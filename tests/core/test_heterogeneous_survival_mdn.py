# Contributor: Joseph Stefurak (joseph93@illinois.edu), Igor Bondar
# (ibondar2@illinois.edu), Ian Wilkey (iwilkey2@illinois.edu)
"""Fast synthetic tests for :class:`pyhealth.models.HeterogeneousSurvivalMDN`."""

import io
import math

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import HeterogeneousSurvivalMDN, SurvivalMDN
from pyhealth.models.heterogeneous_survival_mdn import (
    _HeterogeneousMDNDistribution,
    _HeterogeneousMDNParameters,
)
from pyhealth.models.survival_mdn import _make_time_transform

MIXED = ("gaussian", "logistic", "logistic")
ALL_GAUSSIAN = ("gaussian",) * 3
ALL_LOGISTIC = ("logistic",) * 3


def _make_dataset(durations=(0.8, 1.5, 2.2, 3.1)):
    features = [
        [0.0, 1.0, -0.5, 0.2],
        [1.0, 0.2, 0.4, -0.1],
        [-0.5, 0.7, 0.1, 1.2],
        [0.3, -0.4, 1.0, 0.6],
    ]
    events = [1.0, 0.0, 1.0, 0.0]
    samples = [
        {
            "patient_id": f"p{index}",
            "features": features[index],
            "duration": [durations[index]],
            "event": [events[index]],
        }
        for index in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="heterogeneous_survival_mdn_test",
    )


def _batch(dataset):
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    return next(iter(loader))


def _randomize(model, seed=0):
    """Give every parameter and BatchNorm statistic a non-trivial value."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                0.4 * torch.randn(parameter.shape, generator=generator),
            )
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.copy_(
                    0.2 * torch.randn(module.running_mean.shape, generator=generator),
                )
                module.running_var.copy_(
                    0.5 + torch.rand(module.running_var.shape, generator=generator),
                )


def _distribution(weights, means, scales, shapes, families, transform="softplus"):
    """Build a float64 single-sample distribution from plain lists."""
    as_row = lambda values: torch.tensor([values], dtype=torch.float64)  # noqa: E731
    params = _HeterogeneousMDNParameters(
        log_weights=as_row(weights).log(),
        means=as_row(means),
        scales=as_row(scales),
        raw_weight_logits=as_row(weights).log(),
        raw_scales=as_row(scales),
        shapes=as_row(shapes),
        raw_shapes=as_row(shapes),
    )
    mask = torch.tensor([family == "logistic" for family in families])
    return _HeterogeneousMDNDistribution(
        params,
        mask,
        _make_time_transform(transform),
    )


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _scalar_reference(time, weights, means, scales, shapes, families, transform):
    """Independent scalar evaluation of (log density, survival) at one time."""
    if transform == "softplus":
        latent = math.log(math.expm1(time))
        log_jacobian = -math.log(1.0 - math.exp(-time))
    else:
        latent = math.log(time)
        log_jacobian = -math.log(time)
    density = 0.0
    survival = 0.0
    for weight, mean, scale, shape, family in zip(
        weights, means, scales, shapes, families
    ):
        z = (latent - mean) / scale
        if family == "gaussian":
            density += (
                weight * math.exp(-0.5 * z * z) / (scale * math.sqrt(2.0 * math.pi))
            )
            survival += weight * 0.5 * math.erfc(z / math.sqrt(2.0))
        else:
            density += weight * (shape / scale) * _sigmoid(-z) ** shape * _sigmoid(z)
            survival += weight * _sigmoid(-z) ** shape
    return math.log(density) + log_jacobian, survival


def test_instantiation_and_configuration():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=list(MIXED),
        time_grid=torch.linspace(0.1, 3.0, 11),
    )
    assert model.input_dim == 4
    assert model.num_components == 3
    assert model.component_families == MIXED
    assert model.mode is None
    assert model.label_processors == {}
    assert model.time_grid.shape == (11,)

    default = HeterogeneousSurvivalMDN(dataset)
    assert default.component_families == ("gaussian",) * 5 + ("logistic",) * 5
    assert default.time_grid.shape == (128,)


def test_invalid_configuration_raises():
    dataset = _make_dataset()
    with pytest.raises(ValueError, match="must not be empty"):
        HeterogeneousSurvivalMDN(dataset, component_families=())
    with pytest.raises(ValueError, match="'gaussian' or 'logistic'"):
        HeterogeneousSurvivalMDN(dataset, component_families=("gaussian", "weibull"))
    with pytest.raises(ValueError, match="not a single string"):
        HeterogeneousSurvivalMDN(dataset, component_families="logistic")
    with pytest.raises(ValueError, match="min_shape"):
        HeterogeneousSurvivalMDN(dataset, min_shape=0.0)
    with pytest.raises(ValueError, match="below 1"):
        HeterogeneousSurvivalMDN(dataset, min_shape=1.0)
    unit_floor = HeterogeneousSurvivalMDN(
        dataset, min_shape=1.0, reference_initialization=False
    )
    assert unit_floor.network.min_shape == 1.0
    with pytest.raises(ValueError, match="min_scale"):
        HeterogeneousSurvivalMDN(dataset, min_scale=-1.0)
    with pytest.raises(ValueError, match="time_transform"):
        HeterogeneousSurvivalMDN(dataset, time_transform="identity")
    with pytest.raises(ValueError, match="feature_key"):
        HeterogeneousSurvivalMDN(dataset, feature_key="missing")
    with pytest.raises(ValueError, match="output key"):
        HeterogeneousSurvivalMDN(dataset, event_key="missing")
    with pytest.raises(ValueError, match="strictly positive"):
        HeterogeneousSurvivalMDN(dataset, time_grid=torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("transform", ["softplus", "exp"])
def test_forward_shapes_and_distribution_constraints(transform):
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=MIXED,
        time_transform=transform,
        time_grid=torch.linspace(0.1, 3.0, 11),
    )
    _randomize(model)
    model.eval()
    output = model(**_batch(dataset))

    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    assert output["logit"].shape == (4, 12)
    assert output["y_prob"].shape == (4, 11)
    assert output["y_true"].shape == (4, 2)
    for key in ("weights", "means", "scales", "shapes"):
        assert output[f"mixture_{key}"].shape == (4, 3)
    assert torch.allclose(
        output["mixture_weights"].sum(dim=-1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.all(output["mixture_scales"] > 0)
    assert torch.all(output["mixture_shapes"] > 0)
    # Gaussian components report a shape of exactly one; logistic ones vary.
    assert torch.equal(output["mixture_shapes"][:, 0], torch.ones(4))
    assert not torch.allclose(output["mixture_shapes"][:, 1:], torch.ones(4, 2))
    # The last K logit columns are the raw shape-head outputs.
    expected_shapes = torch.nn.functional.softplus(output["logit"][:, 10:]) + 1e-6
    assert torch.allclose(output["mixture_shapes"][:, 1:], expected_shapes)
    assert torch.all((output["y_prob"] >= 0) & (output["y_prob"] <= 1))
    assert torch.all(output["y_prob"][:, 1:] <= output["y_prob"][:, :-1])
    assert torch.any(output["y_prob"][:, 1:] < output["y_prob"][:, :-1])


@pytest.mark.parametrize("families", [ALL_GAUSSIAN, ALL_LOGISTIC, MIXED])
def test_gradient_computation_is_finite(families):
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=families,
    )
    model.train()
    output = model(**_batch(dataset))
    output["loss"].backward()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    shape_gradient = gradients["network.shape_head.bias"]
    logistic = torch.tensor([family == "logistic" for family in families])
    # Shapes are learned for logistic components and unused by Gaussian ones.
    assert torch.all(shape_gradient[logistic] != 0)
    assert torch.all(shape_gradient[~logistic] == 0)


def test_reference_initialization_starts_logistic_shapes_at_one():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=ALL_LOGISTIC,
    )
    model.eval()
    output = model(**_batch(dataset))
    assert torch.allclose(output["mixture_shapes"], torch.ones(4, 3), atol=1e-6)
    assert torch.allclose(
        output["mixture_means"],
        torch.linspace(-3.0, 3.0, 3).expand(4, 3),
        atol=1e-6,
    )


@pytest.mark.parametrize("transform", ["softplus", "exp"])
def test_all_gaussian_reproduces_survival_mdn(transform):
    dataset = _make_dataset()
    grid = torch.linspace(0.1, 3.0, 11)
    reference = SurvivalMDN(
        dataset,
        hidden_dim=8,
        num_components=3,
        time_transform=transform,
        time_grid=grid,
    )
    _randomize(reference)
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=ALL_GAUSSIAN,
        time_transform=transform,
        time_grid=grid,
    )
    result = model.load_state_dict(reference.state_dict(), strict=False)
    assert sorted(result.missing_keys) == [
        "network.shape_head.bias",
        "network.shape_head.weight",
    ]
    assert result.unexpected_keys == []

    batch = _batch(dataset)
    features = batch["features"]
    times = torch.tensor([0.05, 0.7, 2.0, 9.0])
    for training in (False, True):
        reference.train(training)
        model.train(training)
        expected = reference(**batch)
        actual = model(**batch)
        # Same weights must give the same numbers: tolerance 1e-6 absolute.
        for key, value in expected.items():
            # ``logit`` carries the extra shape-head columns after the first 3K.
            candidate = actual[key][..., :9] if key == "logit" else actual[key]
            assert torch.allclose(candidate, value, rtol=0.0, atol=1e-6), key
    assert torch.allclose(
        model.predict_survival(features, times),
        reference.predict_survival(features, times),
        rtol=0.0,
        atol=1e-6,
    )


def test_component_families_change_the_likelihood():
    dataset = _make_dataset()
    gaussian = HeterogeneousSurvivalMDN(
        dataset, hidden_dim=8, component_families=ALL_GAUSSIAN
    )
    _randomize(gaussian)
    mixed = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    mixed.load_state_dict(gaussian.state_dict())
    gaussian.eval()
    mixed.eval()
    batch = _batch(dataset)
    assert not torch.allclose(gaussian(**batch)["loss"], mixed(**batch)["loss"])
    assert not torch.allclose(gaussian(**batch)["y_prob"], mixed(**batch)["y_prob"])


@pytest.mark.parametrize("transform", ["softplus", "exp"])
@pytest.mark.parametrize("families", [ALL_LOGISTIC, MIXED])
def test_log_density_and_survival_match_scalar_reference(transform, families):
    weights = [0.2, 0.5, 0.3]
    means = [-0.4, 0.6, 1.5]
    scales = [0.7, 1.3, 0.4]
    shapes = [0.6, 1.0, 2.5] if families == ALL_LOGISTIC else [1.0, 0.6, 2.5]
    distribution = _distribution(weights, means, scales, shapes, families, transform)
    for time in (0.05, 0.9, 2.3, 7.0):
        expected_log_density, expected_survival = _scalar_reference(
            time, weights, means, scales, shapes, families, transform
        )
        point = torch.tensor([time], dtype=torch.float64)
        assert distribution.log_prob(point).item() == pytest.approx(
            expected_log_density, abs=1e-10
        )
        assert distribution.survival(point).item() == pytest.approx(
            expected_survival, abs=1e-12
        )


def test_unit_shape_is_the_standard_logistic_distribution():
    mean, scale = 0.3, 0.8
    distribution = _distribution([1.0], [mean], [scale], [1.0], ["logistic"])
    latent = torch.linspace(-6.0, 6.0, 25, dtype=torch.float64)
    time = torch.nn.functional.softplus(latent)
    standard = torch.distributions.TransformedDistribution(
        torch.distributions.Uniform(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
        [
            torch.distributions.SigmoidTransform().inv,
            torch.distributions.AffineTransform(mean, scale),
        ],
    )
    assert torch.allclose(
        distribution.log_prob(time, include_jacobian=False),
        standard.log_prob(latent),
        atol=1e-9,
    )
    assert torch.allclose(
        distribution.survival(time),
        1.0 - standard.cdf(latent),
        atol=1e-12,
    )


def test_logistic_latent_density_integrates_to_one_and_matches_cdf():
    distribution = _distribution(
        weights=[0.2, 0.5, 0.3],
        means=[-1.0, 0.5, 2.0],
        scales=[0.5, 1.5, 3.0],
        shapes=[0.4, 1.0, 2.5],
        families=ALL_LOGISTIC,
    )
    latent = torch.linspace(-200.0, 200.0, 200001, dtype=torch.float64)
    time = torch.nn.functional.softplus(latent)
    density = distribution.log_prob(time, include_jacobian=False).exp()
    cdf = torch.cumulative_trapezoid(density, latent)
    assert cdf[-1].item() == pytest.approx(1.0, abs=1e-8)
    survival = distribution.survival(time)
    assert torch.allclose(1.0 - cdf, survival[1:], rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("transform", ["softplus", "exp"])
def test_positive_time_density_integrates_to_survival_difference(transform):
    distribution = _distribution(
        weights=[0.3, 0.4, 0.3],
        means=[0.5, 1.0, 1.4],
        scales=[0.3, 0.25, 0.2],
        shapes=[1.0, 0.7, 2.0],
        families=MIXED,
        transform=transform,
    )
    time = torch.linspace(1e-3, 40.0, 400001, dtype=torch.float64)
    density = distribution.log_prob(time, include_jacobian=True).exp()
    mass = torch.cumulative_trapezoid(density, time)
    survival = distribution.survival(time)
    assert torch.allclose(survival[0] - mass, survival[1:], rtol=0.0, atol=1e-7)
    # Nearly all probability lies inside the grid, so the check is not vacuous.
    assert mass[-1].item() > 0.999


@pytest.mark.parametrize("transform", ["softplus", "exp"])
@pytest.mark.parametrize("families", [ALL_GAUSSIAN, ALL_LOGISTIC, MIXED])
def test_large_times_give_finite_loss_and_gradients(transform, families):
    dataset = _make_dataset(durations=(200.0, 246.5, 310.0, 1000.0))
    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=families,
        time_transform=transform,
        time_grid=torch.linspace(1.0, 1000.0, 9),
    )
    model.train()
    output = model(**_batch(dataset))
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["y_prob"]).all()
    output["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_unused_gaussian_term_does_not_poison_logistic_gradients():
    # Narrow logistic components far below late times: standardized times of
    # 5e4 to 3e5, where the float32 gradient of the Gaussian term is infinite.
    means = torch.zeros(1, 2, requires_grad=True)
    scales = torch.full((1, 2), 1e-3, requires_grad=True)
    shapes = torch.full((1, 2), 2.0, requires_grad=True)
    log_weights = torch.full((251, 2), 0.5).log()
    params = _HeterogeneousMDNParameters(
        log_weights=log_weights,
        means=means,
        scales=scales,
        raw_weight_logits=log_weights,
        raw_scales=scales,
        shapes=shapes,
        raw_shapes=shapes,
    )
    distribution = _HeterogeneousMDNDistribution(
        params,
        torch.tensor([True, True]),
        _make_time_transform("softplus"),
    )
    time = torch.linspace(50.0, 300.0, 251)
    distribution.log_survival(time).sum().backward()
    # d/dmean of shape * logsigmoid(-z) is shape / scale = 2000 per time, and
    # each of the two equal components carries half of it: 251 * 1000. The
    # float32 sum at these magnitudes is within one percent of that.
    assert torch.allclose(means.grad, torch.full((1, 2), 251000.0), rtol=1e-2)
    assert torch.isfinite(scales.grad).all()
    assert torch.isfinite(shapes.grad).all()


def test_forward_from_embedding_matches_forward():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    _randomize(model)
    model.eval()
    batch = _batch(dataset)
    labels = {key: batch[key] for key in ("duration", "event")}
    expected = model(**batch)
    actual = model.forward_from_embedding(batch["features"], **labels)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_state_dict_round_trip():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    _randomize(model, seed=3)
    restored = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    model.eval()
    restored.eval()
    batch = _batch(dataset)
    assert not torch.allclose(model(**batch)["logit"], restored(**batch)["logit"])

    state = model.state_dict()
    assert "network.shape_head.weight" in state
    assert "network.shape_head.bias" in state
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    expected = model(**batch)
    actual = restored(**batch)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_predict_survival_restores_modes_and_batchnorm_statistics():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    model.train()
    batchnorm = model.network.feature_net[1]
    running_mean = batchnorm.running_mean.clone()
    features = _batch(dataset)["features"]
    times = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])

    survival = model.predict_survival(features, times)

    assert survival.shape == (4, 5)
    assert not survival.requires_grad
    assert torch.all((survival >= 0) & (survival <= 1))
    assert torch.all(survival[:, 1:] <= survival[:, :-1])
    assert all(module.training for module in model.modules())
    assert torch.equal(batchnorm.running_mean, running_mean)

    with pytest.raises(ValueError, match="strictly positive"):
        model.predict_survival(features, torch.tensor([0.0, 1.0]))
    assert all(module.training for module in model.modules())


def test_forward_rejects_malformed_batches():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    model.eval()
    batch = _batch(dataset)

    with pytest.raises(KeyError, match="event"):
        model(features=batch["features"], duration=batch["duration"])
    with pytest.raises(ValueError, match=r"\[batch, input_dim\]"):
        model(**{**batch, "features": batch["features"][:, :3]})
    with pytest.raises(ValueError, match="finite"):
        model(**{**batch, "features": batch["features"] * float("nan")})
    with pytest.raises(ValueError, match="0/1 indicators"):
        model(**{**batch, "event": torch.full((4, 1), 2.0)})
    with pytest.raises(ValueError, match="strictly positive"):
        model(**{**batch, "duration": torch.tensor([[1.0], [0.0], [2.0], [3.0]])})
    with pytest.raises(ValueError, match="same batch size"):
        model(**{**batch, "duration": batch["duration"][:3]})


def test_censored_nll_reductions_agree():
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    _randomize(model)
    model.eval()
    batch = _batch(dataset)
    network = model.network
    params = network.mixture_parameters(batch["features"])
    duration, event = batch["duration"], batch["event"]

    per_sample = network.censored_nll(params, duration, event, reduction="none")
    assert per_sample.shape == (4,)
    assert torch.allclose(
        network.censored_nll(params, duration, event, reduction="sum"),
        per_sample.sum(),
    )
    assert torch.allclose(
        network.censored_nll(params, duration, event, reduction="mean"),
        per_sample.mean(),
    )
    with pytest.raises(ValueError, match="reduction"):
        network.censored_nll(params, duration, event, reduction="median")

    # Censored records contribute -log S(t); events contribute -log f(t).
    survival = network.survival_at(params, duration.reshape(-1))
    censored = event.reshape(-1) == 0
    assert torch.allclose(
        per_sample[censored],
        -survival.diagonal()[censored].log(),
        atol=1e-5,
    )


def test_training_lowers_the_censored_likelihood_loss():
    torch.manual_seed(0)
    dataset = _make_dataset()
    model = HeterogeneousSurvivalMDN(dataset, hidden_dim=8, component_families=MIXED)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    batch = _batch(dataset)
    model.train()
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0] - 0.1
