# Contributor: Joseph Stefurak (joseph93@illinois.edu), Igor Bondar
# (ibondar2@illinois.edu), Ian Wilkey (iwilkey2@illinois.edu)
"""Survival Mixture Density Network with mixed component families.

Survival MDN (Han, Goldstein, and Ranganath, "Survival Mixture Density
Networks", MLHC 2022) places a Gaussian mixture on an unconstrained latent
time and maps it to positive time. This module generalizes the mixture so each
component is either Gaussian or generalized logistic (a logistic distribution
with a learned shape parameter that controls skewness), and the two families
can be combined in one model.

The feature network, the weight/mean/scale heads and the positive-time
transforms are reused from :mod:`pyhealth.models.survival_mdn`. With every
component set to ``"gaussian"`` the model computes the same loss and survival
curves as :class:`~pyhealth.models.SurvivalMDN`.

Paper: https://proceedings.mlr.press/v182/han22a.html
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.survival_mdn import (
    Reduction,
    TimeTransform,
    _infer_input_dim,
    _MDNParameters,
    _PositiveTimeTransform,
    _SurvivalMDNNetwork,
    _validate_positive_time,
    _value_tensor,
)
from pyhealth.processors import LabelProcessor

__all__ = ["HeterogeneousSurvivalMDN"]

ComponentFamily = Literal["gaussian", "logistic"]

_COMPONENT_FAMILIES = ("gaussian", "logistic")
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class _HeterogeneousMDNParameters(_MDNParameters):
    """Latent mixture parameters with one shape value per component.

    ``shapes`` is the generalized-logistic shape for logistic components and
    exactly one for Gaussian components, where it is not used.
    """

    shapes: Tensor
    raw_shapes: Tensor


class _HeterogeneousMDNDistribution:
    """Positive-time distribution induced by a mixed-family latent mixture.

    With ``z = (y - mean) / scale`` for latent time ``y``:

    * Gaussian component: ``S(y) = Phi(-z)``.
    * Generalized-logistic component with shape ``a``:
      ``S(y) = sigmoid(-z) ** a`` and
      ``f(y) = (a / scale) * sigmoid(-z) ** a * sigmoid(z)``.

    All terms are evaluated in log space and components are combined with
    ``logsumexp``.
    """

    def __init__(
        self,
        params: _HeterogeneousMDNParameters,
        logistic_mask: Tensor,
        time_transform: _PositiveTimeTransform,
    ) -> None:
        self.params = params
        self.logistic_mask = logistic_mask
        self.time_transform = time_transform

    def _standardized_latent(self, time: Tensor) -> Tensor:
        latent = self.time_transform.inverse(time)
        return (latent.unsqueeze(-1) - self.params.means) / self.params.scales

    def log_prob(self, time: Tensor, include_jacobian: bool = True) -> Tensor:
        """Return ``log f(time | x)`` for each sample."""
        z = self._standardized_latent(time)
        log_scales = torch.log(self.params.scales)
        gaussian_log_pdf = -0.5 * z.square() - log_scales - 0.5 * _LOG_2PI
        logistic_log_pdf = (
            torch.log(self.params.shapes)
            - log_scales
            + self.params.shapes * F.logsigmoid(-z)
            + F.logsigmoid(z)
        )
        component_log_pdf = torch.where(
            self.logistic_mask,
            logistic_log_pdf,
            gaussian_log_pdf,
        )
        result = torch.logsumexp(
            self.params.log_weights + component_log_pdf,
            dim=-1,
        )
        if include_jacobian:
            result = result + self.time_transform.log_abs_det_inverse_jacobian(time)
        return result

    def log_survival(self, time: Tensor) -> Tensor:
        """Return ``log S(time | x)`` without ``1 - CDF`` cancellation."""
        z = self._standardized_latent(time)
        component_log_survival = torch.where(
            self.logistic_mask,
            self.params.shapes * F.logsigmoid(-z),
            torch.special.log_ndtr(-z),
        )
        return torch.logsumexp(
            self.params.log_weights + component_log_survival,
            dim=-1,
        )

    def survival(self, time: Tensor) -> Tensor:
        """Return survival probability ``S(time | x)``."""
        return self.log_survival(time).exp().clamp(0.0, 1.0)


class _HeterogeneousSurvivalMDNNetwork(_SurvivalMDNNetwork):
    """Survival-MDN network with a shape head and per-component families.

    The feature network and the weight, mean and scale heads are inherited
    unchanged, so their parameter names match the Gaussian network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        component_families: Sequence[ComponentFamily],
        num_hidden_layers: int,
        min_scale: float,
        min_shape: float,
        time_transform: TimeTransform,
    ) -> None:
        families = tuple(component_families)
        if not families:
            raise ValueError("component_families must not be empty")
        unknown = sorted({str(f) for f in families if f not in _COMPONENT_FAMILIES})
        if unknown:
            raise ValueError(
                "component_families entries must be 'gaussian' or 'logistic'; "
                f"received {unknown}"
            )
        if not math.isfinite(min_shape) or min_shape <= 0:
            raise ValueError("min_shape must be positive")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_components=len(families),
            num_hidden_layers=num_hidden_layers,
            min_scale=min_scale,
            time_transform=time_transform,
        )
        self.component_families = families
        self.min_shape = float(min_shape)
        self.shape_head = nn.Linear(hidden_dim, len(families))
        self.shape_softplus = nn.Softplus()
        self.register_buffer(
            "logistic_mask",
            torch.tensor([family == "logistic" for family in families]),
            persistent=False,
        )

    def initialize_reference_heads(self) -> None:
        """Apply the reference head initialization and start shapes at one.

        A shape of one makes every logistic component a standard (symmetric)
        logistic distribution at the start of training.
        """
        super().initialize_reference_heads()
        with torch.no_grad():
            self.shape_head.weight.zero_()
            # Inverse softplus of (1 - min_shape), so the initial shape is one.
            self.shape_head.bias.fill_(math.log(math.expm1(1.0 - self.min_shape)))

    def mixture_parameters(self, features: Tensor) -> _HeterogeneousMDNParameters:
        """Compute latent mixture parameters, including component shapes."""
        hidden = self.feature_net(features)
        weight_logits = self.weight_head(hidden)
        means = self.mean_head(hidden)
        raw_scales = self.scale_head(hidden)
        raw_shapes = self.shape_head(hidden)
        scales = self.scale_softplus(raw_scales).clamp_min(self.min_scale)
        shapes = torch.where(
            self.logistic_mask,
            self.shape_softplus(raw_shapes) + self.min_shape,
            torch.ones_like(raw_shapes),
        )
        return _HeterogeneousMDNParameters(
            log_weights=self.log_softmax(weight_logits),
            means=means,
            scales=scales,
            raw_weight_logits=weight_logits,
            raw_scales=raw_scales,
            shapes=shapes,
            raw_shapes=raw_shapes,
        )

    def censored_nll(
        self,
        params: _HeterogeneousMDNParameters,
        duration: Tensor,
        event: Tensor,
        reduction: Reduction = "mean",
        include_jacobian: bool = True,
    ) -> Tensor:
        """Compute right-censored negative log-likelihood."""
        duration = duration.reshape(-1)
        event = event.reshape(-1).to(dtype=params.means.dtype)
        if params.means.shape[0] != duration.shape[0]:
            raise ValueError("features and duration must have the same batch size")
        if params.means.shape[0] != event.shape[0]:
            raise ValueError("features and event must have the same batch size")
        if torch.any((event != 0) & (event != 1)):
            raise ValueError("event must contain only 0/1 indicators")

        distribution = _HeterogeneousMDNDistribution(
            params,
            self.logistic_mask,
            self.time_transform,
        )
        log_pdf = distribution.log_prob(
            duration,
            include_jacobian=include_jacobian,
        )
        log_survival = distribution.log_survival(duration)
        per_sample = -torch.where(event.bool(), log_pdf, log_survival)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def survival_at(
        self,
        params: _HeterogeneousMDNParameters,
        times: Tensor,
    ) -> Tensor:
        """Return ``S(t | x)`` with shape ``[batch, num_times]``."""
        times = times.reshape(-1).to(
            device=params.means.device,
            dtype=params.means.dtype,
        )
        batch_size = params.means.shape[0]
        expanded = _HeterogeneousMDNParameters(
            log_weights=params.log_weights.unsqueeze(1),
            means=params.means.unsqueeze(1),
            scales=params.scales.unsqueeze(1),
            raw_weight_logits=params.raw_weight_logits.unsqueeze(1),
            raw_scales=params.raw_scales.unsqueeze(1),
            shapes=params.shapes.unsqueeze(1),
            raw_shapes=params.raw_shapes.unsqueeze(1),
        )
        distribution = _HeterogeneousMDNDistribution(
            expanded,
            self.logistic_mask,
            self.time_transform,
        )
        grid = times.unsqueeze(0).expand(batch_size, -1)
        return distribution.survival(grid)


class HeterogeneousSurvivalMDN(BaseModel):
    """Survival MDN whose components are Gaussian, generalized logistic or both.

    The model places a mixture distribution on unconstrained latent time and
    maps it to positive event time, as in Survival MDN. Each mixture component
    belongs to one of two families:

    * ``"gaussian"``: the component used by
      :class:`~pyhealth.models.SurvivalMDN`.
    * ``"logistic"``: a generalized logistic component with latent survival
      ``sigmoid(-z) ** shape``, where ``z = (y - mean) / scale``. Its tails are
      heavier than a Gaussian's and the learned shape makes it asymmetric;
      ``shape == 1`` is the standard logistic distribution.

    Training minimizes the right-censored negative log-likelihood: observed
    events contribute a density term and censored records contribute a
    survival-probability term. The dataset contract and the ``forward`` output
    are those of :class:`~pyhealth.models.SurvivalMDN`, with the component
    shapes added. With only Gaussian components the loss and survival curves
    equal those of :class:`~pyhealth.models.SurvivalMDN` for the same weights.

    Args:
        dataset: Fitted PyHealth ``SampleDataset``. The model expects one dense
            numeric feature vector and two tensor outputs: duration and event.
        feature_key: Input field containing the dense feature vector.
        duration_key: Output field containing strictly positive durations.
        event_key: Output field containing ``1`` for an observed event and
            ``0`` for right censoring.
        input_dim: Width of the dense input vector. If omitted, the model
            infers it from the first processed sample.
        hidden_dim: Width of each shared hidden layer.
        component_families: Family of each mixture component, ``"gaussian"``
            or ``"logistic"``. Its length is the number of components. The
            default is five Gaussian followed by five logistic components.
        num_hidden_layers: Number of Linear-BatchNorm-PReLU hidden blocks.
        time_transform: Positive-time transformation, ``"softplus"`` (paper
            baseline) or ``"exp"``.
        min_scale: Minimum component scale for numerical safety.
        min_shape: Positive constant added to the softplus of the shape head,
            so logistic shapes stay strictly positive.
        time_grid: Times at which ``y_prob`` returns survival probabilities.
            If omitted, a 128-point grid from 0.001 to 5.6 is used. Pass a
            grid that covers the durations of the dataset at hand.
        include_jacobian: Whether event-density likelihoods include the
            transformation Jacobian. Keep this ``True`` when comparing
            different time transformations.
        reference_initialization: If ``True``, initialize the mixture heads as
            :class:`~pyhealth.models.SurvivalMDN` does and start every
            logistic shape at one.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "features": [0.0, 1.0, -0.5],
        ...         "duration": [1.2],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "features": [1.0, 0.2, 0.4],
        ...         "duration": [2.0],
        ...         "event": [0.0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = HeterogeneousSurvivalMDN(
        ...     dataset,
        ...     hidden_dim=8,
        ...     component_families=("gaussian", "gaussian", "logistic"),
        ... )
        >>> model.num_components
        3
        >>> model.component_families
        ('gaussian', 'gaussian', 'logistic')

    References:
        Han, X., Goldstein, M., and Ranganath, R. Survival Mixture Density
        Networks. Proceedings of Machine Learning for Healthcare, 2022.
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """None: the survival outputs are scored by the model, not as a label,
        so the Trainer reports only the loss."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        hidden_dim: int = 32,
        component_families: Sequence[ComponentFamily] = ("gaussian",) * 5
        + ("logistic",) * 5,
        num_hidden_layers: int = 3,
        time_transform: TimeTransform = "softplus",
        min_scale: float = 1e-8,
        min_shape: float = 1e-6,
        time_grid: Tensor | None = None,
        include_jacobian: bool = True,
        reference_initialization: bool = True,
    ) -> None:
        super().__init__(dataset)
        if isinstance(component_families, str):
            raise ValueError(
                "component_families must be a sequence of family names, "
                "not a single string"
            )
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.include_jacobian = bool(include_jacobian)
        self.component_families = tuple(component_families)
        self.num_components = len(self.component_families)
        self.time_transform = time_transform

        if feature_key not in dataset.input_schema:
            raise ValueError(
                f"feature_key {feature_key!r} is not in dataset.input_schema"
            )
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in dataset.output_schema")

        if input_dim is None:
            input_dim = _infer_input_dim(dataset, feature_key)
        self.input_dim = int(input_dim)
        self.network = _HeterogeneousSurvivalMDNNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            component_families=self.component_families,
            num_hidden_layers=num_hidden_layers,
            min_scale=min_scale,
            min_shape=min_shape,
            time_transform=time_transform,
        )
        if reference_initialization:
            self.network.initialize_reference_heads()
        if time_grid is None:
            time_grid = torch.linspace(0.001, 5.6, 128)
        time_grid = torch.as_tensor(time_grid, dtype=torch.float32)
        _validate_positive_time(time_grid)
        self.register_buffer("time_grid", time_grid.reshape(-1))

    def forward(
        self,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run one forward pass.

        Args:
            **kwargs: Batch fields produced by the PyHealth dataloader. The
                batch must include ``feature_key``, ``duration_key``, and
                ``event_key``.

        Returns:
            Dictionary containing ``loss``, ``logit``, ``y_prob``, ``y_true``,
            and the fitted mixture weights, means, scales and shapes.
            ``y_prob`` is the survival curve evaluated on ``self.time_grid``.
            ``logit`` concatenates the raw weight, mean, scale and shape head
            outputs, shape ``[batch, 4 * num_components]``.
            ``mixture_shapes`` is one for Gaussian components.
        """
        missing = [
            key
            for key in (self.feature_key, self.duration_key, self.event_key)
            if key not in kwargs
        ]
        if missing:
            raise KeyError(f"Missing required batch fields: {missing}")

        features = torch.as_tensor(
            _value_tensor(kwargs[self.feature_key]),
            dtype=torch.float32,
            device=self.device,
        )
        duration = torch.as_tensor(
            _value_tensor(kwargs[self.duration_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        event = torch.as_tensor(
            _value_tensor(kwargs[self.event_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )
        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be nonempty and finite")

        # Compute the BatchNorm-containing feature network once per batch.
        params = self.network.mixture_parameters(features)
        loss = self.network.censored_nll(
            params,
            duration,
            event,
            include_jacobian=self.include_jacobian,
        )
        y_prob = self.network.survival_at(params, self.time_grid)
        y_true = torch.stack([duration, event], dim=-1)
        logit = torch.cat(
            [
                params.raw_weight_logits,
                params.means,
                params.raw_scales,
                params.raw_shapes,
            ],
            dim=-1,
        )
        return {
            "loss": loss,
            "logit": logit,
            "y_prob": y_prob,
            "y_true": y_true,
            "mixture_weights": params.weights,
            "mixture_means": params.means,
            "mixture_scales": params.scales,
            "mixture_shapes": params.shapes,
        }

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run ``forward`` with dense features supplied directly."""
        kwargs[self.feature_key] = embeddings
        return self.forward(**kwargs)

    @torch.no_grad()
    def predict_survival(
        self,
        features: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict survival probabilities on an arbitrary time grid.

        Evaluation mode is used temporarily so prediction does not update
        BatchNorm statistics; every module's previous mode is restored
        afterwards, also if prediction raises.

        Args:
            features: Dense input tensor with shape ``[batch, input_dim]``.
            times: Strictly positive time points.

        Returns:
            Survival probabilities with shape ``[batch, num_times]``.
        """
        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
            device=self.device,
        )
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            params = self.network.mixture_parameters(features)
            return self.network.survival_at(params, times)
        finally:
            for module, training in previous_modes:
                module.training = training
