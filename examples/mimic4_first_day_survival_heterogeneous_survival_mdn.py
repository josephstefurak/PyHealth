# Contributor: Joseph Stefurak (joseph93@illinois.edu), Igor Bondar
# (ibondar2@illinois.edu), Ian Wilkey (iwilkey2@illinois.edu)
"""First-day ICU survival on MIMIC-IV with mixed-family Survival MDN, with ablations.

Paper: Han, Goldstein and Ranganath. Survival Mixture Density Networks.
MLHC 2022. https://proceedings.mlr.press/v182/han22a.html

Pipeline: ``MIMIC4EHRDataset`` -> ``FirstDayICUSurvivalMIMIC4`` (processors
fitted on the training patients of a patient-level split) ->
``HeterogeneousSurvivalMDN`` (and ``SurvivalMDN`` as the Gaussian reference)
-> the paper's C^td, IBS and IBLL from ``survival_curve_metrics_fn``, and the
censored test NLL.

Ablations (``--ablation``); each arm changes one setting of the baseline
(mixed components, softplus, in-hospital deaths, all features):

- ``family`` (model): all-Gaussian (``SurvivalMDN``), all-logistic, and mixed
  Gaussian + generalized-logistic components, with the same number of
  components.
- ``transform`` (model): softplus vs exponential map to positive time.
- ``event`` (task): ``event_source="hospital"`` (in-hospital deaths only) vs
  ``"any"`` (any recorded death, including deaths after discharge).
- ``features`` (task): all 101 features vs vitals vs labs.

Every arm uses the same split seeds, optimizer and stopping rule. The two
death definitions give slightly different cohorts, so the same seed does not
assign their patients to the same parts; all other arms share their splits.

Run on synthetic data (the default; demonstrates execution only)::

    python examples/mimic4_first_day_survival_heterogeneous_survival_mdn.py

Run on MIMIC-IV (``--mimic-root`` is the folder that holds ``hosp/`` and
``icu/``; ``--dev`` limits the dataset to 1,000 patients)::

    python examples/mimic4_first_day_survival_heterogeneous_survival_mdn.py \\
        --mimic-root /path/to/mimiciv/3.1 --cache-dir /path/to/cache \\
        --ablation family --seeds 0 1 2 3 4 --output results.json

``--output`` holds only settings, cohort counts and aggregate metrics.

The "Findings" section below reports results on the real data. Synthetic
numbers carry no meaning.

Findings (MIMIC-IV 3.1 through the PyHealth pipeline above; five patient-level
60/20/20 splits, ``PatientSplit`` seeds 0-4; one training run per split; no
hyperparameter tuning):
    Nine components, hidden width 64, three layers, RMSprop lr 2e-3, batch
    1024, early stopping on validation NLL (patience 10), scale floor
    softplus(0.01) for every family. Mean ± standard error over the five
    splits. C^td, IBS and IBLL are the paper's metrics from
    ``survival_curve_metrics_fn``; "full" is the near-untruncated horizon
    (censoring-survival threshold 1e-8) and 0.2 / 0.4 are the truncated ones.

    Cohort (``event_source="hospital"``, the default): 66,131 ICU stays, 10.7%
    in-hospital deaths, median time 2.5 days, longest 247 days, 101 features.

    Component family (model ablation):

    ========  =============  ==============  ==============  =============
    family    C^td (full)    IBLL (full)     IBLL (0.4)      test NLL
    ========  =============  ==============  ==============  =============
    Gaussian  0.659 ± 0.004  -0.589 ± 0.019  -0.080 ± 0.001  0.483 ± 0.010
    logistic  0.665 ± 0.005  -0.345 ± 0.025  -0.082 ± 0.002  0.450 ± 0.008
    mixed     0.662 ± 0.004  -0.350 ± 0.026  -0.081 ± 0.001  0.448 ± 0.007
    ========  =============  ==============  ==============  =============

    - Ranking is the same for all three: no C^td difference between two
      families reaches two standard errors at any horizon (logistic minus
      Gaussian +0.006 ± 0.004 at the full horizon, paired by split).
    - Logistic and mixed components fit the far tail better: full-horizon
      IBLL improves by 0.24 ± 0.04 and test NLL by 0.033 ± 0.006, on all five
      splits. At the truncated horizons there is no gain (IBLL at 0.2: -0.189,
      -0.193, -0.192; at 0.4 the Gaussian model is ahead of the logistic one
      by 0.002 ± 0.001, paired), so the gain comes from late times with few
      patients still at risk.
    - Mixed and all-logistic are close: paired, logistic is ahead by
      0.005 ± 0.002 in full-horizon IBLL and mixed by 0.002 ± 0.001 in test
      NLL. The mixed arm lists its Gaussian components first, which the
      reference initialization starts at the lowest latent means; other
      orders were not tried. IBS does not separate the
      families (0.095, 0.099, 0.098 at the full horizon, standard errors
      0.005 to 0.008).
    - The Gaussian arm is ``SurvivalMDN``; the other two are
      ``HeterogeneousSurvivalMDN``. Their best validation epochs were about
      15, 7 and 6.

    Time transform, mixed components (model ablation):

    =========  =============  ==============  =============
    transform  C^td (full)    IBLL (full)     test NLL
    =========  =============  ==============  =============
    softplus   0.662 ± 0.004  -0.350 ± 0.026  0.448 ± 0.007
    exp        0.666 ± 0.002  -0.343 ± 0.023  0.448 ± 0.007
    =========  =============  ==============  =============

    - No difference reaches two standard errors (C^td +0.004 ± 0.003,
      paired by split).

    Feature set, mixed components (task ablation):

    ========  ========  =============  =============  =============
    features  width     C^td (full)    C^td (0.4)     test NLL
    ========  ========  =============  =============  =============
    all       101       0.662 ± 0.004  0.822 ± 0.004  0.448 ± 0.007
    vitals    29        0.636 ± 0.005  0.778 ± 0.007  0.458 ± 0.007
    labs      75        0.661 ± 0.004  0.796 ± 0.004  0.452 ± 0.007
    ========  ========  =============  =============  =============

    - Labs carry more of the signal than vitals. Labs alone match all
      features at the full horizon but not at the truncated ones, where the
      two groups together are better than either.

    Death definition (task ablation). ``event_source="any"`` gives 66,112
    stays with 41.1% deaths and times up to 5,643 days, because a death
    recorded years after discharge is an event while a survivor is censored
    at ICU discharge. On that cohort:

    ========  =============  ==============  =============  ==========
    family    C^td (full)    IBLL (full)     test NLL       best epoch
    ========  =============  ==============  =============  ==========
    Gaussian  0.626 ± 0.007  -0.310 ± 0.008  3.313 ± 0.093  73
    logistic  0.659 ± 0.002  -0.196 ± 0.004  2.683 ± 0.013  15
    mixed     0.659 ± 0.002  -0.202 ± 0.006  2.695 ± 0.011  16
    ========  =============  ==============  =============  ==========

    - With times spanning three orders of magnitude the Gaussian mixture
      needs about five times as many epochs and ends worse on every metric
      (C^td -0.033 ± 0.007, all five splits). The two cohorts are not
      comparable with each other; this definition makes censoring
      informative and is shown as a stress test of the tails, not as a
      recommended cohort.

    How these numbers relate to others:

    - The paper reports 18% deaths for its MIMIC-IV cohort, built from the
      mimic-code derived tables with the SOFA score and tuned with a
      hyperparameter search. This example uses raw-table aggregates, no SOFA
      and no tuning, on MIMIC-IV 3.1, so it is not a reproduction of the
      paper's table.
    - Before this pipeline existed, the contributors ran the same three
      families in a standalone fork of the authors' released code (not
      PyHealth) on a separately built cohort with the same 10.7% death rate:
      C^td 0.641 / 0.671 / 0.667 and full-horizon IBLL -0.716 / -0.336 /
      -0.355 for Gaussian / logistic / mixed over five splits. That setup
      differs in splits and in feature filtering (96 features, clipped
      outliers). Both show the same IBLL pattern; its larger C^td gap
      came mostly from one split and is not seen here.

    Compute: Apple M1 Max, 32 GB, CPU only. Building the cohort took about
    9 minutes the first time and 1.5 minutes for each further split; training
    one model took 7 to 12 seconds (26 to 94 seconds under ``"any"``).
    PyHealth's event cache could not be built from the full ``chartevents``
    and ``labevents`` files on this machine (the build reached 55 GB of scratch
    space), so both files were first reduced to the rows the task reads: the
    task's item ids, and for labs only patients with an ICU stay (95.6 million
    of 591 million rows). The task ignores every other row, so the samples
    are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit
from pyhealth.metrics import survival_curve_metrics_fn
from pyhealth.models import HeterogeneousSurvivalMDN, SurvivalMDN
from pyhealth.tasks import FirstDayICUSurvivalMIMIC4

#: Scale floor of the authors' released code: softplus(clamp(raw, 0.01, 100)).
CODE_SCALE_FLOOR = math.log1p(math.exp(0.01))
BASELINE = {
    "family": "mixed",
    "transform": "softplus",
    "event_source": "hospital",
    "feature_set": "all",
}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class Part:
    """Dense tensors for one part of a split."""

    x: torch.Tensor
    duration: torch.Tensor
    event: torch.Tensor


def to_part(samples) -> Part:
    """Stacks a processed ``SampleDataset`` into tensors."""
    items = [samples[i] for i in range(len(samples))]
    return Part(
        x=torch.stack([item["features"] for item in items]).float(),
        duration=torch.cat([item["duration"].reshape(-1) for item in items]).float(),
        event=torch.cat([item["event"].reshape(-1) for item in items]).float(),
    )


def write_synthetic(root: Path, n_patients: int = 300, seed: int = 0) -> None:
    """Writes small MIMIC-IV-shaped tables (for the default smoke run).

    Survival depends on age, heart rate and lactate so that the ablations have
    something to find. Values are drawn from the task's plausible ranges and
    do not resemble real MIMIC-IV measurements.
    """
    rng = np.random.default_rng(seed)
    task = FirstDayICUSurvivalMIMIC4
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    (root / "icu").mkdir(parents=True, exist_ok=True)
    patients, admissions, stays, chart, labs = [], [], [], [], []
    origin = datetime(2150, 1, 1)

    def draw(name: str, shift: float = 0.0) -> float:
        low, high = task.PLAUSIBLE_RANGES[name]
        centre = low + (0.35 + 0.1 * shift) * (high - low)
        return float(np.clip(rng.normal(centre, 0.04 * (high - low)), low, high))

    for index in range(n_patients):
        subject, hadm, stay = 10_000 + index, 20_000 + index, 30_000 + index
        age = int(rng.integers(25, 90))
        severity = float(rng.normal())
        intime = origin + timedelta(hours=float(rng.uniform(0, 24 * 300)))
        los_hours = float(rng.uniform(12, 240))
        hazard = math.exp(0.9 * severity + 0.03 * (age - 60) - 2.2)
        death_hours = 24.0 * float(rng.exponential(1.0 / hazard))
        deathtime, dod = "", ""
        if death_hours < los_hours:
            los_hours = death_hours
            deathtime = (intime + timedelta(hours=death_hours)).strftime(TIME_FORMAT)
            dod = deathtime[:10]
        elif death_hours < los_hours + 24 * 60:
            dod = (intime + timedelta(hours=death_hours)).strftime("%Y-%m-%d")
        outtime = intime + timedelta(hours=los_hours)
        race = str(rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "UNKNOWN"]))
        gender = str(rng.choice(["M", "F"]))
        patients.append([subject, gender, age, 2150, "2150 - 2152", dod])
        admissions.append(
            [
                subject,
                hadm,
                (intime - timedelta(hours=6)).strftime(TIME_FORMAT),
                (outtime + timedelta(hours=24)).strftime(TIME_FORMAT),
                deathtime,
                "EW EMER.",
                "EMERGENCY ROOM",
                "Other",
                "ENGLISH",
                "SINGLE",
                race,
                "HOME",
                int(bool(deathtime)),
            ]
        )
        stays.append(
            [
                subject,
                hadm,
                stay,
                intime.strftime(TIME_FORMAT),
                outtime.strftime(TIME_FORMAT),
                "Medical Intensive Care Unit (MICU)",
                "Medical Intensive Care Unit (MICU)",
            ]
        )
        for name, itemids in {**task.VITALS, **task.BODY_SIZE}.items():
            repeats = 1 if name in task.BODY_SIZE else 4
            shift = severity if name == "heart_rate" else 0.0
            for _ in range(repeats):
                when = intime + timedelta(hours=float(rng.uniform(0, 23)))
                value = round(draw(name, shift), 2)
                chart.append(
                    [
                        subject,
                        hadm,
                        stay,
                        when.strftime(TIME_FORMAT),
                        itemids[0],
                        value,
                        value,
                        "",
                    ]
                )
        for name, itemids in task.LABS.items():
            if rng.random() < 0.15:  # some labs are never drawn
                continue
            shift = severity if name == "lactate" else 0.0
            for _ in range(2):
                when = intime + timedelta(hours=float(rng.uniform(-5, 23)))
                value = round(draw(name, shift), 2)
                labs.append(
                    [
                        subject,
                        hadm,
                        itemids[0],
                        when.strftime(TIME_FORMAT),
                        when.strftime(TIME_FORMAT),
                        value,
                        value,
                        "",
                        "",
                        "",
                        "",
                    ]
                )

    lab_items = [
        [ids[0], name, "Blood", "Chemistry"] for name, ids in task.LABS.items()
    ]
    tables = {
        "hosp/patients.csv": (
            "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
            patients,
        ),
        "hosp/admissions.csv": (
            "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,"
            "admission_location,insurance,language,marital_status,race,"
            "discharge_location,hospital_expire_flag",
            admissions,
        ),
        "hosp/d_labitems.csv": ("itemid,label,fluid,category", lab_items),
        "hosp/labevents.csv": (
            "subject_id,hadm_id,itemid,charttime,storetime,value,valuenum,"
            "valueuom,ref_range_lower,ref_range_upper,flag",
            labs,
        ),
        "icu/icustays.csv": (
            "subject_id,hadm_id,stay_id,intime,outtime,first_careunit,last_careunit",
            stays,
        ),
        "icu/chartevents.csv": (
            "subject_id,hadm_id,stay_id,charttime,itemid,value,valuenum,valueuom",
            chart,
        ),
    }
    for name, (header, rows) in tables.items():
        with open(root / name, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header.split(","))
            writer.writerows(rows)


def component_families(family: str, num_components: int) -> tuple[str, ...]:
    """Families of an arm: all logistic, or Gaussian first then logistic."""
    if family == "logistic":
        return ("logistic",) * num_components
    num_gaussian = (num_components + 1) // 2
    return ("gaussian",) * num_gaussian + ("logistic",) * (
        num_components - num_gaussian
    )


def build_model(arm: dict, train_samples, train: Part, args):
    """Creates the model of one arm with a time grid that covers the data."""
    grid_max = float(train.duration.max()) * 1.05
    common = dict(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        time_transform=arm["transform"],
        min_scale=args.min_scale,
        time_grid=torch.linspace(grid_max / 200, grid_max, 200),
    )
    if arm["family"] == "gaussian":
        return SurvivalMDN(train_samples, num_components=args.components, **common)
    return HeterogeneousSurvivalMDN(
        train_samples,
        component_families=component_families(arm["family"], args.components),
        **common,
    )


def censored_nll(model, part: Part) -> float:
    """Mean censored negative log-likelihood of ``part``."""
    model.eval()
    with torch.no_grad():
        params = model.network.mixture_parameters(part.x)
        return float(model.network.censored_nll(params, part.duration, part.event))


def train_model(model, train: Part, valid: Part, seed: int, args) -> tuple[int, float]:
    """Trains with early stopping on validation NLL; keeps the best weights.

    Returns:
        The best epoch and the training time in seconds.
    """
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    batch = min(args.batch_size, len(train.x))
    best, best_epoch, best_state, waited = math.inf, 0, None, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train.x))
        # Drop an incomplete last batch: BatchNorm cannot train on one sample.
        for begin in range(0, len(order) - batch + 1, batch):
            idx = torch.as_tensor(order[begin : begin + batch])
            optimizer.zero_grad()
            loss = model(
                features=train.x[idx],
                duration=train.duration[idx],
                event=train.event[idx],
            )["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
        valid_nll = censored_nll(model, valid)
        if valid_nll < best:
            best, best_epoch, waited = valid_nll, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("validation NLL was never finite; training diverged")
    model.load_state_dict(best_state)
    return best_epoch, time.perf_counter() - start


def evaluate(model, test: Part) -> dict[str, float]:
    """The paper's C^td, IBS and IBLL at three horizons, and the test NLL."""

    def predict(grid: np.ndarray) -> np.ndarray:
        times = torch.as_tensor(grid, dtype=torch.float32)
        rows = [
            model.predict_survival(test.x[begin : begin + 4096], times).numpy()
            for begin in range(0, len(test.x), 4096)
        ]
        return np.concatenate(rows, axis=0)

    scores = survival_curve_metrics_fn(
        test.duration.numpy(), test.event.numpy(), predict
    )
    return {"test_nll": censored_nll(model, test), **scores}


def configurations(ablations: list[str], event_source: str = "hospital") -> list[dict]:
    """Lists the ablation arms; each changes one setting of the baseline."""
    baseline = {**BASELINE, "event_source": event_source}
    options = {
        "family": ("family", ("gaussian", "logistic", "mixed")),
        "transform": ("transform", ("softplus", "exp")),
        "event": ("event_source", ("hospital", "any")),
        "features": ("feature_set", ("all", "vitals", "labs")),
    }
    arms = []
    for ablation in ablations:
        key, values = options[ablation]
        for value in values:
            arms.append({**baseline, key: value, "ablation": ablation})
    return arms


def cohort_summary(parts: dict[str, Part]) -> dict[str, float]:
    """Aggregate description of one split (counts and rates only)."""
    duration = torch.cat([part.duration for part in parts.values()])
    event = torch.cat([part.event for part in parts.values()])
    return {
        **{f"n_{name}": len(part.x) for name, part in parts.items()},
        "num_features": int(parts["train"].x.shape[1]),
        "event_rate": float(event.mean()),
        "median_days": float(duration.median()),
        "max_days": float(duration.max()),
    }


def write_output(path: Path | None, payload: dict) -> None:
    """Writes the aggregate results; called after every run to keep progress."""
    if path is not None:
        path.write_text(json.dumps(payload, indent=1))


def run(args: argparse.Namespace) -> dict:
    """Runs every arm on every split seed."""
    started = time.perf_counter()
    dataset = MIMIC4EHRDataset(
        root=str(args.mimic_root),
        tables=["chartevents", "labevents"],
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
        num_workers=args.num_workers,
        dev=args.dev,
    )
    arms = configurations(args.ablation, args.event_source)
    settings = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in ("mimic_root", "cache_dir", "output")
    }
    payload = {"settings": settings, "cohorts": [], "runs": [], "timing": {}}
    trained: dict[tuple, dict] = {}
    task_keys = list(dict.fromkeys((a["event_source"], a["feature_set"]) for a in arms))
    for event_source, feature_set in task_keys:
        task = FirstDayICUSurvivalMIMIC4(
            event_source=event_source, feature_set=feature_set
        )
        for seed in args.seeds:
            clock = time.perf_counter()
            split = PatientSplit(ratios=(0.6, 0.2, 0.2), seed=seed)
            samples = dataset.set_task(task, split=split)
            parts = dict(zip(("train", "valid", "test"), map(to_part, samples)))
            payload["cohorts"].append(
                {
                    "event_source": event_source,
                    "feature_set": feature_set,
                    "seed": seed,
                    "task_seconds": round(time.perf_counter() - clock, 1),
                    **cohort_summary(parts),
                }
            )
            payload["timing"]["first_cohort_seconds"] = payload["timing"].get(
                "first_cohort_seconds", round(time.perf_counter() - started, 1)
            )
            write_output(args.output, payload)
            for arm in arms:
                if (arm["event_source"], arm["feature_set"]) != (
                    event_source,
                    feature_set,
                ):
                    continue
                key = (arm["family"], arm["transform"], event_source, feature_set, seed)
                if key not in trained:  # the baseline is shared by the ablations
                    torch.manual_seed(seed)
                    model = build_model(arm, samples[0], parts["train"], args)
                    epoch, seconds = train_model(
                        model, parts["train"], parts["valid"], seed, args
                    )
                    trained[key] = {
                        "best_epoch": epoch,
                        "train_seconds": round(seconds, 2),
                        **evaluate(model, parts["test"]),
                    }
                row = {**arm, "seed": seed, **trained[key]}
                payload["runs"].append(row)
                print(
                    f"{row['ablation']:9s} family={row['family']:8s} "
                    f"transform={row['transform']:8s} event={event_source:8s} "
                    f"features={feature_set:6s} seed={seed} "
                    f"C^td={row['c_td']:.3f} IBS={row['ibs']:.3f} "
                    f"IBLL={row['ibll']:.3f} NLL={row['test_nll']:.3f}"
                )
                payload["timing"]["total_seconds"] = round(
                    time.perf_counter() - started, 1
                )
                payload["summary"] = summarize(payload["runs"])
                write_output(args.output, payload)
    return payload


def summarize(runs: list[dict]) -> list[dict]:
    """Mean and standard error over split seeds, for each arm."""
    keys = ("ablation", "family", "transform", "event_source", "feature_set")
    groups: dict[tuple, list[dict]] = {}
    for row in runs:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for group, rows in groups.items():
        entry = {**dict(zip(keys, group)), "num_seeds": len(rows)}
        for metric in ("c_td", "ibs", "ibll", "test_nll"):
            values = np.array([row[metric] for row in rows], dtype=float)
            entry[metric] = float(values.mean())
            entry[f"{metric}_se"] = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else None
            )
        summary.append(entry)
    return summary


def print_summary(summary: list[dict]) -> None:
    """Prints one line per arm."""
    print(
        "\nablation   family    transform  event     features  "
        "C^td            IBS             IBLL            test NLL"
    )
    for entry in summary:
        cells = []
        for metric in ("c_td", "ibs", "ibll", "test_nll"):
            se = entry[f"{metric}_se"]
            cells.append(
                f"{entry[metric]:.3f}" + (" ± n/a  " if se is None else f" ± {se:.3f}")
            )
        print(
            f"{entry['ablation']:10s} {entry['family']:9s} {entry['transform']:10s} "
            f"{entry['event_source']:9s} {entry['feature_set']:9s} " + "   ".join(cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mimic-root",
        type=Path,
        help="MIMIC-IV folder holding hosp/ and icu/. Omit to use synthetic data.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated tables in a temporary folder (the default)",
    )
    parser.add_argument("--cache-dir", type=Path, help="PyHealth cache folder")
    parser.add_argument("--dev", action="store_true", help="First 1,000 patients only")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--ablation",
        nargs="+",
        default=["family", "transform", "event", "features"],
        choices=["family", "transform", "event", "features"],
    )
    parser.add_argument(
        "--event-source",
        default="hospital",
        choices=["hospital", "any"],
        help="Death definition of the baseline arms (default: hospital)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Split seeds")
    parser.add_argument("--components", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--min-scale",
        type=float,
        default=CODE_SCALE_FLOOR,
        help="Lower bound on component scales (default: softplus(0.01), the "
        "floor of the authors' released code)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON of settings and aggregate metrics"
    )
    args = parser.parse_args()
    if args.synthetic and args.mimic_root is not None:
        parser.error("--synthetic cannot be combined with --mimic-root")
    args.synthetic = args.mimic_root is None

    with tempfile.TemporaryDirectory() as tmp:
        if args.synthetic:
            if args.dev:
                parser.error("--dev applies to --mimic-root only")
            args.mimic_root = Path(tmp) / "mimiciv"
            write_synthetic(args.mimic_root)
            if args.cache_dir is None:
                args.cache_dir = Path(tmp) / "cache"
            args.epochs = min(args.epochs, 5)
            args.batch_size = min(args.batch_size, 32)
        payload = run(args)
    print_summary(payload["summary"])
    if args.synthetic:
        print("\nSynthetic data: these numbers carry no meaning.")
    if args.output is not None:
        print(f"\nWrote {len(payload['runs'])} runs to {args.output}")


if __name__ == "__main__":
    main()
