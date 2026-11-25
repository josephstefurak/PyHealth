# Contributor: Joseph Stefurak (joseph93@illinois.edu), Igor Bondar
# (ibondar2@illinois.edu), Ian Wilkey (iwilkey2@illinois.edu)
"""First-day ICU survival cohort on MIMIC-IV.

The MIMIC-IV experiment of Han, Goldstein, and Ranganath, "Survival Mixture
Density Networks" (MLHC 2022): for ICU stays longer than 24 hours, summary
statistics of the first day predict the time from ICU admission to death,
right-censored at ICU discharge. The paper builds its features with the
mimic-code ``first_day_*`` SQL concepts; this task computes the same kind of
summaries (minimum, mean and maximum of vitals and labs) directly from the raw
``chartevents`` and ``labevents`` tables.

Paper: https://proceedings.mlr.press/v182/han22a.html
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar, Literal

import polars as pl

from .base_task import BaseTask

__all__ = ["FirstDayICUSurvivalMIMIC4"]

EventSource = Literal["any", "hospital"]
FeatureSet = Literal["all", "vitals", "labs"]

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_STATISTICS = ("min", "mean", "max")
_FAHRENHEIT_ITEMID = 223761
_INCHES_ITEMID = 226707
_UNKNOWN_RACES = frozenset({"UNKNOWN", "UNABLE TO OBTAIN"})
_MEASUREMENT_SCHEMA = {
    "timestamp": pl.Datetime("ms"),
    "stay_id": pl.Int64,
    "feature": pl.String,
    "value": pl.Float64,
}


def _parse_datetime(value: Any) -> datetime | None:
    """Parse a MIMIC-IV timestamp or date; ``None`` if absent or malformed."""
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    for time_format in _TIME_FORMATS:
        try:
            return datetime.strptime(text, time_format)
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float:
    """Convert an attribute to float; NaN if absent or not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value: Any) -> int | None:
    """Convert an identifier to int; ``None`` if absent or not numeric."""
    number = _to_float(value)
    return None if math.isnan(number) else int(number)


def _text(value: Any) -> str:
    """Return a stripped string, empty when the attribute is absent."""
    return "" if value is None else str(value).strip()


class FirstDayICUSurvivalMIMIC4(BaseTask):
    """Time from ICU admission to death from first-day summaries, on MIMIC-IV.

    The MIMIC-IV cohort of the Survival MDN paper (Han et al., MLHC 2022).
    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` and
    :class:`~pyhealth.datasets.MIMIC4Dataset`, loaded with the ``chartevents``
    and ``labevents`` tables.

    Cohort: every ICU stay that is longer than ``observation_hours`` hours,
    whose patient has a recorded sex and a known race, and whose patient is
    alive at the end of the observation window. A patient with several
    qualifying stays gives several samples unless ``first_stay_only`` is set;
    split by patient.

    Each sample is one stay:

    - ``features``: a flat vector in the order of :attr:`feature_names`:
      age at hospital admission, ``is_male`` and ``is_white``; the minimum,
      mean and maximum of 8 charted vitals; the first charted weight (kg) and
      height (cm); and the minimum, mean and maximum of 24 laboratory values.
      That is 101 values for ``feature_set="all"``, 29 for ``"vitals"`` and
      75 for ``"labs"``. Vitals, weight and height are charted during the
      stay's first ``observation_hours`` hours; labs are drawn from 6 hours
      before ICU admission to the end of that window, as in mimic-code's
      first-day lab concept. Values outside a plausible range are dropped as
      artefacts, temperatures charted in Fahrenheit are converted to Celsius
      and heights charted in inches to centimetres. A feature with no
      measurement is NaN in the sample.
    - ``duration`` and ``event``: days from ICU admission to death
      (``event`` 1), or to ICU discharge when no death is recorded
      (``event`` 0).

    With ``standardize`` the feature processor z-scores each feature using the
    observed training values and fills missing values with zero (the training
    mean). Fit it on the training patients only, with
    ``set_task(task, split=...)``.

    How this differs from :class:`~pyhealth.tasks.InHospitalSurvivalMIMIC4`,
    which follows TrajSurv: that task observes 36 hours, uses the YAIB
    variables with their last value, measures time from the end of the window,
    counts in-hospital deaths only and censors at hospital discharge. This
    task observes 24 hours, uses minimum, mean and maximum, measures time from
    ICU admission, can also count deaths recorded after discharge and censors
    at ICU discharge.

    How this differs from the paper's cohort: features are aggregated from the
    raw tables, not taken from the mimic-code derived tables, and the SOFA
    score is not included. The paper counts any recorded date of death in an
    earlier MIMIC-IV release and reports 18% deaths; neither ``event_source``
    reproduces that share on MIMIC-IV 3.1.

    Requires the ``chartevents`` and ``labevents`` tables (``patients``,
    ``admissions`` and ``icustays`` are always loaded).

    Args:
        event_source: Which deaths count as events. ``"hospital"`` (default)
            counts deaths during the stay's hospital admission. ``"any"`` also
            counts the patient's recorded date of death when the admission has
            no death time, so deaths long after discharge become events while
            survivors are still censored at ICU discharge; on MIMIC-IV 3.1
            this raises the share of events from 10.7% to 41.1% and the
            longest time from 247 to 5,643 days. A death recorded only as a
            date is timed at midnight of that date, so its duration can be
            up to a day early.
        feature_set: ``"all"`` (default), ``"vitals"`` (demographics, vitals,
            weight and height) or ``"labs"`` (demographics and labs).
        observation_hours: Length of the observation window in hours.
        first_stay_only: Keep only each patient's earliest qualifying stay.
        standardize: Z-score the features and fill missing values with zero.
            If ``False``, raw values are returned and missing values stay NaN.
        missing_indicators: Append one missing-value indicator per feature,
            which doubles the width of the processed vector.

    Examples:
        >>> from pyhealth.tasks import FirstDayICUSurvivalMIMIC4
        >>> task = FirstDayICUSurvivalMIMIC4(feature_set="vitals")
        >>> len(task.feature_names)
        29
        >>> task.feature_names[:5]
        ['age', 'is_male', 'is_white', 'heart_rate_min', 'heart_rate_mean']
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["chartevents", "labevents"],
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "FirstDayICUSurvivalMIMIC4"
    input_schema: ClassVar[dict[str, Any]] = {
        "features": ("tensor", {"normalization": "zscore", "impute_strategy": "zero"}),
    }
    output_schema: ClassVar[dict[str, Any]] = {"duration": "tensor", "event": "tensor"}

    LAB_LOOKBACK_HOURS: ClassVar[int] = 6
    """Labs drawn up to this many hours before ICU admission are included."""

    DEMOGRAPHICS: ClassVar[tuple[str, ...]] = ("age", "is_male", "is_white")
    VITALS: ClassVar[dict[str, tuple[int, ...]]] = {
        "heart_rate": (220045,),
        "sbp": (220179, 220050),
        "dbp": (220180, 220051),
        "mbp": (220181, 220052),
        "resp_rate": (220210, 224690),
        "temp_c": (223762, _FAHRENHEIT_ITEMID),
        "spo2": (220277,),
        "glucose_chart": (225664, 220621, 226537),
    }
    """``chartevents`` item ids of each vital sign."""
    BODY_SIZE: ClassVar[dict[str, tuple[int, ...]]] = {
        "weight_kg": (226512,),
        "height_cm": (226730, _INCHES_ITEMID),
    }
    """``chartevents`` item ids of admission weight and height."""
    LABS: ClassVar[dict[str, tuple[int, ...]]] = {
        "albumin": (50862,),
        "aniongap": (50868,),
        "bicarbonate": (50882,),
        "bilirubin_total": (50885,),
        "calcium_total": (50893,),
        "chloride": (50902,),
        "creatinine": (50912,),
        "glucose_lab": (50931,),
        "hematocrit": (51221,),
        "hemoglobin": (51222,),
        "inr": (51237,),
        "lactate": (50813,),
        "platelet": (51265,),
        "potassium": (50971,),
        "ptt": (51275,),
        "pt": (51274,),
        "sodium": (50983,),
        "bun": (51006,),
        "wbc": (51301,),
        "ph": (50820,),
        "pco2": (50818,),
        "po2": (50821,),
        "baseexcess": (50802,),
        "rdw": (51277,),
    }
    """``labevents`` item ids of each laboratory value."""
    PLAUSIBLE_RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "heart_rate": (1, 300),
        "sbp": (20, 300),
        "dbp": (10, 250),
        "mbp": (10, 250),
        "resp_rate": (1, 80),
        "temp_c": (25, 45),
        "spo2": (1, 100),
        "glucose_chart": (10, 2000),
        "weight_kg": (20, 400),
        "height_cm": (100, 250),
        "albumin": (0.5, 6),
        "aniongap": (0, 60),
        "bicarbonate": (2, 60),
        "bilirubin_total": (0, 70),
        "calcium_total": (3, 20),
        "chloride": (50, 150),
        "creatinine": (0, 30),
        "glucose_lab": (10, 3000),
        "hematocrit": (5, 80),
        "hemoglobin": (2, 25),
        "inr": (0.5, 20),
        "lactate": (0, 30),
        "platelet": (1, 2000),
        "potassium": (1.5, 12),
        "ptt": (10, 250),
        "pt": (8, 150),
        "sodium": (90, 190),
        "bun": (1, 250),
        "wbc": (0, 500),
        "ph": (6.5, 8),
        "pco2": (5, 250),
        "po2": (10, 700),
        "baseexcess": (-40, 40),
        "rdw": (8, 40),
    }
    """Inclusive range of plausible values; measurements outside are dropped."""

    def __init__(
        self,
        event_source: EventSource = "hospital",
        feature_set: FeatureSet = "all",
        observation_hours: int = 24,
        first_stay_only: bool = False,
        standardize: bool = True,
        missing_indicators: bool = False,
    ) -> None:
        super().__init__()
        if event_source not in ("any", "hospital"):
            raise ValueError("event_source must be 'any' or 'hospital'")
        if feature_set not in ("all", "vitals", "labs"):
            raise ValueError("feature_set must be 'all', 'vitals' or 'labs'")
        if isinstance(observation_hours, bool) or observation_hours < 1:
            raise ValueError("observation_hours must be at least 1")
        self.event_source = event_source
        self.feature_set = feature_set
        self.observation_hours = int(observation_hours)
        self.first_stay_only = bool(first_stay_only)
        self.standardize = bool(standardize)
        self.missing_indicators = bool(missing_indicators)

        options: dict[str, Any] = {}
        if self.standardize:
            options["normalization"] = "zscore"
            options["impute_strategy"] = "zero"
        if self.missing_indicators:
            options["missing_indicators"] = True
        self.input_schema = {"features": ("tensor", options) if options else "tensor"}

        use_chart = feature_set in ("all", "vitals")
        use_labs = feature_set in ("all", "labs")
        names = list(self.DEMOGRAPHICS)
        if use_chart:
            names += [f"{v}_{s}" for v in self.VITALS for s in _STATISTICS]
            names += list(self.BODY_SIZE)
        if use_labs:
            names += [f"{lab}_{s}" for lab in self.LABS for s in _STATISTICS]
        self.feature_names: list[str] = names

    def _table_items(self) -> dict[str, dict[int, str]]:
        """Item id to feature name, for each measurement table in use."""
        tables: dict[str, dict[str, tuple[int, ...]]] = {}
        if self.feature_set in ("all", "vitals"):
            tables["chartevents"] = {**self.VITALS, **self.BODY_SIZE}
        if self.feature_set in ("all", "labs"):
            tables["labevents"] = dict(self.LABS)
        return {
            table: {itemid: name for name, ids in features.items() for itemid in ids}
            for table, features in tables.items()
        }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drop chart and lab rows whose item id the task does not use.

        Args:
            df: The dataset's event table, one row per event.

        Returns:
            The table without the unused ``chartevents`` and ``labevents`` rows.

        Raises:
            ValueError: If a table the task needs was not loaded.
        """
        columns = df.collect_schema().names()
        items = self._table_items()
        missing = [
            table
            for table in items
            if not any(column.startswith(f"{table}/") for column in columns)
        ]
        if missing:
            raise ValueError(
                f"{self.task_name} requires the MIMIC-IV tables {missing}; "
                "add them to the dataset's `tables`."
            )
        measurement_tables = ("chartevents", "labevents")
        keep = ~pl.col("event_type").is_in(measurement_tables)
        for table, mapping in items.items():
            keep = keep | (
                (pl.col("event_type") == table)
                & pl.col(f"{table}/itemid")
                .cast(pl.String)
                .is_in([str(itemid) for itemid in sorted(mapping)])
            )
        return df.filter(keep)

    def _measurements(self, patient: Any, table: str) -> pl.DataFrame:
        """One patient's usable measurements of one table, in long format.

        Units are converted; implausible, missing and non-numeric values are
        dropped by the range check. Columns are
        ``timestamp``, ``stay_id`` (null for labs), ``feature`` and ``value``.
        """
        mapping = self._table_items()[table]
        events = patient.get_events(event_type=table, return_df=True)
        value_column, item_column = f"{table}/valuenum", f"{table}/itemid"
        if events.height == 0 or value_column not in events.columns:
            return pl.DataFrame(schema=_MEASUREMENT_SCHEMA)
        stay_column = f"{table}/stay_id"
        stay_id = (
            pl.col(stay_column).cast(pl.Int64, strict=False)
            if stay_column in events.columns
            else pl.lit(None, dtype=pl.Int64)
        )
        rows = events.select(
            pl.col("timestamp").cast(pl.Datetime("ms")),
            stay_id.alias("stay_id"),
            pl.col(item_column).cast(pl.Int64, strict=False).alias("itemid"),
            pl.col(value_column).cast(pl.Float64, strict=False).alias("value"),
        ).filter(pl.col("itemid").is_in(list(mapping)))
        value = (
            pl.when(pl.col("itemid") == _FAHRENHEIT_ITEMID)
            .then((pl.col("value") - 32.0) / 1.8)
            .when(pl.col("itemid") == _INCHES_ITEMID)
            .then(pl.col("value") * 2.54)
            .otherwise(pl.col("value"))
        )
        feature = pl.col("itemid").replace_strict(mapping, return_dtype=pl.String)
        rows = rows.with_columns(value.alias("value"), feature.alias("feature"))
        lower = {
            name: float(self.PLAUSIBLE_RANGES[name][0]) for name in mapping.values()
        }
        upper = {
            name: float(self.PLAUSIBLE_RANGES[name][1]) for name in mapping.values()
        }
        rows = rows.filter(
            (pl.col("value") >= pl.col("feature").replace_strict(lower))
            & (pl.col("value") <= pl.col("feature").replace_strict(upper))
        )
        return rows.select(list(_MEASUREMENT_SCHEMA))

    @staticmethod
    def _summaries(
        rows: pl.DataFrame,
        start: datetime,
        end: datetime,
        stay_id: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Per feature: min, mean, max and first value in ``[start, end)``."""
        window = (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
        if stay_id is not None:
            window = window & (pl.col("stay_id") == stay_id)
        summary = (
            rows.filter(window)
            .group_by("feature")
            .agg(
                pl.col("value").min().alias("min"),
                pl.col("value").mean().alias("mean"),
                pl.col("value").max().alias("max"),
                pl.col("value").sort_by("timestamp").first().alias("first"),
            )
        )
        return {row["feature"]: row for row in summary.iter_rows(named=True)}

    def _features(
        self,
        demographics: tuple[float, float, float],
        chart: dict[str, dict[str, float]],
        labs: dict[str, dict[str, float]],
    ) -> list[float]:
        """Assemble one stay's vector in the order of ``feature_names``."""
        values = list(demographics)
        if self.feature_set in ("all", "vitals"):
            for name in self.VITALS:
                stats = chart.get(name)
                values += [stats[s] if stats else math.nan for s in _STATISTICS]
            for name in self.BODY_SIZE:
                stats = chart.get(name)
                values.append(stats["first"] if stats else math.nan)
        if self.feature_set in ("all", "labs"):
            for name in self.LABS:
                stats = labs.get(name)
                values += [stats[s] if stats else math.nan for s in _STATISTICS]
        return [float(value) for value in values]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per qualifying ICU stay of ``patient``.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with MIMIC-IV events.

        Returns:
            Samples with ``patient_id``, ``stay_id``, ``features``,
            ``duration`` (days, strictly positive) and ``event``.
        """
        demographics = patient.get_events(event_type="patients")
        stays = patient.get_events(event_type="icustays")
        if not demographics or not stays:
            return []
        person = demographics[0].attr_dict
        sex = _text(person.get("gender"))
        if not sex:
            return []
        date_of_death = _parse_datetime(person.get("dod"))
        anchor_age = _to_float(person.get("anchor_age"))
        anchor_year = _to_float(person.get("anchor_year"))

        admissions = {}
        for admission in patient.get_events(event_type="admissions"):
            hadm_id = _to_int(admission.attr_dict.get("hadm_id"))
            if hadm_id is not None and hadm_id not in admissions:
                admissions[hadm_id] = admission

        items = self._table_items()
        chart_rows = (
            self._measurements(patient, "chartevents")
            if "chartevents" in items
            else None
        )
        lab_rows = (
            self._measurements(patient, "labevents") if "labevents" in items else None
        )
        window = timedelta(hours=self.observation_hours)
        lookback = timedelta(hours=self.LAB_LOOKBACK_HOURS)

        samples = []
        for stay in sorted(stays, key=lambda event: event.timestamp or datetime.max):
            stay_id = _to_int(stay.attr_dict.get("stay_id"))
            admission = admissions.get(_to_int(stay.attr_dict.get("hadm_id")))
            intime = stay.timestamp
            outtime = _parse_datetime(stay.attr_dict.get("outtime"))
            if stay_id is None or admission is None or intime is None:
                continue
            if outtime is None or outtime - intime <= window:
                continue
            race = _text(admission.attr_dict.get("race")).upper()
            if not race or race in _UNKNOWN_RACES:
                continue

            death = _parse_datetime(admission.attr_dict.get("deathtime"))
            if death is None and self.event_source == "any":
                death = date_of_death
            if death is not None and death - intime <= window:
                continue
            end = outtime if death is None else death
            duration = (end - intime).total_seconds() / 86400.0

            admitted = admission.timestamp
            age = (
                anchor_age + (admitted.year - anchor_year)
                if admitted is not None
                else math.nan
            )
            person_features = (
                age,
                float(sex == "M"),
                float(race.startswith("WHITE")),
            )
            window_end = intime + window
            chart = (
                self._summaries(chart_rows, intime, window_end, stay_id)
                if chart_rows is not None
                else {}
            )
            labs = (
                self._summaries(lab_rows, intime - lookback, window_end)
                if lab_rows is not None
                else {}
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "features": self._features(person_features, chart, labs),
                    "duration": [duration],
                    "event": [float(death is not None)],
                }
            )
            if self.first_stay_only:
                break
        return samples
