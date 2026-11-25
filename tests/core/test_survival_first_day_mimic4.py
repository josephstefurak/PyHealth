# Contributor: Joseph Stefurak (joseph93@illinois.edu), Igor Bondar
# (ibondar2@illinois.edu), Ian Wilkey (iwilkey2@illinois.edu)
"""Fast tests for :class:`pyhealth.tasks.FirstDayICUSurvivalMIMIC4`.

Every patient here is fabricated in the test; no MIMIC-IV file is read.
"""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import HeterogeneousSurvivalMDN
from pyhealth.tasks import FirstDayICUSurvivalMIMIC4

T0 = datetime(2150, 3, 1, 8, 0, 0)
HEART_RATE, TEMP_C, TEMP_F = 220045, 223762, 223761
WEIGHT, HEIGHT_CM, HEIGHT_IN = 226512, 226730, 226707
LACTATE, CREATININE = 50813, 50912


def _hours(hours):
    return timedelta(hours=hours)


def _fmt(time):
    return None if time is None else time.strftime("%Y-%m-%d %H:%M:%S")


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _patient(
    stays,
    chart=(),
    labs=(),
    gender="M",
    race="WHITE",
    dod=None,
    deathtimes=None,
    admit_year=2150,
):
    """Fabricate one patient.

    stays: (stay_id, hadm_id, intime, hours in ICU).
    chart: (stay_id, time, itemid, valuenum); labs: (time, itemid, valuenum).
    deathtimes: hadm_id -> in-hospital death time.
    """
    deathtimes = deathtimes or {}
    rows = [
        _event(
            "patients",
            None,
            gender=gender,
            anchor_age=60,
            anchor_year=2148,
            dod=dod,
        )
    ]
    for hadm_id in sorted({stay[1] for stay in stays}):
        rows.append(
            _event(
                "admissions",
                datetime(admit_year, 2, 27, 12, 0, 0),
                hadm_id=hadm_id,
                race=race,
                deathtime=_fmt(deathtimes.get(hadm_id)),
            )
        )
    for stay_id, hadm_id, intime, hours in stays:
        rows.append(
            _event(
                "icustays",
                intime,
                stay_id=stay_id,
                hadm_id=hadm_id,
                outtime=_fmt(intime + _hours(hours)),
            )
        )
    for stay_id, time, itemid, valuenum in chart:
        rows.append(
            _event(
                "chartevents",
                time,
                stay_id=stay_id,
                itemid=itemid,
                valuenum=valuenum,
            )
        )
    for time, itemid, valuenum in labs:
        rows.append(_event("labevents", time, itemid=itemid, valuenum=valuenum))
    frame = pl.from_dicts(rows, infer_schema_length=None)
    frame = frame.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(name).cast(pl.String) for name in frame.columns if "/" in name),
    )
    return Patient(patient_id="p1", data_source=frame)


def _value(task, sample, name):
    return sample["features"][task.feature_names.index(name)]


def test_feature_names_and_feature_sets():
    task = FirstDayICUSurvivalMIMIC4()
    assert len(task.feature_names) == 101
    assert len(set(task.feature_names)) == 101
    assert task.feature_names[:6] == [
        "age",
        "is_male",
        "is_white",
        "heart_rate_min",
        "heart_rate_mean",
        "heart_rate_max",
    ]
    assert task.feature_names[27:32] == [
        "weight_kg",
        "height_cm",
        "albumin_min",
        "albumin_mean",
        "albumin_max",
    ]
    assert task.feature_names[-1] == "rdw_max"

    vitals = FirstDayICUSurvivalMIMIC4(feature_set="vitals")
    labs = FirstDayICUSurvivalMIMIC4(feature_set="labs")
    assert len(vitals.feature_names) == 29
    assert len(labs.feature_names) == 75
    assert vitals.feature_names == task.feature_names[:29]
    assert labs.feature_names == task.feature_names[:3] + task.feature_names[29:]

    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[(10, T0 + _hours(1), HEART_RATE, 80)],
        labs=[(T0 + _hours(1), LACTATE, 2.0)],
    )
    for candidate in (task, vitals, labs):
        (sample,) = candidate(patient)
        assert len(sample["features"]) == len(candidate.feature_names)
    assert _value(vitals, vitals(patient)[0], "heart_rate_mean") == 80.0
    assert _value(labs, labs(patient)[0], "lactate_mean") == 2.0


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="event_source"):
        FirstDayICUSurvivalMIMIC4(event_source="icu")
    with pytest.raises(ValueError, match="feature_set"):
        FirstDayICUSurvivalMIMIC4(feature_set="everything")
    with pytest.raises(ValueError, match="observation_hours"):
        FirstDayICUSurvivalMIMIC4(observation_hours=0)
    for not_whole_hours in (24.5, 24.0, "24", True):
        with pytest.raises(ValueError, match="observation_hours"):
            FirstDayICUSurvivalMIMIC4(observation_hours=not_whole_hours)


def test_schema_follows_the_processing_options():
    assert FirstDayICUSurvivalMIMIC4.task_name == "FirstDayICUSurvivalMIMIC4"
    default = FirstDayICUSurvivalMIMIC4()
    assert default.output_schema == {"duration": "tensor", "event": "tensor"}
    assert default.input_schema == {
        "features": ("tensor", {"normalization": "zscore", "impute_strategy": "zero"})
    }
    assert default.input_schema == FirstDayICUSurvivalMIMIC4.input_schema
    raw = FirstDayICUSurvivalMIMIC4(standardize=False)
    assert raw.input_schema == {"features": "tensor"}
    flagged = FirstDayICUSurvivalMIMIC4(missing_indicators=True)
    assert flagged.input_schema["features"][1] == {
        "normalization": "zscore",
        "impute_strategy": "zero",
        "missing_indicators": True,
    }


def test_stay_must_be_longer_than_the_window():
    task = FirstDayICUSurvivalMIMIC4()
    assert task(_patient([(10, 1, T0, 20)])) == []
    assert task(_patient([(10, 1, T0, 24)])) == []
    (sample,) = task(_patient([(10, 1, T0, 24.5)]))
    assert sample["stay_id"] == 10
    assert sample["patient_id"] == "p1"

    longer = FirstDayICUSurvivalMIMIC4(observation_hours=48)
    assert longer(_patient([(10, 1, T0, 36)])) == []
    assert len(longer(_patient([(10, 1, T0, 60)]))) == 1


def test_censored_stay_gets_icu_length_of_stay():
    task = FirstDayICUSurvivalMIMIC4()
    (sample,) = task(_patient([(10, 1, T0, 60)]))
    assert sample["event"] == [0.0]
    assert sample["duration"] == [pytest.approx(2.5)]


def test_death_gets_time_from_icu_admission_to_death():
    task = FirstDayICUSurvivalMIMIC4()
    death = T0 + _hours(84)
    (sample,) = task(_patient([(10, 1, T0, 90)], deathtimes={1: death}))
    assert sample["event"] == [1.0]
    assert sample["duration"] == [pytest.approx(3.5)]


def test_death_inside_the_window_is_excluded():
    task = FirstDayICUSurvivalMIMIC4()
    stays = [(10, 1, T0, 30)]
    assert task(_patient(stays, deathtimes={1: T0 + _hours(23)})) == []
    assert task(_patient(stays, deathtimes={1: T0 + _hours(24)})) == []
    assert len(task(_patient(stays, deathtimes={1: T0 + _hours(25)}))) == 1
    # With "any", a recorded date of death inside the window excludes the stay.
    any_death = FirstDayICUSurvivalMIMIC4(event_source="any")
    assert any_death(_patient(stays, dod="2150-03-01")) == []
    assert len(task(_patient(stays, dod="2150-03-01"))) == 1


def test_event_source_differs_only_on_out_of_hospital_death():
    stays = [(10, 1, T0, 48)]
    any_death = FirstDayICUSurvivalMIMIC4(event_source="any")
    hospital = FirstDayICUSurvivalMIMIC4()
    assert hospital.event_source == "hospital"

    # Died after discharge: only a date of death is recorded.
    outside = _patient(stays, dod="2150-03-11")
    (sample_any,) = any_death(outside)
    (sample_hospital,) = hospital(outside)
    assert sample_any["event"] == [1.0]
    expected_days = (datetime(2150, 3, 11) - T0).total_seconds() / 86400.0
    assert sample_any["duration"] == [pytest.approx(expected_days)]
    assert sample_hospital["event"] == [0.0]
    assert sample_hospital["duration"] == [pytest.approx(2.0)]
    assert sample_any["features"] == pytest.approx(
        sample_hospital["features"], nan_ok=True
    )

    # Died in hospital: both definitions agree, and the hospital time wins.
    death = T0 + _hours(40)
    inside = _patient(stays, dod="2150-03-03", deathtimes={1: death})
    assert any_death(inside)[0]["event"] == hospital(inside)[0]["event"] == [1.0]
    assert any_death(inside)[0]["duration"] == [pytest.approx(40 / 24)]
    assert hospital(inside)[0]["duration"] == [pytest.approx(40 / 24)]

    # No death recorded: both censor at ICU discharge.
    alive = _patient(stays)
    assert any_death(alive)[0]["event"] == hospital(alive)[0]["event"] == [0.0]


def test_statistics_use_only_measurements_inside_the_window():
    task = FirstDayICUSurvivalMIMIC4()
    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[
            (10, T0 - _hours(1), HEART_RATE, 200),  # before ICU admission
            (10, T0, HEART_RATE, 70),  # window start is inclusive
            (10, T0 + _hours(5), HEART_RATE, 90),
            (10, T0 + _hours(23.5), HEART_RATE, 110),
            (10, T0 + _hours(24), HEART_RATE, 250),  # window end is exclusive
            (10, T0 + _hours(30), HEART_RATE, 260),
            (99, T0 + _hours(2), HEART_RATE, 150),  # another stay's chart row
        ],
        labs=[
            (T0 - _hours(7), LACTATE, 9.0),  # before the 6-hour lookback
            (T0 - _hours(5), LACTATE, 4.0),  # inside the lookback
            (T0 + _hours(3), LACTATE, 2.0),
            (T0 + _hours(12), LACTATE, 1.5),
            (T0 + _hours(24), LACTATE, 8.0),  # window end is exclusive
        ],
    )
    (sample,) = task(patient)
    assert _value(task, sample, "heart_rate_min") == 70.0
    assert _value(task, sample, "heart_rate_mean") == pytest.approx(90.0)
    assert _value(task, sample, "heart_rate_max") == 110.0
    assert _value(task, sample, "lactate_min") == 1.5
    assert _value(task, sample, "lactate_mean") == pytest.approx(2.5)
    assert _value(task, sample, "lactate_max") == 4.0

    longer = FirstDayICUSurvivalMIMIC4(observation_hours=36)
    (sample,) = longer(patient)
    assert _value(longer, sample, "heart_rate_max") == 260.0
    assert _value(longer, sample, "lactate_max") == 8.0


def test_several_item_ids_feed_one_feature_and_units_are_converted():
    task = FirstDayICUSurvivalMIMIC4()
    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[
            (10, T0 + _hours(1), TEMP_C, 36.0),
            (10, T0 + _hours(2), TEMP_F, 100.4),  # 38.0 C
            (10, T0 + _hours(3), HEIGHT_IN, 70),  # 177.8 cm
        ],
    )
    (sample,) = task(patient)
    assert _value(task, sample, "temp_c_min") == pytest.approx(36.0)
    assert _value(task, sample, "temp_c_mean") == pytest.approx(37.0)
    assert _value(task, sample, "temp_c_max") == pytest.approx(38.0)
    assert _value(task, sample, "height_cm") == pytest.approx(177.8)


def test_implausible_and_non_numeric_values_are_dropped():
    task = FirstDayICUSurvivalMIMIC4()
    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[
            (10, T0 + _hours(1), HEART_RATE, 0),  # below 1
            (10, T0 + _hours(2), HEART_RATE, 80),
            (10, T0 + _hours(3), HEART_RATE, 300),  # range is inclusive
            (10, T0 + _hours(4), HEART_RATE, 999),  # above 300
            (10, T0 + _hours(5), HEART_RATE, None),
            (10, T0 + _hours(6), HEART_RATE, "abnormal"),
            (10, T0 + _hours(7), HEART_RATE, "nan"),
            (10, T0 + _hours(1), TEMP_F, 150),  # 65.6 C after conversion
            (10, T0 + _hours(2), TEMP_F, 98.6),  # 37.0 C: checked after conversion
            (10, T0 + _hours(1), HEIGHT_CM, 17),  # below 100
        ],
        labs=[(T0 + _hours(1), CREATININE, -1.0), (T0 + _hours(2), CREATININE, 1.2)],
    )
    (sample,) = task(patient)
    assert _value(task, sample, "heart_rate_min") == 80.0
    assert _value(task, sample, "heart_rate_mean") == pytest.approx(190.0)
    assert _value(task, sample, "heart_rate_max") == 300.0
    assert _value(task, sample, "temp_c_min") == pytest.approx(37.0)
    assert _value(task, sample, "temp_c_max") == pytest.approx(37.0)
    assert math.isnan(_value(task, sample, "height_cm"))
    assert _value(task, sample, "creatinine_min") == 1.2


def test_weight_and_height_take_the_first_value_in_the_window():
    task = FirstDayICUSurvivalMIMIC4()
    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[
            (10, T0 + _hours(9), WEIGHT, 82.0),
            (10, T0 + _hours(2), WEIGHT, 80.0),
            (10, T0 - _hours(2), WEIGHT, 75.0),
            (10, T0 + _hours(4), HEIGHT_CM, 180.0),
            (10, T0 + _hours(6), HEIGHT_IN, 60),
        ],
    )
    (sample,) = task(patient)
    assert _value(task, sample, "weight_kg") == 80.0
    assert _value(task, sample, "height_cm") == 180.0


def test_missing_features_are_nan_and_the_vector_keeps_its_length():
    task = FirstDayICUSurvivalMIMIC4()
    (sample,) = task(_patient([(10, 1, T0, 72)]))
    features = sample["features"]
    assert len(features) == 101
    assert not any(math.isnan(value) for value in features[:3])
    assert all(math.isnan(value) for value in features[3:])

    (sample,) = task(
        _patient([(10, 1, T0, 72)], chart=[(10, T0 + _hours(1), HEART_RATE, 75)])
    )
    observed = {
        name
        for name, value in zip(task.feature_names, sample["features"])
        if not math.isnan(value)
    }
    assert observed == {
        "age",
        "is_male",
        "is_white",
        "heart_rate_min",
        "heart_rate_mean",
        "heart_rate_max",
    }


def test_demographics_and_cohort_exclusions():
    task = FirstDayICUSurvivalMIMIC4()
    stays = [(10, 1, T0, 72)]
    (sample,) = task(_patient(stays, gender="M", race="WHITE - RUSSIAN"))
    # anchor_age 60 in anchor_year 2148, admitted in 2150.
    assert sample["features"][:3] == [62.0, 1.0, 1.0]
    (sample,) = task(_patient(stays, gender="F", race="ASIAN", admit_year=2149))
    assert sample["features"][:3] == [61.0, 0.0, 0.0]

    assert task(_patient(stays, gender=None)) == []
    assert task(_patient(stays, race=None)) == []
    assert task(_patient(stays, race="UNKNOWN")) == []
    assert task(_patient(stays, race="UNABLE TO OBTAIN")) == []


def test_patient_with_two_qualifying_stays():
    later = T0 + timedelta(days=30)
    stays = [(20, 2, later, 96), (10, 1, T0, 48), (30, 2, later + _hours(120), 12)]
    chart = [
        (10, T0 + _hours(1), HEART_RATE, 60),
        (20, later + _hours(1), HEART_RATE, 120),
    ]
    patient = _patient(stays, chart=chart, deathtimes={2: later + _hours(200)})

    task = FirstDayICUSurvivalMIMIC4()
    first, second = task(patient)
    assert (first["stay_id"], second["stay_id"]) == (10, 20)
    assert _value(task, first, "heart_rate_mean") == 60.0
    assert _value(task, second, "heart_rate_mean") == 120.0
    # In-hospital deaths belong to their own admission only.
    assert (first["event"], second["event"]) == ([0.0], [1.0])
    assert second["duration"] == [pytest.approx(200 / 24)]

    only_first = FirstDayICUSurvivalMIMIC4(first_stay_only=True)
    (sample,) = only_first(patient)
    assert sample["stay_id"] == 10

    # The earliest stay does not qualify, so the next one is used.
    short_first = _patient([(10, 1, T0, 5), (20, 2, later, 96)])
    (sample,) = only_first(short_first)
    assert sample["stay_id"] == 20


def test_patients_without_usable_records_give_no_samples():
    task = FirstDayICUSurvivalMIMIC4()
    no_stay = _patient([])
    assert task(no_stay) == []
    # A stay whose hospital admission is not in the data is skipped.
    patient = _patient([(10, 1, T0, 72)])
    frame = patient.data_source.filter(pl.col("event_type") != "admissions")
    assert task(Patient(patient_id="p1", data_source=frame)) == []


def _event_table():
    patient = _patient(
        [(10, 1, T0, 72)],
        chart=[
            (10, T0 + _hours(1), HEART_RATE, 80),
            (10, T0 + _hours(1), 999999, 1),  # an item the task does not use
        ],
        labs=[(T0 + _hours(1), LACTATE, 2.0), (T0 + _hours(1), 888888, 5.0)],
    )
    return patient.data_source


def test_pre_filter_keeps_only_the_needed_measurements():
    task = FirstDayICUSurvivalMIMIC4()
    table = _event_table()
    filtered = task.pre_filter(table.lazy()).collect()
    assert filtered.height == table.height - 2
    counts = dict(filtered.group_by("event_type").len().iter_rows())
    assert counts == {
        "patients": 1,
        "admissions": 1,
        "icustays": 1,
        "chartevents": 1,
        "labevents": 1,
    }
    assert filtered["chartevents/itemid"].drop_nulls().to_list() == [str(HEART_RATE)]
    assert filtered["labevents/itemid"].drop_nulls().to_list() == [str(LACTATE)]

    vitals = FirstDayICUSurvivalMIMIC4(feature_set="vitals")
    counts = dict(
        vitals.pre_filter(table.lazy())
        .collect()
        .group_by("event_type")
        .len()
        .iter_rows()
    )
    assert counts["chartevents"] == 1
    assert "labevents" not in counts


def test_pre_filter_requires_the_measurement_tables():
    table = _event_table()
    without_labs = table.filter(pl.col("event_type") != "labevents").drop(
        [name for name in table.columns if name.startswith("labevents/")]
    )
    with pytest.raises(ValueError, match="labevents"):
        FirstDayICUSurvivalMIMIC4().pre_filter(without_labs.lazy())
    with pytest.raises(ValueError, match="labevents"):
        FirstDayICUSurvivalMIMIC4(feature_set="labs").pre_filter(without_labs.lazy())
    vitals = FirstDayICUSurvivalMIMIC4(feature_set="vitals")
    assert vitals.pre_filter(without_labs.lazy()).collect().height == 4


def _samples(task):
    samples = []
    for index, (hours, heart_rate, death_hour) in enumerate(
        [(60, 70, None), (90, 95, 80), (48, None, None), (120, 120, 100)]
    ):
        chart = (
            [] if heart_rate is None else [(10, T0 + _hours(2), HEART_RATE, heart_rate)]
        )
        deathtimes = {} if death_hour is None else {1: T0 + _hours(death_hour)}
        (sample,) = task(_patient([(10, 1, T0, hours)], chart, deathtimes=deathtimes))
        samples.append({**sample, "patient_id": f"p{index}"})
    return samples


def test_samples_feed_the_processors_and_a_survival_model():
    task = FirstDayICUSurvivalMIMIC4()
    dataset = create_sample_dataset(
        samples=_samples(task),
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="first_day_icu_survival_test",
    )
    batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
    features = batch["features"]
    assert features.shape == (4, 101)
    assert torch.isfinite(features).all()
    index = task.feature_names.index("heart_rate_mean")
    # Standardized with the observed values; the missing one sits at the mean.
    assert features[2, index] == 0.0
    assert features[[0, 1, 3], index].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert features[0, index] < features[1, index] < features[3, index]
    assert batch["duration"].reshape(-1).tolist() == pytest.approx(
        [2.5, 80 / 24, 2.0, 100 / 24]
    )
    assert batch["event"].reshape(-1).tolist() == [0.0, 1.0, 0.0, 1.0]

    model = HeterogeneousSurvivalMDN(
        dataset,
        hidden_dim=8,
        component_families=("gaussian", "logistic"),
        time_grid=torch.linspace(0.5, 10.0, 7),
    )
    assert model.input_dim == 101
    output = model(**batch)
    assert torch.isfinite(output["loss"])
    assert output["y_prob"].shape == (4, 7)
    output["loss"].backward()


def test_missing_indicators_double_the_processed_width():
    task = FirstDayICUSurvivalMIMIC4(feature_set="vitals", missing_indicators=True)
    dataset = create_sample_dataset(
        samples=_samples(task),
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="first_day_icu_survival_indicator_test",
    )
    features = next(iter(get_dataloader(dataset, batch_size=4)))["features"]
    assert features.shape == (4, 58)
    index = task.feature_names.index("heart_rate_mean")
    assert features[:, 29 + index].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert features[:, 29].tolist() == [0.0, 0.0, 0.0, 0.0]
