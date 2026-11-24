pyhealth.tasks.FirstDayICUSurvivalMIMIC4
========================================

Time from ICU admission to death on MIMIC-IV, predicted from summaries of the
first 24 hours of the stay and right-censored at ICU discharge. This is the
MIMIC-IV cohort of Han, Goldstein and Ranganath, `Survival Mixture Density
Networks <https://proceedings.mlr.press/v182/han22a.html>`_ (MLHC 2022). Each
sample holds one flat ``features`` vector, a ``duration`` in days and an
``event`` indicator, the format consumed by
:class:`~pyhealth.models.SurvivalMDN` and
:class:`~pyhealth.models.HeterogeneousSurvivalMDN`.

Cohort and labels
-----------------

One sample per ICU stay that is longer than the observation window (24 hours
by default), whose patient has a recorded sex and a known race and is alive at
the end of the window. ``duration`` is the time from ICU admission to death, or
to ICU discharge when no death is recorded. ``event_source`` selects which
deaths count:

* ``"any"`` (default, the paper's definition): the death time of the stay's
  hospital admission, otherwise the patient's recorded date of death, which
  also covers deaths after discharge.
* ``"hospital"``: only deaths during the stay's hospital admission.

A patient with several qualifying stays gives several samples (or only the
earliest with ``first_stay_only=True``), so split by patient.

Features
--------

``feature_names`` lists the entries of ``features`` in order: age, ``is_male``
and ``is_white``; the minimum, mean and maximum of 8 charted vitals (heart
rate, systolic, diastolic and mean blood pressure, respiratory rate,
temperature, oxygen saturation, glucose); the first charted weight and height;
and the minimum, mean and maximum of 24 laboratory values. ``feature_set``
selects ``"all"`` (101 values), ``"vitals"`` (29) or ``"labs"`` (75).

Vitals, weight and height come from ``chartevents`` rows of the stay inside the
window. Labs come from ``labevents`` from 6 hours before ICU admission to the
end of the window. Values outside the plausible ranges in
``PLAUSIBLE_RANGES`` are dropped, Fahrenheit temperatures are converted to
Celsius and heights in inches to centimetres. A feature without a measurement
is NaN in the sample; with ``standardize=True`` the processor z-scores each
feature on the observed training values and fills missing values with zero.
Use ``set_task(task, split=...)`` so the statistics come from the training
patients only.

Relation to other components
----------------------------

.. list-table::
   :header-rows: 1

   * -
     - :class:`~pyhealth.tasks.InHospitalSurvivalMIMIC4`
     - ``FirstDayICUSurvivalMIMIC4``
   * - Source setting
     - TrajSurv
     - Survival MDN
   * - Observation window
     - First 36 hours
     - First 24 hours
   * - Flat features
     - Last value of each YAIB variable
     - Minimum, mean and maximum of vitals and labs
   * - Time origin
     - End of the window
     - ICU admission
   * - Event
     - In-hospital death
     - Any recorded death, or in-hospital death
   * - Censoring
     - Hospital discharge
     - ICU discharge

Compared with the paper's own cohort, the features here are aggregated from
the raw ``chartevents`` and ``labevents`` tables, not read from the mimic-code
derived tables, and the SOFA score is not included. No feature is removed for
being rarely measured and no outlier clipping across patients is applied;
both would need statistics of the whole cohort, which a per-patient task does
not have.

.. autoclass:: pyhealth.tasks.FirstDayICUSurvivalMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:
