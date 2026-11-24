pyhealth.models.HeterogeneousSurvivalMDN
========================================

Overview
--------

``HeterogeneousSurvivalMDN`` is a Survival MDN whose mixture components can be
Gaussian, generalized logistic, or a combination of the two. As in Han,
Goldstein and Ranganath, `Survival Mixture Density Networks
<https://proceedings.mlr.press/v182/han22a.html>`_ (MLHC 2022), the mixture is
placed on an unconstrained latent time and mapped to positive event time, and
training minimizes the right-censored negative log-likelihood.

Component families
------------------

``component_families`` lists the family of each component; its length is the
number of components. With latent time ``y`` and ``z = (y - mean) / scale``:

* ``"gaussian"`` has latent survival ``Phi(-z)``, as in
  :class:`~pyhealth.models.SurvivalMDN`.
* ``"logistic"`` is a generalized logistic component with latent survival
  ``sigmoid(-z) ** shape`` and density
  ``(shape / scale) * sigmoid(-z) ** shape * sigmoid(z)``. The shape is
  predicted per sample and per component as ``softplus(raw) + min_shape``.
  A shape of one gives the standard logistic distribution; other values make
  the component asymmetric. Its tails are heavier than a Gaussian's.

For example, ``("gaussian",) * 5 + ("logistic",) * 5`` (the default) mixes five
components of each family, and ``("logistic",) * 10`` uses logistic components
only. With only Gaussian components the model computes the same loss and
survival curves as :class:`~pyhealth.models.SurvivalMDN` given the same
weights.

Relation to SurvivalMDN
-----------------------

The feature network, the weight, mean and scale heads, the reference
initialization and the softplus and exponential time transforms are reused
from ``pyhealth.models.survival_mdn``. This model adds a shape head and the
mixed-family likelihood. A ``SurvivalMDN`` state dict loads into an
all-Gaussian ``HeterogeneousSurvivalMDN`` with ``strict=False``; only the
shape head is absent from it.

Inputs and outputs
------------------

The dataset contract is that of :class:`~pyhealth.models.SurvivalMDN`: input
schema ``features: tensor`` and output schema ``duration: tensor, event:
tensor``, with strictly positive durations and an event indicator that is one
for an observed event and zero for censoring.

``forward`` returns the mean censored NLL in ``loss``, ``y_true`` with columns
duration and event, survival curves in ``y_prob`` with shape
``[batch, len(time_grid)]``, and the mixture weights, means, scales and
shapes. ``mixture_shapes`` is one for Gaussian components. ``logit``
concatenates the raw weight, mean, scale and shape head outputs and has shape
``[batch, 4 * num_components]``; it is not a classification score. Supply a
``time_grid`` in the units of the dataset's durations.

``predict_survival`` temporarily uses evaluation mode, disables gradients and
restores each module's previous mode. BatchNorm requires at least two samples
per training minibatch. ``mode=None`` is intentional: use a survival training
loop and :func:`pyhealth.metrics.survival_curve_metrics_fn`.

API
---

.. autoclass:: pyhealth.models.HeterogeneousSurvivalMDN
   :members:
   :show-inheritance:
