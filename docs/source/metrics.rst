===================
Reading the results
===================

``evaluate`` and ``compare`` report a small set of numbers per model. This page
defines each one, says which direction is better, and lists the caveats that
matter when you use them to choose between models.

All metrics are computed on the rows in your simulation bundle, with labels
from ``--labels`` and ``--positive-group-id``.

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Field
     - Definition
     - Reading it
   * - ``baselines.roc_auc``
     - ROC AUC of the model on the clean cohort.
     - Ordinary discrimination. Higher is better; 0.5 is chance.
   * - ``metric_based.roc_auc.resiliency``
     - ROC AUC on the blackout variant ÷ ROC AUC on the clean cohort.
     - 1.0 means missing the feature cost nothing. 0.9 means the model kept
       90% of its AUC. Higher is better.
   * - ``metric_invariant.jitter_stability``
     - ``1 / (1 + 100 × v)``, where ``v`` is the per-patient variance of the
       predicted probability across the jitter variants, averaged over
       patients.
     - 1.0 means predictions did not move under noise. Lower means individual
       patients' scores move when the measurement is noisy. Higher is better.
   * - ``metric_invariant.flipper_stability``
     - For each sampled pair of opposite-label patients, the position ``t``
       (0 → 1) along the straight line between them where the prediction first
       crosses ``--flip-threshold``. A pair that never crosses counts as 1.0.
       Reported as the mean over pairs.
     - Describes *where* the decision boundary sits between real patients. It
       is a description, not a score. Compare models against each other,
       not against a target value.
   * - ``metric_based.roc_auc.generalizability``
     - **Not implemented.** Always ``1.0`` in this release.
     - Ignore it.

Caveats
-------

**One simulation type per run.** ``evaluate blackout`` loads only
``blackout/``. Metrics for simulation types that were not loaded are reported
at their neutral value: ``resiliency`` and ``jitter_stability`` are ``1.0``,
and ``flipper_stability`` is ``null``. Run once per simulation type and read
the matching metric from each run:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Run
     - Metric to read
   * - ``evaluate blackout``
     - ``resiliency``
   * - ``evaluate jitter``
     - ``jitter_stability`` (needs ``--jitter-iters`` of 2 or more)
   * - ``evaluate flipper``
     - ``flipper_stability``

**ROC AUC needs both classes.** If the clean cohort has no positive rows,
or no negative rows, ROC AUC is reported as 0.5 rather than as an error.
Check your label counts first.

**Tied scores count half.** ROC AUC counts a tied positive/negative pair as
half a win, whatever order the rows are in. Before this release ties were
broken by row order, which mattered for models that emit many identical
scores (tree ensembles, models fed a constant blackout fill).

**Non-integer IDs are labelled negative.** A row is positive only when its ID
parses as an integer and appears in ``member_ids``. Identifiers such as
``A00123`` are silently treated as negative. Map them to integers before you
run Pasteur.

**Missing values reach the model as NaN.** Unless the model declares an
``absent_sentinel`` or you pass ``--null-fill``, blacked-out values are sent
to the model as NaN. Most exported scikit-learn models cannot handle NaN and
Pasteur stops with an error instead of reporting bad numbers. The simplest fix
is to include the imputer in the exported pipeline; see
:doc:`model-selection`.

Per-patient predictions
-----------------------

``compare --predictions-out`` writes one row per patient per variant:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Meaning
   * - ``source_row_id``
     - The patient's original ID (synthetic path ID for flipper rows)
   * - ``variant_name``
     - ``clean``, ``blackout``, ``jitter_0`` … , or ``flipper``
   * - ``label``
     - 1 if the patient is in the positive group, else 0
   * - one column per model
     - Positive-class probability. Named after the model file, without
       ``.onnx``.
   * - ``all_agree``
     - True when every model puts the patient on the same side of 0.5
       (fixed; ``--flip-threshold`` does not change it)

For multiclass and multilabel models, ``label`` becomes one
``label__<class>`` column per class, and each model gets one
``<model>__<class>`` column per class. ``all_agree`` compares the argmax class
(multiclass) or the set of labels at or over each model's thresholds
(multilabel). Compared models must list the same ``output.classes``.

Use it to find the specific patients whose score moves, and to review them
with clinicians.

Multiclass and multilabel models
--------------------------------

For a model with K outputs, every metric is computed per class (multiclass)
or per label (multilabel). The headline fields above hold the **macro
average** over classes, so they read on the same scale as a binary model's.
The extra ``multi`` section of the JSON holds what the averages hide. It is
absent for binary models.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - How it generalizes
   * - ROC AUC
     - Each class against the rest (multiclass), or each label on its own
       (multilabel). ``baselines.roc_auc`` is the mean over labels whose AUC is
       defined. A label with only one class in the clean cohort has no AUC: it
       is listed in ``multi.excluded_labels`` and left out of the mean rather
       than counted as 0.5. ``multi.roc_auc_micro`` pools every
       (patient, label) pair, so common labels dominate it.
   * - Resiliency
     - Blackout AUC ÷ clean AUC for each label, then averaged. Labels whose
       clean AUC is 0.5 or below are skipped: a ratio against chance-level
       performance means nothing. ``multi.worst_label_resiliency`` is the
       minimum, because blacking out one assay often wrecks a single label
       while the average barely moves.
   * - Jitter stability
     - ``v`` is averaged over the K columns before the ``1 / (1 + 100 × v)``
       squash. For a 2-class softmax this equals the binary score, so values
       stay comparable. ``multi.decision_flip_rate`` is the share of patients
       whose *decision* changes between jitter draws: the argmax class
       (multiclass) or the set of labels at or over threshold (multilabel).
   * - Flipper stability
     - Multiclass: pairs are drawn from two different classes a and b, and
       the flip is where ``p[b] − p[a]`` crosses 0; ``--flip-threshold`` is
       not used. ``multi.flipper_detour_rate`` is the share of paths whose
       argmax passes through a third class. Multilabel: each pair is sampled
       *for one label*, a negative and a positive patient for that label, and
       the flip is where that label crosses its threshold (contract
       ``output.thresholds``, else ``--flip-threshold``).
       ``multi.per_label[].flipper_stability`` averages the pairs about each
       label.

**Read ``flipper_never_flipped`` next to ``flipper_stability``.** Pairs that
never cross count as 1.0 in the mean, so a model that rarely changes its mind
along these paths scores high for that reason alone.
``multi.flipper_never_flipped`` reports the share.

**Regenerate the flipper grid when the task or groups change.** Multiclass
grids store class indices in ``--positive-group-id`` order, and multilabel
grids store the label index in ``pair_label``. Evaluating a grid built for a
different task is an error. Evaluating one built with the same task but
different groups is not detected.

What these numbers do not show
------------------------------

They describe behaviour under the perturbations you declared: one feature
removed at one rate, noise at one scale, straight-line paths between real
patients. They do not measure calibration, fairness across subgroups,
performance on another site's population, or clinical benefit.
