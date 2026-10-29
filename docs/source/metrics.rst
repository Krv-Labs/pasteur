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
   * - ``evaluate sweep``
     - ``sweep`` (see `Sweep`_)

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
     - The patient's original ID (synthetic path ID for flipper and sweep
       rows)
   * - ``variant_name``
     - ``clean``, ``blackout``, ``jitter_0`` … , ``flipper``, or ``sweep``
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
     - Multilabel: ``v`` is averaged over the labels before the
       ``1 / (1 + 100 × v)`` squash, like every other per-label score.
       Multiclass: ``v`` is the per-class variances *summed* and halved,
       because probability leaving one class arrives in another. For a
       2-class softmax this equals the binary score, and adding a class the
       model never predicts does not change it. (An average over K would make
       the same movement look more stable for every extra class.)
       ``multi.decision_flip_rate`` is the share of patients
       whose *decision* changes between jitter draws: the argmax class
       (multiclass) or the set of labels at or over threshold (multilabel).
   * - Flipper stability
     - Multiclass: pairs are drawn from two different classes a and b, and
       the flip is where ``p[b] − p[a]`` crosses 0; ``--flip-threshold`` is
       not used. ``multi.flipper_detour_rate`` is the share of paths whose
       argmax passes through a third class. For ordered classes, such as
       normal, prediabetes and diabetes, a path from the first to the last
       *should* pass through the middle one, so a high detour rate there is
       expected rather than a sign of instability. Multilabel: each pair is sampled
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

Sweep
-----

``evaluate sweep`` adds a ``sweep`` list to the JSON, one entry per swept
feature. A *path* is one source row swept over one feature. The model's
decision at each step is the positive class at or over ``--flip-threshold``
(binary), the argmax class (multiclass), or the set of labels at or over their
thresholds (multilabel). A *flip* is a step where the decision differs from
the step before, and its *flip value* is the midpoint of the two bracketing
grid values. Sweep is scored for classifiers only.

The full metrics compare the model with an **oracle**: the correct decision
for every grid row, because moving a feature across its cutoff can change the
correct answer. Rates are shares of paths.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Definition
   * - ``boundary_offset``
     - Median, ``q25`` and ``q75`` of ``(model flip value − oracle flip
       value) / |cutoff|``, over monotonic paths. 0 means the model's boundary
       is where the true one is; positive means it flips later on the grid.
       ``boundary_offset_units`` is the same in the feature's units. Null
       without a cutoff (for the relative form) or without monotonic paths.
   * - ``monotonic_rate``
     - Paths where the model and the oracle each flip exactly once, to the
       same decision. Higher is better.
   * - ``never_flipped``
     - Among paths where the oracle flips, the share where the model never
       does. Read it as "ignores this feature". Lower is better.
   * - ``extra_flips_rate``
     - Paths where the model flips more than once. A model that reads digit
       patterns rather than magnitudes flips back and forth. Lower is better.
   * - ``accuracy``, ``accuracy_by_step``
     - Accuracy against the oracle over every grid row, and at each step
       (with that step's ``value``). The dip around the cutoff shows how wide
       the model's uncertain band is.
   * - ``model_flip_value``
     - Median and quartiles of the model's flip value, in feature units, over
       paths where it flips exactly once.

**Without oracle labels the output is descriptive.** Until per-row oracle
labels are supplied, ``oracle`` is ``false`` and only ``model_flip_value``,
``extra_flips_rate`` and ``never_flipped`` are filled. ``never_flipped`` then
counts every path whose decision never changes, so both ends of the grid get
the same answer. A model may be right not to flip: read it against a range
that you know crosses the cutoff. ``boundary_offset``, ``monotonic_rate`` and
the accuracies are ``null``.

**The grid limits the resolution.** Flip values are grid midpoints, so an
offset is exact only to within one grid step. Use ``--sweep-near`` to add
points close to the cutoff.

Regression models
-----------------

A single-target regression model (``--task regression``) predicts a value in
the target's units instead of a probability. The JSON keeps the same shape,
with these fields:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Definition
   * - ``baselines.rmse``, ``baselines.mae``, ``baselines.r2``
     - Root-mean-square error, mean absolute error, and R² on the clean
       cohort. RMSE and MAE are in the target's units; lower is better. R² is
       1 for a perfect model, 0 for one no better than predicting the mean,
       and negative for one worse than that.
   * - ``metric_based.r2.resiliency``
     - Blackout R² ÷ clean R²: the share of the model's skill over predicting
       the cohort mean that survives blackout. 1.0 means blackout cost
       nothing; 0 means the model fell back to no better than the mean;
       negative means blackout made it worse than the mean. It can exceed
       1.0 when blackout happens to help, which usually means the
       blacked-out feature was hurting the model. A model whose clean R² is
       0 or below has no skill to keep and is reported at the neutral 1.0;
       read ``baselines.r2`` first.
   * - ``metric_invariant.jitter_stability``
     - ``1 / (1 + 100 × v / var(target))``, where ``v`` is the per-patient
       variance of the prediction across jitter variants, averaged over
       patients. Dividing by the target's variance makes the score
       unit-free: the same model scores the same in mmol/mol and in %.
   * - ``metric_invariant.flipper_stability``
     - For each pair of one patient whose true target is below the
       ``--flip-threshold`` cutoff and one at or above it, the position ``t``
       (0 → 1) where the *prediction* first crosses the cutoff. Pairs that
       never cross count as 1.0.

**Why an R² ratio and not an RMSE ratio.** An error ratio rewards a model
for having little accuracy to lose. When a model loses a feature entirely and
falls back to the mean, clean RMSE ÷ blackout RMSE is ``sqrt(1 − R²)``: about
0.58 for a model with R² 0.66, but 0.86 for one with R² 0.26. On a synthetic
HbA1c cohort with glucose blacked out for every patient, a 1-nearest-neighbour
model dropped from R² 0.26 to −0.09 (worse than predicting the mean) and still
had the best RMSE ratio, 0.82, of the models that used glucose. Its R² ratio
is −0.35, the worst.

The ``regression`` section reports the same behaviour in the target's units:
``target_sd``, ``blackout_rmse``, ``blackout_r2``, ``jitter_prediction_sd`` (the mean
per-patient standard deviation of the prediction across jitter draws),
``flip_threshold``, and ``flipper_never_flipped``. It is absent for
classifiers.

**The flipper cutoff is a clinical choice.** There is no decision boundary in
a regression model, so flipper needs one: the threshold your service acts on,
such as HbA1c 6.5%. Pass the same ``--flip-threshold`` to ``simulate`` and to
``evaluate``. A grid evaluated against a cutoff that some pair does not
straddle is rejected. Without a cutoff, ``simulate`` skips flipper.

**Read ``flipper_never_flipped`` here too.** Pairs are chosen by their true
targets, but crossings are read from predictions. A model whose predictions
stay on one side of the cutoff for patients just above it never crosses, and
those pairs count as 1.0. Regression models shrink towards the mean, so when
the cutoff sits in the tail of the target this is common: on a synthetic
cohort with 8% of HbA1c values at or above 6.5%, 57–68% of pairs never
crossed for well-fitted models, and those pairs dominate the mean.

**Jitter depends on the cohort.** Because jitter is scaled by the target's
variance in the clean cohort, compare jitter stability between models on the
same cohort only.

.. _oracle-labels:

Oracle labels
-------------

Every metric above assumes a perturbation keeps the patient's label: a
jittered or blacked-out row is scored against the clean row's label. For a
*decision model* (a rule pack, an encoder or an LLM answering "eligible / not
eligible / cannot determine") that is often wrong. Moving a lab value across a
criterion's threshold should change the answer, and removing it may make the
right answer "cannot determine".

``--oracle-labels`` supplies the correct label for **every simulated row**,
computed by your own oracle (for example the rule engine or generator that
made the case). The file is a parquet with one row per simulated row:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Meaning
   * - ``variant_name``
     - ``clean``, ``blackout``, ``jitter_0`` … , or ``flipper``: the parquet
       file stem, as ``compare --predictions-out`` writes it
   * - ``source_row_id``
     - The row's ID; ``pair_<id>_step_<n>`` for flipper rows
   * - ``label``
     - Binary: 1 or 0
   * - ``group_<id>``
     - Multiclass and multilabel, one per ``--positive-group-id`` in the same
       order: a one-hot (multiclass) or multi-hot (multilabel) 0/1 row

It replaces the labels of every variant, ``clean`` included, so it must cover
every row of the clean data and of the simulation type being evaluated. A
missing row is an error. Other columns are ignored, and so are rows for
variants not loaded, so one file can serve every simulation type. Regression
is not supported.

The labels ``--predictions-out`` writes then come from the oracle too.

With oracle labels the JSON gains an ``oracle`` section. All of it compares
**decisions**, not probabilities: ``p ≥ --flip-threshold`` for binary, the
argmax for multiclass, and the set of labels at or over threshold for
multilabel.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Definition and reading
   * - ``oracle.accuracy.<variant>``
     - Share of rows whose decision equals the oracle label, per variant
       (``clean``, ``blackout``, ``jitter_0`` …, ``flipper``). Multilabel
       counts a row as correct only when every label matches. Under blackout
       this is the accuracy against the oracle's answer for the row with the
       feature missing, which may itself be "cannot determine".
   * - ``oracle.jitter.flip_rate_label_preserved``
     - Over every (row, jitter draw) where the oracle label is the same as on
       the clean row: the share where the model's decision changed anyway.
       **Lower is better**; this is instability the truth does not justify.
   * - ``oracle.jitter.flip_rate_label_changed``
     - Over every (row, jitter draw) where the oracle label changed: the
       share where the model's decision changed *to the new label*. **Higher
       is better**; it is the model following the truth.
   * - ``oracle.jitter.n_label_preserved``, ``n_label_changed``
     - The denominators. A rate is ``null`` when its count is 0.
   * - ``oracle.flipper.boundary_error_median``, ``boundary_error_iqr``
     - For each pair, ``flip_t − oracle_flip_t``: where the model's decision
       flips along the path minus where the oracle's does. Median and
       interquartile range over the pairs where both flip
       (``n_pairs_compared``). 0 means the model's boundary is where the true
       one is; negative means the model flips early, positive late.
   * - ``oracle.flipper.oracle_never_flipped``
     - Share of pairs whose oracle label never changes along the path.
       Those pairs have no true boundary to compare against.
   * - ``oracle.abstain``
     - With ``--abstain-class``, on blackout rows.
       ``abstain_when_missing``: of the rows where the oracle abstains, the
       share where the model abstains too (higher is better).
       ``decided_when_missing``: the share where it chose another class
       instead. ``confusion`` holds the four row counts
       ``oracle_<abstain|decided>_model_<abstain|decided>``;
       ``oracle_decided_model_abstain`` is the model abstaining when the
       oracle could decide.

**Where the oracle flips.** Oracle labels are 0/1, so ``oracle_flip_t`` lands
halfway between the last grid step before the change and the first one
after. With ``--flipper-steps 20`` that is within 1/38 of the true crossing;
add steps when the boundary error you care about is smaller than that.
The model's ``flip_t`` is interpolated from its probabilities as for
``flipper_stability``. Multiclass uses the same ``p[b] − p[a]`` rule for both,
so an oracle label of a third class counts as the ``b`` side.

**Pairs where only one side flips.** A pair where the model flips but the
oracle does not, or the reverse, is left out of ``boundary_error``. Read
``oracle_never_flipped`` and ``n_pairs_compared`` beside it.

**Jitter rates pool draws.** Each (row, draw) counts once, against the clean
row. A row that flips in two draws counts twice.

**The abstain class is matched by name** against the model's
``output.classes``, and must be one of them. It applies to multiclass and
multilabel models; a binary model has no class to abstain with.

What these numbers do not show
------------------------------

They describe behaviour under the perturbations you declared: one feature
removed at one rate, noise at one scale, straight-line paths between real
patients. They do not measure calibration, fairness across subgroups,
performance on another site's population, or clinical benefit.
