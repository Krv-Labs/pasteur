=====================================
Walkthrough: choosing between models
=====================================

A typical use: a research team has trained several candidate risk models for
the same prediction task and wants to know which one stays reliable when a lab
is missing or noisy, before any of them go to clinical review.

This walkthrough compares five candidate models on one laptop, with no data
leaving it. Every command on this page has been run as written on a synthetic
cohort.

1. Prepare the inputs
---------------------

Pasteur needs three things, all as local files:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - File
     - Contents
   * - ``cohort.parquet``
     - One row per patient: an integer ID column plus the model's input
       features, **in training order**, and nothing else. Every non-ID column
       is sent to the model.
   * - ``groups.parquet``
     - Which patient IDs are positive. See :doc:`data-contracts`.
   * - ``*.onnx``
     - One ONNX file per candidate model, each with a **different file name**.

Pasteur reads parquet, not CSV. If your cohort is a CSV with an outcome
column, this converts it and builds the labels file (requires ``polars``):

.. code-block:: python

   import polars as pl

   cohort = pl.read_csv("cohort.csv")

   # Model inputs only: the id column plus the features, in training order.
   features = ["age", "wbc", "crp", "crp_measured", "glucose"]
   cohort.select(["patient_id", *features]).write_parquet("cohort.parquet")

   # Labels: one row per cohort, listing the patient ids in it.
   positives = cohort.filter(pl.col("outcome") == 1)["patient_id"].to_list()
   pl.DataFrame(
       {"group_id": [1], "member_ids": [positives]},
       schema={"group_id": pl.UInt32, "member_ids": pl.List(pl.Int64)},
   ).write_parquet("groups.parquet")

Export the models to ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~

Pasteur scores ONNX models only. Convert scikit-learn models with
`skl2onnx <https://onnx.ai/sklearn-onnx/>`_ and PyTorch models with
``torch.onnx.export``. For scikit-learn:

.. code-block:: python

   from skl2onnx import to_onnx
   from skl2onnx.common.data_types import FloatTensorType

   # pipeline = make_pipeline(SimpleImputer(), StandardScaler(), classifier)
   onx = to_onnx(
       pipeline,
       initial_types=[("input", FloatTensorType([None, len(features)]))],
       options={id(pipeline.steps[-1][1]): {"zipmap": False}},
   )
   open("models/logreg.onnx", "wb").write(onx.SerializeToString())

Three details matter:

- Name the input tensor ``input`` (Pasteur's default), or pass
  ``--input-name``.
- Put the **imputer inside the pipeline** so the exported model handles
  missing values itself. Otherwise blackout fails with a "not NaN-native"
  error; see :doc:`metrics`.
- Give each model its own file name. Pasteur labels models by file name, so
  ``a/model.onnx`` and ``b/model.onnx`` would collide.

Optionally, save a ``metadata.json`` next to the models so Pasteur checks the
column order before scoring:

.. code-block:: json

   {"input": {"feature_order": ["age", "wbc", "crp", "crp_measured", "glucose"]}}

2. Simulate
-----------

Choose the feature whose missingness and noise you want to test, here ``crp``,
together with its ``crp_measured`` flag:

.. code-block:: bash

   pasteur-cli simulate \
     --input cohort.parquet \
     --labels groups.parquet \
     --positive-group-id 1 \
     --id-col patient_id \
     --feature crp \
     --blackout-companion crp_measured \
     --blackout-rate 0.2 \
     --jitter-scale 1.0 \
     --jitter-iters 5 \
     --flipper-pairs 100 \
     --flipper-steps 20 \
     --output ./sim

This writes ``sim/clean/``, ``sim/blackout/``, ``sim/jitter/`` (five
variants), and ``sim/flipper/``. ``--jitter-scale`` is in units of the
feature's standard deviation.

Always pass ``--id-col``, ``--feature``, and ``--positive-group-id``: their
defaults (``node_id``, ``TSH``, ``103``) belong to the example thyroid dataset.

3. Compare the models
---------------------

Run ``compare`` once per simulation type:

.. code-block:: bash

   mkdir -p results
   for sim in blackout jitter flipper; do
     pasteur-cli compare "$sim" \
       --sim-root ./sim \
       --labels groups.parquet \
       --positive-group-id 1 \
       --model models/logreg.onnx \
       --model models/forest.onnx \
       --model models/gbm.onnx \
       --model models/tree.onnx \
       --model models/knn.onnx \
       --predictions-out "results/$sim-predictions.parquet" \
       --output "results/$sim.json"
   done

Create the ``--output`` directory first; ``compare`` does not create it.

4. Read the results
-------------------

Each ``results/<sim>.json`` holds one entry per model:

.. code-block:: json

   {
     "evaluations": [
       {
         "model_label": "gbm",
         "evaluation": {
           "baselines": {"roc_auc": 0.991},
           "evaluations": {
             "metric_invariant": {"jitter_stability": 1.0, "flipper_stability": null},
             "metric_based": {"roc_auc": {"resiliency": 0.957, "generalizability": 1.0}}
           },
           "created_at": "2026-09-30T11:50:26Z"
         }
       }
     ]
   }

Take one metric from each run and put them side by side. From the synthetic
run:

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Model
     - Clean AUC
     - Resiliency (blackout)
     - Jitter stability
     - Flipper stability
   * - logreg
     - 0.800
     - 0.952
     - 0.287
     - 0.771
   * - forest
     - 1.000
     - 0.966
     - 0.297
     - 0.611
   * - gbm
     - 0.991
     - 0.957
     - 0.250
     - 0.577
   * - tree
     - 0.843
     - 0.892
     - 0.179
     - 0.588
   * - knn
     - 0.814
     - 0.941
     - 0.370
     - 0.730

Here the decision tree loses the most AUC when ``crp`` is missing and moves
the most under noise. The forest and gradient-boosted models discriminate best
and hold up under blackout. Every model is sensitive to ``crp`` noise at one
standard deviation, which is itself a finding to take to clinicians. (These
are synthetic numbers; do not read them as typical.)

Then open ``results/<sim>-predictions.parquet`` and filter on
``all_agree == false`` to see the specific patients the models disagree on.
Those rows are the concrete cases to review with the clinical team.

See :doc:`metrics` for exact definitions and caveats.

5. Keep a record
----------------

For each run, keep the Pasteur version (``pasteur-cli --version``), the exact
commands, a checksum of each input file, and the ``metadata.json`` of each
model. Without them the numbers cannot be reproduced. Simulations are
deterministic for a given ``--random-state`` (default ``42``).

The ``sim/`` and ``results/`` directories contain patient data. Store and
delete them under the same rules as the cohort. See :doc:`security`.
