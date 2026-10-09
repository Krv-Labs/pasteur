=============
Output Layout
=============

Simulation bundle
-----------------

.. code-block:: text

   output/
   ├── README.md
   ├── clean/
   │   └── clean.parquet
   ├── blackout/
   │   └── blackout.parquet
   ├── jitter/
   │   ├── jitter_0.parquet
   │   └── jitter_1.parquet
   └── flipper/
       └── flipper.parquet

Every simulation parquet includes:

.. list-table::
   :header-rows: 1
   :widths: 30 24 46

   * - Column
     - Type
     - Meaning
   * - ``source_row_id``
     - string
     - Original row ID, or a synthetic flipper path ID
   * - ``row_ordinal``
     - ``u32``
     - Clean-run row position
   * - Feature columns
     - numeric
     - Model inputs carried from or derived from the source cohort

Flipper metadata
----------------

Flipper output also contains ``pair_id``, ``step``, ``t``, ``source_a_row``,
``source_b_row``, ``label_a``, and ``label_b``. Grids simulated with
``--task multilabel`` also contain ``pair_label``, the index of the label the
pair was sampled for. For multiclass grids ``label_a`` and ``label_b`` are
class indices in ``--positive-group-id`` order. Evaluation strips these
metadata columns before model scoring.

Evaluation JSON
---------------

``evaluate`` writes an ``EvaluationResult``:

.. code-block:: json

   {
     "baselines": {"roc_auc": 0.85},
     "evaluations": {
       "metric_invariant": {
         "jitter_stability": 0.92,
         "flipper_stability": null
       },
       "metric_based": {
         "roc_auc": {
           "resiliency": 0.88,
           "generalizability": 1.0
         }
       }
     },
     "created_at": "2026-07-10T12:00:00Z"
   }

``flipper_stability`` is populated for flipper evaluation and null for blackout
or jitter runs.

Multiclass and multilabel results add a ``multi`` object. The headline fields
above hold macro averages; :doc:`metrics` defines each field.

.. code-block:: json

   "multi": {
     "task": "multilabel",
     "roc_auc_micro": 0.93,
     "worst_label_resiliency": 0.81,
     "decision_flip_rate": 0.06,
     "flipper_never_flipped": 0.37,
     "excluded_labels": [],
     "per_label": [
       {"label": "diabetes", "roc_auc": 0.90, "resiliency": 0.81,
        "jitter_stability": 0.95, "flipper_stability": 0.63}
     ]
   }

Multiclass results also carry ``flipper_detour_rate``.

Comparison predictions
----------------------

When ``compare`` receives ``--predictions-out``, the parquet contains:

- ``source_row_id``
- ``variant_name``
- ``label`` (multi-output: ``label__<class>`` per class)
- one floating-point probability column per model (multi-output:
  ``<model>__<class>`` per class)
- ``all_agree``

The model column name is derived from the model file stem.
