==============
Data Contracts
==============

Pasteur accepts local parquet files. It does not bundle a clinical dataset or
infer labels from diagnosis fields.

Clean cohort
------------

One row represents one source sample.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Column
     - Type
     - Meaning
   * - ID column
     - any
     - Selected by ``--id-col``; emitted as string ``source_row_id``
   * - Target feature
     - numeric
     - Selected by ``--feature`` for blackout and jitter
   * - Companion columns
     - any
     - Repeatable ``--blackout-companion`` values nulled with the feature
   * - Other features
     - numeric
     - Carried through simulation and supplied to models

The original ID is converted to a string, not integer-coerced. A separate
``row_ordinal`` records the clean-run row index.

Every column other than the ID is carried into the outputs and sent to the
model, in file order. Drop outcome, label, and administrative columns before
running Pasteur, and keep the features in the order the model was trained on.
The input must be parquet; :doc:`model-selection` shows how to convert a CSV.

Labels
------

Labels are represented as cohort membership rather than one label per source
row:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Column
     - Type
     - Meaning
   * - ``group_id``
     - ``u32``
     - Cohort identifier
   * - ``member_ids``
     - list of integers
     - Source IDs belonging to that cohort

A source row is positive when its ID parses as an integer and appears in the
``member_ids`` list for ``--positive-group-id``. All other rows are negative.

Multiclass and multilabel models take one cohort per model output: repeat
``--positive-group-id`` once per class or label, **in the order of the model's
``output.classes``**. Pasteur matches groups to output columns by position;
it cannot check that group 12 really means the model's second class.

- ``--task multiclass``: every row must be in exactly one group. A row in
  none (including rows whose ID is not an integer) or in several stops the
  run with a count and example IDs. If the model has a "none of the above"
  class, give it its own group.
- ``--task multilabel``: groups may overlap, and a row may be in none.

.. warning::
   IDs that do not parse as integers are labeled negative. Validate identifier
   types before evaluating a model.

:doc:`model-selection` includes a short script that builds this file from an
outcome column.

Model contract
--------------

ONNX tensors preserve width and order, not clinical feature names. Place
``metadata.json`` next to the model or pass it with ``--contract``:

.. code-block:: json

   {
     "input": {
       "feature_order": ["glucose", "bmi", "age"],
       "absent_sentinel": null
     }
   }

``feature_order`` is compared element by element with the simulation data.
``absent_sentinel`` supplies the model-specific value used for missing inputs.

Multiclass and multilabel models must also declare what their outputs are:

.. code-block:: json

   {
     "task_type": "multilabel",
     "input": {"feature_order": ["glucose", "bmi", "age"]},
     "output": {
       "name": "probabilities",
       "classes": ["diabetes", "hypertension", "ckd"],
       "thresholds": [0.5, 0.4, 0.3]
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``task_type``
     - ``binary`` (default), ``multiclass``, or ``multilabel``. Must match
       ``--task``.
   * - ``output.name``
     - Graph output holding the probabilities. Defaults to
       ``output_probability``, then ``probabilities``.
   * - ``output.classes``
     - Required for multiclass and multilabel. One name per output column, in
       the model's column order. Checked against the graph's output width when
       loading, and used to match ZipMap outputs by key name.
   * - ``output.thresholds``
     - Multilabel only, optional: one decision threshold per label. Defaults
       to ``--flip-threshold`` for every label.

Pasteur reads every probability format scikit-learn models export to ONNX: a
``[n, K]`` tensor, a ZipMap with integer or string keys, and, for multilabel,
``MultiOutputClassifier``'s one ``[n, 2]`` tensor per label. A model declared
``multiclass`` whose rows do not sum to 1 is rejected; independent per-label
probabilities are ``multilabel``.
