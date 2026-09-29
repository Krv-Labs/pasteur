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

.. warning::
   IDs that do not parse as integers are labeled negative. Validate identifier
   types before evaluating a model.

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
