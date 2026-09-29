=============
CLI Reference
=============

``pasteur-cli`` performs local simulation, evaluation, comparison, dataset-card
generation, and staging-cache management.

simulate
--------

.. code-block:: bash

   pasteur-cli simulate [OPTIONS] --input <INPUT>

Important options:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Default
     - Purpose
   * - ``--input``
     - required
     - Local clean parquet
   * - ``--labels``
     - none
     - Cohort-membership parquet; enables flipper
   * - ``--output``
     - ``output``
     - Simulation bundle root
   * - ``--id-col``
     - ``node_id``
     - Source row identifier
   * - ``--feature``
     - ``TSH``
     - Feature targeted by blackout and jitter
   * - ``--blackout-companion``
     - none
     - Repeatable column nulled with the target feature
   * - ``--blackout-rate``
     - ``0.1``
     - Fraction of rows selected for blackout
   * - ``--jitter-scale``
     - ``1.0``
     - Noise scale
   * - ``--jitter-iters``
     - ``3``
     - Number of jitter variants
   * - ``--positive-group-id``
     - ``103``
     - Group treated as the positive cohort
   * - ``--random-state``
     - ``42``
     - Random seed

card
----

Generate a Hugging Face-compatible ``README.md`` for a local bundle:

.. code-block:: bash

   pasteur-cli card \
     --input ./output \
     --source-dataset org/source \
     --dest-repo org/source-simulations \
     --source-info source-info.json

The command writes documentation only. Uploads remain explicit ``hf upload``
operations outside Pasteur.

evaluate
--------

Score one local ONNX model:

.. code-block:: bash

   pasteur-cli evaluate blackout \
     --sim-root ./sim \
     --labels ./labels/groups.parquet \
     --model ./models/model.onnx \
     --output ./scores.json

The positional simulation type can be ``blackout``, ``jitter``, or ``flipper``.
Use ``--contract`` when the model's ``metadata.json`` is not adjacent to the
ONNX file.

compare
-------

Score multiple models over the same variants:

.. code-block:: bash

   pasteur-cli compare blackout \
     --sim-root ./sim \
     --labels ./labels/groups.parquet \
     --model ./models/a.onnx \
     --model ./models/b.onnx \
     --predictions-out ./predictions/blackout.parquet \
     --output ./comparison.json

``--model`` is repeatable. ``--predictions-out`` writes one probability column
per model plus ``all_agree``.

cache
-----

The Pasteur staging cache is separate from the Hugging Face download cache:

.. code-block:: bash

   pasteur-cli cache list
   pasteur-cli cache rm stub.onnx --kind models
   pasteur-cli cache clear --kind simulations

Use ``pasteur-cli cache`` for Pasteur's ``models/`` and ``simulations/``
staging directories. Use ``hf cache`` for files downloaded from the Hub.

Model contracts
---------------

An optional ``metadata.json`` preserves feature order and the model's missing
value sentinel:

.. code-block:: json

   {
     "input": {
       "feature_order": ["TSH", "T3", "T4"],
       "absent_sentinel": 0.0
     }
   }

A declared feature-order mismatch aborts evaluation. If no contract exists,
Pasteur still checks input width, but it cannot infer feature semantics from
the ONNX graph.

Run ``pasteur-cli <command> --help`` for the complete generated option list.
