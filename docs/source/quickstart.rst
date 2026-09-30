.. _quickstart:

===========
Quick Start
===========

Pasteur reads local parquet files and writes a local simulation bundle. This
walkthrough exercises all four simulation outputs without uploading data.

Prerequisites
-------------

- Rust 1.95 or newer
- A parquet file with an ID column and numeric features
- Optionally, a cohort-membership parquet for flipper simulation

Install the CLI:

.. code-block:: bash

   cargo install pasteur-cli
   pasteur-cli --help

Run a simulation
----------------

Choose the feature whose missingness and measurement noise you want to test:

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --id-col patient_id \
     --feature glucose \
     --blackout-companion glucose_measured \
     --blackout-rate 0.2 \
     --jitter-iters 3 \
     --output ./output

The companion option keeps the value and its observation flag consistent:
when ``glucose`` is blacked out, ``glucose_measured`` is nulled on the same
rows.

Add flipper pairs
-----------------

Flipper requires a label-membership table. Select the positive cohort by its
``group_id``:

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --labels groups.parquet \
     --positive-group-id 1 \
     --id-col patient_id \
     --feature glucose \
     --flipper-pairs 20 \
     --flipper-steps 10 \
     --output ./output

See :doc:`data-contracts` for the exact input schemas.

Inspect the bundle
------------------

The output is organized by simulation type:

.. code-block:: text

   output/
   ├── clean/clean.parquet
   ├── blackout/blackout.parquet
   ├── jitter/jitter_0.parquet
   ├── jitter/jitter_1.parquet
   ├── jitter/jitter_2.parquet
   └── flipper/flipper.parquet

Score models
------------

Score one or more ONNX models against the bundle. Create the results
directory first:

.. code-block:: bash

   mkdir -p results
   pasteur-cli compare blackout \
     --sim-root ./output \
     --labels groups.parquet \
     --positive-group-id 1 \
     --model models/model_a.onnx \
     --model models/model_b.onnx \
     --output results/blackout.json

:doc:`model-selection` walks through a full comparison, including exporting
models to ONNX, and :doc:`metrics` explains every number in the output.

Generate a dataset card
-----------------------

Only for bundles made from public or synthetic data that you intend to
publish. Never publish bundles made from patient data.

.. code-block:: bash

   pasteur-cli card \
     --input ./output \
     --source-dataset local/patients \
     --dest-repo org/pasteur-simulations \
     --description "Local clinical-model stress-test bundle"

This writes ``output/README.md``. It does not create or upload a remote
repository.

Next steps
----------

- Learn what each perturbation means in :doc:`simulations`.
- Score an ONNX model with :doc:`cli`.
- Confirm the complete file contract in :doc:`outputs`.
