===========
Simulations
===========

Pasteur applies controlled perturbations to a clean tabular cohort. Every run
keeps source-row identity and records deterministic row order so clean and
shifted variants can be compared.

Blackout
--------

Blackout sets the selected feature to null for a seeded sample of rows.
Companion columns can be nulled at the same time:

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --feature glucose \
     --blackout-companion glucose_measured \
     --blackout-rate 0.2

Use companions for observation flags or other fields that would become
internally inconsistent if the value disappeared alone.

Jitter
------

Jitter adds seeded measurement noise to the selected numeric feature and can
generate several variants:

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --feature glucose \
     --jitter-scale 0.5 \
     --jitter-iters 5

The resulting ``jitter_0.parquet`` through ``jitter_4.parquet`` retain the same
source-row IDs as the clean cohort.

Flipper
-------

Flipper samples patients from opposite labels and interpolates between their
feature vectors. The resulting path can expose where a model crosses its
decision threshold.

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --labels groups.parquet \
     --positive-group-id 1 \
     --flipper-pairs 100 \
     --flipper-steps 20

Flipper rows include ``pair_id``, ``step``, ``t``, source-row references, and
endpoint labels. Their ``source_row_id`` is synthetic because each row lies on
an interpolated path rather than representing one source patient.

Reproducibility
---------------

All simulators use ``--random-state`` (default ``42``). Record the command,
source-data revision, and model contract alongside results; a simulation metric
without those inputs is not independently reproducible.

Interpretation
--------------

- Blackout tests behavior under declared missingness, not every real-world
  missing-data mechanism.
- Jitter tests a chosen noise scale, not instrument calibration in general.
- Flipper examines interpolated paths between observed endpoints; clinical
  plausibility still requires domain review.

These simulations are evidence about a model under explicit conditions. They
are not a substitute for local validation or clinical safety review.
