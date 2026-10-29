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

Sweep
-----

Sweep is Flipper's single-feature sibling. For every source row and every
``--sweep`` feature, it sets **only that feature** to each value of a grid and
copies every other column unchanged. Flipper asks where the boundary sits
between two patients; sweep asks whether the model reads one feature, and
where along it the decision flips.

.. code-block:: bash

   pasteur-cli simulate \
     --input cases.parquet \
     --id-col case_id \
     --sweep glucose=7.0 --sweep age \
     --sweep-range 0.5:2.0 --sweep-relative-to 50 \
     --sweep-steps 25 --sweep-near 0.05

- ``--sweep-range lo:hi`` is absolute by default. With
  ``--sweep-relative-to`` it is a multiplier of each feature's cutoff: the
  feature's own ``=cutoff`` if given, else the ``--sweep-relative-to`` value.
  Above, glucose runs from 3.5 to 14 and age from 25 to 100.
- ``--sweep-steps`` points span the range, both ends included.
  ``--sweep-near f`` adds as many points again within ±f of the cutoff, for a
  denser grid where the decision should flip.
- No labels are needed. The grid is written to ``sweep/sweep.parquet``.

Sweep rows carry prefixed meta columns, so they cannot collide with your data:
``sweep_case_row`` (0-based source row), ``sweep_feature``, ``sweep_step``
(ascending by value), ``sweep_value`` and ``sweep_cutoff`` (null without a
cutoff). Their ``source_row_id`` is ``sweep_<case id>_<feature>_step_<n>``.
Every swept feature is written as a float.

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
- Sweep moves one feature while holding the others fixed, so some grid rows
  are combinations no real patient has.

These simulations are evidence about a model under explicit conditions. They
are not a substitute for local validation or clinical safety review.
