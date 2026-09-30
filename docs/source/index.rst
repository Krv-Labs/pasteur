.. _index:

.. rst-class:: pasteur-sphinx-title

=======
Pasteur
=======

.. raw:: html

   <div class="pasteur-hero">
     <p class="pasteur-eyebrow">Clinical AI stress testing</p>
     <p class="pasteur-wordmark">Pasteur</p>
     <p class="pasteur-tagline">Find where clinical models become brittle before deployment.</p>
     <p class="pasteur-lead"><strong>Test model behavior under realistic data failures.</strong>
       Pasteur simulates missing measurements, measurement noise, and transitions
       between patient cohorts, then quantifies how predictions respond. All
       computation stays on your machine.</p>
     <div class="pasteur-cta-row">
       <a class="pasteur-btn" href="quickstart.html">Get started →</a>
       <a class="pasteur-btn ghost" href="https://github.com/Krv-Labs/pasteur">View on GitHub</a>
     </div>
   </div>

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Blackout
      :link: simulations
      :link-type: doc

      Remove a feature and its measurement companions to test missingness.

   .. grid-item-card:: Jitter
      :link: simulations
      :link-type: doc

      Add controlled measurement noise and measure prediction stability.

   .. grid-item-card:: Flipper
      :link: simulations
      :link-type: doc

      Interpolate between differently labeled patients and locate decision flips.

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - **100% local**
     - Runs on your machine's CPU. No server, daemon, or admin rights.
   * - **No network, no telemetry**
     - No network calls at runtime. Data, models, and results never leave the
       machine.
   * - **Standard tooling**
     - ``pip install pypasteur`` or ``cargo install pasteur-cli``.
       BSD-3-Clause open source.

.. note::
   **Reviewing Pasteur for a hospital or health system?** Start with
   :doc:`security`, or download the
   :download:`one-page overview (PDF) <_static/pasteur-usage-overview.pdf>`.

What Pasteur produces
---------------------

One ``simulate`` run creates a reproducible bundle containing the clean cohort
and each stress-test variant. ``evaluate`` scores one ONNX model;
``compare`` places several models on the same rows and can write a parquet of
per-row predictions.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Command
     - Input
     - Result
   * - ``simulate``
     - Local parquet data and optional cohort labels
     - Clean, blackout, jitter, and flipper parquets
   * - ``evaluate``
     - Simulation bundle, labels, and one ONNX model
     - Baseline and stability metrics (:doc:`metrics`)
   * - ``compare``
     - Simulation bundle and multiple ONNX models
     - Model comparison and optional row-level predictions
   * - ``card``
     - Simulation bundle and provenance
     - A Hugging Face-compatible dataset card

Quick look
----------

.. code-block:: bash

   pasteur-cli simulate \
     --input patients.parquet \
     --id-col patient_id \
     --feature glucose \
     --output ./output

This writes ``clean/``, ``blackout/``, and ``jitter/`` beneath ``./output``.
Add ``--labels groups.parquet`` to generate flipper pairs. For a complete
run from CSV to a model comparison, see :doc:`model-selection`.

Designed for evidence, not a pass/fail badge
--------------------------------------------

Pasteur does not claim that a model is safe. It produces concrete evidence
about model behavior under declared perturbations: what changed, where it
changed, and how strongly. Those results belong alongside intended-use
documentation, local validation, clinical review, and deployment monitoring.
Pasteur is a research and evaluation tool, not a medical device.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   Quick Start <quickstart>
   installation

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden:

   simulations
   Choosing Between Models <model-selection>
   Reading the Results <metrics>
   CLI Reference <cli>

.. toctree::
   :maxdepth: 1
   :caption: For Hospital IT
   :hidden:

   Security & Data Handling <security>

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   Data Contracts <data-contracts>
   Output Layout <outputs>
