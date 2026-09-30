=============================
Security and data handling
=============================

This page is for hospital IT, security, and compliance reviewers. It describes
what Pasteur reads, writes, and connects to, so you can assess it without
reading the source.

A one-page summary is available as a PDF:
:download:`Pasteur usage overview <_static/pasteur-usage-overview.pdf>`.

At a glance
-----------

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - **Where it runs**
     - On the machine where you run it. Local CPU only. No server, daemon,
       background service, or root/administrator rights.
   * - **Network at runtime**
     - None. ``pasteur-cli`` and ``pypasteur`` make no network calls and send
       no telemetry. You can run them with all outbound traffic blocked.
   * - **Network at install**
     - Needed, unless you use an internal mirror. See `Offline installation`_.
   * - **What it reads**
     - Only the files you pass on the command line or to the Python API.
   * - **What it writes**
     - Only the output paths you pass. Nothing else is written.
   * - **Logging**
     - Console output lists file paths and row counts. It never prints patient
       values.
   * - **Licence**
     - BSD-3-Clause, open source.

Data flow
---------

.. code-block:: text

   ┌──────────── your machine / your network ─────────────┐
   │                                                      │
   │  cohort.parquet ─┐                                   │
   │  groups.parquet ─┼─► pasteur-cli ─► sim/  (parquet)  │
   │  model.onnx ─────┘                   results/ (JSON) │
   │                                                      │
   └──────────────────── nothing leaves ──────────────────┘

1. **Inputs**: a local parquet of patients and features, an optional labels
   parquet, and one or more local ONNX models.
2. **Compute**: simulations and model scoring run in memory on the local CPU.
3. **Outputs**: parquet and JSON files written to the directories you choose.

Outputs contain patient data
----------------------------

Treat everything Pasteur writes with the same controls as its input.

- ``source_row_id`` in every output parquet is the **original patient
  identifier** from your ``--id-col``.
- Blackout and jitter outputs carry every feature column from the input,
  with only the targeted feature changed.
- Flipper rows are interpolations between two real patients. They are
  synthetic, but they are derived from, and can point back to, real records
  (``source_a_row``, ``source_b_row``).
- ``compare --predictions-out`` writes a model score for every patient.

Pasteur does not de-identify data. If the input is PHI, the outputs are PHI.

.. warning::
   The ``card`` command and the ``hf`` agent workflows in this repository
   exist for publishing simulations of **public or synthetic** datasets to
   Hugging Face. Never upload bundles made from real patient data.

Network behaviour in detail
---------------------------

**At runtime** Pasteur opens local files with the operating system's file APIs.
It does not resolve URLs, and it has no update check, licence check, or
analytics.

Security scanners may still find HTTP and TLS libraries (``reqwest``,
``hyper``, ``rustls``) inside the ``pasteur-cli`` binary. They come from the
`polars <https://pola.rs>`_ dataframe library, which bundles cloud-storage
support. Pasteur never calls that code. As a defence in depth, run Pasteur
on a host or container with outbound traffic denied.

**At install**:

- ``pip install pypasteur`` downloads a pre-built wheel and ``polars`` from
  PyPI.
- ``cargo install pasteur-cli`` downloads Rust crates from crates.io, and its
  build downloads **ONNX Runtime 1.28.0** from ``cdn.pyke.io``. That archive
  is checked against a SHA-256 digest pinned in the ``ort-sys`` crate.

Offline installation
--------------------

For machines with no internet access, prepare the packages on a connected
machine, then transfer them.

**Python**, on a connected machine of the same OS and architecture:

.. code-block:: bash

   python -m pip download pypasteur -d ./wheels

Then, on the offline machine:

.. code-block:: bash

   python -m pip install --no-index --find-links ./wheels pypasteur

**CLI**: build on a connected machine and copy the single
``pasteur-cli`` executable, or, to build offline, vendor the crates with
``cargo vendor`` and point the build at a local ONNX Runtime with
``ORT_LIB_LOCATION=/path/to/onnxruntime``.

System requirements
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - ``pypasteur``
     - CPython 3.12 or newer. Wheels for Linux x86_64/aarch64, macOS
       x86_64/arm64, Windows x64.
   * - ``pasteur-cli``
     - Built from source with Rust 1.95 or newer on Linux, macOS, or
       Windows.
   * - Memory
     - The cohort and every simulation variant are held in memory. Budget a
       few times the size of your input parquet.

Models are trusted code
-----------------------

``evaluate`` and ``compare`` execute ONNX graphs with ONNX Runtime. Only load
models from sources you trust, the same as any other executable artifact.
Pasteur checks the model's input width, and, when a ``metadata.json`` is
present, its feature order. See :doc:`data-contracts`.

Local cache
-----------

``pasteur-cli cache`` manages an optional staging directory for models and
simulation bundles. Pasteur only touches it when you run a ``cache``
command. Its location is, in order: ``--cache-dir``, ``$PASTEUR_RS_CACHE_DIR``,
the OS cache directory (``~/Library/Caches/pasteur-rs`` on macOS,
``~/.cache/pasteur-rs`` on Linux), or ``./.pasteur-rs-cache``. If you stage
patient data there, remove it with ``pasteur-cli cache clear`` when you are
done.

Intended use
------------

Pasteur is a research and evaluation tool. It produces evidence about how a
model's outputs change under declared perturbations. It is not a medical
device, it does not make or support clinical decisions, and a good Pasteur
result does not show that a model is safe. Use it alongside intended-use
documentation, local validation, clinical review, and monitoring.

Third-party components
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Component
     - Licence
     - Role
   * - `ONNX Runtime <https://onnxruntime.ai>`_
     - MIT
     - Runs ONNX models (bundled into ``pasteur-cli``)
   * - `polars <https://pola.rs>`_
     - MIT
     - Dataframes and parquet I/O
   * - `ort <https://ort.pyke.io>`_
     - MIT / Apache-2.0
     - Rust bindings for ONNX Runtime

The full dependency list is in ``Cargo.lock``; ``cargo tree`` prints it.

Reporting a vulnerability
-------------------------

See `SECURITY.md <https://github.com/Krv-Labs/pasteur/blob/main/SECURITY.md>`_.
