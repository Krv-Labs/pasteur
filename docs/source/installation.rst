============
Installation
============

Pasteur ships as a Rust CLI and Python bindings. Choose the interface that
matches your workflow; both execute locally.

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Use case
     - Install
     - What you get
   * - Simulation and evaluation
     - ``cargo install pasteur-cli``
     - The complete command-line workflow
   * - Python simulation
     - ``pip install pypasteur``
     - Blackout and jitter simulators for Polars dataframes
   * - Development
     - Build this workspace
     - CLI, libraries, bindings, and tests

Install the CLI
---------------

.. code-block:: bash

   cargo install pasteur-cli
   pasteur-cli --version
   pasteur-cli --help

The workspace minimum supported Rust version is 1.95.

Install the Python bindings
---------------------------

.. code-block:: bash

   python -m pip install pypasteur

Then import the package:

.. code-block:: python

   import polars as pl
   import pypasteur

   frame = pl.read_parquet("patients.parquet")
   simulator = pypasteur.BlackoutSimulator(
       "glucose",
       rate=0.1,
       random_state=42,
   )
   simulator.fit(frame)
   shifted = simulator.transform(frame)

Build from source
-----------------

.. code-block:: bash

   git clone https://github.com/Krv-Labs/pasteur.git
   cd pasteur-core
   cargo build -p pasteur-cli
   cargo test --workspace --locked --exclude pypasteur-bindings

The executable is written to ``target/debug/pasteur-cli``.

ONNX Runtime
------------

The ``ort`` crate downloads ONNX Runtime at build time. Once built,
``pasteur-cli`` performs simulation and evaluation without opening network
connections.

Publishing is separate
----------------------

Pasteur never treats a remote dataset name as a local path. Download artifacts
with the official Hugging Face CLI first, then pass their local paths to
Pasteur. Uploads are likewise explicit and external to ``pasteur-cli``.
