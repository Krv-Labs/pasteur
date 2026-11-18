============
Installation
============

Pasteur ships as a CLI (on npm and crates.io), Python bindings, and a Claude
Code plugin. Choose the interface that matches your workflow; all of them
execute locally.

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Use case
     - Install
     - What you get
   * - Simulation and evaluation
     - ``npx -y @krv-labs/pasteur --help``
     - The complete command-line workflow, prebuilt, no Rust toolchain
   * - Simulation and evaluation, compiled locally
     - ``cargo install pasteur-cli``
     - The same CLI
   * - Python simulation
     - ``pip install pypasteur``
     - Blackout and jitter simulators for Polars dataframes
   * - Agent-driven runs
     - Claude Code plugin
     - A Pasteur skill and the ``/pasteur:robustness`` command
   * - Development
     - Build this workspace
     - CLI, libraries, bindings, and tests

Install the CLI from npm
------------------------

The ``@krv-labs/pasteur`` npm package carries a prebuilt ``pasteur-cli`` for
macOS (arm64, x64), Linux (glibc; arm64, x64) and Windows (x64). It needs
Node.js 18 or newer and installs the same program under two names,
``pasteur-cli`` and ``pasteur``.

.. code-block:: bash

   npx -y @krv-labs/pasteur --help        # run without installing
   npm install -g @krv-labs/pasteur       # or put pasteur-cli on PATH
   pasteur-cli --version

npm downloads the binary at install time; the CLI itself still makes no
network calls. The same binaries are attached to each
`GitHub release <https://github.com/Krv-Labs/pasteur/releases>`_.

Install the CLI with Cargo
--------------------------

.. code-block:: bash

   cargo install pasteur-cli
   pasteur-cli --version
   pasteur-cli --help

The workspace minimum supported Rust version is 1.95.

Install the Python bindings
---------------------------

.. code-block:: bash

   python -m pip install pypasteur

The package is ``pypasteur``, which requires CPython 3.12 or newer. The
``pasteur`` package on PyPI is an unrelated project; do not install it.

The simulators take `polars <https://pola.rs>`_ DataFrames. To start from
pandas, convert with ``pl.from_pandas(df)``. Always call ``fit`` before
``transform``:

.. code-block:: python

   import polars as pl
   import pypasteur

   frame = pl.read_parquet("patients.parquet")
   simulator = pypasteur.BlackoutSimulator(
       "glucose",
       rate=0.1,
       companions=["glucose_measured"],
       random_state=42,
   )
   simulator.fit(frame)
   shifted = simulator.transform(frame)

Install the Claude Code plugin
------------------------------

The plugin teaches Claude Code to run Pasteur through ``npx`` and adds a
``/pasteur:robustness`` command that runs scramble and unit-conversion checks
on one feature of an ONNX model. In a Claude Code session:

.. code-block:: text

   /plugin marketplace add Krv-Labs/pasteur
   /plugin install pasteur@pasteur

The plugin keeps Pasteur's rules: the CLI stays local, Hugging Face Hub
transfers go through the ``hf`` CLI, repositories are created private, and
bundles built from patient data are never uploaded.

Build from source
-----------------

.. code-block:: bash

   git clone https://github.com/Krv-Labs/pasteur.git
   cd pasteur
   cargo build -p pasteur-cli
   cargo test --workspace --locked --exclude pypasteur-bindings

The executable is written to ``target/debug/pasteur-cli``.

ONNX Runtime
------------

The ``ort`` crate downloads ONNX Runtime at build time. Once built,
``pasteur-cli`` performs simulation and evaluation without opening network
connections. For machines without internet access, see
:ref:`security:Offline installation`.

Publishing is separate
----------------------

Pasteur never treats a remote dataset name as a local path. Download artifacts
with the official Hugging Face CLI first, then pass their local paths to
Pasteur. Uploads are likewise explicit and external to ``pasteur-cli``.
