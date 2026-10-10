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

Build from source
-----------------

.. code-block:: bash

   git clone https://github.com/Krv-Labs/pasteur.git
   cd pasteur
   cargo build -p pasteur-cli
   export ORT_DYLIB_PATH="$(scripts/fetch-onnxruntime.sh)"
   cargo test --workspace --locked --exclude pypasteur-bindings

The executable is written to ``target/debug/pasteur-cli``.

ONNX Runtime
------------

``evaluate`` and ``compare`` score models with ONNX Runtime 1.17 or newer,
which ``pasteur-cli`` opens as a shared library at run time rather than
linking at build time. That is what lets it build and run on Ubuntu 22.04,
Debian 12, and RHEL 8 and 9: Microsoft's release builds need only glibc 2.27.
``simulate`` and ``card`` do not need it.

``pasteur-cli`` looks for the library in this order:

1. ``ORT_DYLIB_PATH``, if set (and nowhere else if it does not load).
2. ``lib/`` beside the directory holding the executable.
3. The system library path (``LD_LIBRARY_PATH``, ``ldconfig``).

To install one, download ``onnxruntime-<os>-<arch>-<version>.tgz`` from
`ONNX Runtime releases <https://github.com/microsoft/onnxruntime/releases>`_
and point at its library:

.. code-block:: bash

   tar -xzf onnxruntime-linux-x64-1.28.0.tgz
   export ORT_DYLIB_PATH="$PWD/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so"
   pasteur-cli doctor

``pasteur-cli doctor`` prints the library and version it loaded, or every
place it looked and how to fix it. From a source checkout,
``scripts/fetch-onnxruntime.sh`` downloads the pinned release, verifies its
SHA-256, and prints the path.

On hosts with glibc 2.38 or newer (Ubuntu 24.04, Fedora 39) or on macOS, you
can instead link ONNX Runtime in at build time, which downloads it from
``cdn.pyke.io``:

.. code-block:: bash

   cargo install pasteur-cli --no-default-features --features static-ort

Once built, ``pasteur-cli`` performs simulation and evaluation without opening
network connections. For machines without internet access, see
:ref:`security:Offline installation`.

Publishing is separate
----------------------

Pasteur never treats a remote dataset name as a local path. Download artifacts
with the official Hugging Face CLI first, then pass their local paths to
Pasteur. Uploads are likewise explicit and external to ``pasteur-cli``.
