# Pasteur

Stress-test clinical AI models before you trust them. Pasteur simulates the ways real clinical data goes wrong, then measures how much a model's predictions move:

- **Blackout**: a lab or feature goes missing, together with its companion columns
- **Jitter**: measurement noise
- **Flipper**: interpolation between patients with different labels, to find where a decision flips

| | |
|---|---|
| **100% local** | Runs on your machine's CPU. No server, daemon, or admin rights. |
| **No network, no telemetry** | Makes no network calls at runtime. Data, models, and results never leave the machine. |
| **Standard tooling** | `pip install pypasteur` or `cargo install pasteur-cli`. BSD-3-Clause open source. |

```text
local parquet + ONNX models  →  Pasteur (CLI or Python)  →  local parquet + JSON metrics
```

Pasteur produces evidence about model behaviour under declared perturbations, not a pass/fail badge. It is a research and evaluation tool, not a medical device.

**Reviewing Pasteur for your organisation?** Start with [Security and data handling](https://docs.krv.ai/pasteur/security.html) or the [one-page overview (PDF)](docs/source/_static/pasteur-usage-overview.pdf).

## Install

```bash
pip install pypasteur       # Python: blackout and jitter simulators (CPython ≥ 3.12)
cargo install pasteur-cli   # CLI: simulate, evaluate, compare (Rust ≥ 1.95)
```

The PyPI package is **`pypasteur`**. The unrelated `pasteur` package on PyPI is a different project.

## Quickstart

Start from a parquet with an integer id column and numeric model features ([input schemas](docs/source/data-contracts.rst)):

```bash
pasteur-cli simulate \
  --input cohort.parquet \
  --labels groups.parquet \
  --positive-group-id 1 \
  --id-col patient_id \
  --feature crp \
  --blackout-companion crp_measured \
  --output ./sim

pasteur-cli compare blackout \
  --sim-root ./sim \
  --labels groups.parquet \
  --positive-group-id 1 \
  --model models/logreg.onnx \
  --model models/gbm.onnx \
  --output results-blackout.json
```

`simulate` writes `sim/clean/`, `sim/blackout/`, `sim/jitter/`, and (with `--labels`) `sim/flipper/`. `compare` scores each ONNX model on the clean and perturbed rows and writes the metrics as JSON. See [Reading the results](https://docs.krv.ai/pasteur/metrics.html) and the [model-selection walkthrough](https://docs.krv.ai/pasteur/model-selection.html).

From Python:

```python
import polars as pl
import pypasteur

df = pl.read_parquet("cohort.parquet")
blackout = pypasteur.BlackoutSimulator(
    "crp", rate=0.2, companions=["crp_measured"], random_state=42
)
blackout.fit(df)
missing = blackout.transform(df)
```

## Docs

- Full documentation: [docs.krv.ai/pasteur](https://docs.krv.ai/pasteur/)
- [Security and data handling](https://docs.krv.ai/pasteur/security.html): data flow, network behaviour, offline install, PHI in outputs
- [Walkthrough: choosing between models](https://docs.krv.ai/pasteur/model-selection.html)
- [Reading the results](https://docs.krv.ai/pasteur/metrics.html): metric definitions and caveats
- [CLI reference](cli/README.md)

## Workspace

| Crate | Role |
|-------|------|
| [`core`](core/) (`pasteur-core`) | Simulators, evaluators, schemas |
| [`cli`](cli/) (`pasteur-cli`) | Local CLI: `simulate`, `evaluate`, `compare`, `card`, `cache` |
| [`hf`](hf/) (`pasteur-hf`) | Simulation bundle layout + Hugging Face dataset cards |
| [`model`](model/) (`pasteur-model`) | ONNX model loading and scoring |
| [`bindings/python-bindings`](bindings/python-bindings/) (`pypasteur`) | PyO3 Python bindings |

`pasteur-model` downloads ONNX Runtime at **build** time (the `ort` crate's `download-binaries`). At runtime, `pasteur-cli` never opens a network connection.

## Development

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace --exclude pypasteur-bindings
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Pushing a `v*` tag runs `.github/workflows/release.yml`.

Automation: publishing simulations of public or synthetic datasets to Hugging Face is described in [AGENTS.md](AGENTS.md), with agent skills under [`.agents/skills/`](.agents/skills/). Never publish bundles made from patient data.
