# Pasteur

Stress-test clinical AI models before you trust them. Pasteur simulates the ways real clinical data goes wrong, then measures how much a model's predictions move:

- **Blackout**: a lab or feature goes missing, together with its companion columns
- **Jitter**: measurement noise
- **Flipper**: interpolation between patients with different labels, to find where a decision flips

This repo contains the Rust engine, the `pasteur-cli` command-line tool, and the `pypasteur` Python bindings. Everything runs locally; no data leaves your machine.

Licensed under [BSD-3-Clause](LICENSE).

## Install

```bash
cargo install pasteur-cli   # CLI
pip install pypasteur       # Python bindings
```

## Quickstart

Start from any parquet with an id column and numeric features ([input schemas](AGENTS.md#inputs)):

```bash
pasteur-cli simulate \
  --input patients.parquet \
  --id-col patient_id \
  --feature glucose \
  --output ./output
```

This writes `output/clean/`, `output/blackout/`, and `output/jitter/`. Pass `--labels` to also produce flipper pairs. Then score an ONNX model against the bundle with `pasteur-cli evaluate`.

```python
import polars as pl
import pypasteur

df = pl.read_parquet("patients.parquet")
sim = pypasteur.BlackoutSimulator("glucose", rate=0.1, random_state=42)
sim.fit(df)
shifted = sim.transform(df)
```

## Workspace

| Crate | Role |
|-------|------|
| [`core`](core/) (`pasteur-core`) | Simulators, evaluators, schemas |
| [`cli`](cli/) (`pasteur-cli`) | Local CLI: `simulate`, `evaluate`, `compare`, `card`, `cache` |
| [`hf`](hf/) (`pasteur-hf`) | Simulation bundle layout + Hugging Face dataset cards |
| [`model`](model/) (`pasteur-model`) | ONNX model loading and scoring |
| [`bindings/python-bindings`](bindings/python-bindings/) (`pypasteur`) | PyO3 Python bindings |

`pasteur-model` downloads ONNX Runtime at **build** time (the `ort` crate's `download-binaries`). At runtime, `pasteur-cli` never opens a network connection.

## Docs

- Full documentation: [docs.krv.ai/pasteur](https://docs.krv.ai/pasteur/)
- CLI reference: [docs.krv.ai/pasteur/cli](https://docs.krv.ai/pasteur/cli.html)
- Agent workflows with the Hugging Face `hf` CLI: [AGENTS.md](AGENTS.md)
- Simulation bundle and dataset card contracts: [hf/README.md](hf/README.md)
- Agent skills: [hf-cli](.agents/skills/hf-cli/SKILL.md), [hf-datasets](.agents/skills/hf-datasets/SKILL.md), [pasteur-rs](.agents/skills/pasteur-rs/SKILL.md)

## Development

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace --exclude pypasteur-bindings
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Pushing a `v*` tag runs `.github/workflows/release.yml`.
