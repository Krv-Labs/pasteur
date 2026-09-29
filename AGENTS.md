# Agent Guide

Pasteur is a Rust workspace for local clinical-AI simulation and evaluation.

## Safety

- Never commit patient data, PHI, real clinical datasets, or proprietary models.
- `pasteur-cli` performs local compute only; keep it network-free.
- Create Hugging Face model and dataset repositories as **private** by default.

## Tool boundary

- Use `pasteur-cli` for simulation, evaluation, comparison, dataset cards, and its local cache.
- Use the official `hf` CLI for Hub downloads, uploads, repositories, and the Hub cache.
- Download remote inputs before passing local paths to `pasteur-cli`.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
cargo test --workspace --exclude pypasteur-bindings
```

See [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), and the [CLI reference](cli/README.md).
