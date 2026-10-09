# Agent Guide

## Safety

- Never commit patient data, PHI, real clinical datasets, or proprietary models.
- `pasteur-cli` is local compute only; keep it network-free. Hub I/O goes through the official `hf` CLI; download inputs first, then pass local paths.
- Create Hugging Face model and dataset repositories as **private** by default.
- Only publish simulations of public or synthetic data. Bundles keep each patient's row ID, so never upload one built from patient data, even to a private repo.

## Development

Git worktrees must not share a `CARGO_TARGET_DIR`: Cargo hashes workspace crates the same way in each checkout, so a shared target dir silently mixes artifacts from different checkouts.

```bash
cargo fmt --all --check
cargo clippy --workspace --locked -- -D warnings
cargo test --workspace --locked --exclude pypasteur-bindings
```
