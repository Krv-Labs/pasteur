# Contributing

Thanks for helping improve Pasteur.

## Before you open a PR

```bash
cargo fmt --all
cargo clippy --workspace --locked -- -D warnings
cargo test --workspace --locked --exclude pypasteur-bindings
```

CI runs the same checks.

## Guidelines

- One logical change per PR. Link the issue it fixes, if any.
- Behavior changes need a test that fails without the change.
- Add a line under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md) for anything a user would notice. Put breaking changes under `### Breaking`.
- Never commit patient data, PHI, or real clinical datasets, including in test fixtures. Use synthetic frames.

## Security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the [BSD 3-Clause License](LICENSE).
