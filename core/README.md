# Pasteur Core

Pure Rust proprietary core. No Python or Node dependencies.

## Tests

- **Unit tests** live in `#[cfg(test)] mod tests` at the bottom of the source file they exercise (`src/**/*.rs`), same pattern as `blackout.rs` and `jitter.rs`.
- **Integration tests** live in `tests/` at the crate root (`core/tests/`, `cli/tests/`) and cover cross-module or CLI workflows.

Run: `cargo test -p pasteur-core` or `cargo test -p pasteur-cli`.
