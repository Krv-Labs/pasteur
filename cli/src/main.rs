//! Binary entrypoint for the `pasteur-cli` executable.
//!
//! Delegates to [`pasteur_cli::run_or_exit`] so error handling lives in the library crate.

fn main() {
    pasteur_cli::run_or_exit();
}
