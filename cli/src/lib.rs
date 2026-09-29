//! Pasteur simulation CLI — local compute only (no Hugging Face network I/O).
//!
//! Agents download and upload Hub artifacts with the official `hf` CLI.
//! See `AGENTS.md` for end-to-end workflows.

mod args;
mod commands;
mod data;
mod io;
mod model;

pub use args::Cli;

use anyhow::Result;
use clap::Parser;

use args::Command;
use commands::{run_cache, run_card, run_compare, run_evaluate, run_simulate};

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Simulate(args) => run_simulate(args),
        Command::Card(args) => run_card(args),
        Command::Evaluate(args) => run_evaluate(args),
        Command::Compare(args) => run_compare(args),
        Command::Cache(args) => run_cache(args),
    }
}

/// Run the CLI and exit the process with code 1 on failure.
pub fn run_or_exit() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
