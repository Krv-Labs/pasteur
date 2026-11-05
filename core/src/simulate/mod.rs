mod blackout;
pub mod flipper;
mod jitter;
mod patient;
mod rng;
pub mod sweep;

pub use blackout::BlackoutSimulator;
pub use flipper::FlipperSimulator;
pub use jitter::JitterSimulator;
pub use sweep::SweepSimulator;

use crate::error::CoreError;
use polars::prelude::*;

pub trait Simulator {
    fn fit(&mut self, df: &DataFrame) -> Result<(), CoreError>;
    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError>;
}
