mod blackout;
pub mod flipper;
mod jitter;
mod patient;
mod rng;

pub use blackout::BlackoutSimulator;
pub use flipper::FlipperSimulator;
pub use jitter::JitterSimulator;

use crate::error::CoreError;
use polars::prelude::*;

pub trait Simulator {
    fn fit(&mut self, df: &DataFrame) -> Result<(), CoreError>;
    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError>;
}
