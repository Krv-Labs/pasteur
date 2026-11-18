mod blackout;
pub mod flipper;
mod jitter;
mod patient;
mod rng;
mod scramble;
mod site;
pub mod sweep;
mod units;

pub use blackout::BlackoutSimulator;
pub use flipper::FlipperSimulator;
pub use jitter::JitterSimulator;
pub use scramble::ScrambleSimulator;
pub use site::{SiteConfig, SiteSimulator, SiteStep};
pub use sweep::SweepSimulator;
pub use units::{unit_preset, UnitsSimulator, UNIT_PRESETS};

use crate::error::CoreError;
use polars::prelude::*;

pub trait Simulator {
    fn fit(&mut self, df: &DataFrame) -> Result<(), CoreError>;
    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError>;
}
