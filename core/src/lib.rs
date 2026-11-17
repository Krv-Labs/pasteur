pub mod compare;
pub mod error;
pub mod evaluate;
pub mod schema;
pub mod simulate;

pub use compare::Comparer;
pub use error::CoreError;
pub use evaluate::{
    calculate_flipper_stability, find_flip_points, parse_loss, Evaluator, FlipperPairResult,
    FlipperScores, Model, SweepScores,
};
pub use schema::*;
pub use simulate::{
    flipper::{is_flipper_meta_column, FLIPPER_META_COLS, PAIR_ID_COL, PAIR_LABEL_COL, STEP_COL},
    sweep::{is_sweep_meta_column, SWEEP_META_COLS},
    unit_preset, BlackoutSimulator, FlipperSimulator, JitterSimulator, ScrambleSimulator,
    Simulator, SweepSimulator, UnitsSimulator, UNIT_PRESETS,
};
