pub mod compare;
pub mod error;
pub mod evaluate;
pub mod schema;
pub mod simulate;

pub use compare::Comparer;
pub use error::CoreError;
pub use evaluate::{
    calculate_flipper_stability, find_flip_points, Evaluator, FlipperPairResult, Model,
};
pub use schema::*;
pub use simulate::{
    flipper::{is_flipper_meta_column, FLIPPER_META_COLS, PAIR_ID_COL, STEP_COL},
    BlackoutSimulator, FlipperSimulator, JitterSimulator, Simulator,
};
