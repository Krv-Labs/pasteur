//! Metrics against per-row oracle labels. When a perturbation can change the
//! correct answer, each simulated row carries its own true label (from the
//! user's oracle, e.g. a rule engine), and the metrics here score the
//! model's *decisions* against it: per-variant accuracy, jitter flips split
//! by whether the oracle also changed, where the flipper boundary sits
//! relative to the oracle's, and abstention under blackout.

use std::collections::{BTreeMap, HashMap};

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::flipper::{find_flip_points, flipper_model_features, multi_crossing};
use super::multi::{argmax_decision, label_matrix, threshold_decision, Columns};
use super::{predict_matrix, Model};
use crate::error::CoreError;
use crate::schema::{EvaluationConfig, SimulationVariant, TaskType};

/// Marks a run whose variant labels came from an oracle, row by row, rather
/// than being copied from the clean row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OracleConfig {
    /// The output class that means "cannot decide" (e.g.
    /// `cannot_determine`), matched against the model's class names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstain_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleScores {
    /// Share of rows whose decision equals the oracle label exactly (every
    /// label, for multilabel), per variant: `clean`, `blackout`, `jitter_0`…,
    /// `flipper`.
    pub accuracy: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jitter: Option<OracleJitterScores>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flipper: Option<OracleFlipperScores>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstain: Option<AbstainScores>,
}

/// Each (row, jitter draw) compared with the same row in the clean data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleJitterScores {
    /// Share of (row, draw) pairs where the oracle label did not change but
    /// the model's decision did. Lower is better.
    pub flip_rate_label_preserved: Option<f64>,
    /// Share of (row, draw) pairs where the oracle label changed and the
    /// model's decision changed to the new label. Higher is better.
    pub flip_rate_label_changed: Option<f64>,
    pub n_label_preserved: usize,
    pub n_label_changed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleFlipperScores {
    pub n_pairs: usize,
    /// Share of pairs whose oracle label never changes along the path.
    pub oracle_never_flipped: f64,
    /// Pairs where both the model and the oracle flip; `boundary_error` is
    /// taken over these.
    pub n_pairs_compared: usize,
    /// Median over pairs of `flip_t − oracle_flip_t`: negative means the
    /// model flips before the oracle does, positive after.
    pub boundary_error_median: Option<f64>,
    /// Interquartile range of `flip_t − oracle_flip_t`.
    pub boundary_error_iqr: Option<f64>,
}

/// Blackout rows scored against the oracle's abstain class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstainScores {
    pub class: String,
    /// Of the rows where the oracle abstains, the share where the model
    /// abstains too. Higher is better.
    pub abstain_when_missing: Option<f64>,
    /// Of the rows where the oracle abstains, the share where the model
    /// decides something else.
    pub decided_when_missing: Option<f64>,
    pub confusion: AbstainConfusion,
}

/// Row counts, named `oracle_<…>_model_<…>`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbstainConfusion {
    pub oracle_abstain_model_abstain: usize,
    pub oracle_abstain_model_decided: usize,
    pub oracle_decided_model_abstain: usize,
    pub oracle_decided_model_decided: usize,
}

type Decisions = Vec<Vec<bool>>;

/// One variant: model output, oracle labels, and both as decisions.
struct Scored {
    probs: Columns,
    y: Columns,
    decided: Decisions,
    truth: Decisions,
}

/// Oracle metrics for one dataset, or `None` when the run is not
/// oracle-labelled. `names` and `thresholds` are the model's output names and
/// decision thresholds, as `score_classifier` uses them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_oracle(
    model: &dyn Model,
    clean: &SimulationVariant,
    variants: &HashMap<String, SimulationVariant>,
    config: &EvaluationConfig,
    names: &[String],
    thresholds: &[f64],
    clean_probs: Option<&Columns>,
    variant_probs: &HashMap<String, Columns>,
) -> Result<Option<OracleScores>, CoreError> {
    let Some(oracle) = &config.oracle else {
        return Ok(None);
    };
    let task = config.task;
    let abstain = oracle
        .abstain_class
        .as_deref()
        .map(|class| abstain_index(task, names, class).map(|j| (class, j)))
        .transpose()?;
    let score =
        |x: &DataFrame, y: &DataFrame, cached: Option<&Columns>| -> Result<Scored, CoreError> {
            let y = label_matrix(y, task)?;
            let probs = match cached {
                Some(p) => p.clone(),
                None => predict_matrix(model, x, names.len())?,
            };
            let decided = decisions(task, &probs, thresholds);
            let truth = decisions(task, &y, &vec![0.5; y.len()]);
            Ok(Scored {
                probs,
                y,
                decided,
                truth,
            })
        };

    let base = score(&clean.x, &clean.y, clean_probs)?;
    let mut accuracy = BTreeMap::from([("clean".to_string(), exact_match(&base))]);
    let mut jitter: Option<JitterTally> = None;
    let mut flipper = None;
    let mut abstain_scores = None;
    for (name, variant) in variants {
        let is_flipper = name == "flipper";
        let x = if is_flipper {
            flipper_model_features(&variant.x)?
        } else {
            variant.x.clone()
        };
        let scored = score(&x, &variant.y, variant_probs.get(name))?;
        accuracy.insert(name.clone(), exact_match(&scored));
        if name.starts_with("jitter") {
            jitter
                .get_or_insert_with(JitterTally::default)
                .add(&base, &scored, name)?;
        } else if is_flipper {
            flipper = Some(flipper_scores(&variant.x, task, &scored, thresholds)?);
        } else if name == "blackout" {
            abstain_scores = abstain.map(|(class, j)| abstain_on(class, j, &scored));
        }
    }
    Ok(Some(OracleScores {
        accuracy,
        jitter: jitter.as_ref().map(JitterTally::scores),
        flipper,
        abstain: abstain_scores,
    }))
}

fn abstain_index(task: TaskType, names: &[String], class: &str) -> Result<usize, CoreError> {
    if !matches!(task, TaskType::Multiclass | TaskType::Multilabel) {
        return Err(CoreError::InvalidConfig(format!(
            "--abstain-class names one of the model's classes; a {task} model has none to \
             abstain with"
        )));
    }
    names.iter().position(|n| n == class).ok_or_else(|| {
        CoreError::InvalidConfig(format!(
            "abstain class {class:?} is not one of the model's output classes {names:?}"
        ))
    })
}

/// Per row, the decision as one bool per output: argmax for multiclass, at
/// or over threshold otherwise (a binary model is the one-column case).
fn decisions(task: TaskType, cols: &Columns, thresholds: &[f64]) -> Decisions {
    let decide = match task {
        TaskType::Multiclass => argmax_decision,
        _ => threshold_decision,
    };
    let n = cols.first().map_or(0, Vec::len);
    (0..n).map(|i| decide(cols, i, thresholds)).collect()
}

fn exact_match(s: &Scored) -> f64 {
    let n = s.truth.len();
    if n == 0 {
        return 0.0;
    }
    let hits = s.decided.iter().zip(&s.truth).filter(|(d, t)| d == t);
    hits.count() as f64 / n as f64
}

/// (row, draw) counts across every jitter draw.
#[derive(Default)]
struct JitterTally {
    preserved: usize,
    preserved_flipped: usize,
    changed: usize,
    changed_followed: usize,
}

impl JitterTally {
    /// Adds one draw, compared row by row with the clean rows it came from.
    fn add(&mut self, clean: &Scored, draw: &Scored, name: &str) -> Result<(), CoreError> {
        if clean.truth.len() != draw.truth.len() {
            return Err(CoreError::InvalidConfig(format!(
                "{name} has {} rows but the clean data has {}; jitter rows must match clean rows",
                draw.truth.len(),
                clean.truth.len()
            )));
        }
        for i in 0..clean.truth.len() {
            let model_changed = draw.decided[i] != clean.decided[i];
            if draw.truth[i] == clean.truth[i] {
                self.preserved += 1;
                self.preserved_flipped += usize::from(model_changed);
            } else {
                self.changed += 1;
                self.changed_followed +=
                    usize::from(model_changed && draw.decided[i] == draw.truth[i]);
            }
        }
        Ok(())
    }

    fn scores(&self) -> OracleJitterScores {
        let rate = |hits: usize, n: usize| (n > 0).then(|| hits as f64 / n as f64);
        OracleJitterScores {
            flip_rate_label_preserved: rate(self.preserved_flipped, self.preserved),
            flip_rate_label_changed: rate(self.changed_followed, self.changed),
            n_label_preserved: self.preserved,
            n_label_changed: self.changed,
        }
    }
}

/// Where the model's decision flips along each pair's path (`flip_t`, as in
/// `score_flipper`) against where the oracle label flips (`oracle_flip_t`).
/// The oracle's labels are 0/1, so its crossing lands halfway between the
/// last grid step before the change and the first one after it.
fn flipper_scores(
    grid: &DataFrame,
    task: TaskType,
    scored: &Scored,
    thresholds: &[f64],
) -> Result<OracleFlipperScores, CoreError> {
    let flips = |cols: &Columns, thresholds: &[f64]| {
        let crossing = match task {
            TaskType::Multiclass | TaskType::Multilabel => {
                multi_crossing(grid, task, cols, cols.len(), thresholds)?.1
            }
            TaskType::Binary | TaskType::Regression => {
                cols[0].iter().map(|p| p - thresholds[0]).collect()
            }
        };
        find_flip_points(grid, &Series::new("crossing".into(), crossing), 0.0)
    };
    let model = flips(&scored.probs, thresholds)?;
    let oracle = flips(&scored.y, &vec![0.5; scored.y.len()])?;
    let n_pairs = oracle.len();
    let oracle_never = oracle.iter().filter(|r| r.flip_t.is_none()).count();
    let mut errors: Vec<f64> = model
        .iter()
        .zip(&oracle)
        .filter_map(|(m, o)| Some(m.flip_t? - o.flip_t?))
        .collect();
    errors.sort_by(f64::total_cmp);
    Ok(OracleFlipperScores {
        n_pairs,
        oracle_never_flipped: if n_pairs == 0 {
            0.0
        } else {
            oracle_never as f64 / n_pairs as f64
        },
        n_pairs_compared: errors.len(),
        boundary_error_median: quantile(&errors, 0.5),
        boundary_error_iqr: quantile(&errors, 0.75)
            .zip(quantile(&errors, 0.25))
            .map(|(q3, q1)| q3 - q1),
    })
}

/// Linear-interpolation quantile of sorted values (numpy's default).
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let pos = q * last as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo]))
}

fn abstain_on(class: &str, j: usize, scored: &Scored) -> AbstainScores {
    let mut c = AbstainConfusion::default();
    for (decided, truth) in scored.decided.iter().zip(&scored.truth) {
        match (truth[j], decided[j]) {
            (true, true) => c.oracle_abstain_model_abstain += 1,
            (true, false) => c.oracle_abstain_model_decided += 1,
            (false, true) => c.oracle_decided_model_abstain += 1,
            (false, false) => c.oracle_decided_model_decided += 1,
        }
    }
    let missing = c.oracle_abstain_model_abstain + c.oracle_abstain_model_decided;
    let share = |n: usize| (missing > 0).then(|| n as f64 / missing as f64);
    AbstainScores {
        class: class.to_string(),
        abstain_when_missing: share(c.oracle_abstain_model_abstain),
        decided_when_missing: share(c.oracle_abstain_model_decided),
        confusion: c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::flipper::{
        LABEL_A_COL, LABEL_B_COL, PAIR_ID_COL, SOURCE_A_ROW_COL, SOURCE_B_ROW_COL, STEP_COL, T_COL,
    };

    /// Returns its input's `p` column, so tests set the model's output.
    struct Echo;

    impl Model for Echo {
        fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
            Ok(df.select(["p"])?)
        }
    }

    fn variant(p: &[f64], y: &[i64]) -> SimulationVariant {
        SimulationVariant {
            x: df!["p" => p].unwrap(),
            y: df!["label" => y].unwrap(),
        }
    }

    fn config(oracle: bool) -> EvaluationConfig {
        EvaluationConfig {
            n_jitter_iters: 1,
            allow_partial: true,
            metrics: vec![],
            flip_threshold: 0.5,
            task: TaskType::Binary,
            oracle: oracle.then(OracleConfig::default),
        }
    }

    fn run(
        clean: &SimulationVariant,
        variants: Vec<(&str, SimulationVariant)>,
    ) -> Option<OracleScores> {
        let variants = variants
            .into_iter()
            .map(|(n, v)| (n.to_string(), v))
            .collect();
        score_oracle(
            &Echo,
            clean,
            &variants,
            &config(true),
            &["proba".into()],
            &[0.5],
            None,
            &HashMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn nothing_without_oracle_labels() {
        let clean = variant(&[0.9], &[1]);
        let out = score_oracle(
            &Echo,
            &clean,
            &HashMap::new(),
            &config(false),
            &["proba".into()],
            &[0.5],
            None,
            &HashMap::new(),
        )
        .unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn binary_jitter_splits_flips_by_oracle_change() {
        // Rows: 0 stays, model stays; 1 stays, model flips (bad);
        // 2 changes, model follows (good); 3 changes, model stays.
        let clean = variant(&[0.9, 0.9, 0.1, 0.1], &[1, 1, 0, 0]);
        let draw = variant(&[0.8, 0.2, 0.7, 0.3], &[1, 1, 1, 1]);
        let s = run(&clean, vec![("jitter_0", draw)]).unwrap();
        let j = s.jitter.unwrap();
        assert_eq!((j.n_label_preserved, j.n_label_changed), (2, 2));
        assert_eq!(j.flip_rate_label_preserved, Some(0.5));
        assert_eq!(j.flip_rate_label_changed, Some(0.5));
        assert_eq!(s.accuracy["clean"], 1.0);
        assert_eq!(s.accuracy["jitter_0"], 0.5);
    }

    #[test]
    fn rates_pool_over_draws() {
        let clean = variant(&[0.9], &[1]);
        let s = run(
            &clean,
            vec![
                ("jitter_0", variant(&[0.9], &[1])),
                ("jitter_1", variant(&[0.1], &[1])),
            ],
        )
        .unwrap();
        let j = s.jitter.unwrap();
        assert_eq!(j.n_label_preserved, 2);
        assert_eq!(j.flip_rate_label_preserved, Some(0.5));
        assert_eq!(j.flip_rate_label_changed, None);
    }

    #[test]
    fn boundary_error_is_model_minus_oracle_flip() {
        // One pair, t = 0, .25, .5, .75, 1. The model crosses 0.5 at t=0.625
        // (0.4 → 0.6 between steps 2 and 3); the oracle flips between steps
        // 1 and 2, so oracle_flip_t = 0.375.
        let n = 5;
        let grid = DataFrame::new(
            n,
            vec![
                Series::new(PAIR_ID_COL.into(), vec![0u32; n]).into(),
                Series::new(STEP_COL.into(), (0..n as u32).collect::<Vec<_>>()).into(),
                Series::new(T_COL.into(), [0.0, 0.25, 0.5, 0.75, 1.0]).into(),
                Series::new(SOURCE_A_ROW_COL.into(), vec![0u32; n]).into(),
                Series::new(SOURCE_B_ROW_COL.into(), vec![1u32; n]).into(),
                Series::new(LABEL_A_COL.into(), vec![0.0f64; n]).into(),
                Series::new(LABEL_B_COL.into(), vec![1.0f64; n]).into(),
                Series::new("p".into(), [0.1, 0.2, 0.4, 0.6, 0.9]).into(),
            ],
        )
        .unwrap();
        let flipper = SimulationVariant {
            x: grid,
            y: df!["label" => [0i64, 0, 1, 1, 1]].unwrap(),
        };
        let clean = variant(&[0.9], &[1]);
        let f = run(&clean, vec![("flipper", flipper)])
            .unwrap()
            .flipper
            .unwrap();
        assert_eq!((f.n_pairs, f.n_pairs_compared), (1, 1));
        assert_eq!(f.oracle_never_flipped, 0.0);
        assert!((f.boundary_error_median.unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(f.boundary_error_iqr, Some(0.0));
    }

    #[test]
    fn abstain_confusion_on_blackout() {
        let names: Vec<String> = ["yes", "no", "unknown"].map(String::from).into();
        let clean = SimulationVariant {
            x: df!["a" => [0.9], "b" => [0.1], "c" => [0.0]].unwrap(),
            y: df!["g1" => [1i64], "g2" => [0i64], "g3" => [0i64]].unwrap(),
        };
        // Oracle abstains on rows 0..3; the model abstains on rows 0 and 3.
        let blackout = SimulationVariant {
            x: df![
                "a" => [0.1, 0.6, 0.2, 0.1],
                "b" => [0.2, 0.3, 0.7, 0.1],
                "c" => [0.7, 0.1, 0.1, 0.8],
            ]
            .unwrap(),
            y: df![
                "g1" => [0i64, 0, 0, 1],
                "g2" => [0i64, 0, 0, 0],
                "g3" => [1i64, 1, 1, 0],
            ]
            .unwrap(),
        };
        struct Three;
        impl Model for Three {
            fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
                Ok(df.select(["a", "b", "c"])?)
            }
        }
        let mut cfg = config(true);
        cfg.task = TaskType::Multiclass;
        cfg.oracle = Some(OracleConfig {
            abstain_class: Some("unknown".into()),
        });
        let variants = HashMap::from([("blackout".to_string(), blackout)]);
        let s = score_oracle(
            &Three,
            &clean,
            &variants,
            &cfg,
            &names,
            &[0.5; 3],
            None,
            &HashMap::new(),
        )
        .unwrap()
        .unwrap();
        let a = s.abstain.unwrap();
        assert!((a.abstain_when_missing.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((a.decided_when_missing.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(a.confusion.oracle_abstain_model_abstain, 1);
        assert_eq!(a.confusion.oracle_abstain_model_decided, 2);
        assert_eq!(a.confusion.oracle_decided_model_abstain, 1);
        assert_eq!(a.confusion.oracle_decided_model_decided, 0);
        assert_eq!(s.accuracy["blackout"], 0.25);

        cfg.oracle = Some(OracleConfig {
            abstain_class: Some("maybe".into()),
        });
        let err = score_oracle(
            &Three,
            &clean,
            &variants,
            &cfg,
            &names,
            &[0.5; 3],
            None,
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not one of"), "{err}");
    }

    #[test]
    fn quantiles_interpolate() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), Some(2.5));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.25), Some(2.0));
    }
}
