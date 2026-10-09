use std::path::Path;

use anyhow::Result;
use pasteur_core::TaskType;
use polars::prelude::*;

use crate::args::TargetArgs;
use crate::data::labels::derive_label_matrix;
use crate::data::targets::derive_target_column;

/// Where the true outcome for each row comes from: cohort membership for a
/// classifier, a target table for a regression model.
pub enum LabelSource<'a> {
    Cohorts {
        path: &'a Path,
        group_ids: &'a [u32],
    },
    Targets {
        path: &'a Path,
        id_col: &'a str,
        target_col: &'a str,
    },
}

impl<'a> LabelSource<'a> {
    /// The source the flags select for `task`, or `None` when none was given.
    /// Flags for the other kind of task are an error rather than ignored, so a
    /// regression run never silently scores against cohort labels.
    pub fn from_args(
        task: TaskType,
        labels: Option<&'a Path>,
        group_ids: &'a [u32],
        targets: &'a TargetArgs,
    ) -> Result<Option<Self>> {
        match task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
                if targets.targets.is_some() || targets.target_col.is_some() {
                    anyhow::bail!(
                        "--targets and --target-col are for --task regression; a {task} model \
                         is scored against cohorts from --labels"
                    );
                }
                Ok(labels.map(|path| LabelSource::Cohorts { path, group_ids }))
            }
            TaskType::Regression => {
                if labels.is_some() {
                    anyhow::bail!(
                        "--labels holds cohorts for classifiers; a regression model is scored \
                         against --targets with --target-col"
                    );
                }
                match (&targets.targets, &targets.target_col) {
                    (Some(path), Some(target_col)) => Ok(Some(LabelSource::Targets {
                        path,
                        id_col: &targets.targets_id_col,
                        target_col,
                    })),
                    (Some(_), None) => {
                        anyhow::bail!("--targets needs --target-col to name the target column")
                    }
                    (None, Some(_)) => {
                        anyhow::bail!("--target-col names a column of --targets; pass --targets")
                    }
                    (None, None) => Ok(None),
                }
            }
        }
    }

    /// `from_args` for commands that cannot run without labels.
    pub fn required(
        task: TaskType,
        labels: Option<&'a Path>,
        group_ids: &'a [u32],
        targets: &'a TargetArgs,
    ) -> Result<Self> {
        Self::from_args(task, labels, group_ids, targets)?.ok_or_else(|| match task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
                anyhow::anyhow!("a {task} evaluation needs --labels")
            }
            TaskType::Regression => {
                anyhow::anyhow!("a regression evaluation needs --targets and --target-col")
            }
        })
    }

    /// The label matrix (classification) or target column (regression) for
    /// `source_row_ids`, in row order.
    pub fn derive(&self, task: TaskType, source_row_ids: &Series) -> Result<DataFrame> {
        match self {
            LabelSource::Cohorts { path, group_ids } => {
                derive_label_matrix(path, group_ids, task, source_row_ids)
            }
            LabelSource::Targets {
                path,
                id_col,
                target_col,
            } => derive_target_column(path, id_col, target_col, source_row_ids),
        }
    }
}

/// The `--flip-threshold` to score with: 0.5 by default for a classifier,
/// and for regression the user's cutoff, required when flipper is scored.
/// A regression run that does not score flipper never reads it.
pub fn resolve_flip_threshold(task: TaskType, sim_type: &str, given: Option<f64>) -> Result<f64> {
    match (task, given) {
        (TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel, t) => Ok(t.unwrap_or(0.5)),
        (TaskType::Regression, Some(t)) if t.is_finite() => Ok(t),
        (TaskType::Regression, Some(t)) => {
            anyhow::bail!("--flip-threshold must be finite, got {t}")
        }
        (TaskType::Regression, None) if sim_type == "flipper" => anyhow::bail!(
            "regression flipper needs --flip-threshold: the clinical cutoff, in the target's \
             units, that `simulate` sampled pairs across"
        ),
        // Blackout and jitter scoring never read the cutoff.
        (TaskType::Regression, None) => Ok(f64::NAN),
    }
}
