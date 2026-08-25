//! Select top-k features using statistical tests.
//!
//! Provides [`SelectKBest`] and the [`FClassif`] scoring function
//! (ANOVA F-value between each feature and the target).

use crate::traits::{Error, FitSupervised, Result, Transform};
use polars::prelude::*;

/// Scoring function for [`SelectKBest`].
///
/// Implementors compute a score for each feature column indicating
/// how relevant it is for predicting the target. Higher scores are better.
pub trait ScoreFunction: Send + Sync {
    /// Score each feature in `x` against the target `y`.
    ///
    /// Returns a list of `(column_name, score)` pairs for numeric columns.
    /// Scores may be positive infinity when a feature separates the target
    /// perfectly. Equal scores retain their input-column order in [`SelectKBest`].
    fn score(&self, x: &DataFrame, y: &Column) -> Result<Vec<(String, f64)>>;
}

/// ANOVA F-value scoring function.
///
/// Computes the F-statistic between each feature and the target labels:
///
/// ```text
/// F = (SS_between / df_between) / (SS_within / df_within)
/// ```
///
/// Where `SS_between` is the between-group sum of squares and `SS_within`
/// is the within-group sum of squares. Higher F-values indicate stronger
/// class separation. A feature with zero within-class variance and non-zero
/// between-class variance scores positive infinity; a constant feature scores
/// zero. Rows with a null feature value or null target are excluded per column,
/// and the degrees of freedom are computed from the remaining observations. A
/// feature with no usable values, or with fewer than two observed target
/// classes, is rejected.
///
/// Requires the target column to be [`Float64`](DataType::Float64).
pub struct FClassif;

impl FClassif {
    /// Create a new `FClassif` scorer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FClassif {
    fn default() -> Self {
        Self::new()
    }
}

impl ScoreFunction for FClassif {
    fn score(&self, x: &DataFrame, y: &Column) -> Result<Vec<(String, f64)>> {
        let y_ca = y.as_materialized_series().f64().map_err(|_| {
            Error::InvalidInput(format!(
                "FClassif: target column '{}' has dtype {}; expected Float64. \
                     The target must be numeric (0, 1, 2, ...) for ANOVA F-test.",
                y.name(),
                y.dtype()
            ))
        })?;
        let y_vals: Vec<Option<f64>> = y_ca.iter().collect();
        let mut classes: Vec<f64> = y_ca.iter().flatten().collect();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup();

        if classes.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "FClassif: target has only {} unique class(es); need at least 2 \
                 to compute ANOVA F-statistic.",
                classes.len()
            )));
        }

        if y_vals.len() != x.height() {
            return Err(Error::InvalidInput(format!(
                "FClassif: feature rows ({}) and target rows ({}) don't match.",
                x.height(),
                y_vals.len()
            )));
        }

        let mut scores = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            if col.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = col.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "FClassif: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;
            let vals: Vec<Option<f64>> = ca.iter().collect();
            let observed: Vec<(f64, f64)> = vals
                .iter()
                .zip(&y_vals)
                .filter_map(|(&xv, &yv)| match (xv, yv) {
                    (Some(value), Some(target)) => Some((value, target)),
                    _ => None,
                })
                .collect();
            if observed.is_empty() {
                return Err(Error::InvalidInput(format!(
                    "FClassif: column '{name}' has no non-null values. Impute or drop it first."
                )));
            }

            let feature_origin = observed[0].0;
            let feature_mean_offset = observed
                .iter()
                .map(|(value, _)| value - feature_origin)
                .sum::<f64>()
                / observed.len() as f64;
            let mut observed_classes: Vec<f64> =
                observed.iter().map(|(_, target)| *target).collect();
            observed_classes.sort_by(|a, b| a.total_cmp(b));
            observed_classes.dedup();
            if observed_classes.len() < 2 {
                return Err(Error::InvalidInput(format!(
                    "FClassif: column '{name}' has only one target class after excluding nulls. \
                     Impute or drop it first."
                )));
            }

            let mut ss_between = 0.0;
            let mut ss_within = 0.0;

            for &cls in &observed_classes {
                let group_vals: Vec<f64> = observed
                    .iter()
                    .filter_map(|&(value, target)| {
                        if (target - cls).abs() < 1e-10 {
                            Some(value)
                        } else {
                            None
                        }
                    })
                    .collect();

                let group_origin = group_vals[0];
                let g_n = group_vals.len() as f64;
                let group_mean_offset = group_vals
                    .iter()
                    .map(|value| value - group_origin)
                    .sum::<f64>()
                    / g_n;
                let group_to_feature_mean =
                    (group_origin - feature_origin) + group_mean_offset - feature_mean_offset;

                ss_between += g_n * group_to_feature_mean.powi(2);

                for &v in &group_vals {
                    ss_within += ((v - group_origin) - group_mean_offset).powi(2);
                }
            }

            let n_observed = observed.len() as f64;
            let n_classes = observed_classes.len() as f64;
            let df_between = n_classes - 1.0;
            let df_within = n_observed - n_classes;

            let f_stat = if ss_within == 0.0 {
                if ss_between > 0.0 { f64::INFINITY } else { 0.0 }
            } else if df_within <= 0.0 {
                0.0
            } else {
                (ss_between / df_between) / (ss_within / df_within)
            };

            scores.push((name, f_stat));
        }

        Ok(scores)
    }
}

/// Select the top `k` features according to a [`ScoreFunction`].
///
/// `SelectKBest` is supervised: it implements [`FitSupervised`] and requires a
/// target `y` at `fit` time. Only `Float64` feature columns are scored; columns
/// of other dtypes are silently skipped. The target `y` must be a single
/// `Float64` column with at least two distinct classes.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::SelectKBest;
/// use featrs::feature_selection::select_kbest::FClassif;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("noise".into(), &[1.0_f64, 2.0, 3.0, 4.0]));
/// let b = Column::from(Series::new("signal".into(), &[0.0_f64, 1.0, 2.0, 10.0]));
/// let features = DataFrame::new(4, vec![a, b])?;
///
/// let target = Column::from(Series::new("y".into(), &[0.0_f64, 0.0, 1.0, 1.0]));
/// let y = DataFrame::new(4, vec![target])?;
///
/// let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));
/// skb.fit(features.clone(), y.clone())?;
/// let selected = skb.transform(features)?;
/// assert_eq!(selected.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SelectKBest {
    fitted: bool,
    k: usize,
    score_fn: Box<dyn ScoreFunction>,
    selected_columns: Option<Vec<String>>,
    scores: Option<Vec<(String, f64)>>,
}

impl SelectKBest {
    /// Create a new `SelectKBest` transformer.
    ///
    /// * `k` — number of top features to keep
    /// * `score_fn` — scoring function (e.g. [`FClassif`])
    pub fn new(k: usize, score_fn: Box<dyn ScoreFunction>) -> Self {
        Self {
            fitted: false,
            k,
            score_fn,
            selected_columns: None,
            scores: None,
        }
    }

    /// Returns the scores for each feature from the last `fit`.
    ///
    /// Returns `None` if not fitted yet. The list is sorted highest-score first.
    pub fn scores(&self) -> Option<&[(String, f64)]> {
        self.scores.as_deref()
    }
}

impl FitSupervised<DataFrame, DataFrame> for SelectKBest {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "SelectKBest.fit received a DataFrame with 0 columns.".into(),
            ));
        }
        if self.k == 0 {
            return Err(Error::InvalidInput(
                "SelectKBest: k must be greater than 0, got 0. \n\
                 Choose k >= 1 to select at least one feature."
                    .into(),
            ));
        }
        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "SelectKBest.fit: target must have exactly 1 column but got {} columns. \
                 Select a single target column.",
                y.width()
            )));
        }
        let y_col = &y.columns()[0];
        let mut scores = self.score_fn.score(&x, y_col)?;

        if scores.is_empty() {
            return Err(Error::InvalidInput(
                "SelectKBest: no f64 columns found to score. \
                 SelectKBest operates on Float64 columns only."
                    .into(),
            ));
        }

        scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let k = self.k.min(scores.len());
        let selected: Vec<String> = scores.iter().take(k).map(|(n, _)| n.clone()).collect();

        self.scores = Some(scores);
        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for SelectKBest {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "SelectKBest has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "SelectKBest has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            )
        })?;
        if cols.is_empty() {
            // Should not happen if fit succeeded, but handle gracefully
            return Err(Error::Computation(
                "SelectKBest: no columns were selected. \
                 This may mean the scoring function returned no valid scores."
                    .into(),
            ));
        }
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features() -> DataFrame {
        let a = Column::from(Series::new(
            "noise".into(),
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let b = Column::from(Series::new(
            "signal".into(),
            &[0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0],
        ));
        DataFrame::new(6, vec![a, b]).unwrap()
    }

    fn make_target_col() -> Column {
        Column::from(Series::new(
            "target".into(),
            &[0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0],
        ))
    }

    #[test]
    fn test_select_kbest_f_classif() {
        let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));
        let features = make_features();
        let y = DataFrame::new(6, vec![make_target_col()]).unwrap();

        skb.fit(features.clone(), y).unwrap();
        let result = skb.transform(features).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "signal");
    }

    #[test]
    fn test_f_classif_scores() {
        let f = FClassif::new();
        let features = make_features();
        let y_col = make_target_col();

        let scores = f.score(&features, &y_col).unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_f_classif_perfect_separator_scores_infinity() {
        let features = DataFrame::new(
            4,
            vec![
                Column::from(Series::new("constant".into(), &[5.0_f64; 4])),
                Column::from(Series::new("perfect".into(), &[0.0_f64, 0.0, 1.0, 1.0])),
            ],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores[0], ("constant".to_string(), 0.0));
        assert_eq!(scores[1].0, "perfect");
        assert_eq!(scores[1].1, f64::INFINITY);
    }

    #[test]
    fn test_f_classif_two_sample_perfect_separator_scores_infinity() {
        let features = DataFrame::new(
            2,
            vec![Column::from(Series::new("perfect".into(), &[0.0_f64, 1.0]))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores, vec![("perfect".to_string(), f64::INFINITY)]);
    }

    #[test]
    fn test_f_classif_decimal_constant_scores_zero() {
        let features = DataFrame::new(
            3,
            vec![Column::from(Series::new(
                "constant".into(),
                &[0.1_f64, 0.1, 0.1],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores, vec![("constant".to_string(), 0.0)]);
    }

    #[test]
    fn test_f_classif_decimal_perfect_separator_scores_infinity() {
        let features = DataFrame::new(
            5,
            vec![Column::from(Series::new(
                "perfect".into(),
                &[0.1_f64, 0.1, 0.1, 0.2, 0.2],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 0.0, 1.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores, vec![("perfect".to_string(), f64::INFINITY)]);
    }

    #[test]
    fn test_f_classif_preserves_one_ulp_variance_at_large_scale() {
        let low = 1.0e9_f64;
        let low_next = f64::from_bits(low.to_bits() + 1);
        let high = low + 1_000.0;
        let high_next = f64::from_bits(high.to_bits() + 1);
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "varying".into(),
                &[low, low_next, high, high_next],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert!(scores[0].1.is_finite());
        assert!(scores[0].1 > 0.0);
    }

    #[test]
    fn test_f_classif_keeps_half_ulp_means_centered() {
        let base = 1.0e16_f64;
        let next = f64::from_bits(base.to_bits() + 1);
        let next_next = f64::from_bits(base.to_bits() + 2);
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "centered".into(),
                &[base, next, next, next_next],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores, vec![("centered".to_string(), 2.0)]);
    }

    #[test]
    fn test_f_classif_partial_null_uses_observed_degrees_of_freedom() {
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "partial".into(),
                &[Some(0.0_f64), None, Some(2.0), Some(4.0)],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores[0].0, "partial");
        assert!((scores[0].1 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_f_classif_null_target_preserves_row_alignment() {
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "aligned".into(),
                &[0.0_f64, 1_000.0, 2.0, 4.0],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new(
            "target".into(),
            &[Some(0.0_f64), None, Some(1.0), Some(1.0)],
        ));

        let scores = FClassif::new().score(&features, &target).unwrap();

        assert_eq!(scores[0].0, "aligned");
        assert!((scores[0].1 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_f_classif_validates_physical_target_row_count() {
        let features = DataFrame::new(
            3,
            vec![Column::from(Series::new(
                "feature".into(),
                &[0.0_f64, 1.0, 2.0],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new(
            "target".into(),
            &[Some(0.0_f64), None, Some(1.0), Some(1.0)],
        ));

        let error = FClassif::new().score(&features, &target).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("feature rows (3) and target rows (4) don't match")
        );
    }

    #[test]
    fn test_select_kbest_ranks_partial_null_feature_by_observed_rows() {
        let features = DataFrame::new(
            4,
            vec![
                Column::from(Series::new(
                    "partial".into(),
                    &[Some(0.0_f64), None, Some(2.0), Some(4.0)],
                )),
                Column::from(Series::new("complete".into(), &[0.0_f64, 2.0, 3.0, 5.0])),
            ],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));
        let y = DataFrame::new(4, vec![target]).unwrap();
        let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));

        skb.fit(features.clone(), y).unwrap();

        assert_eq!(skb.scores().unwrap()[0], ("complete".to_string(), 4.5));
        assert_eq!(
            skb.transform(features).unwrap().get_column_names(),
            &["complete"]
        );
    }

    #[test]
    fn test_f_classif_all_null_feature_errors() {
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "all_null".into(),
                &[None::<f64>; 4],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let error = FClassif::new().score(&features, &target).unwrap_err();

        assert!(error.to_string().contains("no non-null values"));
    }

    #[test]
    fn test_f_classif_feature_with_one_observed_class_errors() {
        let features = DataFrame::new(
            4,
            vec![Column::from(Series::new(
                "one_class".into(),
                &[Some(0.0_f64), Some(1.0), None, None],
            ))],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));

        let error = FClassif::new().score(&features, &target).unwrap_err();

        assert!(error.to_string().contains("only one target class"));
    }

    #[test]
    fn test_select_kbest_ranks_perfect_separator_first() {
        let features = DataFrame::new(
            6,
            vec![
                make_features().column("signal").unwrap().clone(),
                Column::from(Series::new(
                    "perfect".into(),
                    &[0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0],
                )),
            ],
        )
        .unwrap();
        let y = DataFrame::new(6, vec![make_target_col()]).unwrap();
        let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));

        skb.fit(features.clone(), y).unwrap();

        assert_eq!(skb.scores().unwrap()[0].0, "perfect");
        let selected = skb.transform(features).unwrap();
        assert_eq!(selected.get_column_names(), &["perfect"]);
    }

    #[test]
    fn test_select_kbest_breaks_infinite_ties_by_input_order() {
        let features = DataFrame::new(
            4,
            vec![
                Column::from(Series::new("first".into(), &[0.0_f64, 0.0, 1.0, 1.0])),
                Column::from(Series::new("second".into(), &[2.0_f64, 2.0, 3.0, 3.0])),
            ],
        )
        .unwrap();
        let target = Column::from(Series::new("target".into(), &[0.0_f64, 0.0, 1.0, 1.0]));
        let y = DataFrame::new(4, vec![target]).unwrap();
        let mut skb = SelectKBest::new(2, Box::new(FClassif::new()));

        skb.fit(features, y).unwrap();

        let names: Vec<&str> = skb
            .scores()
            .unwrap()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn test_select_kbest_k_zero_rejected() {
        let mut skb = SelectKBest::new(0, Box::new(FClassif::new()));
        let features = make_features();
        let y = DataFrame::new(6, vec![make_target_col()]).unwrap();
        let result = skb.fit(features, y);
        assert!(result.is_err(), "k=0 should be rejected at fit time");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("k must be greater than 0"),
            "error message should mention k"
        );
    }
}
