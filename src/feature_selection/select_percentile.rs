//! Percentile-based feature selection.
//!
//! [`SelectPercentile`] scores every feature with a
//! [`ScoreFunction`] and keeps the highest-scoring `percentile` percent of
//! them.

use crate::feature_selection::select_kbest::{FClassif, ScoreFunction};
use crate::traits::{Error, FitSupervised, Result, Transform};
use polars::prelude::*;

/// Select the top `percentile` percent of features according to a
/// [`ScoreFunction`].
///
/// `SelectPercentile` is supervised: it implements [`FitSupervised`] and
/// requires a target `y` at `fit` time. Only `Float64` feature columns are
/// scored; columns of other dtypes are silently skipped. The target `y` must be
/// a single column.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::select_percentile::SelectPercentile;
/// use featrs::feature_selection::select_kbest::FClassif;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let noise = Column::from(Series::new("noise".into(), &[1.0_f64, 2.0, 3.0, 4.0]));
/// let signal = Column::from(Series::new("signal".into(), &[0.0_f64, 1.0, 2.0, 10.0]));
/// let features = DataFrame::new(4, vec![noise, signal])?;
///
/// let target = Column::from(Series::new("y".into(), &[0.0_f64, 0.0, 1.0, 1.0]));
/// let y = DataFrame::new(4, vec![target])?;
///
/// let mut sp = SelectPercentile::new()
///     .percentile(50.0)
///     .score_func(Box::new(FClassif::new()));
/// sp.fit(features.clone(), y)?;
/// let selected = sp.transform(features)?;
/// assert_eq!(selected.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SelectPercentile {
    fitted: bool,
    percentile: f32,
    score_fn: Box<dyn ScoreFunction>,
    selected_columns: Option<Vec<String>>,
    scores: Option<Vec<(String, f64)>>,
}

impl SelectPercentile {
    /// Create a new `SelectPercentile` transformer.
    ///
    /// Defaults to `percentile = 10.0` and the [`FClassif`] scorer.
    pub fn new() -> Self {
        Self {
            fitted: false,
            percentile: 10.0,
            score_fn: Box::new(FClassif::new()),
            selected_columns: None,
            scores: None,
        }
    }

    /// Set the percentage of features to keep, in `(0.0, 100.0]`.
    ///
    /// The count is relative to the columns the scoring function actually
    /// scored, rounded up, and clamped to at least one column.
    pub fn percentile(mut self, p: f32) -> Self {
        self.percentile = p;
        self
    }

    /// Set the scoring function (e.g. `Box::new(FRegression)`).
    pub fn score_func(mut self, f: Box<dyn ScoreFunction>) -> Self {
        self.score_fn = f;
        self
    }

    /// Returns the scores for each feature from the last `fit`.
    ///
    /// Returns `None` if not fitted yet. The list is sorted highest-score
    /// first, with ties broken by column name and `NaN` scores last.
    pub fn scores(&self) -> Option<&[(String, f64)]> {
        self.scores.as_deref()
    }
}

impl Default for SelectPercentile {
    fn default() -> Self {
        Self::new()
    }
}

impl FitSupervised<DataFrame, DataFrame> for SelectPercentile {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        // A failed fit must not leave a stale selection behind.
        self.fitted = false;
        self.selected_columns = None;
        self.scores = None;

        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "SelectPercentile.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "SelectPercentile.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }
        if !(self.percentile > 0.0 && self.percentile <= 100.0) {
            // Also rejects NaN, for which both comparisons are false.
            return Err(Error::InvalidInput(format!(
                "SelectPercentile: percentile must be in (0.0, 100.0] but got {}.",
                self.percentile
            )));
        }
        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "SelectPercentile.fit: target must have exactly 1 column but got {} columns. \
                 Select a single target column.",
                y.width()
            )));
        }
        let y_col = &y.columns()[0];
        let mut scores = self.score_fn.score(&x, y_col)?;

        if scores.is_empty() {
            return Err(Error::InvalidInput(
                "SelectPercentile: no f64 columns found to score. \
                 SelectPercentile operates on Float64 columns only."
                    .into(),
            ));
        }

        // Rank highest score first, `NaN` last, ties by column name ascending.
        // `f64::total_cmp` orders `NaN` as the greatest value, so map it to
        // `-inf` for ranking while keeping the raw score in `self.scores`.
        let ranked = |score: f64| {
            if score.is_nan() {
                f64::NEG_INFINITY
            } else {
                score
            }
        };
        scores.sort_by(|a, b| {
            ranked(b.1)
                .total_cmp(&ranked(a.1))
                .then_with(|| a.0.cmp(&b.0))
        });

        let n = scores.len();
        let keep = ((f64::from(self.percentile) / 100.0) * n as f64).ceil() as usize;
        let keep = keep.clamp(1, n);
        let selected: Vec<String> = scores
            .iter()
            .take(keep)
            .map(|(name, _)| name.clone())
            .collect();

        self.selected_columns = Some(selected);
        self.scores = Some(scores);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for SelectPercentile {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "SelectPercentile has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "SelectPercentile has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            )
        })?;
        if cols.is_empty() {
            return Err(Error::Computation(
                "SelectPercentile: no columns were selected. \
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

    /// Four scored columns with distinct scores:
    /// `sig` (perfect separator, +inf), `med` (F = 1.5), then `const` and
    /// `noise` tied at F = 0.
    fn make_features() -> DataFrame {
        let sig = Column::from(Series::new(
            "sig".into(),
            &[0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0],
        ));
        let med = Column::from(Series::new(
            "med".into(),
            &[0.0f64, 1.0, 2.0, 1.0, 2.0, 3.0],
        ));
        let noise = Column::from(Series::new(
            "noise".into(),
            &[1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0],
        ));
        let konst = Column::from(Series::new("const".into(), &[5.0f64; 6]));
        DataFrame::new(6, vec![sig, med, noise, konst]).unwrap()
    }

    fn make_target() -> DataFrame {
        let y = Column::from(Series::new(
            "target".into(),
            &[0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0],
        ));
        DataFrame::new(6, vec![y]).unwrap()
    }

    #[test]
    fn test_percentile_50_keeps_top_half() {
        let mut sp = SelectPercentile::new().percentile(50.0);
        let features = make_features();

        sp.fit(features.clone(), make_target()).unwrap();

        assert_eq!(sp.scores().unwrap()[0], ("sig".to_string(), f64::INFINITY));
        assert_eq!(
            sp.transform(features).unwrap().get_column_names(),
            &["sig", "med"]
        );
    }

    #[test]
    fn test_percentile_100_keeps_all_scored_columns() {
        let mut sp = SelectPercentile::new().percentile(100.0);
        let features = make_features();

        sp.fit(features.clone(), make_target()).unwrap();

        let selected = sp.transform(features).unwrap();
        let names: Vec<String> = selected
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .collect();
        assert_eq!(names.len(), 4);
        assert_eq!(&names[..2], &["sig", "med"]);
        assert!(names[2..].contains(&"const".to_string()));
        assert!(names[2..].contains(&"noise".to_string()));
    }

    #[test]
    fn test_tiny_percentile_rounds_up_to_one_column() {
        let mut sp = SelectPercentile::new().percentile(1.0);
        let features = make_features();

        sp.fit(features.clone(), make_target()).unwrap();

        assert_eq!(sp.transform(features).unwrap().get_column_names(), &["sig"]);
    }

    #[test]
    fn test_percentile_rounds_up_to_ceil_of_scored_columns() {
        // 62.5% of 4 scored columns = 2.5 -> 3 columns.
        let mut sp = SelectPercentile::new().percentile(62.5);
        let features = make_features();

        sp.fit(features.clone(), make_target()).unwrap();

        assert_eq!(sp.transform(features).unwrap().width(), 3);
    }

    #[test]
    fn test_percentile_is_relative_to_scored_columns_only() {
        // The `label` string column is not scored, so 50% of the 4 Float64
        // columns is 2, not 50% of the 5 input columns.
        let mut features = make_features();
        let label = Column::from(Series::new("label".into(), &["a", "b", "c", "d", "e", "f"]));
        features.with_column(label).unwrap();
        let mut sp = SelectPercentile::new().percentile(50.0);

        sp.fit(features.clone(), make_target()).unwrap();

        assert_eq!(sp.scores().unwrap().len(), 4);
        assert_eq!(sp.transform(features).unwrap().width(), 2);
    }

    #[test]
    fn test_no_f64_columns_errors() {
        let label = Column::from(Series::new("label".into(), &["a", "b", "c", "d", "e", "f"]));
        let features = DataFrame::new(6, vec![label]).unwrap();
        let mut sp = SelectPercentile::new();

        let error = sp.fit(features, make_target()).unwrap_err();

        assert!(
            error.to_string().contains("no f64 columns found to score"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let sp = SelectPercentile::new();

        let error = sp.transform(make_features()).unwrap_err();

        assert!(matches!(error, Error::NotFitted(_)));
        assert!(sp.scores().is_none());
    }

    /// Scorer returning fixed scores, for exercising ranking without needing
    /// real statistics.
    struct FixedScores(Vec<(&'static str, f64)>);

    impl ScoreFunction for FixedScores {
        fn score(&self, _x: &DataFrame, _y: &Column) -> Result<Vec<(String, f64)>> {
            Ok(self
                .0
                .iter()
                .map(|(name, score)| (name.to_string(), *score))
                .collect())
        }
    }

    fn fixed(names_and_scores: Vec<(&'static str, f64)>) -> SelectPercentile {
        SelectPercentile::new()
            .percentile(100.0)
            .score_func(Box::new(FixedScores(names_and_scores)))
    }

    fn scored_names(sp: &SelectPercentile) -> Vec<String> {
        sp.scores()
            .unwrap()
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    #[test]
    fn test_equal_scores_break_ties_by_column_name() {
        let mut sp = fixed(vec![("b_tie", 1.0), ("a_tie", 1.0), ("c_tie", 2.0)]);

        sp.fit(make_features(), make_target()).unwrap();

        assert_eq!(scored_names(&sp), ["c_tie", "a_tie", "b_tie"]);
    }

    #[test]
    fn test_nan_scores_rank_last() {
        let mut sp = fixed(vec![("a_nan", f64::NAN), ("b_low", 1.0), ("c_high", 2.0)]);

        sp.fit(make_features(), make_target()).unwrap();

        assert_eq!(scored_names(&sp), ["c_high", "b_low", "a_nan"]);
        // The stored scores keep the raw NaN, only the ranking maps it away.
        assert!(sp.scores().unwrap()[2].1.is_nan());
    }

    #[test]
    fn test_constant_feature_scores_zero_and_ranks_last() {
        let konst = Column::from(Series::new("const".into(), &[5.0f64; 6]));
        let med = Column::from(Series::new(
            "med".into(),
            &[0.0f64, 1.0, 2.0, 1.0, 2.0, 3.0],
        ));
        let features = DataFrame::new(6, vec![konst, med]).unwrap();
        let mut sp = SelectPercentile::new().percentile(100.0);

        sp.fit(features, make_target()).unwrap();

        // A constant feature has zero F and always sinks below a feature with
        // a positive score, whatever the input column order was.
        assert_eq!(sp.scores().unwrap()[0].0, "med");
        assert_eq!(sp.scores().unwrap()[1], ("const".to_string(), 0.0));
    }

    #[test]
    fn test_default_is_ten_percent() {
        let sp = SelectPercentile::default();

        assert_eq!(sp.percentile, 10.0);
    }

    #[test]
    fn test_percentile_out_of_range_is_rejected() {
        for p in [0.0f32, -1.0, 101.0, f32::NAN] {
            let mut sp = SelectPercentile::new().percentile(p);

            let error = sp.fit(make_features(), make_target()).unwrap_err();

            assert!(matches!(error, Error::InvalidInput(_)), "p = {p}: {error}");
            assert!(
                error.to_string().contains(&format!("got {p}")),
                "p = {p}: {error}"
            );
        }
    }

    #[test]
    fn test_zero_column_features_are_rejected() {
        let mut sp = SelectPercentile::new();

        let error = sp.fit(DataFrame::empty(), make_target()).unwrap_err();

        assert!(
            error.to_string().contains("0 columns"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_zero_row_features_are_rejected() {
        let mut sp = SelectPercentile::new();

        let error = sp
            .fit(make_features().slice(0, 0), make_target())
            .unwrap_err();

        assert!(
            error.to_string().contains("0 rows"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_empty_target_is_rejected() {
        let mut sp = SelectPercentile::new();

        let error = sp.fit(make_features(), DataFrame::empty()).unwrap_err();

        assert!(
            error.to_string().contains("got 0 columns"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_multi_column_target_is_rejected() {
        let y = DataFrame::new(
            6,
            vec![
                Column::from(Series::new("a".into(), &[0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0])),
                Column::from(Series::new("b".into(), &[0.0f64; 6])),
            ],
        )
        .unwrap();
        let mut sp = SelectPercentile::new();

        let error = sp.fit(make_features(), y).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("target must have exactly 1 column but got 2"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_failed_refit_resets_state() {
        let features = make_features();
        let mut sp = SelectPercentile::new().percentile(50.0);
        sp.fit(features.clone(), make_target()).unwrap();
        assert!(sp.scores().is_some());

        sp.percentile = 0.0;
        assert!(sp.fit(features.clone(), make_target()).is_err());

        assert!(sp.scores().is_none());
        assert!(matches!(
            sp.transform(features).unwrap_err(),
            Error::NotFitted(_)
        ));
    }

    #[test]
    fn test_f_regression_score_func_builder() {
        use crate::feature_selection::select_kbest::FRegression;

        let linear = Column::from(Series::new(
            "linear".into(),
            &[0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        ));
        let noise = Column::from(Series::new(
            "noise".into(),
            &[5.0f64, 0.0, 5.0, 0.0, 5.0, 0.0],
        ));
        let features = DataFrame::new(6, vec![noise, linear]).unwrap();
        let target = Column::from(Series::new(
            "target".into(),
            &[0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        ));
        let mut sp = SelectPercentile::new()
            .percentile(50.0)
            .score_func(Box::new(FRegression));

        sp.fit(features.clone(), DataFrame::new(6, vec![target]).unwrap())
            .unwrap();

        assert_eq!(sp.scores().unwrap()[0].0, "linear");
        assert_eq!(
            sp.transform(features).unwrap().get_column_names(),
            &["linear"]
        );
    }
}
