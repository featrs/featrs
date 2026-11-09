//! Correlation-based feature selection.
//!
//! Provides [`CorrelationThreshold`], which removes features that are highly
//! correlated with other features. For each group of mutually correlated
//! columns above a configurable threshold, only the first (in input order) is
//! kept, reducing multicollinearity and redundant features.

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::subset_names;
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::require_f64_columns;
use polars::prelude::*;
use std::cmp::Ordering;

/// Method used to measure pairwise feature correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    /// Pearson product-moment correlation coefficient (linear correlation).
    Pearson,
    /// Spearman's rank correlation (rank-transform each column, then Pearson).
    Spearman,
}

/// Remove features that are highly correlated with other features.
///
/// Starting from the first `Float64` column, each feature that has an absolute
/// correlation greater than or equal to `threshold` with any already-selected
/// column is dropped. This keeps a single representative from each group of
/// mutually correlated columns, in input order.
///
/// Only `Float64` columns are considered; columns of other dtypes are silently
/// dropped from the output.
///
/// Correlations are computed over rows where both values are present and
/// finite; null and non-finite (`NaN`/`±Inf`) values are skipped pairwise. When
/// a correlation is undefined (e.g. a constant column produces a zero standard
/// deviation), it is treated as `0.0`, so the column is retained whenever the
/// threshold is greater than `0.0`; at a threshold of exactly `0.0` such a
/// column is dropped like any other (`0.0 >= 0.0`).
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::CorrelationThreshold;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
/// let b = Column::from(Series::new("b".into(), &[2.0_f64, 4.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut ct = CorrelationThreshold::new(); // threshold 0.9, Pearson
/// ct.fit(df.clone())?;
/// let kept = ct.transform(df)?;
/// assert_eq!(kept.width(), 1);
/// assert_eq!(kept.get_column_names()[0].as_str(), "a");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CorrelationThreshold {
    fitted: bool,
    threshold: f64,
    method: CorrelationMethod,
    selected_columns: Option<Vec<String>>,
}

impl CorrelationThreshold {
    /// Create a new `CorrelationThreshold` transformer.
    ///
    /// Defaults to a `threshold` of `0.9` and the [`Pearson`](CorrelationMethod::Pearson)
    /// correlation method. Features with an absolute correlation greater than
    /// or equal to `threshold` with an already-selected feature are dropped.
    pub fn new() -> Self {
        Self {
            fitted: false,
            threshold: 0.9,
            method: CorrelationMethod::Pearson,
            selected_columns: None,
        }
    }

    /// Set the absolute-correlation threshold.
    ///
    /// A valid threshold is in the inclusive range `[0.0, 1.0]`. A threshold
    /// of `1.0` removes only perfectly correlated columns; `0.0` removes all
    /// but the first column. Values outside this range are rejected at fit time.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the correlation method used to measure feature similarity.
    pub fn method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }
}

impl Default for CorrelationThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for CorrelationThreshold {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        self.selected_columns = None;

        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "CorrelationThreshold.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "CorrelationThreshold.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(Error::InvalidInput(format!(
                "CorrelationThreshold: threshold must be in [0.0, 1.0], got {}.",
                self.threshold
            )));
        }

        let names = require_f64_columns(&x, "CorrelationThreshold")?;

        // Keep the raw finite values for every Float64 column. Correlation is
        // computed pairwise (method-dispatch inside `correlation`); for
        // Spearman, ranks are derived from the aligned complete pairs, so they
        // are not affected by rows missing from one of the two columns.
        let mut work: Vec<(String, Vec<Option<f64>>)> = Vec::new();
        for name in &names {
            let s = x.column(name).map_err(|e| {
                Error::InvalidInput(format!(
                    "CorrelationThreshold.fit: column '{name}' not found. {e}"
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "CorrelationThreshold.fit: column '{name}' is not Float64. {e}"
                ))
            })?;
            let values: Vec<Option<f64>> =
                ca.iter().map(|opt| opt.filter(|v| v.is_finite())).collect();
            work.push((name.clone(), values));
        }

        // Greedily select: keep a column unless it is correlated (|r| >= threshold)
        // with an already-selected column. Earlier columns win ties.
        let mut selected: Vec<String> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for i in 0..work.len() {
            let (name, values) = &work[i];
            let mut drop = false;
            for &j in &survivors {
                let (_, other) = &work[j];
                if correlation(values, other, self.method).abs() >= self.threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                survivors.push(i);
                selected.push(name.clone());
            }
        }

        if selected.is_empty() {
            return Err(Error::Computation(
                "CorrelationThreshold: no columns survived correlation filtering.".into(),
            ));
        }

        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CorrelationThreshold {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "CorrelationThreshold has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "CorrelationThreshold has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        if cols.is_empty() {
            return Err(Error::Computation(
                "CorrelationThreshold: no columns were selected.".into(),
            ));
        }
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Compute the correlation between two columns following the selected method.
///
/// Rows where either value is `None` are excluded first (pairwise complete
/// observations). Returns `0.0` when fewer than two complete pairs exist or
/// when the correlation is undefined over the complete pairs (zero variance),
/// so an undefined correlation never falsely drops a feature.
///
/// For [`Spearman`](CorrelationMethod::Spearman), ranks are computed from the
/// aligned complete pairs of *both* columns, so a value missing from one
/// column does not affect the ranking of the other.
fn correlation(a: &[Option<f64>], b: &[Option<f64>], method: CorrelationMethod) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Some((*x, *y)),
            _ => None,
        })
        .collect();
    if pairs.len() < 2 {
        return 0.0;
    }
    match method {
        CorrelationMethod::Pearson => {
            let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
            let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
            pearson_values(&xs, &ys)
        }
        CorrelationMethod::Spearman => {
            let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
            let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
            pearson_values(&ranks_of(&xs), &ranks_of(&ys))
        }
    }
}

/// Compute the Pearson correlation between two aligned, complete value vectors.
///
/// Both inputs must be the same length and contain only finite values.
///
/// The computation is overflow-safe: each raw vector is first scaled by its
/// maximum absolute value (keeping every element in `[-1, 1]`), and its mean
/// is then computed from the scaled values. Pearson correlation is invariant
/// to independently and positively scaling each variable, so this does not
/// change the result, but it keeps all intermediate sums bounded and avoids
/// centering overflow (e.g. `1e308 - (-1e308)` would be `inf`) even for
/// `1e308`-scale inputs, where the naive formula would yield `NaN` and wrongly
/// retain a perfectly correlated column. Returns `0.0` when either vector is
/// constant (zero variance).
fn pearson_values(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let scale_a = a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let scale_b = b.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    if scale_a == 0.0 || scale_b == 0.0 {
        return 0.0;
    }
    let mean_a = a.iter().map(|x| x / scale_a).sum::<f64>() / n as f64;
    let mean_b = b.iter().map(|y| y / scale_b).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x / scale_a;
        let y = y / scale_b;
        cov += (x - mean_a) * (y - mean_b);
        var_a += (x - mean_a).powi(2);
        var_b += (y - mean_b).powi(2);
    }
    let denom = var_a.sqrt() * var_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    let r = cov / denom;
    // Clamp to the valid correlation range and snap near-perfect values to
    // exactly ±1. Floating-point noise can otherwise leave a perfectly
    // correlated pair at 0.9999999999999998, which would wrongly survive a
    // `threshold = 1.0` filter (the issue contract requires it to be removed).
    let r = r.clamp(-1.0, 1.0);
    if (1.0 - r.abs()) < 1e-12 {
        r.signum()
    } else {
        r
    }
}

/// Rank-transform a value vector using average ranks for ties.
///
/// Ranks are `1`-based; tied values all receive the average of the ranks they
/// span. This is the standard tie handling used by Spearman's rank correlation.
fn ranks_of(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out: Vec<f64> = vec![0.0; n];
    let mut present: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    present.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut i = 0;
    while i < present.len() {
        let mut j = i;
        while j + 1 < present.len() && present[j + 1].1.total_cmp(&present[i].1) == Ordering::Equal
        {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            out[present[k].0] = avg;
        }
        i = j + 1;
    }
    out
}

impl FeatureNamesOut for CorrelationThreshold {
    /// The learned column subset, in selection order.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        subset_names(
            self.fitted,
            "CorrelationThreshold",
            self.selected_columns.as_deref(),
            input_columns,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(cols: Vec<Column>) -> DataFrame {
        let h = cols[0].len();
        DataFrame::new(h, cols).unwrap()
    }

    fn f64_col(name: &str, vals: &[f64]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    #[test]
    fn test_perfect_positive_correlated_removed() {
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0]),
            f64_col("b", &[2.0, 4.0, 6.0]),
        ]);
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_perfect_negative_correlated_removed() {
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0]),
            f64_col("b", &[3.0, 2.0, 1.0]),
        ]);
        let mut ct = CorrelationThreshold::new(); // threshold 0.9
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_uncorrelated_columns_kept() {
        // Pearson correlation ≈ 0.452 < 0.9 → both columns survive.
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            f64_col("b", &[6.0, 1.0, 5.0, 2.0, 4.0, 3.0, 8.0, 7.0]),
        ]);
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 2);
    }

    #[test]
    fn test_higher_threshold_removes_fewer() {
        // Perfectly-correlated pair: corr = 1.0.
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0]),
            f64_col("b", &[2.0, 4.0, 6.0]),
        ]);
        let mut loose = CorrelationThreshold::new().threshold(0.999);
        loose.fit(df.clone()).unwrap();
        // At 0.999, corr 1.0 >= 0.999 → removed.
        assert_eq!(loose.transform(df.clone()).unwrap().width(), 1);

        // At 1.0, corr 1.0 >= 1.0 still removed (perfect correlation).
        let mut exact = CorrelationThreshold::new().threshold(1.0);
        exact.fit(df.clone()).unwrap();
        assert_eq!(exact.transform(df.clone()).unwrap().width(), 1);
    }

    #[test]
    fn test_threshold_one_snaps_noisy_perfect_correlation() {
        // [1,2,3] vs [2,4,6] computes a Pearson correlation of
        // 0.9999999999999998 (gap ~2.2e-16) due to floating-point noise, not
        // exactly 1.0. The ±1 snap must still drop the redundant column at
        // `threshold = 1.0`, per the issue contract.
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0]),
            f64_col("b", &[2.0, 4.0, 6.0]),
        ]);
        let mut ct = CorrelationThreshold::new().threshold(1.0);
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_large_finite_values_no_overflow() {
        // Identical 1e308-scale columns are perfectly correlated. The naive
        // covariance formula overflows to inf/NaN here, and a NaN correlation
        // (NaN >= threshold is false) would wrongly retain the duplicate. The
        // overflow-safe computation must still drop it.
        let df = frame(vec![
            f64_col("a", &[1.0e308, 1.5e308]),
            f64_col("b", &[1.0e308, 1.5e308]),
        ]);
        let mut ct = CorrelationThreshold::new(); // threshold 0.9, Pearson
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_opposite_sign_large_values_no_overflow() {
        // Centering on a[0] would compute 1e308 - (-1e308) = inf for these
        // opposite-sign values, but scaling each raw vector by its max absolute
        // value first keeps everything in [-1, 1]. Identical columns are
        // perfectly correlated, so the duplicate must still be dropped.
        let df = frame(vec![
            f64_col("a", &[-1.0e308, 1.0e308]),
            f64_col("b", &[-1.0e308, 1.0e308]),
        ]);
        let mut ct = CorrelationThreshold::new(); // threshold 0.9, Pearson
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_spearman_catches_monotonic_where_pearson_does_not() {
        // x = 1..=10, y = [1..9, 1000]. Linear correlation is low (≈0.529),
        // but the relationship is perfectly monotonically increasing, so
        // Spearman = 1.0. Pearson therefore keeps both features; Spearman
        // drops the second one at a 0.9 threshold.
        let df = frame(vec![
            f64_col("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            f64_col("y", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]),
        ]);

        let mut pear = CorrelationThreshold::new().threshold(0.9);
        pear.fit(df.clone()).unwrap();
        assert_eq!(pear.transform(df.clone()).unwrap().width(), 2);

        let mut spear = CorrelationThreshold::new()
            .threshold(0.9)
            .method(CorrelationMethod::Spearman);
        spear.fit(df.clone()).unwrap();
        assert_eq!(spear.transform(df).unwrap().width(), 1);
    }

    #[test]
    fn test_spearman_handles_ties() {
        // Ties in x ([1,1,2,2,3,3]) and y ([1,1,4,4,9,9]) rank-perfectly, so
        // Spearman = 1.0 → removed at threshold 0.9.
        let df = frame(vec![
            f64_col("a", &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
            f64_col("b", &[1.0, 1.0, 4.0, 4.0, 9.0, 9.0]),
        ]);
        let mut ct = CorrelationThreshold::new()
            .threshold(0.9)
            .method(CorrelationMethod::Spearman);
        ct.fit(df.clone()).unwrap();
        assert_eq!(ct.transform(df).unwrap().width(), 1);
    }

    #[test]
    fn test_spearman_ranks_pairwise_complete_rows() {
        // `a` is present in all 10 rows; `b` is present only at rows 1,2,3,9,10
        // (0-indexed 0,1,2,8,9). Over the five complete pairs both columns are
        // monotonically ordered, so pairwise Spearman = 1.0 and `b` is dropped
        // at threshold 0.9. Ranking each full column first (the previous buggy
        // approach) would correlate a's ranks [1,2,3,9,10] with b's [1,2,3,4,5]
        // (≈0.756) and wrongly keep `b`.
        let a = Column::from(Series::new(
            "a".into(),
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ));
        let b = Column::from(Series::new(
            "b".into(),
            &[
                Some(1.0_f64),
                Some(2.0),
                Some(3.0),
                None,
                None,
                None,
                None,
                None,
                Some(9.0),
                Some(10.0),
            ],
        ));
        let df = DataFrame::new(10, vec![a, b]).unwrap();
        let mut ct = CorrelationThreshold::new()
            .threshold(0.9)
            .method(CorrelationMethod::Spearman);
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_threshold_zero_keeps_only_first() {
        let df = frame(vec![
            f64_col("a", &[1.0, 2.0, 3.0]),
            f64_col("b", &[2.0, 4.0, 6.0]),
            f64_col("c", &[3.0, 6.0, 9.0]),
        ]);
        let mut ct = CorrelationThreshold::new().threshold(0.0);
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_nulls_are_skipped_pairwise() {
        // Complete rows (1,2),(2,4),(3,6) are perfectly correlated → 'b' removed.
        let a = Column::from(Series::new(
            "a".into(),
            &[Some(1.0_f64), None, Some(2.0), Some(3.0)],
        ));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 2.0, 4.0, 6.0]));
        let df = DataFrame::new(4, vec![a, b]).unwrap();
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
    }

    #[test]
    fn test_single_column_all_kept() {
        let df = frame(vec![f64_col("only", &[1.0, 2.0, 3.0])]);
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "only");
    }

    #[test]
    fn test_non_f64_columns_dropped() {
        let a = Column::from(Series::new("f".into(), &[1.0_f64, 2.0, 3.0]));
        let i = Column::from(Series::new("i".into(), &[1_i64, 2, 3]));
        let df = DataFrame::new(3, vec![a, i]).unwrap();
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "f");
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let df = frame(vec![f64_col("a", &[1.0, 2.0, 3.0])]);
        let ct = CorrelationThreshold::new();
        let err = ct.transform(df).unwrap_err();
        assert!(err.to_string().contains("not been fitted"));
    }

    #[test]
    fn test_empty_input_errors() {
        let mut ct = CorrelationThreshold::new();
        let err = ct.fit(DataFrame::new(0, Vec::<Column>::new()).unwrap());
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("0 columns"));
    }

    #[test]
    fn test_zero_rows_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0]));
        let a_empty = a.slice(0, 0);
        let df = DataFrame::new(0, vec![a_empty]).unwrap();
        let mut ct = CorrelationThreshold::new();
        let err = ct.fit(df);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("0 rows"));
    }

    #[test]
    fn test_out_of_range_threshold_rejected() {
        let df = frame(vec![f64_col("a", &[1.0, 2.0, 3.0])]);
        let mut ct = CorrelationThreshold::new().threshold(1.5);
        let err = ct.fit(df).unwrap_err();
        assert!(err.to_string().contains("[0.0, 1.0]"));
    }

    #[test]
    fn test_constant_column_kept() {
        // Constant columns yield undefined (0/0) correlation → treated as 0,
        // so both are kept.
        let df = frame(vec![
            f64_col("c1", &[2.0, 2.0, 2.0]),
            f64_col("c2", &[5.0, 5.0, 5.0]),
        ]);
        let mut ct = CorrelationThreshold::new();
        ct.fit(df.clone()).unwrap();
        let out = ct.transform(df).unwrap();
        assert_eq!(out.width(), 2);
    }
}
