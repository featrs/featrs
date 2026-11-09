//! Expanding window aggregations.
//!
//! [`ExpandingAggregator`] computes mean, sum, min, max, and standard
//! deviation over a growing window: at each row `i` the statistic is
//! computed over all rows `0..=i`. Analogous to `df.expanding()` in pandas.

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_fitted, with_generated_names};
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Expanding window function to apply.
#[derive(Clone, Copy)]
pub enum ExpandingFn {
    /// Expanding mean over all rows seen so far.
    Mean,
    /// Expanding sum over all rows seen so far.
    Sum,
    /// Expanding minimum over all rows seen so far.
    Min,
    /// Expanding maximum over all rows seen so far.
    Max,
    /// Expanding (population) standard deviation over all rows seen so far.
    Std,
}

/// Compute expanding window statistics.
///
/// At each row `i`, the configured statistic is computed over the growing
/// window `0..=i`, so the window only ever gains observations. The
/// standard deviation is the population standard deviation (consistent with
/// [`RollingAggregator`](crate::time_series::rolling::RollingAggregator)).
///
/// Null values do not contribute to the accumulator, but every input row
/// keeps its position in the output: a row with a null value produces
/// `null`, and a row with a value produces the statistic over all non-null
/// values seen so far. No value is produced until at least `min_periods`
/// non-null observations have been seen.
///
/// `NaN` values are treated as values, not as missing: they propagate into
/// the mean/sum/std accumulators (subsequent outputs stay `NaN`) and are
/// ignored by the min/max accumulators (`f64::min`/`f64::max`), matching the
/// behavior of [`RollingAggregator`](crate::time_series::rolling::RollingAggregator).
///
/// New `Float64` columns named `{column}_expanding_{function}` are appended.
/// If a column with a generated name already exists in the input, it is
/// replaced (polars `with_column` semantics); otherwise existing columns
/// pass through unchanged. Configured columns must be `Float64`; this is
/// validated at fit time.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::expanding::{ExpandingAggregator, ExpandingFn};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("value".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut e = ExpandingAggregator::new(&["value"], ExpandingFn::Mean);
/// e.fit(df.clone())?;
/// let expanded = e.transform(df)?;
/// assert_eq!(expanded.height(), 5);
/// assert_eq!(expanded.column("value_expanding_mean").unwrap().f64().unwrap().get(4), Some(3.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ExpandingAggregator {
    fitted: bool,
    columns: Vec<String>,
    function: ExpandingFn,
    min_periods: usize,
}

impl ExpandingAggregator {
    /// Create a new expanding aggregator for `columns`, applying `function`
    /// (e.g. [`ExpandingFn::Mean`]) over the growing window.
    pub fn new(columns: &[&str], function: ExpandingFn) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            function,
            min_periods: 1,
        }
    }

    /// Set the minimum number of non-null observations required before a
    /// non-null value is produced (default: `1`).
    pub fn min_periods(mut self, p: usize) -> Self {
        self.min_periods = p;
        self
    }

    fn expanding_series(&self, s: &Series) -> Result<Series> {
        let ca = s
            .f64()
            .map_err(|_| Error::InvalidInput("column must be f64".into()))?;
        let mut seen: usize = 0;
        let mut sum = 0.0_f64;
        let mut min: Option<f64> = None;
        let mut max: Option<f64> = None;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;

        let mut out: Vec<Option<f64>> = Vec::with_capacity(ca.len());
        for v in ca.iter() {
            if let Some(x) = v {
                seen += 1;
                match self.function {
                    ExpandingFn::Min => {
                        min = Some(min.map_or(x, |cur| cur.min(x)));
                    }
                    ExpandingFn::Max => {
                        max = Some(max.map_or(x, |cur| cur.max(x)));
                    }
                    ExpandingFn::Mean | ExpandingFn::Sum | ExpandingFn::Std => {
                        sum += x;
                        // Welford's online algorithm: stable running mean and M2.
                        let delta = x - mean;
                        mean += delta / seen as f64;
                        m2 += delta * (x - mean);
                    }
                }
            }

            out.push(match v {
                None => None,
                Some(_) if seen < self.min_periods => None,
                Some(_) => match self.function {
                    ExpandingFn::Mean => Some(mean),
                    ExpandingFn::Sum => Some(sum),
                    ExpandingFn::Min => min,
                    ExpandingFn::Max => max,
                    ExpandingFn::Std => Some((m2 / seen as f64).sqrt()),
                },
            });
        }

        let new_ca: ChunkedArray<Float64Type> = out.into_iter().collect();
        Ok(new_ca.into_series())
    }
}

impl Default for ExpandingAggregator {
    fn default() -> Self {
        Self::new(&[], ExpandingFn::Mean)
    }
}

impl Fit<DataFrame> for ExpandingAggregator {
    type Output = ();

    /// Validate the input and mark the aggregator fitted. A failed fit leaves it unfitted.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "ExpandingAggregator.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "ExpandingAggregator: at least one column is required.".into(),
            ));
        }
        if self.min_periods == 0 {
            return Err(Error::InvalidInput(format!(
                "ExpandingAggregator: min_periods must be >= 1, got {}",
                self.min_periods
            )));
        }
        for col in &self.columns {
            let c = x.column(col.as_str()).map_err(|_| {
                Error::InvalidInput(format!("ExpandingAggregator: column '{}' not found.", col))
            })?;
            if c.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "ExpandingAggregator: column '{}' has dtype {}; expected Float64.",
                    col,
                    c.dtype()
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for ExpandingAggregator {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("ExpandingAggregator".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            // Snapshot each source from the immutable input `x`, not the
            // growing output `out`: a generated name can collide with a
            // configured column (e.g. `"x"` and `"x_expanding_mean"`), and
            // reading from `out` would then aggregate the derived values.
            let s = x
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "ExpandingAggregator.transform: column '{}' not found. {}",
                        col, e
                    ))
                })?
                .as_materialized_series()
                .clone();
            let fn_name = match self.function {
                ExpandingFn::Mean => "mean",
                ExpandingFn::Sum => "sum",
                ExpandingFn::Min => "min",
                ExpandingFn::Max => "max",
                ExpandingFn::Std => "std",
            };
            let expanded = self.expanding_series(&s)?;
            let expanded_name = format!("{}_expanding_{}", col, fn_name);
            out.with_column(expanded.with_name(expanded_name.as_str().into()).into())
                .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(out)
    }
}

impl FeatureNamesOut for ExpandingAggregator {
    /// The input columns plus `{column}_expanding_{function}`.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        ensure_fitted(self.fitted, "ExpandingAggregator")?;
        let fn_name = match self.function {
            ExpandingFn::Mean => "mean",
            ExpandingFn::Sum => "sum",
            ExpandingFn::Min => "min",
            ExpandingFn::Max => "max",
            ExpandingFn::Std => "std",
        };
        let generated = self
            .columns
            .iter()
            .map(|col| format!("{col}_expanding_{fn_name}"));
        Ok(with_generated_names(input_columns, generated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_df(vals: &[f64]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), vals));
        DataFrame::new(vals.len(), vec![col]).unwrap()
    }

    #[test]
    fn test_expanding_mean() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        let expected = [1.0, 1.5, 2.0, 2.5, 3.0];
        for (i, want) in expected.iter().enumerate() {
            assert_relative_eq!(expanded.get(i).unwrap(), want, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_expanding_sum() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Sum);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_sum").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(1.0));
        assert_eq!(expanded.get(1), Some(3.0));
        assert_eq!(expanded.get(2), Some(6.0));
    }

    #[test]
    fn test_expanding_min() {
        let df = make_df(&[3.0, 1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Min);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_min").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(3.0));
        assert_eq!(expanded.get(1), Some(1.0));
        assert_eq!(expanded.get(2), Some(1.0));
    }

    #[test]
    fn test_expanding_max() {
        let df = make_df(&[3.0, 1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Max);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_max").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(3.0));
        assert_eq!(expanded.get(1), Some(3.0));
        assert_eq!(expanded.get(2), Some(3.0));
    }

    #[test]
    fn test_expanding_std() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Std);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_std").unwrap().f64().unwrap();
        assert_relative_eq!(expanded.get(0).unwrap(), 0.0, epsilon = 1e-9);
        assert_relative_eq!(expanded.get(1).unwrap(), 0.5, epsilon = 1e-9);
        // Population variance of [1, 2, 3] is 2/3.
        assert_relative_eq!(
            expanded.get(2).unwrap(),
            (2.0_f64 / 3.0).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_min_periods_builder() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean).min_periods(2);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert!(expanded.get(0).is_none());
        assert_relative_eq!(expanded.get(1).unwrap(), 1.5, epsilon = 1e-9);
        assert_relative_eq!(expanded.get(2).unwrap(), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_nulls_skipped_in_accumulator_and_preserved_in_output() {
        let col = Column::from(Series::new(
            "x".into(),
            &[Some(1.0_f64), None, Some(3.0), None, Some(5.0)],
        ));
        let df = DataFrame::new(5, vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(1.0));
        assert!(expanded.get(1).is_none());
        assert_eq!(expanded.get(2), Some(2.0));
        assert!(expanded.get(3).is_none());
        assert_eq!(expanded.get(4), Some(3.0));
        assert_eq!(result.height(), 5);
    }

    #[test]
    fn test_input_columns_pass_through() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0]));
        let b = Column::from(Series::new("b".into(), &[10.0_f64, 20.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();
        let mut e = ExpandingAggregator::new(&["b"], ExpandingFn::Sum);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        assert_eq!(result.width(), 3);
        assert_eq!(
            result.column("b").unwrap().f64().unwrap().get(0),
            Some(10.0)
        );
        assert_eq!(
            result
                .column("b_expanding_sum")
                .unwrap()
                .f64()
                .unwrap()
                .get(1),
            Some(30.0)
        );
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let df = make_df(&[1.0, 2.0]);
        let e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        let err = e.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_fit_empty_columns_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&[], ExpandingFn::Mean);
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fit_empty_input_errors() {
        let col = Column::from(Series::new("x".into(), Vec::<f64>::new()));
        let df = DataFrame::new(0, vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fit_missing_column_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["nope"], ExpandingFn::Mean);
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let good = make_df(&[1.0, 2.0, 3.0]);
        let bad = DataFrame::new(
            3,
            vec![Column::from(Series::new(
                "missing".into(),
                &[1.0_f64, 2.0, 3.0],
            ))],
        )
        .unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);

        e.fit(good.clone()).unwrap();
        assert!(e.fit(bad).is_err());
        assert!(matches!(e.transform(good), Err(Error::NotFitted(_))));

        let recovery = make_df(&[4.0, 6.0]);
        e.fit(recovery.clone()).unwrap();
        let result = e.transform(recovery).unwrap();
        assert_eq!(
            result
                .column("x_expanding_mean")
                .unwrap()
                .f64()
                .unwrap()
                .get(1),
            Some(5.0)
        );
    }

    #[test]
    fn test_invalid_dtype_refit_resets_fitted_state() {
        let good = make_df(&[1.0, 2.0]);
        let bad =
            DataFrame::new(2, vec![Column::from(Series::new("x".into(), &["a", "b"]))]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);

        e.fit(good.clone()).unwrap();
        assert!(matches!(e.fit(bad), Err(Error::InvalidInput(_))));
        assert!(matches!(e.transform(good), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_empty_refit_resets_fitted_state() {
        let good = make_df(&[1.0, 2.0, 3.0]);
        let empty = make_df(&[]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);

        e.fit(good.clone()).unwrap();
        assert!(matches!(e.fit(empty), Err(Error::InvalidInput(_))));
        assert!(matches!(e.transform(good), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_min_periods_zero_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean).min_periods(0);
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_zero_min_periods_refit_resets_fitted_state() {
        let good = make_df(&[1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        e.fit(good.clone()).unwrap();

        let mut e = e.min_periods(0);
        assert!(matches!(e.fit(good.clone()), Err(Error::InvalidInput(_))));
        assert!(matches!(e.transform(good), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_fit_non_f64_column_errors() {
        let col = Column::from(Series::new("x".into(), &["a", "b"]));
        let df = DataFrame::new(2, vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_generated_name_collision_uses_original_input() {
        // Configured column names can coincide with generated names: with
        // `["x", "x_expanding_mean"]`, processing "x" first appends
        // `x_expanding_mean` (replacing the input column of that name). The
        // second configured column must then aggregate the ORIGINAL
        // `x_expanding_mean` input values, not the values just derived from
        // "x" — reading sources from the growing output frame would yield
        // [1, 1.25, 1.5] here instead of [10, 15, 20].
        let a = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new(
            "x_expanding_mean".into(),
            &[10.0_f64, 20.0, 30.0],
        ));
        let df = DataFrame::new(3, vec![a, b]).unwrap();
        let mut e = ExpandingAggregator::new(&["x", "x_expanding_mean"], ExpandingFn::Mean);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let derived = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert_relative_eq!(derived.get(0).unwrap(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(derived.get(1).unwrap(), 1.5, epsilon = 1e-9);
        assert_relative_eq!(derived.get(2).unwrap(), 2.0, epsilon = 1e-9);

        let from_original = result
            .column("x_expanding_mean_expanding_mean")
            .unwrap()
            .f64()
            .unwrap();
        assert_relative_eq!(from_original.get(0).unwrap(), 10.0, epsilon = 1e-9);
        assert_relative_eq!(from_original.get(1).unwrap(), 15.0, epsilon = 1e-9);
        assert_relative_eq!(from_original.get(2).unwrap(), 20.0, epsilon = 1e-9);
    }

    fn run_on_nulled(vals: &[Option<f64>], function: ExpandingFn) -> DataFrame {
        let col = Column::from(Series::new("x".into(), vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], function);
        e.fit(df.clone()).unwrap();
        e.transform(df).unwrap()
    }

    #[test]
    fn test_min_with_nulls() {
        let result = run_on_nulled(
            &[Some(3.0), None, Some(1.0), None, Some(2.0)],
            ExpandingFn::Min,
        );
        let expanded = result.column("x_expanding_min").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(3.0));
        assert!(expanded.get(1).is_none());
        assert_eq!(expanded.get(2), Some(1.0));
        assert!(expanded.get(3).is_none());
        assert_eq!(expanded.get(4), Some(1.0));
    }

    #[test]
    fn test_max_with_nulls() {
        let result = run_on_nulled(
            &[Some(3.0), None, Some(1.0), None, Some(2.0)],
            ExpandingFn::Max,
        );
        let expanded = result.column("x_expanding_max").unwrap().f64().unwrap();
        assert_eq!(expanded.get(0), Some(3.0));
        assert!(expanded.get(1).is_none());
        assert_eq!(expanded.get(2), Some(3.0));
        assert!(expanded.get(3).is_none());
        assert_eq!(expanded.get(4), Some(3.0));
    }

    #[test]
    fn test_std_with_nulls() {
        let result = run_on_nulled(
            &[Some(3.0), None, Some(1.0), None, Some(2.0)],
            ExpandingFn::Std,
        );
        let expanded = result.column("x_expanding_std").unwrap().f64().unwrap();
        assert_relative_eq!(expanded.get(0).unwrap(), 0.0, epsilon = 1e-9);
        assert!(expanded.get(1).is_none());
        assert_relative_eq!(expanded.get(2).unwrap(), 1.0, epsilon = 1e-9);
        assert!(expanded.get(3).is_none());
        // Population std of [3, 1, 2] = sqrt(2/3).
        assert_relative_eq!(
            expanded.get(4).unwrap(),
            (2.0_f64 / 3.0).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_null_rows_stay_none_with_min_periods_above_one() {
        let col = Column::from(Series::new("x".into(), &[Some(1.0_f64), None, Some(3.0)]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean).min_periods(2);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert!(expanded.get(0).is_none());
        assert!(expanded.get(1).is_none());
        assert_relative_eq!(expanded.get(2).unwrap(), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_min_periods_beyond_length_all_none() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean).min_periods(5);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert!(expanded.get(0).is_none());
        assert!(expanded.get(1).is_none());
    }

    #[test]
    fn test_nan_propagates_in_mean() {
        let col = Column::from(Series::new(
            "x".into(),
            &[Some(1.0_f64), Some(f64::NAN), Some(3.0)],
        ));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut e = ExpandingAggregator::new(&["x"], ExpandingFn::Mean);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let expanded = result.column("x_expanding_mean").unwrap().f64().unwrap();
        assert_relative_eq!(expanded.get(0).unwrap(), 1.0, epsilon = 1e-9);
        assert!(expanded.get(1).unwrap().is_nan());
        assert!(expanded.get(2).unwrap().is_nan());
    }
}
