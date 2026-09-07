//! Exponentially weighted moving average (EWMA) statistics.
//!
//! [`ExponentiallyWeightedMovingAverage`] computes a weighted moving average
//! where the weights decay exponentially: the most recent observation carries
//! the highest weight. The smoothing-factor options (`alpha`, `span`, `com`,
//! `half_life`) and the `adjust`, `min_periods`, and `ignore_na` flags follow
//! the pandas `DataFrame.ewm` conventions.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// How the EWMA smoothing factor is configured.
///
/// Every variant is converted to an `alpha` in `(0, 1]` at fit time. For the
/// [`EWMAStatistic::Mean`] statistic an `alpha` of `1.0` disables smoothing
/// (the output equals the input).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EWMASmoothing {
    /// Direct smoothing factor in `(0, 1]`.
    Alpha(f64),
    /// `alpha = 2 / (span + 1)`, with `span >= 1`.
    Span(f64),
    /// Center of mass: `alpha = 1 / (1 + com)`, with `com >= 0`.
    Com(f64),
    /// `alpha = 1 - exp(ln(0.5) / half_life)`, with `half_life > 0`.
    HalfLife(f64),
}

/// The statistic to compute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EWMAStatistic {
    /// Exponentially weighted mean.
    Mean,
    /// Exponentially weighted standard deviation.
    Std,
    /// Exponentially weighted variance.
    Var,
}

/// Compute exponentially weighted moving average statistics.
///
/// At each row `i` the weighted statistic is computed over the series seen so
/// far, with exponentially decaying weights. The most recent observation has
/// the highest weight.
///
/// The smoothing factor is configured through [`EWMASmoothing`]. The `adjust`
/// flag controls the normalization of the [`EWMAStatistic::Mean`] statistic:
/// with `adjust = true` (the default) the mean is a normalized weighted
/// average `(Σ (1-α)^j * x[i-j]) / (Σ (1-α)^j)`; with `adjust = false` it is
/// the recursive form `y[i] = α * x[i] + (1-α) * y[i-1]`, initialised to
/// `x[0]`.
///
/// For [`EWMAStatistic::Var`] and [`EWMAStatistic::Std`], the exponentially
/// weighted variance recurrence
/// `var[i] = (1-α) * (var[i-1] + α * (x[i] - mean[i-1])^2)` is used, with the
/// running mean following the unadjusted recursion. This is the biased
/// estimator (pandas `bias=True` form); pandas defaults to the unbiased
/// `bias=False` estimator, so `Var`/`Std` diverge from pandas' default.
/// Variance requires at least two observations, so the output is `null` until
/// two non-null values have been seen. `adjust` does not affect the
/// variance/standard deviation.
///
/// Null handling follows the `ignore_na` option: when `false` (default) a
/// `null` in the input propagates to `null` in the output from that row on
/// (the recursion cannot continue); when `true`, `null` rows produce `null`
/// output but do not advance the recursive state, so later non-null rows
/// continue correctly. `NaN` values are treated as values, not as missing, and
/// propagate through the recursion.
///
/// `min_periods` (default `0`) is the minimum number of non-null observations
/// required before a non-null value is produced. Rows before that point get
/// `null`.
///
/// New `Float64` columns named `{column}_ewm_{statistic}_{alpha}` are appended
/// (e.g. `value_ewm_mean_0.5`). A generated name that collides with an existing
/// input column or with another generated name is rejected at fit time with
/// [`Error::InvalidInput`]; existing columns otherwise pass through unchanged.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::ewma::{EWMASmoothing, ExponentiallyWeightedMovingAverage};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("value".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut e = ExponentiallyWeightedMovingAverage::new(&["value"], EWMASmoothing::Alpha(0.5));
/// e.fit(df.clone())?;
/// let out = e.transform(df)?;
/// assert_eq!(out.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ExponentiallyWeightedMovingAverage {
    fitted: bool,
    columns: Vec<String>,
    smoothing: EWMASmoothing,
    statistic: EWMAStatistic,
    adjust: bool,
    min_periods: usize,
    ignore_na: bool,
    alpha: f64,
}

impl ExponentiallyWeightedMovingAverage {
    /// Create a new EWMA transformer for `columns` using `smoothing`.
    ///
    /// Defaults to the [`EWMAStatistic::Mean`] statistic, `adjust = true`,
    /// `min_periods = 0`, and `ignore_na = false`.
    pub fn new(columns: &[&str], smoothing: EWMASmoothing) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            smoothing,
            statistic: EWMAStatistic::Mean,
            adjust: true,
            min_periods: 0,
            ignore_na: false,
            alpha: 0.0,
        }
    }

    /// Set the statistic to compute (default: [`EWMAStatistic::Mean`]).
    pub fn statistic(mut self, s: EWMAStatistic) -> Self {
        self.statistic = s;
        self
    }

    /// Set whether the mean uses adjusted (normalized) weights.
    ///
    /// Defaults to `true` (pandas default).
    pub fn adjust(mut self, b: bool) -> Self {
        self.adjust = b;
        self
    }

    /// Set the minimum number of non-null observations required before a
    /// non-null value is produced (default: `0`).
    pub fn min_periods(mut self, p: usize) -> Self {
        self.min_periods = p;
        self
    }

    /// Set whether `null` values are skipped rather than propagated.
    ///
    /// Defaults to `false` (a `null` propagates to `null` output from that row
    /// on).
    pub fn ignore_na(mut self, b: bool) -> Self {
        self.ignore_na = b;
        self
    }

    /// Resolve the configured [`EWMASmoothing`] into an `alpha` in `(0, 1]`,
    /// validating the parameter range.
    fn resolve_alpha(&self) -> Result<f64> {
        let alpha = match self.smoothing {
            EWMASmoothing::Alpha(a) => {
                if a <= 0.0 || a > 1.0 {
                    return Err(Error::InvalidInput(format!(
                        "EWMA: alpha must be in (0, 1], got {}",
                        a
                    )));
                }
                a
            }
            EWMASmoothing::Span(s) => {
                if s < 1.0 {
                    return Err(Error::InvalidInput(format!(
                        "EWMA: span must be >= 1, got {}",
                        s
                    )));
                }
                2.0 / (s + 1.0)
            }
            EWMASmoothing::Com(c) => {
                if c < 0.0 {
                    return Err(Error::InvalidInput(format!(
                        "EWMA: com must be >= 0, got {}",
                        c
                    )));
                }
                1.0 / (1.0 + c)
            }
            EWMASmoothing::HalfLife(h) => {
                if h <= 0.0 {
                    return Err(Error::InvalidInput(format!(
                        "EWMA: half_life must be > 0, got {}",
                        h
                    )));
                }
                1.0 - (0.5_f64).powf(1.0 / h)
            }
        };
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(Error::InvalidInput(format!(
                "EWMA: smoothing yields non-finite alpha {}",
                alpha
            )));
        }
        Ok(alpha)
    }

    /// The generated output column name for `col`.
    fn output_name(&self, col: &str) -> String {
        let stat = match self.statistic {
            EWMAStatistic::Mean => "mean",
            EWMAStatistic::Std => "std",
            EWMAStatistic::Var => "var",
        };
        format!("{col}_ewm_{stat}_{}", self.alpha)
    }

    /// Compute the EWMA statistic for one column of values.
    fn compute_column(&self, vals: &[Option<f64>]) -> Vec<Option<f64>> {
        let alpha = self.alpha;
        let min_obs = match self.statistic {
            EWMAStatistic::Std | EWMAStatistic::Var => self.min_periods.max(2),
            EWMAStatistic::Mean => self.min_periods,
        };

        let mut out = Vec::with_capacity(vals.len());
        let mut broken = false;
        let mut seen = 0usize;

        // Unadjusted mean state.
        let mut prev: Option<f64> = None;
        // Adjusted mean state (running weighted sum and normalizer).
        let mut wsum = 0.0_f64;
        let mut wnorm = 0.0_f64;
        // Variance state (running EWMA mean and second moment).
        let mut emean = 0.0_f64;
        let mut evar = 0.0_f64;

        for &v in vals {
            match v {
                None => {
                    if self.ignore_na {
                        out.push(None);
                    } else {
                        broken = true;
                        out.push(None);
                    }
                }
                Some(x) => {
                    if broken {
                        out.push(None);
                        continue;
                    }
                    seen += 1;
                    let value = match self.statistic {
                        EWMAStatistic::Mean => {
                            if self.adjust {
                                wsum = x + (1.0 - alpha) * wsum;
                                wnorm = 1.0 + (1.0 - alpha) * wnorm;
                                wsum / wnorm
                            } else {
                                let y = match prev {
                                    Some(p) => alpha * x + (1.0 - alpha) * p,
                                    None => x,
                                };
                                prev = Some(y);
                                y
                            }
                        }
                        EWMAStatistic::Var | EWMAStatistic::Std => {
                            if seen == 1 {
                                // First sample: seed the mean, variance is
                                // undefined (not emitted because seen < 2).
                                emean = x;
                                evar = 0.0;
                                0.0
                            } else {
                                let new_mean = alpha * x + (1.0 - alpha) * emean;
                                evar = (1.0 - alpha) * (evar + alpha * (x - emean).powi(2));
                                emean = new_mean;
                                if self.statistic == EWMAStatistic::Var {
                                    evar
                                } else {
                                    evar.sqrt()
                                }
                            }
                        }
                    };
                    if seen < min_obs {
                        out.push(None);
                    } else {
                        out.push(Some(value));
                    }
                }
            }
        }
        out
    }
}

impl Default for ExponentiallyWeightedMovingAverage {
    fn default() -> Self {
        Self::new(&[], EWMASmoothing::Alpha(0.5))
    }
}

impl Fit<DataFrame> for ExponentiallyWeightedMovingAverage {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state at the top so a failed re-fit can't leave stale state.
        self.fitted = false;
        self.alpha = 0.0;

        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "EWMA.fit received a DataFrame with 0 rows. Provide at least one row.".into(),
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "EWMA: at least one column is required.".into(),
            ));
        }

        let alpha = self.resolve_alpha()?;
        self.alpha = alpha;

        let mut generated = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let c = x
                .column(col.as_str())
                .map_err(|_| Error::InvalidInput(format!("EWMA: column '{}' not found.", col)))?;
            if c.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "EWMA: column '{}' has dtype {}; expected Float64.",
                    col,
                    c.dtype()
                )));
            }
            let name = self.output_name(col);
            if x.column(name.as_str()).is_ok() {
                return Err(Error::InvalidInput(format!(
                    "EWMA: generated output name '{}' collides with an existing input column.",
                    name
                )));
            }
            if generated.contains(&name) {
                return Err(Error::InvalidInput(format!(
                    "EWMA: generated output name '{}' collides with another generated column.",
                    name
                )));
            }
            generated.push(name);
        }

        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for ExponentiallyWeightedMovingAverage {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "ExponentiallyWeightedMovingAverage".into(),
            ));
        }
        // Reject generated names that collide with the transform input so
        // `with_column` never silently overwrites a caller-provided column.
        for col in &self.columns {
            let name = self.output_name(col);
            if x.column(name.as_str()).is_ok() {
                return Err(Error::InvalidInput(format!(
                    "EWMA.transform: generated output name '{}' collides with an existing input column.",
                    name
                )));
            }
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = x
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "EWMA.transform: column '{}' not found. {}",
                        col, e
                    ))
                })?
                .as_materialized_series()
                .clone();
            let ca = s
                .f64()
                .map_err(|_| Error::InvalidInput("column must be f64".into()))?;
            let vals: Vec<Option<f64>> = ca.iter().collect();
            let computed = self.compute_column(&vals);
            let new_ca: ChunkedArray<Float64Type> = computed.into_iter().collect();
            let name = self.output_name(col);
            out.with_column(new_ca.into_series().with_name(name.as_str().into()).into())
                .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(out)
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

    fn make_nulled_df(vals: &[Option<f64>]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), vals));
        DataFrame::new(vals.len(), vec![col]).unwrap()
    }

    fn output(result: &DataFrame, name: &str) -> Vec<Option<f64>> {
        result.column(name).unwrap().f64().unwrap().iter().collect()
    }

    #[test]
    fn test_mean_alpha_one_equals_input() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(1.0));
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        assert_eq!(
            output(&result, "x_ewm_mean_1"),
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]
        );
    }

    #[test]
    fn test_mean_alpha_half_unadjusted_recursive() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert_relative_eq!(got[0].unwrap(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(got[1].unwrap(), 1.5, epsilon = 1e-9);
        assert_relative_eq!(got[2].unwrap(), 2.25, epsilon = 1e-9);
    }

    #[test]
    fn test_adjusted_and_unadjusted_differ_early_converge_later() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut adj = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        adj.fit(df.clone()).unwrap();
        let adj_out = output(&adj.transform(df.clone()).unwrap(), "x_ewm_mean_0.5");

        let mut unadj = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false);
        unadj.fit(df.clone()).unwrap();
        let unadj_out = output(&unadj.transform(df).unwrap(), "x_ewm_mean_0.5");

        // Early rows differ.
        assert!(adj_out[1].unwrap() != unadj_out[1].unwrap());
        // Later rows converge.
        assert_relative_eq!(adj_out[7].unwrap(), unadj_out[7].unwrap(), epsilon = 0.05);
    }

    #[test]
    fn test_span_matches_equivalent_alpha() {
        // span = 3 => alpha = 2 / (3 + 1) = 0.5.
        let df = make_df(&[1.0, 2.0, 3.0, 4.0]);
        let mut span = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Span(3.0));
        span.fit(df.clone()).unwrap();
        let span_out = output(&span.transform(df.clone()).unwrap(), "x_ewm_mean_0.5");

        let mut alpha = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        alpha.fit(df.clone()).unwrap();
        let alpha_out = output(&alpha.transform(df).unwrap(), "x_ewm_mean_0.5");

        assert_eq!(span_out, alpha_out);
    }

    #[test]
    fn test_com_matches_equivalent_alpha() {
        // com = 1 => alpha = 1 / (1 + 1) = 0.5.
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut com = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Com(1.0));
        com.fit(df.clone()).unwrap();
        let com_out = output(&com.transform(df.clone()).unwrap(), "x_ewm_mean_0.5");

        let mut alpha = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        alpha.fit(df.clone()).unwrap();
        let alpha_out = output(&alpha.transform(df).unwrap(), "x_ewm_mean_0.5");

        assert_eq!(com_out, alpha_out);
    }

    #[test]
    fn test_half_life_matches_equivalent_alpha() {
        // half_life = 1 => alpha = 1 - 0.5^(1/1) = 0.5.
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut hl = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::HalfLife(1.0));
        hl.fit(df.clone()).unwrap();
        let hl_out = output(&hl.transform(df.clone()).unwrap(), "x_ewm_mean_0.5");

        let mut alpha = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        alpha.fit(df.clone()).unwrap();
        let alpha_out = output(&alpha.transform(df).unwrap(), "x_ewm_mean_0.5");

        assert_eq!(hl_out, alpha_out);
    }

    #[test]
    fn test_var_constant_input_is_zero() {
        let df = make_df(&[2.0, 2.0, 2.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_var_0.5");
        assert!(got[0].is_none());
        for v in got.iter().skip(1) {
            assert_relative_eq!(v.unwrap(), 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_var_increasing_input_monotonic() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_var_0.5");
        assert!(got[0].is_none());
        for i in 1..got.len() - 1 {
            assert!(got[i + 1].unwrap() > got[i].unwrap());
        }
    }

    #[test]
    fn test_std_is_sqrt_of_var() {
        let df = make_df(&[1.0, 3.0, 5.0, 7.0]);
        let mut var = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var);
        var.fit(df.clone()).unwrap();
        let var_out = output(&var.transform(df.clone()).unwrap(), "x_ewm_var_0.5");

        let mut std = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Std);
        std.fit(df.clone()).unwrap();
        let std_out = output(&std.transform(df).unwrap(), "x_ewm_std_0.5");

        for i in 1..var_out.len() {
            assert_relative_eq!(
                std_out[i].unwrap(),
                var_out[i].unwrap().sqrt(),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_ignore_na_false_propagates_null_onward() {
        let df = make_nulled_df(&[Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert_relative_eq!(got[0].unwrap(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(got[1].unwrap(), 1.5, epsilon = 1e-9);
        assert!(got[2].is_none());
        assert!(got[3].is_none());
        assert!(got[4].is_none());
    }

    #[test]
    fn test_ignore_na_true_skips_nulls() {
        let df = make_nulled_df(&[Some(1.0), None, Some(2.0), Some(3.0)]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false)
            .ignore_na(true);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert_relative_eq!(got[0].unwrap(), 1.0, epsilon = 1e-9);
        assert!(got[1].is_none());
        // After skipping the null, the recursion continues from the last value.
        assert_relative_eq!(got[2].unwrap(), 1.5, epsilon = 1e-9);
        assert_relative_eq!(got[3].unwrap(), 2.25, epsilon = 1e-9);
    }

    #[test]
    fn test_min_periods_gating() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false)
            .min_periods(2);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert!(got[0].is_none());
        assert_relative_eq!(got[1].unwrap(), 1.5, epsilon = 1e-9);
        assert_relative_eq!(got[2].unwrap(), 2.25, epsilon = 1e-9);
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let df = make_df(&[1.0, 2.0]);
        let e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        let err = e.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_fit_empty_input_errors() {
        let col = Column::from(Series::new("x".into(), Vec::<f64>::new()));
        let df = DataFrame::new(0, vec![col]).unwrap();
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fit_empty_columns_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&[], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fit_missing_column_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["nope"], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fit_non_f64_column_errors() {
        let col = Column::from(Series::new("x".into(), &["a", "b"]));
        let df = DataFrame::new(2, vec![col]).unwrap();
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_alpha_out_of_range_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.0));
        let err = e.fit(df.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let mut e2 = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(1.5));
        let err2 = e2.fit(df.clone()).unwrap_err();
        assert!(matches!(err2, Error::InvalidInput(_)));
    }

    #[test]
    fn test_span_out_of_range_errors() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Span(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_generated_name_collision_with_input_errors() {
        let a = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let b = Column::from(Series::new("x_ewm_mean_0.5".into(), &[10.0_f64, 20.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_transform_time_collision_errors() {
        // The fit input has only "x", so fit succeeds. But the transform input
        // also carries a column named like the generated output; transform must
        // reject it rather than silently overwrite the caller's column.
        let fit_df = make_df(&[1.0, 2.0, 3.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        e.fit(fit_df).unwrap();

        let a = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new(
            "x_ewm_mean_0.5".into(),
            &[10.0_f64, 20.0, 30.0],
        ));
        let df = DataFrame::new(3, vec![a, b]).unwrap();
        let err = e.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_generated_name_collision_between_columns_errors() {
        let a = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x", "x"], EWMASmoothing::Alpha(0.5));
        let err = e.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_builder_defaults_and_statistic_option() {
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5));
        // Defaults: Mean, adjust=true.
        assert_eq!(e.statistic, EWMAStatistic::Mean);
        assert!(e.adjust);
        assert_eq!(e.min_periods, 0);
        assert!(!e.ignore_na);
        // Builder options mutate the instance.
        e = e
            .statistic(EWMAStatistic::Std)
            .adjust(false)
            .min_periods(1)
            .ignore_na(true);
        assert_eq!(e.statistic, EWMAStatistic::Std);
        assert!(!e.adjust);
        assert_eq!(e.min_periods, 1);
        assert!(e.ignore_na);
    }

    #[test]
    fn test_nan_propagates_in_mean() {
        let col = Column::from(Series::new(
            "x".into(),
            &[Some(1.0_f64), Some(f64::NAN), Some(3.0)],
        ));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert_relative_eq!(got[0].unwrap(), 1.0, epsilon = 1e-9);
        assert!(got[1].unwrap().is_nan());
        assert!(got[2].unwrap().is_nan());
    }

    #[test]
    fn test_min_periods_beyond_length_all_none() {
        let df = make_df(&[1.0, 2.0]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .adjust(false)
            .min_periods(5);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_mean_0.5");
        assert!(got[0].is_none());
        assert!(got[1].is_none());
    }

    #[test]
    fn test_adjust_does_not_affect_variance() {
        let df = make_df(&[1.0, 3.0, 5.0, 7.0]);
        let mut adj = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var)
            .adjust(true);
        adj.fit(df.clone()).unwrap();
        let adj_out = output(&adj.transform(df.clone()).unwrap(), "x_ewm_var_0.5");

        let mut unadj = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var)
            .adjust(false);
        unadj.fit(df.clone()).unwrap();
        let unadj_out = output(&unadj.transform(df).unwrap(), "x_ewm_var_0.5");

        assert_eq!(adj_out, unadj_out);
    }

    #[test]
    fn test_var_ignore_na_true_skips_nulls() {
        let df = make_nulled_df(&[Some(1.0), None, Some(3.0), Some(5.0)]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var)
            .ignore_na(true);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_var_0.5");
        assert!(got[0].is_none());
        assert!(got[1].is_none());
        // Two non-null samples seen by row 2, so a value is produced:
        // var = (1-alpha) * (0 + alpha*(3.0 - 1.0)^2) = 0.5 * 0.5 * 4 = 1.0.
        assert_relative_eq!(got[2].unwrap(), 1.0, epsilon = 1e-9);
        assert!(got[3].unwrap() > got[2].unwrap());
    }

    #[test]
    fn test_var_ignore_na_false_propagates_null_onward() {
        let df = make_nulled_df(&[Some(1.0), Some(2.0), None, Some(4.0)]);
        let mut e = ExponentiallyWeightedMovingAverage::new(&["x"], EWMASmoothing::Alpha(0.5))
            .statistic(EWMAStatistic::Var);
        e.fit(df.clone()).unwrap();
        let result = e.transform(df).unwrap();

        let got = output(&result, "x_ewm_var_0.5");
        assert!(got[0].is_none());
        assert_relative_eq!(got[1].unwrap(), 0.25, epsilon = 1e-9);
        assert!(got[2].is_none());
        assert!(got[3].is_none());
    }
}
