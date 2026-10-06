//! Automatic per-column scaling strategy selection.
//!
//! [`AutoScaler`] inspects the statistical shape of every `Float64` column at
//! fit time and delegates each column to the scaler that best matches it,
//! instead of making the caller choose between [`StandardScaler`],
//! [`MinMaxScaler`], [`RobustScaler`], [`MaxAbsScaler`] and
//! [`PowerTransformer`] by hand.
//!
//! Columns of other dtypes, and `Float64` columns with no usable value, are
//! passed through unchanged.
//!
//! [`StandardScaler`]: crate::preprocessing::scaler::StandardScaler
//! [`MinMaxScaler`]: crate::preprocessing::scaler::MinMaxScaler
//! [`RobustScaler`]: crate::preprocessing::scaler::RobustScaler
//! [`MaxAbsScaler`]: crate::preprocessing::max_abs_scaler::MaxAbsScaler
//! [`PowerTransformer`]: crate::preprocessing::power_transformer::PowerTransformer

use polars::prelude::*;

use super::max_abs_scaler::MaxAbsScaler;
use super::power_transformer::PowerTransformer;
use super::scaler::{MinMaxScaler, RobustScaler, StandardScaler, percentile_sorted};
use crate::pipeline::DataFrameTransformer;
use crate::pipeline::column_transformer::{ColumnTransformer, Remainder};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::require_f64_columns;

/// Smallest IQR treated as a usable spread, and smallest standard deviation
/// treated as non-degenerate. Below it the outlier severity has no meaningful
/// denominator, and `StandardScaler` would reject the column anyway.
const SPREAD_EPSILON: f64 = 1e-12;
/// Fraction of exact zeros above which a column counts as sparse.
const SPARSE_ZERO_FRACTION: f64 = 0.5;
/// `max(|x - median|) / IQR` above which the column counts as outlier-heavy.
const OUTLIER_SEVERITY: f64 = 10.0;
/// `|skewness|` above which the column counts as skewed.
const SKEW_THRESHOLD: f64 = 1.0;
/// Non-excess (raw) kurtosis below which the distribution counts as
/// flat/bounded. Gaussian data has raw kurtosis 3.0 and a uniform distribution
/// 1.8, so 2.0 separates the thin-tailed shapes.
const FLAT_KURTOSIS: f64 = 2.0;

/// Scaling strategy applied to a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingStrategy {
    /// Pick a strategy per column from the column's own statistics.
    Auto,
    /// [`StandardScaler`].
    Standard,
    /// [`MinMaxScaler`].
    MinMax,
    /// [`RobustScaler`].
    Robust,
    /// [`MaxAbsScaler`].
    MaxAbs,
    /// [`PowerTransformer`].
    Power,
}

/// Automatically chooses a scaling strategy per `Float64` column.
///
/// At fit time every `Float64` column is inspected and handed to the scaler
/// that matches its distribution. Columns delegated to the same scaler are
/// scaled together; all other columns (non-`Float64`, and `Float64` columns
/// with no finite value) pass through untouched.
///
/// # Output layout
///
/// Delegation runs through a [`ColumnTransformer`] with
/// [`Remainder::Passthrough`], so output columns come out **grouped by the
/// scaler that was chosen, with the pass-through remainder last** — not in the
/// input order. Column names are preserved, so select by name rather than by
/// position when the order matters.
///
/// # Fallback
///
/// The heuristic only ever picks a delegate that can accept the column, with
/// one exception: a right-skewed column can select
/// [`PowerTransformer`] and then be rejected by it when the power transform
/// collapses the column to a constant. Those columns are re-delegated to
/// [`RobustScaler`], and to [`StandardScaler`] if they have no usable IQR
/// either, rather than failing the whole `fit`.
/// [`AutoScaler::chosen_name`] and [`AutoScaler::column_types`] report the
/// scalers actually applied.
///
/// The fallback applies to the heuristic only. A strategy forced through
/// [`AutoScaler::strategy`] is used as-is, so a delegate that rejects a column
/// surfaces its own error rather than being silently swapped out.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::auto_scaler::AutoScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = AutoScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AutoScaler {
    fitted: bool,
    strategy: ScalingStrategy,
    chosen: Option<Box<dyn DataFrameTransformer>>,
    chosen_name: Option<String>,
    column_types: Option<Vec<(String, ScalingStrategy)>>,
}

impl AutoScaler {
    /// Create a new `AutoScaler` with the heuristic (`ScalingStrategy::Auto`).
    pub fn new() -> Self {
        Self {
            fitted: false,
            strategy: ScalingStrategy::Auto,
            chosen: None,
            chosen_name: None,
            column_types: None,
        }
    }

    /// Force one strategy for every `Float64` column instead of using the
    /// heuristic (default: [`ScalingStrategy::Auto`]).
    ///
    /// The strategy is passed to the delegate as-is: it is not re-checked
    /// against the column's distribution, so forcing a scaler that rejects the
    /// data (e.g. a constant column under `MinMax`) surfaces that scaler's own
    /// `fit` error.
    ///
    /// Changing the strategy after `fit` requires a re-`fit`: the fitted plan is
    /// cleared here so a stale selection cannot be reused.
    pub fn strategy(mut self, s: ScalingStrategy) -> Self {
        self.strategy = s;
        self.fitted = false;
        self.chosen = None;
        self.chosen_name = None;
        self.column_types = None;
        self
    }

    /// The strategy chosen for each scaled column, in the frame order the
    /// columns were inspected. The strategies are the ones actually applied,
    /// so a `Power` column the transform rejected shows its fallback here.
    /// Columns the heuristic left alone (non-`Float64`, and `Float64` with no
    /// finite value) are not listed.
    pub fn column_types(&self) -> Option<&[(String, ScalingStrategy)]> {
        self.column_types.as_deref()
    }

    /// Name of the scaler(s) selected during `fit`, e.g. `"StandardScaler"` or
    /// `"StandardScaler+RobustScaler"` for a mixed selection.
    pub fn chosen_name(&self) -> Option<&str> {
        self.chosen_name.as_deref()
    }
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for AutoScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave a stale plan usable by a
        // later transform.
        self.fitted = false;
        self.chosen = None;
        self.chosen_name = None;
        self.column_types = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "AutoScaler.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "AutoScaler")?;

        // Columns that get a scaler, in frame order. A Float64 column with no
        // finite value (all-null or NaN/Inf-only) cannot be scaled by any
        // delegate, so it is left out and passes through unchanged.
        let mut chosen_per_column: Vec<(String, ScalingStrategy)> = Vec::new();

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!("AutoScaler.fit: column '{name}' not found. {e}"))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "AutoScaler.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            // One pass over the column: collect its finite values and note
            // whether it also carried a `±Inf`. `MinMaxScaler` derives its
            // min/max from non-NaN values only, so a single `±Inf` anywhere in
            // the column makes its fitted range collapse and every value map to
            // NaN or 0; no other delegate has that weakness, so a column
            // containing an infinity never goes to `MinMaxScaler` under the
            // heuristic. A plain `NaN` is harmless here — `MinMaxScaler` skips
            // it when fitting.
            let mut vals: Vec<f64> = Vec::with_capacity(ca.len());
            let mut has_infinite = false;
            for v in ca.iter().flatten() {
                if v.is_finite() {
                    vals.push(v);
                } else if v.is_infinite() {
                    has_infinite = true;
                }
            }
            if vals.is_empty() {
                continue;
            }

            let strat = if self.strategy == ScalingStrategy::Auto {
                choose_strategy(&vals, has_infinite)
            } else {
                self.strategy
            };

            chosen_per_column.push((name.clone(), strat));
        }

        // Delegation goes through `ColumnTransformer`: one scaler per strategy
        // subset, and `Remainder::Passthrough` keeps the columns this scaler
        // left alone (non-Float64, and Float64 with no finite value). Output
        // columns therefore come out grouped by scaler, with the remainder
        // last.
        //
        // `PowerTransformer` is the only delegate that can reject the column the
        // heuristic handed it: a right-skewed column whose power transform
        // collapses to a constant fails its fit. Resolve each such column to the
        // IQR-based scaler (still outlier-robust, which is why `Power` was
        // chosen) and then to the z-score scaler. Resolving per column means one
        // collapsing column neither fails the whole `fit` nor drags its healthy
        // siblings off `PowerTransformer`.
        //
        // A forced `ScalingStrategy` is left alone: the caller asked for that
        // scaler on every column, so the delegate's own rejection is reported
        // rather than silently overridden.
        let mut applied = chosen_per_column;
        if self.strategy == ScalingStrategy::Auto {
            for (name, strat) in applied.iter_mut() {
                if *strat != ScalingStrategy::Power
                    || fits_alone(Box::new(PowerTransformer::new()), name, &x)
                {
                    continue;
                }
                *strat = if fits_alone(Box::new(RobustScaler::new()), name, &x) {
                    ScalingStrategy::Robust
                } else {
                    ScalingStrategy::Standard
                };
            }
        }
        let composite = fit_plan(&applied, &x)?;

        let mut seen: Vec<ScalingStrategy> = Vec::new();
        for (_, s) in &applied {
            if !seen.contains(s) {
                seen.push(*s);
            }
        }
        let chosen_name = seen
            .iter()
            .map(|s| strategy_name(*s))
            .collect::<Vec<_>>()
            .join("+");

        self.column_types = Some(applied);
        self.chosen_name = if chosen_name.is_empty() {
            None
        } else {
            Some(chosen_name)
        };
        self.chosen = Some(Box::new(composite));
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for AutoScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "AutoScaler has not been fitted. Call .fit(dataframe) before .transform().".into(),
            ));
        }
        let chosen = self.chosen.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "AutoScaler has not been fitted. Call .fit(dataframe) before .transform().".into(),
            )
        })?;
        chosen.transform(x)
    }
}

/// Pick a strategy from the column's own statistics.
///
/// `vals` holds the column's non-null, finite values; `has_infinite` records
/// whether the raw column also contained a `±Inf`.
///
/// The checks are ordered most-specific-first: sparse columns keep their zeros
/// under a max-abs scale, near-constant columns go to the only delegate that
/// tolerates zero variance, heavy-tailed outliers go to the IQR-based scaler,
/// skewed columns are normalized by the power transform, and flat low-kurtosis
/// ranges are mapped linearly to `[0, 1]`. Everything else — including
/// near-Gaussian and thin-tailed data — falls back to the z-score scaler.
///
/// The outlier test is skipped when the IQR is at or below
/// [`SPREAD_EPSILON`]: with no usable spread the severity ratio is
/// meaningless, and the IQR-based scaler would reject the column anyway. A
/// column containing a `±Inf` never selects `MinMax`, whose fitted range
/// collapses on it.
fn choose_strategy(vals: &[f64], has_infinite: bool) -> ScalingStrategy {
    let n = vals.len() as f64;

    let zeros = vals.iter().filter(|v| **v == 0.0).count() as f64;
    if zeros / n > SPARSE_ZERO_FRACTION {
        return ScalingStrategy::MaxAbs;
    }

    let mean = vals.iter().sum::<f64>() / n;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if variance <= SPREAD_EPSILON * SPREAD_EPSILON {
        return ScalingStrategy::MaxAbs;
    }
    if !variance.is_finite() {
        // Magnitudes large enough to overflow the second moment make the
        // moment-based tests meaningless (skew and kurtosis come out `NaN`).
        // MaxAbs only divides by `max|x|`, which stays finite for finite
        // input, so it is the one scale-safe choice here.
        return ScalingStrategy::MaxAbs;
    }

    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = percentile_sorted(&sorted, 50.0);
    let iqr = percentile_sorted(&sorted, 75.0) - percentile_sorted(&sorted, 25.0);
    if iqr > SPREAD_EPSILON {
        let severity = vals
            .iter()
            .map(|v| (v - median).abs())
            .fold(0.0f64, f64::max)
            / iqr;
        if severity > OUTLIER_SEVERITY {
            return ScalingStrategy::Robust;
        }
    }

    let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
    let m4 = vals.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let skew = m3 / variance.sqrt().powi(3);
    if skew.is_finite() && skew.abs() > SKEW_THRESHOLD {
        return ScalingStrategy::Power;
    }

    let kurtosis = m4 / (variance * variance);
    if kurtosis.is_finite() && kurtosis < FLAT_KURTOSIS && !has_infinite {
        return ScalingStrategy::MinMax;
    }

    ScalingStrategy::Standard
}

/// Human-readable name of a strategy, matching the scaler type it selects.
fn strategy_name(s: ScalingStrategy) -> &'static str {
    match s {
        ScalingStrategy::Auto => "Auto",
        ScalingStrategy::Standard => "StandardScaler",
        ScalingStrategy::MinMax => "MinMaxScaler",
        ScalingStrategy::Robust => "RobustScaler",
        ScalingStrategy::MaxAbs => "MaxAbsScaler",
        ScalingStrategy::Power => "PowerTransformer",
    }
}

/// Instantiate the scaler for a concrete strategy. `Auto` has no scaler of its
/// own — it is resolved per column before this is called.
fn make_scaler(s: ScalingStrategy) -> Option<Box<dyn DataFrameTransformer>> {
    match s {
        ScalingStrategy::Auto => None,
        ScalingStrategy::Standard => Some(Box::new(StandardScaler::new())),
        ScalingStrategy::MinMax => Some(Box::new(MinMaxScaler::new())),
        ScalingStrategy::Robust => Some(Box::new(RobustScaler::new())),
        ScalingStrategy::MaxAbs => Some(Box::new(MaxAbsScaler::new())),
        ScalingStrategy::Power => Some(Box::new(PowerTransformer::new())),
    }
}

/// Try `scaler` on a single column of `x`, reporting whether its `fit` accepts
/// it. Used to resolve columns the heuristic handed to a delegate that can
/// reject them, so the check stays attributable to one column.
fn fits_alone(scaler: Box<dyn DataFrameTransformer>, name: &str, x: &DataFrame) -> bool {
    match x.select(std::slice::from_ref(&name.to_string())) {
        Ok(subset) => {
            let mut scaler = scaler;
            scaler.fit(subset).is_ok()
        }
        Err(_) => false,
    }
}

/// Group a per-column plan by strategy, build the delegating
/// [`ColumnTransformer`], and fit it.
///
/// One scaler per distinct strategy, each bound to the columns that chose it,
/// plus [`Remainder::Passthrough`] for the columns `AutoScaler` leaves alone.
fn fit_plan(plan: &[(String, ScalingStrategy)], x: &DataFrame) -> Result<ColumnTransformer> {
    let mut entries: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)> = Vec::new();
    for (name, strat) in plan {
        if let Some(scaler) = make_scaler(*strat) {
            match entries
                .iter_mut()
                .find(|(n, _, _)| n.as_str() == strategy_name(*strat))
            {
                Some((_, _, cols)) => cols.push(name.clone()),
                None => entries.push((
                    strategy_name(*strat).to_string(),
                    scaler,
                    vec![name.clone()],
                )),
            }
        }
    }
    let mut composite = ColumnTransformer::new(entries, Remainder::Passthrough);
    composite.fit(x.clone())?;
    Ok(composite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn one_col(name: &str, vals: &[f64]) -> DataFrame {
        let col = Column::from(Series::new(name.into(), vals));
        DataFrame::new(vals.len(), vec![col]).unwrap()
    }

    fn f64_vals(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_zeros_column_chooses_max_abs() {
        let df = one_col("x", &[0.0, 0.0, 0.0, 0.0, 1.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MaxAbsScaler"));
        let out = s.transform(df).unwrap();
        // MaxAbsScaler passes an all-but-one-zero column through by its max abs
        // value: zeros stay zero.
        assert_eq!(f64_vals(&out, "x")[0], 0.0);
    }

    #[test]
    fn test_all_zero_column_chooses_max_abs() {
        let df = one_col("x", &[0.0, 0.0, 0.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MaxAbsScaler"));
        let out = s.transform(df).unwrap();
        assert_eq!(f64_vals(&out, "x"), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_constant_column_chooses_max_abs() {
        let df = one_col("x", &[5.0, 5.0, 5.0, 5.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        // MaxAbs is the only delegate the heuristic can hand a zero-variance
        // column to.
        assert_eq!(s.chosen_name(), Some("MaxAbsScaler"));
        s.transform(df).unwrap();
    }

    #[test]
    fn test_outliers_choose_robust() {
        let df = one_col("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("RobustScaler"));
        let out = s.transform(df).unwrap();
        // (100 - median 3.5) / IQR 2.5
        assert_relative_eq!(f64_vals(&out, "x")[5], 38.6);
    }

    #[test]
    fn test_skewed_column_chooses_power() {
        let df = one_col("x", &[0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("PowerTransformer"));
        s.transform(df).unwrap();
    }

    #[test]
    fn test_normal_like_chooses_standard() {
        let df = one_col(
            "x",
            &[
                -1.8, -1.2, -0.9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.8,
            ],
        );
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("StandardScaler"));
        let out = s.transform(df).unwrap();
        // Z-scored: mean 0.
        let vals = f64_vals(&out, "x");
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_bounded_flat_chooses_minmax() {
        let df = one_col(
            "x",
            &[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        );
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
        let out = s.transform(df).unwrap();
        let vals = f64_vals(&out, "x");
        assert_relative_eq!(vals[0], 0.0);
        assert_relative_eq!(vals[9], 1.0);
    }

    #[test]
    fn test_non_negative_flat_chooses_minmax() {
        // Skew-free non-negative data with thin tails (kurtosis < 2) maps
        // linearly onto [0, 1].
        let df = one_col(
            "x",
            &[
                0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.5, 3.7, 3.9,
            ],
        );
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        let out = s.transform(df).unwrap();
        let vals = f64_vals(&out, "x");
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
        assert_relative_eq!(vals[0], 0.0);
        assert_relative_eq!(vals[12], 1.0);
    }

    #[test]
    fn test_mixed_strategies_build_composite() {
        let a = Column::from(Series::new(
            "a".into(),
            &[-1.8f64, -1.2, -0.9, -0.6, -0.3, -0.1],
        ));
        let b = Column::from(Series::new(
            "b".into(),
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 100.0],
        ));
        let df = DataFrame::new(6, vec![a, b]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        let types = s.column_types().unwrap();
        assert_eq!(types[0].1, ScalingStrategy::Standard);
        assert_eq!(types[1].1, ScalingStrategy::Robust);
        assert_eq!(s.chosen_name(), Some("StandardScaler+RobustScaler"));
        let out = s.transform(df).unwrap();
        assert_eq!(out.width(), 2);
        assert_relative_eq!(f64_vals(&out, "b")[5], 38.6);
    }

    #[test]
    fn test_strategy_override_forces_selection() {
        let df = one_col("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let mut s = AutoScaler::new().strategy(ScalingStrategy::MinMax);
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
        let out = s.transform(df).unwrap();
        assert_relative_eq!(f64_vals(&out, "x")[5], 1.0);
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let df = one_col("x", &[1.0, 2.0, 3.0]);
        let s = AutoScaler::new();
        assert!(matches!(s.transform(df), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_empty_input_errors() {
        let df = DataFrame::empty();
        let mut s = AutoScaler::new();
        assert!(matches!(s.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_no_f64_columns_errors() {
        let col = Column::from(Series::new("s".into(), &["a", "b"]));
        let df = DataFrame::new(2, vec![col]).unwrap();
        let mut s = AutoScaler::new();
        assert!(matches!(s.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_all_null_column_passes_through() {
        let col = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), None);
        let out = s.transform(df).unwrap();
        assert_eq!(out.column("x").unwrap().null_count(), 3);
    }

    #[test]
    fn test_non_f64_columns_passed_through() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let s = Column::from(Series::new("s".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![x, s]).unwrap();
        let mut scaler = AutoScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        assert_eq!(out.width(), 2);
        assert_eq!(out.column("s").unwrap().dtype(), &DataType::String);
    }

    #[test]
    fn test_output_grouped_by_scaler_with_remainder_last() {
        // Input order is std, outlier-heavy, std, pass-through. Both standard
        // columns group together ahead of the robust one, so the output is
        // reordered (["a", "c", "b", "s"]) rather than following the input.
        let a = Column::from(Series::new(
            "a".into(),
            &[-1.8f64, -1.2, -0.9, -0.6, -0.3, -0.1],
        ));
        let b = Column::from(Series::new(
            "b".into(),
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 100.0],
        ));
        let c = Column::from(Series::new("c".into(), &[1.0f64, 2.0, 2.5, 3.0, 3.5, 4.0]));
        let s = Column::from(Series::new("s".into(), &["x", "y", "z", "x", "y", "z"]));
        let df = DataFrame::new(6, vec![a, b, c, s]).unwrap();
        let mut scaler = AutoScaler::new();
        scaler.fit(df.clone()).unwrap();
        assert_eq!(scaler.chosen_name(), Some("StandardScaler+RobustScaler"));
        let out = scaler.transform(df).unwrap();
        let names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .collect();
        assert_eq!(names, vec!["a", "c", "b", "s"]);
    }

    #[test]
    fn test_non_finite_column_avoids_minmax() {
        // The finite subset [1, 2, 3] looks flat enough for MinMaxScaler, but
        // MinMax would derive max = +Inf and emit NaN for every row, so a
        // column carrying a non-finite value falls through to StandardScaler.
        let df = one_col("x", &[1.0, 2.0, 3.0, f64::INFINITY]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("StandardScaler"));
        let out = s.transform(df).unwrap();
        // MinMaxScaler would have derived max = +Inf and emitted NaN for every
        // row; the z-score scaler keeps the finite rows finite.
        let vals = f64_vals(&out, "x");
        assert!(vals[..3].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_all_infinite_column_passes_through() {
        let col = Column::from(Series::new(
            "x".into(),
            &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
        ));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), None);
        let out = s.transform(df).unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.width(), 1);
    }

    #[test]
    fn test_nan_column_with_finite_values_scales() {
        // NaN is dropped from the statistics; a normal-like column with one
        // NaN still z-scores. NaN is a float value, not a null, so it is
        // carried through the scaling arithmetic unchanged.
        let col = Column::from(Series::new(
            "x".into(),
            &[
                -1.8f64,
                -1.2,
                -0.9,
                -0.6,
                -0.3,
                -0.1,
                0.1,
                0.3,
                0.6,
                0.9,
                1.2,
                1.8,
                f64::NAN,
            ],
        ));
        let df = DataFrame::new(13, vec![col]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("StandardScaler"));
        let out = s.transform(df).unwrap();
        assert_eq!(out.height(), 13);
        // NaN is a float value, not a null, so it survives as a non-finite entry.
        let vals = f64_vals(&out, "x");
        assert_eq!(vals.len(), 13);
        assert_eq!(vals.iter().filter(|v| v.is_finite()).count(), 12);
    }

    #[test]
    fn test_strategy_change_after_fit_requires_refit() {
        let df = one_col("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        let mut s = s.strategy(ScalingStrategy::MinMax);
        assert!(matches!(s.transform(df.clone()), Err(Error::NotFitted(_))));
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
    }

    #[test]
    fn test_power_rejected_column_falls_back() {
        // Near-constant data with one high value is right-skewed enough for the
        // power transform, which then collapses the column to a constant and
        // rejects it. `fit` must re-delegate instead of failing. The column's
        // IQR is zero too, so the IQR-based fallback is rejected as well and
        // the z-score scaler takes it.
        let df = one_col("x", &[1000.0, 1000.0, 1000.0, 1000.0, 1262.0]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("StandardScaler"));
        let out = s.transform(df).unwrap();
        assert_eq!(out.height(), 5);
        assert!(f64_vals(&out, "x").iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_failed_refit_leaves_scaler_unfitted() {
        let good = one_col("x", &[1.0, 2.0, 3.0]);
        let mut s = AutoScaler::new();
        s.fit(good.clone()).unwrap();
        assert!(s.fit(DataFrame::empty()).is_err());
        assert!(matches!(s.transform(good), Err(Error::NotFitted(_))));
        assert_eq!(s.chosen_name(), None);
    }

    #[test]
    fn test_small_finite_column_chooses_minmax() {
        // Counterfactual for the non-finite guard: the same finite values that
        // go to MinMaxScaler when the column carries no Inf.
        let df = one_col("x", &[1.0, 2.0, 3.0]);
        let mut s = AutoScaler::new();
        s.fit(df).unwrap();
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
    }

    #[test]
    fn test_huge_magnitudes_stay_finite() {
        // The second moment overflows to Inf, which would make skew/kurtosis
        // NaN and silently fall through to StandardScaler's std = Inf.
        let df = one_col("x", &[1e307, 1.5e307, 1.7e307]);
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MaxAbsScaler"));
        let out = s.transform(df).unwrap();
        assert!(f64_vals(&out, "x").iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rejected_power_column_does_not_demote_healthy_sibling() {
        // One collapsing column makes PowerTransformer reject its group, but
        // only that column may leave `Power`: the healthy skewed column keeps
        // the transform it was chosen for.
        let a = Column::from(Series::new(
            "a".into(),
            &[0.5f64, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
        ));
        let b = Column::from(Series::new(
            "b".into(),
            &[
                1000.0f64, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1262.0,
            ],
        ));
        let df = DataFrame::new(10, vec![a, b]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        let types = s.column_types().unwrap();
        assert_eq!(types[0], ("a".to_string(), ScalingStrategy::Power));
        assert_ne!(types[1].1, ScalingStrategy::Power);
        s.transform(df).unwrap();
    }

    #[test]
    fn test_forced_power_rejection_is_reported_not_overridden() {
        // An explicit strategy is the caller's call: a delegate that rejects the
        // column must surface its error instead of being swapped out.
        let df = one_col("x", &[1000.0, 1000.0, 1000.0, 1000.0, 1262.0]);
        let mut s = AutoScaler::new().strategy(ScalingStrategy::Power);
        assert!(s.fit(df).is_err());
    }

    #[test]
    fn test_nan_does_not_block_minmax() {
        // Only `±Inf` collapses MinMaxScaler's fitted range; MinMaxScaler skips
        // NaN, so a flat bounded column keeps its MinMax selection with a NaN
        // present.
        let col = Column::from(Series::new(
            "x".into(),
            &[
                0.0f64,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                f64::NAN,
            ],
        ));
        let df = DataFrame::new(11, vec![col]).unwrap();
        let mut s = AutoScaler::new();
        s.fit(df.clone()).unwrap();
        assert_eq!(s.chosen_name(), Some("MinMaxScaler"));
        s.transform(df).unwrap();
    }
}
