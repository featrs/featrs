//! Quantile-based winsorization.
//!
//! [`Winsorizer`] clips each `Float64` column's extreme values at
//! configurable quantiles of the training data — e.g. everything below the
//! 1st percentile is clamped to the 1st percentile value, everything above
//! the 99th percentile to the 99th percentile value. This releases outliers
//! gently without removing rows. Analogous to `scipy.stats.mstats.winsorize`
//! (one-sided capping at each tail).

use polars::prelude::*;

use crate::preprocessing::scaler::percentile_sorted;
use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_required_present, identity_names};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// Per-column clipping bounds learned at [`fit`](Fit::fit) time.
struct WinsorParam {
    name: String,
    lo: f64,
    hi: f64,
}

/// Clip extreme values at configurable quantiles of the training data.
///
/// For each fitted column, the lower and upper bounds are learned as the
/// `lower_quantile` and `upper_quantile` percentiles of the training values.
/// [`transform`](Transform::transform) then clamps every value into
/// `[lo, hi]`: values below `lo` are raised to `lo`, values above `hi` are
/// lowered to `hi`, in-sample values inside the range are unchanged.
///
/// # Behaviour
///
/// - Only `Float64` columns are winsorized; columns of other dtypes are
///   passed through unchanged. When no explicit [`columns`](Self::columns)
///   are given, all `Float64` columns are auto-discovered at fit time.
/// - **Nulls are preserved** as null. **`NaN` is preserved** as `NaN`:
///   `f64::clamp` uses IEEE-754 comparisons, under which `NaN` is neither
///   below nor above any bound, so it passes through untouched (and `NaN`
///   and `±Inf` values are excluded when learning the bounds). `±Inf`
///   transform-time values are clipped to the bounds.
/// - **`lo == hi`**: if the chosen quantiles coincide on a column (e.g. on a
///   constant column, or on small data where both percentiles land on the
///   same value), every value collapses to that single bound. For a
///   constant column this is a silent pass-through.
/// - An all-null, all-`NaN`, or all-`±Inf` column is an error at fit time
///   (there are no finite values from which to learn bounds) — impute first
///   or drop the column.
/// - Out-of-sample extremes are clipped to the training-time bounds: that is
///   the whole point of winsorization.
/// - If a fitted column is missing or no longer `Float64` in the frame
///   passed to [`transform`](Transform::transform), transform returns
///   [`Error::InvalidInput`].
///
/// Quantile validity is checked at [`fit`](Fit::fit) time: both quantiles
/// must be finite and satisfy `0.0 <= lower < upper <= 1.0`; anything else
/// returns [`Error::InvalidInput`]. (The builder stores the values as given —
/// it returns `Self` and cannot signal an error at configuration time.)
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::winsorizer::Winsorizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0, 4.0, 100.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut winsorizer = Winsorizer::new().quantiles(0.2, 0.8);
/// winsorizer.fit(df.clone())?;
/// let clipped = winsorizer.transform(df)?;
/// assert_eq!(clipped.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Winsorizer {
    fitted: bool,
    columns: Option<Vec<String>>,
    lower_quantile: f64,
    upper_quantile: f64,
    limits: Option<Vec<WinsorParam>>,
}

impl Winsorizer {
    /// Create a new `Winsorizer` clipping at the 5th and 95th percentiles.
    ///
    /// All `Float64` columns are winsorized; columns of other dtypes are
    /// passed through unchanged.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: None,
            lower_quantile: 0.05,
            upper_quantile: 0.95,
            limits: None,
        }
    }

    /// Restrict winsorization to the named columns.
    ///
    /// When omitted, the transformer auto-discovers all `Float64` columns
    /// at [`fit`](Fit::fit) time.
    ///
    /// Each column must exist in the frame passed to `fit` and have dtype
    /// `Float64`; otherwise `fit` returns [`Error::InvalidInput`].
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = Some(cols.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set the lower and upper clipping quantiles (default: `0.05`, `0.95`).
    ///
    /// Values must satisfy `0.0 <= lower < upper <= 1.0`; the requirement is
    /// enforced at [`fit`](Fit::fit) time, which returns
    /// [`Error::InvalidInput`] for any invalid pair.
    pub fn quantiles(mut self, lower: f64, upper: f64) -> Self {
        self.lower_quantile = lower;
        self.upper_quantile = upper;
        self
    }
}

impl Default for Winsorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for Winsorizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale parameters.
        self.fitted = false;
        self.limits = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "Winsorizer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let lower = self.lower_quantile;
        let upper = self.upper_quantile;
        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper > 1.0 || lower >= upper
        {
            return Err(Error::InvalidInput(format!(
                "Winsorizer.fit: invalid quantiles ({lower}, {upper}). \
                 Both must be finite and satisfy 0.0 <= lower < upper <= 1.0."
            )));
        }

        let col_names = match &self.columns {
            Some(cols) => cols.clone(),
            None => require_f64_columns(&x, "Winsorizer")?,
        };

        if col_names.is_empty() {
            return Err(Error::InvalidInput(
                "Winsorizer.fit: no columns to winsorize. \
                 Provide at least one Float64 column or drop the empty column list."
                    .into(),
            ));
        }

        let mut limits = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!("Winsorizer.fit: column '{name}' not found. {e}"))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "Winsorizer.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            // Non-null, finite values only: NaN and ±Inf must not poison the
            // learned percentiles (NaN-checked aggregation, cf. issue #35).
            let mut vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "Winsorizer: column '{name}' has no non-null, finite values. \
                     Cannot learn quantiles from an all-null or all-NaN column. \
                     Impute first or drop the column."
                )));
            }

            vals.sort_by(|a, b| a.total_cmp(b));

            let lo = percentile_sorted(&vals, lower * 100.0);
            let hi = percentile_sorted(&vals, upper * 100.0);

            limits.push(WinsorParam {
                name: name.clone(),
                lo,
                hi,
            });
        }

        self.limits = Some(limits);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Winsorizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Winsorizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let limits = self.limits.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "Winsorizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut out = x.clone();
        for p in limits {
            // lo <= hi holds for the exact percentile arithmetic (lower <
            // upper and percentiles are monotone non-decreasing), but each
            // interpolated bound is a sum of rounded products, so a 1-ulp
            // inversion is conceivable; f64::clamp panics on min > max, so
            // order the pair defensively.
            let lo = p.lo.min(p.hi);
            let hi = p.lo.max(p.hi);
            replace_f64_column(&mut out, &p.name, "Winsorizer", |v| v.clamp(lo, hi))?;
        }

        Ok(out)
    }
}

impl FeatureNamesOut for Winsorizer {
    /// Output names are the input names: `transform` rewrites columns in place.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        let names = identity_names(self.fitted, "Winsorizer", input_columns)?;
        if let Some(limits) = &self.limits {
            let required: Vec<String> = limits.iter().map(|l| l.name.clone()).collect();
            ensure_required_present(input_columns, &required, "Winsorizer")?;
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn df_with(values: &[f64]) -> DataFrame {
        let a = Column::from(Series::new("a".into(), values));
        let b_vals: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        let b = Column::from(Series::new("b".into(), &b_vals));
        DataFrame::new(values.len(), vec![a, b]).unwrap()
    }

    fn col_values(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_clips_outliers_to_quantile_bounds() {
        // Sorted [1, 2, 3, 4, 100], n = 5:
        //   p20 -> idx 0.8  -> 1.0*0.2 + 2.0*0.8 = 1.8
        //   p80 -> idx 3.2  -> 4.0*0.8 + 100.0*0.2 = 23.2
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.2, 0.8);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 1.8, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3], 4.0, epsilon = 1e-12);
        assert_relative_eq!(vals[4], 23.2, epsilon = 1e-12);
    }

    #[test]
    fn test_default_quantiles_clip_outer_five_percent() {
        // Default (0.05, 0.95) on 1..=20 (n = 20):
        //   p5  -> idx 0.95 -> 1*0.05 + 2*0.95 = 1.95
        //   p95 -> idx 18.05 -> 19*0.95 + 20*0.05 = 19.05
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut w = Winsorizer::new().columns(&["a"]);
        let df = df_with(&data);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 1.95, epsilon = 1e-12);
        assert_relative_eq!(vals[19], 19.05, epsilon = 1e-12); // 20 clipped down
        assert_relative_eq!(vals[10], 11.0, epsilon = 1e-12); // interior untouched
    }

    #[test]
    fn test_zero_and_one_quantiles_are_passthrough() {
        // (0.0, 1.0) -> bounds are exactly min and max -> no clipping.
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.0, 1.0);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[4], 100.0, epsilon = 1e-12);
    }

    #[test]
    fn test_values_exactly_at_bounds_are_unchanged() {
        // n = 9, sorted 1..=9: p25 -> idx 2.0 -> 3; p75 -> idx 6.0 -> 7.
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.25, 0.75);
        let df = df_with(&data);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[2], 3.0, epsilon = 1e-12); // lo bound itself
        assert_relative_eq!(vals[6], 7.0, epsilon = 1e-12); // hi bound itself
        assert_relative_eq!(vals[4], 5.0, epsilon = 1e-12); // interior
        assert_relative_eq!(vals[0], 3.0, epsilon = 1e-12); // 1 clipped up
        assert_relative_eq!(vals[8], 7.0, epsilon = 1e-12); // 9 clipped down
    }

    #[test]
    fn test_out_of_sample_extremes_are_clipped() {
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.25, 0.75);
        w.fit(df_with(&data)).unwrap();

        // New extremes below lo and above hi must be clamped to [3, 7].
        let new_df = df_with(&[0.0, 10.0, 4.0, 6.0, 5.0]);
        let result = w.transform(new_df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 7.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 4.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_and_inf_at_transform_are_handled() {
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.25, 0.75);
        w.fit(df_with(&data)).unwrap();

        let a = Column::from(Series::new(
            "a".into(),
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 5.0, f64::NAN],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let new_df = DataFrame::new(5, vec![a, b]).unwrap();
        let result = w.transform(new_df).unwrap();

        let ca = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(vals[0].unwrap().is_nan(), "NaN must be preserved as NaN");
        assert_relative_eq!(vals[1].unwrap(), 7.0, epsilon = 1e-12); // +Inf clipped to hi
        assert_relative_eq!(vals[2].unwrap(), 3.0, epsilon = 1e-12); // -Inf clipped to lo
        assert_relative_eq!(vals[3].unwrap(), 5.0, epsilon = 1e-12);
        assert!(vals[4].unwrap().is_nan(), "NaN must be preserved as NaN");
    }

    #[test]
    fn test_nulls_are_preserved() {
        let a = Column::from(Series::new(
            "a".into(),
            &[Some(1.0_f64), None, Some(3.0), Some(4.0), Some(100.0)],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let df = DataFrame::new(5, vec![a, b]).unwrap();
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.2, 0.8);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let ca = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(vals[1].is_none(), "null must be preserved as null");
        // Non-null values are [1, 3, 4, 100] (n = 4): p20 -> idx 0.6 ->
        // 1*0.4 + 3*0.6 = 2.2; p80 -> idx 2.4 -> 4*0.6 + 100*0.4 = 42.4.
        assert_relative_eq!(vals[0].unwrap(), 2.2, epsilon = 1e-12);
        assert_relative_eq!(vals[4].unwrap(), 42.4, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_at_fit_is_ignored() {
        // NaN must not poison the learned bounds: same bounds as [1..=9].
        let a = Column::from(Series::new(
            "a".into(),
            &[
                Some(1.0_f64),
                Some(2.0),
                None,
                Some(4.0),
                Some(f64::NAN),
                Some(9.0),
            ],
        ));
        let df = DataFrame::new(6, vec![a]).unwrap();
        let mut w = Winsorizer::new().quantiles(0.25, 0.75);

        w.fit(df).unwrap();
        // bounds are the 25th/75th percentiles of [1, 2, 4, 9] (n = 4):
        //   p25 -> idx 0.75 -> 1*0.25 + 2*0.75 = 1.75; p75 -> idx 2.25 -> 4*0.75 + 9*0.25 = 5.25
        assert!(w.limits.is_some());
        let limits = w.limits.as_ref().unwrap();
        assert_relative_eq!(limits[0].lo, 1.75, epsilon = 1e-12);
        assert_relative_eq!(limits[0].hi, 5.25, epsilon = 1e-12);
    }

    #[test]
    fn test_invalid_quantiles_error() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // lower >= upper
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.6, 0.4);
        assert!(matches!(w.fit(df.clone()), Err(Error::InvalidInput(_))));

        // equal quantiles
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.5, 0.5);
        assert!(matches!(w.fit(df.clone()), Err(Error::InvalidInput(_))));

        // out of range
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(-0.1, 0.9);
        assert!(matches!(w.fit(df.clone()), Err(Error::InvalidInput(_))));

        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.1, 1.5);
        assert!(matches!(w.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_all_identical_column_is_passthrough() {
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.2, 0.8);
        let df = df_with(&[5.0, 5.0, 5.0, 5.0, 5.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        for v in vals {
            assert_relative_eq!(v, 5.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lo_equals_hi_collapses_all_values() {
        // n = 10: p50 and p60 both interpolate between two 3s -> lo == hi == 3.
        let data = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.5, 0.6);
        let df = df_with(&data);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        for v in vals {
            assert_relative_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_single_row_column() {
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.2, 0.8);
        let df = df_with(&[7.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_all_null_column_errors() {
        let a = Column::from(Series::new("a".into(), [None::<f64>, None, None]));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();
        let mut w = Winsorizer::new().columns(&["a"]);

        assert!(matches!(w.fit(df), Err(Error::Computation(_))));
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let w = Winsorizer::new();
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!(matches!(w.transform(df), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_empty_input_errors() {
        let mut w = Winsorizer::new();
        let df = DataFrame::empty();

        assert!(matches!(w.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_explicit_missing_column_errors() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut w = Winsorizer::new().columns(&["nope"]);

        assert!(matches!(w.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_empty_explicit_column_list_errors() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut w = Winsorizer::new().columns(&[]);

        assert!(matches!(w.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_non_f64_explicit_column_errors() {
        let a = Column::from(Series::new("a".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let mut w = Winsorizer::new().columns(&["a"]);

        assert!(matches!(w.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_explicit_columns_only_winsorizes_those() {
        let mut w = Winsorizer::new().columns(&["b"]).quantiles(0.2, 0.8);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        // "a" untouched (contains the outlier), "b" clipped at its own bounds.
        let a_vals = col_values(&result, "a");
        assert_relative_eq!(a_vals[4], 100.0, epsilon = 1e-12);

        // b = [0,1,2,3,4]: p20 -> idx 0.8 -> 0.8; p80 -> idx 3.2 -> 3.2.
        let b_vals = col_values(&result, "b");
        assert_relative_eq!(b_vals[0], 0.8, epsilon = 1e-12);
        assert_relative_eq!(b_vals[4], 3.2, epsilon = 1e-12);
    }

    #[test]
    fn test_transform_missing_fitted_column_errors() {
        let mut w = Winsorizer::new().columns(&["a"]).quantiles(0.2, 0.8);
        w.fit(df_with(&[1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();

        // Frame without the fitted column "a" -> InvalidInput, not a silent pass.
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let df = DataFrame::new(5, vec![b]).unwrap();

        assert!(matches!(w.transform(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_auto_discovers_all_f64_columns() {
        let mut w = Winsorizer::new().quantiles(0.2, 0.8);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        w.fit(df.clone()).unwrap();
        let result = w.transform(df).unwrap();

        // Both f64 columns winsorized.
        let a_vals = col_values(&result, "a");
        assert_relative_eq!(a_vals[4], 23.2, epsilon = 1e-12);
        let b_vals = col_values(&result, "b");
        assert_relative_eq!(b_vals[4], 3.2, epsilon = 1e-12);
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let mut w = Winsorizer::new().columns(&["a"]);
        let good = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        w.fit(good).unwrap();
        assert!(w.fitted);

        // Re-fit on an all-null column fails -> fitted must be reset.
        let a = Column::from(Series::new(
            "a".into(),
            [None::<f64>, None, None, None, None],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let bad = DataFrame::new(5, vec![a, b]).unwrap();
        assert!(matches!(w.fit(bad), Err(Error::Computation(_))));
        assert!(!w.fitted);
        assert!(w.limits.is_none());

        assert!(matches!(
            w.transform(df_with(&[1.0, 2.0, 3.0, 4.0, 5.0])),
            Err(Error::NotFitted(_))
        ));
    }
}
