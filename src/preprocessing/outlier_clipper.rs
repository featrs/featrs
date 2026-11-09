//! Rule-based outlier clipping.
//!
//! [`OutlierClipper`] clips the extreme values of each `Float64` column to
//! bounds learned from the training data using one of three statistical
//! rules: the interquartile range (Tukey's fences), z-scores, or the median
//! absolute deviation. It complements
//! [`Winsorizer`](crate::preprocessing::winsorizer::Winsorizer), which
//! clips at fixed quantiles rather than at rule-derived fences.

use polars::prelude::*;

use crate::preprocessing::scaler::percentile_sorted;
use crate::traits::FeatureNamesOut;
use crate::traits::output_names::identity_names;
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// The rule used to derive per-column clipping bounds at [`fit`](Fit::fit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipMethod {
    /// Tukey's fences: values outside `[Q1 - k·IQR, Q3 + k·IQR]` are
    /// clipped, where `IQR = Q3 - Q1` is the interquartile range of the
    /// training values.
    ///
    /// The default multiplier is `k = 1.5`, the classic choice that marks
    /// points further than `1.5 · IQR` beyond the quartiles as outliers.
    IQR {
        /// Fence multiplier (default `1.5`).
        k: f64,
    },
    /// Values more than `k` population standard deviations from the mean
    /// are clipped.
    ///
    /// The standard deviation is computed with denominator `n` (the
    /// population formula, matching NumPy's default). Z-scores assume the
    /// data is approximately normal; on skewed data the mean and standard
    /// deviation are dragged toward the outliers, which can mask them.
    ZScore {
        /// Number of standard deviations (default `3.0`).
        k: f64,
    },
    /// Robust z-score variant: values more than `k · 1.4826 · MAD` from the
    /// median are clipped, where `MAD` is the median absolute deviation from
    /// the median.
    ///
    /// The `1.4826` factor makes the MAD a consistent estimator of the
    /// standard deviation for normal data, so `k` is comparable to the
    /// z-score rule. The median is robust to outliers, but on bimodal data
    /// it may sit between the modes, so some outliers can be missed.
    MAD {
        /// Median-absolute-deviation multiplier (default `3.0`).
        k: f64,
    },
}

/// Per-column clipping bounds learned at [`fit`](Fit::fit) time.
struct ClipParam {
    name: String,
    lo: f64,
    hi: f64,
}

/// Clip outliers to bounds learned from a statistical rule.
///
/// For each fitted `Float64` column, [`fit`](Fit::fit) learns a lower and
/// upper bound `(lo, hi)` from the training values using the configured
/// [`ClipMethod`], and [`transform`](Transform::transform) clamps every
/// value into `[lo, hi]`: values below `lo` are raised to `lo`, values
/// above `hi` are lowered to `hi`, in-sample values inside the range are
/// unchanged.
///
/// # Behaviour
///
/// - Only `Float64` columns are clipped; columns of other dtypes are passed
///   through unchanged. When no explicit [`columns`](Self::columns) are
///   given, all `Float64` columns are auto-discovered at fit time.
/// - **Nulls are preserved** as null. **`NaN` is preserved** as `NaN`:
///   `f64::clamp` uses IEEE-754 comparisons, under which `NaN` is neither
///   below nor above any bound, so it passes through untouched (and `NaN`
///   and `±Inf` values are excluded when learning the bounds). `±Inf`
///   transform-time values are clipped to the bounds.
/// - **Zero-spread columns pass through unchanged**: with no spread to
///   learn from (IQR `== 0`, `std == 0`, or `MAD == 0`), the bounds would
///   collapse onto a single point and clamp every value onto it, so the
///   column is left untouched instead. This includes constant columns
///   (where the collapse would have been a no-op anyway) and non-constant
///   columns such as `[0, 0, 0, 0, 1000]`, whose `1000` must survive.
/// - **Degenerate bound arithmetic passes through**: if computing a bound
///   overflows (e.g. the z-score variance on extreme-magnitude values near
///   `f64::MAX` makes the mean and standard deviation `±inf`, so `lo`/`hi`
///   become `NaN`), the column is passed through unchanged rather than
///   clamped to a nonsensical bound.
/// - Bounds are learned from the non-null, finite values only; an
///   all-null, all-`NaN`, or all-`±Inf` column is an error at fit time
///   (there are no finite values from which to learn bounds) — impute
///   first or drop the column.
/// - Out-of-sample extremes are clipped to the training-time bounds: that
///   is the whole point of clipping.
/// - If a fitted column is missing or no longer `Float64` in the frame
///   passed to [`transform`](Transform::transform), transform returns
///   [`Error::InvalidInput`].
///
/// The multiplier `k` is validated at [`fit`](Fit::fit) time: it must be
/// finite and strictly positive; anything else (including `0`, which would
/// collapse every value to a single point) returns
/// [`Error::InvalidInput`]. (The builder stores the values as given — it
/// returns `Self` and cannot signal an error at configuration time.)
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::outlier_clipper::{ClipMethod, OutlierClipper};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new(
///     "x".into(),
///     &[1.0_f64, 2.0, 3.0, 4.0, 100.0],
/// ));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut clipper = OutlierClipper::new()
///     .columns(&["x"])
///     .method(ClipMethod::IQR { k: 1.5 });
/// clipper.fit(df.clone())?;
/// let clipped = clipper.transform(df)?;
/// assert_eq!(clipped.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OutlierClipper {
    fitted: bool,
    columns: Option<Vec<String>>,
    method: ClipMethod,
    bounds: Option<Vec<ClipParam>>,
}

impl OutlierClipper {
    /// Create a new `OutlierClipper` using the IQR rule with `k = 1.5`.
    ///
    /// All `Float64` columns are clipped; columns of other dtypes are
    /// passed through unchanged.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: None,
            method: ClipMethod::IQR { k: 1.5 },
            bounds: None,
        }
    }

    /// Restrict clipping to the named columns.
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

    /// Set the rule used to derive the clipping bounds.
    ///
    /// Defaults to [`ClipMethod::IQR`] with `k = 1.5`. The `k` of the given
    /// method must be finite and strictly positive; the requirement is
    /// enforced at [`fit`](Fit::fit) time, which returns
    /// [`Error::InvalidInput`] for any invalid value.
    pub fn method(mut self, m: ClipMethod) -> Self {
        self.method = m;
        self
    }
}

impl Default for OutlierClipper {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for OutlierClipper {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale parameters.
        self.fitted = false;
        self.bounds = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "OutlierClipper.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let method = self.method;
        let k = match method {
            ClipMethod::IQR { k } | ClipMethod::ZScore { k } | ClipMethod::MAD { k } => k,
        };
        if !k.is_finite() || k <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "OutlierClipper.fit: invalid multiplier k = {k}. \
                 k must be finite and strictly positive."
            )));
        }

        let col_names = match &self.columns {
            Some(cols) => cols.clone(),
            None => require_f64_columns(&x, "OutlierClipper")?,
        };

        if col_names.is_empty() {
            return Err(Error::InvalidInput(
                "OutlierClipper.fit: no columns to clip. \
                 Provide at least one Float64 column or drop the empty column list."
                    .into(),
            ));
        }

        let mut bounds = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "OutlierClipper.fit: column '{name}' not found. {e}"
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "OutlierClipper.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            // Non-null, finite values only: NaN and ±Inf must not poison the
            // learned bounds (NaN-checked aggregation, cf. issue #35).
            let mut vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "OutlierClipper: column '{name}' has no non-null, finite values. \
                     Cannot learn bounds from an all-null or all-NaN column. \
                     Impute first or drop the column."
                )));
            }

            vals.sort_by(|a, b| a.total_cmp(b));

            let (lo, hi) = match method {
                ClipMethod::IQR { k } => {
                    let q1 = percentile_sorted(&vals, 25.0);
                    let q3 = percentile_sorted(&vals, 75.0);
                    let iqr = q3 - q1;
                    (q1 - k * iqr, q3 + k * iqr)
                }
                ClipMethod::ZScore { k } => {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                        / vals.len() as f64;
                    let std = var.sqrt();
                    (mean - k * std, mean + k * std)
                }
                ClipMethod::MAD { k } => {
                    let median = percentile_sorted(&vals, 50.0);
                    let mut devs: Vec<f64> = vals.iter().map(|v| (v - median).abs()).collect();
                    devs.sort_by(|a, b| a.total_cmp(b));
                    let mad = percentile_sorted(&devs, 50.0);
                    let spread = 1.4826 * mad;
                    (median - k * spread, median + k * spread)
                }
            };

            // Two degenerate cases must not yield point bounds that flatten
            // the whole column:
            // - arithmetic overflow (the z-score variance overflows on
            //   extreme-magnitude values, making mean and std ±inf so lo or
            //   hi becomes NaN) — f64::clamp panics on a NaN range;
            // - zero spread (Q1 == Q3, std == 0, or MAD == 0), which
            //   collapses the bounds onto a single point and would clamp
            //   every value onto it — e.g. an IQR of 0 on [0,0,0,0,1000]
            //   would destroy the 1000.
            // Both degrade to a pass-through instead, matching the "no
            // outliers" definition.
            let (lo, hi) = if lo.is_nan() || hi.is_nan() || lo == hi {
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                (lo, hi)
            };

            bounds.push(ClipParam {
                name: name.clone(),
                lo,
                hi,
            });
        }

        self.bounds = Some(bounds);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for OutlierClipper {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "OutlierClipper has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let bounds = self.bounds.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "OutlierClipper has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut out = x.clone();
        for p in bounds {
            // lo <= hi holds for every rule (IQR: Q1 - k·IQR <= Q3 + k·IQR
            // since k > 0 and Q1 <= Q3; z-score and MAD are symmetric around
            // a center), but each bound is a sum of rounded products, so a
            // 1-ulp inversion is conceivable; f64::clamp panics on
            // min > max, so order the pair defensively.
            let lo = p.lo.min(p.hi);
            let hi = p.lo.max(p.hi);
            replace_f64_column(&mut out, &p.name, "OutlierClipper", |v| v.clamp(lo, hi))?;
        }

        Ok(out)
    }
}

impl FeatureNamesOut for OutlierClipper {
    /// Output names are the input names: `transform` rewrites columns in place.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        identity_names(self.fitted, "OutlierClipper", input_columns)
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
    fn test_iqr_clips_both_outliers() {
        // [1,2,3,4,100,-50], n = 6:
        //   Q1 -> idx 1.25 -> 1*0.75 + 2*0.25 = 1.25
        //   Q3 -> idx 3.75 -> 3*0.25 + 4*0.75 = 3.75
        //   IQR = 2.5; lo = 1.25 - 1.5*2.5 = -2.5; hi = 3.75 + 1.5*2.5 = 7.5
        let mut c = OutlierClipper::new().columns(&["a"]);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0, -50.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[4], 7.5, epsilon = 1e-12); // 100 clipped down
        assert_relative_eq!(vals[5], -2.5, epsilon = 1e-12); // -50 clipped up
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-12); // interior untouched
        assert_relative_eq!(vals[3], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_default_method_is_iqr_k15() {
        // Same bounds as above on the first five values: hi = 7.
        let mut c = OutlierClipper::new().columns(&["a"]);
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[4], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_zscore_clips_outlier() {
        // [1,2,3,4,100], k = 1.5: mean = 22, population std = sqrt(1522).
        //   lo = 22 - 1.5*sqrt(1522) = -36.51922760939348
        //   hi = 22 + 1.5*sqrt(1522) = 80.51922760939348
        let mut c = OutlierClipper::new()
            .columns(&["a"])
            .method(ClipMethod::ZScore { k: 1.5 });
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[4], 80.51922760939348, epsilon = 1e-9);
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_zscore_k3_passes_through_mild_outlier() {
        // The issue sketch claimed [1,2,3,4,100] with k = 3 clips the 100,
        // but z = (100 - 22)/sqrt(1522) ≈ 2.0 < 3, so by the rule's own
        // definition ("more than k std from the mean") it is NOT an outlier.
        // The definition wins; the larger-sample test below covers real
        // z-score clipping.
        let mut c = OutlierClipper::new()
            .columns(&["a"])
            .method(ClipMethod::ZScore { k: 3.0 });
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[4], 100.0, epsilon = 1e-9); // unchanged
    }

    #[test]
    fn test_zscore_k3_clips_on_larger_sample() {
        // 1..=19 plus 100 (n = 20): mean = 14.5, std = 20.328551350256124,
        // hi = 14.5 + 3*std = 75.48565405076837 -> 100 clipped.
        let data: Vec<f64> = (1..=19).map(|i| i as f64).chain([100.0]).collect();
        let mut c = OutlierClipper::new()
            .columns(&["a"])
            .method(ClipMethod::ZScore { k: 3.0 });
        let df = df_with(&data);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[19], 75.48565405076837, epsilon = 1e-9);
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_mad_clips_outlier() {
        // [1,2,3,4,100]: median = 3, MAD = median(|v-3|) = 1.
        //   lo = 3 - 3*1.4826*1 = -1.4478; hi = 3 + 4.4478 = 7.4478
        let mut c = OutlierClipper::new()
            .columns(&["a"])
            .method(ClipMethod::MAD { k: 3.0 });
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[4], 7.4478, epsilon = 1e-9);
        assert_relative_eq!(vals[2], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_all_identical_column_passes_through() {
        for method in [
            ClipMethod::IQR { k: 1.5 },
            ClipMethod::ZScore { k: 3.0 },
            ClipMethod::MAD { k: 3.0 },
        ] {
            let mut c = OutlierClipper::new().columns(&["a"]).method(method);
            let df = df_with(&[5.0, 5.0, 5.0, 5.0, 5.0]);

            c.fit(df.clone()).unwrap();
            let result = c.transform(df).unwrap();

            for v in col_values(&result, "a") {
                assert_relative_eq!(v, 5.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_zero_std_zscore_passes_through() {
        // Constant column -> std = 0 -> bounds collapse onto the mean, which
        // equals every value -> no-op clamp.
        let mut c = OutlierClipper::new()
            .columns(&["a"])
            .method(ClipMethod::ZScore { k: 3.0 });
        let df = df_with(&[3.0, 3.0, 3.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        for v in col_values(&result, "a") {
            assert_relative_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_extreme_magnitude_zscore_passes_through() {
        // Variance overflow: mean and std both overflow to ±inf, so the raw
        // bounds would be NaN — f64::clamp panics on a NaN range. The
        // transformer must degrade to a pass-through instead: every value
        // must come back bit-for-bit unchanged, not merely finite.
        for data in [
            vec![1e308_f64, 1e308, 1e308],
            vec![-1e308_f64, -1e308, -1e308],
            vec![f64::MAX, f64::MAX, f64::MAX],
        ] {
            let mut c = OutlierClipper::new()
                .columns(&["a"])
                .method(ClipMethod::ZScore { k: 3.0 });
            let df = df_with(&data);

            c.fit(df.clone()).unwrap();
            let result = c.transform(df).unwrap();

            let vals = col_values(&result, "a");
            for (got, expected) in vals.iter().zip(data.iter()) {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "column must pass through bit-for-bit"
                );
            }
        }
    }

    #[test]
    fn test_zero_spread_columns_pass_through() {
        // Zero spread (Q1 == Q3, std == 0, MAD == 0) collapses the raw
        // bounds onto a single point; clamping to it would destroy every
        // value off that point, e.g. the 1000 in [0,0,0,0,1000]. The
        // transformer must leave such columns untouched.
        let cases: [(Vec<f64>, ClipMethod); 4] = [
            // IQR: Q1 == Q3 == 0, lo == hi == 0.
            (
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0],
                ClipMethod::IQR { k: 1.5 },
            ),
            // Negative side of the same IQR degenerate case.
            (
                vec![-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ClipMethod::IQR { k: 1.5 },
            ),
            // MAD: median 0, all deviations except one are 0 -> MAD == 0.
            (vec![0.0, 0.0, 0.0, 0.0, 1000.0], ClipMethod::MAD { k: 3.0 }),
            // ZScore: variance underflows to 0 on tiny magnitudes.
            (
                vec![1e-200_f64, 2e-200, 3e-200],
                ClipMethod::ZScore { k: 3.0 },
            ),
        ];

        for (data, method) in cases {
            let mut c = OutlierClipper::new().columns(&["a"]).method(method);
            let df = df_with(&data);

            c.fit(df.clone()).unwrap();
            let result = c.transform(df).unwrap();

            let vals = col_values(&result, "a");
            for (got, expected) in vals.iter().zip(data.iter()) {
                assert_relative_eq!(got, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_nan_and_inf_at_transform_are_handled() {
        // Fit on 1..=9 (n = 9): Q1 = 3, Q3 = 7, IQR = 4,
        // lo = 3 - 6 = -3, hi = 7 + 6 = 13.
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let mut c = OutlierClipper::new().columns(&["a"]);
        c.fit(df_with(&data)).unwrap();

        let a = Column::from(Series::new(
            "a".into(),
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 5.0, f64::NAN],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let new_df = DataFrame::new(5, vec![a, b]).unwrap();
        let result = c.transform(new_df).unwrap();

        let ca = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(vals[0].unwrap().is_nan(), "NaN must be preserved as NaN");
        assert_relative_eq!(vals[1].unwrap(), 13.0, epsilon = 1e-12); // +Inf clipped to hi
        assert_relative_eq!(vals[2].unwrap(), -3.0, epsilon = 1e-12); // -Inf clipped to lo
        assert_relative_eq!(vals[3].unwrap(), 5.0, epsilon = 1e-12);
        assert!(vals[4].unwrap().is_nan(), "NaN must be preserved as NaN");
    }

    #[test]
    fn test_nulls_are_preserved() {
        // Non-null values [1, 3, 4, 100] (n = 4):
        //   Q1 = 1*0.25 + 3*0.75 = 2.5; Q3 = 4*0.75 + 100*0.25 = 28
        //   IQR = 25.5; lo = 2.5 - 38.25 = -35.75; hi = 28 + 38.25 = 66.25
        let a = Column::from(Series::new(
            "a".into(),
            &[Some(1.0_f64), None, Some(3.0), Some(4.0), Some(100.0)],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let df = DataFrame::new(5, vec![a, b]).unwrap();
        let mut c = OutlierClipper::new().columns(&["a"]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let ca = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(vals[1].is_none(), "null must be preserved as null");
        assert_relative_eq!(vals[0].unwrap(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[4].unwrap(), 66.25, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_at_fit_is_ignored() {
        // NaN and nulls must not poison the learned bounds: same bounds as
        // [1, 2, 4, 9] (n = 4): Q1 = 1.75, Q3 = 5.25, IQR = 3.5,
        // lo = 1.75 - 5.25 = -3.5, hi = 5.25 + 5.25 = 10.5.
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
        let mut c = OutlierClipper::new();

        c.fit(df).unwrap();
        assert!(c.bounds.is_some());
        let bounds = c.bounds.as_ref().unwrap();
        assert_relative_eq!(bounds[0].lo, -3.5, epsilon = 1e-12);
        assert_relative_eq!(bounds[0].hi, 10.5, epsilon = 1e-12);
    }

    #[test]
    fn test_out_of_sample_extremes_are_clipped() {
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let mut c = OutlierClipper::new().columns(&["a"]);
        c.fit(df_with(&data)).unwrap(); // bounds [-3, 13]

        let new_df = df_with(&[-10.0, 50.0, 4.0, 6.0, 5.0]);
        let result = c.transform(new_df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], -3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 13.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 4.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tiny_samples_are_graceful() {
        // n = 1: every rule collapses to lo = hi = the value -> pass-through.
        let mut c = OutlierClipper::new().columns(&["a"]);
        let df = df_with(&[7.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        let vals = col_values(&result, "a");
        assert_relative_eq!(vals[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_invalid_k_errors() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        for method in [
            ClipMethod::IQR { k: 0.0 },
            ClipMethod::IQR { k: -1.5 },
            ClipMethod::IQR { k: f64::NAN },
            ClipMethod::IQR { k: f64::INFINITY },
            ClipMethod::ZScore { k: 0.0 },
            ClipMethod::MAD { k: 0.0 },
        ] {
            let mut c = OutlierClipper::new().columns(&["a"]).method(method);
            assert!(
                matches!(c.fit(df.clone()), Err(Error::InvalidInput(_))),
                "k must be finite and strictly positive: {method:?}"
            );
        }
    }

    #[test]
    fn test_empty_input_errors() {
        let mut c = OutlierClipper::new();
        let df = DataFrame::empty();

        assert!(matches!(c.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_explicit_missing_column_errors() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut c = OutlierClipper::new().columns(&["nope"]);

        assert!(matches!(c.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_empty_explicit_column_list_errors() {
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut c = OutlierClipper::new().columns(&[]);

        assert!(matches!(c.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_non_f64_explicit_column_errors() {
        let a = Column::from(Series::new("a".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let mut c = OutlierClipper::new().columns(&["a"]);

        assert!(matches!(c.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_explicit_columns_only_clips_those() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0, 4.0, 100.0]));
        let b = Column::from(Series::new("b".into(), &[1.0_f64, 2.0, 3.0, 4.0, 100.0]));
        let df = DataFrame::new(5, vec![a, b]).unwrap();
        let mut c = OutlierClipper::new().columns(&["a"]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        // "a" clipped (hi = 7), "b" untouched even though it holds the same
        // values.
        let a_vals = col_values(&result, "a");
        assert_relative_eq!(a_vals[4], 7.0, epsilon = 1e-12);
        let b_vals = col_values(&result, "b");
        assert_relative_eq!(b_vals[4], 100.0, epsilon = 1e-12);
    }

    #[test]
    fn test_transform_missing_fitted_column_errors() {
        let mut c = OutlierClipper::new().columns(&["a"]);
        c.fit(df_with(&[1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();

        // Frame without the fitted column "a" -> InvalidInput, not a silent pass.
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let df = DataFrame::new(5, vec![b]).unwrap();

        assert!(matches!(c.transform(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_transform_non_f64_fitted_column_errors() {
        let mut c = OutlierClipper::new().columns(&["a"]);
        c.fit(df_with(&[1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();

        // Column present but no longer Float64 -> InvalidInput.
        let a = Column::from(Series::new("a".into(), &["x", "y", "z", "w", "v"]));
        let df = DataFrame::new(5, vec![a]).unwrap();

        assert!(matches!(c.transform(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_auto_discovers_all_f64_columns() {
        let mut c = OutlierClipper::new();
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 100.0]);

        c.fit(df.clone()).unwrap();
        let result = c.transform(df).unwrap();

        // Both f64 columns clipped at their own bounds: "a" hi = 7,
        // "b" = [0..4] hi = 6 -> nothing to clip.
        let a_vals = col_values(&result, "a");
        assert_relative_eq!(a_vals[4], 7.0, epsilon = 1e-12);
        let b_vals = col_values(&result, "b");
        assert_relative_eq!(b_vals[4], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let mut c = OutlierClipper::new().columns(&["a"]);
        let good = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        c.fit(good).unwrap();
        assert!(c.fitted);

        // Re-fit on an all-null column fails -> fitted must be reset.
        let a = Column::from(Series::new(
            "a".into(),
            [None::<f64>, None, None, None, None],
        ));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 1.0, 2.0, 3.0, 4.0]));
        let bad = DataFrame::new(5, vec![a, b]).unwrap();
        assert!(matches!(c.fit(bad), Err(Error::Computation(_))));
        assert!(!c.fitted);
        assert!(c.bounds.is_none());

        assert!(matches!(
            c.transform(df_with(&[1.0, 2.0, 3.0, 4.0, 5.0])),
            Err(Error::NotFitted(_))
        ));
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let c = OutlierClipper::new();
        let df = df_with(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!(matches!(c.transform(df), Err(Error::NotFitted(_))));
    }
}
