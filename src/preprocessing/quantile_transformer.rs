use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::traits::inverse::{HasInverse, InverseTransform};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// The target distribution for [`QuantileTransformer`] output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputDistribution {
    /// Values are mapped to ranks in `[0, 1]`, producing a uniform distribution.
    Uniform,
    /// Values are mapped through the probit function, producing a standard normal distribution.
    Normal,
}

/// Per-column quantile reference values learned during `fit`.
struct QuantileRef {
    name: String,
    references: Vec<f64>,
}

/// Transform features using quantile information.
///
/// For each column, the transformer computes an evenly-spaced grid of quantile
/// reference values from the training data.  During `transform` each value is
/// mapped to its rank on this grid, optionally passing it through the probit
/// function (inverse normal CDF) to produce a standard-normal output.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::quantile_transformer::QuantileTransformer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut qt = QuantileTransformer::new();
/// qt.fit(df.clone())?;
/// let out = qt.transform(df)?;
/// assert_eq!(out.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QuantileTransformer {
    fitted: bool,
    n_quantiles: usize,
    output_distribution: OutputDistribution,
    references: Option<Vec<QuantileRef>>,
}

impl QuantileTransformer {
    /// Create a new `QuantileTransformer` with uniform output and
    /// `n_quantiles = 1000`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
            references: None,
        }
    }

    /// Set the number of quantile reference points (default: `1000`).
    ///
    /// The actual number used during `fit` is `min(n_quantiles, n_samples)`.
    /// More quantiles give a finer-grained approximation of the distribution.
    pub fn n_quantiles(mut self, n: usize) -> Self {
        self.n_quantiles = n;
        self
    }

    /// Set the output distribution (default: [`OutputDistribution::Uniform`]).
    pub fn output_distribution(mut self, d: OutputDistribution) -> Self {
        self.output_distribution = d;
        self
    }
}

impl Default for QuantileTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for QuantileTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "QuantileTransformer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        if self.n_quantiles == 0 {
            return Err(Error::InvalidInput(
                "QuantileTransformer.fit: n_quantiles must be positive, got 0. \
                 Set a positive value via .n_quantiles()."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "QuantileTransformer")?;
        let n_quantiles = self.n_quantiles.min(x.height());

        let mut refs = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "QuantileTransformer: column '{}' expected but not found. {}",
                    name, e
                ))
            })?;

            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "QuantileTransformer: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let mut vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "QuantileTransformer: column '{}' has no non-null, finite values. \
                     Cannot compute quantiles for an all-null or all-NaN column. \
                     Impute first or drop the column.",
                    name
                )));
            }

            vals.sort_by(|a, b| a.total_cmp(b));

            let references = compute_quantile_references(&vals, n_quantiles);

            refs.push(QuantileRef {
                name: name.clone(),
                references,
            });
        }

        self.references = Some(refs);
        self.fitted = true;

        Ok(())
    }
}

impl Transform<DataFrame> for QuantileTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "QuantileTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let refs = self.references.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "QuantileTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut out = x.clone();

        let normal = match self.output_distribution {
            OutputDistribution::Normal => Some(Normal::new(0.0, 1.0).map_err(|e| {
                Error::Computation(format!(
                    "QuantileTransformer: failed to create standard normal distribution. {}",
                    e
                ))
            })?),
            OutputDistribution::Uniform => None,
        };

        for qr in refs {
            let n_q = qr.references.len();
            let references = &qr.references;

            if n_q == 0 {
                continue;
            }

            let min_ref = references[0];
            let max_ref = references[n_q - 1];

            replace_f64_column(&mut out, &qr.name, "QuantileTransformer", |v| {
                if v.is_nan() {
                    return f64::NAN;
                }
                let rank = map_value_to_rank(v, min_ref, max_ref, references, n_q);
                match normal {
                    Some(ref dist) => {
                        let clipped = rank.clamp(PROBIT_EPSILON, 1.0 - PROBIT_EPSILON);
                        dist.inverse_cdf(clipped)
                    }
                    None => rank,
                }
            })?;
        }

        Ok(out)
    }
}

/// Map a rank in `[0, 1]` back to a value by interpolating the quantile
/// reference grid in reverse.
///
/// Inverse of [`map_value_to_rank`]: rank `0` maps to the first reference,
/// rank `1` to the last, and intermediate ranks interpolate linearly between
/// neighbouring references. A degenerate grid (all references equal) maps
/// every rank to that value.
fn map_rank_to_value(rank: f64, references: &[f64], n_q: usize) -> f64 {
    if n_q <= 1 {
        return references.first().copied().unwrap_or(f64::NAN);
    }
    let pos = rank.clamp(0.0, 1.0) * (n_q - 1) as f64;
    let lo = pos.floor() as usize;
    if lo + 1 >= n_q {
        return references[n_q - 1];
    }
    let frac = pos - lo as f64;
    references[lo] * (1.0 - frac) + references[lo + 1] * frac
}

impl InverseTransform<DataFrame> for QuantileTransformer {
    type Output = DataFrame;

    /// Undo the quantile transform by interpolating the stored reference grid.
    ///
    /// For [`OutputDistribution::Uniform`] the input is read as a rank in
    /// `[0, 1]`; for [`OutputDistribution::Normal`] it is first mapped back to
    /// a rank through the standard-normal CDF. Ranks outside `[0, 1]` are
    /// clamped, and the result is precision-limited by the reference grid: the
    /// inverse is exact only at the reference points and piecewise-linear
    /// between them.
    fn inverse_transform(&self, x: DataFrame) -> Result<Self::Output> {
        let refs = self.references.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "QuantileTransformer has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            )
        })?;

        let normal = match self.output_distribution {
            OutputDistribution::Normal => Some(Normal::new(0.0, 1.0).map_err(|e| {
                Error::Computation(format!(
                    "QuantileTransformer: failed to create standard normal distribution. {}",
                    e
                ))
            })?),
            OutputDistribution::Uniform => None,
        };

        let mut out = x.clone();

        for qr in refs {
            let n_q = qr.references.len();
            let references = &qr.references;

            if n_q == 0 {
                continue;
            }

            replace_f64_column(&mut out, &qr.name, "QuantileTransformer", |v| {
                if v.is_nan() {
                    return f64::NAN;
                }
                let rank = match normal {
                    Some(ref dist) => dist.cdf(v),
                    None => v,
                };
                map_rank_to_value(rank, references, n_q)
            })?;
        }

        Ok(out)
    }
}

impl HasInverse for QuantileTransformer {}

/// Small epsilon used to clamp uniform ranks before applying probit
/// (inverse normal CDF). `inverse_cdf(0)` = `-inf` and `inverse_cdf(1)` = `+inf`,
/// so we avoid the asymptotes.
const PROBIT_EPSILON: f64 = 1e-15;

/// Compute `n_quantiles` evenly-spaced quantile references from a sorted slice.
///
/// Uses linear interpolation between adjacent values (matching scikit-learn's
/// `np.percentile(..., interpolation='linear')`).
fn compute_quantile_references(sorted: &[f64], n_quantiles: usize) -> Vec<f64> {
    if n_quantiles == 0 {
        return Vec::new();
    }
    if n_quantiles == 1 {
        return vec![percentile_at(sorted, 0.5)];
    }
    let mut refs = Vec::with_capacity(n_quantiles);
    for i in 0..n_quantiles {
        let q = i as f64 / (n_quantiles - 1) as f64;
        refs.push(percentile_at(sorted, q));
    }
    refs
}

/// Compute the `q`-quantile (0 ≤ q ≤ 1) from a sorted slice using linear interpolation.
fn percentile_at(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Whether the span `[lo, hi]` is degenerate, i.e. carries no usable spread.
///
/// Degenerate means the endpoints are equal: there is no spread to rank. Any
/// non-zero gap is real information the input column carried, however small
/// (e.g. `1e-300..5e-300`, span `4e-300`) or however large (e.g.
/// `1e16..1e16 + 2.0`, one representable step). An epsilon test cannot express
/// that, because the same threshold is far too wide for small values and sits
/// below a single ulp at large ones; comparing the endpoints directly is exact
/// at every magnitude. A constant column and an all-zero column both have
/// `lo == hi`, so both stay degenerate and keep their documented mid-value.
fn is_degenerate(lo: f64, hi: f64) -> bool {
    lo == hi
}

/// Map a single value `v` to its rank in `[0, 1]` using the quantile reference grid.
///
/// - A degenerate grid (no usable spread) maps every value to `0.5`.
/// - Values below `min_ref` map to `0.0`.
/// - Values above `max_ref` map to `1.0`.
/// - Values between adjacent references get a linearly interpolated rank.
fn map_value_to_rank(v: f64, min_ref: f64, max_ref: f64, references: &[f64], n_q: usize) -> f64 {
    if is_degenerate(min_ref, max_ref) {
        return 0.5;
    }
    if v <= min_ref {
        return 0.0;
    }
    if v >= max_ref {
        return 1.0;
    }

    let lo = match references.binary_search_by(|r| r.total_cmp(&v)) {
        Ok(i) => {
            return i as f64 / (n_q - 1) as f64;
        }
        Err(i) => {
            if i == 0 {
                return 0.0;
            }
            if i >= n_q {
                return 1.0;
            }
            i - 1
        }
    };

    let hi = lo + 1;

    if hi >= n_q {
        return 1.0;
    }

    let r_lo = references[lo];
    let r_hi = references[hi];
    let span = r_hi - r_lo;

    if is_degenerate(r_lo, r_hi) {
        return lo as f64 / (n_q - 1) as f64;
    }

    let t = (v - r_lo) / span;
    (lo as f64 + t) / (n_q - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[10.0f64, 20.0, 30.0, 40.0, 50.0]));
        DataFrame::new(5, vec![a, b]).unwrap()
    }

    #[test]
    fn test_uniform_output() {
        let mut qt = QuantileTransformer::new();
        let df = make_test_df();
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();

        let vals: Vec<f64> = out
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        assert_eq!(vals.len(), 5);
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.25, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 0.5, epsilon = 1e-6);
        assert_relative_eq!(vals[3], 0.75, epsilon = 1e-6);
        assert_relative_eq!(vals[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normal_output() {
        let mut qt = QuantileTransformer::new().output_distribution(OutputDistribution::Normal);
        let df = make_test_df();
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();

        let vals: Vec<f64> = out
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        assert_eq!(vals.len(), 5);
        // Values should be finite (no -inf/+inf from probit)
        for v in &vals {
            assert!(v.is_finite());
        }
        // Order should be preserved
        for i in 1..vals.len() {
            assert!(vals[i - 1] <= vals[i], "order must be preserved");
        }
    }

    #[test]
    fn test_not_fitted() {
        let qt = QuantileTransformer::new();
        let df = make_test_df();
        let result = qt.transform(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_preservation() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(5.0)]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new().n_quantiles(3);
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
    }

    #[test]
    fn test_nan_preservation() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new().n_quantiles(3);
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(vals[1].unwrap().is_nan(), "NaN input must map to NaN");
    }

    #[test]
    fn test_out_of_range_clipping() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df_fit = DataFrame::new(3, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new();
        qt.fit(df_fit).unwrap();

        let x_new = Column::from(Series::new("x".into(), &[-10.0f64, 100.0]));
        let df_transform = DataFrame::new(2, vec![x_new]).unwrap();
        let out = qt.transform(df_transform).unwrap();
        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_small_dataset() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new().n_quantiles(1000);
        qt.fit(df.clone()).unwrap();

        let out = qt.transform(df).unwrap();
        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        assert_eq!(vals.len(), 2);
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_constant_column() {
        let x = Column::from(Series::new("x".into(), &[5.0f64, 5.0, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new().n_quantiles(3);
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();
        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        // All identical values should map to the same rank
        for v in &vals {
            assert_relative_eq!(*v, 0.5, epsilon = 1e-6);
        }
    }

    fn make_single_col(values: &[f64]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), values));
        DataFrame::new(values.len(), vec![col]).unwrap()
    }

    #[test]
    fn test_small_magnitude_column_not_collapsed() {
        // Genuine spread far below the absolute `f64::EPSILON` threshold must
        // still be ranked, not treated as a constant column.
        let vals = [1e-300, 2e-300, 3e-300, 4e-300, 5e-300];
        let df = make_single_col(&vals);
        let mut qt = QuantileTransformer::new().n_quantiles(5);
        qt.fit(df.clone()).unwrap();
        let out = qt.transform(df).unwrap();
        let got = column_values(&out, "x");
        let want = [0.0, 0.25, 0.5, 0.75, 1.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(*g, *w, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_large_magnitude_column_not_collapsed() {
        // One representable step at a large magnitude is genuine spread: an
        // epsilon threshold would exceed it and collapse the column.
        let df = make_single_col(&[1e16, 1e16 + 2.0]);
        let mut qt = QuantileTransformer::new().n_quantiles(2);
        qt.fit(df.clone()).unwrap();
        let got = column_values(&qt.transform(df).unwrap(), "x");
        assert_relative_eq!(got[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(got[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_zero_only_column_stays_degenerate() {
        // All-zero: span and scale are both 0, so the column is degenerate
        // and keeps the documented mid-value.
        let df = make_single_col(&[0.0, 0.0, 0.0]);
        let mut qt = QuantileTransformer::new().n_quantiles(3);
        qt.fit(df.clone()).unwrap();
        let got = column_values(&qt.transform(df).unwrap(), "x");
        for v in &got {
            assert_relative_eq!(*v, 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let mut qt = QuantileTransformer::new();
        let result = qt.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut qt = QuantileTransformer::new();
        let result = qt.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_quantiles_parameter() {
        // Use an irregularly-spaced dataset where n_quantiles matters.
        let x = Column::from(Series::new(
            "x".into(),
            &[1.0f64, 2.0, 4.0, 8.0, 16.0, 32.0],
        ));
        let df = DataFrame::new(6, vec![x]).unwrap();

        let mut qt_small = QuantileTransformer::new().n_quantiles(3);
        qt_small.fit(df.clone()).unwrap();
        let out_small = qt_small.transform(df.clone()).unwrap();
        let vals_small: Vec<f64> = out_small
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        let mut qt_large = QuantileTransformer::new().n_quantiles(6);
        qt_large.fit(df.clone()).unwrap();
        let out_large = qt_large.transform(df).unwrap();
        let vals_large: Vec<f64> = out_large
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();

        // With 3 quantiles (refs at q=0, 0.5, 1), value 4.0 interpolates
        // between ref[0]=1.0 and ref[1]=6.0 → rank = (0 + 0.6) / 2 = 0.3
        assert_relative_eq!(vals_small[2], 0.3, epsilon = 1e-6);
        // With 6 quantiles (refs at every sample), value 4.0 maps exactly
        // to rank 2/5 = 0.4, which is strictly larger.
        assert_relative_eq!(vals_large[2], 0.4, epsilon = 1e-6);
        assert!(vals_small[2] < vals_large[2]);
    }

    fn column_values(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_inverse_roundtrip_uniform() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();

        let mut qt = QuantileTransformer::new();
        qt.fit(df.clone()).unwrap();
        let transformed = qt.transform(df.clone()).unwrap();
        let restored = qt.inverse_transform(transformed).unwrap();

        let want = column_values(&df, "x");
        let got = column_values(&restored, "x");
        for (w, g) in want.iter().zip(&got) {
            assert_relative_eq!(*w, *g, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_inverse_roundtrip_normal() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();

        let mut qt = QuantileTransformer::new().output_distribution(OutputDistribution::Normal);
        qt.fit(df.clone()).unwrap();
        let transformed = qt.transform(df.clone()).unwrap();
        let restored = qt.inverse_transform(transformed).unwrap();

        let want = column_values(&df, "x");
        let got = column_values(&restored, "x");
        for (w, g) in want.iter().zip(&got) {
            assert_relative_eq!(*w, *g, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_inverse_clamps_ranks_outside_unit_interval() {
        let vals = [1.0, 2.0, 3.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut qt = QuantileTransformer::new();
        qt.fit(df).unwrap();

        let out = Column::from(Series::new("x".into(), &[-5.0f64, 5.0]));
        let ranks = DataFrame::new(2, vec![out]).unwrap();
        let restored = qt.inverse_transform(ranks).unwrap();

        let got = column_values(&restored, "x");
        assert_relative_eq!(got[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(got[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_inverse_before_fit_is_not_fitted() {
        let col = Column::from(Series::new("x".into(), &[0.5f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let err = QuantileTransformer::new()
            .inverse_transform(df)
            .unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "error: {err}");
    }

    #[test]
    fn test_reports_invertible() {
        assert!(QuantileTransformer::new().is_invertible());
    }
}
