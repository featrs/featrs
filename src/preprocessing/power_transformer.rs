//! Power transformations for Gaussian-like feature distributions.
//!
//! Analogous to `sklearn.preprocessing.PowerTransformer`. Supports Yeo-Johnson
//! (any real value) and Box-Cox (strictly positive values only). Per-column
//! optimal lambda is estimated via maximum likelihood (golden-section search),
//! then applied and optionally standardized.
//!
//! # Example
//!
//! ```rust
//! use featrs::preprocessing::power_transformer::{PowerMethod, PowerTransformer};
//! use featrs::traits::{Fit, Transform};
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0, 10.0, 50.0]));
//! let df = DataFrame::new(5, vec![col])?;
//!
//! let mut transformer = PowerTransformer::new().method(PowerMethod::YeoJohnson);
//! transformer.fit(df.clone())?;
//! let result = transformer.transform(df)?;
//! assert_eq!(result.height(), 5);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use polars::prelude::*;

use crate::traits::inverse::{HasInverse, InverseTransform};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// The power transformation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMethod {
    /// Yeo-Johnson: works on positive and negative values.
    YeoJohnson,
    /// Box-Cox: requires strictly positive values.
    BoxCox,
}

/// Lower/upper bounds for the golden-section search over lambda.
const LAMBDA_MIN: f64 = -5.0;
const LAMBDA_MAX: f64 = 5.0;
/// Golden-section search iterations; comfortably converges well past f64
/// precision needs for lambda in `[-5, 5]`.
const GOLDEN_ITERS: usize = 100;

struct PowerParam {
    name: String,
    lambda: f64,
    mean: f64,
    std: f64,
}

/// Apply a power transformation to make data more Gaussian-like.
///
/// For each `Float64` column, estimates the optimal transformation parameter
/// `lambda` via maximum likelihood, applies the Yeo-Johnson or Box-Cox
/// transform, then optionally standardizes the result (zero mean, unit
/// variance), mirroring scikit-learn's default behaviour.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::power_transformer::PowerTransformer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0, 10.0, 50.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut transformer = PowerTransformer::new();
/// transformer.fit(df.clone())?;
/// let result = transformer.transform(df)?;
/// assert_eq!(result.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PowerTransformer {
    fitted: bool,
    method: PowerMethod,
    standardize: bool,
    params: Option<Vec<PowerParam>>,
}

impl PowerTransformer {
    /// Create a new `PowerTransformer` with `YeoJohnson` method and
    /// standardization enabled.
    pub fn new() -> Self {
        Self {
            fitted: false,
            method: PowerMethod::YeoJohnson,
            standardize: true,
            params: None,
        }
    }

    /// Set the power transformation method (default: [`PowerMethod::YeoJohnson`]).
    pub fn method(mut self, m: PowerMethod) -> Self {
        self.method = m;
        self.fitted = false;
        self.params = None;
        self
    }

    /// Whether to standardize (zero mean, unit variance) after transforming
    /// (default: `true`).
    pub fn standardize(mut self, b: bool) -> Self {
        self.standardize = b;
        self.fitted = false;
        self.params = None;
        self
    }
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Box-Cox transform of a single strictly-positive value.
fn box_cox(x: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-10 {
        x.ln()
    } else {
        (x.powf(lambda) - 1.0) / lambda
    }
}

/// Yeo-Johnson transform of a single value (any sign).
fn yeo_johnson(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda.abs() < 1e-10 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-10 {
        -(-x + 1.0).ln()
    } else {
        -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
    }
}

/// Box-Cox inverse of a single value, or `None` when the shifted value would
/// need a non-positive base (`y * lambda + 1 <= 0`).
fn box_cox_inverse(y: f64, lambda: f64) -> Option<f64> {
    if lambda.abs() < 1e-10 {
        return Some(y.exp());
    }
    let base = y * lambda + 1.0;
    (base > 0.0).then(|| base.powf(1.0 / lambda))
}

/// Yeo-Johnson inverse of a single value, or `None` when the shifted value
/// would need a non-positive base. The sign of `y` selects the branch the
/// forward transform used: `x >= 0` always maps to `y >= 0`, and `x < 0`
/// always maps to `y < 0`.
fn yeo_johnson_inverse(y: f64, lambda: f64) -> Option<f64> {
    if y >= 0.0 {
        if lambda.abs() < 1e-10 {
            return Some(y.exp() - 1.0);
        }
        let base = y * lambda + 1.0;
        (base > 0.0).then(|| base.powf(1.0 / lambda) - 1.0)
    } else {
        let two_minus_lambda = 2.0 - lambda;
        if two_minus_lambda.abs() < 1e-10 {
            return Some(1.0 - (-y).exp());
        }
        // Forward: y = -(((1 - x)^(2-λ) - 1) / (2-λ)), so the base to invert is
        // `1 - y * (2-λ)`.
        let base = 1.0 - y * two_minus_lambda;
        (base > 0.0).then(|| 1.0 - base.powf(1.0 / two_minus_lambda))
    }
}

/// Log-likelihood `lambda` under MLE criterion scikit-learn uses:
/// maximize `-n/2 * ln(var(y)) + (lambda - 1) * sum(log_terms)`.
fn log_likelihood(vals: &[f64], lambda: f64, method: PowerMethod) -> f64 {
    let n = vals.len() as f64;
    let transformed: Vec<f64> = vals
        .iter()
        .map(|&v| match method {
            PowerMethod::BoxCox => box_cox(v, lambda),
            PowerMethod::YeoJohnson => yeo_johnson(v, lambda),
        })
        .collect();

    let mean = transformed.iter().sum::<f64>() / n;
    let var = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    if !var.is_finite() || var <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let log_terms: f64 = match method {
        PowerMethod::BoxCox => vals.iter().map(|v| v.ln()).sum(),
        PowerMethod::YeoJohnson => vals.iter().map(|v| v.signum() * (v.abs() + 1.0).ln()).sum(),
    };

    let ll = -0.5 * n * var.ln() + (lambda - 1.0) * log_terms;
    if ll.is_nan() { f64::NEG_INFINITY } else { ll }
}

/// Golden-section search for the lambda in `[LAMBDA_MIN, LAMBDA_MAX]` that
/// maximizes the log-likelihood.
fn optimal_lambda(vals: &[f64], method: PowerMethod) -> f64 {
    let gr: f64 = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618

    let mut a = LAMBDA_MIN;
    let mut b = LAMBDA_MAX;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = log_likelihood(vals, c, method);
    let mut fd = log_likelihood(vals, d, method);

    for _ in 0..GOLDEN_ITERS {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = log_likelihood(vals, c, method);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = log_likelihood(vals, d, method);
        }
    }

    (a + b) / 2.0
}

impl Fit<DataFrame> for PowerTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        self.params = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "PowerTransformer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "PowerTransformer")?;
        let mut params = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "PowerTransformer.fit: column '{name}' not found. {e}"
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "PowerTransformer.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            let vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "PowerTransformer: column '{name}' has no non-null, finite values. \
                     Cannot fit an all-null or all-NaN column. Impute first or drop the column."
                )));
            }

            if self.method == PowerMethod::BoxCox {
                let bad_vals: Vec<f64> = vals.iter().copied().filter(|v| *v <= 0.0).collect();
                if !bad_vals.is_empty() {
                    let show: Vec<String> =
                        bad_vals.iter().take(5).map(|v| format!("{v}")).collect();
                    let extra = bad_vals.len().saturating_sub(5);
                    return Err(Error::InvalidInput(format!(
                        "PowerTransformer: Box-Cox requires strictly positive values, but \
                         column '{name}' contains {} value(s) <= 0: [{}]{}. \
                         Use PowerMethod::YeoJohnson instead, which supports non-positive values.",
                        bad_vals.len(),
                        show.join(", "),
                        if extra > 0 {
                            format!(" ... and {extra} more")
                        } else {
                            String::new()
                        },
                    )));
                }
            }

            let lambda = optimal_lambda(&vals, self.method);

            let transformed: Vec<f64> = vals
                .iter()
                .map(|&v| match self.method {
                    PowerMethod::BoxCox => box_cox(v, lambda),
                    PowerMethod::YeoJohnson => yeo_johnson(v, lambda),
                })
                .collect();

            let (mean, std) = if self.standardize {
                let n = transformed.len() as f64;
                let mean = transformed.iter().sum::<f64>() / n;
                let var = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let std = var.sqrt();
                if std < f64::EPSILON {
                    return Err(Error::Computation(format!(
                        "PowerTransformer: column '{name}' has zero variance after the power \
                         transform (lambda={lambda}). Cannot standardize a constant column. \
                         Try standardize(false)."
                    )));
                }
                (mean, std)
            } else {
                (0.0, 1.0)
            };

            params.push(PowerParam {
                name: name.clone(),
                lambda,
                mean,
                std,
            });
        }

        self.params = Some(params);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for PowerTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "PowerTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "PowerTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut out = x.clone();

        for p in params {
            let lambda = p.lambda;
            let mean = p.mean;
            let std = p.std;
            let method = self.method;

            if method == PowerMethod::BoxCox {
                let s = out.column(p.name.as_str()).map_err(|e| {
                    Error::InvalidInput(format!(
                        "PowerTransformer.transform: column '{}' not found. {e}",
                        p.name
                    ))
                })?;
                let ca = s.f64().map_err(|e| {
                    Error::InvalidInput(format!(
                        "PowerTransformer.transform: column '{}' has dtype {}; expected Float64. {e}",
                        p.name,
                        s.dtype()
                    ))
                })?;
                let bad = ca.iter().flatten().filter(|v| *v <= 0.0).count();
                if bad > 0 {
                    return Err(Error::InvalidInput(format!(
                        "PowerTransformer.transform: Box-Cox requires strictly positive values, \
                         but column '{}' contains {bad} value(s) <= 0.",
                        p.name
                    )));
                }
            }

            replace_f64_column(&mut out, &p.name, "PowerTransformer", move |v| {
                let y = match method {
                    PowerMethod::BoxCox => box_cox(v, lambda),
                    PowerMethod::YeoJohnson => yeo_johnson(v, lambda),
                };
                (y - mean) / std
            })?;
        }

        Ok(out)
    }
}

/// Inverse of a single power-transformed value, dispatching on the method.
fn inverse_power(y: f64, method: PowerMethod, lambda: f64) -> Option<f64> {
    match method {
        PowerMethod::BoxCox => box_cox_inverse(y, lambda),
        PowerMethod::YeoJohnson => yeo_johnson_inverse(y, lambda),
    }
}

impl InverseTransform<DataFrame> for PowerTransformer {
    type Output = DataFrame;

    /// Undo the power transform, and the standardization when it was enabled:
    /// `x = T⁻¹(y * std + mean)` per column.
    ///
    /// Returns [`Error::InvalidInput`] for a value outside the range the
    /// forward transform can produce, i.e. one whose inverse base would be
    /// non-positive.
    fn inverse_transform(&self, x: DataFrame) -> Result<Self::Output> {
        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "PowerTransformer has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            )
        })?;
        let method = self.method;
        let mut out = x.clone();

        for p in params {
            let lambda = p.lambda;
            let mean = p.mean;
            let std = p.std;

            let bad = {
                let s = out.column(p.name.as_str()).map_err(|e| {
                    Error::InvalidInput(format!(
                        "PowerTransformer.inverse_transform: column '{}' not found. {e}",
                        p.name
                    ))
                })?;
                let ca = s.f64().map_err(|e| {
                    Error::InvalidInput(format!(
                        "PowerTransformer.inverse_transform: column '{}' has dtype {}; \
                         expected Float64. {e}",
                        p.name,
                        s.dtype()
                    ))
                })?;
                ca.iter()
                    .flatten()
                    .map(|v| v * std + mean)
                    .find(|t| !t.is_nan() && inverse_power(*t, method, lambda).is_none())
            };

            if let Some(bad) = bad {
                return Err(Error::InvalidInput(format!(
                    "PowerTransformer.inverse_transform: value {bad} in column '{}' is outside \
                     the range of the fitted transform (lambda={lambda}); its inverse is undefined.",
                    p.name
                )));
            }

            replace_f64_column(&mut out, &p.name, "PowerTransformer", move |v| {
                inverse_power(v * std + mean, method, lambda).unwrap_or(f64::NAN)
            })?;
        }

        Ok(out)
    }
}

impl HasInverse for PowerTransformer {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn skewed_positive_df() -> DataFrame {
        let vals: Vec<f64> = (1..=30).map(|i| (i as f64).powi(3)).collect();
        let col = Column::from(Series::new("x".into(), &vals));
        DataFrame::new(vals.len(), vec![col]).unwrap()
    }

    #[test]
    fn test_yeo_johnson_reduces_skew_and_standardizes() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new().method(PowerMethod::YeoJohnson);
        t.fit(df.clone()).unwrap();
        let out = t.transform(df).unwrap();

        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        assert_relative_eq!(mean, 0.0, epsilon = 1e-8);
        assert_relative_eq!(var, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_box_cox_positive_only() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        t.fit(df.clone()).unwrap();
        let out = t.transform(df).unwrap();
        assert_eq!(out.height(), 30);
    }

    #[test]
    fn test_box_cox_rejects_non_positive() {
        let col = Column::from(Series::new("x".into(), &[1.0f64, -2.0, 3.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        let result = t.fit(df);
        assert!(result.is_err(), "Box-Cox must reject non-positive values");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("x"));
    }

    #[test]
    fn test_yeo_johnson_handles_negative_values() {
        let vals: Vec<f64> = (-15..15).map(|i| i as f64).collect();
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();

        let mut t = PowerTransformer::new().method(PowerMethod::YeoJohnson);
        let result = t.fit(df.clone());
        assert!(result.is_ok(), "Yeo-Johnson must accept negative values");
        let out = t.transform(df).unwrap();
        assert_eq!(out.height(), 30);
    }

    #[test]
    fn test_no_standardize() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new()
            .method(PowerMethod::BoxCox)
            .standardize(false);
        t.fit(df.clone()).unwrap();
        let out = t.transform(df).unwrap();

        let params = t.params.as_ref().unwrap();
        assert_relative_eq!(params[0].mean, 0.0);
        assert_relative_eq!(params[0].std, 1.0);
        assert_eq!(out.height(), 30);
    }

    #[test]
    fn test_lambda_one_is_near_identity_shift() {
        // Roughly linear data should push lambda close to 1 for Box-Cox,
        // i.e. y ≈ x - 1 (identity up to the -1/lambda offset).
        let vals: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();

        let mut t = PowerTransformer::new()
            .method(PowerMethod::BoxCox)
            .standardize(false);
        t.fit(df).unwrap();
        let lambda = t.params.as_ref().unwrap()[0].lambda;
        assert!(
            (0.0..=2.5).contains(&lambda),
            "expected lambda in a sane range for near-linear data, got {lambda}"
        );
    }

    #[test]
    fn test_box_cox_transform_rejects_non_positive() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new()
            .method(PowerMethod::BoxCox)
            .standardize(false);
        t.fit(df).unwrap();

        let bad = Column::from(Series::new("x".into(), &[1.0f64, -2.0, 3.0]));
        let bad_df = DataFrame::new(3, vec![bad]).unwrap();
        let result = t.transform(bad_df);
        assert!(
            result.is_err(),
            "transform must reject non-positive Box-Cox input, not silently emit NaN/-Inf"
        );
    }

    #[test]
    fn test_reconfiguring_after_fit_clears_fitted_state() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new().method(PowerMethod::YeoJohnson);
        t.fit(df.clone()).unwrap();

        let t2 = t.method(PowerMethod::BoxCox);
        let result = t2.transform(df);
        assert!(
            result.is_err(),
            "rebinding method() after fit must invalidate fitted state"
        );
    }

    #[test]
    fn test_null_preservation() {
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0f64), None, Some(2.0), Some(3.0), Some(4.0)],
        ));
        let df = DataFrame::new(5, vec![x]).unwrap();

        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        t.fit(df.clone()).unwrap();
        let out = t.transform(df).unwrap();

        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
    }

    #[test]
    fn test_not_fitted_error() {
        let t = PowerTransformer::new();
        let col = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();
        let result = t.transform(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_dataframe_error() {
        let df = DataFrame::empty();
        let mut t = PowerTransformer::new();
        let result = t.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut t = PowerTransformer::new();
        let result = t.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_float64_columns_error() {
        let a = Column::from(Series::new("a".into(), &[1i64, 2, 3]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let mut t = PowerTransformer::new();
        let result = t.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_is_yeo_johnson_standardized() {
        let t = PowerTransformer::default();
        assert_eq!(t.method, PowerMethod::YeoJohnson);
        assert!(t.standardize);
    }

    #[test]
    fn test_constant_column_errors_when_standardized() {
        let col = Column::from(Series::new("x".into(), &[5.0f64, 5.0, 5.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        let result = t.fit(df);
        assert!(
            result.is_err(),
            "constant column must error when standardize=true (zero variance)"
        );
    }

    #[test]
    fn test_multiple_columns_independent_lambdas() {
        let a = Column::from(Series::new(
            "a".into(),
            &(1..=20).map(|i| (i as f64).powi(2)).collect::<Vec<_>>(),
        ));
        let b = Column::from(Series::new(
            "b".into(),
            &(1..=20).map(|i| i as f64).collect::<Vec<_>>(),
        ));
        let df = DataFrame::new(20, vec![a, b]).unwrap();

        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        t.fit(df.clone()).unwrap();
        let out = t.transform(df).unwrap();
        assert_eq!(out.width(), 2);

        let params = t.params.as_ref().unwrap();
        assert_ne!(
            params[0].lambda, params[1].lambda,
            "different distributions should generally yield different lambdas"
        );
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

    fn assert_roundtrip(df: &DataFrame, method: PowerMethod, standardize: bool, eps: f64) {
        let mut t = PowerTransformer::new()
            .method(method)
            .standardize(standardize);
        t.fit(df.clone()).unwrap();
        let transformed = t.transform(df.clone()).unwrap();
        let restored = t.inverse_transform(transformed).unwrap();

        let want = column_values(df, "x");
        let got = column_values(&restored, "x");
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            assert_relative_eq!(*w, *g, epsilon = eps);
        }
    }

    #[test]
    fn test_box_cox_inverse_roundtrip() {
        assert_roundtrip(&skewed_positive_df(), PowerMethod::BoxCox, true, 1e-8);
    }

    #[test]
    fn test_yeo_johnson_inverse_roundtrip() {
        assert_roundtrip(&skewed_positive_df(), PowerMethod::YeoJohnson, true, 1e-8);
    }

    #[test]
    fn test_power_inverse_roundtrip_without_standardize() {
        assert_roundtrip(&skewed_positive_df(), PowerMethod::YeoJohnson, false, 1e-8);
    }

    #[test]
    fn test_yeo_johnson_inverse_roundtrip_negative_values() {
        let vals: Vec<f64> = (-15..15).map(|i| i as f64).collect();
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();

        assert_roundtrip(&df, PowerMethod::YeoJohnson, true, 1e-8);
    }

    #[test]
    fn test_inverse_rejects_out_of_range_value() {
        let df = skewed_positive_df();
        let mut t = PowerTransformer::new().method(PowerMethod::BoxCox);
        t.fit(df).unwrap();

        // Far outside the Box-Cox output range: the inverse base becomes
        // non-positive, which is undefined.
        let bad = Column::from(Series::new("x".into(), &[-1e6f64]));
        let out = DataFrame::new(1, vec![bad]).unwrap();

        let err = t.inverse_transform(out).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "error: {err}");
    }

    #[test]
    fn test_inverse_before_fit_is_not_fitted() {
        let err = PowerTransformer::new()
            .inverse_transform(skewed_positive_df())
            .unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "error: {err}");
    }

    #[test]
    fn test_reports_invertible() {
        assert!(PowerTransformer::new().is_invertible());
    }
}
