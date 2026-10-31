//! Logarithmic transformations for numeric columns.
//!
//! Analogous to `numpy.log`, `numpy.log1p`, `numpy.log2`, `numpy.log10`.
//! Supports auto-discovery of `Float64` columns, configurable positive-value
//! checks, and null preservation.
//!
//! # Example
//!
//! ```rust
//! use featrs::preprocessing::log_transformer::{LogMethod, LogTransformer};
//! use featrs::traits::{Fit, Transform};
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![col])?;
//!
//! let mut transformer = LogTransformer::new().method(LogMethod::Log10);
//! transformer.fit(df.clone())?;
//! let result = transformer.transform(df)?;
//! assert_eq!(result.height(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use polars::prelude::*;

use crate::traits::inverse::{HasInverse, InverseTransform};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{numeric_f64_columns, replace_f64_column};

/// The logarithmic transformation to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogMethod {
    /// Natural logarithm `ln(x)`. Requires `x > 0`.
    Ln,
    /// Natural logarithm of `x + 1`: `ln(1 + x)`. Requires `x > -1`.
    /// Safe at zero: `ln(1) = 0`, identity-preserving.
    Log1p,
    /// Base-2 logarithm `log₂(x)`. Requires `x > 0`.
    Log2,
    /// Base-10 logarithm `log₁₀(x)`. Requires `x > 0`.
    Log10,
    /// Logarithm with arbitrary base `b`: `log_b(x) = ln(x) / ln(b)`.
    /// Requires `x > 0` and `b > 0`, `b ≠ 1`, `b` finite.
    LogBase(f64),
}

/// Apply logarithmic transformations to numeric columns.
///
/// Supports `ln`, `log1p` (identity-preserving at zero), `log2`, `log10`,
/// and arbitrary-base logarithms. Useful for compressing right-skewed
/// distributions and stabilizing variance.
///
/// # Behaviour by method
///
/// | Method | Domain | Inverse |
/// |--------|--------|---------|
/// | `Ln`   | `x > 0` | `exp(x)` |
/// | `Log1p` | `x > -1` | `expm1(x)` |
/// | `Log2` | `x > 0` | `pow(2, x)` |
/// | `Log10` | `x > 0` | `pow(10, x)` |
/// | `LogBase(b)` | `x > 0, b > 0, b ≠ 1` | `pow(b, x)` |
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::log_transformer::LogTransformer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut transformer = LogTransformer::new();
/// transformer.fit(df.clone())?;
/// let result = transformer.transform(df)?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LogTransformer {
    fitted: bool,
    columns: Option<Vec<String>>,
    fitted_columns: Vec<String>,
    method: LogMethod,
    check_positive: bool,
}

impl LogTransformer {
    /// Create a new `LogTransformer` with default `Log1p` method and
    /// positive-value checking enabled.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: None,
            fitted_columns: Vec::new(),
            method: LogMethod::Log1p,
            check_positive: true,
        }
    }

    /// Restrict transformation to the named columns.
    ///
    /// When omitted, the transformer auto-discovers all `Float64` columns
    /// at [`fit`](Fit::fit) time.
    ///
    /// Each column must exist in the frame passed to `fit` and have dtype
    /// `Float64`; otherwise `fit` returns [`Error::InvalidInput`].
    ///
    /// Calling this after [`fit`](Fit::fit) clears the fitted state, so the
    /// transformer must be refit before use.
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = Some(cols.iter().map(|s| s.to_string()).collect());
        self.reset_fit();
        self
    }

    /// Set the logarithmic method (default: [`LogMethod::Log1p`]).
    ///
    /// Calling this after [`fit`](Fit::fit) clears the fitted state: the domain
    /// check ran for the old method, so the transformer must be refit before
    /// use.
    pub fn method(mut self, m: LogMethod) -> Self {
        self.method = m;
        self.reset_fit();
        self
    }

    /// Whether to validate the input domain during [`fit`](Fit::fit).
    ///
    /// When `true` (default), [`fit`](Fit::fit) errors if any non-null value
    /// violates the positivity requirement (`x > 0` for `Ln`/`Log2`/`Log10`/
    /// `LogBase`; `x > -1` for `Log1p`).
    ///
    /// When `false`, no validation is performed — out-of-domain values produce
    /// `NaN`/`-Inf` at [`transform`](Transform::transform) time via standard
    /// `f64` semantics.
    ///
    /// Calling this after [`fit`](Fit::fit) clears the fitted state, so the
    /// transformer must be refit before use.
    pub fn check_positive(mut self, b: bool) -> Self {
        self.check_positive = b;
        self.reset_fit();
        self
    }

    /// Drop the fitted state so a changed configuration cannot reuse columns
    /// validated under the previous one.
    fn reset_fit(&mut self) {
        self.fitted = false;
        self.fitted_columns.clear();
    }
}

impl Default for LogTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for LogTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state up front so a failed (re)fit cannot leave a stale
        // column plan or fitted flag behind, matching the builder-reset rule.
        self.reset_fit();

        // 1. Validate LogBase at construction/config time
        if let LogMethod::LogBase(b) = self.method
            && (!b.is_finite() || b <= 0.0 || b == 1.0)
        {
            return Err(Error::InvalidInput(format!(
                "LogTransformer: invalid log base {b}. \
                 Base must be finite, positive, and not equal to 1."
            )));
        }

        // 2. Validate non-empty
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "LogTransformer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        // 3. Resolve columns
        let resolved = match &self.columns {
            None => {
                let discovered = numeric_f64_columns(&x);
                if discovered.is_empty() {
                    let all_types: Vec<String> = x
                        .get_column_names()
                        .iter()
                        .filter_map(|n| x.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
                        .collect();
                    return Err(Error::InvalidInput(format!(
                        "LogTransformer: no Float64 columns found. \
                         This transformer only operates on f64 columns. \
                         Available columns: [{}].",
                        all_types.join(", ")
                    )));
                }
                discovered
            }
            Some(cols) => {
                if cols.is_empty() {
                    return Err(Error::InvalidInput(
                        "LogTransformer.fit: columns list is empty. \
                         Omit .columns() for auto-discovery or provide at least one column name."
                            .into(),
                    ));
                }
                for col in cols {
                    let c = x.column(col.as_str()).map_err(|e| {
                        Error::InvalidInput(format!(
                            "LogTransformer.fit: column '{col}' not found. {e}"
                        ))
                    })?;
                    if c.dtype() != &DataType::Float64 {
                        return Err(Error::InvalidInput(format!(
                            "LogTransformer.fit: column '{col}' has dtype {}; expected Float64.",
                            c.dtype()
                        )));
                    }
                }
                cols.clone()
            }
        };

        // 4. All-null / all-NaN check — consistent with scaler pattern (#35)
        let mut bad_cols: Vec<&str> = Vec::new();
        for name in &resolved {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "LogTransformer.fit: column '{name}' expected but not found. {e}"
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "LogTransformer.fit: column '{name}' has unexpected dtype. {e}"
                ))
            })?;
            let has_non_nan = ca.iter().flatten().any(|v| !v.is_nan());
            if !has_non_nan {
                bad_cols.push(name.as_str());
            }
        }
        if !bad_cols.is_empty() {
            return Err(Error::Computation(format!(
                "LogTransformer: column(s) [{}] have no non-null, non-NaN values. \
                 Cannot validate or transform columns with no usable data. \
                 Impute first or drop the column(s).",
                bad_cols.join(", ")
            )));
        }

        // 5. Positive-value check
        if self.check_positive {
            let (threshold, bound, condition) = match self.method {
                LogMethod::Log1p => (-1.0, "-1", "x > -1"),
                _ => (0.0, "0", "x > 0"),
            };

            let mut violations: Vec<String> = Vec::new();
            for name in &resolved {
                let s = x.column(name.as_str()).map_err(|e| {
                    Error::InvalidInput(format!(
                        "LogTransformer.fit: column '{name}' expected but not found. {e}"
                    ))
                })?;
                let ca = s.f64().map_err(|e| {
                    Error::InvalidInput(format!(
                        "LogTransformer.fit: column '{name}' has unexpected dtype. {e}"
                    ))
                })?;
                let bad_vals: Vec<f64> = ca.iter().flatten().filter(|v| *v <= threshold).collect();
                if !bad_vals.is_empty() {
                    let show: Vec<String> =
                        bad_vals.iter().take(5).map(|v| format!("{v}")).collect();
                    let extra = bad_vals.len().saturating_sub(5);
                    violations.push(format!(
                        "column '{name}' contains {} value(s) ≤ {} \
                         (requires {}): [{}]{}",
                        bad_vals.len(),
                        bound,
                        condition,
                        show.join(", "),
                        if extra > 0 {
                            format!(" ... and {extra} more")
                        } else {
                            String::new()
                        },
                    ));
                }
            }

            if !violations.is_empty() {
                return Err(Error::InvalidInput(format!(
                    "LogTransformer.fit: {} column(s) violate the positivity requirement:\n  {}",
                    violations.len(),
                    violations.join("\n  "),
                )));
            }
        }

        self.fitted_columns = resolved;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for LogTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LogTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();

        for name in &self.fitted_columns {
            replace_f64_column(&mut out, name, "LogTransformer", |v| match self.method {
                LogMethod::Ln => v.ln(),
                LogMethod::Log1p => v.ln_1p(),
                LogMethod::Log2 => v.log2(),
                LogMethod::Log10 => v.log10(),
                LogMethod::LogBase(b) => v.log(b),
            })?;
        }

        Ok(out)
    }
}

impl InverseTransform<DataFrame> for LogTransformer {
    type Output = DataFrame;

    /// Undo the logarithm on the columns learned at fit time.
    ///
    /// | Method | Inverse |
    /// |--------|---------|
    /// | `Ln` | `exp(y)` |
    /// | `Log1p` | `expm1(y)` |
    /// | `Log2` | `2^y` |
    /// | `Log10` | `10^y` |
    /// | `LogBase(b)` | `b^y` |
    ///
    /// Any real input is inside a logarithm's range, so no value can be
    /// rejected here; extreme inputs simply overflow to `inf` per `f64`
    /// semantics.
    fn inverse_transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LogTransformer has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();

        for name in &self.fitted_columns {
            replace_f64_column(&mut out, name, "LogTransformer", |v| match self.method {
                LogMethod::Ln => v.exp(),
                LogMethod::Log1p => v.exp_m1(),
                LogMethod::Log2 => 2.0_f64.powf(v),
                LogMethod::Log10 => 10.0_f64.powf(v),
                LogMethod::LogBase(b) => b.powf(v),
            })?;
        }

        Ok(out)
    }
}

impl HasInverse for LogTransformer {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::E;

    #[test]
    fn test_builder_after_fit_requires_refit() {
        let col = Column::from(Series::new("x".into(), &[0.0f64, 1.0, 2.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let fitted = || {
            let mut t = LogTransformer::new();
            t.fit(df.clone()).unwrap();
            t
        };

        // `0.0` is valid for Log1p but not for Ln: switching the method must
        // not reuse the Log1p validation and produce `-inf`.
        let mut ln = fitted().method(LogMethod::Ln);
        assert!(matches!(ln.transform(df.clone()), Err(Error::NotFitted(_))));
        assert!(matches!(
            ln.inverse_transform(df.clone()),
            Err(Error::NotFitted(_))
        ));
        assert!(matches!(ln.fit(df.clone()), Err(Error::InvalidInput(_))));

        for t in [fitted().columns(&["x"]), fitted().check_positive(false)] {
            assert!(matches!(t.transform(df.clone()), Err(Error::NotFitted(_))));
        }
    }

    #[test]
    fn test_log1p_values() {
        let vals = [0.0, 1.0, 2.0, 3.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut transformer = LogTransformer::new();
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        assert_eq!(out.len(), 4);
        assert_relative_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], (2.0f64).ln(), epsilon = 1e-12);
        assert_relative_eq!(out[2], (3.0f64).ln(), epsilon = 1e-12);
        assert_relative_eq!(out[3], (4.0f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_ln_values() {
        let col = Column::from(Series::new("x".into(), &[1.0f64, E]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_log10_values() {
        let col = Column::from(Series::new("x".into(), &[1.0, 10.0, 100.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Log10);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_log2_values() {
        let col = Column::from(Series::new("x".into(), &[1.0, 2.0, 4.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Log2);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_log_base_3() {
        let col = Column::from(Series::new("x".into(), &[3.0, 9.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(3.0));
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(out[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_check_positive_errors_on_violations() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 0.0, -2.0]));
        let y = Column::from(Series::new("y".into(), &[-1.0f64, 5.0, 3.0]));
        let df = DataFrame::new(3, vec![x, y]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        let result = transformer.fit(df);
        assert!(
            result.is_err(),
            "check_positive must reject non-positive values"
        );
        let err = result.unwrap_err().to_string();
        assert!(err.contains("x"), "error must mention column 'x'");
        assert!(err.contains("y"), "error must mention column 'y'");
    }

    #[test]
    fn test_log1p_boundary_negative_ok() {
        // Log1p with x > -1 is valid
        let col = Column::from(Series::new("x".into(), &[-0.5f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Log1p);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let val = ca.get(0).unwrap();
        assert_relative_eq!(val, (0.5f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_log1p_exactly_negative_one_with_check_positive() {
        let col = Column::from(Series::new("x".into(), &[-1.0f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Log1p);
        let result = transformer.fit(df);
        assert!(
            result.is_err(),
            "Log1p with x = -1 must error when check_positive=true"
        );
    }

    #[test]
    fn test_check_positive_false_allows_negative() {
        let col = Column::from(Series::new("x".into(), &[-1.0f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let mut transformer = LogTransformer::new()
            .method(LogMethod::Ln)
            .check_positive(false);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let val = ca.get(0).unwrap();
        assert!(
            val.is_nan(),
            "Ln(-1) with check_positive=false must produce NaN"
        );
    }

    #[test]
    fn test_zero_with_ln_produces_neg_inf() {
        let col = Column::from(Series::new("x".into(), &[0.0f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let mut transformer = LogTransformer::new()
            .method(LogMethod::Ln)
            .check_positive(false);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let val = ca.get(0).unwrap();
        assert_eq!(val, f64::NEG_INFINITY, "Ln(0) must produce -Inf");
    }

    #[test]
    fn test_null_preservation() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(E)]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_propagation() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, E]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert!(
            vals[1].unwrap().is_nan(),
            "NaN input must propagate to NaN output"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_infinity_propagation() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::INFINITY]));
        let df = DataFrame::new(2, vec![x]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert_eq!(
            vals[1].unwrap(),
            f64::INFINITY,
            "+Inf input must map to +Inf output"
        );
    }

    #[test]
    fn test_auto_discovery() {
        let a = Column::from(Series::new("f64_col".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("i64_col".into(), &[1i64, 2, 3]));
        let c = Column::from(Series::new("str_col".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![a, b, c]).unwrap();

        let mut transformer = LogTransformer::new();
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        // Only f64_col was transformed (default Log1p)
        let ca = result.column("f64_col").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(vals[0], (2.0f64).ln(), epsilon = 1e-12);
        assert_relative_eq!(vals[1], (3.0f64).ln(), epsilon = 1e-12);
        assert_relative_eq!(vals[2], (4.0f64).ln(), epsilon = 1e-12);

        // Other columns are untouched
        assert_eq!(result.column("i64_col").unwrap().dtype(), &DataType::Int64);
        assert_eq!(result.column("str_col").unwrap().dtype(), &DataType::String);
    }

    #[test]
    fn test_explicit_columns_subset() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[10.0f64, 100.0, 1000.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut transformer = LogTransformer::new()
            .method(LogMethod::Log10)
            .columns(&["b"]);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        // "b" transformed: [1, 2, 3]
        let cb = result.column("b").unwrap().f64().unwrap();
        let vb: Vec<f64> = cb.iter().flatten().collect();
        assert_relative_eq!(vb[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vb[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(vb[2], 3.0, epsilon = 1e-12);

        // "a" unchanged (1.0, 2.0, 3.0 — not log transformed)
        let ca = result.column("a").unwrap().f64().unwrap();
        let va: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(va[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(va[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(va[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_user_column_not_found() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().columns(&["nonexistent"]);
        let result = transformer.fit(df);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent"));
    }

    #[test]
    fn test_user_column_wrong_dtype() {
        let a = Column::from(Series::new("a".into(), &[1i64, 2, 3]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().columns(&["a"]);
        let result = transformer.fit(df);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("expected Float64"),
            "error must mention expected dtype Float64, got: {err}"
        );
    }

    #[test]
    fn test_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new();
        let result = transformer.fit(df);
        assert!(result.is_err(), "fitting an all-null column must error");
    }

    #[test]
    fn test_all_nan_column_error() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, f64::NAN]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new();
        let result = transformer.fit(df);
        assert!(result.is_err(), "fitting an all-NaN column must error");
    }

    #[test]
    fn test_empty_dataframe_error() {
        let df = DataFrame::empty();
        let mut transformer = LogTransformer::new();
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_columns_list_error() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().columns(&[]);
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_fitted_error() {
        let transformer = LogTransformer::new();
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();
        let result = transformer.transform(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_log_base_zero() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(0.0));
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_log_base_one() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(1.0));
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_log_base_negative() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(-2.0));
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_log_base_nan() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(f64::NAN));
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_log_base_infinite() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let df = DataFrame::new(1, vec![a]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::LogBase(f64::INFINITY));
        let result = transformer.fit(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_column() {
        let vals = [5.0f64, 5.0, 5.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Log10);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let out: Vec<f64> = ca.iter().flatten().collect();
        let expected = 5.0f64.log10();
        assert_relative_eq!(out[0], expected, epsilon = 1e-12);
        assert_relative_eq!(out[1], expected, epsilon = 1e-12);
        assert_relative_eq!(out[2], expected, epsilon = 1e-12);
    }

    #[test]
    fn test_default_is_log1p() {
        // fit with zero (Log1p safe at 0)
        let col = Column::from(Series::new("x".into(), &[0.0f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();
        let mut t = LogTransformer::default();
        let result = t.fit(df);
        assert!(
            result.is_ok(),
            "default LogTransformer must be Log1p (safe at zero)"
        );
    }

    #[test]
    fn test_tiny_positive_values() {
        let col = Column::from(Series::new("x".into(), &[1e-300f64]));
        let df = DataFrame::new(1, vec![col]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let val = ca.get(0).unwrap();
        assert!(
            val < 0.0,
            "ln(tiny positive) must be large negative, got {val}"
        );
    }

    #[test]
    fn test_partial_nan_mixed_ok() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert!(
            vals[1].unwrap().is_nan(),
            "NaN input must map to NaN through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 5.0f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_partial_null_mixed_ok() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(E)]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut transformer = LogTransformer::new().method(LogMethod::Ln);
        transformer.fit(df.clone()).unwrap();
        let result = transformer.transform(df).unwrap();

        let ca = result.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_no_float64_columns_error() {
        let a = Column::from(Series::new("a".into(), &[1i64, 2, 3]));
        let b = Column::from(Series::new("b".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut transformer = LogTransformer::new();
        let result = transformer.fit(df);
        assert!(result.is_err());
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
    fn test_inverse_roundtrip_every_method() {
        let vals = [0.5, 1.0, 2.0, 7.0];
        let col = Column::from(Series::new("x".into(), &vals));
        let df = DataFrame::new(4, vec![col]).unwrap();

        let methods = [
            LogMethod::Ln,
            LogMethod::Log1p,
            LogMethod::Log2,
            LogMethod::Log10,
            LogMethod::LogBase(3.0),
        ];

        for method in methods {
            let mut transformer = LogTransformer::new().method(method);
            transformer.fit(df.clone()).unwrap();
            let logged = transformer.transform(df.clone()).unwrap();
            let restored = transformer.inverse_transform(logged).unwrap();

            let want = column_values(&df, "x");
            let got = column_values(&restored, "x");
            for (w, g) in want.iter().zip(&got) {
                assert_relative_eq!(*w, *g, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_inverse_log1p_maps_zero_to_zero() {
        let col = Column::from(Series::new("x".into(), &[0.0f64, 3.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut transformer = LogTransformer::new();
        transformer.fit(df.clone()).unwrap();
        let restored = transformer
            .inverse_transform(transformer.transform(df).unwrap())
            .unwrap();

        assert_relative_eq!(column_values(&restored, "x")[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_inverse_only_touches_fitted_columns() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let other = Column::from(Series::new("other".into(), &[5.0f64, 6.0]));
        let df = DataFrame::new(2, vec![x, other]).unwrap();

        let mut transformer = LogTransformer::new().columns(&["x"]);
        transformer.fit(df.clone()).unwrap();
        let restored = transformer.inverse_transform(df).unwrap();

        assert_eq!(column_values(&restored, "other"), vec![5.0, 6.0]);
    }

    #[test]
    fn test_inverse_before_fit_is_not_fitted() {
        let col = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let err = LogTransformer::new().inverse_transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "error: {err}");
    }

    #[test]
    fn test_reports_invertible() {
        assert!(LogTransformer::new().is_invertible());
    }
}
