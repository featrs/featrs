//! Maximum-absolute-value scaling.
//!
//! [`MaxAbsScaler`] scales each feature by its maximum absolute value,
//! mapping values into `[-1, 1]` without centering, so sparsity is preserved
//! (zeros remain zeros). Analogous to `sklearn.preprocessing.MaxAbsScaler`.

use polars::prelude::*;

use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// Scale features by their maximum absolute value into `[-1, 1]`.
///
/// Each column `x` is scaled as `x / max(|x|)` where `max(|x|)` is learned
/// from the training data. Unlike `StandardScaler`, no centering is applied,
/// so zero entries stay zero and sparsity is preserved — suitable for sparse
/// data, text feature vectors, and any scenario where zero entries matter.
///
/// Only `Float64` columns are scaled; columns of other dtypes are passed
/// through unchanged. Values in out-of-sample data may fall outside `[-1, 1]`
/// (they are divided by the same training-time maximum). A column that is
/// entirely zero at fit time has `max(|x|) = 0.0` and is left unchanged by
/// transform (division by zero is undefined). Nulls and `NaN` values are
/// preserved: `NaN` is ignored when learning the maximum, and `NaN` input
/// maps to `NaN` output. If a fitted column is missing or no longer `Float64`
/// in the frame passed to [`transform`](Transform::transform), transform
/// returns [`Error::InvalidInput`].
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::max_abs_scaler::MaxAbsScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, -2.0]));
/// let b = Column::from(Series::new("b".into(), &[0.5_f64, 1.0]));
/// let df = DataFrame::new(2, vec![a, b])?;
///
/// let mut scaler = MaxAbsScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MaxAbsScaler {
    fitted: bool,
    max_abs: Option<Vec<f64>>,
    column_names: Option<Vec<String>>,
}

impl MaxAbsScaler {
    /// Create a new `MaxAbsScaler`.
    ///
    /// All `Float64` columns are scaled by their per-column maximum absolute
    /// value; columns of other dtypes are passed through unchanged.
    pub fn new() -> Self {
        Self {
            fitted: false,
            max_abs: None,
            column_names: None,
        }
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for MaxAbsScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale parameters.
        self.fitted = false;
        self.max_abs = None;
        self.column_names = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "MaxAbsScaler.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "MaxAbsScaler")?;

        let mut max_abs = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!("MaxAbsScaler.fit: column '{name}' not found. {e}"))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "MaxAbsScaler.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            // Non-null, finite values only: NaN and ±Inf must not poison the
            // learned maximum (NaN-checked aggregation, cf. issue #35).
            let vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "MaxAbsScaler: column '{name}' has no non-null, finite values. \
                     Cannot scale an all-null or all-NaN column. Impute first or drop the column."
                )));
            }

            let col_max_abs = vals.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            if !col_max_abs.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "MaxAbsScaler: column '{name}' max absolute value is not finite."
                )));
            }

            max_abs.push(col_max_abs);
        }

        self.max_abs = Some(max_abs);
        self.column_names = Some(col_names);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for MaxAbsScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "MaxAbsScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let names = self.column_names.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "MaxAbsScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let max_abs = self.max_abs.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "MaxAbsScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut out = x.clone();
        for (name, scale) in names.iter().zip(max_abs) {
            replace_f64_column(&mut out, name, "MaxAbsScaler", |v| {
                if *scale == 0.0 {
                    // All-zero column at fit time: division by zero is
                    // undefined, so the value is passed through unchanged.
                    v
                } else {
                    v / *scale
                }
            })?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, -2.0]));
        let b = Column::from(Series::new("b".into(), &[0.5f64, 1.0]));
        DataFrame::new(2, vec![a, b]).unwrap()
    }

    #[test]
    fn test_basic_fit_transform() {
        let mut scaler = MaxAbsScaler::new();
        let df = make_test_df();

        scaler.fit(df.clone()).unwrap();
        let result = scaler.transform(df).unwrap();

        // max_abs = [2.0, 1.0] → output [[0.5, -1.0], [0.5, 1.0]].
        // (The issue #47 example numbers are internally inconsistent: column
        // "a" = [1.0, -2.0] has max-abs 2.0, so its scaled values are
        // [0.5, -1.0] — no per-column divisor could yield [1.0, -1.0].)
        let a_vals: Vec<f64> = result
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(a_vals[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(a_vals[1], -1.0, epsilon = 1e-12);

        let b_vals: Vec<f64> = result
            .column("b")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(b_vals[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(b_vals[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_negative_values_scale_to_negative_one() {
        let col = Column::from(Series::new("x".into(), &[-5.0f64, 3.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.6, epsilon = 1e-12);
    }

    #[test]
    fn test_zeros_preserved() {
        let col = Column::from(Series::new("x".into(), &[0.0f64, 2.0, -4.0, 0.0]));
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(vals[2], -1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_all_zero_column_unchanged() {
        let a = Column::from(Series::new("zero".into(), &[0.0f64, 0.0, 0.0]));
        let b = Column::from(Series::new("x".into(), &[2.0f64, 4.0, 8.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let zero_vals: Vec<f64> = out
            .column("zero")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(zero_vals, vec![0.0, 0.0, 0.0]);

        let x_vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(x_vals[0], 0.25, epsilon = 1e-12);
        assert_relative_eq!(x_vals[1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(x_vals[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_zero_scale_column_still_validates_dtype_at_transform() {
        let zero = Column::from(Series::new("x".into(), &[0.0f64, 0.0, 0.0]));
        let df = DataFrame::new(3, vec![zero]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df).unwrap();

        // The fitted column is re-declared as Int64 at transform time: the
        // zero-scale pass-through must still validate the dtype and error.
        let int_col = Column::from(Series::new("x".into(), &[1i64, 2, 3]));
        let int_df = DataFrame::new(3, vec![int_col]).unwrap();
        assert!(scaler.transform(int_df).is_err());
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let scaler = MaxAbsScaler::new();
        let df = make_test_df();
        assert!(scaler.transform(df).is_err());
    }

    #[test]
    fn test_fit_empty_dataframe_errors() {
        let mut scaler = MaxAbsScaler::new();
        assert!(scaler.fit(DataFrame::empty()).is_err());
    }

    #[test]
    fn test_fit_zero_columns_errors() {
        // 0 columns with > 0 rows must hit the width guard independently.
        let df = DataFrame::new(5, Vec::<Column>::new()).unwrap();
        let mut scaler = MaxAbsScaler::new();
        assert!(scaler.fit(df).is_err());
    }

    #[test]
    fn test_null_preserved() {
        let col = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(-4.0)]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let vals: Vec<Option<f64>> = out.column("x").unwrap().f64().unwrap().iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.25, epsilon = 1e-12);
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_ignored_in_fit_and_preserved_in_transform() {
        let col = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, -4.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        // max_abs = 4.0 — the NaN is ignored when learning the maximum.
        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.25, epsilon = 1e-12);
        assert!(
            vals[1].is_nan(),
            "NaN input must map to NaN through transform"
        );
        assert_relative_eq!(vals[2], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_all_null_column_errors_at_fit() {
        let col = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let fit_result = scaler.fit(df);
        assert!(
            fit_result.is_err(),
            "fitting an all-null column must error, not silently fit NaN params"
        );
    }

    #[test]
    fn test_all_nan_column_errors_at_fit() {
        let col = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, f64::NAN]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let fit_result = scaler.fit(df);
        assert!(
            fit_result.is_err(),
            "fitting an all-NaN column must error, not silently fit NaN params"
        );
    }

    #[test]
    fn test_all_inf_column_errors_at_fit() {
        let col = Column::from(Series::new("x".into(), &[f64::INFINITY; 3]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let fit_result = scaler.fit(df);
        assert!(
            fit_result.is_err(),
            "fitting an all-±Inf column must error (no finite values to learn from)"
        );
    }

    #[test]
    fn test_non_f64_columns_passed_through() {
        let a = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let s = Column::from(Series::new("label".into(), &["a", "b"]));
        let df = DataFrame::new(2, vec![a, s]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        assert_eq!(out.width(), 2);
        let labels: Vec<&str> = out
            .column("label")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(labels, vec!["a", "b"]);

        let x_vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(x_vals[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(x_vals[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_transform_missing_fitted_column_errors() {
        let a = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();

        // The transform-time frame no longer has the fitted column "x".
        let other = Column::from(Series::new("y".into(), &[1.0f64, 2.0]));
        let other_df = DataFrame::new(2, vec![other]).unwrap();
        assert!(scaler.transform(other_df).is_err());
    }

    #[test]
    fn test_single_column_dataframe() {
        let col = Column::from(Series::new("x".into(), &[3.0f64, -6.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let vals: Vec<f64> = out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(vals[1], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_failed_refit_resets_state() {
        let mut scaler = MaxAbsScaler::new();
        let df = make_test_df();

        scaler.fit(df.clone()).unwrap();
        // A failed re-fit must not leave stale fitted state behind: the
        // transformer must report NotFitted afterwards.
        assert!(scaler.fit(DataFrame::empty()).is_err());
        assert!(
            scaler.transform(df).is_err(),
            "transform after a failed re-fit must error"
        );
    }

    #[test]
    fn test_out_of_sample_values_can_exceed_unit_range() {
        let col = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();

        // An unseen value larger than the training maximum is still divided by
        // the training-time max_abs, so it can fall outside [-1, 1].
        let new_col = Column::from(Series::new("x".into(), &[4.0f64]));
        let new_df = DataFrame::new(1, vec![new_col]).unwrap();
        let out = scaler.transform(new_df).unwrap();

        let v = out.column("x").unwrap().f64().unwrap().get(0).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }

    /// MaxAbsScaler on extreme magnitude data (~1e300) operates via absolute maximum
    /// without squaring, so it scales stably without overflowing.
    #[test]
    fn test_max_abs_scaler_extreme_magnitude_scales_ok() {
        let col = Column::from(Series::new("x".into(), &[1e300_f64, 5e300]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();

        let s = out.column("x").unwrap().f64().unwrap();
        assert_relative_eq!(s.get(0).unwrap(), 0.2, epsilon = 1e-12);
        assert_relative_eq!(s.get(1).unwrap(), 1.0, epsilon = 1e-12);
    }
}
