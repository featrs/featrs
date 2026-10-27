//! Feature binarization.
//!
//! [`Binarizer`] thresholds numeric features: values above the threshold
//! become `1.0`, others become `0.0`. Missing data is preserved: polars
//! nulls stay null, and non-finite `NaN` inputs are emitted as null rather
//! than being conflated with a below-threshold value.
//!
//! Only `Float64` columns are binarized. A frame that contains no `Float64`
//! column at all is rejected with the same descriptive error every sibling
//! numeric transformer emits, at fit time and again at transform time, instead
//! of silently passing strings or dates through. Non-`Float64` columns that
//! arrive alongside at least one `Float64` column are passed through
//! unchanged.

use crate::traits::schema::{RequiresSchema, SchemaSpec};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column_opt, require_f64_columns};
use polars::prelude::*;

/// Binarize data according to a threshold.
///
/// Values `> threshold` become `1.0`; all other finite values become `0.0`.
/// Missing data is preserved through transform: nulls stay null, and `NaN`
/// inputs are emitted as null so downstream models can distinguish missing
/// values from below-threshold ones. ±`Inf` inputs binarize normally against
/// the finite threshold.
///
/// The threshold must be finite (`NaN` or ±`Inf` thresholds are rejected at
/// fit time).
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::binarizer::Binarizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[-1.0_f64, 0.5, 2.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut b = Binarizer::new(0.5);
/// b.fit(df.clone())?;
/// let binarized = b.transform(df)?;
/// assert_eq!(binarized.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Binarizer {
    fitted: bool,
    threshold: f64,
}

impl Binarizer {
    /// Create a new `Binarizer` with the given threshold.
    ///
    /// Values strictly greater than `threshold` are set to `1.0`. The
    /// threshold is validated when [`Fit::fit`] is called; builders return
    /// `Self` and cannot signal errors, so an invalid threshold surfaces as
    /// an `Error::InvalidInput` from `fit`.
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
        }
    }
}

impl Default for Binarizer {
    /// Create a `Binarizer` with a default threshold of `0.0`.
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame> for Binarizer {
    type Output = ();

    /// Validate the input DataFrame has at least one row and one column,
    /// contains at least one `Float64` column, and that the configured
    /// threshold is finite.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset fitted state up front: if this fit fails, transform must
        // not silently apply a stale configuration from a previous fit.
        self.fitted = false;

        if !self.threshold.is_finite() {
            return Err(Error::InvalidInput(format!(
                "Binarizer.fit received a non-finite threshold ({}). \
                 Provide a finite threshold.",
                self.threshold
            )));
        }
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "Binarizer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        require_f64_columns(&x, "Binarizer")?;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Binarizer {
    type Output = DataFrame;

    /// Binarize the data: values above the threshold become `1.0`,
    /// other finite values become `0.0`, and missing data stays missing —
    /// nulls remain null and `NaN` inputs are emitted as null.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFitted` if called before [`Fit::fit`], and
    /// `Error::InvalidInput` if the frame contains no `Float64` column.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Binarizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "Binarizer")?;

        let mut out = x.clone();
        let threshold = self.threshold;

        for name in &col_names {
            replace_f64_column_opt(&mut out, name.as_str(), "Binarizer", |v| {
                if v.is_nan() {
                    None
                } else if v > threshold {
                    Some(1.0)
                } else {
                    Some(0.0)
                }
            })?;
        }

        Ok(out)
    }
}

/// `Binarizer::fit` requires at least one `Float64` column, so the spec does too.
impl RequiresSchema for Binarizer {
    fn required_schema(&self) -> SchemaSpec {
        SchemaSpec::f64_columns(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use polars::prelude::AnyValue;

    fn column_values(col: &Column) -> Vec<Option<f64>> {
        col.f64().unwrap().iter().collect()
    }

    #[test]
    fn test_binarizer_default() {
        let mut b = Binarizer::default();
        let a = Column::from(Series::new("x".into(), &[-1.0f64, 0.0, 2.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let mut b = Binarizer::new(5.0);
        let a = Column::from(Series::new("x".into(), &[1.0f64, 5.0, 10.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_empty_rows_rejected() {
        let a = Column::from(Series::new("x".into(), Vec::<f64>::new()));
        let df = DataFrame::new(0, vec![a]).unwrap();

        let mut b = Binarizer::new(0.5);
        let result = b.fit(df);
        assert!(
            result.is_err(),
            "a 0-row DataFrame should be rejected at fit time"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("empty DataFrame"),
            "error message should mention the empty DataFrame"
        );
    }

    #[test]
    fn test_binarizer_preserves_null_and_nan() {
        let mut b = Binarizer::new(0.5);
        // Rows: 0 -> 2.0 (above), 1 -> null, 2 -> NaN, 3 -> -1.0 (below).
        let a = Column::from(Series::new(
            "x".into(),
            &[Some(2.0), None, Some(f64::NAN), Some(-1.0)],
        ));
        let df = DataFrame::new(4, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals = column_values(result.column("x").unwrap());
        assert_relative_eq!(vals[0].unwrap(), 1.0, epsilon = 1e-6);
        assert!(vals[1].is_none(), "null input must stay null");
        assert!(
            vals[2].is_none(),
            "NaN input must be emitted as null, not 0.0"
        );
        assert_relative_eq!(vals[3].unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_rejects_nan_threshold() {
        let a = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();

        let mut b = Binarizer::new(f64::NAN);
        let err = b.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("non-finite threshold"),
            "error message should mention the non-finite threshold"
        );
        assert!(
            !b.fitted,
            "a failed fit must not mark the transformer fitted"
        );
    }

    #[test]
    fn test_binarizer_rejects_infinite_threshold() {
        let a = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a]).unwrap();

        for threshold in [f64::INFINITY, f64::NEG_INFINITY] {
            let mut b = Binarizer::new(threshold);
            let err = b.fit(df.clone()).unwrap_err();
            assert!(
                err.to_string().contains("non-finite threshold"),
                "error message should mention the non-finite threshold"
            );
        }
    }

    #[test]
    fn test_binarizer_not_fitted() {
        let b = Binarizer::new(0.5);
        let result = b.transform(DataFrame::default());
        assert!(
            matches!(result, Err(Error::NotFitted(_))),
            "transform before fit must return NotFitted"
        );
    }

    #[test]
    fn test_binarizer_refit_resets_fitted_state() {
        let good = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let good_df = DataFrame::new(2, vec![good]).unwrap();

        let mut b = Binarizer::new(0.5);
        b.fit(good_df.clone()).unwrap();
        assert!(b.fitted);

        // A re-fit that fails validation must leave the transformer unfitted
        // rather than silently keeping stale state.
        let bad_df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        assert!(b.fit(bad_df).is_err());
        assert!(!b.fitted, "a failed re-fit must reset the fitted flag");

        b.fit(good_df).unwrap();
        assert!(b.fitted);
    }

    #[test]
    fn test_binarizer_rejects_all_string_frame_at_fit() {
        let s = Column::from(Series::new(
            "str1".into(),
            &["a".to_string(), "b".to_string()],
        ));
        let df = DataFrame::new(2, vec![s]).unwrap();

        let mut b = Binarizer::new(0.5);
        let err = b.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("no Float64 columns found"),
            "error should reuse the shared dtype message, got: {err}"
        );
        assert!(
            err.to_string().contains("Binarizer"),
            "error should name the transformer, got: {err}"
        );
        assert!(!b.fitted);
    }

    #[test]
    fn test_binarizer_rejects_date_frame_at_fit() {
        let d = Series::new("d".into(), &[1i32, 2])
            .cast(&DataType::Date)
            .unwrap();
        let df = DataFrame::new(2, vec![Column::from(d)]).unwrap();

        let mut b = Binarizer::new(0.5);
        let err = b.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("no Float64 columns found"),
            "error should reuse the shared dtype message, got: {err}"
        );
        assert!(!b.fitted);
    }

    #[test]
    fn test_binarizer_transform_rejects_all_string_frame() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0]));
        let good = DataFrame::new(2, vec![x]).unwrap();
        let mut b = Binarizer::new(0.5);
        b.fit(good).unwrap();

        let s = Column::from(Series::new(
            "str1".into(),
            &["a".to_string(), "b".to_string()],
        ));
        let strings = DataFrame::new(2, vec![s]).unwrap();

        let err = b.transform(strings).unwrap_err();
        assert!(
            err.to_string().contains("no Float64 columns found"),
            "a fitted Binarizer must not pass strings through, got: {err}"
        );
    }

    #[test]
    fn test_binarizer_mixed_frame_preserves_non_f64_dtype() {
        // Non-Float64 columns alongside a Float64 column must survive
        // transform untouched, dtype and values intact.
        let d = Series::new("d".into(), &[1i32, 2])
            .cast(&DataType::Date)
            .unwrap();
        let x = Column::from(Series::new("x".into(), &[-1.0f64, 0.9]));
        let df = DataFrame::new(2, vec![Column::from(d), x]).unwrap();

        let mut b = Binarizer::new(0.5);
        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let d = result.column("d").unwrap();
        assert_eq!(d.dtype(), &DataType::Date, "date column dtype must survive");
        assert_eq!(result.column("x").unwrap().dtype(), &DataType::Float64);
        let vals = column_values(result.column("x").unwrap());
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1].unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_non_f64_columns_pass_through() {
        let mut b = Binarizer::new(0.5);
        let x = Column::from(Series::new("x".into(), &[-1.0f64, 0.9]));
        let s = Column::from(Series::new("s".into(), &["a".to_string(), "b".to_string()]));
        let df = DataFrame::new(2, vec![x, s]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-6);
        match result.column("s").unwrap().get(0).unwrap() {
            AnyValue::String(v) => assert_eq!(v, "a"),
            other => panic!("expected string value, got {other:?}"),
        }
    }

    #[test]
    fn test_binarizer_infinite_inputs_keep_comparison_semantics() {
        // ±Inf inputs compare meaningfully against a finite threshold and are
        // binarized like any other finite-or-infinite value; only NaN is
        // treated as missing data.
        let mut b = Binarizer::new(0.5);
        let a = Column::from(Series::new(
            "x".into(),
            &[Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
        ));
        let df = DataFrame::new(2, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals = column_values(result.column("x").unwrap());
        assert_relative_eq!(vals[0].unwrap(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1].unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_requires_float64_columns() {
        use crate::traits::schema::validate_schema;

        // `fit` rejects frames without a Float64 column, so the spec does too.
        let spec = Binarizer::new(0.5).required_schema();
        assert_eq!(spec, SchemaSpec::f64_columns(1));

        let s = Column::from(Series::new("s".into(), &["a", "b"]));
        let strings = DataFrame::new(2, vec![s]).unwrap();
        assert!(validate_schema(&spec, &strings).is_err());

        let x = Column::from(Series::new("x".into(), &[0.0f64, 1.0]));
        assert!(validate_schema(&spec, &DataFrame::new(2, vec![x]).unwrap()).is_ok());
    }
}
