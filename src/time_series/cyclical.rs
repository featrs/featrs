//! Cyclical encoding.
//!
//! [`CyclicalEncoder`] maps cyclical features (hour, day-of-week, month)
//! to `sin` and `cos` components so that `23:00` and `01:00` are close
//! in the encoded space.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Encode cyclical features with sin/cos transformation.
///
/// For each column, computes `sin(2π * x / period)` and
/// `cos(2π * x / period)`, preserving the cyclic relationship.
///
/// Periods are validated at [`fit`](Fit::fit) time: every period must be
/// finite and `>= 1`; otherwise `fit` returns [`Error::InvalidInput`] whose
/// message contains `period must be >= 1`. A zero (or non-positive) period
/// would divide by zero and silently emit `NaN`/`±Inf` encodings.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::cyclical::CyclicalEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("hour".into(), &[0.0_f64, 6.0, 12.0, 18.0]));
/// let df = DataFrame::new(4, vec![col])?;
///
/// let mut enc = CyclicalEncoder::new(&["hour"], 24);
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.height(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CyclicalEncoder {
    fitted: bool,
    columns: Vec<(String, f64)>,
}

impl CyclicalEncoder {
    /// Create an encoder that maps each column to `(sin, cos)` using a single
    /// shared integer period (e.g. `24` for hours, `12` for months).
    pub fn new(columns: &[&str], period: usize) -> Self {
        let period_f = period as f64;
        Self {
            fitted: false,
            columns: columns.iter().map(|s| (s.to_string(), period_f)).collect(),
        }
    }

    /// Create an encoder where each column gets its own integer period, given as
    /// `(column_name, period)` pairs.
    pub fn with_periods(columns: &[(&str, usize)]) -> Self {
        Self {
            fitted: false,
            columns: columns
                .iter()
                .map(|(s, p)| (s.to_string(), *p as f64))
                .collect(),
        }
    }
}

impl Fit<DataFrame> for CyclicalEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        let n_cols = x.width();
        if x.height() == 0 || n_cols == 0 {
            return Err(Error::InvalidInput(
                "CyclicalEncoder.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "CyclicalEncoder: at least one column is required.".into(),
            ));
        }
        for (col, _) in &self.columns {
            let s = x.column(col.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "CyclicalEncoder: column '{}' not found. {}",
                    col, e
                ))
            })?;
            if s.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "CyclicalEncoder: column '{}' has dtype {}; expected Float64.",
                    col,
                    s.dtype()
                )));
            }
        }
        for (col, period) in &self.columns {
            if !period.is_finite() || *period < 1.0 {
                return Err(Error::InvalidInput(format!(
                    "CyclicalEncoder: period must be >= 1 (got {period} for column '{col}')."
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CyclicalEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("CyclicalEncoder".into()));
        }
        let mut out = x.clone();
        let two_pi = 2.0 * std::f64::consts::PI;

        for (col, period) in &self.columns {
            let s = out.column(col.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "CyclicalEncoder.transform: column '{}' not found. {}",
                    col, e
                ))
            })?;
            let s = s.clone();
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "CyclicalEncoder.transform: column '{}' has dtype {}; expected Float64. {}",
                    col,
                    s.dtype(),
                    e
                ))
            })?;

            let sin_vals: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| (two_pi * v / period).sin()))
                .collect();
            let cos_vals: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| (two_pi * v / period).cos()))
                .collect();

            let sin_name = format!("{}_sin", col);
            let cos_name = format!("{}_cos", col);
            out.with_column(
                sin_vals
                    .into_series()
                    .with_name(sin_name.as_str().into())
                    .into(),
            )
            .map_err(|e| Error::Computation(e.to_string()))?;
            out.with_column(
                cos_vals
                    .into_series()
                    .with_name(cos_name.as_str().into())
                    .into(),
            )
            .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cyclical_encoding() {
        let vals = Column::from(Series::new("hour".into(), &[0.0f64, 6.0, 12.0, 18.0]));
        let df = DataFrame::new(4, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 24);

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 3); // hour, hour_sin, hour_cos
        let sin = result.column("hour_sin").unwrap().f64().unwrap();
        let cos = result.column("hour_cos").unwrap().f64().unwrap();

        // hour 0: sin=0, cos=1
        assert_relative_eq!(sin.get(0).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(cos.get(0).unwrap(), 1.0, epsilon = 1e-6);
        // hour 6 (90°): sin=1, cos=0
        assert_relative_eq!(sin.get(1).unwrap(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(cos.get(1).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cyclical_fit_validation() {
        // Test empty DataFrame
        let df_empty = DataFrame::empty();
        let mut enc = CyclicalEncoder::new(&["hour"], 24);
        let err = enc.fit(df_empty).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("received an empty DataFrame")),
            "Expected Error::InvalidInput containing 'received an empty DataFrame', got: {:?}",
            err
        );

        // Test non-Float64 column type (Int32)
        let vals = Column::from(Series::new("hour".into(), &[0_i32, 6, 12, 18]));
        let df_int = DataFrame::new(4, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 24);
        let err = enc.fit(df_int).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("expected Float64")),
            "Expected Error::InvalidInput containing 'expected Float64', got: {:?}",
            err
        );

        // Test missing column
        let vals = Column::from(Series::new("day".into(), &[1.0_f64, 2.0]));
        let df_wrong_col = DataFrame::new(2, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 24);
        let err = enc.fit(df_wrong_col).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("column 'hour' not found")),
            "Expected Error::InvalidInput containing 'column 'hour' not found', got: {:?}",
            err
        );

        // Test empty columns list regression test
        let vals = Column::from(Series::new("hour".into(), &[0.0_f64, 6.0]));
        let df = DataFrame::new(2, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&[], 24);
        let err = enc.fit(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("at least one column is required")),
            "Expected Error::InvalidInput containing 'at least one column is required', got: {:?}",
            err
        );
    }

    #[test]
    fn test_zero_period_rejected_new() {
        let vals = Column::from(Series::new("hour".into(), &[0.0_f64, 6.0]));
        let df = DataFrame::new(2, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 0);
        let err = enc.fit(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("period must be >= 1")),
            "Expected Error::InvalidInput containing 'period must be >= 1', got: {:?}",
            err
        );
    }

    #[test]
    fn test_zero_period_rejected_with_periods() {
        let vals = Column::from(Series::new("hour".into(), &[0.0_f64, 6.0]));
        let df = DataFrame::new(2, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::with_periods(&[("hour", 0)]);
        let err = enc.fit(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("period must be >= 1")),
            "Expected Error::InvalidInput containing 'period must be >= 1', got: {:?}",
            err
        );
    }

    #[test]
    fn test_period_one_boundary() {
        // period = 1 is the smallest valid period: integer values land at
        // multiples of 2π, so sin ≈ 0 and cos ≈ 1.
        let vals = Column::from(Series::new("hour".into(), &[0.0_f64, 1.0, 3.0]));
        let df = DataFrame::new(3, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 1);

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 3); // hour, hour_sin, hour_cos
        let sin = result.column("hour_sin").unwrap().f64().unwrap();
        let cos = result.column("hour_cos").unwrap().f64().unwrap();

        for i in 0..3 {
            let s = sin.get(i).unwrap();
            let c = cos.get(i).unwrap();
            assert!(
                s.is_finite() && c.is_finite(),
                "period=1 outputs must be finite, got sin={s}, cos={c}"
            );
            assert_relative_eq!(s, 0.0, epsilon = 1e-6);
            assert_relative_eq!(c, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        // A successful fit followed by a failed re-fit (zero period) must
        // leave the encoder un-fitted: transform must return NotFitted.
        let vals = Column::from(Series::new("hour".into(), &[0.0_f64, 6.0]));
        let df = DataFrame::new(2, vec![vals]).unwrap();
        let mut enc = CyclicalEncoder::new(&["hour"], 24);

        enc.fit(df.clone()).unwrap();
        assert!(enc.transform(df.clone()).is_ok());

        let bad = CyclicalEncoder::new(&["hour"], 0);
        enc.columns = bad.columns;
        let err = enc.fit(df.clone()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("period must be >= 1")),
            "Expected Error::InvalidInput containing 'period must be >= 1', got: {:?}",
            err
        );
        assert!(matches!(enc.transform(df), Err(Error::NotFitted(_))));
    }
}
