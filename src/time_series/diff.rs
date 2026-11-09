//! Differencing and percentage change.
//!
//! [`Difference`] computes `x[t] - x[t - period]` and percentage change.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Compute differences and percentage changes.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::diff::Difference;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("value".into(), &[10.0_f64, 20.0, 30.0, 40.0]));
/// let df = DataFrame::new(4, vec![col])?;
///
/// let mut d = Difference::new(&["value"], 1, false);
/// d.fit(df.clone())?;
/// let diffed = d.transform(df)?;
/// assert_eq!(diffed.height(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Difference {
    fitted: bool,
    columns: Vec<String>,
    period: i64,
    pct_change: bool,
}

impl Difference {
    /// Create a new differencer for `columns` over `period` lag.
    /// If `pct_change` is `true`, compute percentage change instead of absolute difference.
    pub fn new(columns: &[&str], period: i64, pct_change: bool) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            period,
            pct_change,
        }
    }

    /// Create a diff transformer (`x[t] - x[t-1]`).
    pub fn diff(columns: &[&str], period: i64) -> Self {
        Self::new(columns, period, false)
    }

    /// Create a pct_change transformer (`(x[t] - x[t-1]) / x[t-1]`).
    pub fn pct_change(columns: &[&str], period: i64) -> Self {
        Self::new(columns, period, true)
    }
}

impl Fit<DataFrame> for Difference {
    type Output = ();

    /// Validate the input and mark the transformer fitted. A failed fit leaves it unfitted.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "Difference: at least one column is required.".into(),
            ));
        }
        for col in &self.columns {
            let c = x.column(col.as_str()).map_err(|_| {
                Error::InvalidInput(format!("Difference: column '{}' not found.", col))
            })?;
            if c.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "Difference: column '{}' has dtype {}; expected Float64.",
                    col,
                    c.dtype()
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Difference {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("Difference".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = out.column(col.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "Difference.transform: column '{}' not found. {}",
                    col, e
                ))
            })?;
            let s = s.clone();

            let suffix = if self.pct_change { "pct" } else { "diff" };
            let col_name = format!("{}_{}_{}", col, suffix, self.period);

            let orig = s.f64().map_err(|_| {
                Error::InvalidInput(format!("Difference: column '{}' is not f64", col))
            })?;
            let shifted = s.shift(self.period);
            let shifted = shifted.f64().map_err(|_| {
                Error::InvalidInput(format!("Difference: column '{}' is not f64", col))
            })?;

            let result: ChunkedArray<Float64Type> = orig
                .iter()
                .zip(shifted.iter())
                .map(|(a, b)| match (a, b) {
                    (Some(va), Some(vb)) => {
                        if self.pct_change {
                            if vb.abs() > f64::EPSILON {
                                Some((va - vb) / vb)
                            } else {
                                None
                            }
                        } else {
                            Some(va - vb)
                        }
                    }
                    _ => None,
                })
                .collect();

            out.with_column(
                result
                    .into_series()
                    .with_name(col_name.as_str().into())
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
    fn test_diff() {
        let vals = Column::from(Series::new("x".into(), &[10.0f64, 20.0, 30.0, 40.0]));
        let df = DataFrame::new(4, vec![vals]).unwrap();
        let mut d = Difference::diff(&["x"], 1);

        d.fit(df.clone()).unwrap();
        let result = d.transform(df).unwrap();

        let diffed = result.column("x_diff_1").unwrap().f64().unwrap();
        assert!(diffed.get(0).is_none());
        assert_relative_eq!(diffed.get(1).unwrap(), 10.0, epsilon = 1e-6);
        assert_relative_eq!(diffed.get(2).unwrap(), 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pct_change() {
        let vals = Column::from(Series::new("x".into(), &[100.0f64, 110.0, 121.0]));
        let df = DataFrame::new(3, vec![vals]).unwrap();
        let mut d = Difference::pct_change(&["x"], 1);

        d.fit(df.clone()).unwrap();
        let result = d.transform(df).unwrap();

        let pct = result.column("x_pct_1").unwrap().f64().unwrap();
        assert!(pct.get(0).is_none());
        assert_relative_eq!(pct.get(1).unwrap(), 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let good = DataFrame::new(
            3,
            vec![Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]))],
        )
        .unwrap();
        let bad = DataFrame::new(
            3,
            vec![Column::from(Series::new(
                "missing".into(),
                &[1.0_f64, 2.0, 3.0],
            ))],
        )
        .unwrap();
        let mut d = Difference::diff(&["x"], 1);

        d.fit(good.clone()).unwrap();
        assert!(d.fit(bad).is_err());
        assert!(matches!(d.transform(good), Err(Error::NotFitted(_))));
    }
}
