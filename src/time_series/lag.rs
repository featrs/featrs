//! Lag feature generation.
//!
//! [`Lagger`] creates shifted copies of columns for use as features
//! in forecasting models. Analogous to `df.shift()` in pandas.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashSet;

/// Create lag features by shifting columns.
///
/// Each specified column gets `periods` additional columns, one for each
/// lag value: `col_name_lag_1`, `col_name_lag_2`, etc.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::lag::Lagger;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("value".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut lagger = Lagger::new(&["value"], &[1, 3, 7]);
/// lagger.fit(df.clone())?;
/// let lagged = lagger.transform(df)?;
/// assert_eq!(lagged.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Lagger {
    fitted: bool,
    columns: Vec<String>,
    periods: Vec<i64>,
}

impl Lagger {
    /// Create a new `Lagger`.
    ///
    /// * `columns` — names of columns to lag
    /// * `periods` — shift amounts (e.g. `[1, 7]` for lag-1 and lag-7)
    pub fn new(columns: &[&str], periods: &[i64]) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            periods: periods.to_vec(),
        }
    }
}

impl Fit<DataFrame> for Lagger {
    type Output = ();

    /// Validate the input and mark the transformer fitted. A failed fit leaves it unfitted.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "Lagger: at least one column name is required.".into(),
            ));
        }
        if self.periods.is_empty() {
            return Err(Error::InvalidInput(
                "Lagger: at least one period is required.".into(),
            ));
        }
        for col in &self.columns {
            if x.column(col.as_str()).is_err() {
                return Err(Error::InvalidInput(format!(
                    "Lagger: column '{}' not found in input. Available columns: {:?}",
                    col,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                )));
            }
        }
        let mut seen: HashSet<i64> = HashSet::new();
        for &p in &self.periods {
            if !seen.insert(p) {
                return Err(Error::InvalidInput(format!(
                    "Lagger: duplicate period {} in periods {:?}. \
                     Each lag period must be unique.",
                    p, self.periods
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Lagger {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("Lagger has not been fitted.".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = out.column(col.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "Lagger.transform: column '{}' not found. {}",
                    col, e
                ))
            })?;
            let s = s.clone();
            for &period in &self.periods {
                let shifted = s.shift(period);
                let lag_name = format!("{}_lag_{}", col, period);
                out.with_column(shifted.with_name(lag_name.as_str().into()))
                    .map_err(|e| Error::Computation(e.to_string()))?;
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagger() {
        let vals = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::new(5, vec![vals]).unwrap();
        let mut lagger = Lagger::new(&["x"], &[1, 2]);

        lagger.fit(df.clone()).unwrap();
        let result = lagger.transform(df).unwrap();

        assert_eq!(result.width(), 3); // x, x_lag_1, x_lag_2
        assert_eq!(result.height(), 5);
    }

    #[test]
    fn test_lagger_duplicate_periods_rejected() {
        let vals = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::new(5, vec![vals]).unwrap();
        let mut lagger = Lagger::new(&["x"], &[1, 1]);

        let result = lagger.fit(df);
        assert!(
            result.is_err(),
            "duplicate periods should be rejected at fit"
        );
        let err = result.unwrap_err();
        let payload = match err {
            Error::InvalidInput(msg) => msg,
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        };
        assert!(
            payload.contains("duplicate period"),
            "error message should mention 'duplicate period', got: {payload}"
        );
        assert!(
            payload.contains(" 1 ") || payload.contains(" 1,"),
            "error message should contain the duplicate value '1', got: {payload}"
        );
        assert!(
            payload.contains("[1, 1]"),
            "error message should contain the full periods list '[1, 1]', got: {payload}"
        );
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
        let mut lagger = Lagger::new(&["x"], &[1]);

        lagger.fit(good.clone()).unwrap();
        assert!(lagger.fit(bad).is_err());
        assert!(matches!(lagger.transform(good), Err(Error::NotFitted(_))));
    }
}
