//! Rolling window aggregations.
//!
//! [`RollingAggregator`] computes rolling mean, std, min, max, sum
//! over a fixed window. Analogous to `df.rolling()` in pandas.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Rolling window function to apply.
#[derive(Clone, Copy)]
pub enum RollingFn {
    /// Rolling mean over the window.
    Mean,
    /// Rolling (population) standard deviation over the window.
    Std,
    /// Rolling minimum over the window.
    Min,
    /// Rolling maximum over the window.
    Max,
    /// Rolling sum over the window.
    Sum,
}

/// Compute rolling window statistics.
///
/// Input nulls retain their row positions. A complete window containing a
/// null produces a null result.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::rolling::{RollingAggregator, RollingFn};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("value".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
/// let df = DataFrame::new(5, vec![col])?;
///
/// let mut r = RollingAggregator::new(&["value"], 3, RollingFn::Mean);
/// r.fit(df.clone())?;
/// let rolled = r.transform(df)?;
/// assert_eq!(rolled.height(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RollingAggregator {
    fitted: bool,
    columns: Vec<String>,
    window_size: usize,
    function: RollingFn,
}

impl RollingAggregator {
    /// Create a new rolling aggregator for `columns` over a `window_size`-row
    /// window, applying `function` (e.g. [`RollingFn::Mean`]).
    pub fn new(columns: &[&str], window_size: usize, function: RollingFn) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            window_size,
            function,
        }
    }

    fn rolling_series(&self, s: &Series) -> Result<Series> {
        let ca = s
            .f64()
            .map_err(|_| Error::InvalidInput("column must be f64".into()))?;
        let vals: Vec<Option<f64>> = ca.iter().collect();
        let n = vals.len();
        let w = self.window_size;

        let result: Vec<Option<f64>> = (0..n)
            .map(|i| {
                if i < w - 1 {
                    None
                } else {
                    let start = i + 1 - w;
                    let window = &vals[start..=i];
                    if window.iter().any(Option::is_none) {
                        None
                    } else {
                        match self.function {
                            RollingFn::Mean => {
                                Some(window.iter().flatten().copied().sum::<f64>() / w as f64)
                            }
                            RollingFn::Sum => Some(window.iter().flatten().copied().sum()),
                            RollingFn::Min => window
                                .iter()
                                .flatten()
                                .copied()
                                .fold(f64::INFINITY, f64::min)
                                .into(),
                            RollingFn::Max => window
                                .iter()
                                .flatten()
                                .copied()
                                .fold(f64::NEG_INFINITY, f64::max)
                                .into(),
                            RollingFn::Std => {
                                let mean = window.iter().flatten().copied().sum::<f64>() / w as f64;
                                let var = window
                                    .iter()
                                    .flatten()
                                    .map(|v| (v - mean).powi(2))
                                    .sum::<f64>()
                                    / w as f64;
                                Some(var.sqrt())
                            }
                        }
                    }
                }
            })
            .collect();

        let new_ca: ChunkedArray<Float64Type> = result.into_iter().collect();
        Ok(new_ca.into_series())
    }
}

impl Fit<DataFrame> for RollingAggregator {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "RollingAggregator: at least one column is required.".into(),
            ));
        }
        if self.window_size < 2 {
            return Err(Error::InvalidInput(format!(
                "RollingAggregator: window_size must be >= 2, got {}",
                self.window_size
            )));
        }
        for col in &self.columns {
            if x.column(col.as_str()).is_err() {
                return Err(Error::InvalidInput(format!(
                    "RollingAggregator: column '{}' not found.",
                    col
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for RollingAggregator {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("RollingAggregator".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = out
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "RollingAggregator.transform: column '{}' not found. {}",
                        col, e
                    ))
                })?
                .as_materialized_series()
                .clone();
            let fn_name = match self.function {
                RollingFn::Mean => "mean",
                RollingFn::Std => "std",
                RollingFn::Min => "min",
                RollingFn::Max => "max",
                RollingFn::Sum => "sum",
            };
            let rolled = self.rolling_series(&s)?;
            let rolled_name = format!("{}_{}_{}", col, fn_name, self.window_size);
            out.with_column(rolled.with_name(rolled_name.as_str().into()).into())
                .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rolling_mean(vals: &[Option<f64>], window_size: usize) -> Vec<Option<f64>> {
        let col = Column::from(Series::new("x".into(), vals));
        let df = DataFrame::new(vals.len(), vec![col]).unwrap();
        let mut r = RollingAggregator::new(&["x"], window_size, RollingFn::Mean);
        r.fit(df.clone()).unwrap();
        let result = r.transform(df).unwrap();

        result
            .column(&format!("x_mean_{window_size}"))
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn test_rolling_mean() {
        let vals = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::new(5, vec![vals]).unwrap();
        let mut r = RollingAggregator::new(&["x"], 3, RollingFn::Mean);

        r.fit(df.clone()).unwrap();
        let result = r.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        let rolled = result.column("x_mean_3").unwrap().f64().unwrap();
        assert_eq!(
            rolled.iter().collect::<Vec<_>>(),
            vec![None, None, Some(2.0), Some(3.0), Some(4.0)]
        );
    }

    #[test]
    fn test_interior_nulls_invalidate_containing_windows() {
        assert_eq!(
            rolling_mean(&[Some(1.0), None, Some(3.0), Some(4.0)], 2),
            vec![None, None, None, Some(3.5)]
        );
    }

    #[test]
    fn test_leading_and_trailing_nulls_remain_aligned() {
        assert_eq!(
            rolling_mean(&[None, Some(2.0), Some(4.0), None], 2),
            vec![None, None, Some(3.0), None]
        );
    }
}
