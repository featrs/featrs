//! Missing value imputation.
//!
//! [`SimpleImputer`] replaces missing (`None` / `null`) values with
//! a statistic computed from the non-missing values in each column.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashMap;

/// Strategy for computing imputation values.
#[derive(Clone, Copy)]
pub enum Strategy {
    /// Replace missing values with the column mean.
    Mean,
    /// Replace missing values with the column median.
    Median,
    /// Replace missing values with the most frequent value in the column.
    MostFrequent,
    /// Replace missing values with a constant value.
    Constant(f64),
}

/// Impute missing values using basic statistics.
///
/// Only operates on [`Float64`](DataType::Float64) columns.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::imputer::SimpleImputer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[Some(1.0_f64), None, Some(3.0)]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut imp = SimpleImputer::mean();
/// imp.fit(df.clone())?;
/// let filled = imp.transform(df)?;
/// assert_eq!(filled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SimpleImputer {
    fitted: bool,
    strategy: Strategy,
    fill_values: Option<HashMap<String, f64>>,
}

impl SimpleImputer {
    /// Create a new `SimpleImputer` with the given strategy.
    pub fn new(strategy: Strategy) -> Self {
        Self {
            fitted: false,
            strategy,
            fill_values: None,
        }
    }

    /// Impute using the column mean.
    pub fn mean() -> Self {
        Self::new(Strategy::Mean)
    }

    /// Impute using the column median.
    pub fn median() -> Self {
        Self::new(Strategy::Median)
    }

    /// Impute using the most frequent (mode) value.
    pub fn most_frequent() -> Self {
        Self::new(Strategy::MostFrequent)
    }

    /// Impute with a constant value.
    pub fn constant(value: f64) -> Self {
        Self::new(Strategy::Constant(value))
    }
}

impl Fit<DataFrame> for SimpleImputer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        self.fill_values = None;

        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "SimpleImputer.fit received a DataFrame with 0 rows. \
                 Provide data with at least 1 row."
                    .into(),
            ));
        }
        let mut fill_values = HashMap::new();

        for col in x.columns() {
            let name = col.name().to_string();
            if col.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = col.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "SimpleImputer.fit: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;
            let all_vals: Vec<f64> = ca.iter().flatten().collect();
            let has_missing = ca.iter().any(|v| v.is_none());

            if !has_missing {
                continue;
            }

            let fill = match self.strategy {
                Strategy::Mean => {
                    if all_vals.is_empty() {
                        return Err(Error::Computation(format!(
                            "SimpleImputer(Mean): column '{}' has no non-null values. \
                             Cannot compute the mean of an all-null column. \
                             Use Strategy::Constant instead.",
                            name
                        )));
                    }
                    all_vals.iter().sum::<f64>() / all_vals.len() as f64
                }
                Strategy::Median => {
                    let mut sorted = all_vals.clone();
                    sorted.sort_by(|a, b| a.total_cmp(b));
                    if sorted.is_empty() {
                        return Err(Error::Computation(format!(
                            "SimpleImputer(Median): column '{}' has no non-null values. \
                             Cannot compute the median of an all-null column. \
                             Use Strategy::Constant instead.",
                            name
                        )));
                    }
                    let mid = sorted.len() / 2;
                    if sorted.len().is_multiple_of(2) {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                Strategy::MostFrequent => {
                    if all_vals.is_empty() {
                        return Err(Error::Computation(format!(
                            "SimpleImputer(MostFrequent): column '{}' has no non-null values. \
                             Cannot compute the mode of an all-null column. \
                             Use Strategy::Constant instead.",
                            name
                        )));
                    }
                    let mut freq: HashMap<u64, usize> = HashMap::new();
                    for &v in &all_vals {
                        *freq.entry(v.to_bits()).or_default() += 1;
                    }
                    let (max_key, _) = freq
                        .into_iter()
                        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                        .ok_or_else(|| {
                            Error::Computation(format!(
                                "SimpleImputer(MostFrequent): column '{}' has no non-null values",
                                name
                            ))
                        })?;
                    f64::from_bits(max_key)
                }
                Strategy::Constant(v) => v,
            };

            fill_values.insert(name, fill);
        }

        self.fill_values = Some(fill_values);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for SimpleImputer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "SimpleImputer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let fill_values = self.fill_values.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "SimpleImputer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let mut out = x.clone();

        for (name, fill) in fill_values {
            let s = out.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "SimpleImputer.transform: column '{}' not found. \
                     The imputer was fitted on columns: {:?}. {}",
                    name,
                    fill_values.keys().collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "SimpleImputer.transform: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            let filled: ChunkedArray<Float64Type> =
                ca.iter().map(|opt| opt.or(Some(*fill))).collect();
            out.replace(name.as_str(), filled.into_series().into())
                .map_err(|e| {
                    Error::Computation(format!(
                        "SimpleImputer.transform: failed to replace column '{}'. {}",
                        name, e
                    ))
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
        let a = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(3.0)]));
        let b = Column::from(Series::new("y".into(), &[Some(10.0f64), Some(20.0), None]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_imputer_mean() {
        let mut imp = SimpleImputer::mean();
        let df = make_test_df();

        imp.fit(df.clone()).unwrap();
        let result = imp.transform(df).unwrap();

        let x_vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(x_vals[1], 2.0, epsilon = 1e-6);

        let y_vals: Vec<f64> = result
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(y_vals[2], 15.0, epsilon = 1e-6);
    }

    #[test]
    fn test_imputer_constant() {
        let mut imp = SimpleImputer::constant(0.0);
        let df = make_test_df();

        imp.fit(df.clone()).unwrap();
        let result = imp.transform(df).unwrap();

        let x_vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(x_vals[1], 0.0, epsilon = 1e-6);
    }

    /// Regression: the Median path sorted with `partial_cmp().unwrap()`, which
    /// panicked when a NaN was present. `total_cmp` sorts NaN deterministically.
    #[test]
    fn test_imputer_median_with_nan_does_not_panic() {
        // Column `x` has a NaN among the non-null values; the median sort must
        // not panic.
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0f64), Some(f64::NAN), None, Some(3.0)],
        ));
        let df = DataFrame::new(4, vec![x]).unwrap();
        let mut imp = SimpleImputer::median();
        imp.fit(df.clone()).unwrap();
        let _ = imp.transform(df).unwrap();
    }

    #[test]
    fn test_imputer_median_value() {
        // Non-null values [1.0, 3.0] → median = 2.0; the null at index 2 is
        // imputed with 2.0.
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), Some(3.0), None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut imp = SimpleImputer::median();
        imp.fit(df.clone()).unwrap();
        let result = imp.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_imputer_most_frequent_value() {
        // Non-null values [1.0, 2.0, 2.0] → mode = 2.0; the null is imputed
        // with 2.0.
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0f64), Some(2.0), Some(2.0), None],
        ));
        let df = DataFrame::new(4, vec![x]).unwrap();
        let mut imp = SimpleImputer::most_frequent();
        imp.fit(df.clone()).unwrap();
        let result = imp.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[3], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_imputer_most_frequent_ties() {
        // Tied frequencies: 1.0 appears 2×, 2.0 appears 2×, two nulls.
        // On ties, the smallest value (1.0) must be chosen deterministically.
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0f64), Some(1.0), Some(2.0), Some(2.0), None, None],
        ));
        let df = DataFrame::new(6, vec![x]).unwrap();
        let mut imp = SimpleImputer::most_frequent();
        imp.fit(df.clone()).unwrap();
        let result = imp.transform(df.clone()).unwrap();
        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals[4], 1.0);
        assert_eq!(vals[5], 1.0);
        // Run twice to confirm reproducibility
        let result2 = imp.transform(df).unwrap();
        let vals2: Vec<f64> = result2
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vals2);
    }

    #[test]
    fn test_imputer_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut imp = SimpleImputer::mean();
        let fit_result = imp.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-null column with Mean must error"
        );
    }

    #[test]
    fn test_imputer_not_fitted() {
        let imp = SimpleImputer::mean();
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None]));
        let df = DataFrame::new(2, vec![x]).unwrap();
        assert!(imp.transform(df).is_err());
    }
}
