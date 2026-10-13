//! Row-wise normalization.
//!
//! [`Normalizer`] scales each row (sample) to unit norm independently.
//! Supports L1, L2, and Max norms. Analogous to `sklearn.preprocessing.Normalizer`.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Normalize samples individually to unit norm.
///
/// Each row is divided by its norm so that the row has unit length.
/// Note: this is a **row-wise** operation, not column-wise like the scalers.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::normalizer::Normalizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[3.0_f64, 0.0, 1.0]));
/// let b = Column::from(Series::new("b".into(), &[4.0_f64, 0.0, 2.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut n = Normalizer::l2();
/// n.fit(df.clone())?;
/// let normalized = n.transform(df)?;
/// assert_eq!(normalized.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Normalizer {
    fitted: bool,
    norm: Norm,
}

/// The norm to use for normalization.
#[derive(Clone, Copy)]
pub enum Norm {
    /// L1 norm — sum of absolute values.
    L1,
    /// L2 norm — Euclidean (sqrt of sum of squares).
    L2,
    /// Max norm — maximum absolute value.
    Max,
}

impl Normalizer {
    /// Create a new `Normalizer` with the given norm.
    pub fn new(norm: Norm) -> Self {
        Self {
            fitted: false,
            norm,
        }
    }

    /// Normalize using the L1 norm (sum of absolute values).
    pub fn l1() -> Self {
        Self::new(Norm::L1)
    }

    /// Normalize using the L2 norm (Euclidean).
    pub fn l2() -> Self {
        Self::new(Norm::L2)
    }

    /// Normalize using the Max norm (maximum absolute value).
    pub fn max() -> Self {
        Self::new(Norm::Max)
    }

    fn row_norm(values: &[f64], norm: Norm) -> f64 {
        let clean_vals = values.iter().copied().filter(|v| !v.is_nan());
        match norm {
            Norm::L1 => clean_vals.map(|v| v.abs()).sum(),
            Norm::L2 => clean_vals.map(|v| v * v).sum::<f64>().sqrt(),
            Norm::Max => clean_vals.map(|v| v.abs()).fold(0.0f64, f64::max),
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::l2()
    }
}

impl Fit<DataFrame> for Normalizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "Normalizer.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        for col in x.columns() {
            if col.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "Normalizer.fit: column '{}' has dtype {}; expected Float64. \
                     Normalizer operates on f64 columns only.",
                    col.name(),
                    col.dtype()
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Normalizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Normalizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let n_cols = x.width();
        let n_rows = x.height();
        let col_names: Vec<&str> = x.get_column_names().iter().map(|s| s.as_str()).collect();

        let mut col_data: Vec<Vec<Option<f64>>> = Vec::with_capacity(n_cols);
        for name in &col_names {
            let s = x.column(name).map_err(|e| {
                Error::InvalidInput(format!(
                    "Normalizer.transform: column '{}' not found. {}",
                    name, e
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "Normalizer.transform: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            col_data.push(ca.iter().collect());
        }

        for i in 0..n_rows {
            let row_vals: Vec<f64> = col_data.iter().filter_map(|col| col[i]).collect();
            let norm = Self::row_norm(&row_vals, self.norm);
            if !norm.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "Normalizer.transform: row {} norm overflowed or is not finite. \
                     Input values are too large for f64 norm computation.",
                    i
                )));
            }
            if norm > f64::EPSILON {
                for col in &mut col_data {
                    if let Some(v) = col[i].as_mut()
                        && !v.is_nan()
                    {
                        *v /= norm;
                    }
                }
            }
        }

        let mut out_cols: Vec<Column> = Vec::with_capacity(n_cols);
        for (j, name) in col_names.iter().enumerate() {
            let new_ca: ChunkedArray<Float64Type> = col_data[j].iter().copied().collect();
            let s = new_ca
                .into_series()
                .with_name(name.to_string().as_str().into());
            out_cols.push(s.into());
        }

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[3.0f64, 0.0, 1.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 0.0, 2.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_l2_normalization() {
        let mut n = Normalizer::l2();
        let df = make_test_df();

        n.fit(df.clone()).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 0.6, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 0.8, epsilon = 1e-6);
        assert_relative_eq!(col_a.get(1).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_l1_normalization() {
        let mut n = Normalizer::l1();
        let df = make_test_df();

        n.fit(df.clone()).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 3.0 / 7.0, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 4.0 / 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_max_normalization() {
        let mut n = Normalizer::max();
        let df = make_test_df();

        n.fit(df.clone()).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 0.75, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 1.0, epsilon = 1e-6);
    }

    /// `Max` norm must use absolute values; rows containing negatives must
    /// normalize by `max(|x_i|)` rather than `max(x_i)` (regression test for
    /// issue #9). The existing `test_max_normalization` only used positive
    /// values, so the bug was hidden.
    #[test]
    fn test_max_normalization_negative_values() {
        let a = Column::from(Series::new("a".into(), &[-5.0f64, 1.0]));
        let b = Column::from(Series::new("b".into(), &[3.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut n = Normalizer::max();
        n.fit(df.clone()).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();

        // Row 0 [-5, 3]: norm = max(|-5|, |3|) = 5.0  -> [-1.0, 0.6]
        assert_relative_eq!(col_a.get(0).unwrap(), -1.0, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 0.6, epsilon = 1e-6);

        // Row 1 [1, 2]: norm = max(|1|, |2|) = 2.0  -> [0.5, 1.0]
        assert_relative_eq!(col_a.get(1).unwrap(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(1).unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalizer_with_null_and_nan() {
        // Row 0: [3.0, 4.0] -> norm L2 is 5.0 -> [0.6, 0.8]
        // Row 1: [null, 6.0] -> norm L2 is 6.0 -> [null, 1.0]
        // Row 2: [5.0, NaN] -> norm L2 is 5.0 -> [1.0, NaN]
        let a = Column::from(Series::new("a".into(), &[Some(3.0f64), None, Some(5.0)]));
        let b = Column::from(Series::new(
            "b".into(),
            &[Some(4.0f64), Some(6.0), Some(f64::NAN)],
        ));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut n = Normalizer::l2();
        n.fit(df.clone()).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();

        assert_relative_eq!(col_a.get(0).unwrap(), 0.6, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 0.8, epsilon = 1e-6);

        assert!(col_a.get(1).is_none());
        assert_relative_eq!(col_b.get(1).unwrap(), 1.0, epsilon = 1e-6);

        assert_relative_eq!(col_a.get(2).unwrap(), 1.0, epsilon = 1e-6);
        assert!(col_b.get(2).unwrap().is_nan());
    }

    /// Regression for #181: extreme magnitude values (~1e300) must error
    /// on transform due to L2 norm overflow, instead of silently producing
    /// infinite norm which maps all features to 0.0.
    #[test]
    fn test_normalizer_l2_extreme_magnitude_overflow_error() {
        let a = Column::from(Series::new("a".into(), &[1e300_f64, 2e300]));
        let b = Column::from(Series::new("b".into(), &[3e300_f64, 4e300]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();
        let mut normalizer = Normalizer::l2();
        normalizer.fit(df.clone()).unwrap();
        let res = normalizer.transform(df);
        assert!(
            res.is_err(),
            "Normalizer L2 on 1e300 magnitude row must error on norm overflow"
        );
    }
}
