//! Feature hashing (the hashing trick).
//!
//! [`FeatureHasher`] maps categorical features into a fixed-size vector
//! using a hash function, enabling memory-efficient encoding of
//! high-cardinality categories. Uses the signed hashing trick
//! (Weinberger et al. 2009): a second independent hash determines the sign
//! (`+1.0` / `-1.0`) of each addition, so the expected value of every bucket
//! is zero and collisions do not bias the mean.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Map categorical columns to a fixed number of hash buckets.
///
/// Each string cell is mapped to a `(bucket, sign)` pair via two independent
/// hashes; the bucket is incremented by `sign` (`+1.0` or `-1.0`). This avoids
/// storing a category mapping, works with unseen categories at transform time,
/// and (unlike an unsigned trick) preserves a zero mean in expectation.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::feature_hasher::FeatureHasher;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut fh = FeatureHasher::new(&["color"], 10);
/// fh.fit(df.clone())?;
/// let hashed = fh.transform(df)?;
/// assert_eq!(hashed.width(), 10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FeatureHasher {
    fitted: bool,
    columns: Vec<String>,
    n_features: usize,
}

impl FeatureHasher {
    /// Create a new `FeatureHasher`.
    ///
    /// * `columns` — names of string columns to hash
    /// * `n_features` — number of hash buckets (output dimension)
    pub fn new(columns: &[&str], n_features: usize) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            n_features,
        }
    }

    /// Map `s` to a `(bucket, sign)` pair. Two independent `DefaultHasher`
    /// instances (seeded differently) produce the index and the sign bit, so
    /// collisions across the two are uncorrelated.
    fn hash_to_bucket(s: &str, n: usize) -> (usize, f64) {
        let mut h_idx = DefaultHasher::new();
        0u8.hash(&mut h_idx);
        s.hash(&mut h_idx);
        let idx = (h_idx.finish() as usize) % n;

        let mut h_sign = DefaultHasher::new();
        1u8.hash(&mut h_sign);
        s.hash(&mut h_sign);
        let sign = if h_sign.finish() & 1 == 1 { 1.0 } else { -1.0 };

        (idx, sign)
    }
}

impl Fit<DataFrame> for FeatureHasher {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale state.
        self.fitted = false;

        if self.n_features == 0 {
            return Err(Error::InvalidInput(
                "FeatureHasher: n_features must be >= 1.".into(),
            ));
        }
        for col in &self.columns {
            let s = x.column(col.as_str()).map_err(|_| {
                Error::InvalidInput(format!("FeatureHasher.fit: column '{}' not found.", col))
            })?;
            let s = s.as_materialized_series();
            s.str().map_err(|e| {
                Error::InvalidInput(format!(
                    "FeatureHasher.fit: column '{}' has dtype {}; expected String. {}",
                    col,
                    s.dtype(),
                    e
                ))
            })?;
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for FeatureHasher {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("FeatureHasher".into()));
        }
        let n_rows = x.height();
        let mut buckets = vec![vec![0.0f64; n_rows]; self.n_features];

        for col in &self.columns {
            let s = x
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "FeatureHasher.transform: column '{}' not found. {}",
                        col, e
                    ))
                })?
                .as_materialized_series();
            let ca = s.str().map_err(|e| {
                Error::InvalidInput(format!(
                    "FeatureHasher.transform: column '{}' has dtype {}; expected String. {}",
                    col,
                    s.dtype(),
                    e
                ))
            })?;
            for (i, opt) in ca.iter().enumerate() {
                if let Some(val) = opt {
                    let (idx, sign) = Self::hash_to_bucket(val, self.n_features);
                    buckets[idx][i] += sign;
                }
            }
        }

        let mut out_cols: Vec<Column> = Vec::with_capacity(self.n_features);
        for (idx, bucket) in buckets.iter().enumerate() {
            let name = format!("hashed_{}", idx);
            out_cols.push(Column::from(Series::new(name.as_str().into(), bucket)));
        }

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_hasher() {
        let c = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
        let df = DataFrame::new(3, vec![c]).unwrap();
        let mut fh = FeatureHasher::new(&["color"], 10);

        fh.fit(df.clone()).unwrap();
        let result = fh.transform(df).unwrap();

        assert_eq!(result.width(), 10);
        assert_eq!(result.height(), 3);
    }

    /// The signed hashing trick: `hash_to_bucket` returns a valid index and a
    /// sign of exactly `+1.0` or `-1.0`, and is deterministic across calls.
    #[test]
    fn test_hash_to_bucket_signed_and_deterministic() {
        for s in &["red", "blue", "green", "x", "y", "a very long category"] {
            let (idx, sign) = FeatureHasher::hash_to_bucket(s, 64);
            assert!(idx < 64, "index out of range for '{s}'");
            assert!(sign == 1.0 || sign == -1.0, "sign must be ±1 for '{s}'");
            // Determinism: same input, same output.
            let (idx2, sign2) = FeatureHasher::hash_to_bucket(s, 64);
            assert_eq!((idx, sign), (idx2, sign2));
        }
    }

    /// Every non-zero bucket value produced by `transform` must be an integer
    /// in `-1..=1` per single occurrence; summing repeated occurrences stays
    /// an integer (the signed trick never produces fractional magnitudes).
    #[test]
    fn test_feature_hasher_signed_values() {
        let c = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green", "blue"],
        ));
        let df = DataFrame::new(5, vec![c]).unwrap();
        let mut fh = FeatureHasher::new(&["color"], 32);
        fh.fit(df.clone()).unwrap();
        let result = fh.transform(df).unwrap();

        for col in result.columns() {
            for v in col.as_materialized_series().f64().unwrap().iter().flatten() {
                let frac = v.fract();
                assert!(frac == 0.0, "bucket value {v} must be integral");
            }
        }
    }

    /// `fit` must reject non-String columns early instead of pushing the
    /// dtype error to `transform` time (regression test for issue #6).
    #[test]
    fn test_fit_rejects_non_string_column() {
        let c = Column::from(Series::new("count".into(), &[1_i32, 2, 3]));
        let df = DataFrame::new(3, vec![c]).unwrap();
        let mut fh = FeatureHasher::new(&["count"], 8);

        let err = fh.fit(df.clone()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected String"),
            "error should mention expected String, got: {msg}"
        );

        // A failed fit must not mark the hasher as fitted; transform then
        // surfaces `NotFitted` rather than attempting to hash a numeric column.
        let transform_err = fh.transform(df).unwrap_err().to_string();
        assert!(
            transform_err.contains("not fitted"),
            "transform after a failed fit should report NotFitted, got: {transform_err}"
        );
    }
}
