//! Reversing a transformation: mapping output features back to the input domain.
//!
//! [`InverseTransform`] mirrors scikit-learn's `inverse_transform`: a fitted
//! transformer that stores enough state can map its output back to the units of
//! its input. This is useful for interpreting model predictions in original
//! units, for sanity-checking a pipeline step (`inverse(transform(x)) ≈ x`),
//! and for generating synthetic data in the input domain.
//!
//! Only mathematically invertible transformers implement the trait. Encoders,
//! binarizers, discretizers and feature selectors do not: they either collapse
//! several inputs onto one output or drop information outright.
//!
//! # Semantics
//!
//! - Calling `inverse_transform` before `fit` returns [`Error::NotFitted`](crate::traits::Error::NotFitted).
//! - The inverse is only meaningful for data that came out of `transform`.
//!   Values outside the transform's range are extrapolated rather than
//!   rejected, so the caller owns that contract.
//! - Roundtrips are numerically approximate: compare with a tolerance, never
//!   with exact equality.
//!
//! # Example
//!
//! ```rust
//! use featrs::preprocessing::scaler::StandardScaler;
//! use featrs::traits::inverse::InverseTransform;
//! use featrs::traits::{Fit, Transform};
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![col])?;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df.clone())?;
//! let scaled = scaler.transform(df)?;
//! let restored = scaler.inverse_transform(scaled)?;
//!
//! let restored_col = restored.column("x")?.f64()?;
//! let vals: Vec<f64> = restored_col.iter().flatten().collect();
//! assert!((vals[0] - 1.0).abs() < 1e-12);
//! assert!((vals[2] - 3.0).abs() < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::traits::Result;

/// A transformer that can reverse its transformation.
///
/// Implementors store the state needed to undo `transform` during `fit`
/// (means, scales, quantile references, …) and apply the inverse formula in
/// `inverse_transform`.
pub trait InverseTransform<X> {
    /// The type of the inverse-transformed output. Usually also `DataFrame`.
    type Output;

    /// Map data from the transformer's output space back to its input space.
    ///
    /// Returns [`Error::NotFitted`](crate::traits::Error::NotFitted) if called
    /// before `fit`, and
    /// [`Error::InvalidInput`](crate::traits::Error::InvalidInput) when the
    /// inverse is undefined for the given values (e.g. the Box-Cox inverse of a
    /// value whose base would be non-positive).
    fn inverse_transform(&self, x: X) -> Result<Self::Output>;
}

/// Capability introspection for transformers that may or may not be invertible.
///
/// The default implementation reports `true`, so types only implement this
/// trait when they genuinely support [`InverseTransform`]; callers holding a
/// heterogeneous set of transformers can then ask whether an inverse exists
/// before attempting one.
pub trait HasInverse {
    /// Whether this transformer implements [`InverseTransform`].
    fn is_invertible(&self) -> bool {
        true
    }
}
