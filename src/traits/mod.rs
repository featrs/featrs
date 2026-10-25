//! Core traits and error types for the featrs library.
//!
//! The library is built around four traits that mirror the scikit-learn API:
//!
//! - [`Fit`] — learn parameters from unsupervised data (`fit(X)`)
//! - [`FitSupervised`] — learn parameters from data plus a target (`fit(X, y)`)
//! - [`Transform`] — apply a learned transformation (`transform`)
//! - [`FitTransform`] — convenience trait that provides a default
//!   `fit_transform(X)` for any type implementing both [`Fit`] and [`Transform`]
//!
//! Schema validation lives in [`schema`]: transformers that can describe their
//! input implement [`schema::RequiresSchema`], and callers use
//! [`schema::validate_schema`] (or [`schema::check_schema`] for the structured
//! violations) to check a frame against that description up front.
//!
//! Unsupervised transformers (scalers, encoders, imputers, …) implement
//! [`Fit`]; the few supervised ones (e.g. `SelectKBest`) implement
//! [`FitSupervised`] instead. This mirrors scikit-learn's split between
//! `fit(X)` and `fit(X, y)`, so callers no longer pass a dummy target to
//! unsupervised transformers.
//!
//! # Errors
//!
//! All fallible operations return [`Result<T>`], which wraps [`enum@Error`].
//! [`enum@Error`] has three variants:
//! - [`Error::InvalidInput`] — wrong dimensions, types, or empty data
//! - [`Error::NotFitted`] — `transform` called before `fit`
//! - [`Error::Computation`] — numerical issues (zero variance, singular matrices, etc.)

use thiserror::Error;

pub mod schema;

pub use schema::{
    ColumnRequirement, RequiresSchema, SchemaSpec, SchemaViolation, check_schema, validate_schema,
};

/// Errors that can occur during feature engineering operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Input data has invalid shape, type, or is empty.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// `transform` was called before `fit`.
    #[error("not fitted: {0}")]
    NotFitted(String),

    /// Numerical computation failed (e.g. zero variance, singular matrix).
    #[error("computation error: {0}")]
    Computation(String),
}

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Learn parameters from unsupervised data.
///
/// `X` is the feature data (e.g. `DataFrame`). Implement this trait for
/// transformers that do not need a target (scalers, encoders, imputers, …).
/// Supervised transformers implement [`FitSupervised`] instead.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Fit<X> {
    /// The type returned by `fit`. Usually `()`.
    type Output;

    /// Fit the transformer to the data.
    ///
    /// After calling `fit`, the transformer stores the learned parameters
    /// internally. Calling `transform` before `fit` returns
    /// [`Error::NotFitted`].
    fn fit(&mut self, x: X) -> Result<Self::Output>;
}

/// Learn parameters from data plus a supervised target.
///
/// `X` is the feature data and `Y` is the target data (both typically
/// `DataFrame`). Implement this trait for transformers that need a target,
/// such as [`SelectKBest`](crate::feature_selection::SelectKBest). Unsupervised
/// transformers implement [`Fit`] instead.
pub trait FitSupervised<X, Y> {
    /// The type returned by `fit`. Usually `()`.
    type Output;

    /// Fit the transformer to the data and target.
    ///
    /// After calling `fit`, the transformer stores the learned parameters
    /// internally. Calling `transform` before `fit` returns
    /// [`Error::NotFitted`].
    fn fit(&mut self, x: X, y: Y) -> Result<Self::Output>;
}

/// Apply a learned transformation to data.
///
/// `X` is the input data (e.g. `DataFrame`). The output type [`Output`](Transform::Output)
/// is typically also `DataFrame`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Transform<X> {
    /// The type of the transformed output.
    type Output;

    /// Transform the data using the fitted parameters.
    ///
    /// Returns [`Error::NotFitted`] if the transformer has not been fitted yet.
    fn transform(&self, x: X) -> Result<Self::Output>;
}

/// Convenience trait for types that implement both [`Fit`] and [`Transform`].
///
/// Automatically implemented for any type satisfying both bounds. Provides a
/// default [`fit_transform`](FitTransform::fit_transform) equivalent to
/// `fit(X)` followed by `transform(X)`; types may override it with an optimized
/// single-pass implementation. Also enables type erasure via
/// [`Box<dyn DataFrameTransformer>`](crate::pipeline::DataFrameTransformer).
pub trait FitTransform<X>: Fit<X> + Transform<X> {
    /// Fit to `x`, then transform `x`, returning the transformed output.
    ///
    /// Default implementation clones `x` so it can be both fit on and
    /// transformed; override for a single-pass implementation when `X` need
    /// not be cloned.
    fn fit_transform(&mut self, x: X) -> Result<<Self as Transform<X>>::Output>
    where
        X: Clone,
    {
        self.fit(x.clone())?;
        self.transform(x)
    }
}

impl<T, X> FitTransform<X> for T where T: Fit<X> + Transform<X> {}
