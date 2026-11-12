//! Lazy counterparts of [`Fit`](crate::traits::Fit) and [`Transform`](crate::traits::Transform) that operate on Polars
//! [`LazyFrame`]s.
//!
//! Available with the `lazy` Cargo feature. That feature needs Rust 1.95 or
//! newer, because the Polars lazy engine compiles `polars-stream`.
//!
//! [`TransformLazy`] lets a transformer add its work to a query plan without
//! materialising data, so the plan can be composed with other lazy operations
//! (filters, joins, projections) and collected once at the end. [`FitLazy`]
//! accepts a `LazyFrame` for fitting and collects it before the eager fit runs.

use polars::prelude::{DataFrame, DataType, LazyFrame, PolarsError};

use crate::traits::{Error, Result};

/// Reject plan columns that are not `Float64`, mirroring the eager
/// `{who}.transform` checks in [`crate::util`].
///
/// Resolves the plan's schema only. No rows are read.
pub(crate) fn require_lazy_f64_columns<'a>(
    x: &mut LazyFrame,
    columns: impl IntoIterator<Item = &'a str>,
    who: &str,
) -> Result<()> {
    let schema = x.collect_schema().map_err(|e| {
        Error::Computation(format!(
            "{who}.transform_lazy: could not resolve the LazyFrame schema. {e}"
        ))
    })?;
    for name in columns {
        let Some(dtype) = schema.get(name).cloned() else {
            return Err(Error::InvalidInput(format!(
                "{who}.transform: column '{name}' not found in the LazyFrame schema."
            )));
        };
        if dtype != DataType::Float64 {
            return Err(Error::InvalidInput(format!(
                "{who}.transform: column '{name}' has dtype {dtype}; expected Float64."
            )));
        }
    }
    Ok(())
}

/// Collect a plan for fitting, naming the transformer that asked.
pub(crate) fn collect_fit_frame(x: LazyFrame, who: &str) -> Result<DataFrame> {
    x.collect().map_err(|e: PolarsError| {
        Error::Computation(format!(
            "{who}.fit_lazy: could not collect the LazyFrame. {e}"
        ))
    })
}

/// Learn parameters from a [`LazyFrame`].
///
/// Implemented for [`StandardScaler`], [`MinMaxScaler`], [`Binarizer`],
/// [`LogTransformer`], and [`LazyPipeline`]. Each collects the frame and passes
/// it to [`Fit::fit`](crate::traits::Fit::fit), so the parameters match a fit
/// on the collected `DataFrame`. A failed collect leaves the transformer
/// unfitted, because the fitted state is reset before the plan runs.
///
/// [`StandardScaler`]: crate::preprocessing::scaler::StandardScaler
/// [`MinMaxScaler`]: crate::preprocessing::scaler::MinMaxScaler
/// [`Binarizer`]: crate::preprocessing::binarizer::Binarizer
/// [`LogTransformer`]: crate::preprocessing::log_transformer::LogTransformer
/// [`LazyPipeline`]: crate::pipeline::LazyPipeline
pub trait FitLazy {
    /// The type returned by `fit_lazy`. Usually `()`.
    type Output;

    /// Fit the transformer to the data held by `x`.
    ///
    /// Returns [`Error::Computation`] if the frame cannot be collected, and
    /// any error the eager `fit` returns otherwise. A failed collect leaves the
    /// transformer unfitted, so a later `transform` returns
    /// [`Error::NotFitted`] instead of applying stale
    /// parameters.
    fn fit_lazy(&mut self, x: LazyFrame) -> Result<Self::Output>;
}

/// Apply a learned transformation to a [`LazyFrame`] without collecting it.
///
/// Implementations append their expressions to the plan. Calling
/// `transform_lazy` before fitting returns [`Error::NotFitted`]. Collecting
/// the returned frame yields the same values as the eager
/// [`Transform::transform`](crate::traits::Transform::transform) on the
/// collected input.
pub trait TransformLazy {
    /// The type of the transformed output, usually [`LazyFrame`].
    type Output;

    /// Add the transformation to the plan held by `x` and return the new plan.
    ///
    /// Returns [`Error::NotFitted`] if the transformer has not been fitted yet.
    fn transform_lazy(&self, x: LazyFrame) -> Result<Self::Output>;
}
