//! Lazy pipeline composition over Polars [`LazyFrame`]s.
//!
//! Available with the `lazy` Cargo feature, which needs Rust 1.95 or newer.
//!
//! [`LazyPipeline`] chains steps that implement [`TransformLazy`]. Calling
//! `transform_lazy` appends every step to one query plan, so the whole chain
//! runs only when the caller collects the result. [`Pipeline`](crate::pipeline::Pipeline)
//! stays eager: its steps are type-erased behind a trait whose vtable has no
//! lazy method, so lazy support lives in this separate type.

use crate::pipeline::ensure_non_empty_rows;
use crate::traits::lazy::collect_fit_frame;
use crate::traits::{Error, FeatureNamesOut, Fit, FitLazy, Result, Transform, TransformLazy};
use polars::prelude::{DataFrame, LazyFrame};

/// Trait alias for `Box<dyn ...>` type erasure in [`LazyPipeline`].
///
/// Automatically implemented for any type that is a [`DataFrameTransformer`]
/// (fit on and transform a `DataFrame`) and also implements
/// [`TransformLazy<Output = LazyFrame>`](TransformLazy). [`FeatureNamesOut`] is
/// a supertrait, so every step can report its output schema.
///
/// [`DataFrameTransformer`]: crate::pipeline::DataFrameTransformer
pub trait LazyTransformer:
    Fit<DataFrame, Output = ()>
    + Transform<DataFrame, Output = DataFrame>
    + TransformLazy<Output = LazyFrame>
    + FeatureNamesOut
{
}

impl<T> LazyTransformer for T where
    T: Fit<DataFrame, Output = ()>
        + Transform<DataFrame, Output = DataFrame>
        + TransformLazy<Output = LazyFrame>
        + FeatureNamesOut
{
}

/// Sequential pipeline whose transform stays lazy.
///
/// Each step is `(name, transformer)`. `fit` (or `fit_lazy`) fits each step on
/// the eager `DataFrame` and runs the eager `transform` between steps, so
/// fitting matches [`Pipeline::fit`](crate::pipeline::Pipeline). `transform_lazy`
/// appends every step to the plan of the input `LazyFrame` and returns it
/// without collecting.
///
/// # Validation
///
/// `new` rejects an empty step list. `fit` rejects a `DataFrame` with 0 rows,
/// like [`Pipeline`](crate::pipeline::Pipeline). `transform_lazy` cannot check
/// the row count without collecting, so it does not check it.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::{LazyPipeline, LazyTransformer};
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{FitLazy, TransformLazy};
/// use polars::prelude::{Column, DataFrame, IntoLazy, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let steps: Vec<(String, Box<dyn LazyTransformer>)> =
///     vec![("scale".into(), Box::new(StandardScaler::new()))];
/// let mut pipeline = LazyPipeline::new(steps)?;
/// pipeline.fit_lazy(df.clone().lazy())?;
/// let result = pipeline.transform_lazy(df.lazy())?.collect()?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LazyPipeline {
    steps: Vec<(String, Box<dyn LazyTransformer>)>,
    fitted: bool,
}

impl LazyPipeline {
    /// Create a new lazy pipeline with the given steps.
    ///
    /// Returns [`Error::InvalidInput`] if `steps` is empty.
    pub fn new(steps: Vec<(String, Box<dyn LazyTransformer>)>) -> Result<Self> {
        if steps.is_empty() {
            return Err(Error::InvalidInput(
                "LazyPipeline::new: at least one step is required. \
                 Provide a non-empty Vec of (name, transformer) pairs."
                    .into(),
            ));
        }
        Ok(Self {
            steps,
            fitted: false,
        })
    }

    /// Returns a reference to the pipeline steps.
    pub fn steps(&self) -> &[(String, Box<dyn LazyTransformer>)] {
        &self.steps
    }
}

fn wrap_step_error(e: Error, i: usize, name: &str, phase: &str) -> Error {
    let msg = format!(
        "LazyPipeline: step {} ('{}') failed during {}: {}",
        i, name, phase, e,
    );
    match e {
        Error::NotFitted(_) => Error::NotFitted(msg),
        Error::InvalidInput(_) => Error::InvalidInput(msg),
        Error::Computation(_) => Error::Computation(msg),
    }
}

impl Fit<DataFrame> for LazyPipeline {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        ensure_non_empty_rows(&x, "LazyPipeline.fit")?;
        let mut x_curr = x;
        let n = self.steps.len();
        for (i, (name, transformer)) in self.steps.iter_mut().enumerate() {
            let is_last = i == n - 1;
            transformer
                .fit(x_curr.clone())
                .map_err(|e| wrap_step_error(e, i, name, "fit"))?;
            if !is_last {
                x_curr = transformer
                    .transform(x_curr)
                    .map_err(|e| wrap_step_error(e, i, name, "intermediate transform"))?;
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl FitLazy for LazyPipeline {
    type Output = ();

    fn fit_lazy(&mut self, x: LazyFrame) -> Result<()> {
        self.fitted = false;
        let df = collect_fit_frame(x, "LazyPipeline")?;
        self.fit(df)
    }
}

impl TransformLazy for LazyPipeline {
    type Output = LazyFrame;

    fn transform_lazy(&self, x: LazyFrame) -> Result<LazyFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LazyPipeline has not been fitted. \
                 Call .fit(dataframe) or .fit_lazy(lazy_frame) before .transform_lazy()."
                    .into(),
            ));
        }
        let mut x_curr = x;
        for (i, (name, transformer)) in self.steps.iter().enumerate() {
            x_curr = transformer
                .transform_lazy(x_curr)
                .map_err(|e| wrap_step_error(e, i, name, "transform_lazy"))?;
        }
        Ok(x_curr)
    }
}

impl FeatureNamesOut for LazyPipeline {
    /// Thread the input names through every step's output plan, in order.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LazyPipeline has not been fitted. \
                 Call .fit(dataframe) or .fit_lazy(lazy_frame) before .feature_names_out()."
                    .into(),
            ));
        }
        let mut names = input_columns.to_vec();
        for (i, (name, transformer)) in self.steps.iter().enumerate() {
            names = transformer
                .feature_names_out(&names)
                .map_err(|e| wrap_step_error(e, i, name, "feature_names_out"))?;
        }
        Ok(names)
    }
}
