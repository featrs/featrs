//! Pipeline composition utilities.
//!
//! Analogous to `sklearn.pipeline` and `sklearn.compose`.
//! - [`Pipeline`] chains multiple transformers sequentially.
//! - [`ColumnTransformer`] applies different transformers to different column subsets.
//! - [`DataFrameTransformer`] is a trait alias for type erasure.

pub mod column_transformer;

pub use column_transformer::ColumnTransformer;

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Trait alias for [`Box<dyn ...>`](Box) type erasure in [`Pipeline`] and [`ColumnTransformer`].
///
/// Automatically implemented for any type that satisfies both
/// [`Fit<DataFrame, Output = ()>`](crate::traits::Fit) and
/// [`Transform<DataFrame, Output = DataFrame>`](crate::traits::Transform).
pub trait DataFrameTransformer:
    Fit<DataFrame, Output = ()> + Transform<DataFrame, Output = DataFrame>
{
}
impl<T> DataFrameTransformer for T where
    T: Fit<DataFrame, Output = ()> + Transform<DataFrame, Output = DataFrame>
{
}

/// Sequential pipeline of data transformations.
///
/// Each step is `(name, transformer)`. Calling `fit(X)` fits all steps
/// sequentially (passing each step's output into the next). Calling
/// `transform(X)` passes data through every step.
///
/// # Validation
///
/// Both `fit` and `transform` reject a `DataFrame` with 0 rows
/// ([`Error::InvalidInput`]) — the same policy [`ColumnTransformer`] applies.
/// `fit` rejects an empty step list in [`Pipeline::new`]. There is no
/// column-count requirement: a step that needs columns of its own reports that
/// itself. Because the row check runs in `transform` too, a step that drops
/// every row makes a nested `Pipeline` downstream of it fail rather than pass an
/// empty frame along; no transformer in this crate drops rows.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::Pipeline;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut pipeline = Pipeline::new(vec![
///     ("scale".into(), Box::new(StandardScaler::new())),
/// ])?;
/// pipeline.fit(df.clone())?;
/// let result = pipeline.transform(df)?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Pipeline {
    steps: Vec<(String, Box<dyn DataFrameTransformer>)>,
    fitted: bool,
}

impl Pipeline {
    /// Create a new pipeline with the given steps.
    ///
    /// Each step is a `(name, transformer)` pair. The name is used for
    /// inspection and debugging.
    ///
    /// Returns [`Error::InvalidInput`] if `steps` is empty.
    pub fn new(steps: Vec<(String, Box<dyn DataFrameTransformer>)>) -> Result<Self> {
        if steps.is_empty() {
            return Err(Error::InvalidInput(
                "Pipeline::new: at least one step is required. \
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
    pub fn steps(&self) -> &[(String, Box<dyn DataFrameTransformer>)] {
        &self.steps
    }
}

/// Shared empty-frame policy for the composition layer: `fit` and `transform`
/// both reject a frame with 0 rows, in [`Pipeline`] and `ColumnTransformer`
/// alike. Column-count emptiness is checked separately where it applies.
pub(crate) fn ensure_non_empty_rows(x: &DataFrame, who: &str) -> Result<()> {
    if x.height() == 0 {
        return Err(Error::InvalidInput(format!(
            "{who} received a DataFrame with 0 rows."
        )));
    }
    Ok(())
}

fn wrap_step_error(e: Error, i: usize, name: &str, phase: &str) -> Error {
    let msg = format!(
        "Pipeline: step {} ('{}') failed during {}: {}",
        i, name, phase, e,
    );
    match e {
        Error::NotFitted(_) => Error::NotFitted(msg),
        Error::InvalidInput(_) => Error::InvalidInput(msg),
        Error::Computation(_) => Error::Computation(msg),
    }
}

impl Fit<DataFrame> for Pipeline {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        ensure_non_empty_rows(&x, "Pipeline.fit")?;
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

impl Transform<DataFrame> for Pipeline {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Pipeline has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        ensure_non_empty_rows(&x, "Pipeline.transform")?;
        let mut x_curr = x;
        for (i, (name, transformer)) in self.steps.iter().enumerate() {
            x_curr = transformer
                .transform(x_curr)
                .map_err(|e| wrap_step_error(e, i, name, "transform"))?;
        }
        Ok(x_curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::binarizer::Binarizer;
    use crate::preprocessing::scaler::StandardScaler;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_pipeline_single_step() {
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();

        pipeline.fit(df.clone()).unwrap();
        let result = pipeline.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_pipeline_multi_step() {
        let scaler = StandardScaler::new();
        let binarizer = Binarizer::new(0.0);
        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(scaler)),
            ("binarizer".into(), Box::new(binarizer)),
        ])
        .unwrap();
        let df = make_test_df();

        pipeline.fit(df.clone()).unwrap();
        let result = pipeline.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_pipeline_empty_steps_error() {
        let result = Pipeline::new(vec![]);
        assert!(result.is_err(), "Pipeline::new must reject empty steps");
    }

    #[test]
    fn test_pipeline_not_fitted() {
        let scaler = StandardScaler::new();
        let pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();
        let err = pipeline.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted, got {err:?}",
        );
    }

    #[test]
    fn test_pipeline_fit_preserves_invalid_input() {
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let int_col = Column::from(Series::new("x".into(), &[1_i64, 2, 3]));
        let df_no_f64 = DataFrame::new(3, vec![int_col]).unwrap();
        let err = pipeline.fit(df_no_f64).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
    }

    #[test]
    fn test_pipeline_transform_preserves_invalid_input() {
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();
        pipeline.fit(df.clone()).unwrap();

        // Transform a dataframe missing column "a", causing an InvalidInput error.
        let col_b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        let df_missing_a = DataFrame::new(3, vec![col_b]).unwrap();
        let err = pipeline.transform(df_missing_a).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
    }

    #[test]
    fn test_pipeline_transform_rejects_zero_row_frame() {
        // `fit` rejects a 0-row frame but `transform` happily runs each step on
        // one, so the same degenerate input behaves differently per phase.
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();
        pipeline.fit(df.clone()).unwrap();

        let err = pipeline.transform(df.head(Some(0))).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("0 rows"), "{err}");
    }

    #[test]
    fn test_pipeline_failed_refit_resets_fitted() {
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();

        // First fit succeeds.
        pipeline.fit(df.clone()).unwrap();
        assert!(pipeline.transform(df.clone()).is_ok());

        // Second fit on invalid data fails — fitted should be reset.
        let int_col = Column::from(Series::new("x".into(), &[1_i64, 2, 3]));
        let df_no_f64 = DataFrame::new(3, vec![int_col]).unwrap();
        assert!(pipeline.fit(df_no_f64).is_err());

        // Transform should now return NotFitted, not silently use old state.
        let err = pipeline.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted after failed refit, got {err:?}",
        );
    }
}
