//! Feature union: apply several transformers to the same input in parallel.
//!
//! [`FeatureUnion`] mirrors `sklearn.pipeline.FeatureUnion`. Every transformer
//! receives the **full** input DataFrame and their outputs are concatenated
//! horizontally, so independent feature extractors operating on the same
//! columns contribute complementary features side by side.

use crate::pipeline::DataFrameTransformer;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashMap;

/// Apply multiple transformers to the same input and concatenate their outputs.
///
/// Unlike [`ColumnTransformer`](crate::pipeline::ColumnTransformer), which
/// partitions the columns among transformers, `FeatureUnion` hands the whole
/// input to each transformer and stacks the resulting columns horizontally.
///
/// Duplicate output column names are rejected: [`Transform::transform`] fails
/// before stacking, naming the two transformers that produced the same column.
/// Every transformer output must keep the input row count.
///
/// Note that most transformers in this crate preserve their input column names,
/// so two transformers that both receive the full frame will collide on those
/// names. Pair the union with transformers that emit distinct columns — for
/// example a [`ColumnTransformer`](crate::pipeline::ColumnTransformer) that
/// drops the remainder.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::feature_union::FeatureUnion;
/// use featrs::pipeline::ColumnTransformer;
/// use featrs::pipeline::column_transformer::Remainder;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
/// let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut union = FeatureUnion::new(vec![
///     (
///         "scaled_a".into(),
///         Box::new(ColumnTransformer::new(
///             vec![("s".into(), Box::new(StandardScaler::new()), vec!["a".into()])],
///             Remainder::Drop,
///         )),
///     ),
///     (
///         "scaled_b".into(),
///         Box::new(ColumnTransformer::new(
///             vec![("s".into(), Box::new(StandardScaler::new()), vec!["b".into()])],
///             Remainder::Drop,
///         )),
///     ),
/// ])?;
/// union.fit(df.clone())?;
/// let out = union.transform(df)?;
/// assert_eq!(out.width(), 2);
/// assert_eq!(out.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FeatureUnion {
    transformers: Vec<(String, Box<dyn DataFrameTransformer>)>,
    fitted: bool,
}

impl FeatureUnion {
    /// Create a new feature union from `(name, transformer)` pairs.
    ///
    /// The names are used in error messages. Returns [`Error::InvalidInput`]
    /// if `transformers` is empty.
    pub fn new(transformers: Vec<(String, Box<dyn DataFrameTransformer>)>) -> Result<Self> {
        if transformers.is_empty() {
            return Err(Error::InvalidInput(
                "FeatureUnion::new: at least one transformer is required. \
                 Provide a non-empty Vec of (name, transformer) pairs."
                    .into(),
            ));
        }
        Ok(Self {
            transformers,
            fitted: false,
        })
    }

    /// Returns a reference to the union's transformers.
    pub fn transformers(&self) -> &[(String, Box<dyn DataFrameTransformer>)] {
        &self.transformers
    }

    /// Number of transformers in the union.
    ///
    /// This counts the configured sub-transformers, not the columns they
    /// produce.
    pub fn n_components(&self) -> usize {
        self.transformers.len()
    }
}

impl Default for FeatureUnion {
    /// Create an empty union with no transformers.
    ///
    /// An empty union is not usable: [`Fit::fit`] rejects it as invalid input
    /// and [`Transform::transform`] reports that no output columns were
    /// produced. Prefer [`FeatureUnion::new`], which rejects the empty case up
    /// front.
    fn default() -> Self {
        Self {
            transformers: Vec::new(),
            fitted: false,
        }
    }
}

fn wrap_error(e: Error, name: &str, phase: &str) -> Error {
    let msg = format!(
        "FeatureUnion: transformer '{}' failed during {}: {}",
        name, phase, e
    );
    match e {
        Error::NotFitted(_) => Error::NotFitted(msg),
        Error::InvalidInput(_) => Error::InvalidInput(msg),
        Error::Computation(_) => Error::Computation(msg),
    }
}

impl Fit<DataFrame> for FeatureUnion {
    type Output = ();

    /// Fit every transformer on the full input DataFrame.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset up front: a failed re-fit must not leave stale state behind.
        self.fitted = false;

        if self.transformers.is_empty() {
            return Err(Error::InvalidInput(
                "FeatureUnion.fit: the union has no transformers. \
                 Construct it with FeatureUnion::new and at least one transformer."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "FeatureUnion.fit received a DataFrame with 0 rows.".into(),
            ));
        }

        for (name, transformer) in self.transformers.iter_mut() {
            transformer
                .fit(x.clone())
                .map_err(|e| wrap_error(e, name, "fit"))?;
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for FeatureUnion {
    type Output = DataFrame;

    /// Transform the input with every transformer and stack the outputs.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "FeatureUnion has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut parts: Vec<(&str, DataFrame)> = Vec::with_capacity(self.transformers.len());

        for (name, transformer) in &self.transformers {
            let out = transformer
                .transform(x.clone())
                .map_err(|e| wrap_error(e, name, "transform"))?;
            // Check the row count first: a transformer that drops every column
            // still has to preserve the input height, and a zero-width frame
            // must not be able to skip the check.
            if out.height() != x.height() {
                return Err(Error::InvalidInput(format!(
                    "FeatureUnion: transformer '{}' returned {} rows but the input has {} rows.",
                    name,
                    out.height(),
                    x.height()
                )));
            }
            if out.width() == 0 {
                // A transformer that drops every column contributes nothing.
                continue;
            }
            parts.push((name.as_str(), out));
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "FeatureUnion produced no output columns. \
                 At least one transformer must return at least one column."
                    .into(),
            ));
        }

        // Detect duplicate output names before stacking: hstack would fail with
        // an opaque error, and the useful information is which transformers clash.
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for (owner, part) in &parts {
            for col in part.get_column_names() {
                if let Some(prev) = seen.insert(col.as_str(), owner) {
                    return Err(Error::InvalidInput(format!(
                        "FeatureUnion: transformers '{}' and '{}' both produce a column \
                         named '{}'. Rename one of the outputs.",
                        prev, owner, col
                    )));
                }
            }
        }

        let mut result = parts.remove(0).1;
        for (_, other) in &parts {
            result = result.hstack(other.columns()).map_err(|e| {
                Error::Computation(format!(
                    "FeatureUnion: failed to stack transformer outputs: {}",
                    e
                ))
            })?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ColumnTransformer;
    use crate::pipeline::column_transformer::Remainder;
    use crate::preprocessing::binarizer::Binarizer;
    use crate::preprocessing::scaler::StandardScaler;

    /// Test-only transformer that keeps only the first row, so the row-count
    /// validation can be exercised.
    struct RowDropper;

    impl Fit<DataFrame> for RowDropper {
        type Output = ();
        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for RowDropper {
        type Output = DataFrame;
        fn transform(&self, x: DataFrame) -> Result<DataFrame> {
            Ok(x.head(Some(1)))
        }
    }

    /// Test-only transformer that drops every column but keeps the row count.
    struct ColumnDropper;

    impl Fit<DataFrame> for ColumnDropper {
        type Output = ();
        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for ColumnDropper {
        type Output = DataFrame;
        fn transform(&self, x: DataFrame) -> Result<DataFrame> {
            let none: Vec<&str> = Vec::new();
            x.select(none)
                .map_err(|e| Error::Computation(e.to_string()))
        }
    }

    /// Test-only transformer that returns a completely empty frame, dropping
    /// every column AND every row.
    struct EmptyFrameDropper;

    impl Fit<DataFrame> for EmptyFrameDropper {
        type Output = ();
        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for EmptyFrameDropper {
        type Output = DataFrame;
        fn transform(&self, _x: DataFrame) -> Result<DataFrame> {
            Ok(DataFrame::empty())
        }
    }

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_feature_union_complementary_outputs() {
        // Each column transformer drops the remainder, so the two branches emit
        // disjoint columns (scaling "a" vs binarizing "b") — the classic
        // FeatureUnion use case.
        let mut union = FeatureUnion::new(vec![
            (
                "scale_a".into(),
                Box::new(ColumnTransformer::new(
                    vec![(
                        "s".into(),
                        Box::new(StandardScaler::new()),
                        vec!["a".into()],
                    )],
                    Remainder::Drop,
                )),
            ),
            (
                "binarize_b".into(),
                Box::new(ColumnTransformer::new(
                    vec![("b".into(), Box::new(Binarizer::new(2.0)), vec!["b".into()])],
                    Remainder::Drop,
                )),
            ),
        ])
        .unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let result = union.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
        assert_eq!(result.get_column_names(), &["a", "b"]);
        assert_eq!(union.n_components(), 2);
        assert_eq!(union.transformers().len(), 2);
    }

    #[test]
    fn test_feature_union_single_transformer_matches_output() {
        let mut scaler = StandardScaler::new();
        let df = make_test_df();
        scaler.fit(df.clone()).unwrap();
        let expected = scaler.transform(df.clone()).unwrap();

        let mut union =
            FeatureUnion::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
        union.fit(df.clone()).unwrap();
        let result = union.transform(df).unwrap();

        assert_eq!(result.get_column_names(), expected.get_column_names());
        assert_eq!(result.height(), expected.height());
        assert_eq!(
            result.column("a").unwrap().f64().unwrap().get(1),
            expected.column("a").unwrap().f64().unwrap().get(1)
        );
    }

    #[test]
    fn test_feature_union_duplicate_output_columns_error() {
        let mut union = FeatureUnion::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("binarize".into(), Box::new(Binarizer::new(2.0))),
        ])
        .unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("'scale'"),
            "message should name the transformers: {msg}"
        );
        assert!(
            msg.contains("'binarize'"),
            "message should name the transformers: {msg}"
        );
        assert!(msg.contains("'a'"), "message should name the column: {msg}");
    }

    #[test]
    fn test_feature_union_empty_transformers_error_at_new() {
        let result = FeatureUnion::new(vec![]);
        assert!(
            result.is_err(),
            "FeatureUnion::new must reject an empty transformer list"
        );
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_feature_union_default_is_empty_and_unusable() {
        let mut union = FeatureUnion::default();
        assert_eq!(union.n_components(), 0);

        let df = make_test_df();
        assert!(matches!(union.fit(df.clone()), Err(Error::InvalidInput(_))));

        // A transform on an unfitted union reports NotFitted, never a panic.
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted, got {err:?}",
        );
    }

    #[test]
    fn test_feature_union_not_fitted() {
        let union =
            FeatureUnion::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
        let df = make_test_df();
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted, got {err:?}",
        );
        assert_eq!(
            err.to_string(),
            "not fitted: FeatureUnion has not been fitted. \
             Call .fit(dataframe) before .transform()."
        );
    }

    #[test]
    fn test_feature_union_empty_input_error() {
        let mut union =
            FeatureUnion::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
        let err = union.fit(DataFrame::empty()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
    }

    #[test]
    fn test_feature_union_failed_refit_resets_fitted() {
        let mut union =
            FeatureUnion::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        assert!(union.transform(df.clone()).is_ok());

        assert!(union.fit(DataFrame::empty()).is_err());
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted after failed refit, got {err:?}",
        );
    }

    #[test]
    fn test_feature_union_row_count_mismatch_error() {
        let mut union = FeatureUnion::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("drop_rows".into(), Box::new(RowDropper)),
        ])
        .unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("'drop_rows'"));
    }

    #[test]
    fn test_feature_union_all_transformers_drop_columns_error() {
        let mut union = FeatureUnion::new(vec![("drop".into(), Box::new(ColumnDropper))]).unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("produced no output columns"));
    }

    #[test]
    fn test_feature_union_column_dropper_alongside_others_is_skipped() {
        let mut union = FeatureUnion::new(vec![
            ("drop".into(), Box::new(ColumnDropper)),
            ("binarize".into(), Box::new(Binarizer::new(2.0))),
        ])
        .unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let result = union.transform(df).unwrap();
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_feature_union_empty_width_output_still_checks_row_count() {
        // Regression guard: a transformer that drops every column must not be
        // able to skip the row-count check by returning a 0x0 frame.
        let mut union = FeatureUnion::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("empty".into(), Box::new(EmptyFrameDropper)),
        ])
        .unwrap();
        let df = make_test_df();

        union.fit(df.clone()).unwrap();
        let err = union.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("'empty'"), "got {err}");
    }

    #[test]
    fn test_feature_union_fit_error_names_transformer() {
        // StandardScaler rejects a frame without Float64 columns.
        let int_col = Column::from(Series::new("x".into(), &[1_i64, 2, 3]));
        let df_no_f64 = DataFrame::new(3, vec![int_col]).unwrap();
        let mut union =
            FeatureUnion::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();

        let err = union.fit(df_no_f64).unwrap_err();
        assert!(err.to_string().contains("'scale'"), "got {err}");
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}
