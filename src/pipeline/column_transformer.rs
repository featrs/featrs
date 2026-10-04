//! Column-wise transformation routing.
//!
//! [`ColumnTransformer`] applies different preprocessing pipelines to
//! different column subsets and combines the results into a single
//! [`DataFrame`].

use crate::pipeline::DataFrameTransformer;
use crate::pipeline::ensure_non_empty_rows;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};

/// How to handle columns not specified in any transformer.
pub enum Remainder {
    /// Drop unspecified columns from the output.
    Drop,
    /// Pass unspecified columns through unchanged.
    Passthrough,
}

/// Apply different transformers to different subsets of columns.
///
/// Each transformer receives only its designated columns and produces
/// transformed columns. The results are horizontally stacked.
///
/// # Validation
///
/// A frame with 0 rows is rejected by both `fit` and `transform`, matching
/// [`Pipeline`](crate::pipeline::Pipeline); `fit` additionally rejects a frame
/// with 0 columns. A configuration that cannot produce any output (no
/// transformers and `Remainder::Drop`) is rejected by `fit` rather than by
/// `transform`. `fit` rejects a column claimed by more than one transformer.
///
/// # Output column names
///
/// Transformers preserve their input names or generate new ones, and a
/// remainder passthrough re-emits its columns unchanged. Two sources may
/// therefore agree on a name (e.g. two transformers that both emit `hashed_0`,
/// or a generated `a_missing` next to a passthrough input column of the same
/// name). `transform` detects that before stacking and returns
/// [`Error::InvalidInput`] naming the column and both sources; there is no
/// silent precedence rule.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::ColumnTransformer;
/// use featrs::pipeline::column_transformer::Remainder;
/// use featrs::preprocessing::scaler::StandardScaler;
///
/// let _ct = ColumnTransformer::new(
///     vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
///     Remainder::Passthrough,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ColumnTransformer {
    transformers: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)>,
    remainder: Remainder,
    fitted: bool,
}

impl ColumnTransformer {
    /// Create a new `ColumnTransformer`.
    ///
    /// Each entry in `transformers` is `(name, transformer, columns)`:
    /// - `name`: identifier for debugging
    /// - `transformer`: any [`DataFrameTransformer`]
    /// - `columns`: column names to apply the transformer to
    ///
    /// `remainder` controls how columns not listed in any transformer are handled.
    ///
    /// Configuration is validated by [`Fit::fit`]: a transformer list that can
    /// never produce output (empty with [`Remainder::Drop`]) and a column
    /// claimed by more than one transformer are both rejected there. An empty
    /// list with [`Remainder::Passthrough`] is valid — it passes the whole
    /// frame through.
    pub fn new(
        transformers: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)>,
        remainder: Remainder,
    ) -> Self {
        Self {
            transformers,
            remainder,
            fitted: false,
        }
    }

    fn all_specified_columns(&self) -> HashSet<&str> {
        let mut cols = HashSet::new();
        for (_, _, columns) in &self.transformers {
            for c in columns {
                cols.insert(c.as_str());
            }
        }
        cols
    }
}

impl Fit<DataFrame> for ColumnTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        // Row count first: `DataFrame::empty()` is 0x0, and a 0-row frame built
        // with an empty column list is (0, 0) too, so the width check below
        // cannot distinguish them.
        ensure_non_empty_rows(&x, "ColumnTransformer.fit")?;
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "ColumnTransformer.fit received a DataFrame with 0 columns.".into(),
            ));
        }
        if self.transformers.is_empty() && matches!(self.remainder, Remainder::Drop) {
            return Err(Error::InvalidInput(
                "ColumnTransformer: no transformers were configured and the remainder is \
                 Remainder::Drop, so the output would have no columns. Provide at least one \
                 transformer or use Remainder::Passthrough."
                    .into(),
            ));
        }

        let mut seen: HashMap<&str, &str> = HashMap::new();
        for (t_name, _, columns) in &self.transformers {
            for c in columns {
                if let Some(prev) = seen.insert(c.as_str(), t_name.as_str()) {
                    return Err(Error::InvalidInput(format!(
                        "ColumnTransformer: column '{}' is specified by more than one \
                         transformer ('{}' and '{}'). Each column \
                         may belong to at most one transformer.",
                        c, prev, t_name
                    )));
                }
            }
        }

        for (t_name, transformer, columns) in &mut self.transformers {
            let col_names = columns.clone();
            let subset = x.select(&col_names).map_err(|e| {
                Error::InvalidInput(format!(
                    "ColumnTransformer: transformer '{}' requested columns {:?} \
                     but one or more don't exist in the input. Available columns: {:?}. {}",
                    t_name,
                    col_names,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                    e
                ))
            })?;
            transformer.fit(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during fit: {}",
                    t_name, e
                ))
            })?;
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for ColumnTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "ColumnTransformer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        ensure_non_empty_rows(&x, "ColumnTransformer.transform")?;
        // Each part carries the label used in collision errors.
        let mut parts: Vec<(String, DataFrame)> = Vec::new();

        for (t_name, transformer, columns) in &self.transformers {
            let subset = x.select(columns).map_err(|e| {
                Error::InvalidInput(format!(
                    "ColumnTransformer: transformer '{}' requested columns {:?} \
                     but one or more don't exist in the input. Available columns: {:?}. {}",
                    t_name,
                    columns,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                    e
                ))
            })?;
            let transformed = transformer.transform(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during transform: {}",
                    t_name, e
                ))
            })?;
            parts.push((format!("transformer '{}'", t_name), transformed));
        }

        let specified = self.all_specified_columns();
        match self.remainder {
            Remainder::Passthrough => {
                let remaining_cols: Vec<&str> = x
                    .get_column_names()
                    .iter()
                    .filter(|name| !specified.contains(name.as_str()))
                    .map(|s| s.as_str())
                    .collect();
                if !remaining_cols.is_empty() {
                    let rem = remaining_cols.clone();
                    let remaining = x.select(remaining_cols).map_err(|e| {
                        Error::InvalidInput(format!(
                            "ColumnTransformer: failed to select remainder columns {:?}: {}",
                            rem, e
                        ))
                    })?;
                    parts.push((format!("remainder passthrough {rem:?}"), remaining));
                }
            }
            Remainder::Drop => {}
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "ColumnTransformer produced no output columns. \
                 Check that at least one transformer has matching input columns \
                 or use Remainder::Passthrough to keep unspecified columns."
                    .into(),
            ));
        }

        // Stacking is the first point where the outgoing names are all known,
        // so the collision check has to happen here. `hstack` would replace the
        // earlier column silently (or fail on the frame construction) and name
        // neither source, which is what this replaces.
        let mut owner: HashMap<&str, &str> = HashMap::new();
        for (label, part) in &parts {
            for name in part.get_column_names() {
                if let Some(prev) = owner.insert(name.as_str(), label.as_str()) {
                    return Err(Error::InvalidInput(format!(
                        "ColumnTransformer: column '{}' is produced by more than one output \
                         source ({} and {}). Rename one of the transformers' output columns \
                         or drop the conflicting column from the passthrough remainder.",
                        name, prev, label
                    )));
                }
            }
        }

        let (_, mut result) = parts.remove(0);
        for (_, other) in parts {
            let cols = other.columns().to_vec();
            result = result.hstack(&cols).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: failed to stack transformed columns: {}",
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
    use crate::preprocessing::feature_hasher::FeatureHasher;
    use crate::preprocessing::missing_indicator::MissingIndicator;
    use crate::preprocessing::scaler::StandardScaler;
    use crate::traits::Transform;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        let c = Column::from(Series::new("c".into(), &[10.0f64, 20.0, 30.0]));
        DataFrame::new(3, vec![a, b, c]).unwrap()
    }

    #[test]
    fn test_column_transformer_selective() {
        let scaler = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Passthrough,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_column_transformer_drop_remainder() {
        let scaler = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Drop,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        // Only the transformed "a" column is kept; b and c are dropped.
        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_column_transformer_multiple() {
        let scaler_a = StandardScaler::new();
        let scaler_b = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![
                ("scale_a".into(), Box::new(scaler_a), vec!["a".into()]),
                ("scale_b".into(), Box::new(scaler_b), vec!["b".into()]),
            ],
            Remainder::Passthrough,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        // a (scaled) + b (scaled) + c (passthrough) = 3 columns.
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_column_transformer_not_fitted() {
        let scaler = StandardScaler::new();
        let ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Passthrough,
        );
        let df = make_test_df();
        let err = ct.transform(df).unwrap_err();
        assert!(
            matches!(&err, Error::NotFitted(_)),
            "expected NotFitted, got {err:?}",
        );
        assert_eq!(
            err.to_string(),
            "not fitted: ColumnTransformer has not been fitted. \
             Call .fit(dataframe) before .transform()."
        );
    }

    #[test]
    fn test_column_transformer_failed_refit_resets_fitted() {
        let scaler = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Passthrough,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        assert!(ct.transform(df.clone()).is_ok());

        assert!(ct.fit(DataFrame::empty()).is_err());
        let err = ct.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted after failed refit, got {err:?}",
        );
    }

    #[test]
    fn test_column_transformer_fit_rejects_zero_row_frame() {
        // A 0-row frame reaches the transformers today and comes back as an
        // opaque `Computation` wrapper from the first step instead of a named
        // empty-input error, unlike `Pipeline::fit`.
        let mut ct = ColumnTransformer::new(
            vec![(
                "scale_a".into(),
                Box::new(StandardScaler::new()),
                vec!["a".into()],
            )],
            Remainder::Passthrough,
        );
        let err = ct.fit(make_test_df().head(Some(0))).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("0 rows"), "{err}");
    }

    #[test]
    fn test_column_transformer_transform_rejects_zero_row_frame() {
        let mut ct = ColumnTransformer::new(
            vec![(
                "scale_a".into(),
                Box::new(StandardScaler::new()),
                vec!["a".into()],
            )],
            Remainder::Passthrough,
        );
        let df = make_test_df();
        ct.fit(df.clone()).unwrap();

        let err = ct.transform(df.head(Some(0))).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        assert!(err.to_string().contains("0 rows"), "{err}");
    }

    #[test]
    fn test_column_transformer_empty_config_errors_at_fit() {
        // Zero transformers with `Remainder::Drop` can never produce output;
        // today the failure surfaces only at transform time as "produced no
        // output columns".
        let mut ct = ColumnTransformer::new(vec![], Remainder::Drop);
        let err = ct.fit(make_test_df()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        let msg = err.to_string();
        assert!(msg.contains("no transformers"), "{msg}");
    }

    #[test]
    fn test_column_transformer_output_name_collision_errors_named() {
        // Both hashers emit a column named `hashed_0` from different input
        // columns, so the fit-time duplicate-column check cannot see it; the
        // collision only appears once the transformers have run.
        let a = Column::from(Series::new("a".into(), &["x", "y", "x"]));
        let b = Column::from(Series::new("b".into(), &["p", "q", "p"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut ct = ColumnTransformer::new(
            vec![
                (
                    "hash_a".into(),
                    Box::new(FeatureHasher::new(&["a"], 1)),
                    vec!["a".into()],
                ),
                (
                    "hash_b".into(),
                    Box::new(FeatureHasher::new(&["b"], 1)),
                    vec!["b".into()],
                ),
            ],
            Remainder::Drop,
        );
        ct.fit(df.clone()).unwrap();

        let err = ct.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        let msg = err.to_string();
        assert!(msg.contains("hashed_0"), "{msg}");
        assert!(msg.contains("hash_a") && msg.contains("hash_b"), "{msg}");
    }

    #[test]
    fn test_column_transformer_generated_name_collides_with_remainder() {
        // `MissingIndicator` appends `a_missing` while the input already has a
        // passthrough column of that name, so the outgoing names clash even
        // though no input column is claimed twice.
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let a_missing = Column::from(Series::new("a_missing".into(), &[0.0f64, 1.0, 0.0]));
        let df = DataFrame::new(3, vec![a, a_missing]).unwrap();

        let mut ct = ColumnTransformer::new(
            vec![(
                "ind".into(),
                Box::new(MissingIndicator::new(&["a"])),
                vec!["a".into()],
            )],
            Remainder::Passthrough,
        );
        ct.fit(df.clone()).unwrap();

        let err = ct.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
        let msg = err.to_string();
        assert!(msg.contains("a_missing"), "{msg}");
        assert!(msg.contains("remainder"), "{msg}");
    }

    #[test]
    fn test_column_transformer_empty_config_with_passthrough_passes_frame_through() {
        // The counterpart to the zero-transformer rejection: with
        // `Remainder::Passthrough` nothing is claimed, so the whole frame is the
        // remainder and the configuration is valid.
        let mut ct = ColumnTransformer::new(vec![], Remainder::Passthrough);
        let df = make_test_df();
        ct.fit(df.clone()).unwrap();

        let out = ct.transform(df.clone()).unwrap();
        assert_eq!(out.width(), df.width());
        assert_eq!(out.height(), df.height());
    }

    #[test]
    fn test_column_transformer_transform_rejects_zero_column_frame() {
        // The shared row policy does not cover this: a (3, 0) frame has rows,
        // so the transformer's own column selection is what reports it.
        let mut ct = ColumnTransformer::new(
            vec![(
                "scale_a".into(),
                Box::new(StandardScaler::new()),
                vec!["a".into()],
            )],
            Remainder::Passthrough,
        );
        let df = make_test_df();
        ct.fit(df.clone()).unwrap();

        let no_cols = DataFrame::new(3, vec![]).unwrap();
        let err = ct.transform(no_cols).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}",
        );
    }

    #[test]
    fn test_column_transformer_overlapping_columns_errors_at_fit() {
        let scaler_1 = StandardScaler::new();
        let scaler_2 = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![
                ("s1".into(), Box::new(scaler_1), vec!["a".into()]),
                (
                    "s2".into(),
                    Box::new(scaler_2),
                    vec!["a".into(), "b".into()],
                ),
            ],
            Remainder::Drop,
        );
        let df = make_test_df();

        let err = ct.fit(df).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains('a'));
        assert!(msg.contains("s2"));
    }
}
