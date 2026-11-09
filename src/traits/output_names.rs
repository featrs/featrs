//! Output feature names for fitted transformers.
//!
//! Every transformer in this crate preserves or generates column names in a
//! deterministic way, but that plan was only visible after calling
//! [`transform`](crate::traits::Transform::transform). [`FeatureNamesOut`]
//! exposes the same plan as metadata, so callers can validate a downstream
//! input, log a schema, or plan a query before running the data through the
//! transformer.
//!
//! The trait is implemented by every transformer. Transformers that preserve
//! their input names return the names they are given, selectors return their
//! learned subset, and generators report their output names. Generators may
//! append columns or replace input columns in place. The result is always the
//! exact column list, in order, that `transform` produces for a frame with
//! those input columns.
//!
//! # Example
//!
//! ```rust
//! use featrs::preprocessing::scaler::StandardScaler;
//! use featrs::traits::{FeatureNamesOut, Fit};
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let df = DataFrame::new(
//!     3,
//!     vec![Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]))],
//! )?;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df)?;
//!
//! let names = scaler.feature_names_out(&["x".to_string()])?;
//! assert_eq!(names, vec!["x".to_string()]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::traits::{Error, Result};

/// A fitted transformer that can describe the columns its `transform` emits.
///
/// Implementations must be called only after a successful `fit`; calling
/// `feature_names_out` before `fit` returns [`Error::NotFitted`]. The result
/// depends on the fitted state (learned categories, selected columns, bin
/// edges, …) and on `input_columns`, which are the columns of the frame that
/// would be passed to `transform`, in frame order.
///
/// The returned names are exactly the output column names of `transform` for
/// that input, in output order. When `transform` would reject the input — a
/// missing fitted column, or a generated name that collides with an input
/// column — `feature_names_out` returns the same [`Error::InvalidInput`]
/// instead of a schema. Extra input columns are allowed only when `transform`
/// ignores them, matching `transform` exactly.
///
/// The method has a default body returning the input names unchanged, so
/// existing third-party [`DataFrameTransformer`](crate::pipeline::DataFrameTransformer)
/// implementations keep compiling when this trait grows new requirements.
/// Transformers whose `transform` drops, renames, or generates columns must
/// override it.
pub trait FeatureNamesOut {
    /// Returns the output column names that `transform` will produce for a
    /// frame whose columns are `input_columns`, in output order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFitted`] if the transformer has not been fitted.
    /// Returns [`Error::InvalidInput`] when the transformer cannot describe an
    /// output for this input (for example a single-column encoder given an
    /// empty input).
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        Ok(input_columns.to_vec())
    }
}

/// Return [`Error::NotFitted`] unless `fitted` is `true`.
pub(crate) fn ensure_fitted(fitted: bool, who: &str) -> Result<()> {
    if fitted {
        Ok(())
    } else {
        Err(Error::NotFitted(format!(
            "{who} has not been fitted. Call .fit(dataframe) before .feature_names_out()."
        )))
    }
}

/// The identity plan: a fitted pass-through transformer emits its input names.
pub(crate) fn identity_names(
    fitted: bool,
    who: &str,
    input_columns: &[String],
) -> Result<Vec<String>> {
    ensure_fitted(fitted, who)?;
    Ok(input_columns.to_vec())
}

/// The learned column subset a select-based transformer emits.
///
/// Every learned column must be present in `input_columns`; extra input
/// columns are allowed and ignored, matching `transform`.
pub(crate) fn subset_names(
    fitted: bool,
    who: &str,
    learned: Option<&[String]>,
    input_columns: &[String],
) -> Result<Vec<String>> {
    ensure_fitted(fitted, who)?;
    let learned = learned.ok_or_else(|| {
        Error::NotFitted(format!(
            "{who} has not been fitted. Call .fit(dataframe) before .feature_names_out()."
        ))
    })?;
    ensure_required_present(input_columns, learned, who)?;
    Ok(learned.to_vec())
}

/// Require every column in `required` to be present in `input_columns`.
///
/// Mirrors the missing-column guard in `transform`: a schema prediction for
/// a frame that `transform` would reject is an [`Error::InvalidInput`],
/// not a schema. Extra input columns are allowed.
pub(crate) fn ensure_required_present(
    input_columns: &[String],
    required: &[String],
    who: &str,
) -> Result<()> {
    for name in required {
        if !input_columns.contains(name) {
            return Err(Error::InvalidInput(format!(
                "{who}.feature_names_out: column '{name}' not found in the input columns."
            )));
        }
    }
    Ok(())
}

/// Reject generated names that collide with an input column or with an
/// earlier generated name.
///
/// Mirrors the collision guard in `transform` for generators that return
/// [`Error::InvalidInput`] instead of overwriting: a schema prediction for
/// a frame that `transform` would reject is an error, not a schema. Call
/// this before [`with_generated_names`]; after it passes, the append cannot
/// collide.
pub(crate) fn ensure_generated_fresh(
    input_columns: &[String],
    generated: &[String],
    who: &str,
) -> Result<()> {
    let mut seen: std::collections::HashSet<&str> =
        input_columns.iter().map(|s| s.as_str()).collect();
    for name in generated {
        if !seen.insert(name.as_str()) {
            return Err(Error::InvalidInput(format!(
                "{who}.feature_names_out: generated column '{name}' collides with an input \
                 column or another generated column."
            )));
        }
    }
    Ok(())
}

/// Append generated names the way `DataFrame::with_column` applies them: a
/// generated name that already exists keeps its position and is not repeated.
///
/// This mirrors transformers that append with `with_column`; for those a
/// colliding name replaces the input column rather than creating a duplicate.
pub(crate) fn with_generated_names(
    input_columns: &[String],
    generated: impl IntoIterator<Item = String>,
) -> Vec<String> {
    let mut names = input_columns.to_vec();
    for name in generated {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Reject duplicate names before a composition layer reports a schema.
pub(crate) fn ensure_unique_names(names: &[String], who: &str) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    for name in names {
        if !seen.insert(name.as_str()) {
            return Err(Error::InvalidInput(format!(
                "{who}: output column '{name}' is produced by more than one source. Rename \
                 one of the outputs."
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_names_rejects_unfitted() {
        let err = identity_names(false, "Test", &["a".to_string()]).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn with_generated_names_replaces_existing_names() {
        let input = vec!["a".to_string(), "b".to_string()];
        let names = with_generated_names(&input, ["a_lag_1".to_string(), "b".to_string()]);
        assert_eq!(names, vec!["a", "b", "a_lag_1"]);
    }

    #[test]
    fn subset_names_rejects_missing_selected() {
        let learned = vec!["a".to_string()];
        let err = subset_names(true, "Test", Some(&learned), &["b".to_string()]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        let names = subset_names(
            true,
            "Test",
            Some(&learned),
            &["b".to_string(), "a".to_string()],
        )
        .unwrap();
        assert_eq!(names, vec!["a".to_string()]);
    }

    #[test]
    fn ensure_generated_fresh_rejects_input_and_internal_collisions() {
        let input = vec!["a".to_string()];
        assert!(matches!(
            ensure_generated_fresh(&input, &["a_lag_1".to_string(), "a".to_string()], "Test"),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            ensure_generated_fresh(&input, &["x".to_string(), "x".to_string()], "Test"),
            Err(Error::InvalidInput(_))
        ));
        assert!(
            ensure_generated_fresh(&input, &["x".to_string(), "y".to_string()], "Test").is_ok()
        );
    }

    #[test]
    fn ensure_unique_names_rejects_duplicates() {
        let names = vec!["a".to_string(), "a".to_string()];
        assert!(matches!(
            ensure_unique_names(&names, "Test"),
            Err(Error::InvalidInput(_))
        ));
    }
}
