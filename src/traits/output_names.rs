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
//! learned subset, and generators append the names they will create. The
//! result is always the exact column list, in order, that `transform` produces
//! for a frame with those input columns.
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
/// that input, in output order. Passing a different set of input columns than
/// the transformer was fitted on is allowed only when the transformer's output
/// does not depend on them; the result then describes `transform` for that
/// frame. Selectors and encoders that emit a fixed learned set ignore extra
/// input names, matching `transform`.
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
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>>;
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
pub(crate) fn subset_names(
    fitted: bool,
    who: &str,
    learned: Option<&[String]>,
) -> Result<Vec<String>> {
    ensure_fitted(fitted, who)?;
    learned.map(<[String]>::to_vec).ok_or_else(|| {
        Error::NotFitted(format!(
            "{who} has not been fitted. Call .fit(dataframe) before .feature_names_out()."
        ))
    })
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
    fn ensure_unique_names_rejects_duplicates() {
        let names = vec!["a".to_string(), "a".to_string()];
        assert!(matches!(
            ensure_unique_names(&names, "Test"),
            Err(Error::InvalidInput(_))
        ));
    }
}
