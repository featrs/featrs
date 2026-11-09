//! Duplicate-column removal.
//!
//! [`DuplicateColumnRemover`] detects columns that are perfect duplicates of an
//! earlier column (same values at every position) and keeps only the first
//! occurrence, in frame order. Duplicates commonly arise from joins, feature
//! engineering, or data export quirks.
//!
//! Comparison requires identical dtypes: a `Int32` column and a `Float64`
//! column are never considered duplicates, even when their values coincide.
//! `NaN` follows polars' total-order semantics: `NaN` is equal to `NaN`.
//!
//! Complexity: the naive pairwise comparison is O(rows × columns²), which is
//! fine for typical column counts; hash-based grouping is not used.

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::subset_names;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Remove columns that duplicate an earlier column's content.
///
/// Two columns are duplicates when they share the same dtype and every value
/// (including null positions) is identical. Only the leftmost column of each
/// duplicate group is kept; output order matches the input frame order.
///
/// With `consider_nulls = true`, a null in either column is treated as equal
/// to *any* value, so `[1, null]` and `[1, 5]` are duplicates. With the
/// default `consider_nulls = false`, null positions must match exactly, so
/// `[1, null]` and `[1, 5]` are not duplicates.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::duplicate_column_remover::DuplicateColumnRemover;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
/// let a_dup = Column::from(Series::new("a_dup".into(), &[1.0_f64, 2.0, 3.0]));
/// let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, a_dup, b])?;
///
/// let mut remover = DuplicateColumnRemover::new();
/// remover.fit(df.clone())?;
/// let cleaned = remover.transform(df)?;
/// assert_eq!(cleaned.width(), 2);
/// assert_eq!(cleaned.get_column_names()[0].as_str(), "a");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DuplicateColumnRemover {
    fitted: bool,
    /// Names of the columns that survived `fit`, in frame order.
    selected_columns: Option<Vec<String>>,
    /// Whether nulls are ignored when comparing columns.
    ///
    /// `false` (default): columns must agree at every position, including
    /// nulls. `true`: a null in either column is treated as equal to any
    /// value, so null positions never make columns differ.
    consider_nulls: bool,
}

impl DuplicateColumnRemover {
    /// Create a new `DuplicateColumnRemover` with strict null handling
    /// (`consider_nulls = false`).
    pub fn new() -> Self {
        Self {
            fitted: false,
            selected_columns: None,
            consider_nulls: false,
        }
    }

    /// Set whether nulls are ignored when comparing columns.
    ///
    /// When `false` (default), two columns are duplicates only if they agree
    /// at every position, including nulls (a null position must match a null
    /// position). When `true`, a null in either column is treated as equal to
    /// any value, so `[1, null]` and `[1, 5]` are considered duplicates.
    ///
    /// In both modes the columns must still share the same dtype; the
    /// null-permissive rule never makes columns of different dtypes
    /// duplicates.
    pub fn consider_nulls(mut self, b: bool) -> Self {
        self.consider_nulls = b;
        self
    }
}

impl Default for DuplicateColumnRemover {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether `b` is a duplicate of `a`.
///
/// Columns of different dtypes are never duplicates. When `consider_nulls` is
/// `true`, a null in either column is treated as equal to any value; otherwise
/// null positions must match exactly. Columns that polars cannot compare
/// (e.g. categoricals with distinct category objects) are treated as not
/// duplicates rather than failing the fit.
fn columns_are_duplicates(a: &Column, b: &Column, consider_nulls: bool) -> Result<bool> {
    if a.dtype() != b.dtype() {
        return Ok(false);
    }
    if consider_nulls {
        let lhs = a.as_materialized_series();
        let rhs = b.as_materialized_series();
        // Element-wise null-aware equality: null == null is true, null vs a
        // value is false (or null). OR with either-side null to make null
        // equal to any value. A comparison error (e.g. categoricals with
        // distinct category objects) means "not duplicates" — never remove a
        // column on a failed comparison (an all-null mask would vacuously
        // count as "all equal").
        let eq = match lhs.equal_missing(rhs) {
            Ok(eq) => eq,
            Err(_) => return Ok(false),
        };
        let either_null = &lhs.is_null() | &rhs.is_null();
        let permissive_eq = &eq | &either_null;
        Ok(permissive_eq.all())
    } else {
        Ok(a.equals_missing(b))
    }
}

impl Fit<DataFrame> for DuplicateColumnRemover {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset fitted state up front: if a later fit fails, transform must
        // not silently apply a stale selection from a previous fit.
        self.fitted = false;
        self.selected_columns = None;

        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "DuplicateColumnRemover.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "DuplicateColumnRemover.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }

        let cols = x.columns();
        // Keep the leftmost column of each duplicate group. Permissive null
        // equality is not transitive, so every column is compared against ALL
        // earlier input columns (not just survivors): a column that duplicates
        // any earlier column is removed, keeping only the first. The first
        // column always survives, so `selected` can never be empty.
        let mut survivors: Vec<usize> = Vec::new();
        'columns: for i in 0..cols.len() {
            for j in 0..i {
                if columns_are_duplicates(&cols[j], &cols[i], self.consider_nulls)? {
                    continue 'columns;
                }
            }
            survivors.push(i);
        }

        let selected = survivors
            .iter()
            .map(|&i| cols[i].name().to_string())
            .collect();
        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for DuplicateColumnRemover {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "DuplicateColumnRemover has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "DuplicateColumnRemover has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

impl FeatureNamesOut for DuplicateColumnRemover {
    /// The columns that survived the last fit, in frame order.
    fn feature_names_out(&self, _input_columns: &[String]) -> Result<Vec<String>> {
        subset_names(
            self.fitted,
            "DuplicateColumnRemover",
            self.selected_columns.as_deref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let a_dup = Column::from(Series::new("a_dup".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        DataFrame::new(3, vec![a, a_dup, b]).unwrap()
    }

    #[test]
    fn test_removes_duplicate_float64_columns() {
        let mut remover = DuplicateColumnRemover::new();
        let df = make_df();

        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.get_column_names()[0].as_str(), "a");
        assert_eq!(result.get_column_names()[1].as_str(), "b");
    }

    #[test]
    fn test_removes_duplicate_string_columns() {
        let s = Column::from(Series::new("s".into(), &["x", "y", "z"]));
        let s_dup = Column::from(Series::new("s_dup".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![s, s_dup]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "s");
    }

    #[test]
    fn test_columns_differing_at_one_row_are_kept() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[1.0f64, 9.0, 3.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_keeps_first_occurrence_of_duplicate_group() {
        // [a, b, a, c, a] -> keeps a, b, c (first occurrences).
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        let a2 = Column::from(Series::new("a2".into(), &[1.0f64, 2.0, 3.0]));
        let c = Column::from(Series::new("c".into(), &[7.0f64, 8.0, 9.0]));
        let a3 = Column::from(Series::new("a3".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a, b, a2, c, a3]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        let names: Vec<&str> = result
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_all_unique_columns_pass_through() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        let c = Column::from(Series::new("c".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a, b, c]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_single_column_survives() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_null_only_columns_are_duplicates() {
        let n1 = Column::from(Series::new("n1".into(), &[None::<f64>, None, None]));
        let n2 = Column::from(Series::new("n2".into(), &[None::<f64>, None, None]));
        let b = Column::from(Series::new("b".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![n1, n2, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.get_column_names()[0].as_str(), "n1");
        assert_eq!(result.get_column_names()[1].as_str(), "b");
    }

    #[test]
    fn test_strict_mode_requires_matching_null_positions() {
        // Strict (default): [1, null] vs [1, 5] are NOT duplicates because the
        // null position does not match.
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None]));
        let b = Column::from(Series::new("b".into(), &[Some(1.0f64), Some(5.0)]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_strict_mode_null_positions_identical_is_duplicate() {
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None]));
        let b = Column::from(Series::new("b".into(), &[Some(1.0f64), None]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_permissive_mode_null_equals_any_value() {
        // consider_nulls(true): [1, null] vs [1, 5] ARE duplicates.
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None]));
        let b = Column::from(Series::new("b".into(), &[Some(1.0f64), Some(5.0)]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new().consider_nulls(true);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_permissive_mode_null_only_column_duplicates_value_column() {
        let n = Column::from(Series::new("n".into(), &[None::<f64>, None, None]));
        let v = Column::from(Series::new("v".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![n, v]).unwrap();

        let mut remover = DuplicateColumnRemover::new().consider_nulls(true);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_permissive_mode_still_detects_value_mismatches() {
        // [1, 2] vs [1, 5]: no nulls involved, values differ -> not duplicates.
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0]));
        let b = Column::from(Series::new("b".into(), &[1.0f64, 5.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new().consider_nulls(true);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_strict_mode_shifted_null_positions_not_duplicates() {
        // [1, null] vs [null, 1]: null positions differ -> both survive.
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None]));
        let b = Column::from(Series::new("b".into(), &[None, Some(1.0f64)]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_permissive_mode_nan_equals_null() {
        // NaN is a value, so permissive mode treats [NaN] and [null] as
        // duplicates; strict mode does not.
        let nan_col = Column::from(Series::new("nan".into(), &[f64::NAN]));
        let null_col = Column::from(Series::new("null".into(), &[None::<f64>]));
        let df = DataFrame::new(1, vec![nan_col.clone(), null_col.clone()]).unwrap();

        let mut permissive = DuplicateColumnRemover::new().consider_nulls(true);
        permissive.fit(df.clone()).unwrap();
        let result = permissive.transform(df.clone()).unwrap();
        assert_eq!(result.width(), 1);

        let mut strict = DuplicateColumnRemover::new();
        strict.fit(df.clone()).unwrap();
        let result = strict.transform(df).unwrap();
        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_permissive_mode_non_transitive_null_chain_keeps_only_first() {
        // Permissive null equality is not transitive: [null,1] ~ [2,null] and
        // [2,null] ~ [2,3], so the chain collapses to the first column only.
        let a = Column::from(Series::new("a".into(), &[None, Some(1.0f64)]));
        let b = Column::from(Series::new("b".into(), &[Some(2.0f64), None]));
        let c = Column::from(Series::new("c".into(), &[Some(2.0f64), Some(3.0)]));
        let df = DataFrame::new(2, vec![a, b, c]).unwrap();

        let mut remover = DuplicateColumnRemover::new().consider_nulls(true);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn test_comparison_error_is_never_a_duplicate() {
        // A failed polars comparison (here: different lengths, no broadcast)
        // must be treated as "not duplicates" — never remove a column, and
        // never fail the fit. Regression: an all-null fallback mask would
        // vacuously count as "all equal" and drop a column.
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0]));

        assert!(!columns_are_duplicates(&a, &b, true).unwrap());
        assert!(!columns_are_duplicates(&a, &b, false).unwrap());
    }

    #[test]
    fn test_different_dtypes_not_duplicates() {
        // Same logical content, different dtype: not duplicates.
        let i = Column::from(Series::new("i".into(), &[1i32, 2, 3]));
        let f = Column::from(Series::new("f".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![i, f]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_nan_equal_to_nan() {
        // Polars total-order semantics: NaN == NaN.
        let a = Column::from(Series::new("a".into(), &[f64::NAN, 1.0]));
        let b = Column::from(Series::new("b".into(), &[f64::NAN, 1.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let remover = DuplicateColumnRemover::new();
        let df = make_df();
        assert!(matches!(remover.transform(df), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_empty_dataframe_errors() {
        let df = DataFrame::empty();
        let mut remover = DuplicateColumnRemover::new();
        assert!(matches!(remover.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_transform_missing_column_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let a_dup = Column::from(Series::new("a_dup".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a, a_dup]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();

        // Transform input missing the selected column -> Computation error.
        let only_dup = df.select(["a_dup"]).unwrap();
        assert!(matches!(
            remover.transform(only_dup),
            Err(Error::Computation(_))
        ));
    }

    #[test]
    fn test_refit_resets_state() {
        // After a failed re-fit (empty input), transform must still error
        // with NotFitted rather than applying the first fit's selection.
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let a_dup = Column::from(Series::new("a_dup".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a, a_dup]).unwrap();

        let mut remover = DuplicateColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        assert!(remover.fit(DataFrame::empty()).is_err());
        assert!(matches!(remover.transform(df), Err(Error::NotFitted(_))));
    }
}
