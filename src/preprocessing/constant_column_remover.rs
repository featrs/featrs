//! Constant-column removal.
//!
//! [`ConstantColumnRemover`] removes columns that contain only a single
//! unique value (no variance), as they carry no information for modeling.
//! Unlike [`VarianceThreshold`](crate::feature_selection::VarianceThreshold),
//! it operates on **all** dtypes, not just `Float64`.

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::subset_names;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Remove columns that contain only a single unique value.
///
/// With `ignore_nulls = true` (default), a column is considered constant when
/// it has at most one distinct non-null value; null-only columns are removed.
/// With `ignore_nulls = false`, any column containing a null is preserved.
///
/// `NaN` follows polars' grouping semantics: `[NaN, NaN]` counts as a single
/// distinct value (constant), while `[NaN, 1.0]` counts as two.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::constant_column_remover::ConstantColumnRemover;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let const_col = Column::from(Series::new("const".into(), &[1.0_f64, 1.0, 1.0]));
/// let varying = Column::from(Series::new("varying".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![const_col, varying])?;
///
/// let mut remover = ConstantColumnRemover::new();
/// remover.fit(df.clone())?;
/// let cleaned = remover.transform(df)?;
/// assert_eq!(cleaned.width(), 1);
/// assert_eq!(cleaned.get_column_names()[0].as_str(), "varying");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ConstantColumnRemover {
    fitted: bool,
    /// Names of the columns that survived `fit`, in frame order.
    selected_columns: Option<Vec<String>>,
    /// Whether null-containing columns are checked for constancy.
    ///
    /// `true` (default): null-only columns are constant and are removed, and
    /// nulls are ignored when counting distinct values.
    /// `false`: any column containing a null is preserved.
    ignore_nulls: bool,
}

impl ConstantColumnRemover {
    /// Create a new `ConstantColumnRemover` with nulls ignored
    /// (`ignore_nulls = true`).
    pub fn new() -> Self {
        Self {
            fitted: false,
            selected_columns: None,
            ignore_nulls: true,
        }
    }

    /// Set whether null-containing columns are checked for constancy.
    ///
    /// When `true` (default), only distinct **non-null** values are counted,
    /// so null-only columns are treated as constant and removed. When
    /// `false`, any column that contains at least one null is preserved,
    /// regardless of how many distinct values it holds.
    pub fn ignore_nulls(mut self, b: bool) -> Self {
        self.ignore_nulls = b;
        self
    }
}

impl Default for ConstantColumnRemover {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for ConstantColumnRemover {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset fitted state up front: if a later fit fails, transform must
        // not silently apply a stale selection from a previous fit.
        self.fitted = false;
        self.selected_columns = None;

        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "ConstantColumnRemover.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "ConstantColumnRemover.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }

        let mut selected = Vec::new();
        for col in x.columns() {
            let name = col.name().to_string();
            let n_unique = col.n_unique().map_err(|e| {
                Error::Computation(format!(
                    "ConstantColumnRemover.fit: could not count unique values \
                     for column '{name}'. {e}"
                ))
            })?;
            let null_count = col.null_count();

            let constant = if self.ignore_nulls {
                // `n_unique` counts null as one distinct value; subtract it
                // when the column has any nulls to get distinct non-null count.
                let distinct_non_null = if null_count > 0 {
                    n_unique.saturating_sub(1)
                } else {
                    n_unique
                };
                distinct_non_null <= 1
            } else {
                null_count == 0 && n_unique <= 1
            };

            if !constant {
                selected.push(name);
            }
        }

        if selected.is_empty() {
            let hint = if self.ignore_nulls {
                " Check your data or set ignore_nulls(false) to preserve \
                 columns containing nulls."
            } else {
                " Check your data."
            };
            return Err(Error::InvalidInput(format!(
                "ConstantColumnRemover.fit: all {} column(s) are constant \
                 (single unique value). No columns survive.{hint}",
                x.width()
            )));
        }

        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for ConstantColumnRemover {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "ConstantColumnRemover has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "ConstantColumnRemover has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

impl FeatureNamesOut for ConstantColumnRemover {
    /// The columns that survived the last fit, in frame order.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        subset_names(
            self.fitted,
            "ConstantColumnRemover",
            self.selected_columns.as_deref(),
            input_columns,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df() -> DataFrame {
        let const_col = Column::from(Series::new("const".into(), &[1.0f64, 1.0, 1.0]));
        let varying = Column::from(Series::new("varying".into(), &[1.0f64, 2.0, 3.0]));
        DataFrame::new(3, vec![const_col, varying]).unwrap()
    }

    #[test]
    fn test_removes_constant_column() {
        let mut remover = ConstantColumnRemover::new();
        let df = make_df();

        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "varying");
    }

    #[test]
    fn test_no_constant_columns_pass_through() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_all_constant_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 1.0, 1.0]));
        let b = Column::from(Series::new("b".into(), &["x", "x", "x"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        assert!(matches!(remover.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_null_only_column_removed_by_default() {
        let nulls = Column::from(Series::new("nulls".into(), &[None::<f64>, None, None]));
        let varying = Column::from(Series::new("varying".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![nulls, varying]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "varying");
    }

    #[test]
    fn test_null_only_column_preserved_with_ignore_nulls_false() {
        let nulls = Column::from(Series::new("nulls".into(), &[None::<f64>, None, None]));
        let varying = Column::from(Series::new("varying".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![nulls, varying]).unwrap();

        let mut remover = ConstantColumnRemover::new().ignore_nulls(false);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_single_distinct_non_null_with_nulls_removed_by_default() {
        // [1, null, 1]: one distinct non-null value -> constant by default.
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None, Some(1.0)]));
        let b = Column::from(Series::new("b".into(), &[Some(2.0f64), None, Some(3.0)]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "b");
    }

    #[test]
    fn test_null_preserved_with_ignore_nulls_false() {
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None, Some(1.0)]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut remover = ConstantColumnRemover::new().ignore_nulls(false);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_null_free_constant_removed_with_ignore_nulls_false() {
        // ignore_nulls(false) only preserves null-bearing columns; a
        // constant column without nulls is still removed.
        let const_col = Column::from(Series::new("const".into(), &[7.0f64, 7.0, 7.0]));
        let varying = Column::from(Series::new("varying".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![const_col, varying]).unwrap();

        let mut remover = ConstantColumnRemover::new().ignore_nulls(false);
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "varying");
    }

    #[test]
    fn test_transform_missing_column_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 1.0, 1.0]));
        let b = Column::from(Series::new("b".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();

        // Transform input missing the selected column -> Computation error.
        let only_a = df.select(["a"]).unwrap();
        assert!(matches!(
            remover.transform(only_a),
            Err(Error::Computation(_))
        ));
    }

    #[test]
    fn test_string_constant_removed() {
        let s = Column::from(Series::new("s".into(), &["x", "x", "x"]));
        let varying = Column::from(Series::new("varying".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![s, varying]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "varying");
    }

    #[test]
    fn test_bool_constant_removed() {
        let b = Column::from(Series::new("b".into(), &[true, true, true]));
        let varying = Column::from(Series::new("varying".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![b, varying]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_int_non_constant_kept() {
        let a = Column::from(Series::new("a".into(), &[1i32, 1, 2]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        remover.fit(df.clone()).unwrap();
        let result = remover.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }

    #[test]
    fn test_single_row_all_constant_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0f64]));
        let b = Column::from(Series::new("b".into(), &["x"]));
        let df = DataFrame::new(1, vec![a, b]).unwrap();

        let mut remover = ConstantColumnRemover::new();
        assert!(matches!(remover.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let remover = ConstantColumnRemover::new();
        let df = make_df();
        assert!(matches!(remover.transform(df), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_empty_dataframe_errors() {
        let df = DataFrame::empty();
        let mut remover = ConstantColumnRemover::new();
        assert!(matches!(remover.fit(df), Err(Error::InvalidInput(_))));
    }
}
