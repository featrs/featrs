//! Rare-category grouping for string columns.
//!
//! [`RareCategoryGrouper`] relabels infrequently-appearing categories in
//! selected `String` columns to a configurable `"Other"` label, which improves
//! the resilience of downstream encoders (one-hot, target, etc.) to
//! high-cardinality categoricals with rare categories.

use std::collections::{HashMap, HashSet};

use polars::prelude::*;

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_required_present, identity_names};
use crate::traits::{Error, Fit, Result, Transform};

/// The threshold that decides which categories are considered rare.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// Keep every category that appears at least `n` times.
    ///
    /// Categories with `count < n` are relabeled. `MinCount(0)` and
    /// `MinCount(1)` keep every category observed at fit time; categories
    /// unseen at transform time are still relabeled.
    MinCount(u32),
    /// Keep every category whose relative frequency is at least `p`.
    ///
    /// Frequencies are computed against the number of non-null values in the
    /// column, so observed frequencies sum to approximately `1.0`. `p` must be
    /// finite and in `[0, 1]`; `MinFrequency(0.0)` keeps every category
    /// observed at fit time (unseen categories are still relabeled) and
    /// `MinFrequency(1.0)` keeps only categories that own the whole non-null
    /// column.
    MinFrequency(f64),
}

/// Group infrequent categories in `String` columns into a single `"Other"` label.
///
/// During [`fit`](Fit::fit), each selected column is scanned and the
/// categories that fall below [`Threshold`] are remembered. During
/// [`transform`](Transform::transform), every non-null value that is not a
/// kept category — whether it was rare at fit time or entirely unseen at
/// transform time — is relabeled to `other_label`. This makes the transformer
/// robust to category drift between training and inference.
///
/// # Behaviour
///
/// | Input value | Output |
/// |---|---|
/// | Category kept by the threshold | unchanged |
/// | Rare category (below threshold at fit) | `other_label` |
/// | Category unseen at transform time | `other_label` |
/// | Null | preserved as null (unless `treat_null_as_rare` is set) |
///
/// # Chosen semantics
///
/// - **Threshold boundary**: `count >= n` / `freq >= p` keeps, strict
///   inequality relabels — matching the issue's tie-breaking contract.
/// - **Frequency denominator**: non-null values only, so observed frequencies
///   sum to approximately `1.0`.
/// - **`other_label` conflict**: a category that is literally named
///   `other_label` is treated like any other category; if it is frequent it
///   stays, and rare categories relabeled to the same string merge with it.
/// - **All categories rare**: every non-null value becomes `other_label`
///   (a single-category column); this is valid, not an error.
/// - **Empty string** is a distinct category from null and is counted like any
///   other value (unless `other_label` is set to the empty string, in which
///   case relabeled categories merge with empty-string values).
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::rare_category_grouper::{RareCategoryGrouper, Threshold};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let city = Column::from(Series::new("city".into(), &["Delhi", "Mumbai", "Delhi", "Pune"]));
/// let df = DataFrame::new(4, vec![city])?;
///
/// let mut grouper = RareCategoryGrouper::new()
///     .columns(&["city"])
///     .threshold(Threshold::MinCount(2));
/// grouper.fit(df.clone())?;
/// let grouped = grouper.transform(df)?;
/// assert_eq!(grouped.column("city")?.str()?.get(3), Some("Other"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RareCategoryGrouper {
    fitted: bool,
    columns: Vec<String>,
    threshold: Threshold,
    other_label: String,
    treat_null_as_rare: bool,
    /// Per fitted column, the categories to keep. Every other non-null value
    /// (rare at fit time or unseen at transform time) is relabeled to
    /// `other_label`. Index-aligned with `columns`.
    kept_sets: Option<Vec<HashSet<String>>>,
}

impl RareCategoryGrouper {
    /// Create a new `RareCategoryGrouper`.
    ///
    /// Defaults: threshold [`Threshold::MinCount`] with value 5, `other_label`
    /// `"Other"`, nulls preserved. Select at least one column with
    /// [`columns`](Self::columns) before fitting.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: Vec::new(),
            threshold: Threshold::MinCount(5),
            other_label: "Other".to_string(),
            treat_null_as_rare: false,
            kept_sets: None,
        }
    }

    /// Select the `String` columns to group.
    ///
    /// Duplicate names are ignored after their first occurrence so each
    /// selected column is grouped exactly once. Invalidates any previously
    /// fitted state.
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.invalidate_fit();
        self.columns.clear();
        for column in cols {
            if !self.columns.iter().any(|selected| selected == column) {
                self.columns.push((*column).to_owned());
            }
        }
        self
    }

    /// Set the rarity threshold (default: [`Threshold::MinCount`] with value 5).
    /// Invalidates any previously fitted state.
    pub fn threshold(mut self, threshold: Threshold) -> Self {
        self.invalidate_fit();
        self.threshold = threshold;
        self
    }

    /// Set the label that rare and unseen categories are relabeled to
    /// (default: `"Other"`). Invalidates any previously fitted state.
    pub fn other_label(mut self, label: &str) -> Self {
        self.invalidate_fit();
        self.other_label = label.to_owned();
        self
    }

    /// Whether nulls are relabeled to `other_label` at transform time.
    ///
    /// When `false` (default), nulls are preserved as null. When `true`, nulls
    /// are treated like any rare value and become `other_label`.
    /// Invalidates any previously fitted state.
    pub fn treat_null_as_rare(mut self, treat: bool) -> Self {
        self.invalidate_fit();
        self.treat_null_as_rare = treat;
        self
    }

    fn invalidate_fit(&mut self) {
        self.fitted = false;
        self.kept_sets = None;
    }
}

impl Default for RareCategoryGrouper {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for RareCategoryGrouper {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave stale state behind.
        self.invalidate_fit();

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "RareCategoryGrouper.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "RareCategoryGrouper.fit has no selected columns. \
                 Select at least one String column with .columns(&[...])."
                    .into(),
            ));
        }
        if let Threshold::MinFrequency(p) = self.threshold
            && (!p.is_finite() || !(0.0..=1.0).contains(&p))
        {
            return Err(Error::InvalidInput(format!(
                "RareCategoryGrouper.fit: MinFrequency threshold {p} is out of range. \
                 Use a finite value in [0, 1]."
            )));
        }

        let mut kept_sets = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            let column = x.column(name.as_str()).map_err(|error| {
                Error::InvalidInput(format!(
                    "RareCategoryGrouper.fit: column '{name}' was not found. {error}"
                ))
            })?;
            let ca = column.as_materialized_series().str().map_err(|error| {
                Error::InvalidInput(format!(
                    "RareCategoryGrouper.fit: column '{name}' has dtype {}; expected String. {error}",
                    column.dtype()
                ))
            })?;

            let mut counts: HashMap<&str, u64> = HashMap::new();
            let mut total: u64 = 0;
            for value in ca.iter().flatten() {
                *counts.entry(value).or_insert(0u64) += 1;
                total += 1;
            }

            let mut kept: HashSet<String> = HashSet::new();
            match self.threshold {
                Threshold::MinCount(min_count) => {
                    let min_count = u64::from(min_count);
                    for (category, count) in &counts {
                        if *count >= min_count {
                            kept.insert((*category).to_string());
                        }
                    }
                }
                Threshold::MinFrequency(min_frequency) => {
                    let total_f64 = total as f64;
                    for (category, count) in &counts {
                        if *count as f64 / total_f64 >= min_frequency {
                            kept.insert((*category).to_string());
                        }
                    }
                }
            }
            kept_sets.push(kept);
        }

        self.kept_sets = Some(kept_sets);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for RareCategoryGrouper {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        let kept_sets = self.kept_sets.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "RareCategoryGrouper has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        let mut output = x;
        for (name, kept) in self.columns.iter().zip(kept_sets) {
            let column = output.column(name.as_str()).map_err(|error| {
                Error::InvalidInput(format!(
                    "RareCategoryGrouper.transform: fitted column '{name}' was not found. {error}"
                ))
            })?;
            let strings = column.as_materialized_series().str().map_err(|error| {
                Error::InvalidInput(format!(
                    "RareCategoryGrouper.transform: column '{name}' has dtype {}; expected String. {error}",
                    column.dtype()
                ))
            })?;

            let other = self.other_label.as_str();
            let mut grouped: StringChunked = strings
                .iter()
                .map(|value| match value {
                    Some(value) => {
                        if kept.contains(value) {
                            Some(value.to_owned())
                        } else {
                            Some(other.to_owned())
                        }
                    }
                    None => {
                        if self.treat_null_as_rare {
                            Some(other.to_owned())
                        } else {
                            None
                        }
                    }
                })
                .collect();
            grouped.rename(name.as_str().into());
            output
                .with_column(Column::from(grouped.into_series()))
                .map_err(|error| {
                    Error::Computation(format!(
                        "RareCategoryGrouper.transform: could not replace column '{name}'. {error}"
                    ))
                })?;
        }

        Ok(output)
    }
}

impl FeatureNamesOut for RareCategoryGrouper {
    /// Output names are the input names: `transform` rewrites columns in place.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        let names = identity_names(self.fitted, "RareCategoryGrouper", input_columns)?;
        ensure_required_present(input_columns, &self.columns, "RareCategoryGrouper")?;
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(city: &[Option<&str>]) -> DataFrame {
        let col = Column::from(Series::new("city".into(), city));
        DataFrame::new(city.len(), vec![col]).unwrap()
    }

    fn values(df: &DataFrame) -> Vec<Option<String>> {
        df.column("city")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn test_min_count_groups_rare_categories() {
        // Delhi: 2, Mumbai: 1, Pune: 1 — MinCount(2) keeps only Delhi.
        let df = frame(&[Some("Delhi"), Some("Mumbai"), Some("Delhi"), Some("Pune")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("Delhi"));
        assert_eq!(out[1].as_deref(), Some("Other"));
        assert_eq!(out[2].as_deref(), Some("Delhi"));
        assert_eq!(out[3].as_deref(), Some("Other"));
    }

    #[test]
    fn test_min_frequency_all_categories_rare() {
        // Counts [50, 50], frequency 0.5 each — below 0.6, so both are rare.
        let vals: Vec<Option<&str>> = (0..50)
            .map(|_| Some("a"))
            .chain((0..50).map(|_| Some("b")))
            .collect();
        let df = frame(&vals);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinFrequency(0.6));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out.len(), 100);
        assert!(out.iter().all(|v| v.as_deref() == Some("Other")));
    }

    #[test]
    fn test_min_frequency_keeps_frequent_category() {
        // a: 3 (0.75), b: 1 (0.25) — MinFrequency(0.5) keeps a only.
        let df = frame(&[Some("a"), Some("a"), Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinFrequency(0.5));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("a"));
        assert_eq!(out[3].as_deref(), Some("Other"));
    }

    #[test]
    fn test_custom_other_label() {
        let df = frame(&[Some("Delhi"), Some("Mumbai"), Some("Delhi"), Some("Pune")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2))
            .other_label("ZZZ");
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[1].as_deref(), Some("ZZZ"));
        assert_eq!(out[3].as_deref(), Some("ZZZ"));
    }

    #[test]
    fn test_unseen_category_at_transform_becomes_other() {
        let fit_df = frame(&[Some("Delhi"), Some("Mumbai"), Some("Delhi"), Some("Pune")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(fit_df).unwrap();

        let new_df = frame(&[Some("Delhi"), Some("Tokyo")]);
        let result = grouper.transform(new_df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("Delhi"));
        assert_eq!(out[1].as_deref(), Some("Other"));
    }

    #[test]
    fn test_nulls_preserved_by_default() {
        let df = frame(&[Some("a"), None, Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("a"));
        assert!(out[1].is_none(), "null must stay null by default");
        assert_eq!(out[2].as_deref(), Some("a"));
        assert_eq!(out[3].as_deref(), Some("Other"));
    }

    #[test]
    fn test_treat_null_as_rare() {
        let df = frame(&[Some("a"), None, Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2))
            .treat_null_as_rare(true);
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[1].as_deref(), Some("Other"));
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let df = frame(&[Some("a"), Some("b")]);
        let grouper = RareCategoryGrouper::new().columns(&["city"]);
        let err = grouper.transform(df).unwrap_err();
        assert!(
            err.to_string().contains("not been fitted"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_empty_input_errors() {
        let mut grouper = RareCategoryGrouper::new().columns(&["city"]);
        let df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let err = grouper.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("empty DataFrame"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_all_categories_rare_produces_single_category() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert!(out.iter().all(|v| v.as_deref() == Some("Other")));
    }

    #[test]
    fn test_pass_through_thresholds_keep_fit_categories() {
        // MinCount(0), MinCount(1) and MinFrequency(0.0) keep every category
        // observed at fit time.
        for threshold in [
            Threshold::MinCount(0),
            Threshold::MinCount(1),
            Threshold::MinFrequency(0.0),
        ] {
            let df = frame(&[Some("a"), Some("b"), Some("a")]);
            let mut grouper = RareCategoryGrouper::new()
                .columns(&["city"])
                .threshold(threshold);
            grouper.fit(df.clone()).unwrap();
            let result = grouper.transform(df).unwrap();

            let out = values(&result);
            assert_eq!(out[0].as_deref(), Some("a"));
            assert_eq!(out[1].as_deref(), Some("b"));
            assert_eq!(out[2].as_deref(), Some("a"));
        }
    }

    #[test]
    fn test_pass_through_threshold_unseen_still_relabeled() {
        // Even with a pass-through threshold, categories unseen at transform
        // time are relabeled; the "no-op" promise only covers fit-time data.
        let fit_df = frame(&[Some("a"), Some("a")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(0));
        grouper.fit(fit_df).unwrap();

        let result = grouper.transform(frame(&[Some("a"), Some("zzz")])).unwrap();
        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("a"));
        assert_eq!(out[1].as_deref(), Some("Other"));
    }

    #[test]
    fn test_non_string_column_errors() {
        let col = Column::from(Series::new("num".into(), &[1_i64, 2, 3]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut grouper = RareCategoryGrouper::new().columns(&["num"]);
        let err = grouper.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("expected String"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_missing_column_errors() {
        let df = frame(&[Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new().columns(&["nope"]);
        let err = grouper.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("was not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_empty_columns_list_errors() {
        let df = frame(&[Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new().columns(&[]);
        let err = grouper.fit(df).unwrap_err();
        assert!(
            err.to_string().contains("no selected columns"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_other_label_conflict_merges() {
        // "Other" is frequent (kept), "x" is rare — both end up as "Other".
        let df = frame(&[Some("Other"), Some("Other"), Some("x")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert!(out.iter().all(|v| v.as_deref() == Some("Other")));
    }

    #[test]
    fn test_min_frequency_out_of_range_errors() {
        for p in [1.5, -0.1, f64::NAN] {
            let df = frame(&[Some("a"), Some("b")]);
            let mut grouper = RareCategoryGrouper::new()
                .columns(&["city"])
                .threshold(Threshold::MinFrequency(p));
            let err = grouper.fit(df).unwrap_err();
            assert!(
                err.to_string().contains("out of range"),
                "unexpected error for p={p}: {err}"
            );
        }
    }

    #[test]
    fn test_refit_resets_state() {
        let df = frame(&[Some("a"), Some("a"), Some("b"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(3));
        grouper.fit(df.clone()).unwrap();
        // MinCount(3): both categories have count 2 → everything becomes "Other".
        let first = grouper.transform(df.clone()).unwrap();
        let out = values(&first);
        assert!(out.iter().all(|v| v.as_deref() == Some("Other")));

        // Refit with a no-op threshold: state must be fully reset.
        grouper.threshold = Threshold::MinCount(1);
        grouper.fit(df.clone()).unwrap();
        let second = grouper.transform(df).unwrap();
        let out = values(&second);
        assert_eq!(out[0].as_deref(), Some("a"));
        assert_eq!(out[2].as_deref(), Some("b"));
    }

    #[test]
    fn test_empty_string_is_distinct_category() {
        let df = frame(&[Some(""), Some(""), Some("x")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some(""));
        assert_eq!(out[1].as_deref(), Some(""));
        assert_eq!(out[2].as_deref(), Some("Other"));
    }

    #[test]
    fn test_multiple_columns_grouped_independently() {
        let city = Column::from(Series::new("city".into(), &["Delhi", "Delhi", "Pune"]));
        let region = Column::from(Series::new("region".into(), &["north", "south", "south"]));
        let df = DataFrame::new(3, vec![city, region]).unwrap();

        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city", "region"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let cities: Vec<Option<String>> = result
            .column("city")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(|s| s.to_string()))
            .collect();
        assert_eq!(cities[0].as_deref(), Some("Delhi"));
        assert_eq!(cities[2].as_deref(), Some("Other"));

        let regions: Vec<Option<String>> = result
            .column("region")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(|s| s.to_string()))
            .collect();
        assert_eq!(regions[0].as_deref(), Some("Other"));
        assert_eq!(regions[1].as_deref(), Some("south"));
        assert_eq!(regions[2].as_deref(), Some("south"));
    }

    #[test]
    fn test_default_threshold_is_min_count_five() {
        // Default MinCount(5): "a" appears 5 times (kept), "b" once (rare).
        let vals: Vec<Option<&str>> = (0..5)
            .map(|_| Some("a"))
            .chain(std::iter::once(Some("b")))
            .collect();
        let df = frame(&vals);
        let mut grouper = RareCategoryGrouper::new().columns(&["city"]);
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("a"));
        assert_eq!(out[5].as_deref(), Some("Other"));
    }

    #[test]
    fn test_setters_invalidate_fit() {
        let df = frame(&[Some("a"), Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df).unwrap();
        // Changing configuration after fit must invalidate the learned state.
        let grouper = grouper.threshold(Threshold::MinCount(1));
        let df = frame(&[Some("a"), Some("a"), Some("b")]);
        let err = grouper.transform(df).unwrap_err();
        assert!(
            err.to_string().contains("not been fitted"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_transform_missing_column_errors() {
        let fit_df = frame(&[Some("a"), Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(fit_df).unwrap();

        let num = Column::from(Series::new("other".into(), &[1_i64, 2, 3]));
        let df = DataFrame::new(3, vec![num]).unwrap();
        let err = grouper.transform(df).unwrap_err();
        assert!(
            err.to_string().contains("was not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_transform_non_string_column_errors() {
        let fit_df = frame(&[Some("a"), Some("a"), Some("b")]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(fit_df).unwrap();

        let num = Column::from(Series::new("city".into(), &[1_i64, 2, 3]));
        let df = DataFrame::new(3, vec![num]).unwrap();
        let err = grouper.transform(df).unwrap_err();
        assert!(
            err.to_string().contains("expected String"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_unselected_columns_pass_through() {
        let city = Column::from(Series::new("city".into(), &["Delhi", "Delhi", "Pune"]));
        let num = Column::from(Series::new("num".into(), &[1_i64, 2, 3]));
        let df = DataFrame::new(3, vec![city, num]).unwrap();

        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let num_out: Vec<i64> = result
            .column("num")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(num_out, vec![1, 2, 3]);
        let city_out = values(&result);
        assert_eq!(city_out[2].as_deref(), Some("Other"));
    }

    #[test]
    fn test_duplicate_columns_are_deduped() {
        let city = Column::from(Series::new(
            "city".into(),
            &["Delhi", "Mumbai", "Delhi", "Pune"],
        ));
        let df = DataFrame::new(4, vec![city]).unwrap();

        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city", "city"])
            .threshold(Threshold::MinCount(2));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert_eq!(out[0].as_deref(), Some("Delhi"));
        assert_eq!(out[1].as_deref(), Some("Other"));
    }

    #[test]
    fn test_all_null_column() {
        let df = frame(&[None, None, None]);
        let mut grouper = RareCategoryGrouper::new()
            .columns(&["city"])
            .threshold(Threshold::MinFrequency(0.5));
        grouper.fit(df.clone()).unwrap();
        let result = grouper.transform(df).unwrap();

        let out = values(&result);
        assert!(out.iter().all(|v| v.is_none()));
    }
}
