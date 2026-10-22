//! Predicate-driven column routing.
//!
//! [`ConditionalPipeline`] dispatches every input column to the first rule
//! whose [`ColumnPredicate`] matches it, instead of listing the columns of each
//! transformer by hand the way
//! [`ColumnTransformer`](crate::pipeline::ColumnTransformer) requires. Rules are
//! supplied as `(predicate, transformer)` pairs and evaluated in declaration
//! order, so a dtype rule, a name-pattern rule and a user closure can share one
//! pipeline.

use crate::pipeline::{DataFrameTransformer, ensure_non_empty_rows};
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use regex::Regex;
use std::collections::HashMap;

/// Condition that a column has to satisfy for a rule to claim it.
///
/// Every variant is evaluated on the **input** column, before any transformer
/// runs.
pub enum ColumnPredicate {
    /// Matches columns whose dtype is `Float64`.
    IsFloat64,
    /// Matches columns whose dtype is any signed or unsigned integer.
    IsInt,
    /// Matches columns whose dtype is `String`.
    IsString,
    /// Matches columns whose dtype is `Datetime`.
    ///
    /// `Date` is not included; use a [`Custom`](ColumnPredicate::Custom) closure
    /// (`|col| col.dtype().is_date()`) when date columns should match too.
    IsDatetime,
    /// Matches columns whose dtype is `Boolean`.
    IsBoolean,
    /// Matches columns whose **name** matches the given regular expression.
    ///
    /// The pattern is compiled once per `fit`, not once per column, and an
    /// invalid pattern is reported as [`Error::InvalidInput`].
    NameMatches(String),
    /// Matches columns for which the closure returns `true`.
    ///
    /// The closure runs before any transformer and must not panic: a panic
    /// propagates out of `fit` (there is no `catch_unwind`). It is called once
    /// per column, during `fit` only — `transform` reuses the routing resolved at
    /// `fit` instead of re-evaluating the predicate.
    Custom(Box<dyn Fn(&Column) -> bool + Send + Sync>),
}

impl ColumnPredicate {
    /// Evaluate the predicate against one input column.
    ///
    /// `compiled` is the pre-compiled regex of a
    /// [`NameMatches`](ColumnPredicate::NameMatches) predicate; every other
    /// variant ignores it.
    fn matches(&self, compiled: Option<&Regex>, col: &Column) -> bool {
        match self {
            Self::IsFloat64 => matches!(col.dtype(), DataType::Float64),
            Self::IsInt => col.dtype().is_integer(),
            Self::IsString => col.dtype().is_string(),
            Self::IsDatetime => col.dtype().is_datetime(),
            Self::IsBoolean => col.dtype().is_bool(),
            Self::NameMatches(_) => compiled.is_some_and(|re| re.is_match(col.name().as_str())),
            Self::Custom(f) => f(col),
        }
    }

    /// Short human-readable form used in labels and error messages.
    fn describe(&self) -> String {
        match self {
            Self::IsFloat64 => "IsFloat64".into(),
            Self::IsInt => "IsInt".into(),
            Self::IsString => "IsString".into(),
            Self::IsDatetime => "IsDatetime".into(),
            Self::IsBoolean => "IsBoolean".into(),
            Self::NameMatches(pattern) => format!("NameMatches({pattern:?})"),
            Self::Custom(_) => "Custom".into(),
        }
    }
}

/// Route each column of a frame to the pipeline that matches it.
///
/// Unlike [`ColumnTransformer`](crate::pipeline::ColumnTransformer), columns are
/// not listed per transformer: each [`ColumnPredicate`] rule selects them from
/// the input at `fit` time. Columns claimed by no rule go to the transformer set
/// with [`ConditionalPipeline::fallback`]; without one they are dropped.
///
/// # Routing rules
///
/// - **First match wins.** Rules are evaluated in declaration order and only the
///   first matching rule claims a column, so a column is transformed at most
///   once and never appears twice in the output.
/// - Each rule's transformer receives **only** the columns that rule claimed.
/// - A rule that claims no column contributes nothing and is skipped.
/// - Every input column ends up in exactly one of three buckets: a rule, the
///   fallback, or dropped.
///
/// The routing is resolved **once, at `fit`**, and reused by `transform`
/// ([`assignments`](ConditionalPipeline::assignments) reports it). Columns are
/// consumed by name from the fitted plan, so an input column the plan does not
/// know is ignored and output columns come out grouped by rule — in rule order,
/// with the fallback last — rather than in input order. Re-fitting on a
/// different frame re-routes the columns, which is why a data-dependent
/// [`ColumnPredicate::Custom`] closure is only consulted during `fit`: evaluating
/// it again at transform time could send a column to a transformer that was never
/// fitted on it. A column the fitted plan cannot find is reported as
/// [`Error::InvalidInput`].
///
/// # Empty results
///
/// A configuration with neither rules nor a fallback is rejected by `fit`, as is
/// a `transform` where every transformer dropped the columns it received: in
/// both cases the output would have no columns at all. An individual unmatched
/// column being dropped is not an error, as long as something produced output.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::conditional_pipeline::{ColumnPredicate, ConditionalPipeline};
/// use featrs::preprocessing::encoder::OneHotEncoder;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let num = Column::from(Series::new("age".into(), &[20.0_f64, 30.0, 40.0]));
/// let cat = Column::from(Series::new("city".into(), &["oslo", "rome", "oslo"]));
/// let df = DataFrame::new(3, vec![num, cat])?;
///
/// let mut cp = ConditionalPipeline::new()
///     .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()))
///     .rule(ColumnPredicate::IsString, Box::new(OneHotEncoder::new()));
/// cp.fit(df.clone())?;
/// let out = cp.transform(df)?;
/// // The scaled numeric column, plus one one-hot column per city.
/// assert_eq!(out.width(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ConditionalPipeline {
    rules: Vec<(ColumnPredicate, Box<dyn DataFrameTransformer>)>,
    fallback: Option<Box<dyn DataFrameTransformer>>,
    /// Routing resolved by the last successful `fit`; also the fitted flag, so
    /// there is only one piece of state to keep consistent.
    plan: Option<Plan>,
}

/// Label given to the fallback transformer in error messages and assignments.
const FALLBACK_LABEL: &str = "fallback";

/// Label given to a column that no rule and no fallback claimed.
const DROPPED_LABEL: &str = "dropped";

fn rule_label(i: usize, predicate: &ColumnPredicate) -> String {
    format!("rule[{i}] ({})", predicate.describe())
}

fn wrap_error(e: Error, label: &str, phase: &str) -> Error {
    let msg = format!("ConditionalPipeline: transformer '{label}' failed during {phase}: {e}");
    match e {
        Error::NotFitted(_) => Error::NotFitted(msg),
        Error::InvalidInput(_) => Error::InvalidInput(msg),
        Error::Computation(_) => Error::Computation(msg),
    }
}

/// Columns routed to each transformer, resolved once at `fit`.
struct Plan {
    /// Columns claimed by each rule, indexed like `rules`.
    by_rule: Vec<Vec<String>>,
    /// Columns claimed by no rule, in frame order.
    unmatched: Vec<String>,
    /// `(column, label)` for every input column, in frame order.
    assignments: Vec<(String, String)>,
}

/// Resolve `rules` against `x`: compile every `NameMatches` pattern once, then
/// hand each column to the first rule that matches it.
///
/// `has_fallback` only affects the reported label: an unmatched column goes to
/// the fallback when one is configured, otherwise it is dropped.
fn resolve_plan(
    rules: &[(ColumnPredicate, Box<dyn DataFrameTransformer>)],
    has_fallback: bool,
    x: &DataFrame,
) -> Result<Plan> {
    let mut regexes: Vec<Option<Regex>> = Vec::with_capacity(rules.len());
    for (i, (predicate, _)) in rules.iter().enumerate() {
        match predicate {
            ColumnPredicate::NameMatches(pattern) => {
                let re = Regex::new(pattern).map_err(|e| {
                    Error::InvalidInput(format!(
                        "ConditionalPipeline: {} has an invalid NameMatches pattern {pattern:?}: \
                         {e}",
                        rule_label(i, predicate)
                    ))
                })?;
                regexes.push(Some(re));
            }
            _ => regexes.push(None),
        }
    }
    let mut by_rule: Vec<Vec<String>> = vec![Vec::new(); rules.len()];
    let mut unmatched: Vec<String> = Vec::new();
    let mut assignments: Vec<(String, String)> = Vec::with_capacity(x.width());

    for col in x.columns() {
        let name = col.name().to_string();
        let owner = rules
            .iter()
            .enumerate()
            .find(|(i, (predicate, _))| predicate.matches(regexes[*i].as_ref(), col));
        match owner {
            Some((i, (predicate, _))) => {
                assignments.push((name.clone(), rule_label(i, predicate)));
                by_rule[i].push(name);
            }
            None => {
                assignments.push((
                    name.clone(),
                    if has_fallback {
                        FALLBACK_LABEL.to_string()
                    } else {
                        DROPPED_LABEL.to_string()
                    },
                ));
                unmatched.push(name);
            }
        }
    }

    Ok(Plan {
        by_rule,
        unmatched,
        assignments,
    })
}

impl ConditionalPipeline {
    /// Create a new pipeline with no rules and no fallback.
    ///
    /// A pipeline without rules sends every column to the fallback, so configure
    /// at least one [`rule`](Self::rule) — or a [`fallback`](Self::fallback) —
    /// before fitting.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fallback: None,
            plan: None,
        }
    }

    /// Add a rule: every column matching `predicate` is routed to `transformer`.
    ///
    /// Rules are tried in the order they were added and the first match claims
    /// the column. Adding a rule after `fit` clears the resolved routing, so the
    /// pipeline has to be fitted again before it can transform.
    pub fn rule(
        mut self,
        predicate: ColumnPredicate,
        transformer: Box<dyn DataFrameTransformer>,
    ) -> Self {
        self.rules.push((predicate, transformer));
        self.plan = None;
        self
    }

    /// Set the transformer that receives the columns no rule claimed
    /// (default: none, so unmatched columns are dropped).
    ///
    /// The fallback receives only the unmatched columns, never the whole frame.
    /// Setting it after `fit` clears the resolved routing, so the pipeline has to
    /// be fitted again before it can transform.
    pub fn fallback(mut self, transformer: Box<dyn DataFrameTransformer>) -> Self {
        self.fallback = Some(transformer);
        self.plan = None;
        self
    }

    /// Number of configured rules.
    pub fn n_rules(&self) -> usize {
        self.rules.len()
    }

    /// Which transformer claimed each input column during the last successful
    /// `fit`, in frame order.
    ///
    /// Each entry is `(column, label)`, where the label is
    /// `rule[i] (<predicate>)`, `fallback`, or `dropped`. Returns `None` before
    /// the first successful `fit`.
    pub fn assignments(&self) -> Option<&[(String, String)]> {
        self.plan.as_ref().map(|p| p.assignments.as_slice())
    }

    /// Select the columns `label` claimed, fit `transformer` on them.
    fn fit_subset(
        transformer: &mut Box<dyn DataFrameTransformer>,
        columns: &[String],
        x: &DataFrame,
        label: &str,
    ) -> Result<()> {
        let subset = x.select(columns).map_err(|e| {
            Error::InvalidInput(format!(
                "ConditionalPipeline.fit: failed to select columns {columns:?} for transformer \
                 '{label}'. Available columns: {:?}. {e}",
                x.get_column_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
            ))
        })?;
        transformer
            .fit(subset)
            .map_err(|e| wrap_error(e, label, "fit"))
    }

    /// Select the columns `label` claimed, transform them, and check that the
    /// row count is preserved.
    fn transform_subset(
        transformer: &dyn DataFrameTransformer,
        columns: &[String],
        x: &DataFrame,
        label: &str,
    ) -> Result<DataFrame> {
        let subset = x.select(columns).map_err(|e| {
            Error::InvalidInput(format!(
                "ConditionalPipeline.transform: transformer '{label}' was fitted on columns \
                 {columns:?}, but they are not all present in the input. Available columns: \
                 {:?}. {e}",
                x.get_column_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
            ))
        })?;
        let out = transformer
            .transform(subset)
            .map_err(|e| wrap_error(e, label, "transform"))?;
        if out.height() != x.height() {
            return Err(Error::InvalidInput(format!(
                "ConditionalPipeline: transformer '{label}' returned {} rows but the input has \
                 {} rows.",
                out.height(),
                x.height()
            )));
        }
        Ok(out)
    }
}

impl Default for ConditionalPipeline {
    /// Same as [`ConditionalPipeline::new`]: no rules, no fallback.
    ///
    /// Without either the pipeline cannot produce output, so [`Fit::fit`]
    /// rejects it as invalid input.
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for ConditionalPipeline {
    type Output = ();

    /// Resolve the rules against `x`, then fit each transformer on the columns
    /// it claimed.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first: a failed re-fit must not leave a stale plan behind.
        // `plan.is_some()` is the fitted flag.
        self.plan = None;

        // Row count first: a 0-row frame built from an empty column list is
        // (0, 0) too, so width alone cannot tell it apart from
        // `DataFrame::empty()`.
        ensure_non_empty_rows(&x, "ConditionalPipeline.fit")?;
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "ConditionalPipeline.fit received a DataFrame with 0 columns.".into(),
            ));
        }
        if self.rules.is_empty() && self.fallback.is_none() {
            return Err(Error::InvalidInput(
                "ConditionalPipeline.fit: no rules and no fallback transformer were configured, \
                 so the output would have no columns. Add a rule with .rule(..) or a fallback \
                 with .fallback(..)."
                    .into(),
            ));
        }

        let plan = resolve_plan(&self.rules, self.fallback.is_some(), &x)?;

        for (i, columns) in plan.by_rule.iter().enumerate() {
            if columns.is_empty() {
                continue;
            }
            let label = rule_label(i, &self.rules[i].0);
            Self::fit_subset(&mut self.rules[i].1, columns, &x, &label)?;
        }

        if let Some(fallback) = self.fallback.as_mut()
            && !plan.unmatched.is_empty()
        {
            Self::fit_subset(fallback, &plan.unmatched, &x, FALLBACK_LABEL)?;
        }

        // Publish the plan only once every transformer that received columns
        // fitted, so a rejected configuration leaves nothing usable behind.
        self.plan = Some(plan);
        Ok(())
    }
}

impl Transform<DataFrame> for ConditionalPipeline {
    type Output = DataFrame;

    /// Route every column to the transformer that claimed it during `fit` and
    /// stack the results.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        // Fitted state first, matching `Pipeline`/`ColumnTransformer`: an
        // unfitted pipeline reports `NotFitted` whatever the frame looks like.
        // The routing decided at `fit` is reused as-is; re-evaluating a
        // data-dependent `Custom` predicate here could send a column to a
        // transformer that was never fitted on it.
        let plan = self.plan.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "ConditionalPipeline has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        ensure_non_empty_rows(&x, "ConditionalPipeline.transform")?;

        let mut parts: Vec<(String, DataFrame)> = Vec::new();

        for (i, columns) in plan.by_rule.iter().enumerate() {
            if columns.is_empty() {
                continue;
            }
            let label = rule_label(i, &self.rules[i].0);
            let out = Self::transform_subset(&*self.rules[i].1, columns, &x, &label)?;
            // A transformer that drops every column contributes nothing; the
            // row-count check already ran inside `transform_subset`.
            if out.width() == 0 {
                continue;
            }
            parts.push((label, out));
        }

        if let Some(fallback) = self.fallback.as_ref()
            && !plan.unmatched.is_empty()
        {
            let out = Self::transform_subset(&**fallback, &plan.unmatched, &x, FALLBACK_LABEL)?;
            if out.width() > 0 {
                parts.push((FALLBACK_LABEL.to_string(), out));
            }
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "ConditionalPipeline produced no output columns: every transformer dropped the \
                 columns it was given."
                    .into(),
            ));
        }

        // Stacking is the first point where all outgoing names are known, so the
        // collision check has to happen here: `hstack` would silently replace an
        // earlier column and name neither source. Input columns cannot collide,
        // because each one is routed to exactly one rule; only names a
        // transformer *generates* can clash.
        let mut owner: HashMap<&str, &str> = HashMap::new();
        for (label, part) in &parts {
            for name in part.get_column_names() {
                if let Some(prev) = owner.insert(name.as_str(), label.as_str()) {
                    return Err(Error::InvalidInput(format!(
                        "ConditionalPipeline: column '{}' is produced by more than one output \
                         source ({prev} and {label}). Rename one of the transformers' output \
                         columns.",
                        name
                    )));
                }
            }
        }

        let (_, mut result) = parts.remove(0);
        for (_, other) in parts {
            let cols = other.columns().to_vec();
            result = result.hstack(&cols).map_err(|e| {
                Error::Computation(format!(
                    "ConditionalPipeline: failed to stack the routed columns: {e}"
                ))
            })?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::binarizer::Binarizer;
    use crate::preprocessing::encoder::OneHotEncoder;
    use crate::preprocessing::feature_hasher::FeatureHasher;
    use crate::preprocessing::scaler::StandardScaler;

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

    /// Test-only transformer that passes its input through unchanged, for rules
    /// whose dtype no transformer in the crate accepts (e.g. `Int64`).
    struct Passthrough;

    impl Fit<DataFrame> for Passthrough {
        type Output = ();
        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for Passthrough {
        type Output = DataFrame;
        fn transform(&self, x: DataFrame) -> Result<DataFrame> {
            Ok(x)
        }
    }

    /// Test-only transformer that keeps only the first row.
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

    fn f64_col(name: &str, vals: &[f64]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    /// `age: Float64`, `score: Float64`, `city: String`, `is_active: Boolean`,
    /// `user_id: Int64`, `label: String`.
    fn make_test_df() -> DataFrame {
        let age = f64_col("age", &[20.0, 30.0, 40.0]);
        let score = f64_col("score", &[1.0, 2.0, 3.0]);
        let city = Column::from(Series::new("city".into(), &["oslo", "rome", "oslo"]));
        let active = Column::from(Series::new("is_active".into(), &[true, false, true]));
        let uid = Column::from(Series::new("user_id".into(), &[1_i64, 2, 3]));
        let label = Column::from(Series::new("label".into(), &["a", "b", "c"]));
        DataFrame::new(3, vec![age, score, city, active, uid, label]).unwrap()
    }

    #[test]
    fn test_two_rules_route_float_and_string_columns() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()))
            .rule(ColumnPredicate::IsString, Box::new(OneHotEncoder::new()));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();

        // StandardScaler keeps age + score; OneHotEncoder expands the two
        // string columns (city: 2 categories, label: 3).
        assert_eq!(out.height(), 3);
        assert!(out.get_column_names().iter().any(|n| n.as_str() == "age"));
        assert!(out.get_column_names().iter().any(|n| n.as_str() == "score"));
        // The integer and boolean columns matched no rule and were dropped.
        assert!(
            !out.get_column_names()
                .iter()
                .any(|n| n.as_str() == "user_id")
        );
        assert!(
            !out.get_column_names()
                .iter()
                .any(|n| n.as_str() == "is_active")
        );
    }

    #[test]
    fn test_assignments_report_each_column() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()))
            .rule(ColumnPredicate::IsInt, Box::new(Passthrough));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let a = cp.assignments().unwrap();
        assert_eq!(
            a,
            &[
                ("age".to_string(), "rule[0] (IsFloat64)".to_string()),
                ("score".to_string(), "rule[0] (IsFloat64)".to_string()),
                ("city".to_string(), "dropped".to_string()),
                ("is_active".to_string(), "dropped".to_string()),
                ("user_id".to_string(), "rule[1] (IsInt)".to_string()),
                ("label".to_string(), "dropped".to_string()),
            ]
        );
    }

    #[test]
    fn test_name_matches_selects_by_regex() {
        let mut cp = ConditionalPipeline::new().rule(
            ColumnPredicate::NameMatches(".*_id$".into()),
            Box::new(ColumnDropper),
        );
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        // user_id matched the rule and its transformer dropped it, so no other
        // column survives: the whole output is empty.
        let err = cp.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_name_matches_passes_through_with_fallback() {
        let mut cp = ConditionalPipeline::new()
            .rule(
                ColumnPredicate::NameMatches(".*_id$".into()),
                Box::new(ColumnDropper),
            )
            .fallback(Box::new(Passthrough));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();
        let names: Vec<&str> = out.get_column_names().iter().map(|s| s.as_str()).collect();
        // user_id matched the rule and was dropped; every other column went to
        // the fallback untouched, in frame order.
        assert_eq!(names, vec!["age", "score", "city", "is_active", "label"]);
    }

    #[test]
    fn test_custom_predicate_high_cardinality() {
        // A closure over column statistics: route the high-cardinality string
        // column to FeatureHasher, everything else to the dropping fallback.
        let values: Vec<String> = (0..200).map(|i| format!("cat_{i}")).collect();
        let mut cp = ConditionalPipeline::new()
            .rule(
                ColumnPredicate::Custom(Box::new(|col: &Column| {
                    col.n_unique().unwrap_or(usize::MAX) > 100
                })),
                Box::new(FeatureHasher::new(&["uid"], 4)),
            )
            .fallback(Box::new(ColumnDropper));
        let df =
            DataFrame::new(200, vec![Column::from(Series::new("uid".into(), values))]).unwrap();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 200);
        assert_eq!(cp.assignments().unwrap()[0].1, "rule[0] (Custom)");
    }

    #[test]
    fn test_first_match_wins() {
        // Both rules match Float64; only the first may claim the columns.
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(Binarizer::new(25.0)))
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();
        // Binarizer keeps the input names and binarizes against 25.0, so age
        // becomes 0/1 and score (all < 25) becomes all zeros. A scaled column
        // would have negative values, so checking age's values distinguishes
        // the two rules.
        let age = out.column("age").unwrap().f64().unwrap();
        assert_eq!(age.get(0), Some(0.0));
        assert_eq!(age.get(2), Some(1.0));
        // No column was produced twice.
        let mut names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "output columns must be unique");
        // The second rule claimed nothing, so nothing from it reached the plan.
        assert_eq!(cp.assignments().unwrap()[0].1, "rule[0] (IsFloat64)");
    }

    #[test]
    fn test_unmatched_column_dropped_without_fallback() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();
        assert_eq!(out.width(), 2);
        assert!(out.get_column_names().iter().any(|n| n.as_str() == "age"));
        assert!(out.get_column_names().iter().any(|n| n.as_str() == "score"));
    }

    #[test]
    fn test_unmatched_column_goes_to_fallback() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(ColumnDropper))
            .fallback(Box::new(Passthrough));
        let df = make_test_df();

        cp.fit(df.clone()).unwrap();
        let out = cp.transform(df).unwrap();
        // The two float columns were dropped by the rule; the fallback passes
        // the four unmatched columns through unchanged, in frame order.
        let names: Vec<&str> = out.get_column_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(names, vec!["city", "is_active", "user_id", "label"]);
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn test_transform_reuses_the_fit_plan() {
        // The closure matches only the 200-value frame. If `transform`
        // re-evaluated the predicate it would route the 3-value frame to the
        // fallback, sending the column to a transformer that never fitted it.
        let many: Vec<String> = (0..200).map(|i| format!("cat_{i}")).collect();
        let few: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
        let mut cp = ConditionalPipeline::new()
            .rule(
                ColumnPredicate::Custom(Box::new(|col: &Column| {
                    col.n_unique().unwrap_or(usize::MAX) > 100
                })),
                Box::new(FeatureHasher::new(&["uid"], 2)),
            )
            .fallback(Box::new(Passthrough));

        let train =
            DataFrame::new(200, vec![Column::from(Series::new("uid".into(), many))]).unwrap();
        cp.fit(train).unwrap();

        let test = DataFrame::new(3, vec![Column::from(Series::new("uid".into(), few))]).unwrap();
        let out = cp.transform(test).unwrap();
        // FeatureHasher output, not the pass-through column.
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn test_not_fitted() {
        let cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let err = cp.transform(make_test_df()).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
    }

    #[test]
    fn test_fit_rejects_zero_row_frame() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let empty = DataFrame::new(0, vec![f64_col("a", &[])]).unwrap();
        let err = cp.fit(empty).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_fit_rejects_zero_column_frame() {
        // polars allows a (3 rows, 0 columns) frame, which the row check alone
        // cannot reject.
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let err = cp.fit(DataFrame::new(3, vec![]).unwrap()).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("0 columns"), "got {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_rejects_zero_row_frame() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let df = make_test_df();
        cp.fit(df).unwrap();
        let empty = DataFrame::new(0, vec![f64_col("age", &[])]).unwrap();
        let err = cp.transform(empty).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_fit_rejects_no_rules_and_no_fallback() {
        let mut cp = ConditionalPipeline::new();
        let err = cp.fit(make_test_df()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_default_is_empty_and_unusable() {
        let mut cp = ConditionalPipeline::default();
        assert_eq!(cp.n_rules(), 0);
        assert!(cp.assignments().is_none());
        assert!(cp.fit(make_test_df()).is_err());
    }

    #[test]
    fn test_invalid_regex_is_rejected_at_fit() {
        let mut cp = ConditionalPipeline::new().rule(
            ColumnPredicate::NameMatches("([".into()),
            Box::new(ColumnDropper),
        );
        let err = cp.fit(make_test_df()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
        assert!(cp.assignments().is_none());
    }

    #[test]
    fn test_failed_refit_resets_fitted() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let df = make_test_df();
        cp.fit(df.clone()).unwrap();
        assert!(cp.assignments().is_some());

        // An empty frame must clear the previous plan.
        let empty = DataFrame::new(0, vec![f64_col("a", &[])]).unwrap();
        assert!(cp.fit(empty).is_err());
        assert!(cp.assignments().is_none());
        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
    }

    #[test]
    fn test_transform_rejects_row_count_change() {
        let mut cp =
            ConditionalPipeline::new().rule(ColumnPredicate::IsFloat64, Box::new(RowDropper));
        let df = make_test_df();
        cp.fit(df.clone()).unwrap();
        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_all_rules_drop_every_column_errors() {
        let mut cp =
            ConditionalPipeline::new().rule(ColumnPredicate::IsFloat64, Box::new(ColumnDropper));
        let df = make_test_df();
        cp.fit(df.clone()).unwrap();
        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_generated_name_collision_is_reported() {
        // Both rules claim disjoint input columns but generate the same output
        // name, which `hstack` would resolve silently.
        let mut cp = ConditionalPipeline::new()
            .rule(
                ColumnPredicate::NameMatches("^a$".into()),
                Box::new(FeatureHasher::new(&["a"], 1)),
            )
            .rule(
                ColumnPredicate::NameMatches("^b$".into()),
                Box::new(FeatureHasher::new(&["b"], 1)),
            );
        let df = DataFrame::new(
            3,
            vec![
                Column::from(Series::new("a".into(), &["x", "y", "x"])),
                Column::from(Series::new("b".into(), &["p", "q", "p"])),
            ],
        )
        .unwrap();
        cp.fit(df.clone()).unwrap();
        // Both transformers emit `hashed_0`.
        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_transform_missing_column_errors() {
        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()));
        let df = make_test_df();
        cp.fit(df.clone()).unwrap();
        let renamed = DataFrame::new(3, vec![f64_col("other", &[1.0, 2.0, 3.0])]).unwrap();
        let err = cp.transform(renamed).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_datetime_and_boolean_predicates() {
        let bools = Column::from(Series::new("flag".into(), &[true, false]));
        let datetimes = Column::from(Series::new("ts".into(), &[1_i64, 2]))
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        // `IsDatetime` deliberately excludes `Date`, so this column is claimed by
        // no rule and has to reach the fallback instead.
        let dates = Column::from(Series::new("d".into(), &[1_i32, 2]))
            .cast(&DataType::Date)
            .unwrap();

        let mut cp = ConditionalPipeline::new()
            .rule(ColumnPredicate::IsBoolean, Box::new(Passthrough))
            .rule(ColumnPredicate::IsDatetime, Box::new(ColumnDropper))
            .fallback(Box::new(Passthrough));
        let df = DataFrame::new(2, vec![bools, datetimes, dates]).unwrap();

        cp.fit(df.clone()).unwrap();
        assert_eq!(
            cp.assignments().unwrap(),
            [
                ("flag".to_string(), "rule[0] (IsBoolean)".to_string()),
                ("ts".to_string(), "rule[1] (IsDatetime)".to_string()),
                ("d".to_string(), FALLBACK_LABEL.to_string()),
            ]
        );

        let out = cp.transform(df).unwrap();
        // `flag` claimed by its rule, `ts` dropped by its rule, `d` passed
        // through by the fallback.
        assert_eq!(
            out.get_column_names()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            vec!["flag", "d"]
        );
    }
}
