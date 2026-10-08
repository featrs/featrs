//! Automatic per-column imputation strategy selection.
//!
//! [`AutoImputer`] inspects the missing-value pattern and the distribution of
//! every `Float64` column at fit time and delegates each column to the
//! [`SimpleImputer`] strategy that matches it, instead of making the caller
//! choose between [`Strategy::Mean`], [`Strategy::Median`],
//! [`Strategy::MostFrequent`] and [`Strategy::Constant`] by hand.
//!
//! Columns of other dtypes pass through unchanged: [`SimpleImputer`] only
//! operates on `Float64`, so `String`, integer, boolean and datetime columns
//! keep their missing values. Adding those dtypes needs a `SimpleImputer`
//! extension (a string/instant constant), which is out of scope here.

use polars::prelude::*;

use super::imputer::{SimpleImputer, Strategy};
use crate::pipeline::DataFrameTransformer;
use crate::pipeline::column_transformer::{ColumnTransformer, Remainder};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::require_f64_columns;

/// Default null fraction at or above which a column is left alone.
const DEFAULT_HIGH_NULL_THRESHOLD: f64 = 0.5;

/// Default `|skewness|` above which a column is imputed with its median.
const DEFAULT_SKEW_THRESHOLD: f64 = 1.0;

/// Value used to fill an all-null column under [`NullColumnBehavior::Fill`].
const NULL_FILL_VALUE: f64 = 0.0;

/// What to do with a `Float64` column that is entirely null.
///
/// An all-null column has no value any statistic-based strategy could use:
/// `Mean`, `Median` and `MostFrequent` all reject it. The behavior is therefore
/// configurable rather than derived from the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullColumnBehavior {
    /// Remove the column from the output (default).
    #[default]
    Drop,
    /// Fail `fit` with [`Error::InvalidInput`].
    Error,
    /// Replace every null with the constant `0.0`.
    Fill,
}

/// Imputation strategy selected for a column.
///
/// Mirrors [`Strategy`] for the strategies that are actually delegated, plus
/// the two decisions that have no [`SimpleImputer`] counterpart: a column left
/// untouched and a column removed.
#[derive(Debug, Clone, Copy)]
pub enum ImputationStrategy {
    /// Fill missing values with the column mean.
    Mean,
    /// Fill missing values with the column median.
    Median,
    /// Fill missing values with the column's most frequent value.
    MostFrequent,
    /// Fill missing values with a constant value.
    Constant(f64),
    /// Too many values are missing; the column is left unchanged.
    Skipped,
    /// The column was entirely null and is removed from the output.
    Dropped,
}

/// Equality that compares a [`ImputationStrategy::Constant`] by its bit
/// pattern, so two identical constants always group into one delegate. The
/// derived `f64` comparison would let `Constant(NaN) == Constant(NaN)` be
/// false and split the same column value across two imputers.
impl PartialEq for ImputationStrategy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Mean, Self::Mean)
            | (Self::Median, Self::Median)
            | (Self::MostFrequent, Self::MostFrequent)
            | (Self::Skipped, Self::Skipped)
            | (Self::Dropped, Self::Dropped) => true,
            (Self::Constant(a), Self::Constant(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}
/// Automatically chooses an imputation strategy per `Float64` column.
///
/// At fit time every `Float64` column is inspected and handed to the
/// [`SimpleImputer`] strategy that matches its distribution. Columns delegated
/// to the same strategy are imputed together; all other columns pass through
/// untouched.
///
/// # Decision order
///
/// 1. A column with no missing value needs nothing and is left out of the plan.
/// 2. An **entirely null** column has no value to compute a statistic from, so
///    it follows [`NullColumnBehavior`]: dropped (default), an error, or filled
///    with the constant `0.0`. A forced [`Strategy::Constant`] applies instead,
///    because it needs no statistic.
/// 3. A column whose null fraction is at or above
///    [`high_null_threshold`](AutoImputer::high_null_threshold) (default `0.5`)
///    is recorded as [`ImputationStrategy::Skipped`] and left unchanged: the
///    surviving values are too few to describe the column.
/// 4. Otherwise the heuristic picks [`ImputationStrategy::Median`] when
///    `|skewness| >` [`skew_threshold`](AutoImputer::skew_threshold)
///    (default `1.0`) and [`ImputationStrategy::Mean`] otherwise.
/// 5. [`force_strategy`](AutoImputer::force_strategy) replaces steps 3 and 4
///    with one strategy for every column. Step 2 still applies, because a
///    forced `Mean`/`Median`/`MostFrequent` cannot be computed on an all-null
///    column either.
///
/// # Output layout
///
/// Delegation runs through a [`ColumnTransformer`] with
/// [`Remainder::Passthrough`], so output columns come out **grouped by the
/// strategy that was chosen, with the pass-through remainder last** — not in
/// the input order. Column names are preserved, so select by name rather than
/// by position when the order matters. Columns dropped by
/// [`NullColumnBehavior::Drop`] are removed from the result.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::auto_imputer::AutoImputer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[Some(1.0_f64), None, Some(3.0)]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut imp = AutoImputer::new();
/// imp.fit(df.clone())?;
/// let filled = imp.transform(df)?;
/// assert_eq!(filled.height(), 3);
/// assert_eq!(imp.chosen_name(), Some("Mean"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AutoImputer {
    fitted: bool,
    force_strategy: Option<Strategy>,
    high_null_threshold: f64,
    skew_threshold: f64,
    null_column_behavior: NullColumnBehavior,
    chosen: Option<Box<dyn DataFrameTransformer>>,
    chosen_name: Option<String>,
    column_types: Option<Vec<(String, ImputationStrategy)>>,
    dropped: Vec<String>,
}

impl AutoImputer {
    /// Create a new `AutoImputer` using the per-column heuristic.
    pub fn new() -> Self {
        Self {
            fitted: false,
            force_strategy: None,
            high_null_threshold: DEFAULT_HIGH_NULL_THRESHOLD,
            skew_threshold: DEFAULT_SKEW_THRESHOLD,
            null_column_behavior: NullColumnBehavior::default(),
            chosen: None,
            chosen_name: None,
            column_types: None,
            dropped: Vec::new(),
        }
    }

    /// Clear the fitted plan so a stale selection cannot be reused after a
    /// configuration change.
    fn clear_plan(&mut self) {
        self.fitted = false;
        self.chosen = None;
        self.chosen_name = None;
        self.column_types = None;
        self.dropped.clear();
    }

    /// Force one [`Strategy`] for every imputed column instead of using the
    /// heuristic.
    ///
    /// The forced strategy also replaces the high-null skip, so a column whose
    /// null fraction exceeds
    /// [`high_null_threshold`](AutoImputer::high_null_threshold) is imputed
    /// anyway. A forced [`Strategy::Constant`] also applies to an entirely null
    /// column, which is the one case [`NullColumnBehavior`] otherwise governs;
    /// `Mean`, `Median` and `MostFrequent` still cannot be computed from no
    /// values and fall back to it.
    ///
    /// Changing the strategy after `fit` requires a re-`fit`: the fitted plan is
    /// cleared here so a stale selection cannot be reused.
    pub fn force_strategy(mut self, s: Strategy) -> Self {
        self.force_strategy = Some(s);
        self.clear_plan();
        self
    }

    /// Null fraction at or above which a column is left unchanged instead of
    /// imputed (default `0.5`). Values above `1.0` disable the skip.
    pub fn high_null_threshold(mut self, t: f64) -> Self {
        self.high_null_threshold = t;
        self.clear_plan();
        self
    }

    /// `|skewness|` above which a column is imputed with its median instead of
    /// its mean (default `1.0`). A `NaN` here disables the heuristic and picks
    /// the mean for every column.
    pub fn skew_threshold(mut self, s: f64) -> Self {
        self.skew_threshold = s;
        self.clear_plan();
        self
    }

    /// How to handle a `Float64` column that is entirely null
    /// (default: [`NullColumnBehavior::Drop`]).
    pub fn null_column_behavior(mut self, b: NullColumnBehavior) -> Self {
        self.null_column_behavior = b;
        self.clear_plan();
        self
    }

    /// The strategy chosen for each `Float64` column that needed a decision, in
    /// frame order: imputed columns, [`ImputationStrategy::Skipped`] columns,
    /// and [`ImputationStrategy::Dropped`] columns. Columns with no missing
    /// value are not listed, because they were not touched.
    pub fn column_types(&self) -> Option<&[(String, ImputationStrategy)]> {
        self.column_types.as_deref()
    }

    /// Name of the strategy(s) selected during `fit`, e.g. `"Mean"` or
    /// `"Mean+Median"` for a mixed selection.
    pub fn chosen_name(&self) -> Option<&str> {
        self.chosen_name.as_deref()
    }
}

impl Default for AutoImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for AutoImputer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave a stale plan usable by a
        // later transform.
        self.clear_plan();

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "AutoImputer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "AutoImputer")?;
        let height = x.height() as f64;

        let mut plan: Vec<(String, ImputationStrategy)> = Vec::new();
        let mut dropped: Vec<String> = Vec::new();

        for name in &col_names {
            let col = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!("AutoImputer.fit: column '{name}' not found. {e}"))
            })?;
            let ca = col.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "AutoImputer.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    col.dtype()
                ))
            })?;

            // One pass: the non-null values feed the skew test, the null count
            // feeds the null-fraction test.
            let mut vals: Vec<f64> = Vec::with_capacity(ca.len());
            let mut nulls = 0usize;
            for v in ca.iter() {
                match v {
                    Some(v) => vals.push(v),
                    None => nulls += 1,
                }
            }
            if nulls == 0 {
                continue;
            }

            if vals.is_empty() {
                // A forced `Constant` needs no statistic, so it applies here
                // too; `Mean`/`Median`/`MostFrequent` cannot be computed from
                // no values and follow `NullColumnBehavior` instead.
                if let Some(Strategy::Constant(v)) = self.force_strategy {
                    plan.push((name.clone(), ImputationStrategy::Constant(v)));
                    continue;
                }
                match self.null_column_behavior {
                    NullColumnBehavior::Drop => {
                        dropped.push(name.clone());
                        plan.push((name.clone(), ImputationStrategy::Dropped));
                    }
                    NullColumnBehavior::Error => {
                        return Err(Error::InvalidInput(format!(
                            "AutoImputer.fit: column '{name}' is entirely null, so no \
                             imputation statistic can be computed from it. Drop the column \
                             or set NullColumnBehavior::Fill."
                        )));
                    }
                    NullColumnBehavior::Fill => {
                        plan.push((name.clone(), ImputationStrategy::Constant(NULL_FILL_VALUE)));
                    }
                }
                continue;
            }

            let strat = match self.force_strategy {
                Some(s) => from_delegate_strategy(s),
                None if nulls as f64 / height >= self.high_null_threshold => {
                    ImputationStrategy::Skipped
                }
                None => {
                    if skewness(&vals).abs() > self.skew_threshold {
                        ImputationStrategy::Median
                    } else {
                        ImputationStrategy::Mean
                    }
                }
            };
            plan.push((name.clone(), strat));
        }

        let composite = fit_plan(&plan, &x)?;

        let mut seen: Vec<ImputationStrategy> = Vec::new();
        for (_, s) in &plan {
            if !seen.contains(s) {
                seen.push(*s);
            }
        }
        let chosen_name = seen
            .iter()
            .map(|s| strategy_name(*s))
            .collect::<Vec<_>>()
            .join("+");

        self.dropped = dropped;
        self.column_types = Some(plan);
        self.chosen_name = if chosen_name.is_empty() {
            None
        } else {
            Some(chosen_name)
        };
        self.chosen = Some(Box::new(composite));
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for AutoImputer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "AutoImputer has not been fitted. Call .fit(dataframe) before .transform().".into(),
            ));
        }
        let chosen = self.chosen.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "AutoImputer has not been fitted. Call .fit(dataframe) before .transform().".into(),
            )
        })?;
        let mut out = chosen.transform(x)?;
        if !self.dropped.is_empty() {
            out = out.drop_many(self.dropped.iter().map(|s| s.as_str()));
        }
        Ok(out)
    }
}

/// Sample skewness (third standardised moment) of the non-null values.
///
/// The deviations are divided by the standard deviation before being cubed, so
/// the third moment cannot overflow; skewness is scale-invariant, so the value
/// is unchanged. Returns `0.0` when any step comes out non-finite (a constant
/// column, or a column whose values already overflow the mean), so a degenerate
/// column is treated as symmetric rather than propagating a `NaN` into the
/// comparison.
fn skewness(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    if !mean.is_finite() {
        return 0.0;
    }
    let m2 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sd = m2.sqrt();
    if sd == 0.0 || !sd.is_finite() {
        return 0.0;
    }
    let m3 = vals.iter().map(|v| ((v - mean) / sd).powi(3)).sum::<f64>() / n;
    if m3.is_finite() { m3 } else { 0.0 }
}

/// Map a caller-forced [`Strategy`] onto the reported strategy.
fn from_delegate_strategy(s: Strategy) -> ImputationStrategy {
    match s {
        Strategy::Mean => ImputationStrategy::Mean,
        Strategy::Median => ImputationStrategy::Median,
        Strategy::MostFrequent => ImputationStrategy::MostFrequent,
        Strategy::Constant(v) => ImputationStrategy::Constant(v),
    }
}

/// Human-readable strategy name, used for `chosen_name`.
fn strategy_name(s: ImputationStrategy) -> String {
    match s {
        ImputationStrategy::Mean => "Mean".to_string(),
        ImputationStrategy::Median => "Median".to_string(),
        ImputationStrategy::MostFrequent => "MostFrequent".to_string(),
        ImputationStrategy::Constant(v) => format!("Constant({v})"),
        ImputationStrategy::Skipped => "Skipped".to_string(),
        ImputationStrategy::Dropped => "Dropped".to_string(),
    }
}

/// Instantiate the delegate for a concrete strategy. `Skipped` and `Dropped`
/// have no delegate — those columns are left to the pass-through remainder (and
/// removed afterwards, for `Dropped`).
fn make_imputer(s: ImputationStrategy) -> Option<Box<dyn DataFrameTransformer>> {
    match s {
        ImputationStrategy::Mean => Some(Box::new(SimpleImputer::mean())),
        ImputationStrategy::Median => Some(Box::new(SimpleImputer::median())),
        ImputationStrategy::MostFrequent => Some(Box::new(SimpleImputer::most_frequent())),
        ImputationStrategy::Constant(v) => Some(Box::new(SimpleImputer::constant(v))),
        ImputationStrategy::Skipped | ImputationStrategy::Dropped => None,
    }
}

/// Group a per-column plan by strategy, build the delegating
/// [`ColumnTransformer`], and fit it.
///
/// One imputer per distinct strategy, each bound to the columns that chose it,
/// plus [`Remainder::Passthrough`] for the columns `AutoImputer` leaves alone.
///
/// Grouping is by strategy value, not by name: a forced `Constant(5.0)` next to
/// a [`NullColumnBehavior::Fill`] column using `Constant(0.0)` must produce two
/// delegates, and a name-keyed grouping would silently merge them. A `Constant`
/// compares by bit pattern, so two identical values always share a delegate.
///
/// ponytail: one `ColumnTransformer` per fit, even when every column landed on
/// the same strategy. A uniform-strategy fast path would skip the grouping, at
/// the cost of a second code path to keep in sync; add one if profiling shows
/// the grouping matters.
fn fit_plan(plan: &[(String, ImputationStrategy)], x: &DataFrame) -> Result<ColumnTransformer> {
    let mut entries: Vec<(
        ImputationStrategy,
        Box<dyn DataFrameTransformer>,
        Vec<String>,
    )> = Vec::new();
    for (name, strat) in plan {
        let imputer = match make_imputer(*strat) {
            Some(i) => i,
            None => continue,
        };
        match entries.iter_mut().find(|(s, _, _)| s == strat) {
            Some((_, _, cols)) => cols.push(name.clone()),
            None => entries.push((*strat, imputer, vec![name.clone()])),
        }
    }
    let named = entries
        .into_iter()
        .map(|(s, t, cols)| (strategy_name(s), t, cols))
        .collect();
    let mut composite = ColumnTransformer::new(named, Remainder::Passthrough);
    composite.fit(x.clone())?;
    Ok(composite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn f64_col(name: &str, vals: &[Option<f64>]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    fn str_col(name: &str, vals: &[Option<&str>]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    fn frame(cols: Vec<Column>, height: usize) -> DataFrame {
        DataFrame::new(height, cols).unwrap()
    }

    fn vals(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
        df.column(name).unwrap().f64().unwrap().iter().collect()
    }

    /// Symmetric column: skewness 0.
    fn symmetric_col(name: &str) -> Column {
        f64_col(
            name,
            &[
                Some(-2.0),
                Some(-1.0),
                Some(0.0),
                None,
                Some(1.0),
                Some(2.0),
            ],
        )
    }

    /// Right-skewed column: one large outlier pulls the mean far above the
    /// median, so skewness is well above the default threshold of 1.0.
    fn skewed_col(name: &str) -> Column {
        f64_col(
            name,
            &[
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(4.0),
                Some(100.0),
                None,
            ],
        )
    }

    #[test]
    fn test_symmetric_column_chooses_mean() {
        let df = frame(vec![symmetric_col("x")], 6);
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean"));
        let out = imp.transform(df).unwrap();
        // mean of [-2, -1, 0, 1, 2] = 0.0
        assert_relative_eq!(vals(&out, "x")[3].unwrap(), 0.0);
    }

    #[test]
    fn test_skewed_column_chooses_median() {
        let df = frame(vec![skewed_col("x")], 6);
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Median"));
        let out = imp.transform(df).unwrap();
        // median of [1, 2, 3, 4, 100] = 3.0
        assert_relative_eq!(vals(&out, "x")[5].unwrap(), 3.0);
    }

    #[test]
    fn test_high_null_fraction_column_is_skipped() {
        // 6 of 10 values missing: at the default 0.5 threshold.
        let df = frame(
            vec![f64_col(
                "x",
                &[
                    Some(1.0),
                    Some(2.0),
                    Some(3.0),
                    Some(4.0),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            )],
            10,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Skipped"));
        assert_eq!(
            imp.column_types().unwrap(),
            [("x".to_string(), ImputationStrategy::Skipped)]
        );
        let out = imp.transform(df).unwrap();
        assert!(vals(&out, "x")[9].is_none());
    }

    #[test]
    fn test_all_null_column_dropped_by_default() {
        let df = frame(
            vec![
                f64_col("x", &[Some(1.0), Some(2.0)]),
                f64_col("y", &[None, None]),
            ],
            2,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Dropped"));
        let out = imp.transform(df).unwrap();
        assert_eq!(out.get_column_names(), &["x"]);
        assert_eq!(out.height(), 2);
    }

    #[test]
    fn test_all_null_column_error_behavior() {
        let df = frame(vec![f64_col("y", &[None, None])], 2);
        let mut imp = AutoImputer::new().null_column_behavior(NullColumnBehavior::Error);
        let err = imp.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err}");
        assert!(err.to_string().contains("entirely null"));
    }

    #[test]
    fn test_all_null_column_fill_behavior() {
        let df = frame(vec![f64_col("y", &[None, None, None])], 3);
        let mut imp = AutoImputer::new().null_column_behavior(NullColumnBehavior::Fill);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Constant(0)"));
        let out = imp.transform(df).unwrap();
        assert_eq!(vals(&out, "y"), vec![Some(0.0), Some(0.0), Some(0.0)]);
    }

    #[test]
    fn test_force_strategy_overrides_heuristic() {
        let df = frame(vec![skewed_col("x")], 6);
        let mut imp = AutoImputer::new().force_strategy(Strategy::Mean);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean"));
        let out = imp.transform(df).unwrap();
        // mean of [1, 2, 3, 4, 100] = 22.0, not the median.
        assert_relative_eq!(vals(&out, "x")[5].unwrap(), 22.0);
    }

    #[test]
    fn test_force_strategy_bypasses_high_null_threshold() {
        let df = frame(
            vec![f64_col(
                "x",
                &[
                    Some(2.0),
                    Some(4.0),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            )],
            10,
        );
        let mut imp = AutoImputer::new().force_strategy(Strategy::Median);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Median"));
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "x")[9].unwrap(), 3.0);
    }

    #[test]
    fn test_high_null_threshold_knob() {
        let df = frame(
            vec![f64_col(
                "x",
                &[
                    Some(1.0),
                    Some(2.0),
                    Some(3.0),
                    Some(4.0),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            )],
            10,
        );
        let mut imp = AutoImputer::new().high_null_threshold(0.9);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean"));
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "x")[9].unwrap(), 2.5);
    }

    #[test]
    fn test_skew_threshold_knob() {
        let df = frame(vec![skewed_col("x")], 6);
        let mut imp = AutoImputer::new().skew_threshold(10.0);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean"));
    }

    #[test]
    fn test_transform_before_fit_is_not_fitted() {
        let df = frame(vec![symmetric_col("x")], 6);
        let imp = AutoImputer::new();
        let err = imp.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err}");
    }

    #[test]
    fn test_empty_input_is_invalid() {
        let mut imp = AutoImputer::new();
        assert!(matches!(
            imp.fit(DataFrame::empty()).unwrap_err(),
            Error::InvalidInput(_)
        ));
        let zero_rows = frame(vec![f64_col("x", &[])], 0);
        assert!(matches!(
            imp.fit(zero_rows).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn test_no_float64_columns_is_invalid() {
        let df = frame(vec![str_col("s", &[Some("a"), Some("b")])], 2);
        let mut imp = AutoImputer::new();
        let err = imp.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err}");
        assert!(err.to_string().contains("no Float64 columns"));
    }

    #[test]
    fn test_no_missing_values_is_a_passthrough() {
        let df = frame(vec![f64_col("x", &[Some(1.0), Some(2.0)])], 2);
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), None);
        assert_eq!(imp.column_types().unwrap(), []);
        let out = imp.transform(df.clone()).unwrap();
        assert_eq!(out.get_column_names(), df.get_column_names());
        assert_eq!(vals(&out, "x"), vals(&df, "x"));
    }

    #[test]
    fn test_mixed_strategy_frame_fills_every_null() {
        let df = frame(
            vec![
                symmetric_col("symmetric"),
                skewed_col("skewed"),
                str_col(
                    "label",
                    &[Some("a"), Some("b"), None, Some("a"), Some("b"), None],
                ),
            ],
            6,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean+Median"));
        assert_eq!(
            imp.column_types().unwrap(),
            [
                ("symmetric".to_string(), ImputationStrategy::Mean),
                ("skewed".to_string(), ImputationStrategy::Median),
            ]
        );

        let out = imp.transform(df).unwrap();
        assert_eq!(out.height(), 6);
        assert_eq!(vals(&out, "symmetric")[3].unwrap(), 0.0);
        assert_relative_eq!(vals(&out, "skewed")[5].unwrap(), 3.0);
        // The String column is out of scope: passed through with its nulls.
        assert!(out.column("label").unwrap().str().unwrap().get(2).is_none());
    }

    #[test]
    fn test_all_columns_dropped_keeps_rows() {
        let df = frame(vec![f64_col("y", &[None, None, None])], 3);
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        let out = imp.transform(df).unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.width(), 0);
    }

    #[test]
    fn test_refit_replaces_previous_plan() {
        let mut imp = AutoImputer::new();
        imp.fit(frame(vec![symmetric_col("x")], 6)).unwrap();
        assert_eq!(imp.chosen_name(), Some("Mean"));

        imp.fit(frame(vec![skewed_col("x")], 6)).unwrap();
        assert_eq!(imp.chosen_name(), Some("Median"));
        assert_eq!(
            imp.column_types().unwrap(),
            [("x".to_string(), ImputationStrategy::Median)]
        );
    }

    #[test]
    fn test_failed_refit_clears_previous_plan() {
        let mut imp = AutoImputer::new();
        imp.fit(frame(vec![symmetric_col("x")], 6)).unwrap();
        assert!(imp.fitted);

        // A frame with no Float64 columns fails; the stale plan must be gone.
        let bad = frame(vec![str_col("s", &[Some("a")])], 1);
        assert!(imp.fit(bad).is_err());
        assert!(!imp.fitted);
        assert_eq!(imp.chosen_name(), None);
        assert!(matches!(
            imp.transform(frame(vec![symmetric_col("x")], 6))
                .unwrap_err(),
            Error::NotFitted(_)
        ));
    }

    #[test]
    fn test_builder_knob_invalidates_fitted_plan() {
        let df = frame(vec![skewed_col("x")], 6);
        let mut imp = AutoImputer::new();
        imp.fit(df).unwrap();
        assert_eq!(imp.chosen_name(), Some("Median"));

        imp = imp.skew_threshold(10.0);
        assert!(!imp.fitted);
        assert_eq!(imp.chosen_name(), None);
    }

    #[test]
    fn test_forced_constant_fills_every_column_with_one_delegate() {
        // A forced Constant applies to partially and entirely null columns
        // alike, and both must land on a single delegate.
        let df = frame(
            vec![
                f64_col("forced", &[Some(1.0), None, Some(3.0)]),
                f64_col("empty", &[None, None, None]),
            ],
            3,
        );
        let mut imp = AutoImputer::new().force_strategy(Strategy::Constant(5.0));
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Constant(5)"));
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "forced")[1].unwrap(), 5.0);
        assert_relative_eq!(vals(&out, "empty")[0].unwrap(), 5.0);
        assert!(vals(&out, "empty")[2].is_some());
    }

    #[test]
    fn test_force_constant_fills_all_null_column() {
        // A forced Constant needs no statistic, so it wins over the default
        // Drop behavior for an entirely null column.
        let df = frame(vec![f64_col("y", &[None, None])], 2);
        let mut imp = AutoImputer::new().force_strategy(Strategy::Constant(5.0));
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Constant(5)"));
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "y")[0].unwrap(), 5.0);
    }

    #[test]
    fn test_force_most_frequent() {
        let df = frame(
            vec![f64_col("x", &[Some(2.0), Some(2.0), Some(7.0), None])],
            4,
        );
        let mut imp = AutoImputer::new().force_strategy(Strategy::MostFrequent);
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("MostFrequent"));
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "x")[3].unwrap(), 2.0);
    }

    #[test]
    fn test_output_groups_by_strategy_with_remainder_last() {
        let df = frame(
            vec![
                symmetric_col("mean_col"),
                str_col(
                    "label",
                    &[Some("a"), Some("b"), None, Some("a"), Some("b"), None],
                ),
                skewed_col("median_col"),
            ],
            6,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        let out = imp.transform(df).unwrap();
        // Grouped by strategy in plan order, pass-through remainder last.
        assert_eq!(out.get_column_names(), &["mean_col", "median_col", "label"]);
    }

    #[test]
    fn test_overflowing_skew_statistic_does_not_hide_skew() {
        // The third moment overflows here; the helper must not report that as
        // "symmetric" and pick the mean for a clearly skewed column.
        let df = frame(
            vec![f64_col(
                "x",
                &[Some(1e103), Some(0.0), Some(0.0), Some(0.0), None],
            )],
            5,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        assert_eq!(imp.chosen_name(), Some("Median"));
    }

    #[test]
    fn test_forced_constant_nan_uses_one_delegate() {
        // Constant(NaN) != Constant(NaN) under a derived f64 comparison, which
        // would split the same value across two delegates.
        let df = frame(
            vec![
                f64_col("a", &[Some(1.0), None]),
                f64_col("b", &[Some(2.0), None]),
            ],
            2,
        );
        let mut imp = AutoImputer::new().force_strategy(Strategy::Constant(f64::NAN));
        imp.fit(df.clone()).unwrap();
        let out = imp.transform(df).unwrap();
        assert_eq!(out.get_column_names(), &["a", "b"]);
        assert!(vals(&out, "a")[1].unwrap().is_nan());
        assert!(vals(&out, "b")[1].unwrap().is_nan());
    }

    #[test]
    fn test_datetime_and_int_columns_pass_through() {
        let ints: Vec<Option<i64>> = vec![Some(1), None, Some(3)];
        let df = frame(
            vec![
                f64_col("x", &[Some(1.0), None, Some(3.0)]),
                Column::from(Series::new("n".into(), &ints)),
            ],
            3,
        );
        let mut imp = AutoImputer::new();
        imp.fit(df.clone()).unwrap();
        let out = imp.transform(df).unwrap();
        assert_relative_eq!(vals(&out, "x")[1].unwrap(), 2.0);
        assert!(out.column("n").unwrap().i64().unwrap().get(1).is_none());
    }
}
