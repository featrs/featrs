//! Elapsed time from a reference timestamp.
//!
//! [`TimeSince`] computes, for each value in a `Date`/`Datetime` column, the
//! signed elapsed time from a reference instant and emits it as a `Float64`
//! column named `{column}_since_{unit}` (e.g. `signup_since_days`). It is the
//! companion to [`DatetimeFeatures`](crate::preprocessing::datetime_features::DatetimeFeatures),
//! turning absolute timestamps into a monotonically increasing numeric
//! feature.

use std::collections::HashSet;

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::TimeUnit as PolarsTimeUnit;
use polars::prelude::*;

/// Number of microseconds in one day (used to floor a fixed reference to the
/// containing day for `Date` columns).
const MICROS_PER_DAY_I64: i64 = 86_400_000_000;
/// Number of nanoseconds in one day, as a float for duration conversion.
const NANOS_PER_DAY: f64 = 86_400_000_000_000.0;

/// The unit in which elapsed time is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Elapsed time in seconds.
    Seconds,
    /// Elapsed time in minutes.
    Minutes,
    /// Elapsed time in hours.
    Hours,
    /// Elapsed time in days.
    Days,
    /// Elapsed time in weeks.
    Weeks,
}

impl TimeUnit {
    /// The suffix used in output column names, e.g. `"days"` for
    /// [`Days`](TimeUnit::Days) produces `{col}_since_days`.
    fn suffix(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "seconds",
            TimeUnit::Minutes => "minutes",
            TimeUnit::Hours => "hours",
            TimeUnit::Days => "days",
            TimeUnit::Weeks => "weeks",
        }
    }

    /// Convert a duration measured in days into this unit.
    ///
    /// [`Days`](TimeUnit::Days) returns the input unchanged; [`Weeks`](TimeUnit::Weeks)
    /// divides by `7.0` so whole-week durations stay exact.
    fn convert_days(self, delta_days: f64) -> f64 {
        match self {
            TimeUnit::Seconds => delta_days * 86_400.0,
            TimeUnit::Minutes => delta_days * 1_440.0,
            TimeUnit::Hours => delta_days * 24.0,
            TimeUnit::Days => delta_days,
            TimeUnit::Weeks => delta_days / 7.0,
        }
    }
}

/// The reference instant that elapsed time is measured from.
///
/// [`Fixed`](ReferenceTime::Fixed) is a timezone-naive absolute instant given
/// as **microseconds since the Unix epoch** (`1970-01-01T00:00:00Z`), i.e. the
/// same representation polars uses for a `Datetime(Microseconds, None)`
/// column. [`Min`](ReferenceTime::Min) and [`Max`](ReferenceTime::Max) learn
/// the per-column reference from the fit-time data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceTime {
    /// A user-supplied reference timestamp, as microseconds since the Unix epoch.
    Fixed(i64),
    /// Use each column's minimum value at fit time as its reference.
    Min,
    /// Use each column's maximum value at fit time as its reference.
    Max,
}

/// The per-column learned reference, in canonical physical units.
#[derive(Debug, Clone, PartialEq)]
enum ReferenceValue {
    /// Days since the Unix epoch (`Date` columns).
    Days(i64),
    /// Nanoseconds since the Unix epoch (`Datetime` columns). Nanoseconds are
    /// the finest polars time unit, so a learned reference round-trips exactly
    /// for `Nanoseconds`, `Microseconds`, and `Milliseconds` columns.
    Nanos(i64),
}

/// Compute the elapsed time from a reference timestamp for date/datetime columns.
///
/// Each configured column produces one new `Float64` column named
/// `{column}_since_{unit}` holding `(value - reference)` converted to
/// [`unit`](TimeSince::unit). The original columns are preserved and the new
/// columns are appended.
///
/// # Semantics and edge cases
///
/// - **Signed output** — elapsed time is `value - reference`, so values before
///   the reference produce negative durations and a value equal to the
///   reference produces `0.0`.
/// - **Reference** — [`ReferenceTime::Min`] (the default) and
///   [`ReferenceTime::Max`] learn one reference per column at fit time;
///   [`ReferenceTime::Fixed`] applies a single absolute instant (microseconds
///   since the Unix epoch) to every column.
/// - **Nulls** — a null input value yields null in the output column.
/// - **`Date` columns** carry whole-day resolution: a sub-day unit (`Hours`,
///   `Minutes`, `Seconds`) emits the whole-day multiple (e.g. `3` days →
///   `72.0` hours), never a time-of-day component. `Days`/`Weeks` are the
///   meaningful units for `Date` columns. A [`Fixed`](ReferenceTime::Fixed)
///   reference with a sub-day offset is floored to the containing day.
/// - **Timezone-aware datetimes** — elapsed time is computed from the
///   underlying epoch instants, so timezone metadata does not affect the
///   result and no timezone-mismatch error is raised. The
///   [`Fixed`](ReferenceTime::Fixed) reference is interpreted as the same
///   timezone-naive instant for every column.
/// - **Auto-discovery** — when no columns are configured, all `Date` and
///   `Datetime` columns present at fit time are discovered (on *every* fit).
/// - **Name collisions** — a generated name that matches an existing input
///   column would silently overwrite data in `with_column`, so both `fit` and
///   `transform` return [`Error::InvalidInput`] instead.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::time_since::{ReferenceTime, TimeSince, TimeUnit};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, DataType, NamedFrom, Series, TimeUnit as PolarsTimeUnit};
///
/// // 2024-01-01 and 2024-02-01, as microseconds since the Unix epoch.
/// let s = Series::new(
///     "t".into(),
///     &[Some(1_704_067_200_000_000i64), Some(1_704_067_200_000_000i64 + 31 * 86_400_000_000)],
/// )
/// .cast(&DataType::Datetime(PolarsTimeUnit::Microseconds, None))?;
/// let df = DataFrame::new(2, vec![Column::from(s)])?;
///
/// let mut xf = TimeSince::new()
///     .columns(&["t"])
///     .reference(ReferenceTime::Fixed(1_704_067_200_000_000))
///     .unit(TimeUnit::Days);
/// xf.fit(df.clone())?;
/// let out = xf.transform(df)?;
/// assert_eq!(out.column("t_since_days")?.f64()?.get(0), Some(0.0));
/// assert_eq!(out.column("t_since_days")?.f64()?.get(1), Some(31.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct TimeSince {
    fitted: bool,
    /// Resolved at fit time: either the explicit configuration from
    /// [`columns`](TimeSince::columns) or the auto-discovered `Date`/`Datetime`
    /// columns of the fitted frame.
    columns: Vec<String>,
    /// The user-supplied column configuration (`None` = auto-discover).
    column_config: Option<Vec<String>>,
    unit: TimeUnit,
    reference: ReferenceTime,
    /// Per-column learned reference, aligned with `columns`.
    learned_reference: Vec<(String, ReferenceValue)>,
}

impl TimeSince {
    /// Create a new `TimeSince` with auto-discovered columns, `Days` unit, and
    /// `Min` reference.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: vec![],
            column_config: None,
            unit: TimeUnit::Days,
            reference: ReferenceTime::Min,
            learned_reference: vec![],
        }
    }

    /// Restrict the transformation to the named columns.
    ///
    /// When omitted (or passed an empty list), all `Date`/`Datetime` columns
    /// present at fit time are auto-discovered on every fit. Each column must
    /// exist at fit time with dtype `Date` or `Datetime`; otherwise `fit`
    /// returns [`Error::InvalidInput`].
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.column_config = Some(cols.iter().map(|s| s.to_string()).collect());
        self.fitted = false;
        self
    }

    /// Set the unit in which elapsed time is expressed (default: [`TimeUnit::Days`]).
    pub fn unit(mut self, unit: TimeUnit) -> Self {
        self.unit = unit;
        self.fitted = false;
        self
    }

    /// Set the reference instant (default: [`ReferenceTime::Min`]).
    pub fn reference(mut self, reference: ReferenceTime) -> Self {
        self.reference = reference;
        self.fitted = false;
        self
    }
}

impl Default for TimeSince {
    fn default() -> Self {
        Self::new()
    }
}

/// True for dtypes this transformer operates on.
fn is_datetime_dtype(dt: &DataType) -> bool {
    matches!(dt, DataType::Date | DataType::Datetime(..))
}

/// Deduplicate while preserving first-seen order.
fn dedup_preserve_order<T: Clone + Eq + std::hash::Hash>(items: &[T]) -> Vec<T> {
    let mut seen = HashSet::new();
    items.iter().filter(|i| seen.insert(*i)).cloned().collect()
}

/// Nanoseconds in one native time unit of a `Datetime` column.
fn nanos_per_native_unit(tu: PolarsTimeUnit) -> f64 {
    match tu {
        PolarsTimeUnit::Nanoseconds => 1.0,
        PolarsTimeUnit::Microseconds => 1_000.0,
        PolarsTimeUnit::Milliseconds => 1_000_000.0,
    }
}

/// Convert a native `Datetime` value to nanoseconds since the Unix epoch.
fn native_to_nanos(native: i64, tu: PolarsTimeUnit) -> i64 {
    match tu {
        PolarsTimeUnit::Nanoseconds => native,
        PolarsTimeUnit::Microseconds => native * 1_000,
        PolarsTimeUnit::Milliseconds => native * 1_000_000,
    }
}

/// Convert nanoseconds since the Unix epoch to a native `Datetime` value.
fn nanos_to_native(nanos: i64, tu: PolarsTimeUnit) -> i64 {
    match tu {
        PolarsTimeUnit::Nanoseconds => nanos,
        PolarsTimeUnit::Microseconds => nanos.div_euclid(1_000),
        PolarsTimeUnit::Milliseconds => nanos.div_euclid(1_000_000),
    }
}

/// Learn the reference value for one column, in canonical physical units.
fn learn_reference(s: &Series, reference: ReferenceTime, col: &str) -> Result<ReferenceValue> {
    match s.dtype() {
        DataType::Date => {
            let days = match reference {
                ReferenceTime::Fixed(epoch_us) => epoch_us.div_euclid(MICROS_PER_DAY_I64),
                ReferenceTime::Min => {
                    let v = s.to_physical_repr().min::<i32>().map_err(|e| {
                        Error::Computation(format!("TimeSince.fit: column '{col}': {e}"))
                    })?;
                    v.ok_or_else(|| all_null_error(col, "Min"))? as i64
                }
                ReferenceTime::Max => {
                    let v = s.to_physical_repr().max::<i32>().map_err(|e| {
                        Error::Computation(format!("TimeSince.fit: column '{col}': {e}"))
                    })?;
                    v.ok_or_else(|| all_null_error(col, "Max"))? as i64
                }
            };
            Ok(ReferenceValue::Days(days))
        }
        DataType::Datetime(tu, _) => {
            let nanos = match reference {
                ReferenceTime::Fixed(epoch_us) => epoch_us * 1_000,
                ReferenceTime::Min => {
                    let v = s.to_physical_repr().min::<i64>().map_err(|e| {
                        Error::Computation(format!("TimeSince.fit: column '{col}': {e}"))
                    })?;
                    native_to_nanos(v.ok_or_else(|| all_null_error(col, "Min"))?, *tu)
                }
                ReferenceTime::Max => {
                    let v = s.to_physical_repr().max::<i64>().map_err(|e| {
                        Error::Computation(format!("TimeSince.fit: column '{col}': {e}"))
                    })?;
                    native_to_nanos(v.ok_or_else(|| all_null_error(col, "Max"))?, *tu)
                }
            };
            Ok(ReferenceValue::Nanos(nanos))
        }
        other => Err(Error::InvalidInput(format!(
            "TimeSince.fit: column '{col}' has dtype {other}; expected Date or Datetime."
        ))),
    }
}

/// Error for an all-null column when a `Min`/`Max` reference is requested.
fn all_null_error(col: &str, which: &str) -> Error {
    Error::InvalidInput(format!(
        "TimeSince.fit: column '{col}' is all-null; cannot compute a {which} reference. \
         Use ReferenceTime::Fixed or impute/drop the column."
    ))
}

/// Compute the elapsed-time `Float64` series for one column.
fn elapsed_series(
    s: &Series,
    reference: &ReferenceValue,
    unit: TimeUnit,
    col: &str,
) -> Result<Series> {
    let c_err =
        |e: PolarsError| Error::Computation(format!("TimeSince.transform: column '{col}': {e}"));

    let result: ChunkedArray<Float64Type> = match (s.dtype(), reference) {
        (DataType::Date, ReferenceValue::Days(days)) => {
            let ca = s.to_physical_repr().i32().map_err(c_err)?.clone();
            ca.iter()
                .map(|opt| opt.map(|v| unit.convert_days((v as i64 - *days) as f64)))
                .collect()
        }
        (DataType::Datetime(tu, _), ReferenceValue::Nanos(nanos)) => {
            let ca = s.to_physical_repr().i64().map_err(c_err)?.clone();
            let ref_native = nanos_to_native(*nanos, *tu);
            let nanos_per_native = nanos_per_native_unit(*tu);
            ca.iter()
                .map(|opt| {
                    opt.map(|v| {
                        // i128 keeps the subtraction exact even for extreme
                        // i64 timestamps near their range limits.
                        let delta_nanos =
                            (i128::from(v) - i128::from(ref_native)) as f64 * nanos_per_native;
                        unit.convert_days(delta_nanos / NANOS_PER_DAY)
                    })
                })
                .collect()
        }
        (other, _) => {
            return Err(Error::InvalidInput(format!(
                "TimeSince.transform: column '{col}' has dtype {other}; expected the Date or \
                 Datetime dtype it was fitted with. The reference was learned on a different \
                 dtype, so re-fit on this data before transforming."
            )));
        }
    };
    Ok(result.into_series())
}

impl Fit<DataFrame> for TimeSince {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave stale fitted state, and
        // drop previously resolved columns so auto-discovery re-runs on every
        // fit instead of reusing the previous schema's columns.
        self.fitted = false;
        self.columns = vec![];
        self.learned_reference = vec![];

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "TimeSince.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide at least one row and one column."
                    .into(),
            ));
        }

        match self.column_config.as_deref() {
            // Auto-discovery: no explicit configuration (or an empty list).
            None | Some([]) => {
                let discovered: Vec<String> = x
                    .get_column_names()
                    .iter()
                    .filter(|n| {
                        x.column(n)
                            .map(|c| is_datetime_dtype(c.dtype()))
                            .unwrap_or(false)
                    })
                    .map(|n| n.to_string())
                    .collect();
                if discovered.is_empty() {
                    let all_types: Vec<String> = x
                        .get_column_names()
                        .iter()
                        .filter_map(|n| x.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
                        .collect();
                    return Err(Error::InvalidInput(format!(
                        "TimeSince: no Date or Datetime columns found. This transformer only \
                         operates on Date/Datetime columns. Available columns: [{}]. Cast \
                         non-date columns before fitting.",
                        all_types.join(", ")
                    )));
                }
                self.columns = discovered;
            }
            Some(cfg) => {
                for col in cfg {
                    let c = x.column(col.as_str()).map_err(|e| {
                        Error::InvalidInput(format!("TimeSince.fit: column '{col}' not found. {e}"))
                    })?;
                    if !is_datetime_dtype(c.dtype()) {
                        return Err(Error::InvalidInput(format!(
                            "TimeSince.fit: column '{col}' has dtype {}; expected Date or Datetime.",
                            c.dtype()
                        )));
                    }
                }
                self.columns = dedup_preserve_order(cfg);
            }
        }

        let mut learned = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let s = x
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!("TimeSince.fit: column '{col}' not found. {e}"))
                })?
                .as_materialized_series();
            let reference = learn_reference(s, self.reference, col)?;
            learned.push((col.clone(), reference));
        }

        // Reject name collisions up front: `with_column` silently replaces a
        // same-named column, so a generated name matching an input column (or
        // another generated name) would silently overwrite data.
        let mut seen: HashSet<String> = x
            .get_column_names()
            .iter()
            .map(|n| n.as_str().to_string())
            .collect();
        for col in &self.columns {
            let out_name = format!("{col}_since_{}", self.unit.suffix());
            if !seen.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "TimeSince: generated column '{out_name}' collides with an existing input \
                     column or another generated column. Rename the conflicting input column \
                     or choose a different unit."
                )));
            }
        }

        self.learned_reference = learned;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for TimeSince {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "TimeSince has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();

        // The transform input may contain columns absent at fit time; guard
        // against silently overwriting them, mirroring the fit-time check.
        for (col, _) in &self.learned_reference {
            let out_name = format!("{col}_since_{}", self.unit.suffix());
            if out.column(out_name.as_str()).is_ok() {
                return Err(Error::InvalidInput(format!(
                    "TimeSince.transform: input already contains column '{out_name}', which \
                     would be overwritten by a generated feature. Rename the conflicting \
                     column or choose a different unit."
                )));
            }
        }

        for (col, reference) in &self.learned_reference {
            let s = out
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "TimeSince.transform: column '{col}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.columns
                    ))
                })?
                .as_materialized_series()
                .clone();
            let elapsed = elapsed_series(&s, reference, self.unit, col)?;
            let out_name = format!("{col}_since_{}", self.unit.suffix());
            out.with_column(elapsed.with_name(out_name.as_str().into()).into())
                .map_err(|e| Error::Computation(format!("TimeSince.transform: {e}")))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::TimeUnit as PolarsTimeUnit;

    /// Days since the Unix epoch of 2023-01-01 (a non-leap-year reference).
    const DAY0: i32 = 19_358;
    /// Microseconds since the Unix epoch of 2024-01-01T00:00:00Z.
    const T0: i64 = 1_704_067_200_000_000;
    /// Microseconds in one day.
    const DAY_US: i64 = 86_400_000_000;

    fn date_col(name: &str, vals: &[Option<i32>]) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Date)
            .unwrap()
            .into()
    }

    fn datetime_col(name: &str, vals: &[Option<i64>]) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Datetime(PolarsTimeUnit::Microseconds, None))
            .unwrap()
            .into()
    }

    #[test]
    fn test_min_reference_days() {
        // Jan 1, Feb 1, Apr 1 (2023, non-leap) with the default Min reference.
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 31), Some(DAY0 + 90)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("d_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(0.0));
        assert_eq!(days.get(1), Some(31.0));
        assert_eq!(days.get(2), Some(90.0));
    }

    #[test]
    fn test_max_reference_days() {
        // Max reference = Apr 1, so Jan 1 is 90 days BEFORE it: signed output.
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 31), Some(DAY0 + 90)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["d"])
            .reference(ReferenceTime::Max);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("d_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(-90.0));
        assert_eq!(days.get(1), Some(-59.0));
        assert_eq!(days.get(2), Some(0.0));
    }

    #[test]
    fn test_fixed_reference_signed() {
        // Fixed at 2024-02-01; Jan 1 is before, Apr 1 (leap-year offset) after.
        let col = datetime_col(
            "t",
            &[Some(T0), Some(T0 + 31 * DAY_US), Some(T0 + 91 * DAY_US)],
        );
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Fixed(T0 + 31 * DAY_US))
            .unit(TimeUnit::Days);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("t_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(-31.0));
        assert_eq!(days.get(1), Some(0.0));
        assert_eq!(days.get(2), Some(60.0));
    }

    #[test]
    fn test_unit_conversion_factor() {
        // One day apart: Days -> 1.0, Hours -> 24.0.
        let col = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut days_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Fixed(T0))
            .unit(TimeUnit::Days);
        days_xf.fit(df.clone()).unwrap();
        let days = days_xf.transform(df.clone()).unwrap();
        assert_eq!(
            days.column("t_since_days").unwrap().f64().unwrap().get(1),
            Some(1.0)
        );

        let mut hours_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Fixed(T0))
            .unit(TimeUnit::Hours);
        hours_xf.fit(df.clone()).unwrap();
        let hours = hours_xf.transform(df).unwrap();
        assert_eq!(
            hours.column("t_since_hours").unwrap().f64().unwrap().get(1),
            Some(24.0)
        );
    }

    #[test]
    fn test_datetime_min_max_reference() {
        // 2024 (leap year): Jan 1, Feb 1, Apr 1.
        let col = datetime_col(
            "t",
            &[Some(T0), Some(T0 + 31 * DAY_US), Some(T0 + 91 * DAY_US)],
        );
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut min_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Min);
        min_xf.fit(df.clone()).unwrap();
        let min_out = min_xf.transform(df.clone()).unwrap();
        let min_days = min_out.column("t_since_days").unwrap().f64().unwrap();
        assert_eq!(min_days.get(0), Some(0.0));
        assert_eq!(min_days.get(1), Some(31.0));
        assert_eq!(min_days.get(2), Some(91.0));

        let mut max_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Max);
        max_xf.fit(df.clone()).unwrap();
        let max_out = max_xf.transform(df).unwrap();
        let max_days = max_out.column("t_since_days").unwrap().f64().unwrap();
        assert_eq!(max_days.get(0), Some(-91.0));
        assert_eq!(max_days.get(1), Some(-60.0));
        assert_eq!(max_days.get(2), Some(0.0));
    }

    #[test]
    fn test_milliseconds_timeunit() {
        // 2024-01-01 and 2024-02-01 in milliseconds.
        let t0_ms = T0 / 1_000;
        let col = Column::from(
            Series::new("t".into(), &[Some(t0_ms), Some(t0_ms + 31 * 86_400_000)])
                .cast(&DataType::Datetime(PolarsTimeUnit::Milliseconds, None))
                .unwrap(),
        );
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Fixed(T0))
            .unit(TimeUnit::Days);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("t_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(0.0));
        assert_eq!(days.get(1), Some(31.0));
    }

    #[test]
    fn test_remaining_unit_conversions() {
        // Whole-week boundaries for exact Weeks output.
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 7), Some(DAY0 + 14)]);
        let df = DataFrame::new(3, vec![col]).unwrap();
        let fixed = ReferenceTime::Fixed(DAY0 as i64 * MICROS_PER_DAY_I64);

        let mut weeks_xf = TimeSince::new()
            .columns(&["d"])
            .reference(fixed)
            .unit(TimeUnit::Weeks);
        weeks_xf.fit(df.clone()).unwrap();
        let weeks = weeks_xf.transform(df.clone()).unwrap();
        let w = weeks.column("d_since_weeks").unwrap().f64().unwrap();
        assert_eq!(w.get(0), Some(0.0));
        assert_eq!(w.get(1), Some(1.0));
        assert_eq!(w.get(2), Some(2.0));

        // 1 day -> 1440 minutes and 86_400 seconds.
        let col1 = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let df1 = DataFrame::new(2, vec![col1]).unwrap();

        let mut mins_xf = TimeSince::new()
            .columns(&["d"])
            .reference(fixed)
            .unit(TimeUnit::Minutes);
        mins_xf.fit(df1.clone()).unwrap();
        let mins = mins_xf.transform(df1.clone()).unwrap();
        assert_eq!(
            mins.column("d_since_minutes")
                .unwrap()
                .f64()
                .unwrap()
                .get(1),
            Some(1440.0)
        );

        let mut secs_xf = TimeSince::new()
            .columns(&["d"])
            .reference(fixed)
            .unit(TimeUnit::Seconds);
        secs_xf.fit(df1.clone()).unwrap();
        let secs = secs_xf.transform(df1).unwrap();
        assert_eq!(
            secs.column("d_since_seconds")
                .unwrap()
                .f64()
                .unwrap()
                .get(1),
            Some(86_400.0)
        );
    }

    #[test]
    fn test_null_preservation() {
        let col = date_col("d", &[Some(DAY0), None, Some(DAY0 + 31)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("d_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(0.0));
        assert!(days.get(1).is_none());
        assert_eq!(days.get(2), Some(31.0));
    }

    #[test]
    fn test_date_column_with_hours_is_whole_days() {
        // A Date column carries no time component, so sub-day units emit the
        // whole-day multiple (1 day -> 24 hours), not a zero time-of-day.
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["d"])
            .reference(ReferenceTime::Fixed(DAY0 as i64 * MICROS_PER_DAY_I64))
            .unit(TimeUnit::Hours);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let hours = out.column("d_since_hours").unwrap().f64().unwrap();
        assert_eq!(hours.get(0), Some(0.0));
        assert_eq!(hours.get(1), Some(24.0));
    }

    #[test]
    fn test_nanosecond_timeunit() {
        // Same instants as the microsecond tests, in nanoseconds.
        let t0_ns = T0 * 1_000;
        let col = Column::from(
            Series::new(
                "t".into(),
                &[Some(t0_ns), Some(t0_ns + 31 * DAY_US * 1_000)],
            )
            .cast(&DataType::Datetime(PolarsTimeUnit::Nanoseconds, None))
            .unwrap(),
        );
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Fixed(T0))
            .unit(TimeUnit::Days);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("t_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(0.0));
        assert_eq!(days.get(1), Some(31.0));
    }

    #[test]
    fn test_auto_discovery_only_datetime_columns() {
        let t = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![t, f]).unwrap();

        let mut xf = TimeSince::new();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        assert!(out.column("t_since_days").is_ok());
        assert!(out.column("x_since_days").is_err());
    }

    #[test]
    fn test_duplicate_config_deduped() {
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 31)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = TimeSince::new().columns(&["d", "d"]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // one input column + one deduped d_since_days
        assert_eq!(out.width(), 2);
        assert!(out.column("d_since_days").is_ok());
    }

    #[test]
    fn test_not_fitted_error() {
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let df = DataFrame::new(2, vec![col]).unwrap();
        let xf = TimeSince::new().columns(&["d"]);
        let err = xf.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_empty_dataframe_rejected() {
        let mut xf = TimeSince::new();
        let err = xf.fit(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_missing_column_errors() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut xf = TimeSince::new().columns(&["nope"]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_non_datetime_column_errors() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut xf = TimeSince::new().columns(&["x"]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_all_null_column_min_reference_errors() {
        let col = date_col("d", &[None, None]);
        let df = DataFrame::new(2, vec![col]).unwrap();
        let mut xf = TimeSince::new().columns(&["d"]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_fixed_reference_with_nulls_ok() {
        let col = date_col("d", &[Some(DAY0), None, Some(DAY0 + 31)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = TimeSince::new()
            .columns(&["d"])
            .reference(ReferenceTime::Fixed(DAY0 as i64 * MICROS_PER_DAY_I64));
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let days = out.column("d_since_days").unwrap().f64().unwrap();
        assert_eq!(days.get(0), Some(0.0));
        assert!(days.get(1).is_none());
        assert_eq!(days.get(2), Some(31.0));
    }

    #[test]
    fn test_name_collision_at_fit() {
        // "d_since_days" is a real input column; generating it from "d" would
        // silently overwrite it.
        let d = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let d_since_days = date_col("d_since_days", &[Some(DAY0), Some(DAY0 + 1)]);
        let df = DataFrame::new(2, vec![d, d_since_days]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        let err = xf.fit(df).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("d_since_days"), "got: {msg}"),
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_input_collision_rejected() {
        let d = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let fit_df = DataFrame::new(2, vec![d]).unwrap();

        let d2 = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let d_since_days = date_col("d_since_days", &[Some(DAY0 + 5), Some(DAY0 + 6)]);
        let transform_df = DataFrame::new(2, vec![d2, d_since_days]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        xf.fit(fit_df).unwrap();
        let err = xf.transform(transform_df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_default_equals_new() {
        let d = TimeSince::default();
        let n = TimeSince::new();
        assert_eq!(d.unit, n.unit);
        assert_eq!(d.reference, n.reference);
        assert_eq!(d.fitted, n.fitted);
        assert!(d.columns.is_empty());
        assert!(d.learned_reference.is_empty());
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let d = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let df1 = DataFrame::new(2, vec![d]).unwrap();
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df2 = DataFrame::new(2, vec![f]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        xf.fit(df1).unwrap();
        // re-fit on data missing the configured column must fail...
        assert!(xf.fit(df2).is_err());
        // ...and must NOT leave the transformer in a fitted state
        let err = xf.transform(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_nanosecond_reference_preserves_precision() {
        // Sub-microsecond values: the learned Min/Max reference must round-trip
        // exactly so the extreme row yields exactly 0.0 elapsed (no truncation
        // to whole microseconds).
        let col = Column::from(
            Series::new("t".into(), &[Some(1_001i64), Some(2_002i64)])
                .cast(&DataType::Datetime(PolarsTimeUnit::Nanoseconds, None))
                .unwrap(),
        );
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut min_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Min)
            .unit(TimeUnit::Days);
        min_xf.fit(df.clone()).unwrap();
        let min_days = min_xf
            .transform(df.clone())
            .unwrap()
            .column("t_since_days")
            .unwrap()
            .f64()
            .unwrap()
            .clone();
        assert_eq!(min_days.get(0), Some(0.0));

        let mut max_xf = TimeSince::new()
            .columns(&["t"])
            .reference(ReferenceTime::Max)
            .unit(TimeUnit::Days);
        max_xf.fit(df.clone()).unwrap();
        let max_days = max_xf
            .transform(df)
            .unwrap()
            .column("t_since_days")
            .unwrap()
            .f64()
            .unwrap()
            .clone();
        assert_eq!(max_days.get(1), Some(0.0));
    }

    #[test]
    fn test_setter_after_fit_invalidates() {
        let col = date_col("d", &[Some(DAY0), Some(DAY0 + 1)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = TimeSince::new().columns(&["d"]);
        xf.fit(df.clone()).unwrap();
        // Reconfiguring after fit must invalidate fitted state so transform
        // cannot silently apply the previous configuration.
        xf = xf.columns(&["d"]);
        assert!(matches!(
            xf.transform(df.clone()).unwrap_err(),
            Error::NotFitted(_)
        ));

        xf.fit(df.clone()).unwrap();
        xf = xf.unit(TimeUnit::Hours);
        assert!(matches!(
            xf.transform(df.clone()).unwrap_err(),
            Error::NotFitted(_)
        ));

        xf.fit(df.clone()).unwrap();
        xf = xf.reference(ReferenceTime::Max);
        assert!(matches!(xf.transform(df).unwrap_err(), Error::NotFitted(_)));
    }
}
