//! Datetime feature extraction.
//!
//! [`DatetimeFeatures`] extracts calendar components (year, month, weekday,
//! hour, quarter, …) from `Date` and `Datetime` columns, appending one new
//! numeric column per requested component, e.g. `date_month`,
//! `timestamp_hour`, `timestamp_is_weekend`. The original columns are
//! preserved unchanged.

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_fitted, with_generated_names};
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashSet;
use std::ops::Not;

/// A calendar component that can be extracted from a `Date`/`Datetime` column.
///
/// [`DayOfYear`](DatetimeComponent::DayOfYear) follows the ISO calendar
/// (`1..=366`) and [`WeekOfYear`](DatetimeComponent::WeekOfYear) the ISO
/// week-number calendar (`1..=53`). [`Weekday`](DatetimeComponent::Weekday)
/// is reported with `0 = Monday … 6 = Sunday` (contrast polars' ISO `weekday`,
/// which is `1 = Monday … 7 = Sunday`). [`IsWeekend`](DatetimeComponent::IsWeekend)
/// is `1.0` for Saturday/Sunday and `0.0` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatetimeComponent {
    /// Calendar year (e.g. `2024`).
    Year,
    /// Month number, `1..=12`.
    Month,
    /// Day of month, `1..=31`.
    Day,
    /// Hour of day, `0..=23`.
    Hour,
    /// Minute of hour, `0..=59`.
    Minute,
    /// Second of minute, `0..=59`.
    Second,
    /// Weekday with `0 = Monday` and `6 = Sunday`.
    Weekday,
    /// Day of year, `1..=366`.
    DayOfYear,
    /// ISO week of year, `1..=53`.
    WeekOfYear,
    /// Quarter of year, `1..=4`.
    Quarter,
    /// `1.0` if the date falls on a weekend (Saturday/Sunday), else `0.0`.
    IsWeekend,
}

impl DatetimeComponent {
    /// The suffix used in output column names, e.g. `"month"` for
    /// [`Month`](DatetimeComponent::Month) produces `{col}_month`.
    fn suffix(self) -> &'static str {
        match self {
            DatetimeComponent::Year => "year",
            DatetimeComponent::Month => "month",
            DatetimeComponent::Day => "day",
            DatetimeComponent::Hour => "hour",
            DatetimeComponent::Minute => "minute",
            DatetimeComponent::Second => "second",
            DatetimeComponent::Weekday => "weekday",
            DatetimeComponent::DayOfYear => "day_of_year",
            DatetimeComponent::WeekOfYear => "week_of_year",
            DatetimeComponent::Quarter => "quarter",
            DatetimeComponent::IsWeekend => "is_weekend",
        }
    }
}

/// Extract calendar components from `Date`/`Datetime` columns.
///
/// Each configured `(column, component)` pair produces one new column named
/// `{column}_{component}` (e.g. `date_month`, `timestamp_is_weekend`).
/// Integer-valued components (`Year`, `Month`, …) are emitted as `Int32`;
/// [`DatetimeComponent::IsWeekend`] is emitted as `Float64` (`0.0`/`1.0`) for
/// consistency with the downstream `Float64`-based transformers.
///
/// Semantics and edge cases:
///
/// - **Auto-discovery** — when no columns are configured (or the list is
///   empty), all `Date` and `Datetime` columns present at fit time are
///   discovered — on *every* fit, so re-fitting on a changed schema
///   re-discovers instead of reusing the previous schema's columns.
/// - **Default components** — when no components are configured, the sensible
///   default set `[Year, Month, Day]` is used.
/// - **`Date` columns carry no time** — [`DatetimeComponent::Hour`],
///   [`DatetimeComponent::Minute`], and [`DatetimeComponent::Second`] on a
///   `Date` column are zero-filled (`0` for every non-null value) rather than
///   erroring; only `Datetime` columns produce real time-of-day values.
/// - **Nulls** are preserved: a null input value yields null in every
///   extracted component column.
/// - **Weekday** is 0-based (`0 = Monday … 6 = Sunday`), following the issue
///   spec; [`DatetimeComponent::IsWeekend`] is derived from it.
/// - **Output ordering** — generated columns are appended grouped by input
///   column (in the fitted column order), each input column followed by its
///   components in configured order.
/// - **Name collisions** are rejected at fit time: a generated name that
///   matches an existing input column would silently overwrite data in
///   `with_column`, so `fit` returns [`Error::InvalidInput`] instead.
///   `transform` applies the same guard against columns that appear only in
///   the transform input.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::datetime_features::{DatetimeComponent, DatetimeFeatures};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, DataType, NamedFrom, Series, TimeUnit};
///
/// let s = Series::new(
///     "t".into(),
///     &[
///         Some(1_704_067_200_000_000i64), // 2024-01-01T00:00:00Z
///         Some(1_704_067_200_000_000i64 + 31 * 86_400_000_000), // 2024-02-01
///     ],
/// )
/// .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
/// let df = DataFrame::new(2, vec![Column::from(s)])?;
///
/// let mut xf = DatetimeFeatures::new()
///     .columns(&["t"])
///     .components(&[DatetimeComponent::Year, DatetimeComponent::Month]);
/// xf.fit(df.clone())?;
/// let out = xf.transform(df)?;
/// assert_eq!(out.width(), 3); // t, t_year, t_month
/// assert_eq!(out.column("t_year")?.i32()?.get(0), Some(2024));
/// assert_eq!(out.column("t_month")?.i32()?.get(1), Some(2));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DatetimeFeatures {
    fitted: bool,
    /// Resolved at fit time: either the explicit configuration from
    /// [`columns`](DatetimeFeatures::columns) or the auto-discovered
    /// `Date`/`Datetime` columns of the fitted frame.
    columns: Vec<String>,
    /// The user-supplied column configuration (`None` = auto-discover).
    column_config: Option<Vec<String>>,
    components: Vec<DatetimeComponent>,
}

impl DatetimeFeatures {
    /// Create a new `DatetimeFeatures` with no columns and no components
    /// configured.
    ///
    /// Both are resolved at fit time: columns default to auto-discovery of all
    /// `Date`/`Datetime` columns, and components default to `[Year, Month, Day]`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: vec![],
            column_config: None,
            components: vec![],
        }
    }

    /// Restrict extraction to the named columns.
    ///
    /// When omitted (or passed an empty list), all `Date`/`Datetime` columns
    /// present at fit time are auto-discovered on every fit. Each column must
    /// exist at fit time with dtype `Date` or `Datetime`; otherwise `fit`
    /// returns [`Error::InvalidInput`].
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.column_config = Some(cols.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set the components to extract.
    ///
    /// When omitted, the default set `[Year, Month, Day]` is used at fit time.
    pub fn components(mut self, comps: &[DatetimeComponent]) -> Self {
        self.components = comps.to_vec();
        self
    }
}

impl Default for DatetimeFeatures {
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

impl Fit<DataFrame> for DatetimeFeatures {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave stale fitted state, and
        // drop previously resolved columns so auto-discovery re-runs on every
        // fit instead of reusing the previous schema's columns.
        self.fitted = false;
        self.columns = vec![];

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "DatetimeFeatures.fit received an empty DataFrame (0 rows or 0 columns). \
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
                        "DatetimeFeatures: no Date or Datetime columns found. This transformer only \
                         operates on Date/Datetime columns. Available columns: [{}]. Cast non-date \
                         columns before fitting.",
                        all_types.join(", ")
                    )));
                }
                self.columns = discovered;
            }
            Some(cfg) => {
                for col in cfg {
                    let c = x.column(col.as_str()).map_err(|e| {
                        Error::InvalidInput(format!(
                            "DatetimeFeatures.fit: column '{col}' not found. {e}"
                        ))
                    })?;
                    if !is_datetime_dtype(c.dtype()) {
                        return Err(Error::InvalidInput(format!(
                            "DatetimeFeatures.fit: column '{col}' has dtype {}; expected Date or Datetime.",
                            c.dtype()
                        )));
                    }
                }
                self.columns = dedup_preserve_order(cfg);
            }
        }

        if self.components.is_empty() {
            self.components = vec![
                DatetimeComponent::Year,
                DatetimeComponent::Month,
                DatetimeComponent::Day,
            ];
        } else {
            self.components = dedup_preserve_order(&self.components);
        }

        // Reject name collisions up front: `with_column` silently replaces a
        // same-named column, so a generated name matching an input column (or
        // another generated name) would silently overwrite data.
        let mut seen: HashSet<String> = x
            .get_column_names()
            .iter()
            .map(|n| n.as_str().to_string())
            .collect();
        for (_, out_name, _) in self.planned() {
            if !seen.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "DatetimeFeatures: generated column '{out_name}' collides with an existing \
                     input column or another generated column. Rename the conflicting input \
                     column or drop the conflicting component."
                )));
            }
        }

        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for DatetimeFeatures {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "DatetimeFeatures has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();

        // The transform input may contain columns absent at fit time; guard
        // against silently overwriting them, mirroring the fit-time check.
        for (_, out_name, _) in self.planned() {
            if out.column(out_name.as_str()).is_ok() {
                return Err(Error::InvalidInput(format!(
                    "DatetimeFeatures.transform: input already contains column '{out_name}', \
                     which would be overwritten by a generated feature. Rename the conflicting \
                     column or drop the conflicting component."
                )));
            }
        }

        for col in &self.columns {
            let s = out
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "DatetimeFeatures.transform: column '{col}' not found. The transformer \
                         was fitted on columns: {:?}. {e}",
                        self.columns
                    ))
                })?
                .as_materialized_series()
                .clone();

            for comp in &self.components {
                let result = extract_component(&s, *comp, col)?;
                let out_name = format!("{col}_{}", comp.suffix());
                out.with_column(result.with_name(out_name.as_str().into()).into())
                    .map_err(|e| Error::Computation(format!("DatetimeFeatures.transform: {e}")))?;
            }
        }

        Ok(out)
    }
}

impl DatetimeFeatures {
    /// The resolved `(column, output_name, component)` triples, in output order.
    fn planned(&self) -> Vec<(String, String, DatetimeComponent)> {
        self.columns
            .iter()
            .flat_map(|col| {
                self.components
                    .iter()
                    .map(move |comp| (col.clone(), format!("{col}_{}", comp.suffix()), *comp))
            })
            .collect()
    }
}

/// Zero-filled `Int32` with the same length and null pattern as `s`.
///
/// Used for time-of-day components (`Hour`, `Minute`, `Second`) on `Date`
/// columns, which have no time component: every non-null value is `0`, nulls
/// stay null.
///
/// The result is built from the validity mask, so the input series' values
/// are never copied (rechunking it just to read nulls would be an unnecessary
/// full copy).
fn zero_fill_i32(s: &Series) -> Result<Series> {
    let zeros = Int32Chunked::from_iter(std::iter::repeat_n(Some(0i32), s.len())).into_series();
    let not_null = s.is_null().not();
    zeros
        .zip_with(
            &not_null,
            &Series::full_null(s.name().clone(), s.len(), &DataType::Int32),
        )
        .map_err(|e| Error::Computation(format!("DatetimeFeatures.transform: {e}")))
}

/// Convert an `Int8` chunked array to an `Int32` series by element-wise
/// iteration.
///
/// polars' temporal accessors return `Int8`/`Int16` chunked arrays for most
/// components, but this crate compiles with polars' default features, which do
/// NOT enable `dtype-i8`/`dtype-i16` — casting such an array (`to_physical`)
/// panics with "not implemented". Iterating the logical values avoids that
/// feature-gated path entirely.
fn i8_to_i32(ca: &Int8Chunked) -> Series {
    let out: Int32Chunked = ca.iter().map(|o| o.map(|v| v as i32)).collect();
    out.into_series()
}

/// Convert an `Int16` chunked array to an `Int32` series (see [`i8_to_i32`]).
fn i16_to_i32(ca: &Int16Chunked) -> Series {
    let out: Int32Chunked = ca.iter().map(|o| o.map(|v| v as i32)).collect();
    out.into_series()
}

/// Extract a single component from a `Date`/`Datetime` series.
fn extract_component(s: &Series, comp: DatetimeComponent, col: &str) -> Result<Series> {
    let c_err = |e: PolarsError| {
        Error::Computation(format!("DatetimeFeatures.transform: column '{col}': {e}"))
    };

    let result = match comp {
        DatetimeComponent::Year => s.year().map_err(c_err)?.into_series(),
        DatetimeComponent::Month => i8_to_i32(&s.month().map_err(c_err)?),
        DatetimeComponent::Day => i8_to_i32(&s.day().map_err(c_err)?),
        DatetimeComponent::Hour => match s.dtype() {
            DataType::Date => zero_fill_i32(s)?,
            _ => i8_to_i32(&s.hour().map_err(c_err)?),
        },
        DatetimeComponent::Minute => match s.dtype() {
            DataType::Date => zero_fill_i32(s)?,
            _ => i8_to_i32(&s.minute().map_err(c_err)?),
        },
        DatetimeComponent::Second => match s.dtype() {
            DataType::Date => zero_fill_i32(s)?,
            _ => i8_to_i32(&s.second().map_err(c_err)?),
        },
        DatetimeComponent::Weekday => {
            // polars weekday() is ISO 1..=7 (Monday=1); shift to 0..=6 per spec.
            let wd = s.weekday().map_err(c_err)?;
            let shifted: Int32Chunked = wd.iter().map(|o| o.map(|v| (v - 1) as i32)).collect();
            shifted.into_series()
        }
        DatetimeComponent::DayOfYear => i16_to_i32(&s.ordinal_day().map_err(c_err)?),
        DatetimeComponent::WeekOfYear => i8_to_i32(&s.week().map_err(c_err)?),
        DatetimeComponent::Quarter => i8_to_i32(&s.quarter().map_err(c_err)?),
        DatetimeComponent::IsWeekend => {
            // polars weekday(): ISO 6=Saturday, 7=Sunday.
            let flags: ChunkedArray<Float64Type> = s
                .weekday()
                .map_err(c_err)?
                .iter()
                .map(|o| o.map(|v| if v >= 6 { 1.0 } else { 0.0 }))
                .collect();
            flags.into_series()
        }
    };
    Ok(result)
}

impl FeatureNamesOut for DatetimeFeatures {
    /// The input columns plus `{column}_{component}` for every planned pair.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        ensure_fitted(self.fitted, "DatetimeFeatures")?;
        let generated = self.planned().into_iter().map(|(_, name, _)| name);
        Ok(with_generated_names(input_columns, generated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::TimeUnit;

    /// Epoch microseconds of 2024-01-01T00:00:00Z (a Monday).
    const T0: i64 = 1_704_067_200_000_000;
    const DAY_US: i64 = 86_400_000_000;
    /// Days since epoch of 2024-01-01 (a Monday).
    const D0: i32 = 19_723;

    fn datetime_col(name: &str, vals: &[Option<i64>]) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap()
            .into()
    }

    fn date_col(name: &str, vals: &[Option<i32>]) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Date)
            .unwrap()
            .into()
    }

    #[test]
    fn test_year_month_day_from_datetime() {
        // 2024-01-01, 2024-02-01, 2024-04-01
        let col = datetime_col(
            "t",
            &[Some(T0), Some(T0 + 31 * DAY_US), Some(T0 + 91 * DAY_US)],
        );
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["t"]).components(&[
            DatetimeComponent::Year,
            DatetimeComponent::Month,
            DatetimeComponent::Day,
        ]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        assert_eq!(out.width(), 4);
        let year = out.column("t_year").unwrap().i32().unwrap();
        assert_eq!(year.get(0), Some(2024));
        assert_eq!(year.get(1), Some(2024));
        assert_eq!(year.get(2), Some(2024));
        let month = out.column("t_month").unwrap().i32().unwrap();
        assert_eq!(month.get(0), Some(1));
        assert_eq!(month.get(1), Some(2));
        assert_eq!(month.get(2), Some(4));
        let day = out.column("t_day").unwrap().i32().unwrap();
        assert_eq!(day.get(0), Some(1));
        assert_eq!(day.get(1), Some(1));
        assert_eq!(day.get(2), Some(1));
    }

    #[test]
    fn test_weekday_zero_based() {
        // Mon, Fri, Sat, Sun of 2024-01
        let col = datetime_col(
            "t",
            &[
                Some(T0),
                Some(T0 + 4 * DAY_US),
                Some(T0 + 5 * DAY_US),
                Some(T0 + 6 * DAY_US),
            ],
        );
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::Weekday]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let wd = out.column("t_weekday").unwrap().i32().unwrap();
        assert_eq!(wd.get(0), Some(0)); // Monday
        assert_eq!(wd.get(1), Some(4)); // Friday
        assert_eq!(wd.get(2), Some(5)); // Saturday
        assert_eq!(wd.get(3), Some(6)); // Sunday
    }

    #[test]
    fn test_is_weekend() {
        // Mon, Fri, Sat, Sun
        let col = datetime_col(
            "t",
            &[
                Some(T0),
                Some(T0 + 4 * DAY_US),
                Some(T0 + 5 * DAY_US),
                Some(T0 + 6 * DAY_US),
            ],
        );
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::IsWeekend]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let flag = out.column("t_is_weekend").unwrap();
        assert_eq!(flag.dtype(), &DataType::Float64);
        let flag = flag.f64().unwrap();
        assert_eq!(flag.get(0), Some(0.0));
        assert_eq!(flag.get(1), Some(0.0));
        assert_eq!(flag.get(2), Some(1.0));
        assert_eq!(flag.get(3), Some(1.0));
    }

    #[test]
    fn test_quarter() {
        // 2024-01-01 (Q1), 2024-03-01 (Q1), 2024-04-01 (Q2), 2024-10-01 (Q4)
        let col = datetime_col(
            "t",
            &[
                Some(T0),
                Some(T0 + 60 * DAY_US),
                Some(T0 + 91 * DAY_US),
                Some(T0 + 274 * DAY_US),
            ],
        );
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::Quarter]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let q = out.column("t_quarter").unwrap().i32().unwrap();
        assert_eq!(q.get(0), Some(1));
        assert_eq!(q.get(1), Some(1));
        assert_eq!(q.get(2), Some(2));
        assert_eq!(q.get(3), Some(4));
    }

    #[test]
    fn test_day_of_year_and_week() {
        // 2024-01-01 (DOY 1, ISO week 1), 2024-02-29 (DOY 60, week 9),
        // 2024-03-04 (DOY 64, ISO week 10)
        let col = datetime_col(
            "t",
            &[Some(T0), Some(T0 + 59 * DAY_US), Some(T0 + 63 * DAY_US)],
        );
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::DayOfYear, DatetimeComponent::WeekOfYear]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let doy = out.column("t_day_of_year").unwrap().i32().unwrap();
        assert_eq!(doy.get(0), Some(1));
        assert_eq!(doy.get(1), Some(60));
        assert_eq!(doy.get(2), Some(64));
        let wk = out.column("t_week_of_year").unwrap().i32().unwrap();
        assert_eq!(wk.get(0), Some(1));
        assert_eq!(wk.get(1), Some(9));
        assert_eq!(wk.get(2), Some(10));
    }

    #[test]
    fn test_hour_minute_second_on_datetime() {
        // 00:00:00 and 13:00:00 on 2024-01-01
        let col = datetime_col("t", &[Some(T0), Some(T0 + 13 * 3_600 * 1_000_000)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["t"]).components(&[
            DatetimeComponent::Hour,
            DatetimeComponent::Minute,
            DatetimeComponent::Second,
        ]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let hour = out.column("t_hour").unwrap().i32().unwrap();
        assert_eq!(hour.get(0), Some(0));
        assert_eq!(hour.get(1), Some(13));
        let minute = out.column("t_minute").unwrap().i32().unwrap();
        assert_eq!(minute.get(0), Some(0));
        assert_eq!(minute.get(1), Some(0));
        let second = out.column("t_second").unwrap().i32().unwrap();
        assert_eq!(second.get(0), Some(0));
        assert_eq!(second.get(1), Some(0));
    }

    #[test]
    fn test_date_column_components() {
        // 2024-01-01, 2024-02-01, 2024-03-01 as Date (days since epoch)
        let col = date_col("d", &[Some(D0), Some(D0 + 31), Some(D0 + 60)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["d"])
            .components(&[DatetimeComponent::Year, DatetimeComponent::Month]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let year = out.column("d_year").unwrap().i32().unwrap();
        assert_eq!(year.get(0), Some(2024));
        let month = out.column("d_month").unwrap().i32().unwrap();
        assert_eq!(month.get(0), Some(1));
        assert_eq!(month.get(1), Some(2));
        assert_eq!(month.get(2), Some(3));
    }

    #[test]
    fn test_date_column_hour_zero_fill_with_nulls() {
        let col = date_col("d", &[Some(D0), None, Some(D0 + 1)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["d"])
            .components(&[DatetimeComponent::Hour]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let hour = out.column("d_hour").unwrap().i32().unwrap();
        assert_eq!(hour.get(0), Some(0));
        assert!(hour.get(1).is_none());
        assert_eq!(hour.get(2), Some(0));
    }

    #[test]
    fn test_null_preservation() {
        let col = datetime_col("t", &[Some(T0), None, Some(T0 + 31 * DAY_US)]);
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::Month, DatetimeComponent::IsWeekend]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let month = out.column("t_month").unwrap().i32().unwrap();
        assert_eq!(month.get(0), Some(1));
        assert!(month.get(1).is_none());
        assert_eq!(month.get(2), Some(2));
        let flag = out.column("t_is_weekend").unwrap().f64().unwrap();
        assert!(flag.get(1).is_none());
    }

    #[test]
    fn test_auto_discovery_only_datetime_columns() {
        let t = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let d = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let df = DataFrame::new(2, vec![t, f, d]).unwrap();

        let mut xf = DatetimeFeatures::new().components(&[DatetimeComponent::Month]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // t, x, d + t_month + d_month (x untouched)
        assert_eq!(out.width(), 5);
        assert!(out.column("t_month").is_ok());
        assert!(out.column("d_month").is_ok());
        assert!(out.column("x_month").is_err());
        assert_eq!(out.column("x").unwrap().f64().unwrap().get(1), Some(2.0));
    }

    #[test]
    fn test_default_components() {
        let col = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["t"]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // default components: Year, Month, Day
        assert_eq!(out.width(), 4);
        assert!(out.column("t_year").is_ok());
        assert!(out.column("t_month").is_ok());
        assert!(out.column("t_day").is_ok());
        assert!(out.column("t_hour").is_err());
    }

    #[test]
    fn test_duplicate_config_deduped() {
        let col = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t", "t"])
            .components(&[DatetimeComponent::Month, DatetimeComponent::Month]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // one input column + one deduped t_month
        assert_eq!(out.width(), 2);
        assert!(out.column("t_month").is_ok());
    }

    #[test]
    fn test_not_fitted_error() {
        let col = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df = DataFrame::new(2, vec![col]).unwrap();
        let xf = DatetimeFeatures::new().columns(&["t"]);
        let err = xf.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_empty_dataframe_rejected() {
        let mut xf = DatetimeFeatures::new();
        let err = xf.fit(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_missing_column_errors() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut xf = DatetimeFeatures::new().columns(&["nope"]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_non_datetime_column_errors() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut xf = DatetimeFeatures::new().columns(&["x"]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_component_name_collision_at_fit() {
        // "d_day" is a real input column; extracting Day from "d" would
        // generate "d_day" and silently overwrite it.
        let d = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let d_day = date_col("d_day", &[Some(D0), Some(D0 + 1)]);
        let df = DataFrame::new(2, vec![d, d_day]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["d"])
            .components(&[DatetimeComponent::Day]);
        let err = xf.fit(df).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("d_day"), "got: {msg}"),
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_input_collision_rejected() {
        let d = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let fit_df = DataFrame::new(2, vec![d]).unwrap();

        // transform input gains a column named like a future generated column
        let d2 = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let d_day = date_col("d_day", &[Some(D0 + 5), Some(D0 + 6)]);
        let transform_df = DataFrame::new(2, vec![d2, d_day]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["d"])
            .components(&[DatetimeComponent::Day]);
        xf.fit(fit_df).unwrap();
        let err = xf.transform(transform_df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_generated_vs_input_collision_at_fit() {
        // columns "a" and "a_month" (both real datetimes) with Month on "a"
        // generates "a_month", colliding with the input column "a_month".
        let a = datetime_col("a", &[Some(T0), Some(T0 + DAY_US)]);
        let a_month = datetime_col("a_month", &[Some(T0), Some(T0 + DAY_US)]);
        let df = DataFrame::new(2, vec![a, a_month]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["a"])
            .components(&[DatetimeComponent::Month, DatetimeComponent::Year]);
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
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
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
        );
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new()
            .columns(&["t"])
            .components(&[DatetimeComponent::Year, DatetimeComponent::Month]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        assert_eq!(
            out.column("t_year").unwrap().i32().unwrap().get(0),
            Some(2024)
        );
        assert_eq!(
            out.column("t_month").unwrap().i32().unwrap().get(1),
            Some(2)
        );
    }

    #[test]
    fn test_date_column_calendar_components() {
        // 2024-01-01 (Mon), 2024-01-06 (Sat), 2024-01-07 (Sun), 2024-04-01 (Mon)
        let col = date_col("d", &[Some(D0), Some(D0 + 5), Some(D0 + 6), Some(D0 + 91)]);
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["d"]).components(&[
            DatetimeComponent::Weekday,
            DatetimeComponent::IsWeekend,
            DatetimeComponent::Quarter,
            DatetimeComponent::DayOfYear,
        ]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let wd = out.column("d_weekday").unwrap().i32().unwrap();
        assert_eq!(wd.get(0), Some(0));
        assert_eq!(wd.get(1), Some(5));
        assert_eq!(wd.get(2), Some(6));
        assert_eq!(wd.get(3), Some(0));
        let we = out.column("d_is_weekend").unwrap().f64().unwrap();
        assert_eq!(we.get(0), Some(0.0));
        assert_eq!(we.get(1), Some(1.0));
        assert_eq!(we.get(2), Some(1.0));
        assert_eq!(we.get(3), Some(0.0));
        let q = out.column("d_quarter").unwrap().i32().unwrap();
        assert_eq!(q.get(0), Some(1));
        assert_eq!(q.get(3), Some(2));
        let doy = out.column("d_day_of_year").unwrap().i32().unwrap();
        assert_eq!(doy.get(0), Some(1));
        assert_eq!(doy.get(1), Some(6));
        assert_eq!(doy.get(2), Some(7));
        assert_eq!(doy.get(3), Some(92)); // 2024 is a leap year
    }

    #[test]
    fn test_integer_components_emit_int32() {
        let col = datetime_col("t", &[Some(T0), Some(T0 + 31 * DAY_US)]);
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["t"]).components(&[
            DatetimeComponent::Year,
            DatetimeComponent::Month,
            DatetimeComponent::Hour,
            DatetimeComponent::DayOfYear,
        ]);
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        assert_eq!(out.column("t_year").unwrap().dtype(), &DataType::Int32);
        assert_eq!(out.column("t_month").unwrap().dtype(), &DataType::Int32);
        assert_eq!(out.column("t_hour").unwrap().dtype(), &DataType::Int32);
        assert_eq!(
            out.column("t_day_of_year").unwrap().dtype(),
            &DataType::Int32
        );
    }

    #[test]
    fn test_refit_rediscovery_on_changed_schema() {
        let t1 = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df1 = DataFrame::new(2, vec![t1]).unwrap();
        // second frame has a DIFFERENT datetime column name
        let u1 = datetime_col("u", &[Some(T0), Some(T0 + DAY_US)]);
        let df2 = DataFrame::new(2, vec![u1]).unwrap();

        let mut xf = DatetimeFeatures::new().components(&[DatetimeComponent::Month]);
        xf.fit(df1.clone()).unwrap();
        let out1 = xf.transform(df1).unwrap();
        assert!(out1.column("t_month").is_ok());

        // auto-discovery must re-run on the new schema instead of erroring
        xf.fit(df2.clone()).unwrap();
        let out2 = xf.transform(df2).unwrap();
        assert!(out2.column("u_month").is_ok());
        assert!(out2.column("t_month").is_err());
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let t1 = datetime_col("t", &[Some(T0), Some(T0 + DAY_US)]);
        let df1 = DataFrame::new(2, vec![t1]).unwrap();
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df2 = DataFrame::new(2, vec![f]).unwrap();

        let mut xf = DatetimeFeatures::new().columns(&["t"]);
        xf.fit(df1).unwrap();
        // re-fit on data missing the configured column must fail...
        assert!(xf.fit(df2).is_err());
        // ...and must NOT leave the transformer in a fitted state
        let err = xf.transform(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }
}
