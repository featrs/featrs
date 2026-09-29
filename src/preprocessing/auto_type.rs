//! Automatic feature type detection.
//!
//! [`AutoTypeDetector`] infers the semantic type of each column and
//! applies appropriate transformations (e.g., one-hot for low-cardinality
//! strings, pass-through for floats).

use crate::pipeline::DataFrameTransformer;
use crate::preprocessing::encoder::OneHotEncoder;
use crate::preprocessing::feature_hasher::FeatureHasher;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// How to treat each detected column type.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnType {
    /// Pass through unchanged.
    ///
    /// This covers every numeric dtype (`Float64`, `Float32`, `Int8`..=`Int64`,
    /// `UInt8`..=`UInt64`), which are forwarded with their original dtype and
    /// values — no scaling, imputation, or widening is applied. `String`
    /// columns whose values are at least `numeric_string_threshold` parseable
    /// as numbers or ISO 8601 dates land here too, and are likewise forwarded
    /// untouched as strings. Scaling stays explicit and up to the caller.
    Numeric,
    /// One-hot encode (low-cardinality strings).
    Categorical,
    /// Feature hashing (high-cardinality strings).
    HighCardinality,
}

/// How many non-null string values per column are inspected when deciding
/// whether a `String` column really holds numbers or dates. The first
/// [`NUMERIC_STRING_SAMPLE`] non-null values in frame order are used, so
/// detection is deterministic.
const NUMERIC_STRING_SAMPLE: usize = 1000;

/// `true` for the numeric dtypes the detector matches explicitly. Every other
/// dtype falls through to the same `ColumnType::Numeric` catch-all arm.
fn is_numeric_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// `true` when a value parses as a number or matches an ISO 8601 date /
/// datetime. The value is whitespace-trimmed first, so `" 42 "` counts; note
/// that a bare `" 42 ".parse::<f64>()` by a caller does not.
///
/// Counting as a number means `f64::from_str` accepts the trimmed value —
/// including signs, exponents, and the non-finite spellings `inf`, `infinity`,
/// `NaN` (case-insensitive), and overflow such as `1e400`. Empty and
/// whitespace-only strings do not count.
fn parses_as_number_or_date(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.parse::<f64>().is_ok() || is_iso_datetime(trimmed)
}

/// `true` for `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` with a real calendar day
/// and a valid time of day.
///
/// Only these two layouts are supported — fractional seconds, timezone
/// suffixes, a space as the date/time separator, and every non-ISO layout are
/// deliberately out of scope, as are currency symbols and thousands
/// separators (`"$1.5"`, `"1,234.5"`).
fn is_iso_datetime(value: &str) -> bool {
    if !value.is_ascii() || (value.len() != 10 && value.len() != 19) {
        return false;
    }
    let (date, rest) = value.split_at(10);
    if !date.bytes().enumerate().all(|(i, b)| {
        if i == 4 || i == 7 {
            b == b'-'
        } else {
            b.is_ascii_digit()
        }
    }) {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        date[0..4].parse::<i32>(),
        date[5..7].parse::<u32>(),
        date[8..10].parse::<u32>(),
    ) else {
        return false;
    };
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return false;
    }
    let Some(time) = rest.strip_prefix('T') else {
        return rest.is_empty();
    };
    if time.len() != 8 {
        return false;
    }
    let parts: Vec<&str> = time.split(':').collect();
    if parts.len() != 3
        || !parts
            .iter()
            .all(|p| p.len() == 2 && p.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    let (Ok(hour), Ok(minute), Ok(second)) = (
        parts[0].parse::<u32>(),
        parts[1].parse::<u32>(),
        parts[2].parse::<u32>(),
    ) else {
        return false;
    };
    hour < 24 && minute < 60 && second < 60
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

/// Decide from a deterministic sample whether a `String` column is really
/// numeric. An empty sample (all-null column) is never numeric, which also
/// avoids dividing by zero.
fn looks_numeric(ca: &StringChunked, threshold: f64) -> bool {
    let mut sampled = 0usize;
    let mut parsed = 0usize;
    for value in ca.iter().flatten().take(NUMERIC_STRING_SAMPLE) {
        sampled += 1;
        if parses_as_number_or_date(value) {
            parsed += 1;
        }
    }
    sampled > 0 && (parsed as f64 / sampled as f64) >= threshold
}

/// Auto-detect column types apply default transformations.
///
/// Detection rules:
/// - Any numeric dtype (`Float64`, `Float32`, `Int8`..=`Int64`, `UInt8`..=`UInt64`)
///   → `Numeric`, passed through **untouched** (no scaling, imputation, or
///   widening is applied)
/// - `String` whose first 1000 non-null values are at least
///   `numeric_string_threshold` (default `0.95`) parseable as `f64` or as an
///   ISO 8601 date/datetime → `Numeric`, passed through untouched (still a
///   `String` column; parsing it is up to the caller). Values are trimmed
///   before parsing, empty strings never count, and a column with no non-null
///   value at all is never numeric (it falls back to the categorical path below,
///   where `OneHotEncoder` rejects a column with no observed value)
/// - other `String` with < `cat_threshold` unique values → `Categorical` (one-hot)
/// - other `String` with ≥ `cat_threshold` unique values → `HighCardinality` (hash)
/// - any other dtype → `Numeric` passthrough, as before
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::auto_type::AutoTypeDetector;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let num = Column::from(Series::new("num".into(), &[1.0_f64, 2.0, 3.0]));
/// let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
/// let df = DataFrame::new(3, vec![num, cat])?;
///
/// let mut atd = AutoTypeDetector::new();
/// atd.fit(df.clone())?;
/// let typed = atd.transform(df)?;
/// assert_eq!(typed.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AutoTypeDetector {
    fitted: bool,
    cat_threshold: usize,
    hash_buckets: usize,
    numeric_string_threshold: f64,
    column_types: Option<Vec<(String, ColumnType)>>,
    /// Fitted sub-transformers for the non-numeric columns, in frame order.
    /// Learned once during `fit` and reused on every `transform` call so the
    /// learned categories / hash mapping do not drift with the transform data.
    encoders: Option<Vec<(String, Box<dyn DataFrameTransformer>)>>,
}

impl AutoTypeDetector {
    /// Create a new detector with defaults: `cat_threshold = 20`,
    /// `hash_buckets = 100`, `numeric_string_threshold = 0.95`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            cat_threshold: 20,
            hash_buckets: 100,
            numeric_string_threshold: 0.95,
            column_types: None,
            encoders: None,
        }
    }

    /// Set the maximum unique values for a string column to be treated
    /// as categorical (one-hot). Default: 20.
    pub fn cat_threshold(mut self, value: usize) -> Self {
        self.cat_threshold = value;
        self
    }

    /// Set the number of hash buckets for high-cardinality columns. Default: 100.
    pub fn hash_buckets(mut self, value: usize) -> Self {
        self.hash_buckets = value;
        self
    }

    /// Set the fraction of sampled non-null string values that must parse as
    /// `f64` or an ISO 8601 date/datetime for a `String` column to be
    /// classified [`ColumnType::Numeric`]. Default: `0.95`.
    ///
    /// Up to `NUMERIC_STRING_SAMPLE` non-null values per column are sampled, in
    /// frame order. A threshold of `0.0` classifies every string column with at
    /// least one non-empty value as `Numeric`; the value is only checked in
    /// [`Fit::fit`], which rejects `NaN` and anything outside `0.0..=1.0` with
    /// [`Error::InvalidInput`].
    pub fn numeric_string_threshold(mut self, value: f64) -> Self {
        self.numeric_string_threshold = value;
        self
    }

    /// Return the inferred column types.
    pub fn column_types(&self) -> Option<&[(String, ColumnType)]> {
        self.column_types.as_deref()
    }
}

impl Default for AutoTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for AutoTypeDetector {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed fit cannot leave a stale plan usable by a
        // later transform.
        self.fitted = false;
        self.column_types = None;
        self.encoders = None;

        if self.numeric_string_threshold.is_nan()
            || !(0.0..=1.0).contains(&self.numeric_string_threshold)
        {
            return Err(Error::InvalidInput(format!(
                "AutoTypeDetector.fit: numeric_string_threshold must be in 0.0..=1.0, got {}",
                self.numeric_string_threshold
            )));
        }

        let mut types = Vec::new();
        let mut encoders: Vec<(String, Box<dyn DataFrameTransformer>)> = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let dtype = col.dtype();

            let detected = match dtype {
                dt if is_numeric_dtype(dt) => ColumnType::Numeric,
                dt if dt == &DataType::String => {
                    let ca = col.as_materialized_series().str().map_err(|_| {
                        Error::Computation(format!("could not read string column '{}'", name))
                    })?;
                    if looks_numeric(ca, self.numeric_string_threshold) {
                        ColumnType::Numeric
                    } else {
                        let n_unique = ca
                            .iter()
                            .flatten()
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        if n_unique < self.cat_threshold {
                            ColumnType::Categorical
                        } else {
                            ColumnType::HighCardinality
                        }
                    }
                }
                // Booleans, dates, datetimes, lists, ... — anything left passes
                // through untouched, same as the numeric dtypes above.
                _ => ColumnType::Numeric,
            };

            // For non-numeric columns, fit the corresponding sub-transformer on
            // the fit data now, so transform() never has to re-fit.
            match detected {
                ColumnType::Categorical => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let mut enc = OneHotEncoder::new();
                    enc.fit(subset.clone()).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType.fit: one-hot failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    encoders.push((name.clone(), Box::new(enc)));
                }
                ColumnType::HighCardinality => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let mut fh = FeatureHasher::new(&[name.as_str()], self.hash_buckets);
                    fh.fit(subset.clone()).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType.fit: hashing failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    encoders.push((name.clone(), Box::new(fh)));
                }
                ColumnType::Numeric => {}
            }

            types.push((name, detected));
        }

        self.column_types = Some(types);
        self.encoders = Some(encoders);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for AutoTypeDetector {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("AutoTypeDetector".into()));
        }
        let types = self
            .column_types
            .as_ref()
            .ok_or_else(|| Error::NotFitted("AutoTypeDetector has not been fitted.".into()))?;
        let encoders = self
            .encoders
            .as_ref()
            .ok_or_else(|| Error::NotFitted("AutoTypeDetector has not been fitted.".into()))?;
        let mut parts: Vec<DataFrame> = Vec::new();
        let mut numeric_cols: Vec<Column> = Vec::new();

        for (name, ctype) in types {
            match ctype {
                ColumnType::Numeric => {
                    if let Ok(col) = x.column(name.as_str()) {
                        numeric_cols.push(col.clone());
                    }
                }
                ColumnType::Categorical | ColumnType::HighCardinality => {
                    // Sub-transformers were fitted during `fit`; reuse them so
                    // learned categories / hash mapping are stable across calls.
                    let enc = encoders
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, e)| e)
                        .ok_or_else(|| {
                            Error::Computation(format!(
                                "AutoTypeDetector: no fitted encoder for column '{}'",
                                name
                            ))
                        })?;
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let out = enc.transform(subset).map_err(|e| {
                        Error::Computation(format!(
                            "AutoTypeDetector.transform: '{}' failed: {}",
                            name, e
                        ))
                    })?;
                    if !out.columns().is_empty() {
                        parts.push(out);
                    }
                }
            }
        }

        if !numeric_cols.is_empty() {
            let h = x.height();
            parts.push(
                DataFrame::new(h, numeric_cols).map_err(|e| Error::Computation(e.to_string()))?,
            );
        }

        if parts.is_empty() {
            return Err(Error::Computation(
                "AutoTypeDetector produced no output columns.".into(),
            ));
        }

        let mut result = parts.remove(0);
        for other in &parts {
            let cols = other.columns().to_vec();
            result = result
                .hstack(&cols)
                .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let num = Column::from(Series::new("num".into(), &[1.0f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let high = Column::from(Series::new("high".into(), &["x1", "x2", "x3"]));
        DataFrame::new(3, vec![num, cat, high]).unwrap()
    }

    #[test]
    fn test_auto_type_detect_and_transform() {
        let mut atd = AutoTypeDetector::new().cat_threshold(5).hash_buckets(8);
        let df = make_test_df();
        atd.fit(df.clone()).unwrap();

        // `cat` has 2 uniques (< threshold 5) -> Categorical; `high` has 3
        // uniques but threshold is 5 so also Categorical here. Bump threshold
        // down to push `high` into HighCardinality.
        let types: std::collections::HashMap<&str, ColumnType> = atd
            .column_types()
            .unwrap()
            .iter()
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        assert_eq!(types.get("num"), Some(&ColumnType::Numeric));
        assert_eq!(types.get("cat"), Some(&ColumnType::Categorical));

        let out = atd.transform(df).unwrap();
        assert!(out.width() >= 1);
        assert_eq!(out.height(), 3);
    }

    /// Regression: `transform` used to re-fit its sub-transformers on every
    /// call, so learned categories could drift. Verify two consecutive
    /// transforms on the same data produce identical output schemas.
    #[test]
    fn test_auto_type_transform_is_idempotent() {
        let mut atd = AutoTypeDetector::new().cat_threshold(5).hash_buckets(8);
        let df = make_test_df();
        atd.fit(df.clone()).unwrap();

        let out1 = atd.transform(df.clone()).unwrap();
        let out2 = atd.transform(df).unwrap();

        let names1: Vec<String> = out1
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let names2: Vec<String> = out2
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            names1, names2,
            "transform schema must be stable across calls"
        );
        assert_eq!(out1.height(), out2.height());
    }

    #[test]
    fn test_auto_type_not_fitted() {
        let atd = AutoTypeDetector::new();
        let df = make_test_df();
        assert!(atd.transform(df).is_err());
    }

    fn string_df(name: &str, values: &[&str]) -> DataFrame {
        DataFrame::new(
            values.len(),
            vec![Column::from(Series::new(name.into(), values))],
        )
        .unwrap()
    }

    fn detected(atd: &AutoTypeDetector, name: &str) -> ColumnType {
        atd.column_types()
            .unwrap()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .unwrap()
    }

    #[test]
    fn test_numeric_strings_are_numeric() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &["1.5", "42", "3.25"])).unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_mixed_strings_below_threshold_stay_categorical() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &["1.5", "abc"])).unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Categorical);
    }

    #[test]
    fn test_threshold_boundary_is_inclusive() {
        // 19 of 20 values parse: ratio 0.95 == default threshold -> Numeric.
        let mut values: Vec<String> = (0..19).map(|i| i.to_string()).collect();
        values.push("abc".to_string());
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &refs)).unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_empty_string_is_a_parse_failure() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &["1.5", "", "42"])).unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Categorical);

        let mut lenient = AutoTypeDetector::new().numeric_string_threshold(0.6);
        lenient.fit(string_df("s", &["1.5", "", "42"])).unwrap();
        assert_eq!(detected(&lenient, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_iso_date_strings_pass_through() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df(
            "s",
            &["2024-01-01", "2024-02-02", "2024-03-03T10:00:00"],
        ))
        .unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_currency_and_thousands_separators_are_out_of_scope() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &["$1.5", "1,234.5"])).unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Categorical);
    }

    #[test]
    fn test_numeric_string_threshold_builder_changes_outcome() {
        let df = string_df("s", &["1", "2", "3", "4", "abc", "def"]);
        let mut strict = AutoTypeDetector::new();
        strict.fit(df.clone()).unwrap();
        assert_eq!(detected(&strict, "s"), ColumnType::Categorical);

        let mut lenient = AutoTypeDetector::new().numeric_string_threshold(0.5);
        lenient.fit(df).unwrap();
        assert_eq!(detected(&lenient, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_integer_subtypes_pass_through_as_numeric() {
        // `polars`' non-default `dtype-*` features gate the narrow integer
        // chunked types, so probe the ones this crate's dependency set can
        // build and assert the routing for every dtype name the detector lists.
        let mut cols = vec![
            Column::from(Series::new("f32".into(), &[1.0f32, 2.0, 3.0])),
            Column::from(Series::new("f64".into(), &[1.0f64, 2.0, 3.0])),
            Column::from(Series::new("i32".into(), &[1i32, 2, 3])),
            Column::from(Series::new("i64".into(), &[1i64, 2, 3])),
            Column::from(Series::new("u32".into(), &[1u32, 2, 3])),
            Column::from(Series::new("u64".into(), &[1u64, 2, 3])),
        ];
        let df = DataFrame::new(3, std::mem::take(&mut cols)).unwrap();

        let mut atd = AutoTypeDetector::new();
        atd.fit(df.clone()).unwrap();
        for name in ["f32", "f64", "i32", "i64", "u32", "u64"] {
            assert_eq!(detected(&atd, name), ColumnType::Numeric, "{name}");
        }

        // Untouched means the dtypes survive the transform unscaled.
        let out = atd.transform(df).unwrap();
        assert_eq!(out.column("f32").unwrap().dtype(), &DataType::Float32);
        assert_eq!(out.column("u32").unwrap().dtype(), &DataType::UInt32);
        assert_eq!(out.column("i64").unwrap().dtype(), &DataType::Int64);
    }

    #[test]
    fn test_all_null_string_column_is_not_numeric() {
        let series = Series::new("s".into(), &[None::<&str>, None]);
        let ca = series.str().unwrap();
        // An empty sample never clears the threshold — no divide by zero.
        assert!(!looks_numeric(ca, 0.95));
        assert!(!looks_numeric(ca, 0.0));

        // Fitting such a frame must not panic; the detector falls back to the
        // categorical path, which the one-hot encoder rejects for a column with
        // no observed value.
        let col = Column::from(series);
        let df = DataFrame::new(2, vec![col]).unwrap();
        let mut atd = AutoTypeDetector::new();
        assert!(atd.fit(df).is_err());
    }

    #[test]
    fn test_detection_is_deterministic() {
        let df = string_df("s", &["1.5", "42", "3.25"]);
        let mut a = AutoTypeDetector::new();
        let mut b = AutoTypeDetector::new();
        a.fit(df.clone()).unwrap();
        b.fit(df).unwrap();
        assert_eq!(a.column_types().unwrap(), b.column_types().unwrap());
    }

    #[test]
    fn test_padded_numbers_count_as_numeric() {
        let mut atd = AutoTypeDetector::new();
        atd.fit(string_df("s", &[" 42 ", "1.5", "\t3.25\n"]))
            .unwrap();
        assert_eq!(detected(&atd, "s"), ColumnType::Numeric);
    }

    #[test]
    fn test_leap_day_is_a_date_but_a_bogus_day_is_not() {
        let mut lenient = AutoTypeDetector::new();
        lenient
            .fit(string_df("s", &["2024-02-29", "2024-01-31", "2023-11-30"]))
            .unwrap();
        assert_eq!(detected(&lenient, "s"), ColumnType::Numeric);

        let mut bogus = AutoTypeDetector::new();
        bogus
            .fit(string_df("s", &["2023-02-29", "2023-02-30", "2023-13-01"]))
            .unwrap();
        assert_eq!(detected(&bogus, "s"), ColumnType::Categorical);

        let mut impossible_time = AutoTypeDetector::new();
        impossible_time
            .fit(string_df(
                "s",
                &["2024-01-01T24:00:00", "2024-01-01T00:60:00"],
            ))
            .unwrap();
        assert_eq!(detected(&impossible_time, "s"), ColumnType::Categorical);
    }

    #[test]
    fn test_out_of_range_numeric_string_threshold_is_rejected() {
        for bad in [f64::NAN, -0.1, 1.5] {
            let mut atd = AutoTypeDetector::new().numeric_string_threshold(bad);
            let err = atd.fit(string_df("s", &["1.5", "42"])).unwrap_err();
            assert!(
                matches!(err, Error::InvalidInput(_)),
                "threshold {bad} must be rejected, got {err:?}"
            );
            assert!(atd.transform(string_df("s", &["1.5"])).is_err());
        }
    }

    #[test]
    fn test_failed_refit_leaves_no_stale_state() {
        let mut atd = AutoTypeDetector::new();
        let df = string_df("s", &["1.5", "42"]);
        atd.fit(df.clone()).unwrap();
        assert!(atd.column_types().is_some());

        // An all-null String column cannot be one-hot encoded, so this fit
        // fails; the previous successful fit must not remain usable.
        let bad = DataFrame::new(
            2,
            vec![Column::from(Series::new("s".into(), &[None::<&str>, None]))],
        )
        .unwrap();
        assert!(atd.fit(bad).is_err());
        assert!(atd.column_types().is_none());
        assert!(atd.transform(df).is_err());
    }
}
