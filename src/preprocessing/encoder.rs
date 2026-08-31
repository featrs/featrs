//! Categorical encoding.
//!
//! Analogous to `sklearn.preprocessing`. Provides:
//! - [`BinaryEncoder`] — encode categories as binary digit columns
//! - [`OneHotEncoder`] — create dummy/binary columns for each category
//! - [`LabelEncoder`] — encode labels as `0..n_classes-1` integers
//! - [`OrdinalEncoder`] — encode categorical features as integer columns
//! - [`CountEncoder`] — replace categories with their raw occurrence counts
//! - [`FrequencyEncoder`] — replace categories with their observed relative frequencies

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};

fn column_unique_strings(col: &Column) -> Result<Vec<String>> {
    let s = col.as_materialized_series();
    let ca = s.str().map_err(|e| {
        Error::InvalidInput(format!(
            "Encoder: column '{}' has dtype {}; expected String. \
             Only string columns can be encoded. {}",
            col.name(),
            col.dtype(),
            e
        ))
    })?;
    let mut unique: Vec<String> = ca
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort();
    Ok(unique)
}

/// Encode categorical features as a one-hot numeric array.
///
/// Creates a binary column for each category value. Non-string columns
/// are ignored. Output columns are named `{col}_{category}`; a generated
/// name that collides with an input column or another generated column is
/// rejected at fit with [`Error::InvalidInput`].
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::OneHotEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut enc = OneHotEncoder::new().drop_first(false);
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.width(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OneHotEncoder {
    fitted: bool,
    categories: Option<Vec<OneHotCategory>>,
    drop_first: bool,
}

struct OneHotCategory {
    column: String,
    categories: Vec<String>,
}

/// The planned `(column, category)` one-hot pairs, in output order, honoring
/// `drop_first`.
///
/// This is the single source of truth for what `transform` emits: both the
/// fit-time name-collision guard and the transform loop derive their planned
/// columns from it, so they can never drift apart.
fn planned_pairs(cats: &[OneHotCategory], drop_first: bool) -> Vec<(String, String)> {
    let start = if drop_first { 1 } else { 0 };
    cats.iter()
        .flat_map(|cat| {
            cat.categories
                .iter()
                .skip(start)
                .map(move |category| (cat.column.clone(), category.clone()))
        })
        .collect()
}

impl OneHotEncoder {
    /// Create a new `OneHotEncoder`.
    ///
    /// By default, a column is created for every category. Use
    /// [`drop_first`](Self::drop_first) to drop the first category
    /// and avoid multicollinearity.
    pub fn new() -> Self {
        Self {
            fitted: false,
            categories: None,
            drop_first: false,
        }
    }

    /// Whether to drop the first category of each feature (default: `false`).
    ///
    /// When `true`, the first category (alphabetically) is omitted from the
    /// output, producing `k-1` columns for a feature with `k` categories.
    pub fn drop_first(mut self, value: bool) -> Self {
        self.drop_first = value;
        self
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for OneHotEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed fit (including the collision guard
        // below) cannot leave a stale output plan usable by a later transform.
        self.fitted = false;
        self.categories = None;
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "OneHotEncoder.fit received a DataFrame with 0 rows. \
                 Provide at least 1 row."
                    .into(),
            ));
        }
        let mut cats = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let unique = column_unique_strings(col)?;

            if !unique.is_empty() {
                cats.push(OneHotCategory {
                    column: name,
                    categories: unique,
                });
            }
        }

        if cats.is_empty() {
            return Err(Error::InvalidInput(
                "OneHotEncoder.fit: no string columns found. \
                 OneHotEncoder operates on String columns only. \
                 Cast categorical columns to String first."
                    .into(),
            ));
        }

        // Reject name collisions up front: `transform` emits `{col}_{category}`
        // for every fitted category, and a generated name equal to an input
        // column (or another generated name) would silently overwrite data.
        // Mirror the plan `transform` will produce so the two can never drift.
        let mut seen: HashSet<String> = x
            .get_column_names()
            .iter()
            .map(|n| n.as_str().to_string())
            .collect();
        for (col, category) in planned_pairs(&cats, self.drop_first) {
            let out_name = format!("{col}_{category}");
            if !seen.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "OneHotEncoder: generated column '{out_name}' collides with an \
                     existing input column or another generated column. Rename the \
                     conflicting input column."
                )));
            }
        }

        self.categories = Some(cats);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for OneHotEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "OneHotEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let cats = self.categories.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "OneHotEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let mut new_cols: Vec<Column> = Vec::new();
        let n_rows = x.height();

        for (col, category) in &planned_pairs(cats, self.drop_first) {
            let s = x.column(col).map_err(|e| {
                Error::InvalidInput(format!(
                    "OneHotEncoder.transform: column '{}' not found in input. \
                     The encoder was fitted on columns: {:?}. {}",
                    col,
                    cats.iter().map(|c| &c.column).collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "OneHotEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    col,
                    s.dtype(),
                    e
                ))
            })?;

            let mut vals = vec![0.0f64; n_rows];
            for (i, opt) in ca.iter().enumerate() {
                if let Some(v) = opt
                    && v == *category
                {
                    vals[i] = 1.0;
                }
            }
            let col_name = format!("{col}_{category}");
            new_cols.push(Column::from(Series::new(col_name.as_str().into(), &vals)));
        }

        DataFrame::new(n_rows, new_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Encode labels as integers `0` to `n_classes - 1`.
///
/// Operates on a single string column. The mapping is sorted alphabetically.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::LabelEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut enc = LabelEncoder::new();
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LabelEncoder {
    fitted: bool,
    classes: Option<Vec<String>>,
    mapping: Option<HashMap<String, usize>>,
}

impl LabelEncoder {
    /// Create a new `LabelEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            classes: None,
            mapping: None,
        }
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for LabelEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "LabelEncoder.fit expects a single column but got {} columns. \
                 Select one column before calling fit, e.g. df.select(['target']).",
                x.width()
            )));
        }
        let col = &x.columns()[0];
        let classes = column_unique_strings(col)?;

        if classes.is_empty() {
            return Err(Error::InvalidInput(format!(
                "LabelEncoder.fit: column '{}' contains no unique values. \
                 Provide data with at least one non-null string.",
                col.name()
            )));
        }

        let mapping: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        self.classes = Some(classes);
        self.mapping = Some(mapping);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for LabelEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LabelEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let mapping = self.mapping.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "LabelEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let s = x.columns()[0].as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "LabelEncoder.transform: column '{}' has dtype {}; expected String. {}",
                s.name(),
                s.dtype(),
                e
            ))
        })?;

        let encoded: ChunkedArray<UInt32Type> = ca
            .iter()
            .map(|opt| opt.and_then(|v| mapping.get(v).copied().map(|x| x as u32)))
            .collect();

        let mut series = encoded.into_series();
        series.rename(s.name().clone());
        DataFrame::new(x.height(), vec![Column::from(series)])
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Encode categorical features as integer columns.
///
/// Similar to [`LabelEncoder`] but operates on multiple columns at once.
/// Each column receives its own `0..n_categories` mapping.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::OrdinalEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut enc = OrdinalEncoder::new();
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OrdinalEncoder {
    fitted: bool,
    categories: Option<Vec<(String, HashMap<String, u32>)>>,
}

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            categories: None,
        }
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for OrdinalEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "OrdinalEncoder.fit received a DataFrame with 0 columns. \
                 Provide at least one string column to encode."
                    .into(),
            ));
        }
        let mut cats = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let classes = column_unique_strings(col)?;

            if classes.is_empty() {
                continue;
            }

            let mapping: HashMap<String, u32> = classes
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i as u32))
                .collect();

            cats.push((name, mapping));
        }

        if cats.is_empty() {
            return Err(Error::InvalidInput(
                "OrdinalEncoder.fit: no string columns found. \
                 OrdinalEncoder operates on String columns only."
                    .into(),
            ));
        }

        self.categories = Some(cats);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for OrdinalEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "OrdinalEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let mut out_cols = Vec::new();

        let cats = self.categories.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "OrdinalEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        for (name, mapping) in cats {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "OrdinalEncoder.transform: column '{}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    name,
                    cats.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "OrdinalEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let encoded: ChunkedArray<UInt32Type> = ca
                .iter()
                .map(|opt| opt.and_then(|v| mapping.get(v).copied()))
                .collect();

            let mut series = encoded.into_series();
            series.rename(name.as_str().into());
            out_cols.push(Column::from(series));
        }

        DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Replace categorical string values with their raw occurrence counts.
///
/// Each category is replaced by the number of times it was observed in the
/// training data (`fit`). This is useful when the popularity of a category is
/// itself informative (e.g. "this city appears 457 times in our customer
/// base"). Non-string columns are ignored.
///
/// Categories seen during `transform` but not during `fit` are encoded as
/// `0`; null values are preserved as null. Output columns are `UInt32`.
///
/// Note: the output is an integer dtype. Many downstream transformers in this
/// crate operate on `Float64` columns only (see `require_f64_columns`), so you
/// may need to cast the result (e.g. `with_column(col("*").cast(Float64))`)
/// before feeding it to a scaler or normalizer.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::CountEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut enc = CountEncoder::new();
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CountEncoder {
    fitted: bool,
    counts: Option<Vec<(String, HashMap<String, u32>)>>,
}

impl CountEncoder {
    /// Create a new `CountEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            counts: None,
        }
    }
}

impl Default for CountEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for CountEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "CountEncoder.fit received a DataFrame with 0 rows. \
                 Provide at least 1 row."
                    .into(),
            ));
        }
        let mut counts = Vec::new();

        for col in x.columns() {
            // Non-string columns are ignored; only String columns are counted.
            if col.dtype() != &DataType::String {
                continue;
            }
            let name = col.name().to_string();
            let ca = col.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "CountEncoder.fit: column '{}' has dtype {}; expected String. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;

            let mut mapping: HashMap<String, u32> = HashMap::new();
            for opt in ca.iter().flatten() {
                *mapping.entry(opt.to_string()).or_insert(0) += 1;
            }

            // Skip columns with no observed (non-null) category.
            if mapping.is_empty() {
                continue;
            }

            counts.push((name, mapping));
        }

        if counts.is_empty() {
            return Err(Error::InvalidInput(
                "CountEncoder.fit: no string columns found. \
                 CountEncoder operates on String columns only."
                    .into(),
            ));
        }

        self.counts = Some(counts);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CountEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "CountEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let mut out_cols = Vec::new();

        let counts = self.counts.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "CountEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        for (name, mapping) in counts {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "CountEncoder.transform: column '{}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    name,
                    counts.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "CountEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let encoded: ChunkedArray<UInt32Type> = ca
                .iter()
                .map(|opt| opt.map(|v| mapping.get(v).copied().unwrap_or(0)))
                .collect();

            let mut series = encoded.into_series();
            series.rename(name.as_str().into());
            out_cols.push(Column::from(series));
        }

        DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Replace categorical string values with their observed relative frequencies.
///
/// Each category is replaced by its relative frequency — the number of times it
/// was observed in the training data divided by the total number of non-null
/// observations in that column — so the encoded values of a column sum to
/// approximately `1.0` (up to floating-point rounding). This is the
/// proportion-based counterpart to [`CountEncoder`]: raw counts are normalized
/// by the column total. Non-string columns are ignored.
///
/// Categories seen during `transform` but not during `fit` are encoded as
/// `0.0`; null values are preserved as null. Output columns are `Float64`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::FrequencyEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut enc = FrequencyEncoder::new();
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// assert_eq!(encoded.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FrequencyEncoder {
    fitted: bool,
    column_names: Option<Vec<String>>,
    mappings: Option<Vec<HashMap<String, f64>>>,
}

impl FrequencyEncoder {
    /// Create a new `FrequencyEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            column_names: None,
            mappings: None,
        }
    }
}

impl Default for FrequencyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for FrequencyEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset any previously learned state so a failed re-fit cannot leave
        // stale mappings behind.
        self.fitted = false;
        self.column_names = None;
        self.mappings = None;

        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "FrequencyEncoder.fit received a DataFrame with 0 rows. \
                 Provide at least 1 row."
                    .into(),
            ));
        }

        let mut names = Vec::new();
        let mut mappings = Vec::new();

        for col in x.columns() {
            // Non-string columns are ignored; only String columns are encoded.
            if col.dtype() != &DataType::String {
                continue;
            }
            let name = col.name().to_string();
            let ca = col.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "FrequencyEncoder.fit: column '{}' has dtype {}; expected String. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;

            let mut mapping: HashMap<String, f64> = HashMap::new();
            let mut total: u64 = 0;
            for opt in ca.iter().flatten() {
                *mapping.entry(opt.to_string()).or_insert(0.0) += 1.0;
                total += 1;
            }

            // Skip columns with no observed (non-null) category.
            if mapping.is_empty() {
                continue;
            }

            // Normalize counts to relative frequencies (proportions in [0, 1]).
            // The denominator is the number of non-null observations, so the
            // frequencies of a column sum to 1.0 even when nulls are present.
            let total_f = total as f64;
            for v in mapping.values_mut() {
                *v /= total_f;
            }

            names.push(name);
            mappings.push(mapping);
        }

        if names.is_empty() {
            return Err(Error::InvalidInput(
                "FrequencyEncoder.fit: no string columns found. \
                 FrequencyEncoder operates on String columns only."
                    .into(),
            ));
        }

        self.column_names = Some(names);
        self.mappings = Some(mappings);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for FrequencyEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "FrequencyEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let mut out_cols = Vec::new();

        let names = self.column_names.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "FrequencyEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let mappings = self.mappings.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "FrequencyEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        for (name, mapping) in names.iter().zip(mappings.iter()) {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "FrequencyEncoder.transform: column '{}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    name, names, e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "FrequencyEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let encoded: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| mapping.get(v).copied().unwrap_or(0.0)))
                .collect();

            let mut series = encoded.into_series();
            series.rename(name.as_str().into());
            out_cols.push(Column::from(series));
        }

        DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Encode categorical string values as binary digit columns.
///
/// Each category is assigned an ordinal ID `0..n_categories` (in sorted
/// alphabetical order) during [`fit`](Fit::fit), and each ID is decomposed
/// into its binary representation, producing `max(1, ⌈log₂(n_categories)⌉)`
/// output columns per input column, named `{column}_bit_{i}` for `i` in
/// `0..n_bits` (bit `i` is the `2^i`-place bit — least-significant bit
/// first). The number of output columns grows logarithmically with the
/// number of categories instead of linearly as one-hot encoding would, which
/// is useful for high-cardinality categoricals.
///
/// Details:
/// - A column with a single category is encoded as one all-zero bit column.
/// - String columns that are entirely null at fit time are skipped (they
///   have no categories to map) and pass through the output unchanged.
/// - Categories seen during `transform` but not during `fit` are encoded as
///   an all-zeros vector (a safe, uninformative default).
/// - Null values are preserved as null across every bit column.
/// - Non-string columns pass through the output unchanged; each fitted
///   string column is replaced, in place, by its bit columns.
/// - Output columns are `UInt32` (`0`/`1`). Many downstream transformers in
///   this crate operate on `Float64` columns only (see `require_f64_columns`),
///   so you may need to cast the result (e.g.
///   `with_column(col("*").cast(Float64))`) before feeding it to a scaler or
///   normalizer.
/// - If a generated name (e.g. `color_bit_0`) would collide with another
///   output column, [`transform`](Transform::transform) returns
///   [`Error::InvalidInput`] instead of silently overwriting data. Rename
///   the conflicting input column before encoding.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::BinaryEncoder;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("color".into(), &["red", "blue", "red", "green"]));
/// let df = DataFrame::new(4, vec![col])?;
///
/// let mut enc = BinaryEncoder::new();
/// enc.fit(df.clone())?;
/// let encoded = enc.transform(df)?;
/// // 3 categories -> 2 bit columns (ceil(log2(3)))
/// assert_eq!(encoded.width(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BinaryEncoder {
    fitted: bool,
    column_names: Option<Vec<String>>,
    category_ids: Option<Vec<HashMap<String, u32>>>,
    n_bits: Option<Vec<usize>>,
}

impl BinaryEncoder {
    /// Create a new `BinaryEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            column_names: None,
            category_ids: None,
            n_bits: None,
        }
    }
}

impl Default for BinaryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for BinaryEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset any previously learned state so a failed re-fit cannot leave
        // stale mappings behind.
        self.fitted = false;
        self.column_names = None;
        self.category_ids = None;
        self.n_bits = None;

        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "BinaryEncoder.fit received a DataFrame with 0 rows. \
                 Provide at least 1 row."
                    .into(),
            ));
        }

        let mut names = Vec::new();
        let mut ids = Vec::new();
        let mut bits = Vec::new();

        for col in x.columns() {
            // Non-string columns are ignored; only String columns are encoded.
            if col.dtype() != &DataType::String {
                continue;
            }
            let name = col.name().to_string();
            let unique = column_unique_strings(col)?;

            // Skip columns with no observed (non-null) category.
            if unique.is_empty() {
                continue;
            }

            // Smallest number of bits that can represent `unique.len()`
            // distinct IDs; a single category still gets one (all-zero) bit
            // column. Integer arithmetic keeps this exact.
            let n_categories = unique.len();
            debug_assert!(
                n_categories <= u32::MAX as usize,
                "BinaryEncoder: too many categories to assign u32 IDs"
            );
            let mut n_bits = 1usize;
            let mut capacity = 2usize;
            while capacity < n_categories {
                capacity <<= 1;
                n_bits += 1;
            }

            let mapping: HashMap<String, u32> = unique
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i as u32))
                .collect();

            names.push(name);
            ids.push(mapping);
            bits.push(n_bits);
        }

        if names.is_empty() {
            return Err(Error::InvalidInput(
                "BinaryEncoder.fit: no string columns with non-null values \
                 found. BinaryEncoder operates on String columns only."
                    .into(),
            ));
        }

        self.column_names = Some(names);
        self.category_ids = Some(ids);
        self.n_bits = Some(bits);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for BinaryEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "BinaryEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let names = self.column_names.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "BinaryEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let ids = self.category_ids.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "BinaryEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let bits = self.n_bits.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "BinaryEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        // Every fitted column must exist in the transform input and still be
        // String. Without this up-front check, a missing column would silently
        // pass through the output, hiding bugs.
        for name in names.iter() {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "BinaryEncoder.transform: column '{}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    name, names, e
                ))
            })?;
            if s.dtype() != &DataType::String {
                return Err(Error::InvalidInput(format!(
                    "BinaryEncoder.transform: column '{}' has dtype {}; expected String. \
                     The encoder was fitted on String columns.",
                    name,
                    s.dtype()
                )));
            }
        }

        let fitted_idx: std::collections::HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let mut out_cols: Vec<Column> = Vec::new();
        let mut output_names: std::collections::HashSet<String> = std::collections::HashSet::new();

        for col in x.columns() {
            let name = col.name();
            match fitted_idx.get(name.as_str()) {
                Some(&i) => {
                    let ca = col.as_materialized_series().str().map_err(|e| {
                        Error::InvalidInput(format!(
                            "BinaryEncoder.transform: column '{}' has dtype {}; expected String. {}",
                            name,
                            col.dtype(),
                            e
                        ))
                    })?;
                    let mapping = &ids[i];
                    let n_bits = bits[i];

                    for j in 0..n_bits {
                        let bit_name = format!("{name}_bit_{j}");
                        if !output_names.insert(bit_name.clone()) {
                            return Err(Error::InvalidInput(format!(
                                "BinaryEncoder.transform: generated column '{bit_name}' \
                                 collides with another output column. Rename the \
                                 conflicting input column before encoding."
                            )));
                        }
                        // Bit `j` is the `2^j`-place (least-significant) bit of
                        // the category ID. Unseen categories (no mapping entry)
                        // encode as ID 0, i.e. an all-zeros vector.
                        let encoded: ChunkedArray<UInt32Type> = ca
                            .iter()
                            .map(|opt| {
                                opt.map(|v| (mapping.get(v).copied().unwrap_or(0) >> j as u32) & 1)
                            })
                            .collect();
                        let mut series = encoded.into_series();
                        series.rename(bit_name.as_str().into());
                        out_cols.push(Column::from(series));
                    }
                }
                None => {
                    if !output_names.insert(name.to_string()) {
                        return Err(Error::InvalidInput(format!(
                            "BinaryEncoder.transform: column '{name}' collides with another \
                             output column. Rename the conflicting input column before encoding."
                        )));
                    }
                    out_cols.push(col.clone());
                }
            }
        }

        DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_categorical_df() -> DataFrame {
        let a = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green"],
        ));
        let b = Column::from(Series::new("size".into(), &["S", "M", "L", "M"]));
        DataFrame::new(4, vec![a, b]).unwrap()
    }

    #[test]
    fn test_one_hot_encoder() {
        let mut enc = OneHotEncoder::new();
        let df = make_categorical_df();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 4);
    }

    #[test]
    fn test_one_hot_encoder_drop_first() {
        let mut enc = OneHotEncoder::new().drop_first(true);
        let df = make_categorical_df();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 4);
    }

    #[test]
    fn test_one_hot_name_collision_errors() {
        // Column "a" (category "b_c") and column "a_b" (category "c") both
        // generate the output name "a_b_c"; fit must reject the ambiguity.
        let a = Column::from(Series::new("a".into(), &["b_c", "x"]));
        let a_b = Column::from(Series::new("a_b".into(), &["c", "y"]));
        let df = DataFrame::new(2, vec![a, a_b]).unwrap();

        let mut enc = OneHotEncoder::new();
        let err = enc.fit(df).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("a_b_c"), "got: {msg}"),
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_one_hot_generated_vs_input_collision_errors() {
        // Column "a" (category "b") generates "a_b", which shadows the real
        // input column "a_b"; fit must reject it.
        let a = Column::from(Series::new("a".into(), &["b", "c"]));
        let a_b = Column::from(Series::new("a_b".into(), &["z", "z"]));
        let df = DataFrame::new(2, vec![a, a_b]).unwrap();

        let mut enc = OneHotEncoder::new();
        let err = enc.fit(df).unwrap_err();
        match err {
            Error::InvalidInput(msg) => assert!(msg.contains("a_b"), "got: {msg}"),
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_one_hot_drop_first_skips_colliding_category() {
        // With drop_first, "a"'s first category "b" (which would generate
        // "a_b", shadowing the input column "a_b") is dropped, so fit succeeds.
        let a = Column::from(Series::new("a".into(), &["b", "c"]));
        let a_b = Column::from(Series::new("a_b".into(), &["z", "w"]));
        let df = DataFrame::new(2, vec![a, a_b]).unwrap();

        let mut enc = OneHotEncoder::new().drop_first(true);
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();
        // "a" emits "a_c"; "a_b" emits "a_b_w".
        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_one_hot_failed_refit_clears_state() {
        // Fit successfully, then re-fit on colliding data. The failed re-fit
        // must clear the plan so a later transform returns NotFitted rather
        // than reusing the stale output plan.
        let ok =
            DataFrame::new(2, vec![Column::from(Series::new("a".into(), &["b", "x"]))]).unwrap();
        let mut enc = OneHotEncoder::new();
        enc.fit(ok.clone()).unwrap();

        let bad = DataFrame::new(
            2,
            vec![
                Column::from(Series::new("a".into(), &["b_c", "x"])),
                Column::from(Series::new("a_b".into(), &["c", "y"])),
            ],
        )
        .unwrap();
        let err = enc.fit(bad).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let err2 = enc.transform(ok).unwrap_err();
        assert!(
            matches!(err2, Error::NotFitted(_)),
            "failed re-fit must clear the output plan"
        );
    }

    #[test]
    fn test_label_encoder() {
        let mut enc = LabelEncoder::new();
        let colors = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green"],
        ));
        let df = DataFrame::new(4, vec![colors]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<u32> = result
            .column("color")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![2, 0, 2, 1]);
    }

    #[test]
    fn test_ordinal_encoder() {
        let mut enc = OrdinalEncoder::new();
        let df = make_categorical_df();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let color_vals: Vec<u32> = result
            .column("color")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(color_vals, vec![2, 0, 2, 1]);

        let size_vals: Vec<u32> = result
            .column("size")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(size_vals, vec![2, 1, 0, 1]);
    }

    #[test]
    fn test_count_encoder_counts() {
        let mut enc = CountEncoder::new();
        let df = make_categorical_df();
        let n_rows = df.height();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // color: red x2, blue x1, green x1
        let color_vals: Vec<u32> = result
            .column("color")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(color_vals, vec![2, 1, 2, 1]);

        // size: S x1, M x2, L x1
        let size_vals: Vec<u32> = result
            .column("size")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(size_vals, vec![1, 2, 1, 2]);

        // Counts observed at each row sum to the number of fitted
        // occurrences; every category's count appears exactly `count` times,
        // so the per-column sum of (count * 1) over distinct categories
        // equals n_rows. Here both columns cover all rows.
        assert_eq!(result.height(), n_rows);
    }

    #[test]
    fn test_count_encoder_counts_sum_to_n_rows() {
        // Two-category column: counts for the distinct categories must sum to
        // the number of rows.
        let col = Column::from(Series::new("c".into(), &["a", "b", "a", "a", "b", "a"]));
        let df = DataFrame::new(6, vec![col]).unwrap();
        let n_rows = df.height();

        let mut enc = CountEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<u32> = result
            .column("c")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // a x4, b x2
        assert_eq!(vals, vec![4, 2, 4, 4, 2, 4]);

        // Sum over distinct categories: 4 + 2 == n_rows.
        let distinct_sum: u32 = {
            let mut seen = std::collections::HashSet::new();
            vals.iter().filter(|v| seen.insert(*v)).sum()
        };
        assert_eq!(distinct_sum as usize, n_rows);
    }

    #[test]
    fn test_count_encoder_unseen_category_maps_to_zero() {
        let mut enc = CountEncoder::new();
        let train = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "a", "b"]))],
        )
        .unwrap();
        enc.fit(train).unwrap();

        // "zzz" was never seen during fit -> 0.
        let test = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "zzz", "b"]))],
        )
        .unwrap();
        let result = enc.transform(test).unwrap();

        let vals: Vec<u32> = result
            .column("c")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![2, 0, 1]);
    }

    #[test]
    fn test_count_encoder_not_fitted() {
        let enc = CountEncoder::new();
        let df = make_categorical_df();
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_count_encoder_nulls_preserved() {
        let mut enc = CountEncoder::new();
        let col = Column::from(Series::new(
            "c".into(),
            &[Some("a"), None, Some("a"), Some("b")],
        ));
        let df = DataFrame::new(4, vec![col]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let ca = result.column("c").unwrap().u32().unwrap();
        let vals: Vec<Option<u32>> = ca.iter().collect();
        // null stays null; nulls do not contribute to counts (a x2, b x1).
        assert_eq!(vals, vec![Some(2), None, Some(2), Some(1)]);
    }

    #[test]
    fn test_count_encoder_output_dtype_is_uint32() {
        let mut enc = CountEncoder::new();
        let df = make_categorical_df();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.column("color").unwrap().dtype(), &DataType::UInt32);
    }

    #[test]
    fn test_count_encoder_single_category_maps_to_n_rows() {
        let col = Column::from(Series::new("c".into(), &["only", "only", "only"]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut enc = CountEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<u32> = result
            .column("c")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![3, 3, 3]);
    }

    #[test]
    fn test_count_encoder_default_and_missing_column_error() {
        let mut enc = CountEncoder::default();
        let df = make_categorical_df();
        enc.fit(df).unwrap();

        // Transform a frame missing the fitted columns.
        let other =
            DataFrame::new(2, vec![Column::from(Series::new("x".into(), &["a", "b"]))]).unwrap();
        let err = enc.transform(other).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_count_encoder_skips_non_string_columns() {
        let city = Column::from(Series::new("city".into(), &["a", "b", "a"]));
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![city, x]).unwrap();

        let mut enc = CountEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // Only the string column is encoded; the numeric column is ignored.
        assert_eq!(result.width(), 1);
        let vals: Vec<u32> = result
            .column("city")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![2, 1, 2]);
    }

    #[test]
    fn test_count_encoder_no_string_columns_errors() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut enc = CountEncoder::new();
        let err = enc.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_frequency_encoder_unequal_counts() {
        let mut enc = FrequencyEncoder::new();
        let col = Column::from(Series::new("c".into(), &["a", "b", "a", "a"]));
        let df = DataFrame::new(4, vec![col]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("c")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // a x3/4, b x1/4
        assert_eq!(vals, vec![0.75, 0.25, 0.75, 0.75]);
    }

    #[test]
    fn test_frequency_encoder_equal_counts() {
        let mut enc = FrequencyEncoder::new();
        let col = Column::from(Series::new("c".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("c")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        for v in vals {
            assert_relative_eq!(v, 1.0 / 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_frequency_encoder_single_category_maps_to_one() {
        let mut enc = FrequencyEncoder::new();
        let col = Column::from(Series::new("c".into(), &["only", "only", "only"]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("c")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_frequency_encoder_unseen_category_maps_to_zero() {
        let mut enc = FrequencyEncoder::new();
        let train = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "a", "b"]))],
        )
        .unwrap();
        enc.fit(train).unwrap();

        // "zzz" was never seen during fit -> 0.0.
        let test = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "zzz", "b"]))],
        )
        .unwrap();
        let result = enc.transform(test).unwrap();

        let vals: Vec<f64> = result
            .column("c")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // a x2/3, b x1/3, unseen -> 0.0
        assert_relative_eq!(vals[0], 2.0 / 3.0, epsilon = 1e-12);
        assert_eq!(vals[1], 0.0);
        assert_relative_eq!(vals[2], 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_frequency_encoder_nulls_preserved() {
        let mut enc = FrequencyEncoder::new();
        let col = Column::from(Series::new(
            "c".into(),
            &[Some("a"), None, Some("a"), Some("b")],
        ));
        let df = DataFrame::new(4, vec![col]).unwrap();

        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let ca = result.column("c").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        // null stays null; the denominator counts non-null values only (3),
        // so frequencies still sum to ~1.0: a x2/3, b x1/3.
        assert_relative_eq!(vals[0].unwrap(), 2.0 / 3.0, epsilon = 1e-12);
        assert_eq!(vals[1], None);
        assert_relative_eq!(vals[2].unwrap(), 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3].unwrap(), 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_frequency_encoder_not_fitted() {
        let enc = FrequencyEncoder::new();
        let df = make_categorical_df();
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_frequency_encoder_empty_input_errors() {
        let mut enc = FrequencyEncoder::new();
        let df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let err = enc.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_frequency_encoder_output_dtype_is_float64() {
        let mut enc = FrequencyEncoder::new();
        let df = make_categorical_df();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.column("color").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("size").unwrap().dtype(), &DataType::Float64);
    }

    #[test]
    fn test_frequency_encoder_skips_non_string_columns() {
        let city = Column::from(Series::new("city".into(), &["a", "b", "a"]));
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![city, x]).unwrap();

        let mut enc = FrequencyEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // Only the string column is encoded; the numeric column is ignored.
        assert_eq!(result.width(), 1);
        let vals: Vec<f64> = result
            .column("city")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_frequency_encoder_no_string_columns_errors() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut enc = FrequencyEncoder::new();
        let err = enc.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_frequency_encoder_default_and_missing_column_error() {
        let mut enc = FrequencyEncoder::default();
        let df = make_categorical_df();
        enc.fit(df).unwrap();

        // Transform a frame missing the fitted columns.
        let other =
            DataFrame::new(2, vec![Column::from(Series::new("x".into(), &["a", "b"]))]).unwrap();
        let err = enc.transform(other).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_frequency_encoder_wrong_dtype_at_transform_errors() {
        let mut enc = FrequencyEncoder::new();
        let df = make_categorical_df();
        enc.fit(df.clone()).unwrap();

        // The fitted column exists but is no longer String at transform time.
        let other = DataFrame::new(
            df.height(),
            vec![Column::from(Series::new(
                "color".into(),
                &[1.0f64, 2.0, 3.0, 4.0],
            ))],
        )
        .unwrap();
        let err = enc.transform(other).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_frequency_encoder_all_null_column_skipped() {
        let a = Column::from(Series::new("a".into(), &[None::<&str>, None, None]));
        let b = Column::from(Series::new("b".into(), &["x", "x", "y"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut enc = FrequencyEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // The all-null column contributes no mapping; only "b" is encoded.
        assert_eq!(result.width(), 1);
        assert!(result.column("b").is_ok());
    }

    #[test]
    fn test_frequency_encoder_refit_resets_state() {
        let mut enc = FrequencyEncoder::new();
        let df = make_categorical_df();
        enc.fit(df.clone()).unwrap();
        let r1 = enc.transform(df.clone()).unwrap();
        assert_eq!(r1.width(), 2);

        // Re-fit on a frame with no string columns must fail and must NOT
        // leave the previous fitted state usable.
        let bad = DataFrame::new(
            2,
            vec![Column::from(Series::new("x".into(), &[1.0f64, 2.0]))],
        )
        .unwrap();
        let err = enc.fit(bad).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    fn make_binary_df() -> DataFrame {
        let color = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green"],
        ));
        DataFrame::new(4, vec![color]).unwrap()
    }

    #[test]
    fn test_binary_encoder_four_categories_two_bits() {
        let mut enc = BinaryEncoder::new();
        let df = make_binary_df();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // 3 categories -> 2 bit columns; sorted IDs: blue=0, green=1, red=2.
        assert_eq!(result.width(), 2);
        assert_eq!(result.get_column_names()[0].as_str(), "color_bit_0");
        assert_eq!(result.get_column_names()[1].as_str(), "color_bit_1");

        // LSB first: red=2 -> [0,1], blue=0 -> [0,0], green=1 -> [1,0].
        let bit0: Vec<u32> = result
            .column("color_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let bit1: Vec<u32> = result
            .column("color_bit_1")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![0, 0, 0, 1]);
        assert_eq!(bit1, vec![1, 0, 1, 0]);
    }

    #[test]
    fn test_binary_encoder_eight_categories_three_bits() {
        let vals = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let col = Column::from(Series::new("c".into(), &vals));
        let df = DataFrame::new(8, vec![col]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 3);
        // IDs a=0..h=7; bit i holds the 2^i-place digit of each ID.
        let bit0: Vec<u32> = result
            .column("c_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let bit1: Vec<u32> = result
            .column("c_bit_1")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let bit2: Vec<u32> = result
            .column("c_bit_2")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(bit1, vec![0, 0, 1, 1, 0, 0, 1, 1]);
        assert_eq!(bit2, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_binary_encoder_id_five_decomposes_to_101() {
        // Six sorted categories assign "f" the ID 5 -> binary 101 (LSB first:
        // bit_0=1, bit_1=0, bit_2=1).
        let col = Column::from(Series::new("c".into(), &["f", "a", "b", "c", "d", "e"]));
        let df = DataFrame::new(6, vec![col]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 3);
        // Rows: f=5 -> 101, a=0 -> 000, b=1 -> 001 (LSB first), c=2 -> 010,
        // d=3 -> 011, e=4 -> 100.
        let bit0: Vec<u32> = result
            .column("c_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let bit1: Vec<u32> = result
            .column("c_bit_1")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let bit2: Vec<u32> = result
            .column("c_bit_2")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![1, 0, 1, 0, 1, 0]);
        assert_eq!(bit1, vec![0, 0, 0, 1, 1, 0]);
        assert_eq!(bit2, vec![1, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_binary_encoder_two_categories_one_bit() {
        let col = Column::from(Series::new("c".into(), &["x", "y", "x"]));
        let df = DataFrame::new(3, vec![col]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        // x=0 -> 0, y=1 -> 1.
        let bit0: Vec<u32> = result
            .column("c_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![0, 1, 0]);
    }

    #[test]
    fn test_binary_encoder_single_category_all_zeros() {
        let col = Column::from(Series::new("c".into(), &["only", "only"]));
        let df = DataFrame::new(2, vec![col]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        let bit0: Vec<u32> = result
            .column("c_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![0, 0]);
    }

    #[test]
    fn test_binary_encoder_unseen_category_all_zeros() {
        let mut enc = BinaryEncoder::new();
        let train = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "a", "b"]))],
        )
        .unwrap();
        enc.fit(train).unwrap();

        let test = DataFrame::new(
            3,
            vec![Column::from(Series::new("c".into(), &["a", "zzz", "b"]))],
        )
        .unwrap();
        let result = enc.transform(test).unwrap();

        // a=0, b=1; unseen "zzz" -> all-zero vector.
        let bit0: Vec<u32> = result
            .column("c_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(bit0, vec![0, 0, 1]);
    }

    #[test]
    fn test_binary_encoder_nulls_preserved() {
        let col = Column::from(Series::new(
            "c".into(),
            &[Some("a"), None, Some("b"), Some("a")],
        ));
        let df = DataFrame::new(4, vec![col]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        let ca = result.column("c_bit_0").unwrap().u32().unwrap();
        let vals: Vec<Option<u32>> = ca.iter().collect();
        // a=0 -> 0, null stays null across the bit columns, b=1 -> 1.
        assert_eq!(vals, vec![Some(0), None, Some(1), Some(0)]);
    }

    #[test]
    fn test_binary_encoder_not_fitted() {
        let enc = BinaryEncoder::new();
        let df = make_binary_df();
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_binary_encoder_empty_input_errors() {
        let mut enc = BinaryEncoder::new();
        let df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let err = enc.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_binary_encoder_no_string_columns_errors() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();

        let mut enc = BinaryEncoder::new();
        let err = enc.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_binary_encoder_non_string_columns_pass_through() {
        let city = Column::from(Series::new("city".into(), &["a", "b", "a", "c"]));
        let price = Column::from(Series::new("price".into(), &[1.5f64, 2.5, 3.5, 4.5]));
        let df = DataFrame::new(4, vec![city, price]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // city (3 categories) -> 2 bit columns; price passes through unchanged.
        assert_eq!(result.width(), 3);
        assert_eq!(result.get_column_names()[0].as_str(), "city_bit_0");
        assert_eq!(result.get_column_names()[1].as_str(), "city_bit_1");
        assert_eq!(result.get_column_names()[2].as_str(), "price");
        assert_eq!(result.column("price").unwrap().dtype(), &DataType::Float64);
        let prices: Vec<f64> = result
            .column("price")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(prices, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_binary_encoder_output_dtype_is_uint32() {
        let mut enc = BinaryEncoder::new();
        let df = make_binary_df();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(
            result.column("color_bit_0").unwrap().dtype(),
            &DataType::UInt32
        );
    }

    #[test]
    fn test_binary_encoder_default_and_missing_column_error() {
        let mut enc = BinaryEncoder::default();
        let df = make_binary_df();
        enc.fit(df).unwrap();

        // Transform a frame missing the fitted columns.
        let other =
            DataFrame::new(2, vec![Column::from(Series::new("x".into(), &["a", "b"]))]).unwrap();
        let err = enc.transform(other).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_binary_encoder_wrong_dtype_at_transform_errors() {
        let mut enc = BinaryEncoder::new();
        let df = make_binary_df();
        enc.fit(df.clone()).unwrap();

        // The fitted column exists but is no longer String at transform time.
        let other = DataFrame::new(
            df.height(),
            vec![Column::from(Series::new(
                "color".into(),
                &[1.0f64, 2.0, 3.0, 4.0],
            ))],
        )
        .unwrap();
        let err = enc.transform(other).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_binary_encoder_all_null_column_skipped() {
        let a = Column::from(Series::new("a".into(), &[None::<&str>, None, None]));
        let b = Column::from(Series::new("b".into(), &["x", "x", "y"]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // The all-null column contributes no mapping, so it is not encoded and
        // passes through unchanged; only "b" is encoded.
        assert_eq!(result.width(), 2);
        assert!(result.column("a").is_ok());
        assert_eq!(result.column("a").unwrap().dtype(), &DataType::String);
        assert!(result.column("b_bit_0").is_ok());
    }

    #[test]
    fn test_binary_encoder_refit_resets_state() {
        let mut enc = BinaryEncoder::new();
        let df = make_binary_df();
        enc.fit(df.clone()).unwrap();
        let r1 = enc.transform(df.clone()).unwrap();
        assert_eq!(r1.width(), 2);

        // Re-fit on a frame with no string columns must fail and must NOT
        // leave the previous fitted state usable.
        let bad = DataFrame::new(
            2,
            vec![Column::from(Series::new("x".into(), &[1.0f64, 2.0]))],
        )
        .unwrap();
        let err = enc.fit(bad).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_binary_encoder_generated_name_collision_errors() {
        // Fitting "a" (2 categories) generates "a_bit_0"; an input column
        // literally named "a_bit_0" would collide -> error, not overwrite.
        let mut enc = BinaryEncoder::new();
        let train =
            DataFrame::new(2, vec![Column::from(Series::new("a".into(), &["x", "y"]))]).unwrap();
        enc.fit(train).unwrap();

        // Pass-through column after the fitted one: the generated name is
        // inserted first and the pass-through insert fails.
        let a = Column::from(Series::new("a".into(), &["x", "y"]));
        let clash = Column::from(Series::new("a_bit_0".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![a, clash]).unwrap();
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        // Reverse ordering: the pass-through name is inserted first and the
        // generated insert fails.
        let a = Column::from(Series::new("a".into(), &["x", "y"]));
        let clash = Column::from(Series::new("a_bit_0".into(), &[1.0f64, 2.0]));
        let df = DataFrame::new(2, vec![clash, a]).unwrap();
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_binary_encoder_multiple_fitted_columns() {
        // Two fitted String columns are each replaced in place by their bit
        // columns, interleaved with a pass-through column in input order.
        let a = Column::from(Series::new("a".into(), &["x", "y", "x"]));
        let num = Column::from(Series::new("num".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &["p", "p", "q"]));
        let df = DataFrame::new(3, vec![a, num, b]).unwrap();

        let mut enc = BinaryEncoder::new();
        enc.fit(df.clone()).unwrap();
        let result = enc.transform(df).unwrap();

        // a (2 cats) -> a_bit_0, num passes through, b (2 cats) -> b_bit_0.
        assert_eq!(result.width(), 3);
        assert_eq!(result.get_column_names()[0].as_str(), "a_bit_0");
        assert_eq!(result.get_column_names()[1].as_str(), "num");
        assert_eq!(result.get_column_names()[2].as_str(), "b_bit_0");

        let a_bits: Vec<u32> = result
            .column("a_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let b_bits: Vec<u32> = result
            .column("b_bit_0")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // x=0 -> 0, y=1 -> 1; p=0 -> 0, q=1 -> 1.
        assert_eq!(a_bits, vec![0, 1, 0]);
        assert_eq!(b_bits, vec![0, 0, 1]);
    }
}
