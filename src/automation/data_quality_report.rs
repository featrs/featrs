//! Data-quality diagnostics for a [`DataFrame`].
//!
//! [`DataQualityReport`] summarises a frame before modelling: missing values,
//! per-column statistics and outliers, cardinality, duplicated rows and
//! columns, constant columns, and the preprocessing steps it recommends.
//! It is a diagnostic, not a transformer: it holds no fitted state and does
//! not modify the frame it inspects.
//!
//! ```rust
//! use featrs::automation::data_quality_report::DataQualityReport;
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let x = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![x])?;
//!
//! let report = DataQualityReport::from_dataframe(&df)?;
//! assert_eq!(report.columns.len(), 1);
//! assert_eq!(report.overall.duplicated_rows, 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;

use polars::prelude::*;

use crate::preprocessing::scaler::percentile_sorted;
use crate::traits::{Error, Result};

/// Cardinality bucket of a column, from its count of distinct non-null values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// Fewer than 20 distinct non-null values.
    Low,
    /// Between 20 and 999 distinct non-null values.
    Medium,
    /// 1000 or more distinct non-null values.
    High,
}

/// Summary statistics of a `Float64` column.
///
/// Computed over the column's non-null, **finite** values only: `NaN` and
/// `±Inf` are excluded, matching the convention used by the crate's fitted
/// transformers. `skew` and `kurtosis` are `NaN` when the variance is zero or
/// not finite (a constant column, or magnitudes large enough to overflow the
/// second moment), because the standardized moments are undefined there.
/// `kurtosis` is the raw (non-excess) fourth standardized moment: `3.0` for a
/// Gaussian, as in [`AutoScaler`](crate::preprocessing::auto_scaler::AutoScaler).
#[derive(Debug, Clone, PartialEq)]
pub struct NumericStats {
    /// Arithmetic mean of the finite values.
    pub mean: f64,
    /// Population standard deviation (denominator `n`).
    pub std: f64,
    /// Smallest finite value.
    pub min: f64,
    /// Largest finite value.
    pub max: f64,
    /// Linear-interpolated median.
    pub median: f64,
    /// Third standardized moment, or `NaN` when undefined.
    pub skew: f64,
    /// Raw (non-excess) fourth standardized moment, or `NaN` when undefined.
    pub kurtosis: f64,
}

/// Outlier counts for a `Float64` column under the three rules
/// [`OutlierClipper`](crate::preprocessing::outlier_clipper::OutlierClipper)
/// applies, each with its default multiplier.
///
/// Counts are over the column's non-null, finite values; `NaN` and `±Inf` are
/// ignored. A rule whose spread is zero has no meaningful fence, so it reports
/// no outliers rather than flagging every value.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlierInfo {
    /// Lower Tukey fence, `Q1 - 1.5 · IQR`; `-Inf` when the IQR is zero, so a
    /// degenerate spread flags no value.
    pub iqr_low_fence: f64,
    /// Upper Tukey fence, `Q3 + 1.5 · IQR`; `+Inf` when the IQR is zero, so a
    /// degenerate spread flags no value.
    pub iqr_high_fence: f64,
    /// Values outside the Tukey fences.
    pub iqr_outlier_count: usize,
    /// Values more than 3 population standard deviations from the mean.
    pub zscore_outlier_count: usize,
    /// Values more than `3 · 1.4826 · MAD` from the median.
    pub mad_outlier_count: usize,
}

/// Summary statistics of a `String` column.
///
/// Computed over the column's non-null values; lengths are in UTF-8 bytes,
/// the unit Polars' string length kernel uses.
#[derive(Debug, Clone, PartialEq)]
pub struct StringStats {
    /// Most frequent value and its count, or `None` for an empty column.
    ///
    /// Ties are broken by first appearance in the column.
    pub most_frequent: Option<(String, usize)>,
    /// Mean byte length of the non-null values (`0.0` when there are none).
    pub mean_length: f64,
    /// Longest non-null value in bytes (`0` when there are none).
    pub max_length: usize,
}

/// Quality summary of a single column.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnQualityReport {
    /// Column name.
    pub name: String,
    /// Column dtype as Polars reports it.
    pub dtype: DataType,
    /// Number of non-null values. `NaN` counts as non-null, as in Polars.
    pub non_null_count: u64,
    /// Number of null values. `NaN` is **not** counted here.
    pub null_count: u64,
    /// `null_count / row count`, in `[0, 1]`.
    pub null_fraction: f64,
    /// Number of `NaN` values; always `0` for non-float columns.
    ///
    /// Kept separately from [`null_count`](Self::null_count) because Polars
    /// treats `NaN` as a present value: an all-`NaN` column has
    /// `null_fraction == 0.0` yet holds no usable number.
    pub nan_count: u64,
    /// Number of `±Inf` values; always `0` for non-float columns.
    ///
    /// Like `NaN`, an infinity is a present value in Polars, so it counts
    /// toward [`non_null_count`](Self::non_null_count) and is excluded from the
    /// moments. It is reported on its own because an infinity collapses a
    /// fitted range: a column holding one cannot be scaled by
    /// `MinMaxScaler`, so the report never offers that scaler for it.
    pub inf_count: u64,
    /// Distinct **non-null** values.
    ///
    /// Polars counts null as one distinct value, so this subtracts it when the
    /// column has nulls; a constant column therefore has `unique_count <= 1`.
    pub unique_count: usize,
    /// Bucket derived from [`unique_count`](Self::unique_count).
    pub cardinality: Cardinality,
    /// Numeric summary, for `Float64` columns with at least one finite value.
    pub statistics: Option<NumericStats>,
    /// Outlier counts, for `Float64` columns with at least one finite value.
    pub outliers: Option<OutlierInfo>,
    /// String summary, for `String` columns.
    pub string_stats: Option<StringStats>,
}

/// Frame-level quality summary.
#[derive(Debug, Clone, PartialEq)]
pub struct OverallQuality {
    /// `rows × columns`.
    pub total_cells: u64,
    /// Sum of the per-column null counts.
    pub total_nulls: u64,
    /// `total_nulls / total_cells`, in `[0, 1]`.
    pub overall_null_fraction: f64,
    /// Rows that repeat another row, as Polars' `is_duplicated` marks them:
    /// **every** member of a duplicate group counts, so a group of three
    /// identical rows contributes three. Nulls and `NaN` compare equal here, so
    /// two rows that are null (or `NaN`) in the same cells are duplicates.
    pub duplicated_rows: u64,
    /// Columns whose values repeat an earlier column's, by name. The earlier
    /// (kept) column is not listed.
    pub duplicate_columns: Vec<String>,
    /// Columns with at most one distinct non-null value, including all-null
    /// columns.
    pub constant_columns: Vec<String>,
}

/// A preprocessing step the report suggests for a column or the frame.
///
/// Names are the crate's own type names, so a caller can map a variant to a
/// transformer without re-deriving the heuristic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recommendation {
    /// Drop the column: it carries no information.
    DropColumn {
        /// Column name.
        name: String,
        /// Why the column is droppable.
        reason: String,
    },
    /// Impute missing values in the column.
    ImputeColumn {
        /// Column name.
        name: String,
        /// Suggested [`SimpleImputer`](crate::preprocessing::imputer::SimpleImputer)
        /// strategy, e.g. `"Median"`.
        suggested_strategy: String,
    },
    /// Scale the column.
    ScaleColumn {
        /// Column name.
        name: String,
        /// Suggested scaler, e.g. `"RobustScaler"`.
        suggested_scaler: String,
    },
    /// Encode the column's categories as numbers.
    EncodeColumn {
        /// Column name.
        name: String,
        /// Suggested encoder, e.g. `"OneHotEncoder"`.
        suggested_encoder: String,
    },
    /// Normalize the column's strings (trim, case, replacements).
    ///
    /// Reserved for a later revision: the report does not yet inspect string
    /// content, so this variant is never emitted. Callers should treat it as a
    /// future addition rather than a reachable recommendation.
    CleanStrings {
        /// Column name.
        name: String,
    },
    /// Remove duplicated rows or columns from the frame.
    RemoveDuplicates {
        /// One entry per reason, e.g. `"3 duplicated rows"`.
        reasons: Vec<String>,
    },
}

/// Structured quality report for a [`DataFrame`].
///
/// Built by [`from_dataframe`](Self::from_dataframe), which walks the frame
/// once per column plus a `group_by` for the duplicated-row count. Cost is
/// `O(rows × columns)` plus the pairwise duplicate-column comparison, which is
/// `O(rows × columns²)` — fine for typical column counts, and the same bound
/// [`DuplicateColumnRemover`](crate::preprocessing::duplicate_column_remover::DuplicateColumnRemover)
/// uses.
///
/// # Nulls, `NaN` and `±Inf`
///
/// `NaN` is a *present* value in Polars, so it counts toward `non_null_count`
/// but not toward `null_count`; [`ColumnQualityReport::nan_count`] reports it
/// separately, and [`ColumnQualityReport::inf_count`] does the same for `±Inf`.
/// `NaN` and `±Inf` are left out of [`NumericStats`] and [`OutlierInfo`], and a
/// column holding an infinity is never offered `MinMaxScaler`.
///
/// Numeric summaries are built for `Float64` columns only, matching
/// [`SimpleImputer`](crate::preprocessing::imputer::SimpleImputer) and
/// [`AutoScaler`](crate::preprocessing::auto_scaler::AutoScaler). A `Float32`
/// or integer column keeps null/unique/cardinality counts but reports no
/// `statistics`, no `outliers` and `nan_count == 0`, so a `NaN` in a `Float32`
/// column is not counted. `String` columns additionally get
/// [`ColumnQualityReport::string_stats`]; every other dtype is summarised by
/// the null and cardinality counts alone.
///
/// # Deferred
///
/// Correlations, dtype-mismatch detection, `Datetime`/`Binary` breakdowns and
/// `to_json` are out of scope: JSON output waits on an optional `serde` feature
/// (see issue #28), and the rest add disproportionate code for a first
/// diagnostic. Imputation advice covers every partly-null `String` column and
/// every `Float64` column with a usable finite summary; a `Float32`, integer,
/// boolean or datetime column with nulls is left
/// to the caller, because the crate's only imputer
/// ([`SimpleImputer`](crate::preprocessing::imputer::SimpleImputer)) accepts
/// `Float64` alone. Scaler
/// advice reads the distribution shape only, so a sparse column (more than half
/// exact zeros) is offered `StandardScaler` where
/// [`AutoScaler`](crate::preprocessing::auto_scaler::AutoScaler) would pick
/// `MaxAbsScaler`.
///
/// # Example
///
/// ```rust
/// use featrs::automation::data_quality_report::DataQualityReport;
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let x = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![x])?;
///
/// let report = DataQualityReport::from_dataframe(&df)?;
/// assert_eq!(report.overall.duplicated_rows, 0);
/// assert!(report.to_markdown().contains("| x |"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DataQualityReport {
    /// Per-column summaries, in frame order.
    pub columns: Vec<ColumnQualityReport>,
    /// Frame-level summary.
    pub overall: OverallQuality,
    /// Suggested preprocessing steps, in a stable order: drops, imputes,
    /// scales, encodes, then frame-level duplicate removal.
    pub recommendations: Vec<Recommendation>,
}

impl DataQualityReport {
    /// Inspect `df` and build the report.
    ///
    /// Returns [`Error::InvalidInput`] for a frame with no rows or no columns:
    /// the report is not defined without at least one cell.
    pub fn from_dataframe(df: &DataFrame) -> Result<Self> {
        if df.height() == 0 || df.width() == 0 {
            return Err(Error::InvalidInput(
                "DataQualityReport::from_dataframe received an empty DataFrame \
                 (0 rows or 0 columns). Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let mut columns = Vec::with_capacity(df.width());
        for col in df.columns() {
            columns.push(inspect_column(col, df.height())?);
        }

        let total_cells = (df.height() as u64) * (df.width() as u64);
        let total_nulls: u64 = columns.iter().map(|c| c.null_count).sum();

        let duplicated_rows = df
            .is_duplicated()
            .map_err(|e| {
                Error::Computation(format!(
                    "DataQualityReport: could not detect duplicated rows. {e}"
                ))
            })?
            .iter()
            .flatten()
            .filter(|dup| *dup)
            .count() as u64;

        let mut duplicate_columns = Vec::new();
        let cols = df.columns();
        for (i, col) in cols.iter().enumerate() {
            // Compare against every earlier column, not just the survivors:
            // this report uses the strict (null-aware) equality mode, where a
            // non-transitive chain cannot wrongly keep a later duplicate.
            if cols[..i]
                .iter()
                .any(|earlier| is_duplicate_of(col, earlier))
            {
                duplicate_columns.push(col.name().to_string());
            }
        }

        let constant_columns = columns
            .iter()
            .filter(|c| c.unique_count <= 1)
            .map(|c| c.name.clone())
            .collect();

        let recommendations = build_recommendations(&columns, duplicated_rows, &duplicate_columns);

        Ok(Self {
            columns,
            overall: OverallQuality {
                total_cells,
                total_nulls,
                overall_null_fraction: total_nulls as f64 / total_cells as f64,
                duplicated_rows,
                duplicate_columns,
                constant_columns,
            },
            recommendations,
        })
    }

    /// Print the report to stdout.
    pub fn print_summary(&self) {
        println!("{}", self.to_markdown());
    }

    /// Render the report as GitHub-flavoured markdown.
    ///
    /// Intended for issue trackers, PR descriptions and monitoring logs.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from("# Data quality report\n\n## Columns\n\n");
        out.push_str(
            "| Column | Dtype | Non-null | Nulls | Null % | Unique | Cardinality | Statistics |\n",
        );
        out.push_str("|---|---|---|---|---|---|---|---|\n");

        for c in &self.columns {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.1}% | {} | {:?} | {} |\n",
                md_cell(&c.name),
                c.dtype,
                c.non_null_count,
                c.null_count,
                c.null_fraction * 100.0,
                c.unique_count,
                c.cardinality,
                md_cell(&describe_statistics(c)),
            ));
        }

        out.push_str("\n## Overall\n\n");
        out.push_str(&format!("- Total cells: {}\n", self.overall.total_cells));
        out.push_str(&format!(
            "- Total nulls: {} ({:.1}%)\n",
            self.overall.total_nulls,
            self.overall.overall_null_fraction * 100.0
        ));
        out.push_str(&format!(
            "- Duplicated rows: {}\n",
            self.overall.duplicated_rows
        ));
        out.push_str(&format!(
            "- Duplicate columns: {}\n",
            list_or_none(&self.overall.duplicate_columns)
        ));
        out.push_str(&format!(
            "- Constant columns: {}\n",
            list_or_none(&self.overall.constant_columns)
        ));

        out.push_str("\n## Recommendations\n\n");
        if self.recommendations.is_empty() {
            out.push_str("None.\n");
        } else {
            out.push_str("| Recommendation | Column | Detail |\n|---|---|---|\n");
            for r in &self.recommendations {
                let (kind, column, detail): (&str, &str, String) = match r {
                    Recommendation::DropColumn { name, reason } => {
                        ("DropColumn", name, reason.clone())
                    }
                    Recommendation::ImputeColumn {
                        name,
                        suggested_strategy,
                    } => ("ImputeColumn", name, suggested_strategy.clone()),
                    Recommendation::ScaleColumn {
                        name,
                        suggested_scaler,
                    } => ("ScaleColumn", name, suggested_scaler.clone()),
                    Recommendation::EncodeColumn {
                        name,
                        suggested_encoder,
                    } => ("EncodeColumn", name, suggested_encoder.clone()),
                    Recommendation::CleanStrings { name } => {
                        ("CleanStrings", name, "clean string values".into())
                    }
                    Recommendation::RemoveDuplicates { reasons } => {
                        ("RemoveDuplicates", "-", reasons.join("; "))
                    }
                };
                out.push_str(&format!(
                    "| {} | {} | {} |\n",
                    md_cell(kind),
                    md_cell(column),
                    md_cell(&detail)
                ));
            }
        }

        out
    }
}

/// Escape a value for a GitHub-flavoured markdown table cell.
///
/// A `|` in a column name or a modal string value would otherwise split the
/// cell in two and break the table; a newline would end the row. Newlines are
/// the only character that cannot be escaped, so they become spaces.
fn md_cell(value: &str) -> String {
    value.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// Build the per-column report for one column of a frame with `height` rows.
fn inspect_column(col: &Column, height: usize) -> Result<ColumnQualityReport> {
    let name = col.name().to_string();
    let dtype = col.dtype().clone();
    let null_count = col.null_count() as u64;

    // `n_unique` counts null as one distinct value, so remove it to get the
    // count of distinct non-null values.
    let n_unique = col.n_unique().map_err(|e| {
        Error::Computation(format!(
            "DataQualityReport: could not count unique values for column '{name}'. {e}"
        ))
    })?;
    let unique_count = if null_count > 0 {
        n_unique.saturating_sub(1)
    } else {
        n_unique
    };

    let mut report = ColumnQualityReport {
        name,
        dtype: dtype.clone(),
        non_null_count: height as u64 - null_count,
        null_count,
        null_fraction: null_count as f64 / height as f64,
        nan_count: 0,
        inf_count: 0,
        unique_count,
        cardinality: cardinality_of(unique_count),
        statistics: None,
        outliers: None,
        string_stats: None,
    };

    if dtype == DataType::Float64 {
        let ca = col.f64().map_err(|e| {
            Error::Computation(format!(
                "DataQualityReport: column '{}' has dtype {}; expected Float64. {e}",
                report.name, dtype
            ))
        })?;
        let mut vals: Vec<f64> = Vec::with_capacity(ca.len());
        let mut nan_count = 0u64;
        let mut inf_count = 0u64;
        for v in ca.iter().flatten() {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                vals.push(v);
            }
        }
        report.nan_count = nan_count;
        report.inf_count = inf_count;

        if !vals.is_empty() {
            report.statistics = Some(numeric_stats(&vals));
            report.outliers = Some(outlier_info(&vals));
        }
    } else if dtype == DataType::String {
        let ca = col.str().map_err(|e| {
            Error::Computation(format!(
                "DataQualityReport: column '{}' has dtype {}; expected String. {e}",
                report.name, dtype
            ))
        })?;
        report.string_stats = Some(string_stats(ca));
    }

    Ok(report)
}

/// Cardinality bucket for a distinct-value count.
fn cardinality_of(unique_count: usize) -> Cardinality {
    if unique_count < 20 {
        Cardinality::Low
    } else if unique_count < 1000 {
        Cardinality::Medium
    } else {
        Cardinality::High
    }
}

/// Mean, spread and standardized moments of the finite values.
///
/// `vals` must be non-empty and hold only finite values.
fn numeric_stats(vals: &[f64]) -> NumericStats {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // The standardized moments divide by a power of the standard deviation, so
    // they are undefined for a constant column (and for magnitudes that
    // overflow the second moment) — report NaN rather than a fake 0.
    //
    // Each deviation is divided by the standard deviation before being raised
    // to its power, the same way `AutoImputer::skewness` does it: the raw third
    // and fourth moments overflow to `Inf` for large-but-finite values, and
    // `Inf / Inf` would collapse a perfectly measurable skew to `NaN`.
    let (skew, kurtosis) = if variance > 0.0 && variance.is_finite() {
        let sd = variance.sqrt();
        let skew = vals.iter().map(|v| ((v - mean) / sd).powi(3)).sum::<f64>() / n;
        let kurtosis = vals.iter().map(|v| ((v - mean) / sd).powi(4)).sum::<f64>() / n;
        (skew, kurtosis)
    } else {
        (f64::NAN, f64::NAN)
    };

    NumericStats {
        mean,
        std,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        median: percentile_sorted(&sorted, 50.0),
        skew,
        kurtosis,
    }
}

/// Outlier counts under the IQR, z-score and MAD rules.
///
/// `vals` must be non-empty and hold only finite values. Each rule degrades to
/// "no outliers" when its spread is zero, because a point-sized fence would
/// flag every remaining value.
fn outlier_info(vals: &[f64]) -> OutlierInfo {
    let n = vals.len() as f64;
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    let iqr = q3 - q1;
    let (iqr_lo, iqr_hi) = if iqr > 0.0 {
        (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    };
    let iqr_outlier_count = vals.iter().filter(|v| **v < iqr_lo || **v > iqr_hi).count();

    let mean = vals.iter().sum::<f64>() / n;
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let zscore_outlier_count = if std > 0.0 && std.is_finite() {
        vals.iter()
            .filter(|v| (**v - mean).abs() > 3.0 * std)
            .count()
    } else {
        0
    };

    let median = percentile_sorted(&sorted, 50.0);
    let mut devs: Vec<f64> = vals.iter().map(|v| (v - median).abs()).collect();
    devs.sort_by(|a, b| a.total_cmp(b));
    let mad = percentile_sorted(&devs, 50.0);
    let spread = 1.4826 * mad;
    let mad_outlier_count = if spread > 0.0 {
        vals.iter()
            .filter(|v| (**v - median).abs() > 3.0 * spread)
            .count()
    } else {
        0
    };

    OutlierInfo {
        iqr_low_fence: iqr_lo,
        iqr_high_fence: iqr_hi,
        iqr_outlier_count,
        zscore_outlier_count,
        mad_outlier_count,
    }
}

/// Mode, mean length and max length of the non-null strings.
fn string_stats(ca: &StringChunked) -> StringStats {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut total_len = 0usize;
    let mut max_length = 0usize;
    let mut n = 0usize;

    for v in ca.iter().flatten() {
        n += 1;
        total_len += v.len();
        max_length = max_length.max(v.len());
        *counts.entry(v).or_insert(0) += 1;
    }

    // Ties go to the value seen first, so scan the column in order for the
    // first value whose count reaches the maximum.
    let max_count = counts.values().copied().max().unwrap_or(0);
    let most_frequent = ca
        .iter()
        .flatten()
        .find(|v| counts.get(v) == Some(&max_count))
        .map(|v| (v.to_string(), max_count));

    StringStats {
        most_frequent,
        mean_length: if n == 0 {
            0.0
        } else {
            total_len as f64 / n as f64
        },
        max_length,
    }
}

/// Whether two columns hold the same values, nulls included.
///
/// Different dtypes are never duplicates, and a comparison Polars rejects
/// (e.g. categoricals with distinct category objects) is treated as "not
/// duplicates" rather than failing the report.
fn is_duplicate_of(a: &Column, b: &Column) -> bool {
    a.dtype() == b.dtype() && a.equals_missing(b)
}

/// Suggested preprocessing steps, grouped: drops, imputes, scales, encodes,
/// then frame-level duplicate removal.
fn build_recommendations(
    columns: &[ColumnQualityReport],
    duplicated_rows: u64,
    duplicate_columns: &[String],
) -> Vec<Recommendation> {
    let mut out = Vec::new();

    // Drop advice first: a dropped column needs no impute/scale/encode step.
    let mut dropped: Vec<&str> = Vec::new();
    for c in columns {
        let reason = if c.non_null_count == 0 {
            Some("100% null")
        } else if c.unique_count <= 1 {
            Some("constant column (single distinct value)")
        } else {
            None
        };
        if let Some(reason) = reason {
            dropped.push(&c.name);
            out.push(Recommendation::DropColumn {
                name: c.name.clone(),
                reason: reason.into(),
            });
        }
    }

    for c in columns {
        if c.null_count == 0 || dropped.contains(&c.name.as_str()) {
            continue;
        }
        let strategy = match &c.statistics {
            Some(s) if s.skew.abs() > 1.0 => "Median",
            Some(_) => "Mean",
            // Non-float columns: a string column's missing values are still
            // missing, and `MostFrequent` is the only strategy that applies.
            None if c.dtype == DataType::String => "MostFrequent",
            None => continue,
        };
        out.push(Recommendation::ImputeColumn {
            name: c.name.clone(),
            suggested_strategy: strategy.into(),
        });
    }

    for c in columns {
        if dropped.contains(&c.name.as_str()) {
            continue;
        }
        if let Some(scaler) = scaling_advice(c) {
            out.push(Recommendation::ScaleColumn {
                name: c.name.clone(),
                suggested_scaler: scaler.into(),
            });
        }
    }

    for c in columns {
        if dropped.contains(&c.name.as_str()) || c.dtype != DataType::String {
            continue;
        }
        // A high-cardinality string is hashed by this crate's own detector
        // (`AutoTypeDetector` maps `HighCardinality` to `FeatureHasher`); a
        // label encoding would just print an integer per distinct value.
        let encoder = if c.cardinality == Cardinality::Low {
            "OneHotEncoder"
        } else {
            "FeatureHasher"
        };
        out.push(Recommendation::EncodeColumn {
            name: c.name.clone(),
            suggested_encoder: encoder.into(),
        });
    }

    let mut reasons = Vec::new();
    if duplicated_rows > 0 {
        reasons.push(format!("{duplicated_rows} duplicated rows"));
    }
    if !duplicate_columns.is_empty() {
        reasons.push(format!(
            "duplicate columns: {}",
            duplicate_columns.join(", ")
        ));
    }
    if !reasons.is_empty() {
        out.push(Recommendation::RemoveDuplicates { reasons });
    }

    out
}

/// Scaler suggestion for a numeric column, following the same
/// "inspect the distribution, then pick" order as
/// [`AutoScaler`](crate::preprocessing::auto_scaler::AutoScaler).
///
/// `None` for columns with no usable numeric summary: non-`Float64` columns
/// and `Float64` columns holding no finite value.
///
/// A heavy tail is judged from the IQR and z-score outlier counts only. The
/// MAD rule uses the `1.4826` consistency factor, which on a smooth skewed
/// column (e.g. `x³`) flags values the other two rules call in-range; those
/// columns are better served by `PowerTransformer` than by a robust scaler.
///
/// Magnitudes large enough to overflow the second moment leave `skew` and
/// `kurtosis` as `NaN`; every comparison against `NaN` is false, so those
/// columns fall through both shape tests and are offered `MaxAbsScaler`, the
/// one delegate that only divides by `max|x|`.
fn scaling_advice(c: &ColumnQualityReport) -> Option<&'static str> {
    let stats = c.statistics.as_ref()?;
    let outliers = c.outliers.as_ref()?;

    if outliers.iqr_outlier_count > 0 || outliers.zscore_outlier_count > 0 {
        return Some("RobustScaler");
    }
    if stats.skew.is_nan() || stats.kurtosis.is_nan() {
        return Some("MaxAbsScaler");
    }
    if stats.skew.abs() > 1.0 {
        return Some("PowerTransformer");
    }
    // `MinMaxScaler` fits its range on every non-`NaN` value, so a single `±Inf`
    // collapses the range and maps the whole column to `NaN`. No other scaler
    // has that weakness, so a column holding an infinity never gets it.
    if stats.kurtosis < 2.0 && c.inf_count == 0 {
        return Some("MinMaxScaler");
    }
    Some("StandardScaler")
}

/// One-line statistics cell for the markdown table.
fn describe_statistics(c: &ColumnQualityReport) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &c.statistics {
        parts.push(format!(
            "mean={:.3}, std={:.3}, min={:.3}, max={:.3}, median={:.3}, skew={:.3}, kurtosis={:.3}",
            s.mean, s.std, s.min, s.max, s.median, s.skew, s.kurtosis
        ));
    }
    if let Some(o) = &c.outliers {
        parts.push(format!(
            "IQR outliers={} (fences {:.3}..{:.3}), z-score={}, MAD={}",
            o.iqr_outlier_count,
            o.iqr_low_fence,
            o.iqr_high_fence,
            o.zscore_outlier_count,
            o.mad_outlier_count
        ));
    }
    if let Some(s) = &c.string_stats {
        let mode = match &s.most_frequent {
            Some((v, count)) => format!("mode={v} ({count})"),
            None => "mode=-".into(),
        };
        parts.push(format!(
            "{mode}, mean length={:.1}, max length={}",
            s.mean_length, s.max_length
        ));
    }
    if c.nan_count > 0 {
        parts.push(format!("NaN={}", c.nan_count));
    }
    if c.inf_count > 0 {
        parts.push(format!("Inf={}", c.inf_count));
    }
    parts.join("; ")
}

/// Comma-separated list, or `none` when empty.
fn list_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".into()
    } else {
        items.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn col(name: &str, vals: &[f64]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    fn scol(name: &str, vals: &[&str]) -> Column {
        Column::from(Series::new(name.into(), vals))
    }

    fn df(cols: Vec<Column>) -> DataFrame {
        let n = cols[0].len();
        DataFrame::new(n, cols).unwrap()
    }

    fn report(df: &DataFrame) -> DataQualityReport {
        DataQualityReport::from_dataframe(df).unwrap()
    }

    fn column<'a>(r: &'a DataQualityReport, name: &str) -> &'a ColumnQualityReport {
        r.columns.iter().find(|c| c.name == name).unwrap()
    }

    #[test]
    fn test_empty_dataframe_errors() {
        let empty = DataFrame::empty();
        assert!(matches!(
            DataQualityReport::from_dataframe(&empty),
            Err(Error::InvalidInput(_))
        ));
        // A DataFrame::new(rows, vec![]) is (rows, 0) — no columns either.
        let zero_width = DataFrame::new(3, vec![]).unwrap();
        assert!(matches!(
            DataQualityReport::from_dataframe(&zero_width),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_column_report_fields_match_expected() {
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0_f64), None, Some(3.0), Some(5.0)],
        ));
        let r = report(&df(vec![x]));
        let c = column(&r, "x");

        assert_eq!(c.dtype, DataType::Float64);
        assert_eq!(c.null_count, 1);
        assert_eq!(c.non_null_count, 3);
        assert_relative_eq!(c.null_fraction, 0.25);
        assert_eq!(c.unique_count, 3);
        assert_eq!(c.cardinality, Cardinality::Low);
        assert_eq!(c.nan_count, 0);

        let s = c.statistics.as_ref().unwrap();
        assert_relative_eq!(s.mean, 3.0);
        assert_relative_eq!(s.min, 1.0);
        assert_relative_eq!(s.max, 5.0);
        assert_relative_eq!(s.median, 3.0);
    }

    #[test]
    fn test_overall_fields_match_expected() {
        let r = report(&df(vec![
            col("a", &[1.0, 2.0, 3.0]),
            col("b", &[4.0, 5.0, 6.0]),
        ]));
        assert_eq!(r.overall.total_cells, 6);
        assert_eq!(r.overall.total_nulls, 0);
        assert_relative_eq!(r.overall.overall_null_fraction, 0.0);
    }

    #[test]
    fn test_duplicated_rows_counted() {
        // Polars marks every member of a duplicate group, so `[1, 1, 2]`
        // reports two duplicated rows, not one.
        let r = report(&df(vec![col("a", &[1.0, 1.0, 2.0])]));
        assert_eq!(r.overall.duplicated_rows, 2);
        assert!(matches!(
            r.recommendations
                .iter()
                .find(|x| matches!(x, Recommendation::RemoveDuplicates { .. })),
            Some(Recommendation::RemoveDuplicates { .. })
        ));
    }

    #[test]
    fn test_no_duplicated_rows_recommendation_when_unique() {
        let r = report(&df(vec![col("a", &[1.0, 2.0, 3.0])]));
        assert_eq!(r.overall.duplicated_rows, 0);
        assert!(
            !r.recommendations
                .iter()
                .any(|x| matches!(x, Recommendation::RemoveDuplicates { .. }))
        );
    }

    #[test]
    fn test_constant_column_listed_and_dropped() {
        let r = report(&df(vec![
            col("k", &[7.0, 7.0, 7.0]),
            col("v", &[1.0, 2.0, 3.0]),
        ]));
        assert_eq!(r.overall.constant_columns, vec!["k".to_string()]);
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::DropColumn { name, .. } if name == "k"
        )));
    }

    #[test]
    fn test_constant_string_column_listed() {
        let r = report(&df(vec![scol("s", &["a", "a", "a"])]));
        assert_eq!(r.overall.constant_columns, vec!["s".to_string()]);
    }

    #[test]
    fn test_duplicate_column_listed() {
        let r = report(&df(vec![
            col("a", &[1.0, 2.0, 3.0]),
            col("a_copy", &[1.0, 2.0, 3.0]),
            col("b", &[4.0, 5.0, 6.0]),
        ]));
        assert_eq!(r.overall.duplicate_columns, vec!["a_copy".to_string()]);
    }

    #[test]
    fn test_near_duplicate_columns_not_listed() {
        let r = report(&df(vec![
            col("a", &[1.0, 2.0, 3.0]),
            col("b", &[1.0, 2.0, 3.5]),
        ]));
        assert!(r.overall.duplicate_columns.is_empty());
    }

    #[test]
    fn test_all_null_column_reports_drop_and_no_stats() {
        let x = Column::from(Series::new("n".into(), &[None::<f64>, None, None]));
        let r = report(&df(vec![x]));
        let c = column(&r, "n");
        assert_eq!(c.null_count, 3);
        assert_eq!(c.unique_count, 0);
        assert!(c.statistics.is_none());
        assert!(c.outliers.is_none());
        assert_relative_eq!(c.null_fraction, 1.0);
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::DropColumn { name, reason } if name == "n" && reason.contains("null")
        )));
    }

    #[test]
    fn test_all_null_column_is_constant() {
        let x = Column::from(Series::new("n".into(), &[None::<f64>, None]));
        let r = report(&df(vec![x]));
        assert_eq!(r.overall.constant_columns, vec!["n".to_string()]);
    }

    #[test]
    fn test_nan_is_non_null_and_counted_separately() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, 2.0]));
        let r = report(&df(vec![x]));
        let c = column(&r, "x");
        assert_eq!(c.null_count, 0);
        assert_eq!(c.nan_count, 2);
        assert_eq!(c.non_null_count, 3);
        // Only the finite value feeds the statistics.
        assert_relative_eq!(c.statistics.as_ref().unwrap().mean, 2.0);
    }

    #[test]
    fn test_infinite_values_are_counted_separately() {
        let x = Column::from(Series::new(
            "x".into(),
            &[
                Some(1.0_f64),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
                None,
            ],
        ));
        let r = report(&df(vec![x]));
        let c = column(&r, "x");
        assert_eq!(c.null_count, 1);
        assert_eq!(c.nan_count, 0);
        assert_eq!(c.inf_count, 2);
        // Only the finite value feeds the moments.
        assert_eq!(c.statistics.as_ref().unwrap().mean, 1.0);
    }

    #[test]
    fn test_all_infinite_column_reports_no_statistics() {
        let x = Column::from(Series::new("x".into(), &[f64::INFINITY, f64::NEG_INFINITY]));
        let r = report(&df(vec![x]));
        let c = column(&r, "x");
        assert_eq!(c.inf_count, 2);
        assert!(c.statistics.is_none());
        assert!(c.outliers.is_none());
    }

    #[test]
    fn test_infinite_column_is_not_offered_minmax_scaler() {
        // `MinMaxScaler` fits its range on non-NaN values, so an infinity
        // collapses the range and maps every value to NaN.
        let vals = [1.0_f64, 2.0, 3.0, 4.0, f64::INFINITY];
        let r = report(&df(vec![col("x", &vals)]));
        assert!(!r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ScaleColumn { suggested_scaler, .. } if suggested_scaler == "MinMaxScaler"
        )));
    }

    #[test]
    fn test_overflowing_third_moment_still_yields_finite_skew() {
        // |x| ~ 1e150 keeps the variance finite but overflows the raw third
        // moment to inf; dividing by the standard deviation before cubing keeps
        // the moment exact.
        let vals = [1e150_f64, 2e150, 3e150, 4e150, 5e150, 6e150];
        let r = report(&df(vec![col("x", &vals)]));
        let c = column(&r, "x");
        let s = c.statistics.as_ref().unwrap();
        assert!(s.skew.is_finite());
        assert!(s.kurtosis.is_finite());
        // The moments are usable, so the shape tests run: an overflowing but
        // finite column must not fall through to the overflow sentinel.
        assert!(
            !r.recommendations.iter().any(|x| matches!(
                x,
                Recommendation::ScaleColumn { suggested_scaler, .. }
                    if suggested_scaler == "MaxAbsScaler"
            )),
            "a finite column with usable moments must not fall through to MaxAbsScaler"
        );
    }

    #[test]
    fn test_skewed_column_with_overflowing_moment_advises_median() {
        // Mostly one value with a single large one: right-skewed (skew > 1) and
        // large enough that the raw third moment overflows.
        let mut vals: Vec<Option<f64>> = [1e150_f64, 1e150, 1e150, 1e150, 1e150, 1e151]
            .into_iter()
            .map(Some)
            .collect();
        vals.push(None);
        let x = Column::from(Series::new("x".into(), vals));
        let r = report(&df(vec![x]));
        let s = column(&r, "x").statistics.as_ref().unwrap();
        assert!(s.skew.abs() > 1.0, "skew should be measurable: {}", s.skew);
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ImputeColumn { suggested_strategy, .. } if suggested_strategy == "Median"
        )));
    }

    #[test]
    fn test_all_nan_column_has_no_statistics() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN]));
        let r = report(&df(vec![x]));
        let c = column(&r, "x");
        assert_eq!(c.nan_count, 2);
        assert_relative_eq!(c.null_fraction, 0.0);
        assert!(c.statistics.is_none());
    }

    #[test]
    fn test_null_adjusted_unique_count_ignores_null() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0_f64), None, Some(1.0)]));
        let r = report(&df(vec![x]));
        assert_eq!(column(&r, "x").unique_count, 1);
        assert_eq!(r.overall.constant_columns, vec!["x".to_string()]);
    }

    #[test]
    fn test_single_row_degenerate_stats() {
        let r = report(&df(vec![col("x", &[4.0])]));
        let s = column(&r, "x").statistics.as_ref().unwrap();
        assert_relative_eq!(s.mean, 4.0);
        assert_relative_eq!(s.min, 4.0);
        assert_relative_eq!(s.max, 4.0);
        assert_relative_eq!(s.median, 4.0);
        assert_relative_eq!(s.std, 0.0);
        assert!(s.skew.is_nan());
        assert!(s.kurtosis.is_nan());
    }

    #[test]
    fn test_constant_column_stats_are_nan_moments() {
        let r = report(&df(vec![col("k", &[2.0, 2.0, 2.0])]));
        let s = column(&r, "k").statistics.as_ref().unwrap();
        assert_relative_eq!(s.std, 0.0);
        assert!(s.skew.is_nan());
        assert!(s.kurtosis.is_nan());
    }

    #[test]
    fn test_cardinality_buckets() {
        let low: Vec<f64> = (0..19).map(|v| v as f64).collect();
        let medium: Vec<f64> = (0..20).map(|v| v as f64).collect();
        let edge: Vec<f64> = (0..1000).map(|v| v as f64).collect();
        let mut frames = vec![low, medium, edge];
        frames.push((0..1001).map(|v| v as f64).collect());

        let mut out = Vec::new();
        for vals in frames {
            let r = report(&df(vec![col("x", &vals)]));
            out.push(column(&r, "x").cardinality);
        }
        assert_eq!(
            out,
            vec![
                Cardinality::Low,
                Cardinality::Medium,
                Cardinality::High,
                Cardinality::High
            ]
        );
    }

    #[test]
    fn test_outlier_counts() {
        // One planted extreme value in an otherwise uniform spread; the tight
        // `10.0` block keeps the IQR from swallowing the 1000.
        let mut vals: Vec<f64> = (0..21).map(|v| v as f64).collect();
        vals.push(1000.0);
        let r = report(&df(vec![col("x", &vals)]));
        let o = column(&r, "x").outliers.as_ref().unwrap();
        assert!(o.iqr_high_fence < 1000.0);
        assert_eq!(o.iqr_outlier_count, 1);
        assert_eq!(o.zscore_outlier_count, 1);
        assert_eq!(o.mad_outlier_count, 1);
    }

    #[test]
    fn test_no_outliers_on_tight_spread() {
        let vals: Vec<f64> = (0..30).map(|v| v as f64).collect();
        let r = report(&df(vec![col("x", &vals)]));
        let o = column(&r, "x").outliers.as_ref().unwrap();
        assert_eq!(o.iqr_outlier_count, 0);
        assert_eq!(o.zscore_outlier_count, 0);
        assert_eq!(o.mad_outlier_count, 0);
    }

    #[test]
    fn test_zero_spread_reports_no_outliers() {
        let r = report(&df(vec![col("k", &[3.0, 3.0, 3.0, 3.0])]));
        let o = column(&r, "k").outliers.as_ref().unwrap();
        assert_eq!(o.iqr_outlier_count, 0);
        assert_eq!(o.zscore_outlier_count, 0);
        assert_eq!(o.mad_outlier_count, 0);
    }

    #[test]
    fn test_string_stats() {
        let r = report(&df(vec![scol("s", &["aa", "bbb", "aa", "c"])]));
        let c = column(&r, "s");
        let s = c.string_stats.as_ref().unwrap();
        assert_eq!(s.most_frequent, Some(("aa".to_string(), 2)));
        assert_relative_eq!(s.mean_length, (2.0 + 3.0 + 2.0 + 1.0) / 4.0);
        assert_eq!(s.max_length, 3);
        assert_eq!(c.unique_count, 3);
        assert_eq!(c.cardinality, Cardinality::Low);
        assert!(c.statistics.is_none());
        assert!(c.outliers.is_none());
    }

    #[test]
    fn test_string_most_frequent_tie_breaks_on_first_appearance() {
        let r = report(&df(vec![scol("s", &["b", "a"])]));
        let s = column(&r, "s").string_stats.as_ref().unwrap();
        assert_eq!(s.most_frequent, Some(("b".to_string(), 1)));
    }

    #[test]
    fn test_string_most_frequent_tie_on_max_count_follows_first_appearance() {
        // Both values reach count 2, but "b" appeared first, so it is the mode.
        let r = report(&df(vec![scol("s", &["b", "a", "a", "b"])]));
        let s = column(&r, "s").string_stats.as_ref().unwrap();
        assert_eq!(s.most_frequent, Some(("b".to_string(), 2)));
    }

    #[test]
    fn test_string_column_with_null_ignores_null_in_mode() {
        let x = Column::from(Series::new("s".into(), &[Some("a"), None, Some("a")]));
        let r = report(&df(vec![x]));
        let c = column(&r, "s");
        assert_eq!(c.null_count, 1);
        assert_eq!(c.unique_count, 1);
        let s = c.string_stats.as_ref().unwrap();
        assert_eq!(s.most_frequent, Some(("a".to_string(), 2)));
        assert_relative_eq!(s.mean_length, 1.0);
        assert_eq!(s.max_length, 1);
    }

    #[test]
    fn test_recommendations_cover_impute_drop_encode_scale() {
        let x = Column::from(Series::new(
            "num".into(),
            &[Some(1.0_f64), None, Some(3.0), Some(4.0)],
        ));
        let r = report(&df(vec![
            x,
            scol("cat", &["a", "b", "a", "b"]),
            col("k", &[5.0, 5.0, 5.0, 5.0]),
        ]));

        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ImputeColumn { name, suggested_strategy }
                if name == "num" && suggested_strategy == "Mean"
        )));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ScaleColumn { name, .. } if name == "num"
        )));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::EncodeColumn { name, .. } if name == "cat"
        )));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::DropColumn { name, .. } if name == "k"
        )));
    }

    #[test]
    fn test_low_cardinality_string_recommends_one_hot() {
        let r = report(&df(vec![scol("cat", &["a", "b", "a"])]));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::EncodeColumn { suggested_encoder, .. } if suggested_encoder == "OneHotEncoder"
        )));
    }

    #[test]
    fn test_high_cardinality_string_recommends_feature_hasher() {
        let vals: Vec<String> = (0..1000).map(|v| format!("c{v}")).collect();
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        let r = report(&df(vec![scol("cat", &refs)]));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::EncodeColumn { suggested_encoder, .. } if suggested_encoder == "FeatureHasher"
        )));
    }

    #[test]
    fn test_skewed_column_recommends_power_transformer() {
        // Cubed values: right-skewed (skew > 1) but with no IQR/z-score/MAD
        // outlier, so the scaler advice falls through to the skew test.
        let vals: Vec<f64> = (1..=30).map(|v| (v * v * v) as f64).collect();
        let r = report(&df(vec![col("x", &vals)]));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ScaleColumn { suggested_scaler, .. } if suggested_scaler == "PowerTransformer"
        )));
    }

    #[test]
    fn test_outlier_heavy_column_recommends_robust_scaler() {
        // Geometric growth: positive IQR, and `max|x - median| / IQR` well past
        // the outlier-severity threshold, so the scaler advice stops at Robust.
        let vals: Vec<f64> = (0..30).map(|i| 1.5_f64.powi(i)).collect();
        let r = report(&df(vec![col("x", &vals)]));
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ScaleColumn { suggested_scaler, .. } if suggested_scaler == "RobustScaler"
        )));
    }

    #[test]
    fn test_clean_frame_has_only_scale_recommendation() {
        let vals: Vec<f64> = (0..12).map(|v| f64::from(v) * 0.5).collect();
        let r = report(&df(vec![col("x", &vals)]));
        assert!(
            r.recommendations
                .iter()
                .all(|x| matches!(x, Recommendation::ScaleColumn { .. }))
        );
        assert_eq!(r.recommendations.len(), 1);
    }

    #[test]
    fn test_recommendation_order_is_stable() {
        let x = Column::from(Series::new("num".into(), &[Some(1.0_f64), None, Some(3.0)]));
        let r = report(&df(vec![
            x,
            scol("cat", &["a", "b", "a"]),
            col("k", &[1.0, 1.0, 1.0]),
        ]));
        let kinds: Vec<&str> = r
            .recommendations
            .iter()
            .map(|x| match x {
                Recommendation::DropColumn { .. } => "drop",
                Recommendation::ImputeColumn { .. } => "impute",
                Recommendation::ScaleColumn { .. } => "scale",
                Recommendation::EncodeColumn { .. } => "encode",
                Recommendation::CleanStrings { .. } => "clean",
                Recommendation::RemoveDuplicates { .. } => "dedup",
            })
            .collect();
        assert_eq!(kinds, vec!["drop", "impute", "scale", "encode"]);
    }

    #[test]
    fn test_to_markdown_escapes_pipe_in_cell() {
        let r = report(&df(vec![
            scol("a|b", &["x|y", "x|y", "z"]),
            col("n", &[1.0, 2.0, 3.0]),
        ]));
        let md = r.to_markdown();

        assert!(
            md.contains("a\\|b"),
            "column name pipe must be escaped: {md}"
        );
        assert!(
            md.contains("x\\|y"),
            "modal value pipe must be escaped: {md}"
        );
        // The escaped pipes must not add cells: with the escapes removed, the
        // columns-table row still has the header's 9 pipes.
        for line in md
            .lines()
            .filter(|l| l.contains("a\\|b") && l.contains("| str |"))
        {
            let plain = line.replace("\\|", "");
            assert_eq!(
                plain.matches('|').count(),
                9,
                "row gained/lost a cell: {line}"
            );
        }
    }

    #[test]
    fn test_overflowing_moments_recommend_max_abs_scaler() {
        // |x| around 1e200 squares to inf, so skew and kurtosis come out NaN
        // and the shape tests cannot fire.
        let vals = [1e200_f64, -1e200, 1e200, -1e200, 5e199, -5e199];
        let r = report(&df(vec![col("x", &vals)]));
        let c = column(&r, "x");
        assert!(c.statistics.as_ref().unwrap().skew.is_nan());
        assert!(r.recommendations.iter().any(|x| matches!(
            x,
            Recommendation::ScaleColumn { suggested_scaler, .. } if suggested_scaler == "MaxAbsScaler"
        )));
    }

    #[test]
    fn test_to_markdown_contains_columns_and_overall() {
        let x = Column::from(Series::new("num".into(), &[Some(1.0_f64), None, Some(3.0)]));
        let r = report(&df(vec![x, scol("cat", &["a", "b", "c"])]));
        let md = r.to_markdown();

        assert!(md.contains("| num |"));
        assert!(md.contains("| cat |"));
        assert!(md.contains("Duplicated rows"));
        assert!(md.contains("Recommendations"));
        assert!(md.contains("ImputeColumn"));
        assert!(md.contains("null"));
    }

    #[test]
    fn test_markdown_is_well_formed_gfm_table() {
        let r = report(&df(vec![
            col("x", &[1.0, 2.0, 3.0]),
            scol("s", &["a", "b", "a"]),
        ]));
        let md = r.to_markdown();

        // Split the markdown into runs of consecutive table lines; each run is
        // one table (header, separator, data rows) with a uniform cell count.
        let mut tables: Vec<Vec<&str>> = Vec::new();
        for line in md.lines() {
            if line.starts_with('|') {
                if tables.is_empty() {
                    tables.push(Vec::new());
                }
                tables.last_mut().unwrap().push(line);
            } else if !tables.is_empty() && !tables.last().unwrap().is_empty() {
                tables.push(Vec::new());
            }
        }
        tables.retain(|t| !t.is_empty());

        assert_eq!(
            tables.len(),
            2,
            "expected a columns table and a recommendations table, got {tables:?}"
        );
        for table in &tables {
            assert!(table.len() >= 3, "table needs header, separator and a row");
            let cells = |l: &str| l.matches('|').count();
            let header = cells(table[0]);
            for row in table {
                assert_eq!(cells(row), header, "row `{row}` has a different cell count");
            }
            assert!(table[1].contains("---"), "second row must be the separator");
        }
    }

    #[test]
    fn test_print_summary_does_not_panic() {
        let r = report(&df(vec![col("x", &[1.0, 2.0, 3.0])]));
        r.print_summary();
    }

    #[test]
    fn test_non_float_non_string_column_has_no_stats() {
        let b = Column::from(Series::new("flag".into(), &[true, false, true]));
        let r = report(&df(vec![b, col("x", &[1.0, 2.0, 3.0])]));
        let c = column(&r, "flag");
        assert_eq!(c.dtype, DataType::Boolean);
        assert_eq!(c.unique_count, 2);
        assert!(c.statistics.is_none());
        assert!(c.string_stats.is_none());
        assert_eq!(c.nan_count, 0);
    }

    #[test]
    fn test_duplicate_columns_compared_against_all_earlier_columns() {
        // a, a_copy and a_copy2 all hold the same values: both later columns
        // are redundant, not just the second one.
        let r = report(&df(vec![
            col("a", &[1.0, 2.0, 3.0]),
            col("a_copy", &[1.0, 2.0, 3.0]),
            col("a_copy2", &[1.0, 2.0, 3.0]),
        ]));
        assert_eq!(
            r.overall.duplicate_columns,
            vec!["a_copy".to_string(), "a_copy2".to_string()]
        );
    }

    #[test]
    fn test_different_dtypes_are_not_duplicates() {
        let ints = Column::from(Series::new("i".into(), &[1_i32, 2, 3]));
        let floats = col("f", &[1.0, 2.0, 3.0]);
        let r = report(&df(vec![ints, floats]));
        assert!(r.overall.duplicate_columns.is_empty());
    }
}
