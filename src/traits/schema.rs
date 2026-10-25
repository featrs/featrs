//! Input schema validation for transformers.
//!
//! Transformers in this crate discover their working columns at `fit` time and
//! otherwise process whatever the input happens to contain, so a missing or
//! wrongly-typed column surfaces late as an opaque Polars error. This module
//! lets a transformer describe the input it needs ([`SchemaSpec`]), and lets
//! callers check a frame against that description up front
//! ([`validate_schema`]) with one aggregated, human-readable error.
//!
//! # Example
//!
//! ```rust
//! use featrs::traits::schema::{SchemaSpec, validate_schema};
//! use polars::prelude::{Column, DataFrame, DataType, NamedFrom, Series};
//!
//! let df = DataFrame::new(
//!     2,
//!     vec![Column::from(Series::new("x".into(), &[1.0_f64, 2.0]))],
//! )?;
//!
//! let spec = SchemaSpec::exact(&[("x", DataType::Float64)]);
//! assert!(validate_schema(&spec, &df).is_ok());
//!
//! let missing = SchemaSpec::exact(&[("target", DataType::Float64)]);
//! let err = validate_schema(&missing, &df).unwrap_err();
//! assert!(format!("{err}").contains("Missing column 'target'"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use polars::prelude::{DataFrame, DataType};

use crate::traits::{Error, Result};

/// A single requirement a transformer places on its input frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnRequirement {
    /// A column with this exact name and dtype must exist.
    Named {
        /// Column name.
        name: String,
        /// Required dtype.
        dtype: DataType,
    },
    /// At least `min_count` columns of this dtype must exist.
    Dtype {
        /// Required dtype.
        dtype: DataType,
        /// Minimum number of columns with that dtype.
        min_count: usize,
    },
    /// A column with this exact name must exist, whatever its dtype.
    NameOnly {
        /// Column name.
        name: String,
    },
    /// Every column in the frame must have this dtype.
    AllDtype {
        /// Dtype every column must have.
        dtype: DataType,
    },
}

/// The input schema a transformer requires.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SchemaSpec {
    /// Requirements checked in order; every one is validated.
    pub required_columns: Vec<ColumnRequirement>,
}

impl SchemaSpec {
    /// Require these `(name, dtype)` columns to be present with that dtype.
    ///
    /// Extra columns are allowed; the spec only states what must be there.
    pub fn exact(names: &[(&str, DataType)]) -> Self {
        Self {
            required_columns: names
                .iter()
                .map(|(name, dtype)| ColumnRequirement::Named {
                    name: (*name).to_string(),
                    dtype: dtype.clone(),
                })
                .collect(),
        }
    }

    /// Require at least `min` `Float64` columns.
    pub fn f64_columns(min: usize) -> Self {
        Self::dtype_columns(DataType::Float64, min)
    }

    /// Require at least `min` `String` columns.
    pub fn string_columns(min: usize) -> Self {
        Self::dtype_columns(DataType::String, min)
    }

    /// Require at least `min` columns of `dtype`.
    ///
    /// `min == 0` is vacuously satisfied.
    pub fn dtype_columns(dtype: DataType, min: usize) -> Self {
        Self {
            required_columns: vec![ColumnRequirement::Dtype {
                dtype,
                min_count: min,
            }],
        }
    }

    /// Require every column of the frame to have `dtype`.
    ///
    /// Use this for transformers that hand each column to a typed operation,
    /// so one wrongly-typed column fails the whole frame.
    pub fn all_columns(dtype: DataType) -> Self {
        Self {
            required_columns: vec![ColumnRequirement::AllDtype { dtype }],
        }
    }

    /// Build a spec from `(name, dtype)` pairs computed at runtime.
    ///
    /// Fitted transformers use this to turn the columns they learned at `fit`
    /// into named requirements, so `validate_schema` catches a frame that
    /// carries the right dtypes under the wrong names.
    pub(crate) fn from_names<I>(names: I) -> Self
    where
        I: IntoIterator<Item = (String, DataType)>,
    {
        Self {
            required_columns: names
                .into_iter()
                .map(|(name, dtype)| ColumnRequirement::Named { name, dtype })
                .collect(),
        }
    }
}

/// One way a frame can fail a [`SchemaSpec`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaViolation {
    /// A required named column is absent.
    MissingColumn {
        /// Column name.
        name: String,
        /// Dtype the requirement asked for; `None` when any dtype would do
        /// (from [`ColumnRequirement::NameOnly`]).
        expected_dtype: Option<DataType>,
    },
    /// A required named column exists with a different dtype.
    DtypeMismatch {
        /// Column name.
        name: String,
        /// Dtype the requirement asked for.
        expected: DataType,
        /// Dtype actually present.
        actual: DataType,
    },
    /// Fewer columns of a dtype than required.
    InsufficientColumns {
        /// Dtype the requirement asked for.
        dtype: DataType,
        /// Minimum required count.
        min_required: usize,
        /// Count present in the frame.
        found: usize,
    },
}

impl std::fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingColumn {
                name,
                expected_dtype,
            } => match expected_dtype {
                Some(dtype) => write!(f, "Missing column '{name}' (expected: {dtype:?})"),
                None => write!(f, "Missing column '{name}'"),
            },
            Self::DtypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Dtype mismatch on '{name}' (expected: {expected:?}, actual: {actual:?})"
            ),
            Self::InsufficientColumns {
                dtype,
                min_required,
                found,
            } => write!(
                f,
                "Require >= {min_required} {dtype:?} columns, found {found}"
            ),
        }
    }
}

/// A transformer that can describe the input schema its `fit`/`transform` needs.
///
/// Implementors return a [`SchemaSpec`] built from their current configuration,
/// so callers can validate a frame before handing it over. Transformers that
/// accept any input simply do not implement this trait.
pub trait RequiresSchema {
    /// Describe the schema this transformer requires.
    fn required_schema(&self) -> SchemaSpec;
}

/// Validate `df` against `spec`, returning every violation as one
/// [`Error::InvalidInput`].
///
/// The check is schema-only: a column that exists with the required dtype
/// satisfies the spec even when all its values are null.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when at least one requirement is unmet. The
/// message starts with `SchemaValidator: input does not satisfy requirements:`
/// and then lists every violation on its own `  - ` line, so one call reports
/// every problem in the frame rather than only the first.
pub fn validate_schema(spec: &SchemaSpec, df: &DataFrame) -> Result<()> {
    let violations = check_schema(spec, df);
    if violations.is_empty() {
        return Ok(());
    }

    let mut msg = String::from("SchemaValidator: input does not satisfy requirements:");
    for v in &violations {
        msg.push_str("\n  - ");
        msg.push_str(&v.to_string());
    }
    Err(Error::InvalidInput(msg))
}

/// Collect every way `df` fails `spec`, in requirement order.
///
/// This is the structured form of [`validate_schema`]: callers that want to
/// route on individual violations (rather than read one message) use this.
pub fn check_schema(spec: &SchemaSpec, df: &DataFrame) -> Vec<SchemaViolation> {
    let mut violations = Vec::new();
    for req in &spec.required_columns {
        match req {
            ColumnRequirement::Named { name, dtype } => match df.column(name) {
                Ok(col) => {
                    if col.dtype() != dtype {
                        violations.push(SchemaViolation::DtypeMismatch {
                            name: name.clone(),
                            expected: dtype.clone(),
                            actual: col.dtype().clone(),
                        });
                    }
                }
                // `DataFrame::column` only fails for an absent (or ambiguous)
                // name, so a lookup error is a missing column here.
                Err(_) => violations.push(SchemaViolation::MissingColumn {
                    name: name.clone(),
                    expected_dtype: Some(dtype.clone()),
                }),
            },
            ColumnRequirement::NameOnly { name } => {
                if df.column(name).is_err() {
                    violations.push(SchemaViolation::MissingColumn {
                        name: name.clone(),
                        expected_dtype: None,
                    });
                }
            }
            ColumnRequirement::AllDtype { dtype } => {
                // One mismatch per offending column, matching the aggregation
                // contract of `check_schema`: a caller correcting errors should
                // see every column that needs casting, not just the first.
                violations.extend(df.columns().iter().filter(|col| col.dtype() != dtype).map(
                    |col| SchemaViolation::DtypeMismatch {
                        name: col.name().to_string(),
                        expected: dtype.clone(),
                        actual: col.dtype().clone(),
                    },
                ));
            }
            ColumnRequirement::Dtype { dtype, min_count } => {
                let found = df
                    .columns()
                    .iter()
                    .filter(|col| col.dtype() == dtype)
                    .count();
                if found < *min_count {
                    violations.push(SchemaViolation::InsufficientColumns {
                        dtype: dtype.clone(),
                        min_required: *min_count,
                        found,
                    });
                }
            }
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{Column, NamedFrom, Series};

    fn frame() -> DataFrame {
        let x = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let cat = Column::from(Series::new("category".into(), &["a", "b"]));
        DataFrame::new(2, vec![x, cat]).unwrap()
    }

    #[test]
    fn matching_frame_passes() {
        let spec = SchemaSpec::exact(&[("x", DataType::Float64), ("category", DataType::String)]);
        assert!(validate_schema(&spec, &frame()).is_ok());
    }

    #[test]
    fn missing_named_column_is_reported() {
        let spec = SchemaSpec::exact(&[("target", DataType::Float64)]);
        let err = validate_schema(&spec, &frame()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        let msg = err.to_string();
        assert!(msg.contains("SchemaValidator"), "message: {msg}");
        assert!(
            msg.contains("Missing column 'target' (expected: Float64)"),
            "message: {msg}"
        );
    }

    #[test]
    fn dtype_mismatch_reports_expected_and_actual() {
        let spec = SchemaSpec::exact(&[("category", DataType::Float64)]);
        let msg = validate_schema(&spec, &frame()).unwrap_err().to_string();
        assert!(
            msg.contains("Dtype mismatch on 'category' (expected: Float64, actual: String)"),
            "message: {msg}"
        );
    }

    #[test]
    fn insufficient_float64_columns_reported() {
        let spec = SchemaSpec::f64_columns(2);
        let msg = validate_schema(&spec, &frame()).unwrap_err().to_string();
        assert!(
            msg.contains("Require >= 2 Float64 columns, found 1"),
            "message: {msg}"
        );
    }

    #[test]
    fn name_only_accepts_any_dtype() {
        let spec = SchemaSpec {
            required_columns: vec![
                ColumnRequirement::NameOnly {
                    name: "category".into(),
                },
                ColumnRequirement::NameOnly { name: "x".into() },
            ],
        };
        assert!(validate_schema(&spec, &frame()).is_ok());
    }

    #[test]
    fn name_only_missing_column_names_no_dtype() {
        let spec = SchemaSpec {
            required_columns: vec![ColumnRequirement::NameOnly {
                name: "missing".into(),
            }],
        };
        let msg = validate_schema(&spec, &frame()).unwrap_err().to_string();
        assert!(
            msg.contains("  - Missing column 'missing'"),
            "message: {msg}"
        );
        // NameOnly accepts any dtype, so the message must not claim one.
        assert!(!msg.contains("expected:"), "message: {msg}");
    }

    #[test]
    fn all_columns_reports_every_offender() {
        let spec = SchemaSpec::all_columns(DataType::String);
        // `frame()` has one Float64 and one String column: only `x` offends.
        let violations = check_schema(&spec, &frame());
        assert_eq!(
            violations,
            vec![SchemaViolation::DtypeMismatch {
                name: "x".into(),
                expected: DataType::String,
                actual: DataType::Float64,
            }]
        );

        let df =
            DataFrame::new(2, vec![Column::from(Series::new("s".into(), &["a", "b"]))]).unwrap();
        assert!(validate_schema(&spec, &df).is_ok());

        // Two offending columns produce two violations, not just the first.
        let both_numeric = DataFrame::new(
            2,
            vec![
                Column::from(Series::new("a".into(), &[1.0f64, 2.0])),
                Column::from(Series::new("b".into(), &[3.0f64, 4.0])),
            ],
        )
        .unwrap();
        assert_eq!(check_schema(&spec, &both_numeric).len(), 2);
    }

    #[test]
    fn check_schema_returns_structured_violations() {
        let spec = SchemaSpec::exact(&[("target", DataType::Float64)]);
        let violations = check_schema(&spec, &frame());
        assert_eq!(
            violations,
            vec![SchemaViolation::MissingColumn {
                name: "target".into(),
                expected_dtype: Some(DataType::Float64),
            }]
        );
        assert!(check_schema(&SchemaSpec::default(), &frame()).is_empty());
    }

    #[test]
    fn null_only_column_satisfies_dtype_requirement() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None]));
        let df = DataFrame::new(2, vec![x]).unwrap();
        assert!(validate_schema(&SchemaSpec::f64_columns(1), &df).is_ok());
    }

    #[test]
    fn all_violations_listed_in_one_message() {
        let spec = SchemaSpec {
            required_columns: vec![
                ColumnRequirement::Named {
                    name: "target".into(),
                    dtype: DataType::Float64,
                },
                ColumnRequirement::Named {
                    name: "category".into(),
                    dtype: DataType::Int32,
                },
                ColumnRequirement::Dtype {
                    dtype: DataType::Float64,
                    min_count: 3,
                },
            ],
        };
        let msg = validate_schema(&spec, &frame()).unwrap_err().to_string();
        let lines: Vec<&str> = msg.lines().collect();
        assert_eq!(lines.len(), 4, "message: {msg}");
        assert!(lines[0].contains("SchemaValidator"), "message: {msg}");
        assert_eq!(lines[1], "  - Missing column 'target' (expected: Float64)");
        assert_eq!(
            lines[2],
            "  - Dtype mismatch on 'category' (expected: Int32, actual: String)"
        );
        assert_eq!(lines[3], "  - Require >= 3 Float64 columns, found 1");
    }

    #[test]
    fn empty_spec_passes() {
        assert!(validate_schema(&SchemaSpec::default(), &frame()).is_ok());
    }
}
