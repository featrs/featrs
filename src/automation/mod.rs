//! Data-quality diagnostics.
//!
//! [`DataQualityReport`] summarises a [`DataFrame`](polars::prelude::DataFrame)
//! before modelling: missing values, per-column statistics and outliers,
//! cardinality, duplicated rows and columns, constant columns, and the
//! preprocessing steps it recommends.
//!
//! [`DataQualityReport`]: data_quality_report::DataQualityReport

pub mod data_quality_report;
