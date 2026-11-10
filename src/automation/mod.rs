//! Data-quality diagnostics.
//!
//! [`DataQualityReport`] summarises a [`DataFrame`](polars::prelude::DataFrame)
//! before modelling: missing values, per-column statistics and outliers,
//! cardinality, duplicated rows and columns, constant columns, and the
//! preprocessing steps it recommends.
//!
//! [`AutoPipeline`](auto_pipeline::AutoPipeline) turns that kind of schema
//! inspection into action: it assembles a fitted, inspectable preprocessing
//! [`Pipeline`](crate::pipeline::Pipeline) from the columns it finds.
//!
//! [`DataQualityReport`]: data_quality_report::DataQualityReport

pub mod auto_pipeline;
pub mod data_quality_report;
