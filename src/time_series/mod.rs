//! Time-series feature engineering.
//!
//! Provides lag features, rolling window aggregations, differencing,
//! and cyclical encoding for temporal data.

pub mod cyclical;
pub mod diff;
pub mod ewma;
pub mod expanding;
pub mod lag;
pub mod rolling;
