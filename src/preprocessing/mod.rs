//! Data preprocessing transformations.
//!
//! Analogous to `sklearn.preprocessing`. Each sub-module provides a transformer
//! that implements [`Fit`](crate::traits::Fit) and [`Transform`](crate::traits::Transform)
//! and operates on [`DataFrame`](polars::prelude::DataFrame).

pub mod auto_type;
pub mod binarizer;
pub mod constant_column_remover;
pub mod datetime_features;
pub mod duplicate_column_remover;
pub mod encoder;
pub mod encoder_loo;
pub mod encoder_target;
pub mod feature_hasher;
pub mod imputer;
pub mod interaction_features;
pub mod kbins_discretizer;
pub mod log_transformer;
pub mod max_abs_scaler;
pub mod missing_indicator;
pub mod normalizer;
pub mod outlier_clipper;
pub mod polynomial_features;
pub mod power_transformer;
/// Quantile-based transformation to uniform or normal distribution.
pub mod quantile_transformer;
pub mod rare_category_grouper;
pub mod ratio_features;
pub mod scaler;
pub mod string_cleaner;
pub mod time_since;
pub mod winsorizer;
