//! `featrs` — feature engineering for Rust, inspired by scikit-learn.
//!
//! Built on [Polars](https://pola.rs), all transformations operate on
//! `DataFrame` and preserve column names throughout.
//!
//! # Quick start
//!
//! ```rust
//! use featrs::prelude::*;
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![col])?;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df.clone())?;
//! let scaled = scaler.transform(df)?;
//! assert_eq!(scaled.height(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`prelude`] | Convenient glob-import of the most common types |
//! | [`preprocessing`] | Scaling, encoding, normalization, imputation, binarization, polynomial features, feature hashing, log transformation, auto-type detection |
//! | [`pipeline`] | `Pipeline` (sequential) and `ColumnTransformer` (per-column transforms) |
//! | [`feature_selection`] | `VarianceThreshold`, `SelectKBest`, `CorrelationThreshold` |
//! | [`traits`] | Core `Fit`, `Transform`, `FitTransform` traits and error types |
//! | [`time_series`] | Lag features, rolling windows, difference, cyclical encoding |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Production code must not `unwrap()`/`expect()` Polars results — route every
// failure through `Error` instead. Tests are exempt.
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod feature_selection;
pub mod pipeline;
pub mod preprocessing;
pub mod time_series;
pub mod traits;
pub mod util;

/// Convenient glob import of the most common types.
///
/// ```rust
/// use featrs::prelude::*;
///
/// let _scaler = StandardScaler::new();
/// ```
pub mod prelude {
    pub use crate::feature_selection::CorrelationThreshold;
    pub use crate::feature_selection::SelectKBest;
    pub use crate::feature_selection::VarianceThreshold;
    pub use crate::feature_selection::select_kbest::FClassif;
    pub use crate::pipeline::ColumnTransformer;
    pub use crate::pipeline::DataFrameTransformer;
    pub use crate::pipeline::Pipeline;
    pub use crate::pipeline::column_transformer::Remainder;
    pub use crate::preprocessing::auto_type::{AutoTypeDetector, ColumnType};
    pub use crate::preprocessing::binarizer::Binarizer;
    pub use crate::preprocessing::constant_column_remover::ConstantColumnRemover;
    pub use crate::preprocessing::datetime_features::DatetimeComponent;
    pub use crate::preprocessing::datetime_features::DatetimeFeatures;
    pub use crate::preprocessing::duplicate_column_remover::DuplicateColumnRemover;
    pub use crate::preprocessing::encoder::BinaryEncoder;
    pub use crate::preprocessing::encoder::CountEncoder;
    pub use crate::preprocessing::encoder::FrequencyEncoder;
    pub use crate::preprocessing::encoder::LabelEncoder;
    pub use crate::preprocessing::encoder::OneHotEncoder;
    pub use crate::preprocessing::encoder::OrdinalEncoder;
    pub use crate::preprocessing::encoder_loo::LeaveOneOutEncoder;
    pub use crate::preprocessing::encoder_target::TargetEncoder;
    pub use crate::preprocessing::feature_hasher::FeatureHasher;
    pub use crate::preprocessing::imputer::SimpleImputer;
    pub use crate::preprocessing::imputer::Strategy;
    pub use crate::preprocessing::interaction_features::InteractionFeatures;
    pub use crate::preprocessing::interaction_features::InteractionFeaturesBuilder;
    pub use crate::preprocessing::kbins_discretizer::BinStrategy;
    pub use crate::preprocessing::kbins_discretizer::EncodeMode;
    pub use crate::preprocessing::kbins_discretizer::KBinsDiscretizer;
    pub use crate::preprocessing::log_transformer::LogMethod;
    pub use crate::preprocessing::log_transformer::LogTransformer;
    pub use crate::preprocessing::max_abs_scaler::MaxAbsScaler;
    pub use crate::preprocessing::missing_indicator::MissingIndicator;
    pub use crate::preprocessing::normalizer::Norm;
    pub use crate::preprocessing::normalizer::Normalizer;
    pub use crate::preprocessing::outlier_clipper::ClipMethod;
    pub use crate::preprocessing::outlier_clipper::OutlierClipper;
    pub use crate::preprocessing::polynomial_features::PolynomialFeatures;
    pub use crate::preprocessing::polynomial_features::PolynomialFeaturesBuilder;
    pub use crate::preprocessing::power_transformer::PowerMethod;
    pub use crate::preprocessing::power_transformer::PowerTransformer;
    pub use crate::preprocessing::quantile_transformer::OutputDistribution;
    pub use crate::preprocessing::quantile_transformer::QuantileTransformer;
    pub use crate::preprocessing::rare_category_grouper::RareCategoryGrouper;
    pub use crate::preprocessing::rare_category_grouper::Threshold;
    pub use crate::preprocessing::ratio_features::RatioFeatures;
    pub use crate::preprocessing::scaler::MinMaxScaler;
    pub use crate::preprocessing::scaler::RobustScaler;
    pub use crate::preprocessing::scaler::StandardScaler;
    pub use crate::preprocessing::string_cleaner::{CaseStyle, StringCleaner, StringReplacement};
    pub use crate::preprocessing::winsorizer::Winsorizer;
    pub use crate::time_series::cyclical::CyclicalEncoder;
    pub use crate::time_series::diff::Difference;
    pub use crate::time_series::ewma::{
        EWMASmoothing, EWMAStatistic, ExponentiallyWeightedMovingAverage,
    };
    pub use crate::time_series::expanding::ExpandingAggregator;
    pub use crate::time_series::lag::Lagger;
    pub use crate::time_series::rolling::RollingAggregator;
    pub use crate::traits::{Error, Fit, FitSupervised, FitTransform, Result, Transform};
}

// --- Shallow re-exports at crate root ---
// The canonical list lives in `prelude`; the root re-exports it so that
// `featrs::StandardScaler` and `featrs::prelude::StandardScaler` both work.
// Add new public types to `prelude` only.
pub use crate::prelude::*;
