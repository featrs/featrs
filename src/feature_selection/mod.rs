//! Feature selection transformers.
//!
//! Analogous to `sklearn.feature_selection`. Reduce the number of features
//! by removing low-variance columns or selecting the top-k features according
//! to a statistical test.

pub mod correlation_threshold;
pub mod select_kbest;
pub mod select_percentile;
pub mod variance_threshold;

pub use correlation_threshold::CorrelationThreshold;
pub use select_kbest::SelectKBest;
pub use select_percentile::SelectPercentile;
pub use variance_threshold::VarianceThreshold;
