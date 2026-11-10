//! Schema-driven automatic preprocessing pipeline.
//!
//! [`AutoPipeline`] inspects the input [`DataFrame`] once, decides how every
//! column should be treated, assembles the matching cleaning, encoding and
//! scaling steps into a [`Pipeline`], and exposes that fitted pipeline for
//! inspection. It is the "one call, sensible defaults" entry point of the
//! crate: instead of hand-wiring a [`ColumnTransformer`]
//! for every dataset, callers describe policy once and reuse the result.
//!
//! # What is decided from the data
//!
//! | Column kind | Treatment |
//! |---|---|
//! | all-null | dropped (always) |
//! | constant (≤ 1 non-null distinct value) | dropped when [`remove_constant`](AutoPipelineConfig::remove_constant) |
//! | exact duplicate of an earlier column | dropped when [`remove_duplicates`](AutoPipelineConfig::remove_duplicates) |
//! | `Date` / `Datetime` | component columns added, originals dropped, when [`detect_datetime`](AutoPipelineConfig::detect_datetime) |
//! | `String`, low cardinality | [`StringCleaner`] when configured, [`RareCategoryGrouper`] when configured, one-hot encoded |
//! | `String`, high cardinality | [`StringCleaner`], optional grouper, then target encoding (classification target) or feature hashing |
//! | `String` that looks numeric | forwarded unchanged as `String` (matching [`AutoTypeDetector`]) |
//! | numeric dtype | [`AutoImputer`] when configured, then [`AutoScaler`] when configured |
//! | anything else (`Boolean`, nested …) | forwarded unchanged |
//!
//! The target column named by [`target`](AutoPipelineConfig::target) is never
//! part of the output. It is only used, and only required, when a high
//! cardinality `String` column is target-encoded: that path needs a
//! classification target, detected by dtype (`Int*`, `UInt*` or `Boolean`) and
//! cast to `Float64` for [`TargetEncoder`]. A `Float32`/`Float64` target is
//! treated as continuous, so high-cardinality strings are hashed instead.
//!
//! # Schema contract
//!
//! `fit` freezes the feature columns and dtypes it saw (after removing the
//! target). `transform` requires exactly those columns with those dtypes and
//! rejects missing, retyped and extra columns with
//! [`Error::InvalidInput`]. A model fitted
//! on `(age, city)` therefore cannot silently score a frame that renamed
//! `city` or switched it to `Int64`.
//!
//! # Example
//!
//! ```rust
//! use featrs::automation::auto_pipeline::{
//!     AutoPipeline, AutoPipelineConfig, AutoScalerConfig,
//! };
//! use featrs::traits::{Fit, Transform};
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let x = DataFrame::new(
//!     3,
//!     vec![Column::from(Series::new(
//!         "height".into(),
//!         &[1.0_f64, 2.0, 3.0],
//!     ))],
//! )?;
//!
//! let mut auto = AutoPipeline::new(
//!     AutoPipelineConfig::default().scaler(AutoScalerConfig::default()),
//! );
//! auto.fit(x.clone())?;
//! let out = auto.transform(x)?;
//!
//! assert_eq!(out.get_column_names(), vec!["height"]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{HashMap, HashSet};

use polars::prelude::*;

use crate::pipeline::column_transformer::Remainder;
use crate::pipeline::{ColumnTransformer, DataFrameTransformer, Pipeline};
use crate::preprocessing::auto_imputer::{AutoImputer, NullColumnBehavior};
use crate::preprocessing::auto_scaler::{AutoScaler, ScalingStrategy};
use crate::preprocessing::auto_type::{AutoTypeDetector, ColumnType};
use crate::preprocessing::constant_column_remover::ConstantColumnRemover;
use crate::preprocessing::datetime_features::DatetimeFeatures;
use crate::preprocessing::duplicate_column_remover::DuplicateColumnRemover;
use crate::preprocessing::encoder::OneHotEncoder;
use crate::preprocessing::encoder_target::TargetEncoder;
use crate::preprocessing::feature_hasher::FeatureHasher;
use crate::preprocessing::imputer::Strategy;
use crate::preprocessing::rare_category_grouper::{RareCategoryGrouper, Threshold};
use crate::preprocessing::string_cleaner::StringCleaner;
use crate::traits::output_names::{ensure_fitted, identity_names};
use crate::traits::schema::{SchemaSpec, validate_schema};
use crate::traits::{Error, FeatureNamesOut, Fit, FitSupervised, Result, Transform};

/// Configuration for the [`AutoScaler`] step.
///
/// Mirrors the builder of [`AutoScaler`]. Kept as plain data so
/// [`AutoPipelineConfig`] stays cloneable and a pipeline can be re-fitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoScalerConfig {
    /// Scaling strategy passed to [`AutoScaler::strategy`].
    pub strategy: ScalingStrategy,
}

impl Default for AutoScalerConfig {
    fn default() -> Self {
        Self {
            strategy: ScalingStrategy::Auto,
        }
    }
}

impl AutoScalerConfig {
    /// Create a config using [`ScalingStrategy::Auto`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Force one scaling strategy instead of the per-column heuristic.
    pub fn strategy(mut self, strategy: ScalingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    fn build(&self) -> AutoScaler {
        AutoScaler::new().strategy(self.strategy)
    }
}

/// Configuration for the [`AutoImputer`] step.
///
/// Mirrors the builders of [`AutoImputer`]. Kept as plain data so
/// [`AutoPipelineConfig`] stays cloneable and a pipeline can be re-fitted.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoImputerConfig {
    /// Force one imputation strategy for every imputed column.
    pub force_strategy: Option<Strategy>,
    /// Null fraction at or above which a column is left unchanged
    /// (default `0.5`).
    pub high_null_threshold: f64,
    /// `|skewness|` above which the median is used instead of the mean
    /// (default `1.0`).
    pub skew_threshold: f64,
    /// How to handle a `Float64` column that is entirely missing.
    pub null_column_behavior: NullColumnBehavior,
}

impl Default for AutoImputerConfig {
    fn default() -> Self {
        Self {
            force_strategy: None,
            high_null_threshold: 0.5,
            skew_threshold: 1.0,
            null_column_behavior: NullColumnBehavior::Drop,
        }
    }
}

impl AutoImputerConfig {
    /// Create a config using the per-column heuristic.
    pub fn new() -> Self {
        Self::default()
    }

    /// Force one [`Strategy`] for every imputed column.
    pub fn force_strategy(mut self, strategy: Strategy) -> Self {
        self.force_strategy = Some(strategy);
        self
    }

    /// Set the null fraction at or above which a column is left unchanged.
    pub fn high_null_threshold(mut self, threshold: f64) -> Self {
        self.high_null_threshold = threshold;
        self
    }

    /// Set the `|skewness|` above which the median is used.
    pub fn skew_threshold(mut self, threshold: f64) -> Self {
        self.skew_threshold = threshold;
        self
    }

    /// Set how an entirely missing `Float64` column is handled.
    pub fn null_column_behavior(mut self, behavior: NullColumnBehavior) -> Self {
        self.null_column_behavior = behavior;
        self
    }

    fn build(&self) -> AutoImputer {
        let mut imputer = AutoImputer::new()
            .high_null_threshold(self.high_null_threshold)
            .skew_threshold(self.skew_threshold)
            .null_column_behavior(self.null_column_behavior);
        if let Some(strategy) = self.force_strategy {
            imputer = imputer.force_strategy(strategy);
        }
        imputer
    }
}

/// Configuration for the [`RareCategoryGrouper`] step.
///
/// Mirrors the builders of [`RareCategoryGrouper`]. Kept as plain data so
/// [`AutoPipelineConfig`] stays cloneable and a pipeline can be re-fitted.
#[derive(Debug, Clone, PartialEq)]
pub struct RareCategoryGrouperConfig {
    /// Categories below this threshold are relabeled (default
    /// [`Threshold::MinCount`]).
    pub threshold: Threshold,
    /// Label used for rare and unseen categories (default `"Other"`).
    pub other_label: String,
    /// Whether nulls are relabeled too (default `false`).
    pub treat_null_as_rare: bool,
}

impl Default for RareCategoryGrouperConfig {
    fn default() -> Self {
        Self {
            threshold: Threshold::MinCount(5),
            other_label: "Other".to_string(),
            treat_null_as_rare: false,
        }
    }
}

impl RareCategoryGrouperConfig {
    /// Create a config with the grouper defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rarity threshold.
    pub fn threshold(mut self, threshold: Threshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the label for rare and unseen categories.
    pub fn other_label(mut self, label: impl Into<String>) -> Self {
        self.other_label = label.into();
        self
    }

    /// Relabel nulls along with rare categories.
    pub fn treat_null_as_rare(mut self, treat: bool) -> Self {
        self.treat_null_as_rare = treat;
        self
    }

    fn build(&self, columns: &[String]) -> RareCategoryGrouper {
        let refs = column_refs(columns);
        RareCategoryGrouper::new()
            .columns(&refs)
            .threshold(self.threshold)
            .other_label(&self.other_label)
            .treat_null_as_rare(self.treat_null_as_rare)
    }
}

/// Policy for [`AutoPipeline`].
///
/// The defaults are deliberately conservative: only all-null, constant and
/// datetime columns are touched. Enable scaling, imputation or string cleanup
/// explicitly (or use [`AutoPipeline::from_inferred_schema`]).
#[derive(Debug, Clone, PartialEq)]
pub struct AutoPipelineConfig {
    /// Column to use as the supervised target. Never emitted by `transform`.
    pub target: Option<String>,
    /// `String` columns with fewer distinct non-null values than this are
    /// one-hot encoded; the rest are high cardinality (default `20`).
    pub cat_threshold: usize,
    /// Number of hashed output columns for high-cardinality strings
    /// (default `100`).
    pub hash_buckets: usize,
    /// Expand `Date`/`Datetime` columns into component columns
    /// (default `true`).
    pub detect_datetime: bool,
    /// Drop columns with a single distinct non-null value (default `true`).
    pub remove_constant: bool,
    /// Drop columns that exactly duplicate an earlier column
    /// (default `false`).
    pub remove_duplicates: bool,
    /// Trim and collapse whitespace in `String` columns before encoding
    /// (default `false`).
    pub normalize_strings: bool,
    /// Optional rare-category grouping before encoding.
    pub rare_category_grouper: Option<RareCategoryGrouperConfig>,
    /// Optional automatic scaling of numeric columns.
    pub scaler: Option<AutoScalerConfig>,
    /// Optional automatic imputation of numeric columns.
    pub imputer: Option<AutoImputerConfig>,
}

impl Default for AutoPipelineConfig {
    fn default() -> Self {
        Self {
            target: None,
            cat_threshold: 20,
            hash_buckets: 100,
            detect_datetime: true,
            remove_constant: true,
            remove_duplicates: false,
            normalize_strings: false,
            rare_category_grouper: None,
            scaler: None,
            imputer: None,
        }
    }
}

impl AutoPipelineConfig {
    /// Create the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the supervised target column.
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set the one-hot cardinality threshold.
    pub fn cat_threshold(mut self, threshold: usize) -> Self {
        self.cat_threshold = threshold;
        self
    }

    /// Set the number of feature-hashing buckets.
    pub fn hash_buckets(mut self, buckets: usize) -> Self {
        self.hash_buckets = buckets;
        self
    }

    /// Enable or disable datetime component extraction.
    pub fn detect_datetime(mut self, detect: bool) -> Self {
        self.detect_datetime = detect;
        self
    }

    /// Enable or disable constant-column removal.
    pub fn remove_constant(mut self, remove: bool) -> Self {
        self.remove_constant = remove;
        self
    }

    /// Enable or disable duplicate-column removal.
    pub fn remove_duplicates(mut self, remove: bool) -> Self {
        self.remove_duplicates = remove;
        self
    }

    /// Enable or disable string whitespace normalization.
    pub fn normalize_strings(mut self, normalize: bool) -> Self {
        self.normalize_strings = normalize;
        self
    }

    /// Add rare-category grouping before encoding.
    pub fn rare_category_grouper(mut self, config: RareCategoryGrouperConfig) -> Self {
        self.rare_category_grouper = Some(config);
        self
    }

    /// Add automatic scaling of numeric columns.
    pub fn scaler(mut self, config: AutoScalerConfig) -> Self {
        self.scaler = Some(config);
        self
    }

    /// Add automatic imputation of numeric columns.
    pub fn imputer(mut self, config: AutoImputerConfig) -> Self {
        self.imputer = Some(config);
        self
    }
}

/// Preprocessing pipeline assembled from an inferred data schema.
///
/// Call [`fit`](Fit::fit) once on training data, then
/// [`transform`](Transform::transform) any frame with the same feature schema.
/// The assembled steps are available through
/// [`prepared_pipeline`](AutoPipeline::prepared_pipeline).
///
/// `AutoPipeline` owns the assembled [`Pipeline`], so the pipeline is refitted
/// as a unit and its steps keep their configuration across re-fits.
pub struct AutoPipeline {
    fitted: bool,
    config: AutoPipelineConfig,
    inner: Option<Pipeline>,
    /// Feature columns and dtypes seen at fit, in input order (target removed).
    feature_schema: Vec<(String, DataType)>,
}

impl AutoPipeline {
    /// Create an unfitted pipeline from `config`.
    ///
    /// `config` is not validated here; invalid values (for example
    /// [`hash_buckets`](AutoPipelineConfig::hash_buckets) = 0) are reported by
    /// [`fit`](Fit::fit).
    pub fn new(config: AutoPipelineConfig) -> Self {
        Self {
            fitted: false,
            config,
            inner: None,
            feature_schema: Vec::new(),
        }
    }

    /// Build a pipeline whose policy is inferred from `df`.
    ///
    /// The returned pipeline uses the default configuration plus automatic
    /// imputation when a numeric column has missing values and automatic
    /// scaling when the frame has at least one numeric column.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when `df` has 0 rows or 0 columns.
    pub fn from_inferred_schema(df: &DataFrame) -> Result<Self> {
        if df.height() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline::from_inferred_schema: the DataFrame has 0 rows.".to_string(),
            ));
        }
        if df.width() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline::from_inferred_schema: the DataFrame has 0 columns.".to_string(),
            ));
        }

        let mut config = AutoPipelineConfig::default();
        let numeric = df
            .columns()
            .iter()
            .filter(|column| is_numeric_dtype(column.dtype()));
        let mut has_numeric = false;
        let mut has_missing_numeric = false;
        for column in numeric {
            has_numeric = true;
            if column.null_count() > 0 {
                has_missing_numeric = true;
            }
        }
        if has_missing_numeric {
            config.imputer = Some(AutoImputerConfig::default());
        }
        if has_numeric {
            config.scaler = Some(AutoScalerConfig::default());
        }
        Ok(Self::new(config))
    }

    /// The policy this pipeline was built with.
    pub fn config(&self) -> &AutoPipelineConfig {
        &self.config
    }

    /// Whether [`fit`](Fit::fit) has completed successfully.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// The assembled, fitted pipeline, for inspection or reuse.
    ///
    /// Before a successful `fit` this returns [`Error::NotFitted`]; there is no
    /// assembled pipeline to inspect yet.
    pub fn prepared_pipeline(&self) -> Result<&Pipeline> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "AutoPipeline has not been fitted. Call .fit(dataframe) before .prepared_pipeline()."
                    .to_string(),
            ));
        }
        self.inner.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "AutoPipeline has not been fitted. Call .fit(dataframe) before .prepared_pipeline()."
                    .to_string(),
            )
        })
    }

    /// Remove the target column if it is present, leaving the features.
    fn strip_target(&self, x: DataFrame) -> Result<DataFrame> {
        let Some(target) = self.config.target.as_deref() else {
            return Ok(x);
        };
        if !x
            .get_column_names()
            .iter()
            .any(|name| name.as_str() == target)
        {
            return Ok(x);
        }
        let keep: Vec<&str> = x
            .get_column_names()
            .iter()
            .filter(|name| name.as_str() != target)
            .map(|name| name.as_str())
            .collect();
        x.select(keep).map_err(|error| {
            Error::Computation(format!(
                "AutoPipeline: failed to remove target column '{target}': {error}"
            ))
        })
    }

    /// Build the step list for a feature frame, using `y` for target encoding.
    fn assemble(
        &self,
        features: &DataFrame,
        y: Option<&DataFrame>,
        classification: bool,
    ) -> Result<Pipeline> {
        let config = &self.config;
        let height = features.height();
        let all_names: Vec<String> = features
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();

        // Columns that carry no information and cannot be encoded: all null.
        let all_null: Vec<String> = features
            .columns()
            .iter()
            .filter(|column| column.null_count() == height)
            .map(|column| column.name().to_string())
            .collect();
        let after_null: Vec<String> = all_names
            .iter()
            .filter(|name| !all_null.contains(name))
            .cloned()
            .collect();
        if after_null.is_empty() {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: every feature column is entirely null; no columns remain to model."
                    .to_string(),
            ));
        }
        let working = select_columns(features, &after_null, "AutoPipeline.fit")?;

        // Constant columns are a fit-time decision, not a transform-time one.
        let constant: Vec<String> = if config.remove_constant {
            let mut constant = Vec::new();
            for column in working.columns() {
                if is_constant(column)? {
                    constant.push(column.name().to_string());
                }
            }
            if constant.len() == working.width() {
                return Err(Error::InvalidInput(format!(
                    "AutoPipeline.fit: all {} feature column(s) are constant (≤ 1 distinct non-null value). No columns remain to model.",
                    working.width()
                )));
            }
            constant
        } else {
            Vec::new()
        };

        let survivors_after_constant: Vec<String> = working
            .get_column_names()
            .iter()
            .filter(|name| {
                !constant
                    .iter()
                    .any(|constant| constant.as_str() == name.as_str())
            })
            .map(|name| name.to_string())
            .collect();
        let after_constant =
            select_columns(&working, &survivors_after_constant, "AutoPipeline.fit")?;

        // Duplicate removal is probed on the post-constant frame so the plan
        // names the columns that actually reach the encoder.
        let mut deduplicated = false;
        let mut planned_names = survivors_after_constant.clone();
        if config.remove_duplicates {
            let mut remover = DuplicateColumnRemover::new();
            remover.fit(after_constant.clone())?;
            let names: Vec<String> = after_constant
                .get_column_names()
                .iter()
                .map(|name| name.to_string())
                .collect();
            let survivors = remover.feature_names_out(&names)?;
            if survivors.len() < names.len() {
                deduplicated = true;
                planned_names = survivors;
            }
        }
        let planned = select_columns(&after_constant, &planned_names, "AutoPipeline.fit")?;

        let string_columns: Vec<String> = planned
            .columns()
            .iter()
            .filter(|column| matches!(column.dtype(), DataType::String))
            .map(|column| column.name().to_string())
            .collect();

        // Column classification reuses the crate's detector: it owns the
        // numeric-looking-string and cardinality-threshold rules. When strings
        // are normalized, classify the cleaned values so trimming cannot push
        // a column over the cardinality threshold.
        let detection_frame = if config.normalize_strings && !string_columns.is_empty() {
            let refs = column_refs(&string_columns);
            let mut cleaner = StringCleaner::new()
                .columns(&refs)
                .trim()
                .collapse_internal_ws();
            cleaner.fit(planned.clone())?;
            cleaner.transform(planned.clone())?
        } else {
            planned.clone()
        };
        let mut detector = AutoTypeDetector::new()
            .cat_threshold(config.cat_threshold)
            .hash_buckets(config.hash_buckets);
        detector.fit(detection_frame)?;
        let detected: HashMap<String, ColumnType> = detector
            .column_types()
            .unwrap_or(&[])
            .iter()
            .map(|(name, kind)| (name.clone(), kind.clone()))
            .collect();

        let mut numeric = Vec::new();
        let mut datetime = Vec::new();
        let mut low_cardinality = Vec::new();
        let mut high_cardinality = Vec::new();

        for column in planned.columns() {
            let name = column.name().to_string();
            let dtype = column.dtype();
            if is_datetime_dtype(dtype) {
                if config.detect_datetime {
                    datetime.push(name);
                }
                continue;
            }
            if is_numeric_dtype(dtype) {
                numeric.push(name);
                continue;
            }
            if !matches!(dtype, DataType::String) {
                continue;
            }
            match detected.get(&name) {
                Some(ColumnType::Categorical) => low_cardinality.push(name),
                Some(ColumnType::HighCardinality) => high_cardinality.push(name),
                // Numeric-looking strings are forwarded unchanged; anything
                // the detector left undecided is a passthrough too.
                _ => {}
            }
        }

        let use_target_encoder =
            config.target.is_some() && classification && !high_cardinality.is_empty();

        let mut steps: Vec<(String, Box<dyn DataFrameTransformer>)> = Vec::new();
        if !all_null.is_empty() {
            steps.push((
                "drop_all_null".to_string(),
                Box::new(DropColumns::new(all_null)),
            ));
        }
        if !constant.is_empty() {
            steps.push((
                "remove_constant".to_string(),
                Box::new(ConstantColumnRemover::new()),
            ));
        }
        if deduplicated {
            steps.push((
                "remove_duplicates".to_string(),
                Box::new(DuplicateColumnRemover::new()),
            ));
        }
        if !datetime.is_empty() {
            let refs = column_refs(&datetime);
            steps.push((
                "datetime_features".to_string(),
                Box::new(DatetimeFeatures::new().columns(&refs)),
            ));
            steps.push((
                "drop_datetime_sources".to_string(),
                Box::new(DropColumns::new(datetime)),
            ));
        }
        if config.normalize_strings && !string_columns.is_empty() {
            let refs = column_refs(&string_columns);
            steps.push((
                "clean_strings".to_string(),
                Box::new(
                    StringCleaner::new()
                        .columns(&refs)
                        .trim()
                        .collapse_internal_ws(),
                ),
            ));
        }
        if let Some(grouper) = &config.rare_category_grouper {
            let encoded: Vec<String> = low_cardinality
                .iter()
                .chain(high_cardinality.iter())
                .cloned()
                .collect();
            if !encoded.is_empty() {
                steps.push((
                    "group_rare_categories".to_string(),
                    Box::new(grouper.build(&encoded)),
                ));
            }
        }
        if use_target_encoder {
            let y = y.ok_or_else(|| {
                Error::InvalidInput(
                    "AutoPipeline.fit: target encoding requires the configured target column."
                        .to_string(),
                )
            })?;
            let target_series = y
                .columns()
                .first()
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "AutoPipeline.fit: target encoding requires a non-empty target."
                            .to_string(),
                    )
                })?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(|error| {
                    Error::InvalidInput(format!(
                        "AutoPipeline.fit: the target must be numeric for target encoding: {error}"
                    ))
                })?;
            let target_frame = DataFrame::new(y.height(), vec![Column::from(target_series)])
                .map_err(|error| {
                    Error::Computation(format!(
                        "AutoPipeline.fit: failed to prepare the target for target encoding: {error}"
                    ))
                })?;
            steps.push((
                "target_encode_high_cardinality".to_string(),
                Box::new(TargetEncoderStep::new(
                    high_cardinality.clone(),
                    target_frame,
                )),
            ));
        }

        let mut entries: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)> = Vec::new();
        if !numeric.is_empty() && (config.imputer.is_some() || config.scaler.is_some()) {
            let mut numeric_steps: Vec<(String, Box<dyn DataFrameTransformer>)> = Vec::new();
            if let Some(imputer) = &config.imputer {
                numeric_steps.push(("imputer".to_string(), Box::new(imputer.build())));
            }
            if let Some(scaler) = &config.scaler {
                numeric_steps.push(("scaler".to_string(), Box::new(scaler.build())));
            }
            entries.push((
                "numeric".to_string(),
                Box::new(Pipeline::new(numeric_steps)?),
                numeric,
            ));
        }
        if !low_cardinality.is_empty() {
            entries.push((
                "categorical".to_string(),
                Box::new(OneHotEncoder::new()),
                low_cardinality,
            ));
        }
        if !high_cardinality.is_empty() && !use_target_encoder {
            let refs = column_refs(&high_cardinality);
            entries.push((
                "high_cardinality".to_string(),
                Box::new(FeatureHasher::new(&refs, config.hash_buckets)),
                high_cardinality,
            ));
        }
        if !entries.is_empty() {
            steps.push((
                "column_transformers".to_string(),
                Box::new(ColumnTransformer::new(entries, Remainder::Passthrough)),
            ));
        }

        if steps.is_empty() {
            steps.push(("identity".to_string(), Box::new(Identity::new())));
        }
        Pipeline::new(steps)
    }
}

impl Fit<DataFrame> for AutoPipeline {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first: a failed re-fit must not leave the previous plan
        // usable with the new data.
        self.fitted = false;
        self.inner = None;
        self.feature_schema.clear();

        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: the DataFrame has 0 rows.".to_string(),
            ));
        }
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: the DataFrame has 0 columns.".to_string(),
            ));
        }
        if self.config.hash_buckets == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: hash_buckets must be at least 1.".to_string(),
            ));
        }
        if let Some(target) = self.config.target.as_deref()
            && target.trim().is_empty()
        {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: the target column name must not be empty.".to_string(),
            ));
        }

        let (features, y, classification) = if let Some(target) = self.config.target.clone() {
            let target_column = x.column(target.as_str()).map_err(|_| {
                Error::InvalidInput(format!(
                    "AutoPipeline.fit: target column '{target}' is not present in the DataFrame."
                ))
            })?;
            let classification = target_column.dtype().is_integer()
                || matches!(target_column.dtype(), DataType::Boolean);
            let y = DataFrame::new(
                x.height(),
                vec![Column::from(target_column.as_materialized_series().clone())],
            )
            .map_err(|error| {
                Error::Computation(format!(
                    "AutoPipeline.fit: failed to isolate target column '{target}': {error}"
                ))
            })?;
            let keep: Vec<&str> = x
                .get_column_names()
                .iter()
                .filter(|name| name.as_str() != target)
                .map(|name| name.as_str())
                .collect();
            let features = x.select(keep).map_err(|error| {
                Error::Computation(format!(
                    "AutoPipeline.fit: failed to remove target column '{target}': {error}"
                ))
            })?;
            (features, Some(y), classification)
        } else {
            (x, None, false)
        };

        if features.width() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline.fit: no feature columns remain after removing the target."
                    .to_string(),
            ));
        }

        self.feature_schema = features
            .columns()
            .iter()
            .map(|column| (column.name().to_string(), column.dtype().clone()))
            .collect();

        let mut pipeline = self.assemble(&features, y.as_ref(), classification)?;
        pipeline.fit(features)?;
        self.inner = Some(pipeline);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for AutoPipeline {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let pipeline = self.prepared_pipeline()?;
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "AutoPipeline.transform: the DataFrame has 0 rows.".to_string(),
            ));
        }
        let features = self.strip_target(x)?;

        // Freeze the feature schema: missing or retyped columns are drift.
        let pairs: Vec<(&str, DataType)> = self
            .feature_schema
            .iter()
            .map(|(name, dtype)| (name.as_str(), dtype.clone()))
            .collect();
        validate_schema(&SchemaSpec::exact(&pairs), &features)?;

        let expected: HashSet<&str> = self
            .feature_schema
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        for name in features.get_column_names() {
            if !expected.contains(name.as_str()) {
                return Err(Error::InvalidInput(format!(
                    "AutoPipeline.transform: unexpected column '{name}'. The feature schema was frozen at fit time."
                )));
            }
        }

        pipeline.transform(features)
    }
}

impl FeatureNamesOut for AutoPipeline {
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        let pipeline = self.prepared_pipeline()?;
        let feature_columns: Vec<String> = match self.config.target.as_deref() {
            Some(target) => input_columns
                .iter()
                .filter(|name| name.as_str() != target)
                .cloned()
                .collect(),
            None => input_columns.to_vec(),
        };
        pipeline.feature_names_out(&feature_columns)
    }
}

/// Pass-through step used when no other transformation applies.
struct Identity {
    fitted: bool,
}

impl Identity {
    fn new() -> Self {
        Self { fitted: false }
    }
}

impl Fit<DataFrame> for Identity {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "Identity.fit: the DataFrame has 0 rows.".to_string(),
            ));
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Identity {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        ensure_fitted(self.fitted, "Identity")?;
        Ok(x)
    }
}

impl FeatureNamesOut for Identity {
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        identity_names(self.fitted, "Identity", input_columns)
    }
}

/// Remove a fixed set of columns, preserving the order of the survivors.
struct DropColumns {
    columns: Vec<String>,
    fitted: bool,
}

impl DropColumns {
    fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            fitted: false,
        }
    }
}

impl Fit<DataFrame> for DropColumns {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        let present: HashSet<&str> = x
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect();
        for column in &self.columns {
            if !present.contains(column.as_str()) {
                return Err(Error::InvalidInput(format!(
                    "DropColumns.fit: column '{column}' is not present in the input DataFrame."
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for DropColumns {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        ensure_fitted(self.fitted, "DropColumns")?;
        let keep: Vec<&str> = x
            .get_column_names()
            .iter()
            .filter(|name| {
                !self
                    .columns
                    .iter()
                    .any(|column| column.as_str() == name.as_str())
            })
            .map(|name| name.as_str())
            .collect();
        x.select(keep).map_err(|error| {
            Error::Computation(format!(
                "DropColumns.transform: failed to select columns: {error}"
            ))
        })
    }
}

impl FeatureNamesOut for DropColumns {
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        ensure_fitted(self.fitted, "DropColumns")?;
        Ok(input_columns
            .iter()
            .filter(|name| !self.columns.contains(name))
            .cloned()
            .collect())
    }
}

/// Adapter that makes [`TargetEncoder`] (a `FitSupervised` transformer) usable
/// as a sequential [`Pipeline`] step on the whole feature frame.
struct TargetEncoderStep {
    encoder: TargetEncoder,
    columns: Vec<String>,
    target: DataFrame,
    fitted: bool,
}

impl TargetEncoderStep {
    fn new(columns: Vec<String>, target: DataFrame) -> Self {
        Self {
            encoder: TargetEncoder::new(),
            columns,
            target,
            fitted: false,
        }
    }
}

impl Fit<DataFrame> for TargetEncoderStep {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.fitted = false;
        let subset = select_columns(&x, &self.columns, "TargetEncoderStep.fit")?;
        self.encoder.fit(subset, self.target.clone())?;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for TargetEncoderStep {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        ensure_fitted(self.fitted, "TargetEncoderStep")?;
        let subset = select_columns(&x, &self.columns, "TargetEncoderStep.transform")?;
        let encoded = self.encoder.transform(subset)?;
        let mut out = x;
        for column in encoded.columns() {
            let name = column.name().to_string();
            out.replace(name.as_str(), column.clone())
                .map_err(|error| {
                    Error::Computation(format!(
                        "TargetEncoderStep.transform: failed to replace column '{name}': {error}"
                    ))
                })?;
        }
        Ok(out)
    }
}

impl FeatureNamesOut for TargetEncoderStep {
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        identity_names(self.fitted, "TargetEncoderStep", input_columns)
    }
}

/// `true` for the numeric dtypes `AutoScaler` and `AutoImputer` understand.
fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
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

/// `true` for `Date` and `Datetime` columns.
fn is_datetime_dtype(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Date | DataType::Datetime(..))
}

/// A column is constant when it has at most one distinct non-null value.
fn is_constant(column: &Column) -> Result<bool> {
    let mut distinct = column.n_unique().map_err(|error| {
        Error::Computation(format!(
            "AutoPipeline.fit: failed to count unique values in column '{}': {error}",
            column.name()
        ))
    })?;
    if column.null_count() > 0 {
        // `n_unique` counts null as one distinct value.
        distinct = distinct.saturating_sub(1);
    }
    Ok(distinct <= 1)
}

fn column_refs(columns: &[String]) -> Vec<&str> {
    columns.iter().map(String::as_str).collect()
}

fn select_columns(df: &DataFrame, columns: &[String], who: &str) -> Result<DataFrame> {
    let refs = column_refs(columns);
    df.select(refs)
        .map_err(|error| Error::Computation(format!("{who}: failed to select columns: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{NamedFrom, Series};

    fn frame(columns: Vec<Column>) -> DataFrame {
        let height = columns.first().map(|column| column.len()).unwrap_or(0);
        DataFrame::new(height, columns).unwrap()
    }

    fn f64_column(name: &str, values: &[f64]) -> Column {
        Column::from(Series::new(name.into(), values))
    }

    fn str_column(name: &str, values: &[&str]) -> Column {
        Column::from(Series::new(name.into(), values))
    }

    #[test]
    fn all_numeric_with_scaler_matches_manual_pipeline() {
        let x = frame(vec![
            f64_column("a", &[1.0, 2.0, 3.0, 4.0]),
            f64_column("b", &[10.0, 20.0, 30.0, 40.0]),
        ]);

        let mut manual = Pipeline::new(vec![(
            "scaler".to_string(),
            Box::new(AutoScaler::new()) as Box<dyn DataFrameTransformer>,
        )])
        .unwrap();
        manual.fit(x.clone()).unwrap();
        let expected = manual.transform(x.clone()).unwrap();

        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().scaler(AutoScalerConfig::default()));
        auto.fit(x.clone()).unwrap();
        let actual = auto.transform(x).unwrap();

        assert!(actual.equals_missing(&expected));
        assert_eq!(
            auto.prepared_pipeline().unwrap().steps().len(),
            1,
            "only the column transformer should be needed"
        );
    }

    #[test]
    fn mixed_numeric_and_categorical_uses_one_hot() {
        let x = frame(vec![
            f64_column("height", &[1.0, 2.0, 3.0, 4.0]),
            str_column("city", &["a", "b", "a", "b"]),
        ]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().scaler(AutoScalerConfig::default()));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        let names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert!(names.contains(&"height".to_string()));
        assert!(names.contains(&"city_a".to_string()));
        assert!(names.contains(&"city_b".to_string()));
        assert_eq!(out.width(), 3);
    }

    #[test]
    fn low_cardinality_strings_are_one_hot_encoded() {
        let x = frame(vec![str_column("city", &["a", "b", "a", "c"])]);
        // Exactly 3 distinct values vs threshold 3: not below the threshold,
        // so this is high-cardinality and hashed.
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().cat_threshold(3));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        assert!(out.get_column_names()[0].starts_with("hashed_"));

        // Exactly 2 distinct values is below the threshold: one-hot.
        let x = frame(vec![str_column("city", &["a", "b", "a", "b"])]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().cat_threshold(3));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        let names: Vec<&str> = out
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect();
        assert_eq!(names, vec!["city_a", "city_b"]);
    }

    #[test]
    fn classification_target_encodes_high_cardinality_strings() {
        let x = frame(vec![
            str_column("city", &["a", "b", "c", "d", "e", "f"]),
            Column::from(Series::new("y".into(), &[0_i32, 1, 0, 1, 0, 1])),
        ]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().target("y").cat_threshold(3));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        let names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert_eq!(names, vec!["city"]);
        assert_eq!(out.column("city").unwrap().dtype(), &DataType::Float64);
        assert!(
            !names.iter().any(|name| name.starts_with("hashed_")),
            "target encoding replaces feature hashing"
        );
    }

    #[test]
    fn continuous_target_hashes_high_cardinality_strings() {
        let x = frame(vec![
            str_column("city", &["a", "b", "c", "d", "e", "f"]),
            f64_column("y", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ]);
        let mut auto = AutoPipeline::new(
            AutoPipelineConfig::default()
                .target("y")
                .cat_threshold(3)
                .hash_buckets(8),
        );
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        assert_eq!(out.width(), 8);
        let names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert!(names[0].starts_with("hashed_"));
        assert!(!names.contains(&"y".to_string()));
    }

    #[test]
    fn default_config_is_a_passthrough_for_plain_numeric_data() {
        let x = frame(vec![
            f64_column("a", &[1.0, 2.0, 3.0]),
            f64_column("b", &[4.0, 5.0, 6.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x.clone()).unwrap();

        assert!(out.equals_missing(&x));
        assert_eq!(auto.prepared_pipeline().unwrap().steps()[0].0, "identity");
        let names: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn all_null_columns_are_dropped() {
        let x = frame(vec![
            Column::from(Series::new("empty".into(), &[None::<f64>, None, None])),
            f64_column("a", &[1.0, 2.0, 3.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        assert_eq!(out.get_column_names().len(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "a");
    }

    #[test]
    fn all_null_features_error() {
        let x = frame(vec![Column::from(Series::new(
            "empty".into(),
            &[None::<f64>, None, None],
        ))]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        let error = auto.fit(x).unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    #[test]
    fn datetime_features_replace_the_source_column() {
        let dates = Series::new(
            "when".into(),
            &[
                19_723_i32, // 2024-01-01
                19_724, 19_725,
            ],
        )
        .cast(&DataType::Date)
        .unwrap();
        let x = frame(vec![Column::from(dates), f64_column("a", &[1.0, 2.0, 3.0])]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        let names: Vec<&str> = out
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "when_year", "when_month", "when_day"]);
    }

    #[test]
    fn normalize_strings_trims_before_encoding() {
        let x = frame(vec![str_column("city", &["  a ", "b", "a"])]);
        let mut auto = AutoPipeline::new(
            AutoPipelineConfig::default()
                .cat_threshold(3)
                .normalize_strings(true),
        );
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        let names: Vec<&str> = out
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect();
        assert_eq!(names, vec!["city_a", "city_b"]);
    }

    #[test]
    fn rare_category_grouper_runs_before_encoding() {
        let x = frame(vec![str_column("city", &["a", "a", "b", "c"])]);
        let mut auto = AutoPipeline::new(
            AutoPipelineConfig::default()
                .cat_threshold(10)
                .rare_category_grouper(
                    RareCategoryGrouperConfig::default().threshold(Threshold::MinCount(2)),
                ),
        );
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        let mut names: Vec<&str> = out
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(names, vec!["city_Other", "city_a"]);
    }

    #[test]
    fn schema_drift_is_rejected() {
        let x = frame(vec![
            f64_column("a", &[1.0, 2.0, 3.0]),
            f64_column("b", &[4.0, 5.0, 6.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        auto.fit(x.clone()).unwrap();

        let missing = frame(vec![f64_column("a", &[1.0, 2.0, 3.0])]);
        assert!(matches!(
            auto.transform(missing).unwrap_err(),
            Error::InvalidInput(_)
        ));

        let extra = frame(vec![
            f64_column("a", &[1.0, 2.0, 3.0]),
            f64_column("b", &[4.0, 5.0, 6.0]),
            f64_column("c", &[7.0, 8.0, 9.0]),
        ]);
        let error = auto.transform(extra).unwrap_err();
        assert!(
            error.to_string().contains("unexpected column 'c'"),
            "unexpected error: {error}"
        );

        let retyped = frame(vec![
            Column::from(Series::new("a".into(), &[1_i64, 2, 3])),
            f64_column("b", &[4.0, 5.0, 6.0]),
        ]);
        assert!(matches!(
            auto.transform(retyped).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn not_fitted_errors() {
        let auto = AutoPipeline::new(AutoPipelineConfig::default());
        let x = frame(vec![f64_column("a", &[1.0, 2.0, 3.0])]);
        let names: Vec<String> = vec!["a".to_string()];

        assert!(matches!(
            auto.transform(x).unwrap_err(),
            Error::NotFitted(_)
        ));
        assert!(matches!(auto.prepared_pipeline(), Err(Error::NotFitted(_))));
        assert!(matches!(
            auto.feature_names_out(&names).unwrap_err(),
            Error::NotFitted(_)
        ));
    }

    #[test]
    fn empty_frames_are_rejected() {
        let empty = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        assert!(matches!(
            auto.fit(empty).unwrap_err(),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            AutoPipeline::from_inferred_schema(&DataFrame::new(3, Vec::<Column>::new()).unwrap()),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn from_inferred_schema_enables_scaler_and_imputer() {
        let x = frame(vec![Column::from(Series::new(
            "a".into(),
            &[Some(1.0_f64), None, Some(3.0)],
        ))]);
        let auto = AutoPipeline::from_inferred_schema(&x).unwrap();
        assert!(auto.config().scaler.is_some());
        assert!(auto.config().imputer.is_some());

        let strings = frame(vec![str_column("city", &["a", "b"])]);
        let auto = AutoPipeline::from_inferred_schema(&strings).unwrap();
        assert!(auto.config().scaler.is_none());
        assert!(auto.config().imputer.is_none());
    }

    #[test]
    fn forced_scaler_strategy_is_kept() {
        let x = frame(vec![f64_column("a", &[1.0, 2.0, 3.0, 4.0])]);
        let mut auto = AutoPipeline::new(
            AutoPipelineConfig::default()
                .scaler(AutoScalerConfig::default().strategy(ScalingStrategy::MinMax)),
        );
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        let column = out.column("a").unwrap();
        let series = column.f64().unwrap();
        let values: Vec<Option<f64>> = (0..series.len()).map(|index| series.get(index)).collect();
        // Min-max scales the four distinct values to 0, 1/3, 2/3, 1.
        assert!((values[0].unwrap() - 0.0).abs() < 1e-9);
        assert!((values[3].unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn feature_names_out_matches_transform_output() {
        let x = frame(vec![
            f64_column("height", &[1.0, 2.0, 3.0, 4.0]),
            str_column("city", &["a", "b", "a", "b"]),
        ]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().scaler(AutoScalerConfig::default()));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x.clone()).unwrap();

        let input: Vec<String> = x
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        let expected: Vec<String> = auto.feature_names_out(&input).unwrap();
        let actual: Vec<String> = out
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn imputer_fills_missing_numeric_values() {
        let x = frame(vec![Column::from(Series::new(
            "a".into(),
            &[Some(1.0_f64), None, Some(3.0), Some(5.0)],
        ))]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().imputer(AutoImputerConfig::default()));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();

        assert_eq!(out.column("a").unwrap().null_count(), 0);
        // The heuristic uses the mean for an approximately symmetric column:
        // (1 + 3 + 5) / 3 = 3.
        let filled = out.column("a").unwrap().f64().unwrap().get(1).unwrap();
        assert!((filled - 3.0).abs() < 1e-9, "filled value was {filled}");
    }

    #[test]
    fn empty_string_target_is_rejected() {
        let config = AutoPipelineConfig {
            target: Some("  ".to_string()),
            ..AutoPipelineConfig::default()
        };
        let mut auto = AutoPipeline::new(config);
        let x = frame(vec![f64_column("a", &[1.0, 2.0])]);
        assert!(matches!(auto.fit(x).unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn missing_target_is_rejected() {
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().target("y"));
        let x = frame(vec![f64_column("a", &[1.0, 2.0])]);
        let error = auto.fit(x).unwrap_err();
        assert!(
            error.to_string().contains("target column 'y'"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn zero_hash_buckets_is_rejected() {
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().hash_buckets(0));
        let x = frame(vec![f64_column("a", &[1.0, 2.0])]);
        assert!(matches!(auto.fit(x).unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn all_constant_features_error() {
        let x = frame(vec![
            f64_column("a", &[1.0, 1.0, 1.0]),
            f64_column("b", &[2.0, 2.0, 2.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default());
        assert!(matches!(auto.fit(x).unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn constant_columns_survive_when_removal_is_disabled() {
        let x = frame(vec![
            f64_column("a", &[1.0, 1.0, 1.0]),
            f64_column("b", &[1.0, 2.0, 3.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().remove_constant(false));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        assert_eq!(out.width(), 2);
    }

    #[test]
    fn duplicate_columns_are_removed_when_enabled() {
        let x = frame(vec![
            f64_column("a", &[1.0, 2.0, 3.0]),
            f64_column("a_copy", &[1.0, 2.0, 3.0]),
        ]);
        let mut auto = AutoPipeline::new(AutoPipelineConfig::default().remove_duplicates(true));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        assert_eq!(out.get_column_names(), vec!["a"]);
    }

    #[test]
    fn transform_accepts_the_target_column_in_input() {
        let x = frame(vec![
            str_column("city", &["a", "b", "c", "d", "e", "f"]),
            Column::from(Series::new("y".into(), &[0_i32, 1, 0, 1, 0, 1])),
        ]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().target("y").cat_threshold(3));
        auto.fit(x.clone()).unwrap();
        let out = auto.transform(x).unwrap();
        assert_eq!(out.get_column_names().len(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "city");
    }

    #[test]
    fn refit_replaces_the_previous_plan() {
        let first = frame(vec![f64_column("a", &[1.0, 2.0, 3.0])]);
        let mut auto =
            AutoPipeline::new(AutoPipelineConfig::default().scaler(AutoScalerConfig::default()));
        auto.fit(first.clone()).unwrap();
        assert_eq!(
            auto.prepared_pipeline().unwrap().steps()[0].0,
            "column_transformers"
        );

        let second = frame(vec![f64_column("b", &[4.0, 5.0, 6.0])]);
        auto.fit(second.clone()).unwrap();
        let out = auto.transform(second).unwrap();
        assert_eq!(out.get_column_names(), vec!["b"]);
    }
}
