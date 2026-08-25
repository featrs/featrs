# featrs

[![Crates.io](https://img.shields.io/crates/v/featrs)](https://crates.io/crates/featrs)
[![Docs.rs](https://img.shields.io/docsrs/featrs)](https://docs.rs/featrs)
[![CI](https://github.com/featrs/featrs/actions/workflows/ci.yml/badge.svg)](https://github.com/featrs/featrs/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/featrs/featrs?style=social)](https://github.com/featrs/featrs)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Feature engineering library for Rust, inspired by scikit-learn.

Built on [Polars](https://pola.rs) — all transformations operate natively on `DataFrame` and preserve column names.

## Installation

```toml
[dependencies]
featrs = "0.4"
```

## Quick start

```rust
use featrs::prelude::*;

let mut scaler = StandardScaler::new();
scaler.fit(data.clone())?;
let scaled = scaler.transform(data)?;
```

## Features

| Category | Component | Description |
|---|---|---|
| **Scaling** | `StandardScaler` | Z-score normalization (mean 0, variance 1) |
| | `MinMaxScaler` | Scale to `[0, 1]` or custom range |
| | `RobustScaler` | Scale using median and IQR (outlier-robust) |
| | `MaxAbsScaler` | Scale by maximum absolute value; training values map to `[-1, 1]`, sparsity preserved |
| | `Winsorizer` | Clip extreme values at configurable quantiles (outlier capping) |
| | `OutlierClipper` | Clip outliers via IQR, Z-score, or MAD statistical fences |
| | `KBinsDiscretizer` | Bin continuous features into `n_bins` discrete bins (uniform, quantile, or k-means) as ordinal or one-hot |
| **Normalization** | `Normalizer` | Row-wise L1, L2, or Max normalization |
| | `Binarizer` | Threshold-based binarization; in Float64 columns, nulls and NaN pass through as null |
| **Encoding** | `OneHotEncoder` | Create binary dummy columns for categories |
| | `LabelEncoder` | Encode labels as `0..n_classes-1` integers |
| | `OrdinalEncoder` | Per-column category → integer encoding |
| | `CountEncoder` | Replace categories with their raw occurrence counts |
| | `FrequencyEncoder` | Replace categories with their observed relative frequencies |
| | `BinaryEncoder` | Encode categories as binary digit columns using `max(1, ceil(log₂(n_categories)))` columns |
| | `TargetEncoder` | Replace categories with the smoothed mean of a supervised target |
| | `LeaveOneOutEncoder` | Replace categories with a leave-one-out supervised target mean (no self-leak on training rows) |
| | `CyclicalEncoder` | Sin/cos encoding for cyclical features (hour, month) |
| | `FeatureHasher` | Hash strings into a fixed number of buckets |
| | `StringCleaner` | Trim, normalize case, and regex-replace string values |
| | `RareCategoryGrouper` | Group rare and unseen categories into `"Other"` |
| **Imputation** | `SimpleImputer` | Fill nulls with mean, median, mode, or constant |
| | `MissingIndicator` | Binary columns marking where values were missing |
| **Feature Generation** | `PolynomialFeatures` | Generate polynomial and interaction features |
| | `Lagger` | Create lag features for time-series forecasting |
| | `RollingAggregator` | Rolling window mean, std, min, max, sum |
| | `ExpandingAggregator` | Expanding (cumulative) window mean, sum, min, max, std |
| | `Difference` | Differencing (`x[t] - x[t-1]`) and percentage change |
| | `DatetimeFeatures` | Extract year/month/weekday/quarter/… components from date columns |
| **Pipeline** | `Pipeline` | Sequentially chain multiple transformers |
| | `ColumnTransformer` | Apply different transformers to different columns |
| **Selection** | `VarianceThreshold` | Remove low-variance features |
| | `SelectKBest` | Select top-k features by statistical test (ANOVA F) |
| | `CorrelationThreshold` | Drop features highly correlated with others |
| **Auto** | `AutoTypeDetector` | Auto-detect column types and apply default transforms |

## Examples

### StandardScaler

```rust
use featrs::prelude::*;

let mut scaler = StandardScaler::new();
scaler.fit(df.clone())?;
let scaled = scaler.transform(df)?;
```

### Pipeline

```rust
use featrs::prelude::*;

let mut pipeline = Pipeline::new(vec![
    ("scale".into(), Box::new(StandardScaler::new())),
    ("poly".into(), Box::new(PolynomialFeatures::new(2)?)),
])?;
pipeline.fit(df.clone())?;
let result = pipeline.transform(df)?;
```

### ColumnTransformer

```rust
use featrs::prelude::*;

let ct = ColumnTransformer::new(
    vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
    Remainder::Passthrough,
);
```

### PolynomialFeatures (builder pattern)

```rust
use featrs::prelude::*;

let pf = PolynomialFeatures::builder()
    .degree(3)
    .include_bias(false)
    .interaction_only(true)
    .build()?;
```

### Feature Selection

```rust
use featrs::prelude::*;

let mut vt = VarianceThreshold::new(0.01);
vt.fit(features.clone())?;
let filtered = vt.transform(features)?;

let mut skb = SelectKBest::new(5, Box::new(FClassif::new()));
skb.fit(features.clone(), target)?;
let selected = skb.transform(features)?;
```

### Time Series — Lag Features

```rust
use featrs::prelude::*;

let mut lagger = Lagger::new(&["sales", "revenue"], &[1, 7, 30]);
lagger.fit(df.clone())?;
let lagged = lagger.transform(df)?;  // adds sales_lag_1, sales_lag_7, ...
```

### Time Series — Rolling Windows

```rust
use featrs::prelude::*;
use featrs::time_series::rolling::RollingFn;

let mut rolling = RollingAggregator::new(&["price"], 7, RollingFn::Mean);
rolling.fit(df.clone())?;
let result = rolling.transform(df)?;  // adds price_mean_7
```

### Time Series — Differencing

```rust
use featrs::prelude::*;

let mut diff = Difference::diff(&["sales"], 1);
diff.fit(df.clone())?;
let result = diff.transform(df)?;  // adds sales_diff_1

let mut pct = Difference::pct_change(&["price"], 1);
pct.fit(df.clone())?;
let result = pct.transform(df)?;  // adds price_pct_1
```

### Cyclical Encoding

```rust
use featrs::prelude::*;

let mut enc = CyclicalEncoder::new(&["hour"], 24);
enc.fit(df.clone())?;
let result = enc.transform(df)?;  // adds hour_sin, hour_cos
```

### Feature Hasher

```rust
use featrs::prelude::*;

let mut fh = FeatureHasher::new(&["user_id", "category"], 100);
fh.fit(df.clone())?;
let hashed = fh.transform(df)?;  // 100 hashed columns
```

### Missing Indicator

```rust
use featrs::prelude::*;

let mut ind = MissingIndicator::all();
ind.fit(df.clone())?;
let marked = ind.transform(df)?;  // adds {col}_missing where nulls exist
```

### Auto-Type Detection

```rust
use featrs::prelude::*;

let mut atd = AutoTypeDetector::new()
    .cat_threshold(30)     // one-hot if < 30 unique values
    .hash_buckets(200);    // hash to 200 buckets otherwise
atd.fit(df.clone())?;
let result = atd.transform(df)?;
```

## Roadmap

Work is tracked in [GitHub milestones](https://github.com/featrs/featrs/milestones). Open issues are grouped by release so contributors can pick a target and see what ships with it.

| Milestone | Focus | Target |
|---|---|---|
| [v0.4.0](https://github.com/featrs/featrs/milestone/1) | New transformers: scalers, encoders, cleaners | Sep 2026 |
| [v0.5.0](https://github.com/featrs/featrs/milestone/2) | Text, time-series, feature selection | Oct 2026 |
| [v0.6.0](https://github.com/featrs/featrs/milestone/3) | Pipeline composition & automation (FeatureUnion, AutoPipeline, SchemaValidator) | Nov 2026 |
| [v0.7.0](https://github.com/featrs/featrs/milestone/4) | Performance workstream (benchmarks, Polars-native kernels, rayon, ahash) + LazyFrame & streaming | Dec 2026 |
| [v0.8.0](https://github.com/featrs/featrs/milestone/5) | Quality & docs hardening (lints, doc tests) | Dec 2026 |
| [v1.0.0](https://github.com/featrs/featrs/milestone/6) | Persistence (serde), tracing, ONNX export, pipeline versioning | Mar 2027 |

The performance milestones are ordered: benchmarks (Phase 0) land first, then Polars-native kernels, rayon, and ahash. ahash must stabilize before serde persistence ships so the on-disk hash format never changes.

New contributors: start with a `good first issue` in the next milestone.

## Resources

- [Crates.io](https://crates.io/crates/featrs)
- [Docs.rs](https://docs.rs/featrs)
- [GitHub](https://github.com/featrs/featrs)

## Star History

[![Star History](https://starchart.cc/featrs/featrs.svg?variant=adaptive)](https://starchart.cc/featrs/featrs)

## License

MIT
