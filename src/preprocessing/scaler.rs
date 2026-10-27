//! Feature scaling and centering.
//!
//! Analogous to `sklearn.preprocessing.scaler`. Provides:
//! - [`StandardScaler`] — z-score normalization
//! - [`MinMaxScaler`] — min-max scaling to a range
//! - [`RobustScaler`] — scaling robust to outliers via IQR

use polars::prelude::*;

use crate::traits::inverse::{HasInverse, InverseTransform};
use crate::traits::schema::{RequiresSchema, SchemaSpec};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};

/// Standardize features by removing the mean and scaling to unit variance.
///
/// For each column `x`, computes `(x - mean) / std` where `mean` and `std`
/// are learned from the training data.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StandardScaler {
    fitted: bool,
    params: Option<Vec<ScaleParam>>,
    with_mean: bool,
    with_std: bool,
}

struct ScaleParam {
    name: String,
    mean: f64,
    std: f64,
}

impl StandardScaler {
    /// Create a new `StandardScaler`.
    ///
    /// Both centering and scaling are enabled by default.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            with_mean: true,
            with_std: true,
        }
    }

    /// Whether to center the data by subtracting the mean (default: `true`).
    pub fn with_mean(mut self, value: bool) -> Self {
        self.with_mean = value;
        self
    }

    /// Whether to scale the data to unit variance (default: `true`).
    pub fn with_std(mut self, value: bool) -> Self {
        self.with_std = value;
        self
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for StandardScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        let n_cols = x.width();
        if x.height() == 0 || n_cols == 0 {
            return Err(Error::InvalidInput(
                "StandardScaler.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        let col_names = require_f64_columns(&x, "StandardScaler")?;

        let mut params = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "StandardScaler: column '{}' expected but not found. {}",
                    name, e
                ))
            })?;

            let _ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "StandardScaler: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let vals: Vec<f64> = _ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "StandardScaler: column '{}' has no non-null, finite values. \
                     Cannot scale an all-null or all-NaN column. Impute first or drop the column.",
                    name
                )));
            }

            let (col_mean, col_std) = if self.with_mean || self.with_std {
                let fitted_mean = vals.iter().sum::<f64>() / vals.len() as f64;
                if !fitted_mean.is_finite() {
                    return Err(Error::InvalidInput(format!(
                        "StandardScaler: column '{}' mean overflowed or is not finite. \
                         Input values are too large for f64 computation.",
                        name
                    )));
                }
                let mean = if self.with_mean { fitted_mean } else { 0.0 };

                let std = if self.with_std {
                    let var = vals.iter().map(|v| (v - fitted_mean).powi(2)).sum::<f64>()
                        / vals.len() as f64;
                    if !var.is_finite() || var < 0.0 {
                        return Err(Error::InvalidInput(format!(
                            "StandardScaler: column '{}' variance overflowed or is not finite. \
                             Input values are too large for f64 computation.",
                            name
                        )));
                    }
                    let s = var.sqrt();
                    if !s.is_finite() {
                        return Err(Error::InvalidInput(format!(
                            "StandardScaler: column '{}' standard deviation overflowed or is not finite.",
                            name
                        )));
                    }
                    if s < f64::EPSILON {
                        return Err(Error::Computation(format!(
                            "StandardScaler: column '{}' has zero variance. \
                             Try removing it with VarianceThreshold or setting with_std(false).",
                            name
                        )));
                    }
                    s
                } else {
                    1.0
                };
                (mean, std)
            } else {
                (0.0, 1.0)
            };

            params.push(ScaleParam {
                name: name.clone(),
                mean: col_mean,
                std: col_std,
            });
        }

        self.params = Some(params);
        self.fitted = true;

        Ok(())
    }
}

impl Transform<DataFrame> for StandardScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "StandardScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "StandardScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let mut out = x.clone();

        for p in params {
            let mean = p.mean;
            let std = p.std;
            replace_f64_column(&mut out, &p.name, "StandardScaler", |v| (v - mean) / std)?;
        }

        Ok(out)
    }
}

impl InverseTransform<DataFrame> for StandardScaler {
    type Output = DataFrame;

    /// Undo the z-score transform: `x * std + mean` per column.
    ///
    /// Columns whose scaling was disabled at fit time store `std = 1`, and
    /// columns whose centering was disabled store `mean = 0`, so the same
    /// formula restores both configurations.
    fn inverse_transform(&self, x: DataFrame) -> Result<DataFrame> {
        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "StandardScaler has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            )
        })?;
        let mut out = x.clone();

        for p in params {
            let mean = p.mean;
            let std = p.std;
            replace_f64_column(&mut out, &p.name, "StandardScaler", |v| v * std + mean)?;
        }

        Ok(out)
    }
}

impl HasInverse for StandardScaler {}

/// Scale features to a given range (default `[0, 1]`).
///
/// For each column `x`, computes `(x - min) / (max - min) * range + range_min`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::MinMaxScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = MinMaxScaler::new().feature_range((-1.0, 1.0));
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MinMaxScaler {
    fitted: bool,
    params: Option<Vec<MinMaxParam>>,
    feature_range: (f64, f64),
}

struct MinMaxParam {
    name: String,
    min: f64,
    scale: f64,
}

impl MinMaxScaler {
    /// Create a new `MinMaxScaler` that scales to `[0, 1]`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Set the output feature range (default `(0.0, 1.0)`).
    pub fn feature_range(mut self, range: (f64, f64)) -> Self {
        self.feature_range = range;
        self
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for MinMaxScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "MinMaxScaler.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        let col_names = require_f64_columns(&x, "MinMaxScaler")?;
        let r_min = self.feature_range.0;
        let r_max = self.feature_range.1;
        let mut params = Vec::new();

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "MinMaxScaler.fit: column '{}' not found. {}",
                    name, e
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "MinMaxScaler.fit: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            let vals: Vec<f64> = ca.iter().flatten().filter(|v| !v.is_nan()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "MinMaxScaler: column '{}' has no non-null, non-NaN values. \
                     Cannot scale an all-null or all-NaN column. Impute first or drop the column.",
                    name
                )));
            }

            let col_min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let col_max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            if (col_max - col_min).abs() < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "MinMaxScaler: column '{}' is constant (all values = {}). \
                     Cannot scale a constant column. Remove it or use StandardScaler \
                     with with_std(false).",
                    name, col_min
                )));
            }

            let scale = (r_max - r_min) / (col_max - col_min);
            params.push(MinMaxParam {
                name: name.clone(),
                min: col_min,
                scale,
            });
        }

        self.params = Some(params);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for MinMaxScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "MinMaxScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let r_min = self.feature_range.0;
        let mut out = x.clone();

        for p in self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "MinMaxScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })? {
            let min = p.min;
            let scale = p.scale;
            replace_f64_column(&mut out, &p.name, "MinMaxScaler", |v| {
                (v - min) * scale + r_min
            })?;
        }

        Ok(out)
    }
}

impl InverseTransform<DataFrame> for MinMaxScaler {
    type Output = DataFrame;

    /// Undo the min-max scaling: `(x - range_min) / scale + min` per column.
    ///
    /// The denominator comes from the stored `scale`, so no range is
    /// re-derived. Values outside the fitted range extrapolate past the
    /// original training min/max instead of being clipped.
    fn inverse_transform(&self, x: DataFrame) -> Result<DataFrame> {
        let r_min = self.feature_range.0;
        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "MinMaxScaler has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            )
        })?;
        let mut out = x.clone();

        for p in params {
            let min = p.min;
            let scale = p.scale;
            replace_f64_column(&mut out, &p.name, "MinMaxScaler", |v| {
                (v - r_min) / scale + min
            })?;
        }

        Ok(out)
    }
}

impl HasInverse for MinMaxScaler {}

/// Scale features using statistics robust to outliers.
///
/// For each column `x`, computes `(x - median) / IQR` using the
/// interquartile range, which is insensitive to outliers.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::RobustScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = RobustScaler::new().with_centering(true);
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RobustScaler {
    fitted: bool,
    params: Option<Vec<RobustParam>>,
    with_centering: bool,
    with_scaling: bool,
}

struct RobustParam {
    name: String,
    center: f64,
    scale: f64,
}

impl RobustScaler {
    /// Create a new `RobustScaler` with centering and scaling enabled.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            with_centering: true,
            with_scaling: true,
        }
    }

    /// Whether to center by subtracting the median (default: `true`).
    pub fn with_centering(mut self, value: bool) -> Self {
        self.with_centering = value;
        self
    }

    /// Whether to scale by the IQR (default: `true`).
    pub fn with_scaling(mut self, value: bool) -> Self {
        self.with_scaling = value;
        self
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear-interpolated percentile of a sorted slice.
///
/// `p` is a percentage in `[0, 100]` (e.g. `25.0` for the first quartile).
/// Interpolates between the two nearest elements when the percentile falls
/// between them, per the `numpy.percentile` default (linear) method.
/// Returns `0.0` for an empty slice; for a single element the element
/// itself. Shared by `RobustScaler` and the quantile-based transformers
/// (e.g. `Winsorizer`).
pub(crate) fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = (p / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl Fit<DataFrame> for RobustScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "RobustScaler.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        let col_names = require_f64_columns(&x, "RobustScaler")?;
        let mut params = Vec::new();

        for name in &col_names {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "RobustScaler.fit: column '{}' not found. {}",
                    name, e
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "RobustScaler.fit: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            let mut vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();

            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "RobustScaler: column '{}' has no non-null, finite values. \
                     Cannot scale an all-null or all-NaN column. Impute first or drop the column.",
                    name
                )));
            }

            vals.sort_by(|a, b| a.total_cmp(b));

            let median = percentile_sorted(&vals, 50.0);
            let q1 = percentile_sorted(&vals, 25.0);
            let q3 = percentile_sorted(&vals, 75.0);
            let iqr = q3 - q1;

            if self.with_centering && !median.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "RobustScaler: column '{}' median is not finite.",
                    name
                )));
            }

            if self.with_scaling && !iqr.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "RobustScaler: column '{}' IQR overflowed or is not finite. \
                     Input values are too large for f64 computation.",
                    name
                )));
            }

            if self.with_scaling && iqr < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "RobustScaler: column '{}' has zero IQR (Q1=Q3={}). \
                     All values are the same. Remove the column or use a different scaler.",
                    name, median
                )));
            }

            params.push(RobustParam {
                name: name.clone(),
                center: if self.with_centering { median } else { 0.0 },
                scale: if self.with_scaling { iqr } else { 1.0 },
            });
        }

        self.params = Some(params);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for RobustScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "RobustScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let mut out = x.clone();

        for p in self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "RobustScaler has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })? {
            let center = p.center;
            let scale = p.scale;
            replace_f64_column(&mut out, &p.name, "RobustScaler", |v| (v - center) / scale)?;
        }

        Ok(out)
    }
}

impl InverseTransform<DataFrame> for RobustScaler {
    type Output = DataFrame;

    /// Undo the robust scaling: `x * scale + center` per column.
    ///
    /// Centering and scaling are stored as `center = 0` / `scale = 1` when
    /// they were disabled at fit time, so the same formula inverts every
    /// builder configuration.
    fn inverse_transform(&self, x: DataFrame) -> Result<DataFrame> {
        let params = self.params.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "RobustScaler has not been fitted. \
                 Call .fit(dataframe) before .inverse_transform()."
                    .into(),
            )
        })?;
        let mut out = x.clone();

        for p in params {
            let center = p.center;
            let scale = p.scale;
            replace_f64_column(&mut out, &p.name, "RobustScaler", |v| v * scale + center)?;
        }

        Ok(out)
    }
}

impl HasInverse for RobustScaler {}

/// `StandardScaler` normalizes the `Float64` columns it learned at fit.
impl RequiresSchema for StandardScaler {
    fn required_schema(&self) -> SchemaSpec {
        match &self.params {
            Some(params) => {
                SchemaSpec::from_names(params.iter().map(|p| (p.name.clone(), DataType::Float64)))
            }
            None => SchemaSpec::f64_columns(1),
        }
    }
}

/// `MinMaxScaler` scales the `Float64` columns it learned at fit.
impl RequiresSchema for MinMaxScaler {
    fn required_schema(&self) -> SchemaSpec {
        match &self.params {
            Some(params) => {
                SchemaSpec::from_names(params.iter().map(|p| (p.name.clone(), DataType::Float64)))
            }
            None => SchemaSpec::f64_columns(1),
        }
    }
}

/// `RobustScaler` scales the `Float64` columns it learned at fit.
impl RequiresSchema for RobustScaler {
    fn required_schema(&self) -> SchemaSpec {
        match &self.params {
            Some(params) => {
                SchemaSpec::from_names(params.iter().map(|p| (p.name.clone(), DataType::Float64)))
            }
            None => SchemaSpec::f64_columns(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_standard_scaler_fit_transform() {
        let mut scaler = StandardScaler::new();
        let df = make_test_df();

        scaler.fit(df.clone()).unwrap();
        let result = scaler.transform(df).unwrap();

        let scaled_a = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<f64> = scaled_a.iter().flatten().collect();

        assert_relative_eq!(vals[0], -1.22474487, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.22474487, epsilon = 1e-6);
    }

    #[test]
    fn test_min_max_scaler() {
        let mut scaler = MinMaxScaler::new();
        let df = make_test_df();

        scaler.fit(df.clone()).unwrap();
        let result = scaler.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    /// Regression for #12: an all-null column must error at `fit` time instead
    /// of silently fitting with NaN parameters and propagating NaN on transform.
    #[test]
    fn test_min_max_scaler_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = MinMaxScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-null column must error, not silently fit NaN params"
        );
    }

    /// Regression for #12: an all-NaN column (real `f64::NAN` values, not
    /// Polars nulls) must also error instead of silently propagating NaN.
    #[test]
    fn test_min_max_scaler_all_nan_column_error() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, f64::NAN]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = MinMaxScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-NaN column must error, not silently fit NaN params"
        );
    }

    /// Nulls mixed with valid values must keep working: the null position is
    /// preserved through transform, and valid values scale to the range ends.
    #[test]
    fn test_min_max_scaler_partial_null_ok() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(5.0)]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = MinMaxScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-6);
    }

    /// `f64::NAN` mixed with valid values must keep working: NaN is ignored for
    /// fit statistics, and the NaN position maps to NaN on transform.
    #[test]
    fn test_min_max_scaler_partial_nan_ok() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = MinMaxScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), 0.0, epsilon = 1e-6);
        assert!(
            vals[1].unwrap().is_nan(),
            "NaN input must map to NaN through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_scaler() {
        let mut scaler = RobustScaler::new();
        let df = make_test_df();

        scaler.fit(df.clone()).unwrap();
        let result = scaler.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let df = make_test_df();
        let result = scaler.transform(df);
        assert!(result.is_err());
    }

    /// Regression: `partial_cmp().unwrap()` panicked when a column contained NaN.
    /// `total_cmp` sorts NaN deterministically without panicking.
    #[test]
    fn test_robust_scaler_with_nan_does_not_panic() {
        let a = Column::from(Series::new("a".into(), &[1.0f64, f64::NAN, 5.0, 3.0]));
        let df = DataFrame::new(4, vec![a]).unwrap();
        let mut scaler = RobustScaler::new();
        // fit must not panic on the NaN-bearing sort.
        scaler.fit(df.clone()).unwrap();
        let _ = scaler.transform(df).unwrap();
    }

    /// Regression for #35: an all-null column must error at `fit` time instead
    /// of silently fitting with NaN parameters.
    #[test]
    fn test_standard_scaler_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = StandardScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-null column must error, not silently fit NaN params"
        );
    }

    /// Regression for #35: an all-NaN column must error at `fit` time.
    #[test]
    fn test_standard_scaler_all_nan_column_error() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, f64::NAN]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = StandardScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-NaN column must error, not silently fit NaN params"
        );
    }

    /// Nulls mixed with valid values must keep working: the null position is
    /// preserved through transform, and valid values scale correctly.
    #[test]
    fn test_standard_scaler_partial_null_ok() {
        let x = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(5.0)]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert!(
            vals[1].is_none(),
            "null input must stay null through transform"
        );
        assert_relative_eq!(vals[0].unwrap(), -1.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-6);
    }

    /// `f64::NAN` mixed with valid values must keep working: NaN is ignored for
    /// fit statistics, and the NaN position maps to NaN on transform.
    #[test]
    fn test_standard_scaler_partial_nan_ok() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), -1.0, epsilon = 1e-6);
        assert!(
            vals[1].unwrap().is_nan(),
            "NaN input must map to NaN through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-6);
    }

    /// Regression for #35: an all-null column must error at `fit` time.
    #[test]
    fn test_robust_scaler_all_null_column_error() {
        let x = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = RobustScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-null column must error, not silently fit NaN params"
        );
    }

    /// Regression for #35: an all-NaN column must error at `fit` time.
    #[test]
    fn test_robust_scaler_all_nan_column_error() {
        let x = Column::from(Series::new("x".into(), &[f64::NAN, f64::NAN, f64::NAN]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = RobustScaler::new();
        let fit_result = scaler.fit(df.clone());
        assert!(
            fit_result.is_err(),
            "fitting an all-NaN column must error, not silently fit NaN params"
        );
    }

    /// `f64::NAN` mixed with valid values must keep working: NaN is ignored for
    /// fit statistics, and the NaN position maps to NaN on transform.
    #[test]
    fn test_robust_scaler_partial_nan_ok() {
        let x = Column::from(Series::new("x".into(), &[1.0f64, f64::NAN, 5.0]));
        let df = DataFrame::new(3, vec![x]).unwrap();
        let mut scaler = RobustScaler::new();
        scaler.fit(df.clone()).unwrap();
        let out = scaler.transform(df).unwrap();
        let ca = out.column("x").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        assert_relative_eq!(vals[0].unwrap(), -1.0, epsilon = 1e-6);
        assert!(
            vals[1].unwrap().is_nan(),
            "NaN input must map to NaN through transform"
        );
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-6);
    }

    /// Regression for #181: extreme magnitude values (~1e300) must error
    /// on fit due to variance overflow, instead of silently producing
    /// infinite std which maps all features to +/-0.0 during transform.
    #[test]
    fn test_standard_scaler_extreme_magnitude_overflow_error() {
        let x = Column::from(Series::new(
            "x".into(),
            &[1e300_f64, 2e300, 3e300, 4e300, 5e300],
        ));
        let df = DataFrame::new(5, vec![x]).unwrap();
        let mut scaler = StandardScaler::new();
        let fit_result = scaler.fit(df);
        assert!(
            fit_result.is_err(),
            "StandardScaler on 1e300 magnitude data must error on variance overflow"
        );
    }

    /// When both with_mean and with_std are disabled, StandardScaler is an identity
    /// operation and must not fail on extreme magnitude data where mean would overflow.
    #[test]
    fn test_standard_scaler_disabled_mean_and_std_extreme_magnitude_ok() {
        let x = Column::from(Series::new(
            "x".into(),
            &[1e300_f64, 2e300, 3e300, 4e300, 5e300],
        ));
        let df = DataFrame::new(5, vec![x]).unwrap();
        let mut scaler = StandardScaler::new().with_mean(false).with_std(false);
        assert!(scaler.fit(df.clone()).is_ok());
        let res = scaler.transform(df).unwrap();
        let vals: Vec<f64> = res
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(vals, vec![1e300, 2e300, 3e300, 4e300, 5e300]);
    }

    /// RobustScaler on extreme magnitude data (~1e300) operates via median and IQR
    /// without squaring, so it scales stably without overflowing.
    #[test]
    fn test_robust_scaler_extreme_magnitude_scales_ok() {
        let x = Column::from(Series::new(
            "x".into(),
            &[1e300_f64, 2e300, 3e300, 4e300, 5e300],
        ));
        let df = DataFrame::new(5, vec![x]).unwrap();
        let mut scaler = RobustScaler::new();
        assert!(scaler.fit(df.clone()).is_ok());
        let res = scaler.transform(df).unwrap();
        let vals: Vec<f64> = res
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_relative_eq!(vals[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], -0.5, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[3], 0.5, epsilon = 1e-6);
        assert_relative_eq!(vals[4], 1.0, epsilon = 1e-6);
    }

    /// RobustScaler must reject columns where IQR computation overflows f64.
    #[test]
    fn test_robust_scaler_iqr_overflow_error() {
        let x = Column::from(Series::new(
            "x".into(),
            &[-1.5e308_f64, -1.5e308, 1.5e308, 1.5e308],
        ));
        let df = DataFrame::new(4, vec![x]).unwrap();
        let mut scaler = RobustScaler::new();
        let fit_result = scaler.fit(df);
        assert!(
            fit_result.is_err(),
            "RobustScaler on overflowing IQR data must error"
        );
    }

    #[test]
    fn test_scalers_require_float64_columns() {
        use crate::traits::schema::validate_schema;

        let strings = Column::from(Series::new("s".into(), &["a", "b"]));
        let no_f64 = DataFrame::new(2, vec![strings]).unwrap();

        // Before fit each scaler only knows the dtype it needs.
        let specs = [
            StandardScaler::new().required_schema(),
            MinMaxScaler::new().required_schema(),
            RobustScaler::new().required_schema(),
        ];
        for spec in &specs {
            assert_eq!(*spec, SchemaSpec::f64_columns(1));
            assert!(validate_schema(spec, &no_f64).is_err());
            assert!(validate_schema(spec, &make_test_df()).is_ok());
        }

        // After fit the spec names the columns the scaler learned, so a frame
        // that swaps in a different Float64 column no longer passes.
        let mut scaler = StandardScaler::new();
        scaler.fit(make_test_df()).unwrap();
        let fitted = scaler.required_schema();
        assert_eq!(
            fitted,
            SchemaSpec::exact(&[("a", DataType::Float64), ("b", DataType::Float64)])
        );
        let other = Column::from(Series::new("c".into(), &[1.0f64, 2.0, 3.0]));
        let wrong_name = DataFrame::new(3, vec![other]).unwrap();
        let msg = validate_schema(&fitted, &wrong_name)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("Missing column 'a'"), "message: {msg}");
    }

    fn column_values(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    fn assert_roundtrip(inverse: &DataFrame, original: &DataFrame) {
        for name in original.get_column_names() {
            let want = column_values(original, name);
            let got = column_values(inverse, name);
            assert_eq!(want.len(), got.len());
            for (w, g) in want.iter().zip(&got) {
                assert_relative_eq!(*w, *g, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_standard_scaler_inverse_roundtrip() {
        let df = make_test_df();
        let mut scaler = StandardScaler::new();
        scaler.fit(df.clone()).unwrap();

        let scaled = scaler.transform(df.clone()).unwrap();
        let restored = scaler.inverse_transform(scaled).unwrap();

        assert_roundtrip(&restored, &df);
    }

    #[test]
    fn test_standard_scaler_inverse_without_mean_or_std() {
        let df = make_test_df();

        for scaler in [
            StandardScaler::new().with_mean(false),
            StandardScaler::new().with_std(false),
            StandardScaler::new().with_mean(false).with_std(false),
        ] {
            let mut scaler = scaler;
            scaler.fit(df.clone()).unwrap();
            let scaled = scaler.transform(df.clone()).unwrap();
            let restored = scaler.inverse_transform(scaled).unwrap();
            assert_roundtrip(&restored, &df);
        }
    }

    #[test]
    fn test_min_max_scaler_inverse_roundtrip() {
        let df = make_test_df();

        for scaler in [
            MinMaxScaler::new(),
            MinMaxScaler::new().feature_range((-1.0, 1.0)),
        ] {
            let mut scaler = scaler;
            scaler.fit(df.clone()).unwrap();
            let scaled = scaler.transform(df.clone()).unwrap();
            let restored = scaler.inverse_transform(scaled).unwrap();
            assert_roundtrip(&restored, &df);
        }
    }

    #[test]
    fn test_min_max_scaler_inverse_extrapolates_out_of_range() {
        let col = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut scaler = MinMaxScaler::new();
        scaler.fit(df).unwrap();

        // 2.0 is outside the fitted output range [0, 1]; the inverse
        // extrapolates past the training maximum instead of clipping.
        let out = Column::from(Series::new("a".into(), &[2.0f64]));
        let extreme = DataFrame::new(1, vec![out]).unwrap();
        let restored = scaler.inverse_transform(extreme).unwrap();

        assert_relative_eq!(column_values(&restored, "a")[0], 9.0, epsilon = 1e-9);
    }

    #[test]
    fn test_robust_scaler_inverse_roundtrip() {
        let df = make_test_df();

        for scaler in [
            RobustScaler::new(),
            RobustScaler::new().with_centering(false),
            RobustScaler::new().with_scaling(false),
            RobustScaler::new()
                .with_centering(false)
                .with_scaling(false),
        ] {
            let mut scaler = scaler;
            scaler.fit(df.clone()).unwrap();
            let scaled = scaler.transform(df.clone()).unwrap();
            let restored = scaler.inverse_transform(scaled).unwrap();
            assert_roundtrip(&restored, &df);
        }
    }

    #[test]
    fn test_scaler_inverse_before_fit_is_not_fitted() {
        let df = make_test_df();

        let errors = [
            StandardScaler::new()
                .inverse_transform(df.clone())
                .unwrap_err(),
            MinMaxScaler::new()
                .inverse_transform(df.clone())
                .unwrap_err(),
            RobustScaler::new().inverse_transform(df).unwrap_err(),
        ];

        for err in errors {
            assert!(matches!(err, Error::NotFitted(_)), "error: {err}");
        }
    }

    #[test]
    fn test_scalers_report_invertible() {
        assert!(StandardScaler::new().is_invertible());
        assert!(MinMaxScaler::new().is_invertible());
        assert!(RobustScaler::new().is_invertible());
    }
}
