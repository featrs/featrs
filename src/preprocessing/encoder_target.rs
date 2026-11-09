//! Supervised target encoding.
//!
//! [`TargetEncoder`] replaces categorical string values with the smoothed
//! mean of a supervised target variable, following scikit-learn's
//! `TargetEncoder` design. It is the only encoder in this crate that
//! implements [`FitSupervised`] (it needs the target `y` at fit time).

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_required_present, identity_names};
use crate::traits::{Error, FitSupervised, Result, Transform};
use polars::prelude::*;
use std::collections::HashMap;

/// Encode categorical string values with the smoothed mean of a target.
///
/// For each fitted String column, every category is replaced by
///
/// ```text
/// encoded = (n * cat_mean + alpha * global_mean) / (n + alpha)
/// ```
///
/// where `n` is the number of training rows of that category, `cat_mean` is
/// its mean target value, `global_mean` is the mean of the whole target, and
/// `alpha` is the smoothing strength (default `1.0`). Higher `alpha` pulls
/// the encoding toward the global mean, which regularizes high-cardinality
/// categories and mitigates target leakage.
///
/// Note that the statistics are **not cross-fitted** (unlike scikit-learn's
/// `TargetEncoder.fit_transform`, which encodes training rows with
/// out-of-fold statistics): every encoding here is computed from the full
/// fit data, so transforming the same frame you fitted on still exposes
/// target information in the features. Fit on a training split only and
/// apply `transform` to validation or test folds.
///
/// # Behaviour
///
/// - Only `String` columns are encoded; non-string columns pass through the
///   output unchanged. The encoded columns keep their names and positions
///   but become `Float64`. String columns that were all-null or had no
///   usable target rows at fit time are skipped and pass through un-encoded,
///   as do String columns that only appear at transform time.
/// - Every fitted column must be present and `String`-typed in the transform
///   input; otherwise [`transform`](Transform::transform) returns
///   [`Error::InvalidInput`].
/// - Categories seen at [`transform`](Transform::transform) time but not
///   during [`fit`](FitSupervised::fit) are encoded as `global_mean`.
/// - Null category values are preserved as null.
/// - Rows whose target value is null or non-finite are excluded from the
///   category statistics and from `global_mean`, so a category observed only
///   with a null target falls back to `global_mean`. With alpha `0.0` (pure
///   category means) such categories are simply absent from the mapping and
///   consequently encode as `global_mean`.
/// - `alpha` must be finite and non-negative; `fit` returns
///   [`Error::InvalidInput`] otherwise.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder_target::TargetEncoder;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
/// let x = DataFrame::new(3, vec![cat])?;
/// let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
/// let y = DataFrame::new(3, vec![target])?;
///
/// let mut enc = TargetEncoder::new().alpha(1.0);
/// enc.fit(x.clone(), y)?;
/// let encoded = enc.transform(x)?;
/// assert_eq!(encoded.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TargetEncoder {
    fitted: bool,
    alpha: f64,
    column_names: Option<Vec<String>>,
    encodings: Option<Vec<HashMap<String, f64>>>,
    global_mean: Option<f64>,
}

impl TargetEncoder {
    /// Create a new `TargetEncoder` with smoothing `alpha = 1.0`.
    ///
    /// The encoder must be fitted with [`fit`](FitSupervised::fit), which
    /// takes both the feature data and a single-column `Float64` target.
    pub fn new() -> Self {
        Self {
            fitted: false,
            alpha: 1.0,
            column_names: None,
            encodings: None,
            global_mean: None,
        }
    }

    /// Set the smoothing strength (default: `1.0`).
    ///
    /// `alpha` must be finite and non-negative. Larger values pull every
    /// encoding closer to the global target mean; `0.0` disables smoothing
    /// (pure per-category means). Invalid values are rejected at
    /// [`fit`](FitSupervised::fit) time with [`Error::InvalidInput`].
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }
}

impl Default for TargetEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FitSupervised<DataFrame, DataFrame> for TargetEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        // Reset any previously learned state so a failed re-fit cannot leave
        // stale mappings behind.
        self.fitted = false;
        self.column_names = None;
        self.encodings = None;
        self.global_mean = None;

        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(Error::InvalidInput(format!(
                "TargetEncoder: invalid smoothing alpha {}. \
                 Alpha must be finite and >= 0.",
                self.alpha
            )));
        }

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "TargetEncoder.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "TargetEncoder.fit: target must have exactly 1 column but got {} columns. \
                 Select a single target column.",
                y.width()
            )));
        }

        if y.height() != x.height() {
            return Err(Error::InvalidInput(format!(
                "TargetEncoder.fit: feature rows ({}) and target rows ({}) don't match.",
                x.height(),
                y.height()
            )));
        }

        let y_col = &y.columns()[0];
        let y_ca = y_col.as_materialized_series().f64().map_err(|e| {
            Error::InvalidInput(format!(
                "TargetEncoder.fit: target column '{}' has dtype {}; expected Float64. {}",
                y_col.name(),
                y_col.dtype(),
                e
            ))
        })?;

        // Global target mean over all usable (non-null, finite) rows.
        let y_vals: Vec<Option<f64>> = y_ca.iter().collect();
        let mut global_sum = 0.0;
        let mut global_n = 0u64;
        for opt in &y_vals {
            if let Some(v) = opt.filter(|v| v.is_finite()) {
                global_sum += v;
                global_n += 1;
            }
        }
        if global_n == 0 {
            return Err(Error::InvalidInput(
                "TargetEncoder.fit: target column contains no non-null, non-NaN values. \
                 Provide a target with at least one usable value."
                    .into(),
            ));
        }
        let global_mean = global_sum / global_n as f64;
        if !global_mean.is_finite() {
            return Err(Error::InvalidInput(
                "TargetEncoder.fit: target values overflow to a non-finite global mean. \
                 Scale the target values before fitting."
                    .into(),
            ));
        }

        let mut names = Vec::new();
        let mut encodings = Vec::new();

        for col in x.columns() {
            // Non-string columns are ignored; only String columns are encoded.
            if col.dtype() != &DataType::String {
                continue;
            }
            let name = col.name().to_string();
            let ca = col.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "TargetEncoder.fit: column '{}' has dtype {}; expected String. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;

            let mut stats: HashMap<String, (f64, u64)> = HashMap::new();
            for (opt_cat, opt_y) in ca.iter().zip(y_vals.iter()) {
                let (Some(cat), Some(v)) = (opt_cat, opt_y) else {
                    continue;
                };
                if !v.is_finite() {
                    continue;
                }
                let (sum, n) = stats.entry(cat.to_string()).or_insert((0.0, 0u64));
                *sum += *v;
                *n += 1;
            }

            // Skip columns with no observed (non-null, usable-target) category.
            if stats.is_empty() {
                continue;
            }

            // encoded = (n * cat_mean + alpha * global_mean) / (n + alpha)
            //         = (sum + alpha * global_mean) / (n + alpha)
            let mapping: HashMap<String, f64> = stats
                .into_iter()
                .map(|(c, (sum, n))| {
                    let encoded = (sum + self.alpha * global_mean) / (n as f64 + self.alpha);
                    if encoded.is_finite() {
                        Ok((c, encoded))
                    } else {
                        Err(Error::InvalidInput(format!(
                            "TargetEncoder.fit: column '{name}' would encode to a \
                             non-finite value (sum = {sum}, n = {n}). Target values \
                             or alpha are too large; scale the target or reduce alpha."
                        )))
                    }
                })
                .collect::<Result<HashMap<String, f64>>>()?;

            names.push(name);
            encodings.push(mapping);
        }

        if names.is_empty() {
            return Err(Error::InvalidInput(
                "TargetEncoder.fit: no string columns with usable categories found. \
                 TargetEncoder operates on String columns with at least one non-null \
                 value paired with a usable target row."
                    .into(),
            ));
        }

        self.column_names = Some(names);
        self.encodings = Some(encodings);
        self.global_mean = Some(global_mean);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for TargetEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "TargetEncoder has not been fitted. \
                 Call .fit(x, y) before .transform()."
                    .into(),
            ));
        }
        let (names, encodings, global_mean) =
            match (&self.column_names, &self.encodings, self.global_mean) {
                (Some(n), Some(e), Some(g)) => (n, e, g),
                _ => {
                    return Err(Error::NotFitted(
                        "TargetEncoder has not been fitted. \
                         Call .fit(x, y) before .transform()."
                            .into(),
                    ));
                }
            };

        let mut out = x;
        for (name, mapping) in names.iter().zip(encodings.iter()) {
            let encoded = {
                let s = out.column(name.as_str()).map_err(|e| {
                    Error::InvalidInput(format!(
                        "TargetEncoder.transform: column '{}' not found. \
                         The encoder was fitted on columns: {:?}. {}",
                        name,
                        names.iter().collect::<Vec<_>>(),
                        e
                    ))
                })?;
                let ca = s.as_materialized_series().str().map_err(|e| {
                    Error::InvalidInput(format!(
                        "TargetEncoder.transform: column '{}' has dtype {}; expected String. {}",
                        name,
                        s.dtype(),
                        e
                    ))
                })?;

                let encoded: ChunkedArray<Float64Type> = ca
                    .iter()
                    .map(|opt| opt.map(|v| mapping.get(v).copied().unwrap_or(global_mean)))
                    .collect();
                let mut series = encoded.into_series();
                series.rename(name.as_str().into());
                Column::from(series)
            };
            out.replace(name.as_str(), encoded).map_err(|e| {
                Error::Computation(format!(
                    "TargetEncoder.transform: failed to replace column '{}'. {}",
                    name, e
                ))
            })?;
        }

        Ok(out)
    }
}

impl FeatureNamesOut for TargetEncoder {
    /// Output names are the input names: `transform` rewrites columns in place.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        let names = identity_names(self.fitted, "TargetEncoder", input_columns)?;
        if let Some(column_names) = &self.column_names {
            ensure_required_present(input_columns, column_names, "TargetEncoder")?;
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_data() -> (DataFrame, DataFrame) {
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();
        (x, y)
    }

    #[test]
    fn test_smoothed_encoding() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        // global_mean = 2/3; "a": (2 + 2/3) / 3 = 8/9; "b": (0 + 2/3) / 2 = 1/3
        assert_relative_eq!(vals[0], 8.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 8.0 / 9.0, epsilon = 1e-12);
        assert_eq!(result.column("cat").unwrap().dtype(), &DataType::Float64);
    }

    #[test]
    fn test_pure_category_means_with_alpha_zero() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new().alpha(0.0);
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_unseen_category_falls_back_to_global_mean() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x, y).unwrap();

        let unseen = Column::from(Series::new("cat".into(), &["a", "zzz"]));
        let df = DataFrame::new(2, vec![unseen]).unwrap();
        let result = enc.transform(df).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        assert_relative_eq!(vals[0], 8.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_null_category_values_preserved() {
        let cat = Column::from(Series::new("cat".into(), &[Some("a"), None, Some("a")]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new(
            "y".into(),
            &[Some(1.0f64), Some(1.0), Some(0.0)],
        ));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<Option<f64>> = ca.iter().collect();
        // global_mean = 2/3; "a": (1 + 2/3) / 3 = 5/9
        assert_relative_eq!(vals[0].unwrap(), 5.0 / 9.0, epsilon = 1e-12);
        assert!(vals[1].is_none(), "null category must stay null");
        assert_relative_eq!(vals[2].unwrap(), 5.0 / 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_null_target_rows_excluded() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a", "b"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[Some(1.0f64), None, Some(0.0)]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        // global_mean = 0.5; "a": (1 + 0.5) / 2 = 0.75; "b": (0 + 0.5) / 2 = 0.25
        assert_relative_eq!(vals[0], 0.75, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.75, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_non_finite_target_values_excluded() {
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, f64::NAN, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let ca = result.column("cat").unwrap().f64().unwrap();
        let vals: Vec<f64> = ca.iter().flatten().collect();
        // global_mean = 1.0; "a": (2 + 1.0) / 3 = 1.0; "b" never usable -> global_mean
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_encodes_all_string_columns_keeps_non_string() {
        let c1 = Column::from(Series::new("c1".into(), &["a", "b", "a"]));
        let c2 = Column::from(Series::new("c2".into(), &["x", "x", "y"]));
        let num = Column::from(Series::new("num".into(), &[10.0f64, 20.0, 30.0]));
        let x = DataFrame::new(3, vec![c1, c2, num]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        assert_eq!(result.width(), 3);
        assert_eq!(result.column("c1").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("c2").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("num").unwrap().dtype(), &DataType::Float64);

        // c1: "a" 8/9, "b" 1/3; c2: "x" (1 + 2/3)/3 = 5/9, "y" (1 + 2/3)/2 = 5/6
        let c1_vals: Vec<f64> = result
            .column("c1")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(c1_vals[0], 8.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(c1_vals[1], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(c1_vals[2], 8.0 / 9.0, epsilon = 1e-12);

        let c2_vals: Vec<f64> = result
            .column("c2")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(c2_vals[0], 5.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(c2_vals[1], 5.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(c2_vals[2], 5.0 / 6.0, epsilon = 1e-12);

        let num_vals: Vec<f64> = result
            .column("num")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(num_vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_transform_before_fit_is_not_fitted() {
        let (x, _y) = make_data();
        let enc = TargetEncoder::new();
        let err = enc.transform(x).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "expected NotFitted, got {err:?}"
        );
    }

    #[test]
    fn test_empty_input_rejected() {
        let empty = Column::from(Series::new("cat".into(), Vec::<&str>::new()));
        let x = DataFrame::new(0, vec![empty]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64]));
        let y = DataFrame::new(1, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_multi_column_target_rejected() {
        let (x, _y) = make_data();
        let t1 = Column::from(Series::new("y1".into(), &[1.0f64, 0.0, 1.0]));
        let t2 = Column::from(Series::new("y2".into(), &[1.0f64, 1.0, 1.0]));
        let y = DataFrame::new(3, vec![t1, t2]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_row_count_mismatch_rejected() {
        let (x, _y) = make_data();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_non_f64_target_rejected() {
        let (x, _y) = make_data();
        let target = Column::from(Series::new("y".into(), &[1i64, 0, 1]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_negative_alpha_rejected() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new().alpha(-1.0);
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_non_finite_alpha_rejected() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new().alpha(f64::NAN);
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_no_string_columns_rejected() {
        let num = Column::from(Series::new("num".into(), &[1.0f64, 2.0, 3.0]));
        let x = DataFrame::new(3, vec![num]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_all_null_target_rejected() {
        let (x, _y) = make_data();
        let target = Column::from(Series::new("y".into(), &[None::<f64>, None, None]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_failed_refit_resets_state() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();

        // A failed re-fit must not leave the encoder fitted.
        let target = Column::from(Series::new("y".into(), &[1i64, 0, 1]));
        let bad_y = DataFrame::new(3, vec![target]).unwrap();
        assert!(enc.fit(make_data().0, bad_y).is_err());

        let err = enc.transform(x).unwrap_err();
        assert!(
            matches!(err, Error::NotFitted(_)),
            "failed re-fit must reset fitted=false, got {err:?}"
        );
    }

    #[test]
    fn test_successful_refit_updates_encodings() {
        let (x1, y1) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x1, y1).unwrap();

        // Refit on different data: "cat" all "a" with y all 0.0.
        let cat2 = Column::from(Series::new("cat".into(), &["a", "a", "a"]));
        let x2 = DataFrame::new(3, vec![cat2]).unwrap();
        let target2 = Column::from(Series::new("y".into(), &[0.0_f64, 0.0, 0.0]));
        let y2 = DataFrame::new(3, vec![target2]).unwrap();
        enc.fit(x2.clone(), y2).unwrap();

        let result = enc.transform(x2).unwrap();
        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        for v in &vals {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_missing_fitted_column_at_transform_rejected() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x, y).unwrap();

        let other = Column::from(Series::new("other".into(), &["a", "b"]));
        let df = DataFrame::new(2, vec![other]).unwrap();
        let err = enc.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_wrong_dtype_at_transform_rejected() {
        let (x, y) = make_data();
        let mut enc = TargetEncoder::new();
        enc.fit(x, y).unwrap();

        let cat = Column::from(Series::new("cat".into(), &[1i64, 2, 3]));
        let df = DataFrame::new(3, vec![cat]).unwrap();
        let err = enc.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_all_null_string_column_skipped_when_others_usable() {
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let nulls = Column::from(Series::new("nulls".into(), &[None::<&str>, None, None]));
        let x = DataFrame::new(3, vec![cat, nulls]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.column("cat").unwrap().dtype(), &DataType::Float64);
        assert_eq!(
            result.column("nulls").unwrap().dtype(),
            &DataType::String,
            "skipped all-null column must pass through un-encoded"
        );
    }

    #[test]
    fn test_overflowing_category_sums_rejected() {
        // alpha * global_mean overflows; the per-category encoded value would
        // be non-finite, which must be reported instead of silently emitted.
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1e200_f64, 0.0, 1e200]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = TargetEncoder::new().alpha(f64::MAX);
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn test_overflowing_global_mean_rejected() {
        // The summed targets overflow to infinity; the global mean would be
        // non-finite and poison every encoding.
        let cat = Column::from(Series::new("cat".into(), &["a", "b"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[f64::MAX, f64::MAX]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = TargetEncoder::new();
        let err = enc.fit(x, y).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }
}
