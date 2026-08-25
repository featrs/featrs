//! Leave-one-out supervised target encoding.
//!
//! [`LeaveOneOutEncoder`] is a variant of [`TargetEncoder`](
//! crate::preprocessing::encoder_target::TargetEncoder) that replaces
//! categorical string values with a supervised target's mean, but for each
//! *training* row the statistics are computed from all **other** rows that
//! share its category (`leave-one-out` cross-fitting). Excluding the row's own
//! target value prevents target leakage into the training features, which
//! dramatically reduces overfitting compared to plain target encoding.

use crate::traits::{Error, FitSupervised, Result, Transform};
use polars::prelude::*;
use std::collections::HashMap;

/// Encode categorical string values with a leave-one-out smoothed target mean.
///
/// For every String column of the feature data, each row's category is
/// replaced by the mean of the target over the **other** training rows that
/// belong to that category:
///
/// ```text
/// loo = (category_sum - y_i) / (category_n - 1)
/// ```
///
/// optionally smoothed toward the global target mean:
///
/// ```text
/// loo = ((category_sum - y_i) + alpha * global_mean) / ((category_n - 1) + alpha)
/// ```
///
/// In both formulas `category_sum` and `category_n` count only *usable*
/// target rows (non-null and finite) within the row's category.
///
/// Because the row's own target is excluded from its own statistic, LOO
/// encoding does **not** leak the target into the training features (unlike
/// [`TargetEncoder`](crate::preprocessing::encoder_target::TargetEncoder)).
///
/// # Behaviour
///
/// - [`fit`](FitSupervised::fit) is supervised: it takes the feature data and
///   a single-column `Float64` target. Every `String` column is encoded in
///   place as `Float64` (even a column with no usable target rows — its
///   non-null values become the global mean); non-string columns pass through
///   unchanged at their original position/dtype.
/// - [`transform`](Transform::transform) accepts only the **exact training
///   frame** (the frame passed to [`fit`](FitSupervised::fit), or that frame
///   with its columns reordered — matched by column name, value-for-value
///   including nulls, with rows in the original order). It returns the per-row
///   leave-one-out encodings. Any
///   **other** frame — a row subset, new rows, or a different column set —
///   returns [`Error::InvalidInput`] instead of silently falling back to
///   full-sample means, which would leak each row's own target into the
///   training features. For genuinely new data, call
///   [`transform_unseen`](LeaveOneOutEncoder::transform_unseen), which applies
///   the full-sample per-category means explicitly.
/// - A category with exactly one usable training row is a *singleton*: leaving
///   its single row out leaves `0 / 0`. In the training (leave-one-out) path
///   that row is therefore encoded as the global target mean; in the
///   unseen-data path ([`transform_unseen`](LeaveOneOutEncoder::transform_unseen))
///   it is encoded with the smoothed full-sample mean (see `alpha`).
/// - Categories seen at [`transform_unseen`](LeaveOneOutEncoder::transform_unseen)
///   time but not during fit are encoded as the global target mean.
/// - Null category values are preserved as null. Rows whose target value is
///   null or non-finite are excluded from the category statistics and from the
///   global mean; such a row is encoded from the remaining usable rows of its
///   category (equivalent to the full-sample mean if no others contribute).
/// - `alpha` must be finite and non-negative; `fit` returns
///   [`Error::InvalidInput`] otherwise.
/// - All statistics are accumulated as running means (never raw sums), so
///   very large target values (e.g. `f64::MAX`) do not overflow to infinity.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder_loo::LeaveOneOutEncoder;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
/// let x = DataFrame::new(2, vec![cat])?;
/// let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
/// let y = DataFrame::new(2, vec![target])?;
///
/// let mut enc = LeaveOneOutEncoder::new();
/// enc.fit(x.clone(), y)?;
/// // Each training row is encoded with the *other* row's target (leave-one-out).
/// let enc_train = enc.transform(x)?;
/// let vals: Vec<f64> = enc_train.column("cat")?.f64()?.iter().flatten().collect();
/// assert_eq!(vals, vec![10.0, 0.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LeaveOneOutEncoder {
    fitted: bool,
    /// Smoothing strength. `0.0` disables smoothing (default).
    alpha: f64,
    /// Names of the fitted String columns, in fit order.
    column_names: Option<Vec<String>>,
    /// Full-sample per-category means, one entry per fitted column, used at
    /// transform time for data that is not the training frame.
    encodings: Option<Vec<HashMap<String, f64>>>,
    /// Global target mean over all usable (finite) rows.
    global_mean: Option<f64>,
    /// The feature frame passed to `fit`, used to detect when `transform` is
    /// applied to the training data (so the per-row LOO encodings are used).
    /// Matched by column name in any order with rows in the original order, so a
    /// column reorder of the training frame is still detected as training data.
    training_source: Option<DataFrame>,
    /// The training frame with fitted columns replaced by their per-row
    /// leave-one-out `Float64` encodings.
    training_loo: Option<DataFrame>,
}

impl LeaveOneOutEncoder {
    /// Create a new `LeaveOneOutEncoder` with smoothing `alpha = 0.0`.
    ///
    /// The encoder must be fitted with [`fit`](FitSupervised::fit), which
    /// takes both the feature data and a single-column `Float64` target.
    pub fn new() -> Self {
        Self {
            fitted: false,
            alpha: 0.0,
            column_names: None,
            encodings: None,
            global_mean: None,
            training_source: None,
            training_loo: None,
        }
    }

    /// Set the smoothing strength (default: `0.0`).
    ///
    /// `alpha` must be finite and non-negative. Larger values pull every
    /// encoding closer to the global target mean; `0.0` disables smoothing
    /// (pure per-category leave-one-out means). Invalid values are rejected at
    /// [`fit`](FitSupervised::fit) time with [`Error::InvalidInput`].
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Encode data with the full-sample per-category target means, opting in
    /// explicitly to encoding rows that were not present during fit.
    ///
    /// [`transform`](Transform::transform) rejects any frame that is not the
    /// exact training frame with [`Error::InvalidInput`], because leave-one-out
    /// is only defined for the rows seen during fit and silently returning
    /// full-sample means would leak each row's own target into the training
    /// features. `transform_unseen` is the explicit escape hatch for genuinely
    /// new (unseen) data: every category is replaced by its full-sample mean
    /// (including the row's own category contribution), categories not seen
    /// during fit are replaced by the global target mean, and null category
    /// values are preserved as null.
    ///
    /// The encoder must be fitted first; otherwise [`Error::NotFitted`] is
    /// returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use featrs::preprocessing::encoder_loo::LeaveOneOutEncoder;
    /// use featrs::traits::FitSupervised;
    /// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
    ///
    /// let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
    /// let x = DataFrame::new(2, vec![cat])?;
    /// let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
    /// let y = DataFrame::new(2, vec![target])?;
    ///
    /// let mut enc = LeaveOneOutEncoder::new();
    /// enc.fit(x, y)?;
    /// let new = DataFrame::new(2, vec![Column::from(Series::new(
    ///     "cat".into(),
    ///     &["a", "zzz"],
    /// ))])?;
    /// // Known and unseen categories both resolve to the full-sample mean (5.0).
    /// let vals: Vec<f64> = enc
    ///     .transform_unseen(new)?
    ///     .column("cat")?
    ///     .f64()?
    ///     .iter()
    ///     .flatten()
    ///     .collect();
    /// assert_eq!(vals, vec![5.0, 5.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transform_unseen(&self, x: DataFrame) -> Result<DataFrame> {
        let (names, encodings, global_mean) =
            match (&self.column_names, &self.encodings, self.global_mean) {
                (Some(n), Some(e), Some(g)) => (n, e, g),
                _ => {
                    return Err(Error::NotFitted(
                        "LeaveOneOutEncoder has not been fitted. \
                         Call .fit(x, y) before .transform_unseen()."
                            .into(),
                    ));
                }
            };

        // Full-sample per-category means: leave-one-out is impossible for rows
        // that were not present during fit. Include each row's own category
        // contribution (this is the explicit opt-in path for new data).
        let mut out = x;
        for (name, mapping) in names.iter().zip(encodings.iter()) {
            let s = out.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "LeaveOneOutEncoder.transform_unseen: column '{name}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    names.iter().collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "LeaveOneOutEncoder.transform_unseen: column '{name}' has dtype {}; \
                     expected String. {}",
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
            out.replace(name.as_str(), Column::from(series))
                .map_err(|e| {
                    Error::Computation(format!(
                        "LeaveOneOutEncoder.transform_unseen: failed to replace column \
                     '{name}'. {}",
                        e
                    ))
                })?;
        }

        Ok(out)
    }
}

impl Default for LeaveOneOutEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FitSupervised<DataFrame, DataFrame> for LeaveOneOutEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        // Reset any previously learned state so a failed re-fit cannot leave
        // stale mappings behind.
        self.fitted = false;
        self.column_names = None;
        self.encodings = None;
        self.global_mean = None;
        self.training_source = None;
        self.training_loo = None;

        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(Error::InvalidInput(format!(
                "LeaveOneOutEncoder: invalid smoothing alpha {}. \
                 Alpha must be finite and >= 0.",
                self.alpha
            )));
        }

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "LeaveOneOutEncoder.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "LeaveOneOutEncoder.fit: target must have exactly 1 column but got {} columns. \
                 Select a single target column.",
                y.width()
            )));
        }

        if y.height() != x.height() {
            return Err(Error::InvalidInput(format!(
                "LeaveOneOutEncoder.fit: feature rows ({}) and target rows ({}) don't match.",
                x.height(),
                y.height()
            )));
        }

        let y_col = &y.columns()[0];
        let y_ca = y_col.as_materialized_series().f64().map_err(|e| {
            Error::InvalidInput(format!(
                "LeaveOneOutEncoder.fit: target column '{}' has dtype {}; expected Float64. {}",
                y_col.name(),
                y_col.dtype(),
                e
            ))
        })?;

        // Global target mean over all usable (non-null, finite) rows,
        // accumulated incrementally so running sums of very large target
        // values cannot overflow to infinity.
        let y_vals: Vec<Option<f64>> = y_ca.iter().collect();
        let mut global_mean = 0.0;
        let mut global_n = 0u64;
        for opt in &y_vals {
            if let Some(v) = opt.filter(|v| v.is_finite()) {
                global_n += 1;
                global_mean += (v - global_mean) / global_n as f64;
            }
        }
        if global_n == 0 {
            return Err(Error::InvalidInput(
                "LeaveOneOutEncoder.fit: target column contains no non-null, non-NaN values. \
                 Provide a target with at least one usable value."
                    .into(),
            ));
        }
        if !global_mean.is_finite() {
            return Err(Error::InvalidInput(
                "LeaveOneOutEncoder.fit: target values accumulate to a non-finite global mean. \
                 Scale the target values before fitting."
                    .into(),
            ));
        }

        let mut names = Vec::new();
        let mut encodings = Vec::new();
        let mut loo_columns = Vec::new();

        for col in x.columns() {
            // Non-string columns are ignored; only String columns are encoded.
            if col.dtype() != &DataType::String {
                continue;
            }
            let name = col.name().to_string();
            let ca = col.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "LeaveOneOutEncoder.fit: column '{}' has dtype {}; expected String. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;

            // Per-category running mean and count over usable (finite,
            // non-null) target rows, accumulated incrementally so large
            // targets cannot overflow a sum.
            let mut stats: HashMap<String, (f64, u64)> = HashMap::new();
            for (opt_cat, opt_y) in ca.iter().zip(y_vals.iter()) {
                let (Some(cat), Some(v)) = (opt_cat, opt_y) else {
                    continue;
                };
                if !v.is_finite() {
                    continue;
                }
                let (mean, n) = stats.entry(cat.to_string()).or_insert((0.0, 0u64));
                *n += 1;
                *mean += (v - *mean) / *n as f64;
            }

            // Full-sample per-category means, used at transform time for data
            // that is not the training frame (new rows cannot be leak-free).
            // A String column with no usable (finite, non-null) target rows
            // still gets registered with an empty mapping so its output is
            // consistently `Float64` (non-null values -> global mean).
            let mapping: HashMap<String, f64> = stats
                .iter()
                .map(|(c, (mean, n))| {
                    // encoded = (n * mean + alpha * global_mean) / (n + alpha),
                    // expressed as a convex combination so it cannot overflow
                    // for finite inputs.
                    let w = *n as f64 / (*n as f64 + self.alpha);
                    let encoded = w * mean + (1.0 - w) * global_mean;
                    if encoded.is_finite() {
                        Ok((c.clone(), encoded))
                    } else {
                        Err(Error::InvalidInput(format!(
                            "LeaveOneOutEncoder.fit: column '{name}' would encode to a \
                             non-finite value for category '{c}' (n = {n}). Target values \
                             or alpha are too large; scale the target or reduce alpha."
                        )))
                    }
                })
                .collect::<Result<HashMap<String, f64>>>()?;

            // Per-row leave-one-out encodings for the training data: exclude
            // each row's own target value (if usable) from its category's
            // statistic. A singleton category (n == 1) leaves 0 / 0, so the
            // row falls back to the global target mean.
            let loo: ChunkedArray<Float64Type> = ca
                .iter()
                .zip(y_vals.iter())
                .map(|(opt_cat, opt_y)| {
                    let cat = opt_cat?;
                    let (mean, n) = stats.get(cat).copied().unwrap_or((global_mean, 0u64));
                    // Number of usable rows left after excluding this row's own
                    // target (rows with a null/non-finite target never counted).
                    let n_excl = match opt_y.filter(|v| v.is_finite()) {
                        Some(_) => n.saturating_sub(1),
                        None => n,
                    };
                    if n_excl >= 1 {
                        // Mean of the leave-one-out set, computed without
                        // forming an overflowing category sum.
                        let base = match opt_y.filter(|v| v.is_finite()) {
                            Some(v) => mean + (mean - v) / n_excl as f64,
                            None => mean,
                        };
                        // Smooth toward the global mean via a convex
                        // combination (see the mapping above).
                        let w = n_excl as f64 / (n_excl as f64 + self.alpha);
                        let loo = w * base + (1.0 - w) * global_mean;
                        if loo.is_finite() {
                            Some(loo)
                        } else {
                            Some(global_mean)
                        }
                    } else {
                        Some(global_mean)
                    }
                })
                .collect();
            let mut loo_series = loo.into_series();
            loo_series.rename(name.as_str().into());
            loo_columns.push(Column::from(loo_series));

            names.push(name);
            encodings.push(mapping);
        }

        if names.is_empty() {
            return Err(Error::InvalidInput(
                "LeaveOneOutEncoder.fit: no String columns found to encode. \
                 LeaveOneOutEncoder operates on String columns of the feature data."
                    .into(),
            ));
        }

        // Stash the leave-one-out-encoded training frame (the training data
        // with each fitted column replaced by its Float64 LOO encodings).
        let mut training_loo = x.clone();
        for (name, loo_col) in names.iter().zip(loo_columns.iter()) {
            training_loo
                .replace(name.as_str(), loo_col.clone())
                .map_err(|e| {
                    Error::Computation(format!(
                        "LeaveOneOutEncoder.fit: failed to replace column '{}'. {}",
                        name, e
                    ))
                })?;
        }

        self.column_names = Some(names);
        self.encodings = Some(encodings);
        self.global_mean = Some(global_mean);
        self.training_source = Some(x);
        self.training_loo = Some(training_loo);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for LeaveOneOutEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LeaveOneOutEncoder has not been fitted. \
                 Call .fit(x, y) before .transform()."
                    .into(),
            ));
        }
        let (names, training_source, training_loo) = match (
            &self.column_names,
            &self.training_source,
            &self.training_loo,
        ) {
            (Some(n), Some(s), Some(l)) => (n, s, l),
            _ => {
                return Err(Error::NotFitted(
                    "LeaveOneOutEncoder has not been fitted. \
                         Call .fit(x, y) before .transform()."
                        .into(),
                ));
            }
        };

        // The per-row leave-one-out encodings are only meaningful for the exact
        // training frame. Detect it by matching every training column by name,
        // value-for-value including nulls, in any order — so a column reorder of
        // the training frame still counts as training data.
        let is_training_frame = x.width() == training_source.width()
            && training_source.columns().iter().all(|sc| {
                x.column(sc.name())
                    .map(|xc| xc.equals_missing(sc))
                    .unwrap_or(false)
            });

        if !is_training_frame {
            // Reject anything that is not the training frame rather than
            // silently returning full-sample means. For a row subset of the
            // training frame — or genuinely new rows — those means would include
            // each row's own target value and leak it into the training
            // features. New data must opt in explicitly via `transform_unseen`.
            return Err(Error::InvalidInput(format!(
                "LeaveOneOutEncoder.transform requires the exact training frame \
                 (its columns may be reordered). Got a frame with {} columns x {} \
                 rows but the training frame has {} columns x {} rows. Leave-one-out \
                 is only defined for the rows present during fit, so a row subset or \
                 new data cannot be encoded without leaking each row's own target. \
                 Call .transform_unseen(x) to encode new data with full-sample \
                 per-category means.",
                x.width(),
                x.height(),
                training_source.width(),
                training_source.height()
            )));
        }

        let mut out = x;
        for name in names {
            let loo_col = training_loo.column(name.as_str()).map_err(|e| {
                Error::Computation(format!(
                    "LeaveOneOutEncoder.transform: missing stored encoding for column \
                     '{}'. {}\n        This is an internal inconsistency: the encoder was fitted \
                     successfully but has no leave-one-out column for a fitted name.",
                    name, e
                ))
            })?;
            out.replace(name.as_str(), loo_col.clone()).map_err(|e| {
                Error::Computation(format!(
                    "LeaveOneOutEncoder.transform: failed to replace column '{}'. {}",
                    name, e
                ))
            })?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_leave_one_out_excludes_own_target() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // Each row gets the *other* row's target (no self-leak).
        assert_relative_eq!(vals[0], 10.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-12);
        assert_eq!(result.column("cat").unwrap().dtype(), &DataType::Float64);
    }

    #[test]
    fn test_two_category_clear_target_difference() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a", "b", "b"]));
        let x = DataFrame::new(4, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 0.0, 10.0, 10.0]));
        let y = DataFrame::new(4, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 10.0, epsilon = 1e-12);
        assert_relative_eq!(vals[3], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_alpha_smoothing_on_leave_one_out() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new().alpha(2.0);
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // global = 5; row0: (10 - 0 + 2*5) / (1 + 2) = 20/3; row1: (10 - 10 + 10)/3 = 10/3
        assert_relative_eq!(vals[0], 20.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 10.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_singleton_category_falls_back_to_global_mean() {
        let cat = Column::from(Series::new("cat".into(), &["a"]));
        let x = DataFrame::new(1, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[5.0_f64]));
        let y = DataFrame::new(1, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // global_mean = 5; singleton row cannot leave out a usable row -> global_mean.
        assert_relative_eq!(vals[0], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_unseen_data_uses_full_sample_mean_not_loo() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x, y).unwrap();

        let new_cat = Column::from(Series::new("cat".into(), &["a", "zzz"]));
        let new_data = DataFrame::new(2, vec![new_cat]).unwrap();
        let result = enc.transform_unseen(new_data).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // "a" full-sample mean = 5.0; unseen "zzz" -> global_mean = 5.0.
        assert_relative_eq!(vals[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_unseen_category_falls_back_to_global_mean() {
        let cat = Column::from(Series::new("cat".into(), &["a", "b"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x, y).unwrap();

        let unseen = Column::from(Series::new("cat".into(), &["a", "zzz"]));
        let df = DataFrame::new(2, vec![unseen]).unwrap();
        let result = enc.transform_unseen(df).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // "a" full-sample mean = 1.0; unseen -> global_mean = 0.5.
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 0.5, epsilon = 1e-12);
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

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let cat_vals: Vec<Option<f64>> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect();
        // global = 2/3; row0 "a" target 1 -> (1 - 1) / 1 = 0; row2 "a" target 0 -> 1 / 1 = 1.
        assert_relative_eq!(cat_vals[0].unwrap(), 0.0, epsilon = 1e-12);
        assert!(cat_vals[1].is_none(), "null category must stay null");
        assert_relative_eq!(cat_vals[2].unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_null_target_rows_excluded() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a", "b"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[Some(1.0f64), None, Some(0.0)]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // global over usable (1, 0) = 0.5.
        // "a": only row0 usable -> row0 singleton -> global = 0.5.
        // row1 (null target, "a") not counted -> encoded from row0 -> 1.0.
        // "b": singleton -> global = 0.5.
        assert_relative_eq!(vals[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(vals[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_non_finite_target_values_excluded() {
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, f64::NAN, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // global = 1.0.
        // "a": sum=2, n=2. row0(target 1) -> (2-1)/1 = 1; row2(target 1) -> 1.
        // "b": NaN target not counted -> n=0 -> falls back to global = 1.0.
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

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        assert_eq!(result.width(), 3);
        assert_eq!(result.column("c1").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("c2").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("num").unwrap().dtype(), &DataType::Float64);

        let c1_vals: Vec<f64> = result
            .column("c1")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let c2_vals: Vec<f64> = result
            .column("c2")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // global = 2/3.
        // c1 "a": sum=2,n=2. row0(1)-> (2-1)/1=1; row2(1)->1. c1 "b": singleton -> g.
        // c2 "x": sum=1,n=2. row0(1)->(1-1)/1=0; row1(0)->(1-0)/1=1. c2 "y": singleton -> g.
        assert_relative_eq!(c1_vals[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c1_vals[1], 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(c1_vals[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c2_vals[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(c2_vals[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c2_vals[2], 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_transform_before_fit_returns_not_fitted() {
        let enc = LeaveOneOutEncoder::new();
        let cat = Column::from(Series::new("cat".into(), &["a"]));
        let x = DataFrame::new(1, vec![cat]).unwrap();
        match enc.transform(x) {
            Err(Error::NotFitted(_)) => {}
            other => panic!("expected NotFitted, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_fit_empty_input_returns_invalid_input() {
        let mut enc = LeaveOneOutEncoder::new();
        let x = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let y = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        match enc.fit(x, y) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!("expected InvalidInput, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_fit_invalid_alpha_rejected() {
        let cat = Column::from(Series::new("cat".into(), &["a"]));
        let x = DataFrame::new(1, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64]));
        let y = DataFrame::new(1, vec![target]).unwrap();

        let mut neg = LeaveOneOutEncoder::new().alpha(-1.0);
        match neg.fit(x.clone(), y.clone()) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput for negative alpha, got {:?}",
                other.map(|_| ())
            ),
        }

        let mut nan = LeaveOneOutEncoder::new().alpha(f64::NAN);
        match nan.fit(x, y) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput for NaN alpha, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[test]
    fn test_fit_requires_single_float_column_target() {
        let cat = Column::from(Series::new("cat".into(), &["a"]));
        let x = DataFrame::new(1, vec![cat]).unwrap();

        // Multi-column target.
        let t1 = Column::from(Series::new("y1".into(), &[1.0f64]));
        let t2 = Column::from(Series::new("y2".into(), &[2.0f64]));
        let y2 = DataFrame::new(1, vec![t1, t2]).unwrap();
        let mut enc = LeaveOneOutEncoder::new();
        match enc.fit(x.clone(), y2) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput for multi-column target, got {:?}",
                other.map(|_| ())
            ),
        }

        // Non-f64 target.
        let t = Column::from(Series::new("y".into(), &["a"]));
        let ys = DataFrame::new(1, vec![t]).unwrap();
        let mut enc = LeaveOneOutEncoder::new();
        match enc.fit(x, ys) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput for non-f64 target, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[test]
    fn test_refit_overwrites_prior_state() {
        let c1 = Column::from(Series::new("cat".into(), &["a", "b"]));
        let x1 = DataFrame::new(2, vec![c1]).unwrap();
        let t1 = Column::from(Series::new("y".into(), &[1.0f64, 0.0]));
        let y1 = DataFrame::new(2, vec![t1]).unwrap();

        let c2 = Column::from(Series::new("cat".into(), &["q", "q"]));
        let x2 = DataFrame::new(2, vec![c2]).unwrap();
        let t2 = Column::from(Series::new("y".into(), &[2.0f64, 8.0]));
        let y2 = DataFrame::new(2, vec![t2]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x1, y1).unwrap();
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
        assert_relative_eq!(vals[0], 8.0, epsilon = 1e-12);
        assert_relative_eq!(vals[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_string_column_with_no_usable_targets_is_still_encoded() {
        // cat_b has no usable target rows at all, but must still become Float64.
        let cat_a = Column::from(Series::new("cat_a".into(), &["a", "b"]));
        let cat_b = Column::from(Series::new("cat_b".into(), &["p", "q"]));
        let x = DataFrame::new(2, vec![cat_a, cat_b]).unwrap();
        let target = Column::from(Series::new("y".into(), &[Some(5.0f64), None]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();
        let result = enc.transform(x).unwrap();

        assert_eq!(result.column("cat_a").unwrap().dtype(), &DataType::Float64);
        assert_eq!(result.column("cat_b").unwrap().dtype(), &DataType::Float64);
        // global = 5. cat_b has no usable stats -> all rows global mean; nulls preserved.
        let b_vals: Vec<Option<f64>> = result
            .column("cat_b")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect();
        assert_relative_eq!(b_vals[0].unwrap(), 5.0, epsilon = 1e-12);
        assert_relative_eq!(b_vals[1].unwrap(), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_large_target_values_do_not_overflow() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[f64::MAX, f64::MAX]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x, y).unwrap(); // true mean is f64::MAX (finite); must not return InvalidInput.

        let new_cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let new_data = DataFrame::new(2, vec![new_cat]).unwrap();
        let result = enc.transform_unseen(new_data).unwrap();
        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert!(vals[0].is_finite());
        assert!(vals[1].is_finite());
    }

    #[test]
    fn test_row_subset_of_fit_frame_is_rejected() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a", "b"]));
        let x = DataFrame::new(3, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0, 5.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();

        // A row subset of the training frame must not fall back to full-sample
        // means (which would leak each row's own target); it errors instead.
        let subset = x.slice(0, 2);
        match enc.transform(subset) {
            Err(Error::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput for a row subset, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[test]
    fn test_reordered_columns_same_values_as_training_frame() {
        let c1 = Column::from(Series::new("c1".into(), &["a", "b", "a"]));
        let c2 = Column::from(Series::new("c2".into(), &["x", "x", "y"]));
        let x = DataFrame::new(3, vec![c1, c2]).unwrap();
        let target = Column::from(Series::new("y".into(), &[1.0_f64, 0.0, 1.0]));
        let y = DataFrame::new(3, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x.clone(), y).unwrap();

        let baseline = enc.transform(x.clone()).unwrap();

        // Reorder the columns (same data, different layout): the per-row LOO
        // encodings must be identical to the fit-frame transform.
        let reordered = DataFrame::new(
            3,
            vec![
                x.column("c2").unwrap().clone(),
                x.column("c1").unwrap().clone(),
            ],
        )
        .unwrap();
        let result = enc.transform(reordered).unwrap();

        let names = result.get_column_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0].as_str(), "c2");
        assert_eq!(names[1].as_str(), "c1");

        let base_c1: Vec<f64> = baseline
            .column("c1")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let got_c1: Vec<f64> = result
            .column("c1")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let base_c2: Vec<f64> = baseline
            .column("c2")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let got_c2: Vec<f64> = result
            .column("c2")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        // The stored LOO columns are copied verbatim, so values are identical.
        assert_eq!(base_c1, got_c1);
        assert_eq!(base_c2, got_c2);
    }

    #[test]
    fn test_single_row_known_category_via_unseen() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x, y).unwrap();

        // A single row of a known category resolves to its full-sample mean.
        let single =
            DataFrame::new(1, vec![Column::from(Series::new("cat".into(), &["a"]))]).unwrap();
        let result = enc.transform_unseen(single).unwrap();
        let vals: Vec<f64> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_transform_unseen_preserves_null_categories() {
        let cat = Column::from(Series::new("cat".into(), &["a", "a"]));
        let x = DataFrame::new(2, vec![cat]).unwrap();
        let target = Column::from(Series::new("y".into(), &[0.0_f64, 10.0]));
        let y = DataFrame::new(2, vec![target]).unwrap();

        let mut enc = LeaveOneOutEncoder::new();
        enc.fit(x, y).unwrap();

        let unseen = DataFrame::new(
            2,
            vec![Column::from(Series::new("cat".into(), &[Some("a"), None]))],
        )
        .unwrap();
        let result = enc.transform_unseen(unseen).unwrap();
        let vals: Vec<Option<f64>> = result
            .column("cat")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect();
        assert_relative_eq!(vals[0].unwrap(), 5.0, epsilon = 1e-12);
        assert!(
            vals[1].is_none(),
            "null category must stay null in unseen data"
        );
    }

    #[test]
    fn test_transform_unseen_before_fit_returns_not_fitted() {
        let enc = LeaveOneOutEncoder::new();
        let cat = Column::from(Series::new("cat".into(), &["a"]));
        let x = DataFrame::new(1, vec![cat]).unwrap();
        match enc.transform_unseen(x) {
            Err(Error::NotFitted(_)) => {}
            other => panic!("expected NotFitted, got {:?}", other.map(|_| ())),
        }
    }
}
