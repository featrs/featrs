//! Ratio feature generation from numeric column pairs.
//!
//! [`RatioFeatures`] creates the element-wise ratio `a / b` for every pair
//! of input columns, with an additive `epsilon` floor on the divisor to
//! avoid division-by-zero. Useful for feature engineering where the relative
//! proportion between two quantities is informative (e.g. price/earnings,
//! clicks/impressions, bedroom_count/square_footage).
//!
//! # Output semantics
//!
//! The original input columns are **preserved** and the new ratio columns
//! are **appended** in pair order. For input columns `[a, b, c]` the
//! default output columns are `[a, b, c, a_div_b, a_div_c, b_div_c]`.
//! Pairs are emitted in **lexicographic column order** for deterministic
//! output, regardless of the order columns appear in the fitted frame or
//! the order they were supplied via [`columns`](RatioFeatures::columns).
//! With [`include_reciprocal`](RatioFeatures::include_reciprocal) enabled,
//! the reciprocal `b / a` is emitted right after each `a / b`:
//! `[a, b, c, a_div_b, b_div_a, a_div_c, c_div_a, b_div_c, c_div_b]`.
//!
//! # Division by zero
//!
//! Each ratio is computed as `col_i / (col_j + epsilon)` with
//! `epsilon = 1e-12` by default, so an exact-zero divisor yields a very
//! large but finite value instead of `NaN`/`Inf`. Setting
//! [`epsilon(0.0)`](RatioFeatures::epsilon) disables the floor: divisions
//! by zero then produce `±Inf`/`NaN` per IEEE-754 semantics. The floor
//! slightly biases ratios whose divisor is very small.
//!
//! The floor is sign-aware: it is applied with the divisor's sign
//! (`col_j + copysign(epsilon, col_j)`), so divisors in `(-epsilon, 0)`
//! stay negative and ratios keep their correct sign, and the floor can
//! never be cancelled by a divisor of exactly `-epsilon`.
//!
//! # Null and `NaN` propagation
//!
//! A null dividend or divisor produces a null output (a null divisor stays
//! null after adding `epsilon`); `NaN` values propagate through the
//! underlying `f64` division.
//!
//! # Name collisions
//!
//! Generated names use the `_div_` separator (e.g. `a_div_b`). If an input
//! column already carries a generated name — for example the input already
//! contains a column literally named `a_div_b` — [`transform`](Transform::transform)
//! returns [`Error::InvalidInput`] instead of silently overwriting it.
//! Similarly, input column names that themselves contain `_div_` (e.g.
//! `p_div_q`) can make two distinct pairs produce the same generated name;
//! that is also rejected. Rename such columns before fitting.

use std::collections::HashSet;

use polars::prelude::*;

use crate::traits::FeatureNamesOut;
use crate::traits::output_names::{ensure_fitted, with_generated_names};
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{require_f64_columns, series_div};

/// Suffix inserted between the two source column names to form a ratio
/// column name (e.g. `a_div_b`).
///
/// Note: if an input column name itself contains `_div_`, generated names
/// may be ambiguous and are rejected at transform time. Rename such inputs
/// before fitting.
const NAME_SEP: &str = "_div_";

/// Create ratio features (`a / b`) from pairs of numeric columns.
///
/// Stateless in the sense that no numeric parameters are learned; [`fit`](Fit::fit)
/// resolves and validates the column set, and [`transform`](Transform::transform)
/// emits the element-wise ratio for every pair. Implements [`Fit`] and
/// [`Transform`] on [`DataFrame`].
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::ratio_features::RatioFeatures;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
/// let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut rf = RatioFeatures::new();
/// rf.fit(df.clone())?;
/// let out = rf.transform(df)?;
/// // a, b, a_div_b = [3.0, 4.0, 3.0]
/// assert_eq!(out.width(), 3);
/// let v = out.column("a_div_b")?.f64()?.get(1).unwrap();
/// assert!((v - 4.0).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RatioFeatures {
    fitted: bool,
    /// User-configured columns; empty means auto-discover all `Float64`
    /// columns at fit time.
    columns: Vec<String>,
    /// Resolved column list (lexicographically sorted) stored at fit time.
    fitted_columns: Vec<String>,
    include_reciprocal: bool,
    epsilon: f64,
}

impl RatioFeatures {
    /// Create a new `RatioFeatures` transformer.
    ///
    /// Defaults: auto-discover `Float64` columns at fit time, only `a / b`
    /// pairs (`include_reciprocal = false`), and an `epsilon` floor of
    /// `1e-12` on the divisor.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: Vec::new(),
            fitted_columns: Vec::new(),
            include_reciprocal: false,
            epsilon: 1e-12,
        }
    }

    /// Restrict ratio generation to the named columns.
    ///
    /// When omitted, the transformer auto-discovers all `Float64` columns
    /// at [`fit`](Fit::fit) time. An empty slice also selects
    /// auto-discovery.
    ///
    /// Each column must exist in the frame passed to `fit` and have dtype
    /// `Float64`; otherwise `fit` returns [`Error::InvalidInput`].
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = cols.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Whether to also emit the reciprocal ratio `b / a` for every pair.
    ///
    /// Default: `false` — only `a / b` for pairs `(i, j)` with `i < j` in
    /// lexicographic column order. When `true`, `b / a` is emitted
    /// immediately after each `a / b`.
    pub fn include_reciprocal(mut self, b: bool) -> Self {
        self.include_reciprocal = b;
        self
    }

    /// Set the additive floor added to the divisor before dividing.
    ///
    /// Default: `1e-12`. Each ratio is `col_i / (col_j + epsilon)`, so an
    /// exact-zero divisor yields a very large but finite value instead of
    /// `NaN`/`Inf`. Set `0.0` to disable the floor: divisions by zero then
    /// produce `±Inf`/`NaN` per IEEE-754 semantics.
    ///
    /// The floor is sign-aware (applied as `copysign(epsilon, col_j)`), so
    /// small negative divisors keep their sign instead of being flipped
    /// positive.
    ///
    /// `epsilon` must be finite and non-negative; anything else makes
    /// [`fit`](Fit::fit) return [`Error::InvalidInput`].
    pub fn epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
        self
    }
}

impl Default for RatioFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for RatioFeatures {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale columns
        // behind that `transform` would otherwise use.
        self.fitted = false;
        self.fitted_columns = Vec::new();

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "RatioFeatures.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }

        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(Error::InvalidInput(format!(
                "RatioFeatures: epsilon must be a finite, non-negative number; got {}. \
                 Use 0.0 to disable the divisor floor, or keep the default 1e-12.",
                self.epsilon
            )));
        }

        let resolved = if self.columns.is_empty() {
            require_f64_columns(&x, "RatioFeatures")?
        } else {
            for col in &self.columns {
                let c = x.column(col.as_str()).map_err(|e| {
                    Error::InvalidInput(format!("RatioFeatures.fit: column '{col}' not found. {e}"))
                })?;
                if c.dtype() != &DataType::Float64 {
                    return Err(Error::InvalidInput(format!(
                        "RatioFeatures.fit: column '{col}' has dtype {}; expected Float64.",
                        c.dtype()
                    )));
                }
            }
            self.columns.clone()
        };

        // Lexicographic order for deterministic pair emission; dedupe so a
        // repeated name (e.g. `.columns(&["a", "a"])`) cannot produce a
        // self-ratio `a_div_a` or a spurious name collision.
        let mut fitted = resolved;
        fitted.sort();
        fitted.dedup();
        self.fitted_columns = fitted;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for RatioFeatures {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "RatioFeatures has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();
        let who = "RatioFeatures";

        // Verify every fitted column exists up front, so the n < 2
        // pass-through behaves the same as the pair-generating path.
        for name in &self.fitted_columns {
            out.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "{who}.transform: column '{name}' not found. The transformer was \
                     fitted on columns: {:?}. {e}",
                    self.fitted_columns
                ))
            })?;
        }

        let n = self.fitted_columns.len();
        if n < 2 {
            // Fewer than two columns → no pairs → pass-through.
            return Ok(out);
        }

        // (i, j) index pairs into the fitted column list; j strictly greater
        // than i. With include_reciprocal, (j, i) is appended after (i, j).
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j));
                if self.include_reciprocal {
                    pairs.push((j, i));
                }
            }
        }

        // Reject name collisions up front: `DataFrame::with_column` silently
        // REPLACES an existing column with the same name, so without this
        // check a collision would silently destroy input data. The `used`
        // set starts with the input columns and also catches two distinct
        // pairs that produce the same generated name (possible when input
        // names contain `_div_`).
        let mut used: HashSet<String> = out
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for (i, j) in &pairs {
            let out_name = format!(
                "{}{}{}",
                self.fitted_columns[*i], NAME_SEP, self.fitted_columns[*j]
            );
            if !used.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "{who}.transform: generated column '{out_name}' collides with an \
                     existing input column or a previously generated column. Rename the \
                     conflicting column(s) or choose different column names."
                )));
            }
        }

        for (i, j) in pairs {
            let name_i = &self.fitted_columns[i];
            let name_j = &self.fitted_columns[j];

            let s_i = out
                .column(name_i.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "{who}.transform: column '{name_i}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.fitted_columns
                    ))
                })?
                .as_materialized_series();
            let s_j = out
                .column(name_j.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "{who}.transform: column '{name_j}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.fitted_columns
                    ))
                })?
                .as_materialized_series();

            let mut ratio = series_div(s_i, s_j, self.epsilon, who)?;
            let out_name = format!("{name_i}{NAME_SEP}{name_j}");
            ratio.rename(out_name.as_str().into());

            out.with_column(ratio.into())
                .map_err(|e| Error::Computation(format!("{who}.transform: {e}")))?;
        }

        Ok(out)
    }
}

impl FeatureNamesOut for RatioFeatures {
    /// The input columns plus each generated `_div_` ratio, in pair order.
    fn feature_names_out(&self, input_columns: &[String]) -> Result<Vec<String>> {
        ensure_fitted(self.fitted, "RatioFeatures")?;
        let n = self.fitted_columns.len();
        let mut generated = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                generated.push(format!(
                    "{}{}{}",
                    self.fitted_columns[i], NAME_SEP, self.fitted_columns[j]
                ));
                if self.include_reciprocal {
                    generated.push(format!(
                        "{}{}{}",
                        self.fitted_columns[j], NAME_SEP, self.fitted_columns[i]
                    ));
                }
            }
        }
        Ok(with_generated_names(input_columns, generated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_two_col_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    fn make_three_col_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
        let c = Column::from(Series::new("c".into(), &[3.0_f64, 4.0, 2.0]));
        DataFrame::new(3, vec![a, b, c]).unwrap()
    }

    fn ratio_values(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
        df.column(name).unwrap().f64().unwrap().iter().collect()
    }

    #[test]
    fn test_ratio_two_columns_default() {
        let df = make_two_col_df();
        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        // a, b  +  a_div_b
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);
        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 3.0, epsilon = 1e-9);
        assert_relative_eq!(vals[1].unwrap(), 4.0, epsilon = 1e-9);
        assert_relative_eq!(vals[2].unwrap(), 3.0, epsilon = 1e-9);
        assert!(
            out.column("b_div_a").is_err(),
            "reciprocal must be off by default"
        );
    }

    #[test]
    fn test_ratio_include_reciprocal() {
        let df = make_two_col_df();
        let mut rf = RatioFeatures::new().include_reciprocal(true);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        // a, b  +  a_div_b, b_div_a
        assert_eq!(out.width(), 4);
        let fwd = ratio_values(&out, "a_div_b");
        let rev = ratio_values(&out, "b_div_a");
        assert_relative_eq!(fwd[0].unwrap(), 3.0, epsilon = 1e-9);
        assert_relative_eq!(rev[0].unwrap(), 1.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(rev[1].unwrap(), 1.0 / 4.0, epsilon = 1e-9);
        assert_relative_eq!(rev[2].unwrap(), 1.0 / 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ratio_three_columns_three_pairs() {
        let df = make_three_col_df();
        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        // a, b, c  +  a_div_b, a_div_c, b_div_c
        assert_eq!(out.width(), 6);
        assert!(out.column("a_div_b").is_ok());
        assert!(out.column("a_div_c").is_ok());
        assert!(out.column("b_div_c").is_ok());
        // no self-ratios and no reciprocals
        assert!(out.column("a_div_a").is_err());
        assert!(out.column("b_div_a").is_err());
    }

    #[test]
    fn test_ratio_division_by_zero_epsilon_floor() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 0.0, 0.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        let eps = 1e-12;
        assert_relative_eq!(vals[0].unwrap(), 1.0 / eps, epsilon = 1e-6);
        assert_relative_eq!(vals[1].unwrap(), 2.0 / eps, epsilon = 1e-6);
        assert_relative_eq!(vals[2].unwrap(), 3.0 / eps, epsilon = 1e-6);
        for v in vals.iter().flatten() {
            assert!(v.is_finite(), "epsilon floor must keep ratios finite");
        }
    }

    #[test]
    fn test_ratio_epsilon_zero_produces_inf() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, -1.0, 2.0]));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 0.0, 0.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new().epsilon(0.0);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_eq!(vals[0].unwrap(), f64::INFINITY);
        assert_eq!(vals[1].unwrap(), f64::NEG_INFINITY);
        assert_eq!(vals[2].unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_ratio_null_propagation() {
        let a = Column::from(Series::new("a".into(), &[Some(6.0_f64), None, Some(18.0)]));
        let b = Column::from(Series::new("b".into(), &[Some(2.0_f64), Some(3.0), None]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 3.0, epsilon = 1e-9);
        assert!(vals[1].is_none(), "null dividend must stay null");
        assert!(vals[2].is_none(), "null divisor must stay null");
    }

    #[test]
    fn test_ratio_nan_propagation() {
        let a = Column::from(Series::new("a".into(), &[6.0_f64, f64::NAN, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 3.0, epsilon = 1e-9);
        assert!(vals[1].unwrap().is_nan(), "NaN dividend must propagate");
        assert_relative_eq!(vals[2].unwrap(), 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ratio_identical_columns_is_one() {
        let a = Column::from(Series::new("a".into(), &[5.0_f64, 5.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[5.0_f64, 5.0, 5.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(vals[1].unwrap(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(vals[2].unwrap(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ratio_negative_values_signed() {
        let a = Column::from(Series::new("a".into(), &[-6.0_f64, 12.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), -3.0, epsilon = 1e-9);
        assert_relative_eq!(vals[1].unwrap(), 4.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ratio_lexicographic_pair_order() {
        // Supplied in reverse order; lexicographic sort must still yield a_div_b.
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new().columns(&["b", "a"]);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        assert!(out.column("a_div_b").is_ok(), "pairs must be lexicographic");
        assert!(out.column("b_div_a").is_err());
    }

    #[test]
    fn test_ratio_explicit_columns() {
        let df = make_three_col_df();
        let mut rf = RatioFeatures::new().columns(&["a", "c"]);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        // a, b, c  +  a_div_c  (b not selected)
        assert_eq!(out.width(), 4);
        assert!(out.column("a_div_c").is_ok());
        assert!(out.column("a_div_b").is_err());
    }

    #[test]
    fn test_ratio_explicit_missing_column_errors() {
        let df = make_two_col_df();
        let mut rf = RatioFeatures::new().columns(&["a", "missing"]);
        let err = rf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_ratio_explicit_non_f64_column_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a, cat]).unwrap();

        let mut rf = RatioFeatures::new().columns(&["a", "cat"]);
        let err = rf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_ratio_auto_discovery_skips_non_f64() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![a, cat, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        // a, cat, b  +  a_div_b  (cat skipped)
        assert_eq!(out.width(), 4);
        assert!(out.column("a_div_b").is_ok());
    }

    #[test]
    fn test_ratio_auto_discovery_no_f64_errors() {
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![cat]).unwrap();

        let mut rf = RatioFeatures::new();
        let err = rf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_ratio_single_column_no_new_cols() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn test_ratio_empty_dataframe_rejected() {
        let df = DataFrame::empty();
        let mut rf = RatioFeatures::new();
        let err = rf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_ratio_not_fitted_error() {
        let df = make_two_col_df();
        let rf = RatioFeatures::new();
        let err = rf.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_ratio_name_collision_errors() {
        // Input already contains a column literally named a_div_b.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let clash = Column::from(Series::new("a_div_b".into(), &[7.0_f64, 8.0, 9.0]));
        let df = DataFrame::new(3, vec![a, b, clash]).unwrap();

        let mut rf = RatioFeatures::new().columns(&["a", "b"]);
        rf.fit(df.clone()).unwrap();
        let err = rf.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "collision with an existing column must error, not overwrite"
        );
        assert!(err.to_string().contains("a_div_b"));
    }

    #[test]
    fn test_ratio_separator_in_names_still_generates() {
        // Column names containing the separator produce distinct names here;
        // this is the non-colliding separator case.
        let p = Column::from(Series::new("p".into(), &[1.0_f64, 2.0, 3.0]));
        let q_div_r = Column::from(Series::new("q_div_r".into(), &[4.0_f64, 5.0, 6.0]));
        let r = Column::from(Series::new("r".into(), &[7.0_f64, 8.0, 9.0]));
        let df = DataFrame::new(3, vec![p, q_div_r, r]).unwrap();

        // fitted lexicographic: [p, q_div_r, r]
        // pairs: (p, q_div_r) -> p_div_q_div_r, (p, r) -> p_div_r,
        //        (q_div_r, r) -> q_div_r_div_r
        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();
        assert!(out.column("p_div_q_div_r").is_ok());
        assert!(out.column("p_div_r").is_ok());
        assert!(out.column("q_div_r_div_r").is_ok());
    }

    #[test]
    fn test_ratio_generated_name_collision_errors() {
        // Two distinct pairs produce the same generated name because the
        // input names contain the separator: (a, b_div_c) and (a_div_b, c)
        // both map to a_div_b_div_c. Must error, not overwrite.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let a_div_b = Column::from(Series::new("a_div_b".into(), &[4.0_f64, 5.0, 6.0]));
        let b_div_c = Column::from(Series::new("b_div_c".into(), &[7.0_f64, 8.0, 9.0]));
        let c = Column::from(Series::new("c".into(), &[10.0_f64, 11.0, 12.0]));
        let df = DataFrame::new(3, vec![a, a_div_b, b_div_c, c]).unwrap();

        // fitted lexicographic: [a, a_div_b, b_div_c, c]
        let mut rf = RatioFeatures::new().columns(&["a", "a_div_b", "b_div_c", "c"]);
        rf.fit(df.clone()).unwrap();
        let err = rf.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "generated-vs-generated name collision must error"
        );
        assert!(err.to_string().contains("a_div_b_div_c"));
    }

    #[test]
    fn test_ratio_duplicate_config_columns_deduped() {
        // Repeating a column in .columns() must not produce a self-ratio or
        // a spurious collision.
        let df = make_two_col_df();
        let mut rf = RatioFeatures::new().columns(&["a", "a", "b"]);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df.clone()).unwrap();

        assert_eq!(out.width(), 3, "deduped [a, b] must yield exactly a_div_b");
        assert!(out.column("a_div_b").is_ok());
        assert!(out.column("a_div_a").is_err(), "no self-ratios");

        // Same config with reciprocal must not hit a bogus collision either.
        let mut rf = RatioFeatures::new()
            .columns(&["a", "a", "b"])
            .include_reciprocal(true);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();
        assert_eq!(out.width(), 4);
        assert!(out.column("a_div_b").is_ok());
        assert!(out.column("b_div_a").is_ok());
    }

    #[test]
    fn test_ratio_zero_over_zero_epsilon_zero_is_nan() {
        let a = Column::from(Series::new("a".into(), &[0.0_f64, 1.0]));
        let b = Column::from(Series::new("b".into(), &[0.0_f64, 0.0]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new().epsilon(0.0);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert!(vals[0].unwrap().is_nan(), "0/0 must be NaN with epsilon=0");
        assert_eq!(vals[1].unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_ratio_nan_divisor_propagates() {
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, f64::NAN, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 3.0, epsilon = 1e-9);
        assert!(vals[1].unwrap().is_nan(), "NaN divisor must propagate");
        assert_relative_eq!(vals[2].unwrap(), 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ratio_inf_divisor_propagates() {
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, f64::INFINITY]));
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        assert_relative_eq!(vals[0].unwrap(), 3.0, epsilon = 1e-9);
        assert_eq!(vals[1].unwrap(), 0.0, "finite / Inf must be 0.0");
    }

    #[test]
    fn test_ratio_neg_divisor_in_floor_interval_keeps_sign() {
        // Sign-aware floor (CodeRabbit #114): divisors in (-epsilon, 0)
        // must stay negative so ratios keep their correct sign, and the
        // floor is never cancelled by a divisor of exactly -epsilon.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, -1.0, 6.0]));
        let b = Column::from(Series::new("b".into(), &[-5e-13_f64, -5e-13, -1e-12]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let vals = ratio_values(&out, "a_div_b");
        // 1.0 / (-5e-13 - 1e-12) = 1.0 / -1.5e-12: negative and finite.
        assert!(
            vals[0].unwrap().is_sign_negative(),
            "positive / small-negative divisor must stay negative"
        );
        assert!(vals[0].unwrap().is_finite());
        // -1.0 / -1.5e-12: positive.
        assert!(vals[1].unwrap().is_sign_positive());
        // 6.0 / (-1e-12 - 1e-12) = 6.0 / -2e-12: no cancellation, finite.
        assert!(vals[2].unwrap().is_sign_negative());
        assert!(vals[2].unwrap().is_finite());
        assert_relative_eq!(vals[2].unwrap(), 6.0 / -2e-12, epsilon = 1e-6);
    }

    #[test]
    fn test_ratio_reciprocal_output_order() {
        // With include_reciprocal, b_div_a must come immediately after a_div_b.
        let a = Column::from(Series::new("a".into(), &[6.0_f64, 12.0, 18.0]));
        let b = Column::from(Series::new("b".into(), &[2.0_f64, 3.0, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new().include_reciprocal(true);
        rf.fit(df.clone()).unwrap();
        let out = rf.transform(df).unwrap();

        let names: Vec<&str> = out.get_column_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(names, &["a", "b", "a_div_b", "b_div_a"]);
    }

    #[test]
    fn test_ratio_transform_missing_fitted_column_errors() {
        // Transform receives a frame that lacks a fitted column.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();

        let only_a = DataFrame::new(3, vec![df.column("a").unwrap().clone()]).unwrap();
        let err = rf.transform(only_a).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "missing fitted column must error, not silently pass through"
        );
    }

    #[test]
    fn test_ratio_single_column_transform_missing_errors() {
        // The n < 2 pass-through must still verify the fitted column exists.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut rf = RatioFeatures::new().columns(&["a"]);
        rf.fit(df.clone()).unwrap();

        let other = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let other_df = DataFrame::new(3, vec![other]).unwrap();
        let err = rf.transform(other_df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "n < 2 path must still reject a missing fitted column"
        );
    }

    #[test]
    fn test_ratio_refit_resets_state() {
        let df = make_two_col_df();
        let mut rf = RatioFeatures::new();
        rf.fit(df.clone()).unwrap();
        assert!(rf.transform(df.clone()).is_ok());

        // A failed re-fit must leave the transformer unfitted.
        let empty = DataFrame::empty();
        assert!(rf.fit(empty).is_err());
        let err = rf.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_ratio_epsilon_invalid_rejected() {
        let df = make_two_col_df();

        let mut rf = RatioFeatures::new().epsilon(f64::NAN);
        let err = rf.fit(df.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let mut rf = RatioFeatures::new().epsilon(f64::INFINITY);
        let err = rf.fit(df.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));

        let mut rf = RatioFeatures::new().epsilon(-1.0);
        let err = rf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}
