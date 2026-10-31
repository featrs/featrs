//! Polynomial feature generation.
//!
//! [`PolynomialFeatures`] generates all polynomial combinations of the
//! input features up to a specified degree. Analogous to
//! `sklearn.preprocessing.PolynomialFeatures`.

use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{require_f64_columns, series_mul, series_pow};
use polars::prelude::*;
use std::collections::HashSet;

/// Generate polynomial and interaction features.
///
/// Creates new features that are polynomial combinations of the original
/// features. For example, with `degree=2` and inputs `[a, b]`, the output
/// includes `[a, b, a², a·b, b²]` plus an optional bias term.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::polynomial_features::PolynomialFeatures;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
/// let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut pf = PolynomialFeatures::new(2)?
///     .include_bias(true)
///     .interaction_only(false);
/// pf.fit(df.clone())?;
/// let result = pf.transform(df)?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PolynomialFeatures {
    fitted: bool,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
    input_columns: Option<Vec<String>>,
}

impl PolynomialFeatures {
    /// Create a new `PolynomialFeatures` with the given maximum degree.
    ///
    /// Returns [`Error::InvalidInput`] if `degree` is `0`.
    pub fn new(degree: usize) -> Result<Self> {
        if degree == 0 {
            return Err(Error::InvalidInput(format!(
                "PolynomialFeatures::new: degree must be >= 1, got {degree}"
            )));
        }
        Ok(Self {
            fitted: false,
            degree,
            interaction_only: false,
            include_bias: true,
            input_columns: None,
        })
    }

    /// Create a builder for more ergonomic configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use featrs::preprocessing::polynomial_features::PolynomialFeatures;
    ///
    /// let _pf = PolynomialFeatures::builder()
    ///     .degree(3)
    ///     .include_bias(false)
    ///     .interaction_only(true)
    ///     .build()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> PolynomialFeaturesBuilder {
        PolynomialFeaturesBuilder::default()
    }

    /// Whether to include only interaction features (`a·b`, `a·b·c`, ...)
    /// and exclude pure powers (`a²`, `a³`, ...). Default: `false`.
    pub fn interaction_only(mut self, value: bool) -> Self {
        self.interaction_only = value;
        self
    }

    /// Whether to include a bias column (all ones). Default: `true`.
    pub fn include_bias(mut self, value: bool) -> Self {
        self.include_bias = value;
        self
    }

    /// The output column name for one power vector, or `None` for the
    /// all-zero vector (which `transform` skips). `fit` uses this to reject
    /// colliding names, so it must stay the single source of truth.
    fn term_name(power: &[usize], input_columns: &[String]) -> Option<String> {
        // Only the all-zero power vector produces no term; `transform` skips
        // it. An input column whose name happens to be empty still yields a
        // (possibly empty) term name, matching the pre-refactor behaviour.
        if power.iter().all(|&p| p == 0) {
            return None;
        }
        let mut name = power
            .iter()
            .zip(input_columns)
            .filter(|(p, _)| **p > 0)
            .map(|(_, col)| col.as_str())
            .collect::<Vec<_>>()
            .join("_");
        if power.iter().filter(|&&p| p > 0).count() > 1 {
            name.push_str("_inter");
        } else if power.iter().any(|&p| p > 1) {
            name.push('^');
        }
        Some(name)
    }

    fn generate_powers(
        n_features: usize,
        degree: usize,
        interaction_only: bool,
    ) -> Vec<Vec<usize>> {
        let mut result = Vec::new();

        if interaction_only {
            let max_mask = 1usize << n_features;
            for mask in 0..max_mask {
                let sum_bits = mask.count_ones() as usize;
                if sum_bits >= 2 && sum_bits <= degree {
                    let mut powers = vec![0usize; n_features];
                    for (j, cell) in powers.iter_mut().enumerate() {
                        if mask & (1 << j) != 0 {
                            *cell = 1;
                        }
                    }
                    result.push(powers);
                }
            }
            result.sort_by_key(|p| p.iter().sum::<usize>());
        } else {
            fn recurse(
                result: &mut Vec<Vec<usize>>,
                current: &mut Vec<usize>,
                idx: usize,
                remaining: usize,
                n: usize,
                max_degree: usize,
            ) {
                if idx == n {
                    let total: usize = current.iter().sum();
                    if total >= 1 && total <= max_degree {
                        result.push(current.clone());
                    }
                    return;
                }
                let max_power = remaining.min(max_degree);
                for p in 0..=max_power {
                    current.push(p);
                    let new_remaining = remaining.saturating_sub(p);
                    recurse(result, current, idx + 1, new_remaining, n, max_degree);
                    current.pop();
                }
            }

            let mut current = Vec::with_capacity(n_features);
            recurse(&mut result, &mut current, 0, degree, n_features, degree);

            result.sort_by_key(|p| (p.iter().sum::<usize>(), p.clone()));
        }

        result
    }
}

/// Builder for [`PolynomialFeatures`].
///
/// Provides a more ergonomic way to configure polynomial feature generation.
#[derive(Default)]
pub struct PolynomialFeaturesBuilder {
    degree: Option<usize>,
    interaction_only: bool,
    include_bias: bool,
}

impl PolynomialFeaturesBuilder {
    /// Set the maximum degree of polynomial features. Required.
    ///
    /// `degree == 0` is rejected at [`build`](PolynomialFeaturesBuilder::build) time.
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = Some(degree);
        self
    }

    /// Only include interaction terms (exclude pure powers). Default: `false`.
    pub fn interaction_only(mut self, value: bool) -> Self {
        self.interaction_only = value;
        self
    }

    /// Include a bias (intercept) column. Default: `true`.
    pub fn include_bias(mut self, value: bool) -> Self {
        self.include_bias = value;
        self
    }

    /// Build the `PolynomialFeatures` instance.
    ///
    /// Returns [`Error::InvalidInput`] if `degree` was not set or is `0`.
    pub fn build(self) -> Result<PolynomialFeatures> {
        let degree = self.degree.ok_or_else(|| {
            Error::InvalidInput(
                "PolynomialFeaturesBuilder: degree is required. Call .degree(n) before .build()."
                    .into(),
            )
        })?;
        if degree == 0 {
            return Err(Error::InvalidInput(
                "PolynomialFeaturesBuilder: degree must be >= 1, got 0".into(),
            ));
        }
        Ok(PolynomialFeatures {
            fitted: false,
            degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            input_columns: None,
        })
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        // `new(2)` always succeeds; construct directly to avoid unwrapping a
        // `Result` in `Default::default`.
        Self {
            fitted: false,
            degree: 2,
            interaction_only: false,
            include_bias: true,
            input_columns: None,
        }
    }
}

impl Fit<DataFrame> for PolynomialFeatures {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state up front so a failed (re)fit cannot leave a stale
        // column plan or fitted flag behind.
        self.fitted = false;
        self.input_columns = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "PolynomialFeatures.fit received an empty DataFrame (0 rows or 0 columns).".into(),
            ));
        }
        let col_names = require_f64_columns(&x, "PolynomialFeatures")?;
        if self.include_bias && col_names.iter().any(|c| c == "bias") {
            return Err(Error::InvalidInput(
                "PolynomialFeatures: input contains a column named 'bias', which collides \
                 with the synthetic bias column. Rename the input column or set \
                 include_bias(false)."
                    .into(),
            ));
        }
        // Reject name collisions up front: `transform` names each term with
        // `term_name`, and two equal names (e.g. an input column called
        // `a_b_inter` next to the `a * b` interaction) only fail later, inside
        // a fitted pipeline, as an opaque duplicate-column error.
        let powers = Self::generate_powers(col_names.len(), self.degree, self.interaction_only);
        let mut seen: HashSet<String> = HashSet::new();
        if self.include_bias {
            seen.insert("bias".to_string());
        }
        for power in &powers {
            if let Some(out_name) = Self::term_name(power, &col_names)
                && !seen.insert(out_name.clone())
            {
                return Err(Error::InvalidInput(format!(
                    "PolynomialFeatures.fit: generated column '{out_name}' collides with \
                     another generated column. Output names do not encode exponents, so \
                     degree >= 3 without interaction_only(true) always collides; otherwise \
                     rename the conflicting input column."
                )));
            }
        }

        self.input_columns = Some(col_names);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for PolynomialFeatures {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "PolynomialFeatures has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let input_columns = self.input_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "PolynomialFeatures has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let powers = Self::generate_powers(input_columns.len(), self.degree, self.interaction_only);

        let mut columns: Vec<Column> = Vec::new();
        let n_rows = x.height();

        if self.include_bias {
            let bias = Column::from(Series::new("bias".into(), vec![1.0f64; n_rows]));
            columns.push(bias);
        }

        for power in &powers {
            let mut has_terms = false;
            let mut series_vec: Option<Series> = None;

            for (j, &p) in power.iter().enumerate() {
                if p == 0 {
                    continue;
                }

                let name = &input_columns[j];
                let orig_series = x
                    .column(name)
                    .map_err(|e| {
                        Error::InvalidInput(format!(
                            "PolynomialFeatures.transform: column '{}' not found. \
                             The transformer was fitted on columns: {:?}. {}",
                            name, input_columns, e
                        ))
                    })?
                    .as_materialized_series()
                    .clone();

                if !has_terms {
                    series_vec = Some(if p > 1 {
                        series_pow(&orig_series, p, "PolynomialFeatures")?
                    } else {
                        orig_series
                    });
                    has_terms = true;
                } else {
                    let powered = if p > 1 {
                        series_pow(&orig_series, p, "PolynomialFeatures")?
                    } else {
                        orig_series
                    };
                    let prev = series_vec.as_ref().ok_or_else(|| {
                        Error::Computation(
                            "PolynomialFeatures: internal state error — series_vec unset before mul"
                                .into(),
                        )
                    })?;
                    series_vec = Some(series_mul(prev, &powered, "PolynomialFeatures")?);
                }
            }

            if let (Some(s), Some(col_name)) = (series_vec, Self::term_name(power, input_columns)) {
                let mut renamed = s.clone();
                renamed.rename(col_name.as_str().into());
                columns.push(Column::from(renamed));
            }
        }

        if columns.is_empty() {
            return Err(Error::Computation(
                "PolynomialFeatures: no features were generated. \
                 Ensure degree >= 1 and the input has at least one f64 column."
                    .into(),
            ));
        }

        DataFrame::new(n_rows, columns)
            .map_err(|e| Error::Computation(format!("failed to create polynomial features: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_generate_powers_degree2_2features() {
        let powers = PolynomialFeatures::generate_powers(2, 2, false);
        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0].iter().sum::<usize>(), 1);
        assert_eq!(powers[1].iter().sum::<usize>(), 1);
        assert_eq!(powers[2].iter().sum::<usize>(), 2);
        assert_eq!(powers[3].iter().sum::<usize>(), 2);
        assert_eq!(powers[4].iter().sum::<usize>(), 2);
    }

    #[test]
    fn test_polynomial_features_fit_transform() {
        let mut pf = PolynomialFeatures::new(2).unwrap().include_bias(true);
        let df = make_test_df();

        pf.fit(df.clone()).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let mut pf = PolynomialFeatures::new(2).unwrap().include_bias(false);
        let df = make_test_df();

        pf.fit(df.clone()).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 5);
    }

    #[test]
    fn test_bias_column_collision_detected_at_fit() {
        let bias_col = Column::from(Series::new("bias".into(), &[1.0f64, 2.0, 3.0]));
        let a = Column::from(Series::new("a".into(), &[4.0f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![bias_col, a]).unwrap();

        let mut pf = PolynomialFeatures::new(2).unwrap().include_bias(true);
        let err = pf.fit(df).unwrap_err();
        match &err {
            Error::InvalidInput(msg) => {
                assert!(
                    msg.contains("bias"),
                    "expected message containing 'bias', got: {msg}"
                );
                assert!(
                    msg.contains("collides"),
                    "expected message containing 'collides', got: {msg}"
                );
            }
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_bias_column_collision_skipped_when_no_bias() {
        let bias_col = Column::from(Series::new("bias".into(), &[1.0f64, 2.0, 3.0]));
        let a = Column::from(Series::new("a".into(), &[4.0f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![bias_col, a]).unwrap();

        let mut pf = PolynomialFeatures::new(2).unwrap().include_bias(false);
        pf.fit(df.clone()).unwrap();
        let result = pf.transform(df).unwrap();
        assert_eq!(result.width(), 5);
    }

    #[test]
    fn test_fit_rejects_generated_name_colliding_with_another() {
        // The `a * b` interaction is named `a_b_inter`, the same as the
        // degree-1 term for the input column `a_b_inter`.
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        let ab = Column::from(Series::new("a_b_inter".into(), &[7.0f64, 8.0, 9.0]));
        let df = DataFrame::new(3, vec![a, b, ab]).unwrap();
        let mut pf = PolynomialFeatures::new(2).unwrap().include_bias(false);
        let err = pf.fit(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("'a_b_inter'")),
            "{err}"
        );
    }

    #[test]
    fn test_fit_rejects_degree3_power_name_collision() {
        // `a^2` and `a^3` are both named `a^`; fit says so instead of
        // transform failing on a duplicate column.
        let df = DataFrame::new(3, vec![make_test_df().column("a").unwrap().clone()]).unwrap();
        let mut pf = PolynomialFeatures::new(3).unwrap();
        let err = pf.fit(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("degree >= 3")),
            "{err}"
        );
    }

    #[test]
    fn test_output_names_unchanged() {
        let mut pf = PolynomialFeatures::new(2).unwrap();
        let df = make_test_df();
        pf.fit(df.clone()).unwrap();
        let out = pf.transform(df).unwrap();
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, ["bias", "b", "a", "b^", "a_b_inter", "a^"]);
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        let mut pf = PolynomialFeatures::new(2)
            .unwrap()
            .include_bias(false)
            .interaction_only(true);
        let df = make_test_df();

        pf.fit(df.clone()).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }
}
