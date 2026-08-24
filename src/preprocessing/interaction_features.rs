//! Pairwise interaction feature generation.
//!
//! [`InteractionFeatures`] generates the element-wise product of every pair
//! of input columns, without the pure-power terms (`a²`, `b²`, …) or
//! higher-degree crosses (`a·b·c`) that [`PolynomialFeatures`] would
//! produce. It is a focused alternative to
//! `PolynomialFeatures::builder().degree(2).interaction_only(true).build()`,
//! offered as a dedicated, more discoverable transformer.
//!
//! # Output semantics
//!
//! The original input columns are **preserved** and the new interaction
//! columns are **appended** in column order. For input columns
//! `[a, b, c]` (no self-products), the output columns are
//! `[a, b, c, a_x_b, a_x_c, b_x_c]`.
//!
//! When [`include_self_products`](InteractionFeaturesBuilder::include_self_products)
//! is enabled, pure-square terms are also emitted:
//! `[a, b, c, a_x_a, a_x_b, a_x_c, b_x_b, b_x_c, c_x_c]`.
//!
//! Pairs are generated in the **order the columns appear** in the fitted
//! column list (frame order for auto-discovered columns; the user-supplied
//! order for explicit columns) — no lexicographic reordering is applied.
//!
//! # Name collisions
//!
//! Generated names use the `_x_` separator (e.g. `a_x_b`). If an input
//! column already carries a generated name — for example the input already
//! contains a column literally named `a_x_b` — [`transform`](Transform::transform)
//! returns [`Error::InvalidInput`] instead of silently overwriting it.
//! Similarly, input column names that themselves contain `_x_` (e.g.
//! `p_x_q`) can make two distinct pairs produce the same generated name;
//! that is also rejected. Rename such columns before fitting.
//!
//! [`PolynomialFeatures`]: crate::preprocessing::polynomial_features::PolynomialFeatures

use std::collections::HashSet;

use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{numeric_f64_columns, series_mul};
use polars::prelude::*;

/// Suffix inserted between the names of the two factor columns to form an
/// interaction column name (e.g. `a_x_b`).
///
/// Note: if an input column name itself contains `_x_`, the generated name
/// may be ambiguous; configurations where two distinct pairs produce the
/// same generated name are rejected at transform time rather than silently
/// overwritten.
const NAME_SEP: &str = "_x_";

/// Generate pairwise interaction features (`a·b`) without the full
/// polynomial expansion.
///
/// Stateful only in the sense that it remembers which columns to operate
/// on; no numeric parameters are learned. Implements [`Fit`] and
/// [`Transform`] on [`DataFrame`].
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::interaction_features::InteractionFeatures;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
/// let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
/// let df = DataFrame::new(3, vec![a, b])?;
///
/// let mut xf = InteractionFeatures::builder().build();
/// xf.fit(df.clone())?;
/// let out = xf.transform(df)?;
/// // a, b, a_x_b = [4.0, 10.0, 18.0]
/// assert_eq!(out.width(), 3);
/// assert_eq!(out.column("a_x_b")?.f64()?.get(1).unwrap(), 10.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct InteractionFeatures {
    fitted: bool,
    columns: Vec<String>,
    include_self_products: bool,
}

impl InteractionFeatures {
    /// Create a new transformer with auto-discovered columns and
    /// `include_self_products = false`.
    ///
    /// Equivalent to
    /// [`InteractionFeatures::builder().build()`](InteractionFeaturesBuilder::build).
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: vec![],
            include_self_products: false,
        }
    }

    /// Begin builder configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use featrs::preprocessing::interaction_features::InteractionFeatures;
    ///
    /// let _xf = InteractionFeatures::builder()
    ///     .columns(&["a", "b"])
    ///     .include_self_products(true)
    ///     .build();
    /// ```
    pub fn builder() -> InteractionFeaturesBuilder {
        InteractionFeaturesBuilder::default()
    }
}

impl Default for InteractionFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`InteractionFeatures`].
///
/// Provides a more ergonomic way to configure interaction feature
/// generation, mirroring the [`PolynomialFeaturesBuilder`] pattern.
///
/// [`PolynomialFeaturesBuilder`]: crate::preprocessing::polynomial_features::PolynomialFeaturesBuilder
#[derive(Default)]
pub struct InteractionFeaturesBuilder {
    columns: Option<Vec<String>>,
    include_self_products: bool,
}

impl InteractionFeaturesBuilder {
    /// Restrict interaction generation to the named columns.
    ///
    /// When omitted, [`build`](InteractionFeaturesBuilder::build) produces a
    /// transformer that auto-discovers all `Float64` columns at [`Fit`]
    /// time.
    ///
    /// Each column must exist in the frame passed to `fit` and have dtype
    /// `Float64`; otherwise `fit` returns
    /// [`Error::InvalidInput`].
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = Some(cols.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Whether to also emit pure-square terms (`a·a`, `b·b`, …).
    ///
    /// Default: `false` — only distinct pairs (`i < j`) are generated.
    pub fn include_self_products(mut self, value: bool) -> Self {
        self.include_self_products = value;
        self
    }

    /// Build the [`InteractionFeatures`] instance.
    ///
    /// This never fails: invalid column selections surface at [`Fit`] time.
    pub fn build(self) -> InteractionFeatures {
        InteractionFeatures {
            fitted: false,
            columns: self.columns.unwrap_or_default(),
            include_self_products: self.include_self_products,
        }
    }
}

impl Fit<DataFrame> for InteractionFeatures {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "InteractionFeatures.fit received an empty DataFrame (0 rows or 0 columns).".into(),
            ));
        }

        if self.columns.is_empty() {
            let discovered = numeric_f64_columns(&x);
            if discovered.is_empty() {
                let all_types: Vec<String> = x
                    .get_column_names()
                    .iter()
                    .filter_map(|n| x.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
                    .collect();
                return Err(Error::InvalidInput(format!(
                    "InteractionFeatures: no Float64 columns found. This transformer only \
                     operates on f64 columns. Available columns: [{}]. Cast non-f64 columns \
                     before fitting.",
                    all_types.join(", ")
                )));
            }
            self.columns = discovered;
        } else {
            for col in &self.columns {
                let c = x.column(col.as_str()).map_err(|e| {
                    Error::InvalidInput(format!(
                        "InteractionFeatures.fit: column '{col}' not found. {e}"
                    ))
                })?;
                if c.dtype() != &DataType::Float64 {
                    return Err(Error::InvalidInput(format!(
                        "InteractionFeatures.fit: column '{col}' has dtype {}; expected Float64.",
                        c.dtype()
                    )));
                }
            }
        }

        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for InteractionFeatures {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "InteractionFeatures has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();
        let who = "InteractionFeatures";
        let n = self.columns.len();

        // (i, j) index pairs into the fitted column list.
        // - default (no self products): j strictly greater than i
        // - include_self_products: j >= i  (covers i==j squares)
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| {
                let start = if self.include_self_products { i } else { i + 1 };
                (start..n).map(move |j| (i, j))
            })
            .collect();

        // Reject name collisions up front: `DataFrame::with_column` silently
        // REPLACES an existing column with the same name, so without this
        // check a collision would silently destroy input data. The `used`
        // set starts with the input columns and also catches two distinct
        // pairs that produce the same generated name (possible when input
        // names contain `_x_`).
        let mut used: HashSet<String> = out
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for (i, j) in &pairs {
            let out_name = format!("{}{}{}", self.columns[*i], NAME_SEP, self.columns[*j]);
            if !used.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "{who}.transform: generated column '{out_name}' collides with an \
                     existing input column or a previously generated column. Rename the \
                     conflicting column(s) or choose different column names."
                )));
            }
        }

        for (i, j) in pairs {
            let name_i = &self.columns[i];
            let name_j = &self.columns[j];

            let s_i = out
                .column(name_i.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "{who}.transform: column '{name_i}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.columns
                    ))
                })?
                .as_materialized_series()
                .clone();
            let s_j = out
                .column(name_j.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "{who}.transform: column '{name_j}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.columns
                    ))
                })?
                .as_materialized_series()
                .clone();

            let product = series_mul(&s_i, &s_j, who)?;
            let out_name = format!("{name_i}{NAME_SEP}{name_j}");

            let mut product = product;
            product.rename(out_name.as_str().into());

            out.with_column(product.into())
                .map_err(|e| Error::Computation(format!("{who}.transform: {e}")))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_col_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    fn make_three_col_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let c = Column::from(Series::new("c".into(), &[7.0_f64, 8.0, 9.0]));
        DataFrame::new(3, vec![a, b, c]).unwrap()
    }

    #[test]
    fn test_interaction_two_columns_default() {
        let df = make_two_col_df();
        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // originals: a, b  + new: a_x_b
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);

        let prod = out.column("a_x_b").unwrap().f64().unwrap();
        assert_eq!(prod.get(0).unwrap(), 4.0);
        assert_eq!(prod.get(1).unwrap(), 10.0);
        assert_eq!(prod.get(2).unwrap(), 18.0);
    }

    #[test]
    fn test_interaction_include_self_products() {
        let df = make_two_col_df();
        let mut xf = InteractionFeatures::builder()
            .include_self_products(true)
            .build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // a, b, a_x_a, a_x_b, b_x_b
        assert_eq!(out.width(), 5);

        let aa = out.column("a_x_a").unwrap().f64().unwrap();
        assert_eq!(aa.get(0).unwrap(), 1.0);
        assert_eq!(aa.get(1).unwrap(), 4.0);
        assert_eq!(aa.get(2).unwrap(), 9.0);

        let bb = out.column("b_x_b").unwrap().f64().unwrap();
        assert_eq!(bb.get(0).unwrap(), 16.0);
        assert_eq!(bb.get(1).unwrap(), 25.0);
        assert_eq!(bb.get(2).unwrap(), 36.0);

        // a_x_b still present
        assert!(out.column("a_x_b").is_ok());
    }

    #[test]
    fn test_interaction_three_columns_three_pairs() {
        let df = make_three_col_df();
        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // a, b, c  +  a_x_b, a_x_c, b_x_c
        assert_eq!(out.width(), 6);
        assert!(out.column("a_x_b").is_ok());
        assert!(out.column("a_x_c").is_ok());
        assert!(out.column("b_x_c").is_ok());
        // no self products
        assert!(out.column("a_x_a").is_err());
    }

    #[test]
    fn test_interaction_single_column_no_new_cols() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // kept originals; no pairs generated
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn test_interaction_single_column_self_products() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        let mut xf = InteractionFeatures::builder()
            .include_self_products(true)
            .build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // a, a_x_a
        assert_eq!(out.width(), 2);
        let aa = out.column("a_x_a").unwrap().f64().unwrap();
        assert_eq!(aa.get(0).unwrap(), 1.0);
        assert_eq!(aa.get(1).unwrap(), 4.0);
        assert_eq!(aa.get(2).unwrap(), 9.0);
    }

    #[test]
    fn test_interaction_null_propagation() {
        let a = Column::from(Series::new("a".into(), &[Some(1.0_f64), None, Some(3.0)]));
        let b = Column::from(Series::new("b".into(), &[Some(4.0_f64), Some(5.0), None]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let prod = out.column("a_x_b").unwrap().f64().unwrap();
        assert_eq!(prod.get(0).unwrap(), 4.0);
        assert!(prod.get(1).is_none()); // a null
        assert!(prod.get(2).is_none()); // b null
    }

    #[test]
    fn test_interaction_nan_propagation() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, f64::NAN, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![a, b]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let prod = out.column("a_x_b").unwrap().f64().unwrap();
        assert_eq!(prod.get(0).unwrap(), 4.0);
        assert!(prod.get(1).unwrap().is_nan());
        assert_eq!(prod.get(2).unwrap(), 18.0);
    }

    #[test]
    fn test_interaction_not_fitted_error() {
        let df = make_two_col_df();
        let xf = InteractionFeatures::builder().build();
        let err = xf.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_interaction_auto_discovery_all_f64() {
        let df = make_two_col_df();
        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();
        // auto-discovered a, b → one pair
        assert_eq!(out.width(), 3);
    }

    #[test]
    fn test_interaction_auto_discovery_skips_non_f64() {
        // mixed dtypes: only f64 columns participate
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let df = DataFrame::new(3, vec![a, cat, b]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // a, cat, b  +  a_x_b   (cat skipped)
        assert_eq!(out.width(), 4);
        assert!(out.column("a_x_b").is_ok());
    }

    #[test]
    fn test_interaction_auto_discovery_no_f64_errors() {
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![cat]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_interaction_explicit_columns() {
        let df = make_three_col_df();
        let mut xf = InteractionFeatures::builder().columns(&["a", "c"]).build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        // a, b, c  +  a_x_c   (b not selected)
        assert_eq!(out.width(), 4);
        assert!(out.column("a_x_c").is_ok());
        assert!(out.column("a_x_b").is_err());
    }

    #[test]
    fn test_interaction_explicit_missing_column_errors() {
        let df = make_two_col_df();
        let mut xf = InteractionFeatures::builder()
            .columns(&["a", "missing"])
            .build();
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_interaction_explicit_non_f64_column_errors() {
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["x", "y", "z"]));
        let df = DataFrame::new(3, vec![a, cat]).unwrap();

        let mut xf = InteractionFeatures::builder()
            .columns(&["a", "cat"])
            .build();
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_interaction_empty_dataframe_rejected() {
        let df = DataFrame::empty();
        let mut xf = InteractionFeatures::builder().build();
        let err = xf.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_interaction_zero_column_propagates() {
        // products involving an all-zero column are all zeros
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let z = Column::from(Series::new("z".into(), &[0.0_f64, 0.0, 0.0]));
        let df = DataFrame::new(3, vec![a, z]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        let prod = out.column("a_x_z").unwrap().f64().unwrap();
        assert_eq!(prod.get(0).unwrap(), 0.0);
        assert_eq!(prod.get(1).unwrap(), 0.0);
        assert_eq!(prod.get(2).unwrap(), 0.0);
    }

    #[test]
    fn test_interaction_column_order_preserved() {
        // explicit column order [b, a] → pair name b_x_a, not a_x_b
        let df = make_two_col_df();
        let mut xf = InteractionFeatures::builder().columns(&["b", "a"]).build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();

        assert!(out.column("b_x_a").is_ok());
        assert!(out.column("a_x_b").is_err());
    }

    #[test]
    fn test_default_equals_new() {
        let d = InteractionFeatures::default();
        let n = InteractionFeatures::new();
        assert_eq!(d.columns, n.columns);
        assert_eq!(d.include_self_products, n.include_self_products);
        assert_eq!(d.fitted, n.fitted);
    }

    #[test]
    fn test_interaction_name_collision_errors() {
        // Input already contains a column literally named a_x_b.
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0_f64, 5.0, 6.0]));
        let clash = Column::from(Series::new("a_x_b".into(), &[7.0_f64, 8.0, 9.0]));
        let df = DataFrame::new(3, vec![a, b, clash]).unwrap();

        let mut xf = InteractionFeatures::builder().columns(&["a", "b"]).build();
        xf.fit(df.clone()).unwrap();
        let err = xf.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "collision with an existing column must error, not overwrite"
        );
        assert!(err.to_string().contains("a_x_b"));
    }

    #[test]
    fn test_interaction_separator_in_names_still_generates() {
        // Column names containing the separator produce distinct names here;
        // this is the non-colliding separator case.
        let p = Column::from(Series::new("p".into(), &[1.0_f64, 2.0, 3.0]));
        let q_x_r = Column::from(Series::new("q_x_r".into(), &[4.0_f64, 5.0, 6.0]));
        let r = Column::from(Series::new("r".into(), &[7.0_f64, 8.0, 9.0]));
        let df = DataFrame::new(3, vec![p, q_x_r, r]).unwrap();

        // fitted order: [p, q_x_r, r]
        // pairs: (p, q_x_r) -> p_x_q_x_r, (p, r) -> p_x_r,
        //        (q_x_r, r) -> q_x_r_x_r
        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let out = xf.transform(df).unwrap();
        assert!(out.column("p_x_q_x_r").is_ok());
        assert!(out.column("p_x_r").is_ok());
        assert!(out.column("q_x_r_x_r").is_ok());
    }

    #[test]
    fn test_interaction_generated_name_collision_errors() {
        // Two distinct pairs produce the same generated name because the
        // input names contain the separator: (a, b_x_c) and (a_x_b, c)
        // both generate "a_x_b_x_c".
        let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0]));
        let b_x_c = Column::from(Series::new("b_x_c".into(), &[4.0_f64, 5.0, 6.0]));
        let a_x_b = Column::from(Series::new("a_x_b".into(), &[7.0_f64, 8.0, 9.0]));
        let c = Column::from(Series::new("c".into(), &[10.0_f64, 11.0, 12.0]));
        let df = DataFrame::new(3, vec![a, b_x_c, a_x_b, c]).unwrap();

        let mut xf = InteractionFeatures::builder().build();
        xf.fit(df.clone()).unwrap();
        let err = xf.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert!(err.to_string().contains("a_x_b_x_c"));
    }
}
