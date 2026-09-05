//! Discretization of continuous features into bins.
//!
//! [`KBinsDiscretizer`] converts `Float64` columns into `n_bins` discrete bins
//! using one of three strategies — equal-width ([`BinStrategy::Uniform`]),
//! equal-population ([`BinStrategy::Quantile`]), or 1-D k-means
//! ([`BinStrategy::KMeans`]) — and encodes each bin either as an ordinal
//! integer column ([`EncodeMode::Ordinal`]) or as one-hot binary columns
//! ([`EncodeMode::OneHot`] / [`EncodeMode::OneHotDropFirst`]). This is the Rust
//! analogue of `sklearn.preprocessing.KBinsDiscretizer`.
//!
//! The effective number of bins for a column can be **less** than the
//! configured `n_bins`: duplicate bin boundaries (from heavily-tied quantile
//! data, collapsed k-means centers, or a constant column) collapse, so the
//! one-hot output width may be smaller than configured. Use
//! [`KBinsDiscretizer::effective_bins`] to read the actual count per fitted
//! column rather than sizing by `n_bins`.

use crate::preprocessing::scaler::percentile_sorted;
use crate::traits::{Error, Fit, Result, Transform};
use crate::util::{replace_f64_column, require_f64_columns};
use polars::prelude::*;
use std::collections::HashMap;

/// Strategy used to choose bin boundaries for each column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinStrategy {
    /// Equal-width bins spanning the training `[min, max]` range.
    Uniform,
    /// Bins holding (approximately) equal numbers of observations.
    Quantile,
    /// Bins whose boundaries are midpoints between 1-D k-means cluster centers.
    KMeans,
}

/// How each fitted column's bins are encoded in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeMode {
    /// Replace the column with a single `Float64` column of bin indices
    /// `0 .. n_bins-1` (nulls preserved as null).
    Ordinal,
    /// Replace the column with `n_bins` binary columns named
    /// `{column}_bin_{j}` for `j in 0..n_bins`.
    ///
    /// When duplicate bin boundaries collapse the effective bin count may be
    /// **less** than `n_bins` (see [`KBinsDiscretizer::effective_bins`]), so
    /// the number of emitted columns may be smaller than configured.
    OneHot,
    /// Like [`EncodeMode::OneHot`] but the first bin column (`j == 0`) is
    /// dropped, producing `n_bins - 1` columns to avoid multicollinearity.
    ///
    /// As with [`EncodeMode::OneHot`], when duplicate boundaries collapse the
    /// effective bin count (and thus the column count) may be **less** than
    /// `n_bins`; a column that collapses to a single bin is rejected at fit
    /// time because dropping its only bin would remove the feature entirely.
    OneHotDropFirst,
}

/// Per-column bin boundaries captured during [`Fit`].
struct BinEdges {
    column: String,
    /// Ascending bin boundaries; `edges.len() - 1` is the number of bins.
    /// The bin of a value `v` is the index `b` with
    /// `edges[b] <= v < edges[b + 1]` (the last bin includes its upper edge).
    edges: Vec<f64>,
}

/// Bin continuous features into `n_bins` discrete bins.
///
/// Only `Float64` columns are considered; columns of other dtypes are
/// preserved unchanged in the output. The chosen bin boundaries (`bin_edges`)
/// are learned during [`fit`](Fit::fit) and applied later by
/// [`transform`](Transform::transform). Values seen at transform time that
/// fall outside the fitted range are clipped to the nearest bin. Missing
/// values are handled per output mode: in `Ordinal` mode, `null` stays `null`
/// and non-finite inputs (`NaN`/`±Inf`) are emitted as `NaN`; in one-hot
/// modes (`OneHot` / `OneHotDropFirst`) both `null` and non-finite inputs
/// produce all-zero bin columns.
///
/// The effective number of bins for a column can be **less** than the
/// configured `n_bins` when duplicate bin boundaries collapse (heavily-tied
/// quantile data, collapsed k-means centers, or a constant column). This
/// makes the one-hot output width smaller than configured, so downstream
/// consumers should read [`effective_bins`](Self::effective_bins) per fitted
/// column rather than sizing inputs by `n_bins`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::kbins_discretizer::KBinsDiscretizer;
/// use featrs::preprocessing::kbins_discretizer::EncodeMode;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0, 4.0]));
/// let df = DataFrame::new(4, vec![col])?;
///
/// let mut kb = KBinsDiscretizer::new().n_bins(2).encode(EncodeMode::Ordinal);
/// kb.fit(df.clone())?;
/// let out = kb.transform(df)?;
/// // Two equal-width bins over [1, 4]: [1, 2.5) -> 0, [2.5, 4] -> 1.
/// assert_eq!(out.column("x")?.f64()?.get(0).unwrap(), 0.0);
/// assert_eq!(out.column("x")?.f64()?.get(3).unwrap(), 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct KBinsDiscretizer {
    fitted: bool,
    n_bins: usize,
    strategy: BinStrategy,
    encode: EncodeMode,
    bin_edges: Option<Vec<BinEdges>>,
}

impl KBinsDiscretizer {
    /// Create a new `KBinsDiscretizer`.
    ///
    /// Defaults: `n_bins = 5`, [`BinStrategy::Uniform`],
    /// [`EncodeMode::Ordinal`].
    pub fn new() -> Self {
        Self {
            fitted: false,
            n_bins: 5,
            strategy: BinStrategy::Uniform,
            encode: EncodeMode::Ordinal,
            bin_edges: None,
        }
    }

    /// Set the number of bins (default: `5`).
    ///
    /// Must be at least `2`; an [`Error::InvalidInput`] is returned from
    /// [`fit`](Fit::fit) otherwise.
    pub fn n_bins(mut self, k: usize) -> Self {
        self.n_bins = k;
        self
    }

    /// Set the bin-boundary strategy (default: [`BinStrategy::Uniform`]).
    pub fn strategy(mut self, s: BinStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set how bins are encoded (default: [`EncodeMode::Ordinal`]).
    pub fn encode(mut self, e: EncodeMode) -> Self {
        self.encode = e;
        self
    }

    /// Number of effective bins produced for a fitted discretized column.
    ///
    /// This is `edges.len() - 1` for the column. When duplicate bin
    /// boundaries collapse (e.g. heavily-tied quantile data or collapsed
    /// k-means centers), the effective count can be **less** than the
    /// configured `n_bins`. The one-hot output width is `effective_bins` for
    /// [`EncodeMode::OneHot`] and `effective_bins - 1` for
    /// [`EncodeMode::OneHotDropFirst`] (the first bin is dropped).
    ///
    /// Returns `Some(n)` for a column that was discretized at fit time, and
    /// `None` for a column that was not discretized (non-`Float64` or not
    /// present in the fitted data) or if the transformer has not been fitted
    /// yet. Consumers that size inputs by `n_bins` should use this instead.
    pub fn effective_bins(&self, column: &str) -> Option<usize> {
        self.bin_edges
            .as_ref()?
            .iter()
            .find(|b| b.column == column)
            .map(|b| b.edges.len() - 1)
    }
}

impl Default for KBinsDiscretizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the bin index of a finite `v` given ascending `edges`.
///
/// `edges` has `n_bins + 1` entries. The returned index is in
/// `0 .. n_bins`, with values below the first edge or above the last edge
/// clipped to the nearest bin.
fn bin_index(edges: &[f64], v: f64) -> usize {
    let n_bins = edges.len() - 1;
    let count = edges.partition_point(|&e| e <= v);
    (count.saturating_sub(1)).clamp(0, n_bins - 1)
}

/// Build equal-width bin edges over sorted `values`.
///
/// `edges[i] = min + (max - min) * i / k`. For a constant column
/// (`min == max`) every edge equals `min`, which yields a single effective
/// bin. The endpoints are pinned to the exact observed `min`/`max`: without
/// it `min + (max - min) * k / k` can round off `max` (e.g. `min=3.0`,
/// `max=6.7`, `k=3` yields `6.700000000000001`), breaking the
/// inclusive-last-bin contract documented on [`BinEdges`].
fn uniform_edges(sorted: &[f64], k: usize) -> Vec<f64> {
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let span = max - min;
    let mut edges: Vec<f64> = (0..=k)
        .map(|i| min + span * (i as f64) / (k as f64))
        .collect();
    // Pin the endpoints so the last edge is exactly the observed maximum
    // (`span * k / k` does not always round back to `span`).
    edges[0] = min;
    edges[k] = max;
    edges
}

/// Build equal-population bin edges over sorted `values`.
///
/// `edges[i]` is the percentile at `100 * i / k`, interpolated between the
/// two nearest elements (see [`percentile_sorted`]).
fn quantile_edges(sorted: &[f64], k: usize) -> Vec<f64> {
    (0..=k)
        .map(|i| percentile_sorted(sorted, 100.0 * i as f64 / k as f64))
        .collect()
}

/// Turn a set of sorted distinct cluster centers into bin edges.
///
/// The outer edges are the training minimum and maximum (so out-of-range
/// transform values clip to the first/last bin); interior edges are
/// midpoints between adjacent centers.
fn edges_from_centers(values: &[f64], centers: &[f64]) -> Vec<f64> {
    let min = values[0];
    let max = values[values.len() - 1];
    if centers.len() == 1 {
        return vec![min, max];
    }
    let mut edges = Vec::with_capacity(centers.len() + 1);
    edges.push(min);
    for pair in centers.windows(2) {
        edges.push((pair[0] + pair[1]) / 2.0);
    }
    edges.push(max);
    edges
}

/// Run 1-D Lloyd k-means over sorted `values` and return the resulting bin
/// edges.
///
/// If the data has fewer distinct values than `k` requested centers, one bin
/// per distinct value is produced. Convergence is bounded by `max_iter`
/// iterations; centers that collapse together (possible when values are
/// tightly clustered) are merged via deduplication of the final sorted
/// centers, so the effective number of bins may be less than `k`.
fn kmeans_edges(values: &[f64], k: usize, max_iter: usize) -> Vec<f64> {
    let mut distinct: Vec<f64> = Vec::new();
    for &v in values {
        if distinct.last().is_none_or(|&l| l != v) {
            distinct.push(v);
        }
    }
    let nd = distinct.len();

    // Fewer distinct training values than requested centers: one bin per
    // distinct value.
    if nd <= k {
        return edges_from_centers(values, &distinct);
    }

    // Seed centers at evenly spaced ranks within the distinct values.
    let mut centers: Vec<f64> = (0..k).map(|i| distinct[(i * (nd - 1)) / (k - 1)]).collect();

    let mut assignment: Vec<usize> = Vec::new();
    for iter in 0..max_iter {
        let mut next = vec![0usize; values.len()];
        // The initial `assignment` is empty; the first iteration must always
        // be treated as a change (otherwise an all-center-0 first pass would
        // look "converged" and skip the center update entirely).
        let mut changed = iter == 0;
        for (i, &v) in values.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (ci, &c) in centers.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            next[i] = best;
            if iter > 0 && next[i] != assignment[i] {
                changed = true;
            }
        }
        assignment = next;
        if !changed {
            break;
        }
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        for (i, &v) in values.iter().enumerate() {
            sums[assignment[i]] += v;
            counts[assignment[i]] += 1;
        }
        for (ci, c) in centers.iter_mut().enumerate() {
            if counts[ci] > 0 {
                *c = sums[ci] / counts[ci] as f64;
            }
        }
    }

    let mut sorted = centers.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut final_centers: Vec<f64> = Vec::new();
    for &c in &sorted {
        if final_centers.last().is_none_or(|&l| l != c) {
            final_centers.push(c);
        }
    }
    edges_from_centers(values, &final_centers)
}

impl Fit<DataFrame> for KBinsDiscretizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset at the top so a failed re-fit cannot leave stale state.
        self.fitted = false;
        self.bin_edges = None;

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "KBinsDiscretizer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide at least 1 row and 1 column."
                    .into(),
            ));
        }
        if self.n_bins < 2 {
            return Err(Error::InvalidInput(format!(
                "KBinsDiscretizer.fit: n_bins must be at least 2, got {}.",
                self.n_bins
            )));
        }

        let cols = require_f64_columns(&x, "KBinsDiscretizer")?;
        let mut bins = Vec::with_capacity(cols.len());

        for name in &cols {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "KBinsDiscretizer.fit: column '{name}' not found. {e}"
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "KBinsDiscretizer.fit: column '{name}' has dtype {}; expected Float64. {e}",
                    s.dtype()
                ))
            })?;
            let mut vals: Vec<f64> = ca.iter().flatten().filter(|v| v.is_finite()).collect();
            if vals.is_empty() {
                return Err(Error::Computation(format!(
                    "KBinsDiscretizer.fit: column '{name}' has no non-null, finite values. \
                     Cannot discretize an all-null or all-NaN column. Impute first or drop the column."
                )));
            }
            vals.sort_by(|a, b| a.total_cmp(b));

            // A constant column admits only a single bin. Give it a
            // degenerate `[min, min]` edge list so every value maps to bin 0,
            // uniformly across strategies.
            let raw = if vals.first() == vals.last() {
                vec![vals[0], vals[0]]
            } else {
                match self.strategy {
                    BinStrategy::Uniform => uniform_edges(&vals, self.n_bins),
                    BinStrategy::Quantile => quantile_edges(&vals, self.n_bins),
                    BinStrategy::KMeans => kmeans_edges(&vals, self.n_bins, 100),
                }
            };

            // Collapse repeated boundaries (e.g. duplicate quantile
            // percentiles on skewed data). Kept as adjacent duplicates they
            // would create unreachable bins that leave gaps in the ordinal
            // indices and always-zero one-hot columns.
            let mut edges: Vec<f64> = Vec::with_capacity(raw.len());
            for &e in &raw {
                if edges.last().is_none_or(|&l| l != e) {
                    edges.push(e);
                }
            }
            // Guarantee at least two edges (one interval) so a fully
            // collapsed column still discretizes into a single bin.
            if edges.len() < 2 {
                let first = edges[0];
                edges.push(first);
            }

            // `OneHotDropFirst` would drop the only bin of a single-bin column,
            // silently removing the feature entirely. Reject that configuration
            // here instead of returning a frame with fewer columns.
            if self.encode == EncodeMode::OneHotDropFirst && edges.len() - 1 < 2 {
                return Err(Error::InvalidInput(format!(
                    "KBinsDiscretizer.fit: column '{name}' collapses to a single bin; \
                     EncodeMode::OneHotDropFirst would drop its only bin and remove the \
                     feature entirely. Use EncodeMode::OneHot, EncodeMode::Ordinal, or \
                     remove the column."
                )));
            }
            bins.push(BinEdges {
                column: name.clone(),
                edges,
            });
        }

        self.bin_edges = Some(bins);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for KBinsDiscretizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "KBinsDiscretizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let bins = self.bin_edges.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "KBinsDiscretizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;

        match self.encode {
            EncodeMode::Ordinal => {
                let mut out = x.clone();
                for be in bins {
                    let edges = be.edges.clone();
                    replace_f64_column(&mut out, &be.column, "KBinsDiscretizer", |v| {
                        if v.is_finite() {
                            bin_index(&edges, v) as f64
                        } else {
                            f64::NAN
                        }
                    })?;
                }
                Ok(out)
            }
            EncodeMode::OneHot | EncodeMode::OneHotDropFirst => {
                let by_name: HashMap<&str, &BinEdges> =
                    bins.iter().map(|b| (b.column.as_str(), b)).collect();
                let mut out_cols: Vec<Column> = Vec::new();

                for col in x.columns() {
                    let name = col.name();
                    let Some(be) = by_name.get(name.as_str()) else {
                        // Non-discretized column: pass through unchanged.
                        out_cols.push(col.clone());
                        continue;
                    };
                    let ca = col.as_materialized_series().f64().map_err(|e| {
                        Error::InvalidInput(format!(
                            "KBinsDiscretizer.transform: column '{name}' has dtype {}; expected Float64. {e}",
                            col.dtype()
                        ))
                    })?;
                    let n_bins = be.edges.len() - 1;
                    let start = if self.encode == EncodeMode::OneHotDropFirst {
                        1
                    } else {
                        0
                    };
                    for j in start..n_bins {
                        let vals: Vec<f64> = ca
                            .iter()
                            .map(|opt| match opt {
                                Some(v) if v.is_finite() && bin_index(&be.edges, v) == j => 1.0,
                                _ => 0.0,
                            })
                            .collect();
                        let col_name = format!("{name}_bin_{j}");
                        out_cols.push(Column::from(Series::new(col_name.as_str().into(), &vals)));
                    }
                }

                DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_df(values: &[f64]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), values));
        DataFrame::new(values.len(), vec![col]).unwrap()
    }

    fn col_vals(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
        df.column(name).unwrap().f64().unwrap().iter().collect()
    }

    #[test]
    fn test_uniform_ordinal_two_bins() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(2)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // edges: [1.0, 2.5, 4.0] -> [0, 0, 1, 1]
        assert_eq!(
            col_vals(&out, "x"),
            vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)]
        );
    }

    #[test]
    fn test_uniform_ordinal_out_of_range_clips() {
        let mut kb = KBinsDiscretizer::new()
            .n_bins(2)
            .encode(EncodeMode::Ordinal);
        kb.fit(make_df(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        // Transform values outside the fitted [1, 4] range clip to nearest bin.
        let out = kb.transform(make_df(&[0.0, 1.5, 3.5, 100.0])).unwrap();
        assert_eq!(
            col_vals(&out, "x"),
            vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)]
        );
    }

    #[test]
    fn test_quantile_ordinal_equal_population() {
        let df = make_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Quantile)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // Equal-population bins over 6 points -> 2 per bin.
        assert_eq!(
            col_vals(&out, "x"),
            vec![
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(1.0),
                Some(2.0),
                Some(2.0)
            ]
        );
    }

    #[test]
    fn test_quantile_skewed_duplicate_edges_collapse() {
        // Heavily skewed data: many zeros make repeated (duplicate) quantile
        // boundaries. These must collapse so no bin is unreachable and the
        // ordinal indices stay contiguous.
        let df = make_df(&[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Quantile)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // Duplicate edges collapse to 2 effective bins: zeros -> 0, 1/2 -> 1.
        assert_eq!(
            col_vals(&out, "x"),
            vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(1.0)
            ]
        );
    }

    #[test]
    fn test_kmeans_ordinal_two_clusters() {
        let df = make_df(&[0.0, 0.0, 1.0, 10.0, 10.0, 11.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(2)
            .strategy(BinStrategy::KMeans)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // Two tight clusters separate cleanly into the two bins.
        assert_eq!(
            col_vals(&out, "x"),
            vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(1.0),
                Some(1.0)
            ]
        );
    }

    #[test]
    fn test_onehot_creates_bin_columns() {
        let col = Column::from(Series::new("x".into(), &[1.0_f64, 3.0, 5.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeMode::OneHot);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // 1 -> bin 0 (x_bin_0 = 1, others 0), etc.
        assert_eq!(out.width(), 3);
        assert_eq!(
            col_vals(&out, "x_bin_0"),
            vec![Some(1.0), Some(0.0), Some(0.0)]
        );
        assert_eq!(
            col_vals(&out, "x_bin_1"),
            vec![Some(0.0), Some(1.0), Some(0.0)]
        );
        assert_eq!(
            col_vals(&out, "x_bin_2"),
            vec![Some(0.0), Some(0.0), Some(1.0)]
        );
        assert!(out.column("x").is_err()); // original replaced
    }

    #[test]
    fn test_onehot_drop_first() {
        let col = Column::from(Series::new("x".into(), &[1.0_f64, 3.0, 5.0]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .encode(EncodeMode::OneHotDropFirst);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        assert_eq!(out.width(), 2);
        assert!(out.column("x_bin_0").is_err()); // first bin dropped
        assert!(out.column("x_bin_1").is_ok());
        assert!(out.column("x_bin_2").is_ok());
    }

    #[test]
    fn test_onehot_passes_through_non_f64_column() {
        let x = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![x, cat]).unwrap();
        let mut kb = KBinsDiscretizer::new().n_bins(2).encode(EncodeMode::OneHot);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // x -> 2 bin columns; cat passed through unchanged.
        assert_eq!(out.width(), 3);
        assert_eq!(out.column("cat").unwrap().dtype(), &DataType::String);
        assert_eq!(
            out.column("cat").unwrap().str().unwrap().get(0).unwrap(),
            "a"
        );
    }

    #[test]
    fn test_null_preservation_ordinal() {
        let col = Column::from(Series::new(
            "x".into(),
            &[Some(1.0_f64), None, Some(4.0), Some(f64::NAN)],
        ));
        let df = DataFrame::new(4, vec![col]).unwrap();
        let mut kb = KBinsDiscretizer::new()
            .n_bins(2)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        let vals = col_vals(&out, "x");
        assert_eq!(vals[0], Some(0.0));
        assert!(vals[1].is_none()); // null preserved
        assert_eq!(vals[2], Some(1.0));
        assert!(vals[3].unwrap().is_nan()); // NaN preserved
    }

    #[test]
    fn test_constant_column_single_bin() {
        let df = make_df(&[5.0, 5.0, 5.0, 5.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        // All identical values collapse to a single bin (index 0).
        assert_eq!(col_vals(&out, "x"), vec![Some(0.0); 4]);
    }

    #[test]
    fn test_onehot_drop_first_constant_column_rejected() {
        let df = make_df(&[5.0, 5.0, 5.0, 5.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .encode(EncodeMode::OneHotDropFirst);
        let err = kb.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_transform_before_fit_not_fitted() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let kb = KBinsDiscretizer::new();
        let err = kb.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)));
    }

    #[test]
    fn test_empty_input_rejected() {
        let mut kb = KBinsDiscretizer::new();
        let err = kb.fit(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_n_bins_below_two_rejected() {
        let df = make_df(&[1.0, 2.0, 3.0]);
        let mut kb = KBinsDiscretizer::new().n_bins(1);
        let err = kb.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_all_null_column_computation_error() {
        let col = Column::from(Series::new("x".into(), &[None::<f64>, None, None]));
        let df = DataFrame::new(3, vec![col]).unwrap();
        let mut kb = KBinsDiscretizer::new();
        let err = kb.fit(df).unwrap_err();
        assert!(matches!(err, Error::Computation(_)));
    }

    #[test]
    fn test_missing_f64_columns_error() {
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "c"]));
        let df = DataFrame::new(3, vec![cat]).unwrap();
        let mut kb = KBinsDiscretizer::new();
        let err = kb.fit(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_n_bins_two_minimum() {
        let df = make_df(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(2)
            .encode(EncodeMode::Ordinal);
        kb.fit(df.clone()).unwrap();
        let out = kb.transform(df).unwrap();
        let vals = col_vals(&out, "x");
        assert!(vals.iter().all(|v| matches!(v, Some(0.0) | Some(1.0))));
        assert!(vals[0] == Some(0.0) && vals[vals.len() - 1] == Some(1.0));
    }

    #[test]
    fn test_bin_edges_are_ascending_uniform() {
        let db = KBinsDiscretizer::new().n_bins(4);
        let edges = uniform_edges(&[1.0, 3.0, 5.0, 9.0], db.n_bins);
        // Ascending, first = min, last = max
        assert_relative_eq!(edges[0], 1.0);
        assert_relative_eq!(edges[4], 9.0);
        for w in edges.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_uniform_max_falls_in_last_bin() {
        // The documented inclusive-last-bin contract: the observed maximum maps
        // to the LAST bin (index n_bins-1), and the final edge is exactly the
        // observed max. The [3.0, 6.7] span is adversarial — `span * k / k` does
        // not round back to `span` for several k (e.g. k=3 yields
        // 6.700000000000001), so without pinning the final edge drifts off the
        // max and the contract is broken even though `bin_index` clamps the
        // value into the last bin.
        for k in 2..=50usize {
            let mut vals: Vec<f64> = (0..=k).map(|i| 3.0 + 3.7 * i as f64 / k as f64).collect();
            vals[k] = 6.7; // observed maximum, pinned explicitly
            vals.sort_by(|a, b| a.total_cmp(b));

            let edges = uniform_edges(&vals, k);
            assert_eq!(edges[0], vals[0]);
            assert_eq!(edges[k], vals[k]); // final edge == observed max exactly

            let df = make_df(&vals);
            let mut kb = KBinsDiscretizer::new()
                .n_bins(k)
                .strategy(BinStrategy::Uniform)
                .encode(EncodeMode::Ordinal);
            kb.fit(df.clone()).unwrap();
            let out = kb.transform(df).unwrap();
            let col = out.column("x").unwrap().f64().unwrap();
            let last = col.get(col.len() - 1).unwrap();
            assert_relative_eq!(last, (k - 1) as f64); // max -> last bin
        }
    }

    #[test]
    fn test_quantile_effective_bins_exposed() {
        // Heavily-tied quantile data: duplicate boundaries collapse, so the
        // effective bin count (2) is LESS than the configured n_bins (3), and
        // the one-hot output width matches the effective count, not n_bins.
        let df = make_df(&[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        let mut kb = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Quantile)
            .encode(EncodeMode::OneHot);
        kb.fit(df.clone()).unwrap();
        assert_eq!(kb.effective_bins("x"), Some(2));

        let out = kb.transform(df).unwrap();
        assert_eq!(out.width(), 2); // one-hot width == effective bins

        // Unknown / non-discretized column -> None.
        assert_eq!(kb.effective_bins("nonexistent"), None);
    }

    #[test]
    fn test_effective_bins_none_before_fit() {
        let kb = KBinsDiscretizer::new().n_bins(3);
        assert_eq!(kb.effective_bins("x"), None);
    }
}
