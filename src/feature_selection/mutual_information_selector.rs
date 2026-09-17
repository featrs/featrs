//! Mutual-information-based feature selection.
//!
//! [`MutualInformationSelector`] ranks the `Float64` features of a frame by an
//! estimate of their mutual information (MI) with a single target column and
//! keeps the top `k`. Unlike correlation, MI captures any dependency, not only
//! linear ones. It is the equivalent of scikit-learn's `mutual_info_classif` /
//! `mutual_info_regression` used with `SelectKBest`.

use crate::traits::{Error, FitSupervised, Result, Transform};
use crate::util::require_f64_columns;
use polars::prelude::*;
use statrs::function::gamma::digamma;

/// Whether the target of [`MutualInformationSelector`] is discrete or
/// continuous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MITask {
    /// A discrete target (class labels), as with scikit-learn's
    /// `mutual_info_classif`.
    #[default]
    Classification,
    /// A continuous target, as with scikit-learn's `mutual_info_regression`.
    Regression,
}

/// Select the top `k` features by mutual information with the target.
///
/// `MutualInformationSelector` is supervised: it implements [`FitSupervised`]
/// and requires a target `y` at `fit` time. Only `Float64` feature columns are
/// scored; columns of other dtypes are silently skipped from the selection. The
/// target must be a single `Float64` column.
///
/// # Estimator
///
/// Scores are estimated with the Kraskov-Stögbauer-Grassberger k-nearest
/// neighbour estimator (algorithm 1, *Estimating mutual information*, 2004)
/// using Chebyshev (max-norm) distances:
///
/// ```text
/// MI = ψ(k) + ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)>
/// ```
///
/// `k` is [`n_neighbors`](MutualInformationSelector::n_neighbors), `N` the
/// number of usable rows, `ψ` the digamma function, and `n_x` / `n_y` the number
/// of other rows closer to the row under consideration than its `k`-th nearest
/// neighbour in the joint feature/target space, measured in the feature space
/// and in the target space respectively. Pruning the marginal counts at the
/// joint neighbour's distance is what keeps the estimate insensitive to the
/// units each variable is measured in.
///
/// The feature — and, for [`MITask::Regression`], the target — is min-max
/// scaled to `[0, 1]` before the distances are taken. Mutual information is
/// invariant under rescaling either variable, but the max-norm distance is not,
/// so without scaling a feature whose values span wide units would dominate the
/// joint radius and flatten every score to zero.
///
/// The estimator is asymptotic, so estimates carry a finite-sample bias: with
/// few usable rows (in practice fewer than a few dozen) they are noisy, and an
/// independent feature can score slightly above zero. Negative estimates are
/// clamped to `0.0`, and a feature or target with a single distinct value is
/// scored exactly `0.0`. Features with many repeated values (a binary flag, for
/// instance) sit outside the estimator's assumption of continuous variables and
/// may be scored misleadingly.
///
/// # Target treatment
///
/// [`MITask::Regression`] treats the target as continuous and applies the
/// estimator above to both variables. [`MITask::Classification`] treats it as a
/// set of labels and scores the feature with the continuous/discrete estimator
/// of Ross (2014), the one scikit-learn's `mutual_info_classif` uses:
///
/// ```text
/// MI = ψ(N) + <ψ(k_i)> - <ψ(c_i)> - <ψ(m_i)>
/// ```
///
/// where `c_i` is the number of rows sharing row `i`'s label, `k_i` the smaller
/// of `n_neighbors` and `c_i - 1`, and `m_i` the number of rows of any label
/// within row `i`'s distance to its `k_i`-th same-label neighbour in the feature
/// space. Rows whose label occurs only once are left out of the means. Compared
/// with running the continuous estimator on a jittered target, this costs no
/// arbitrary offset and does not depend on the order of the rows.
///
/// `n_neighbors` is clamped to `[1, rows - 1]` per feature, so asking for more
/// neighbours than there are usable rows is safe rather than an error — but a
/// neighbourhood that wide is a poor estimate, and clamping all the way to
/// `rows - 1` scores every feature `0.0`. Keep `n_neighbors` well below the
/// number of usable rows.
///
/// # Rows and columns
///
/// Only `Float64` feature columns are scored; columns of other dtypes are
/// silently skipped. Rows with a null or non-finite (`NaN`/`±Inf`) feature value
/// or target are excluded per feature column, so a column with missing values is
/// scored on the rows that remain. A column with fewer than two usable rows, or
/// with a single distinct value, is scored `0.0`.
///
/// Ties in the scores are broken by column name ascending, and a `k` larger than
/// the number of scored columns keeps every scored column.
///
/// # Cost
///
/// The estimator is brute force with no spatial index: each feature costs
/// `O(rows²)` time, so fitting is roughly `O(rows² × columns)`.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::mutual_information_selector::MutualInformationSelector;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let signal = Column::from(Series::new(
///     "signal".into(),
///     &[0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
/// ));
/// let constant = Column::from(Series::new("constant".into(), &[5.0_f64; 12]));
/// let features = DataFrame::new(12, vec![signal, constant])?;
///
/// let target = Column::from(Series::new(
///     "y".into(),
///     &[0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
/// ));
/// let y = DataFrame::new(12, vec![target])?;
///
/// let mut mi = MutualInformationSelector::new().k(1);
/// mi.fit(features.clone(), y)?;
/// let selected = mi.transform(features)?;
/// assert_eq!(selected.get_column_names(), &["signal"]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MutualInformationSelector {
    fitted: bool,
    k: usize,
    task: MITask,
    n_neighbors: usize,
    selected_columns: Option<Vec<String>>,
    scores: Option<Vec<(String, f64)>>,
}

impl MutualInformationSelector {
    /// Create a new `MutualInformationSelector`.
    ///
    /// Defaults to `k = 10`, [`MITask::Classification`] and
    /// `n_neighbors = 3`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            k: 10,
            task: MITask::Classification,
            n_neighbors: 3,
            selected_columns: None,
            scores: None,
        }
    }

    /// Set the number of top-scoring features to keep.
    ///
    /// A `k` of zero is rejected at `fit` time. A `k` larger than the number of
    /// scored columns keeps every scored column.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set whether the target is a discrete label ([`MITask::Classification`])
    /// or a continuous value ([`MITask::Regression`]).
    pub fn task(mut self, t: MITask) -> Self {
        self.task = t;
        self
    }

    /// Set the number of neighbors `k` used by the Kraskov estimator.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Returns the mutual information score of each scored feature.
    ///
    /// Returns `None` if not fitted yet. The list is sorted highest score
    /// first, with ties broken by column name ascending.
    pub fn scores(&self) -> Option<&[(String, f64)]> {
        self.scores.as_deref()
    }
}

impl Default for MutualInformationSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl FitSupervised<DataFrame, DataFrame> for MutualInformationSelector {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        // A failed fit must not leave a stale selection behind.
        self.fitted = false;
        self.selected_columns = None;
        self.scores = None;

        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "MutualInformationSelector.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "MutualInformationSelector.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }
        if self.k == 0 {
            return Err(Error::InvalidInput(
                "MutualInformationSelector: k must be greater than 0, got 0. \n\
                 Choose k >= 1 to select at least one feature."
                    .into(),
            ));
        }
        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "MutualInformationSelector.fit: target must have exactly 1 column but got {} \
                 columns. Select a single target column.",
                y.width()
            )));
        }
        if y.height() != x.height() {
            return Err(Error::InvalidInput(format!(
                "MutualInformationSelector.fit: feature rows ({}) and target rows ({}) don't match.",
                x.height(),
                y.height()
            )));
        }

        let y_col = &y.columns()[0];
        let y_ca = y_col.as_materialized_series().f64().map_err(|_| {
            Error::InvalidInput(format!(
                "MutualInformationSelector.fit: target column '{}' has dtype {}; \
                 expected Float64. The target must be numeric.",
                y_col.name(),
                y_col.dtype()
            ))
        })?;
        let targets: Vec<Option<f64>> = y_ca.iter().collect();

        let columns = require_f64_columns(&x, "MutualInformationSelector")?;

        let mut scores = Vec::with_capacity(columns.len());
        for name in columns {
            let col = x.column(&name).map_err(|e| {
                Error::InvalidInput(format!(
                    "MutualInformationSelector.fit: column '{name}' not found. {e}"
                ))
            })?;
            let ca = col.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "MutualInformationSelector.fit: column '{name}' has dtype {}; \
                     expected Float64. {e}",
                    col.dtype()
                ))
            })?;

            let pairs: Vec<(f64, f64)> = ca
                .iter()
                .zip(&targets)
                .filter_map(|(value, &target)| match (value, target) {
                    (Some(v), Some(t)) if v.is_finite() && t.is_finite() => Some((v, t)),
                    _ => None,
                })
                .collect();

            scores.push((name, score_feature(&pairs, self.task, self.n_neighbors)));
        }

        // Rank highest score first with ties broken by column name ascending.
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let k = self.k.min(scores.len());
        let selected: Vec<String> = scores
            .iter()
            .take(k)
            .map(|(name, _)| name.clone())
            .collect();

        self.selected_columns = Some(selected);
        self.scores = Some(scores);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for MutualInformationSelector {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "MutualInformationSelector has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "MutualInformationSelector has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            )
        })?;
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Score one feature against the target, in nats.
///
/// `pairs` holds the usable `(feature, target)` rows. The feature (and, for
/// regression, the target) is min-max scaled to `[0, 1]` first: mutual
/// information is invariant under rescaling, but the estimator's max-norm
/// distance is not, so without this a feature measured in wide units would
/// dominate the joint radius and flatten every regression score to zero.
///
/// A feature or target that never changes, and a feature with fewer than two
/// usable rows, is scored `0.0`.
fn score_feature(pairs: &[(f64, f64)], task: MITask, n_neighbors: usize) -> f64 {
    if pairs.len() < 2 {
        return 0.0;
    }

    let (x_lo, x_hi) = range_of(pairs, |&(x, _)| x);
    let (y_lo, y_hi) = range_of(pairs, |&(_, y)| y);
    let (x_span, y_span) = (x_hi - x_lo, y_hi - y_lo);
    if x_span <= 0.0 || y_span <= 0.0 {
        // A constant feature carries no information about the target, and a
        // constant target carries none about any feature.
        return 0.0;
    }

    let features: Vec<f64> = pairs.iter().map(|&(x, _)| (x - x_lo) / x_span).collect();
    let targets: Vec<f64> = pairs.iter().map(|&(_, y)| y).collect();

    match task {
        MITask::Classification => {
            estimate_mi_discrete_target(&features, &targets, n_neighbors).max(0.0)
        }
        MITask::Regression => {
            let scaled: Vec<(f64, f64)> = features
                .iter()
                .zip(&targets)
                .map(|(&x, &y)| (x, (y - y_lo) / y_span))
                .collect();
            estimate_mi(&scaled, n_neighbors).max(0.0)
        }
    }
}

/// Smallest and largest value of one coordinate of `pairs`.
fn range_of(pairs: &[(f64, f64)], coordinate: fn(&(f64, f64)) -> f64) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for pair in pairs {
        let value = coordinate(pair);
        lo = lo.min(value);
        hi = hi.max(value);
    }
    (lo, hi)
}

/// Mutual information estimate (in nats) between a continuous feature and a
/// discrete target, following Ross (2014):
///
/// ```text
/// MI = ψ(N) + <ψ(k_i)> - <ψ(c_i)> - <ψ(m_i)>
/// ```
///
/// over the rows whose label occurs more than once, where `c_i` counts the rows
/// sharing row `i`'s label, `k_i` is `n_neighbors` clamped to `c_i - 1`, and
/// `m_i` counts the rows of any label within row `i`'s distance to its `k_i`-th
/// same-label neighbour in the feature space. Only distances within one feature
/// are ever compared, so the score does not depend on the units the feature is
/// measured in, nor on the order of the rows.
fn estimate_mi_discrete_target(features: &[f64], labels: &[f64], n_neighbors: usize) -> f64 {
    let n = features.len();
    if n < 2 {
        return 0.0;
    }

    let label_counts: Vec<usize> = (0..n)
        .map(|i| labels.iter().filter(|&&label| label == labels[i]).count())
        .collect();

    let mut k_sum = 0.0;
    let mut count_sum = 0.0;
    let mut m_sum = 0.0;
    let mut scored = 0usize;
    let mut same_label_distances = Vec::with_capacity(n);

    for i in 0..n {
        if label_counts[i] < 2 {
            // A label seen once has no same-label neighbour to measure against.
            continue;
        }
        let k = n_neighbors.clamp(1, label_counts[i] - 1);
        same_label_distances.clear();
        for (j, &label) in labels.iter().enumerate() {
            if j != i && label == labels[i] {
                same_label_distances.push((features[i] - features[j]).abs());
            }
        }
        same_label_distances.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
        // The `<= radius` count below must not pick up rows sitting exactly at
        // the k-th distance, so the radius is nudged one float below it.
        let radius = shrink(same_label_distances[k - 1]);

        let m = features
            .iter()
            .filter(|&&value| (features[i] - value).abs() <= radius)
            .count();

        k_sum += digamma(k as f64);
        count_sum += digamma(label_counts[i] as f64);
        m_sum += digamma(m as f64);
        scored += 1;
    }

    if scored == 0 {
        return 0.0;
    }
    let scored = scored as f64;
    digamma(scored) + (k_sum - count_sum - m_sum) / scored
}

/// The float immediately below `radius` (and `0.0` for a zero radius).
fn shrink(radius: f64) -> f64 {
    if radius > 0.0 {
        f64::from_bits(radius.to_bits() - 1)
    } else {
        0.0
    }
}

/// Kraskov-Stögbauer-Grassberger mutual information estimate (algorithm 1)
/// with Chebyshev (max-norm) distances, in nats.
///
/// `n_neighbors` is clamped to `[1, len - 1]`; callers pass at least two rows.
fn estimate_mi(pairs: &[(f64, f64)], n_neighbors: usize) -> f64 {
    let n = pairs.len();
    if n < 2 {
        return 0.0;
    }
    let k = n_neighbors.clamp(1, n - 1);

    let mut joint_distances = vec![f64::INFINITY; n];
    let mut marginal_sum = 0.0;
    for i in 0..n {
        let (x_i, y_i) = pairs[i];
        for (j, distance) in joint_distances.iter_mut().enumerate() {
            *distance = if i == j {
                f64::INFINITY
            } else {
                let (x_j, y_j) = pairs[j];
                (x_i - x_j).abs().max((y_i - y_j).abs())
            };
        }
        joint_distances.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
        let radius = joint_distances[k - 1];

        let mut n_x = 0usize;
        let mut n_y = 0usize;
        for (j, &(x_j, y_j)) in pairs.iter().enumerate() {
            if i == j {
                continue;
            }
            if (x_i - x_j).abs() < radius {
                n_x += 1;
            }
            if (y_i - y_j).abs() < radius {
                n_y += 1;
            }
        }
        marginal_sum += digamma((n_x + 1) as f64) + digamma((n_y + 1) as f64);
    }

    digamma(k as f64) + digamma(n as f64) - marginal_sum / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Rows in the fixture frames: 20 rows of each class.
    const N: usize = 40;

    /// Deterministic uniform values in `[-1, 1)`, so tests that need unrelated
    /// ("noise") data stay reproducible without an RNG dependency.
    fn uniform(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn col(name: &str, values: &[f64]) -> Column {
        Column::from(Series::new(name.into(), values))
    }

    fn target_of(values: &[f64]) -> DataFrame {
        DataFrame::new(values.len(), vec![col("target", values)]).unwrap()
    }

    /// `[0, 0, ..., 1, 1, ...]` with `n / 2` rows of each class.
    fn classes(n: usize) -> Vec<f64> {
        (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect()
    }

    /// A feature that determines the class: each class owns a band of the line,
    /// the bands sit far apart, and no two rows share a value inside a band.
    fn separating(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i < n / 2 {
                    i as f64
                } else {
                    100.0 + i as f64
                }
            })
            .collect()
    }

    /// `informative` determines the class, `unrelated` is drawn independently
    /// of it, and `constant` never changes. `order` picks the rows, so a test
    /// can hand the same data over in a different row order.
    fn classification_features_in_order(n: usize, order: &[usize]) -> (DataFrame, DataFrame) {
        let pick = |values: &[f64]| -> Vec<f64> { order.iter().map(|&i| values[i]).collect() };
        let constant = vec![5.0; n];
        let features = DataFrame::new(
            n,
            vec![
                col("informative", &pick(&separating(n))),
                col("unrelated", &pick(&uniform(n, 12_345))),
                col("constant", &constant),
            ],
        )
        .unwrap();
        (features, target_of(&pick(&classes(n))))
    }

    /// [`classification_features_in_order`] in the natural row order.
    fn classification_features(n: usize) -> (DataFrame, DataFrame) {
        classification_features_in_order(n, &(0..n).collect::<Vec<usize>>())
    }

    /// `linear` tracks the target with noise; `unrelated` is independent of it.
    /// `order` picks the rows, so a test can permute the frame.
    fn regression_frames_in_order(n: usize, order: &[usize]) -> (DataFrame, DataFrame) {
        let pick = |values: &[f64]| -> Vec<f64> { order.iter().map(|&i| values[i]).collect() };
        let x_linear: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let noise = uniform(n, 7);
        let target: Vec<f64> = x_linear
            .iter()
            .zip(&noise)
            .map(|(x, e)| x + 20.0 * e)
            .collect();
        let features = DataFrame::new(
            n,
            vec![
                col("unrelated", &pick(&uniform(n, 99))),
                col("linear", &pick(&x_linear)),
            ],
        )
        .unwrap();
        (features, target_of(&pick(&target)))
    }

    /// [`regression_frames_in_order`] in the natural row order.
    fn regression_frames(n: usize) -> (DataFrame, DataFrame) {
        regression_frames_in_order(n, &(0..n).collect::<Vec<usize>>())
    }

    /// A binary feature that repeats a binary target exactly, so the joint k-th
    /// neighbour distances collapse to zero without the classification offset.
    fn tie_heavy_frames() -> (DataFrame, DataFrame) {
        let labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let features = DataFrame::new(
            8,
            vec![
                col("signal", &labels),
                col("unrelated", &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
                col("constant", &[5.0; 8]),
            ],
        )
        .unwrap();
        (features, target_of(&labels))
    }

    fn fitted(
        mut mi: MutualInformationSelector,
        x: DataFrame,
        y: DataFrame,
    ) -> MutualInformationSelector {
        mi.fit(x, y).unwrap();
        mi
    }

    fn score_of(mi: &MutualInformationSelector, name: &str) -> f64 {
        mi.scores()
            .unwrap()
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no score for column '{name}'"))
            .1
    }

    fn selected(mi: &MutualInformationSelector, x: DataFrame) -> Vec<String> {
        mi.transform(x)
            .unwrap()
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .collect()
    }

    #[test]
    fn test_classification_selects_the_determining_feature() {
        let (features, target) = classification_features(N);
        let mi = fitted(
            MutualInformationSelector::new().k(1),
            features.clone(),
            target,
        );

        assert_eq!(selected(&mi, features), ["informative"]);
    }

    #[test]
    fn test_classification_ranks_informative_above_unrelated() {
        let (features, target) = classification_features(N);
        let mi = fitted(MutualInformationSelector::new(), features, target);

        assert!(score_of(&mi, "informative") > score_of(&mi, "unrelated"));
        assert!(score_of(&mi, "informative") > score_of(&mi, "constant"));
        assert_eq!(score_of(&mi, "constant"), 0.0);
    }

    #[test]
    fn test_independent_feature_scores_far_below_the_informative_one() {
        let (features, target) = classification_features(N);
        let mi = fitted(MutualInformationSelector::new(), features, target);

        // The estimator's finite-sample bias keeps an unrelated feature above
        // zero; what must hold is that it stays far below a real dependency.
        let unrelated = score_of(&mi, "unrelated");
        assert!(
            unrelated >= 0.0 && unrelated < 0.25 * score_of(&mi, "informative"),
            "unrelated feature scored {unrelated}"
        );
    }

    #[test]
    fn test_constant_feature_scores_exactly_zero() {
        let (features, target) = classification_features(N);
        let mi = fitted(MutualInformationSelector::new(), features, target);

        assert_eq!(score_of(&mi, "constant"), 0.0);
    }

    #[test]
    fn test_scores_are_non_negative_and_sorted_descending() {
        let (features, target) = classification_features(N);
        let mi = fitted(MutualInformationSelector::new(), features, target);
        let scores = mi.scores().unwrap();

        assert_eq!(scores.len(), 3);
        for (name, score) in scores {
            assert!(*score >= 0.0, "{name} scored {score}");
            assert!(score.is_finite(), "{name} scored {score}");
        }
        for pair in scores.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "{scores:?}");
        }
    }

    #[test]
    fn test_classification_estimate_is_close_to_the_true_mutual_information() {
        // A balanced binary target and a feature that determines it: the true
        // mutual information is `ln 2` nats. The estimator is finite-sample
        // biased, so this asserts the band around it that the struct documents
        // rather than the exact value.
        let n = 200;
        let features = DataFrame::new(
            n,
            vec![col("signal", &separating(n)), col("noise", &uniform(n, 42))],
        )
        .unwrap();
        let mi = fitted(
            MutualInformationSelector::new(),
            features,
            target_of(&classes(n)),
        );

        assert_relative_eq!(score_of(&mi, "signal"), 2.0_f64.ln(), epsilon = 0.5);
        assert!(score_of(&mi, "signal") > 3.0 * score_of(&mi, "noise"));
    }

    #[test]
    fn test_scores_do_not_depend_on_row_order() {
        // Mutual information is a property of the joint distribution, not of
        // the order the rows arrive in; only float summation may differ.
        let reversed: Vec<usize> = (0..N).rev().collect();

        let (features, target) = classification_features(N);
        let straight = fitted(MutualInformationSelector::new(), features, target);
        let (features, target) = classification_features_in_order(N, &reversed);
        let shuffled = fitted(MutualInformationSelector::new(), features, target);
        for name in ["informative", "unrelated", "constant"] {
            assert_relative_eq!(
                score_of(&straight, name),
                score_of(&shuffled, name),
                epsilon = 1e-9
            );
        }

        let (features, target) = regression_frames(N);
        let straight = fitted(
            MutualInformationSelector::new().task(MITask::Regression),
            features,
            target,
        );
        let (features, target) = regression_frames_in_order(N, &reversed);
        let shuffled = fitted(
            MutualInformationSelector::new().task(MITask::Regression),
            features,
            target,
        );
        for name in ["linear", "unrelated"] {
            assert_relative_eq!(
                score_of(&straight, name),
                score_of(&shuffled, name),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_regression_selects_the_linearly_related_feature() {
        let (features, target) = regression_frames(200);
        let mi = fitted(
            MutualInformationSelector::new()
                .k(1)
                .task(MITask::Regression),
            features.clone(),
            target,
        );

        assert_eq!(selected(&mi, features), ["linear"]);
        assert!(score_of(&mi, "linear") > 0.0);
        assert!(score_of(&mi, "linear") > 3.0 * score_of(&mi, "unrelated"));
    }

    #[test]
    fn test_regression_independent_feature_scores_near_zero() {
        let (features, target) = regression_frames(200);
        let mi = fitted(
            MutualInformationSelector::new().task(MITask::Regression),
            features,
            target,
        );

        assert!(
            score_of(&mi, "unrelated") < 0.1,
            "independent feature scored {}",
            score_of(&mi, "unrelated")
        );
    }

    #[test]
    fn test_task_builder_switches_the_target_treatment() {
        let (features, target) = tie_heavy_frames();
        let classification = fitted(
            MutualInformationSelector::new().task(MITask::Classification),
            features.clone(),
            target.clone(),
        );
        let regression = fitted(
            MutualInformationSelector::new().task(MITask::Regression),
            features,
            target,
        );

        assert!(score_of(&classification, "signal") > 0.0);
        assert_ne!(
            score_of(&classification, "signal"),
            score_of(&regression, "signal")
        );
    }

    #[test]
    fn test_n_neighbors_builder_changes_the_estimate() {
        // The estimate smooths downward as the neighbourhood grows; the
        // ranking must not change with it.
        let (features, target) = regression_frames(200);
        let one = fitted(
            MutualInformationSelector::new()
                .task(MITask::Regression)
                .n_neighbors(1),
            features.clone(),
            target.clone(),
        );
        let ten = fitted(
            MutualInformationSelector::new()
                .task(MITask::Regression)
                .n_neighbors(10),
            features,
            target,
        );

        assert!(score_of(&one, "linear") > 0.0);
        assert!(score_of(&ten, "linear") > 0.0);
        assert!(
            score_of(&one, "linear") > score_of(&ten, "linear") + 0.1,
            "k = 1 scored {}, k = 10 scored {}",
            score_of(&one, "linear"),
            score_of(&ten, "linear")
        );
        assert!(score_of(&one, "linear") > score_of(&one, "unrelated"));
        assert!(score_of(&ten, "linear") > score_of(&ten, "unrelated"));
    }

    #[test]
    fn test_n_neighbors_at_or_above_sample_count_is_clamped() {
        let (features, target) = classification_features(N);
        let mi = fitted(
            MutualInformationSelector::new().k(1).n_neighbors(999),
            features.clone(),
            target,
        );

        // Clamping to `rows - 1` is the documented behaviour; the resulting
        // neighbourhood is too wide to be informative, but it must not panic
        // and every score must stay a finite, non-negative number.
        let scores = mi.scores().unwrap();
        assert_eq!(scores.len(), 3);
        for (name, score) in scores {
            assert!(*score >= 0.0 && score.is_finite(), "{name} scored {score}");
        }
        assert_eq!(mi.transform(features).unwrap().width(), 1);
    }

    #[test]
    fn test_zero_n_neighbors_is_clamped_to_one() {
        let (features, target) = classification_features(N);
        let mi = fitted(
            MutualInformationSelector::new().n_neighbors(0),
            features,
            target,
        );

        assert!(score_of(&mi, "informative") > 0.0);
        assert!(score_of(&mi, "informative").is_finite());
    }

    #[test]
    fn test_k_larger_than_feature_count_keeps_every_feature() {
        let (features, target) = classification_features(N);
        let mi = fitted(
            MutualInformationSelector::new().k(10),
            features.clone(),
            target,
        );

        assert_eq!(mi.transform(features).unwrap().width(), 3);
    }

    #[test]
    fn test_k_keeps_exactly_k_columns() {
        let (features, target) = classification_features(N);
        let mi = fitted(
            MutualInformationSelector::new().k(2),
            features.clone(),
            target,
        );

        assert_eq!(mi.transform(features).unwrap().width(), 2);
    }

    #[test]
    fn test_k_zero_is_rejected() {
        let mut mi = MutualInformationSelector::new().k(0);
        let (features, target) = classification_features(N);

        let error = mi.fit(features, target).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(
            error.to_string().contains("k must be greater than 0"),
            "{error}"
        );
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let (features, _) = classification_features(N);
        let mi = MutualInformationSelector::new();

        let error = mi.transform(features).unwrap_err();

        assert!(matches!(error, Error::NotFitted(_)), "{error}");
        assert!(mi.scores().is_none());
    }

    #[test]
    fn test_failed_refit_resets_state() {
        let (features, target) = classification_features(N);
        let mut mi = fitted(
            MutualInformationSelector::new().k(1),
            features.clone(),
            target.clone(),
        );
        assert!(mi.scores().is_some());

        mi.k = 0;
        assert!(mi.fit(features.clone(), target).is_err());

        assert!(mi.scores().is_none());
        assert!(matches!(
            mi.transform(features).unwrap_err(),
            Error::NotFitted(_)
        ));
    }

    #[test]
    fn test_zero_column_features_are_rejected() {
        let (_, target) = classification_features(N);
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(DataFrame::empty(), target).unwrap_err();

        assert!(error.to_string().contains("0 columns"), "{error}");
    }

    #[test]
    fn test_zero_row_features_are_rejected() {
        let (features, target) = classification_features(N);
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(features.slice(0, 0), target).unwrap_err();

        assert!(error.to_string().contains("0 rows"), "{error}");
    }

    #[test]
    fn test_multi_column_target_is_rejected() {
        let (features, _) = classification_features(N);
        let constant = vec![0.0; N];
        let y = DataFrame::new(N, vec![col("a", &classes(N)), col("b", &constant)]).unwrap();
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(features, y).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("target must have exactly 1 column but got 2"),
            "{error}"
        );
    }

    #[test]
    fn test_mismatched_target_rows_are_rejected() {
        let (features, _) = classification_features(N);
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(features, target_of(&classes(N - 1))).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("feature rows (40) and target rows (39) don't match"),
            "{error}"
        );
    }

    #[test]
    fn test_non_float64_target_is_rejected() {
        let (features, _) = classification_features(N);
        let labels: Vec<i64> = (0..N).map(|i| i64::from(i >= N / 2)).collect();
        let y = DataFrame::new(
            N,
            vec![Column::from(Series::new("target".into(), &labels[..]))],
        )
        .unwrap();
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(features, y).unwrap_err();

        assert!(
            error.to_string().contains("MutualInformationSelector"),
            "{error}"
        );
        assert!(error.to_string().contains("Float64"), "{error}");
    }

    #[test]
    fn test_no_float64_feature_columns_are_rejected() {
        let (_, target) = classification_features(N);
        let labels: Vec<String> = (0..N).map(|i| format!("row{i}")).collect();
        let features =
            DataFrame::new(N, vec![Column::from(Series::new("label".into(), labels))]).unwrap();
        let mut mi = MutualInformationSelector::new();

        let error = mi.fit(features, target).unwrap_err();

        assert!(
            error.to_string().contains("no Float64 columns found"),
            "{error}"
        );
    }

    #[test]
    fn test_non_float64_feature_columns_are_skipped() {
        let (mut features, target) = classification_features(N);
        let labels: Vec<String> = (0..N).map(|i| format!("row{i}")).collect();
        let labels = Column::from(Series::new("label".into(), labels));
        features.with_column(labels).unwrap();
        let mi = fitted(MutualInformationSelector::new(), features.clone(), target);

        assert_eq!(mi.scores().unwrap().len(), 3);
        assert_eq!(selected(&mi, features)[0], "informative");
        assert!(mi.scores().unwrap().iter().all(|(name, _)| name != "label"));
    }

    #[test]
    fn test_ties_break_by_column_name() {
        let features = DataFrame::new(
            N,
            vec![col("b_dup", &separating(N)), col("a_dup", &separating(N))],
        )
        .unwrap();
        let mi = fitted(
            MutualInformationSelector::new().k(1),
            features.clone(),
            target_of(&classes(N)),
        );

        assert_eq!(score_of(&mi, "a_dup"), score_of(&mi, "b_dup"));
        assert_eq!(selected(&mi, features), ["a_dup"]);
    }

    #[test]
    fn test_null_and_nan_rows_are_excluded() {
        // Two rows of the separating feature are unusable: one null, one NaN.
        let values = separating(N);
        let dirty: Vec<Option<f64>> = values
            .iter()
            .enumerate()
            .map(|(i, v)| match i {
                3 => None,
                23 => Some(f64::NAN),
                _ => Some(*v),
            })
            .collect();
        let constant = vec![5.0; N];
        let features = DataFrame::new(
            N,
            vec![
                Column::from(Series::new("dirty".into(), &dirty[..])),
                col("unrelated", &uniform(N, 12_345)),
                col("constant", &constant),
            ],
        )
        .unwrap();
        let mi = fitted(
            MutualInformationSelector::new().k(1),
            features.clone(),
            target_of(&classes(N)),
        );

        // 38 usable rows remain and the feature still determines the class.
        assert!(score_of(&mi, "dirty") > 0.0);
        assert_eq!(selected(&mi, features), ["dirty"]);
    }

    #[test]
    fn test_all_null_feature_scores_zero_and_is_not_selected() {
        let (mut features, target) = classification_features(N);
        let empty = Column::from(Series::new("empty".into(), &[None::<f64>; N]));
        features.with_column(empty).unwrap();
        let mi = fitted(
            MutualInformationSelector::new().k(3),
            features.clone(),
            target,
        );

        assert_eq!(score_of(&mi, "empty"), 0.0);
        assert!(!selected(&mi, features).contains(&"empty".to_string()));
    }

    #[test]
    fn test_default_builder_values() {
        let mi = MutualInformationSelector::default();

        assert_eq!(mi.k, 10);
        assert_eq!(mi.task, MITask::Classification);
        assert_eq!(mi.n_neighbors, 3);
        assert!(!mi.fitted);
        assert!(mi.selected_columns.is_none());
    }

    #[test]
    fn test_mitask_default_is_classification() {
        assert_eq!(MITask::default(), MITask::Classification);
    }
}
