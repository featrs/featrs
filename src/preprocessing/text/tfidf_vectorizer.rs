//! TF-IDF vectorization.
//!
//! [`TFIDFVectorizer`] turns a `String` column into a TF-IDF feature matrix. It
//! counts the terms of every document with
//! [`CountVectorizer`]
//! and reweights each count by the term's inverse document frequency, so terms
//! occurring in many documents (and therefore discriminating little) contribute
//! less than terms specific to a few.

use crate::preprocessing::text::count_vectorizer::{CountVectorizer, Tokenizer};
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// How a term's raw count is turned into its term frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TfWeighting {
    /// The raw term count.
    RawCount,
    /// The term count divided by the sum of the document's counted terms
    /// (`count / total_terms_in_doc`, both after any `sublinear_tf` rescaling).
    /// This is the default.
    #[default]
    TermFrequency,
    /// `1.0` when the term occurs in the document, `0.0` otherwise.
    Binary,
    /// `1 + ln(count)` for `count > 0`, `0.0` otherwise.
    ///
    /// This is the same transform as [`TfWeighting::RawCount`] combined with
    /// [`TFIDFVectorizer::sublinear_tf`]; the two spellings are both part of
    /// the API, but selecting both applies the logarithm twice.
    LogTf,
}

/// How a term's inverse document frequency is derived from its document
/// frequency `df` and the number of documents seen at fit time, `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IDFWeighting {
    /// `ln(N / df)`.
    Unsmoothed,
    /// `ln((1 + N) / (1 + df)) + 1`. This is the default, and matches
    /// scikit-learn's `TfidfTransformer`.
    #[default]
    Smoothed,
    /// `ln((N - df + 0.5) / (df + 0.5))`.
    Probabilistic,
}

/// How each output row is scaled after the TF-IDF weights are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormMethod {
    /// No scaling; the weights are emitted as they are.
    None,
    /// Divide each row by the sum of its absolute values.
    L1,
    /// Divide each row by its Euclidean length. This is the default, and
    /// matches scikit-learn.
    #[default]
    L2,
}

/// Convert a text column into TF-IDF weights.
///
/// `fit` learns a vocabulary by delegating to a
/// [`CountVectorizer`]
/// configured with the same tokenization settings (`column`, `tokenizer`,
/// `ngram_range`, `max_features`, `min_df`, `max_df`, `stop_words`,
/// `lowercase`), then computes one inverse document frequency per learned term
/// from the document frequency of that term. `transform` multiplies the term
/// counts of every document — computed by the same vectorizer, so the counts
/// are identical to the ones `fit` saw — by the term's weights, and returns a
/// new `DataFrame` with one `Float64` column per term, named after the term
/// itself and ordered like the vocabulary. Terms outside the vocabulary are
/// dropped and the input text column is not carried over, so the output width
/// is the vocabulary size.
///
/// The weighting pipeline runs in scikit-learn's order:
///
/// 1. `sublinear_tf` rescales the raw counts (`count > 0` becomes
///    `1 + ln(count)`, `0` stays `0`). [`TfWeighting::LogTf`] performs the same
///    transform as part of the weighting step, so enabling both applies the
///    logarithm twice.
/// 2. `tf_weighting` turns each rescaled count into a term frequency. For
///    [`TfWeighting::TermFrequency`] the numerator is the rescaled count and
///    the denominator is the sum of the document's rescaled counts — with
///    `sublinear_tf` that sum is `Σ (1 + ln(count))`, and terms outside the
///    vocabulary do not contribute to it at all.
/// 3. The term frequency is multiplied by the term's `idf_weighting` weight.
/// 4. `norm` scales every row (this step is skipped for
///    [`NormMethod::None`]).
///
/// A `null` or empty document counts no terms and becomes an all-zero row
/// (never a `null` row), and normalization leaves all-zero rows untouched
/// rather than dividing by zero, so the output holds no `NaN` or `±Inf` for
/// such rows. Every learned term occurred in at least one document, so every
/// inverse document frequency is finite.
///
/// A vocabulary that ends up empty — everything filtered out by `min_df`,
/// `max_df` or `stop_words` — transforms to a 0x0 frame, mirroring
/// [`CountVectorizer`],
/// because polars drops the height of a frame without columns.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::text::tfidf_vectorizer::TFIDFVectorizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new(
///     "text".into(),
///     &["the cat sat", "the dog ran", "cat and dog"],
/// ));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut v = TFIDFVectorizer::new().column("text");
/// v.fit(df.clone())?;
/// let tfidf = v.transform(df)?;
///
/// // The vocabulary — and therefore the column order — is the one learned by
/// // the underlying bag-of-words counts.
/// let names: Vec<String> = tfidf
///     .get_column_names()
///     .iter()
///     .map(|n| n.to_string())
///     .collect();
/// assert_eq!(names, vec!["cat", "dog", "the", "and", "ran", "sat"]);
/// assert_eq!(tfidf.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TFIDFVectorizer {
    fitted: bool,
    column: Option<String>,
    tokenizer: Tokenizer,
    ngram_range: (usize, usize),
    max_features: Option<usize>,
    min_df: usize,
    max_df: f64,
    stop_words: Option<Vec<String>>,
    lowercase: bool,
    tf_weighting: TfWeighting,
    idf_weighting: IDFWeighting,
    norm: NormMethod,
    sublinear_tf: bool,
    /// The fitted count vectorizer, which owns the learned vocabulary and the
    /// tokenization state, so `transform` counts terms exactly the way `fit`
    /// did. Its columns are the vocabulary entries in index order, which is
    /// also the order of `idf`.
    count_vectorizer: Option<CountVectorizer>,
    /// Per-term inverse document frequency, in vocabulary-index order.
    idf: Option<Vec<f64>>,
}

impl TFIDFVectorizer {
    /// Create a new `TFIDFVectorizer` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name of the `String` column to vectorize.
    pub fn column(mut self, c: &str) -> Self {
        self.column = Some(c.to_string());
        self.reset();
        self
    }

    /// Set the tokenizer (default [`Tokenizer::Whitespace`]).
    pub fn tokenizer(mut self, t: Tokenizer) -> Self {
        self.tokenizer = t;
        self.reset();
        self
    }

    /// Set the inclusive n-gram length range (default `(1, 1)`, unigrams).
    pub fn ngram_range(mut self, lo: usize, hi: usize) -> Self {
        self.ngram_range = (lo, hi);
        self.reset();
        self
    }

    /// Keep at most `n` terms, ranked by corpus frequency (`n >= 1`).
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self.reset();
        self
    }

    /// Drop terms appearing in fewer than `n` documents (default `1`).
    pub fn min_df(mut self, n: usize) -> Self {
        self.min_df = n;
        self.reset();
        self
    }

    /// Drop terms appearing in more than this fraction of documents
    /// (default `1.0`, i.e. no upper bound).
    pub fn max_df(mut self, p: f64) -> Self {
        self.max_df = p;
        self.reset();
        self
    }

    /// Set the words to remove from every document before n-grams are formed.
    pub fn stop_words(mut self, words: Vec<String>) -> Self {
        self.stop_words = Some(words);
        self.reset();
        self
    }

    /// Lowercase the text before tokenizing (default `true`).
    pub fn lowercase(mut self, b: bool) -> Self {
        self.lowercase = b;
        self.reset();
        self
    }

    /// Set how a term count becomes a term frequency (default
    /// [`TfWeighting::TermFrequency`]).
    pub fn tf_weighting(mut self, w: TfWeighting) -> Self {
        self.tf_weighting = w;
        self.reset();
        self
    }

    /// Set how a term's inverse document frequency is computed (default
    /// [`IDFWeighting::Smoothed`]).
    pub fn idf_weighting(mut self, w: IDFWeighting) -> Self {
        self.idf_weighting = w;
        self.reset();
        self
    }

    /// Set how each output row is normalized (default [`NormMethod::L2`]).
    pub fn norm(mut self, n: NormMethod) -> Self {
        self.norm = n;
        self.reset();
        self
    }

    /// Apply `1 + ln(count)` to the raw counts, before the term weighting
    /// (default `false`).
    pub fn sublinear_tf(mut self, b: bool) -> Self {
        self.sublinear_tf = b;
        self.reset();
        self
    }

    /// Discard the learned vocabulary and inverse document frequencies.
    ///
    /// Every configuration change invalidates what was learned from the old
    /// configuration, so a setter called after `fit` cannot leave `transform`
    /// applying stale weights.
    fn reset(&mut self) {
        self.fitted = false;
        self.count_vectorizer = None;
        self.idf = None;
    }

    /// Compose the inner [`CountVectorizer`] from this transformer's
    /// tokenization settings.
    fn build_count_vectorizer(&self) -> CountVectorizer {
        let mut vectorizer = CountVectorizer::new()
            .tokenizer(self.tokenizer.clone())
            .ngram_range(self.ngram_range.0, self.ngram_range.1)
            .min_df(self.min_df)
            .max_df(self.max_df)
            .lowercase(self.lowercase);
        if let Some(max) = self.max_features {
            vectorizer = vectorizer.max_features(max);
        }
        if let Some(words) = &self.stop_words {
            vectorizer = vectorizer.stop_words(words.clone());
        }
        if let Some(column) = &self.column {
            vectorizer = vectorizer.column(column);
        }
        vectorizer
    }
}

impl Default for TFIDFVectorizer {
    fn default() -> Self {
        Self {
            fitted: false,
            column: None,
            tokenizer: Tokenizer::Whitespace,
            ngram_range: (1, 1),
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            stop_words: None,
            lowercase: true,
            tf_weighting: TfWeighting::TermFrequency,
            idf_weighting: IDFWeighting::Smoothed,
            norm: NormMethod::L2,
            sublinear_tf: false,
            count_vectorizer: None,
            idf: None,
        }
    }
}

/// Re-label an error raised by the composed [`CountVectorizer`].
///
/// The inner vectorizer words its messages for itself; they are forwarded
/// under this transformer's name, so a caller is never told to configure a
/// `CountVectorizer` it never created.
fn relabel(error: Error) -> Error {
    fn rename(message: String) -> String {
        message.replace("CountVectorizer", "TFIDFVectorizer")
    }
    match error {
        Error::InvalidInput(message) => Error::InvalidInput(rename(message)),
        Error::NotFitted(message) => Error::NotFitted(rename(message)),
        Error::Computation(message) => Error::Computation(rename(message)),
    }
}

/// Rescale a raw term count when `sublinear_tf` is enabled.
///
/// A count of zero stays zero (instead of `1 + ln(0)`), so absent terms never
/// become `-Inf`.
fn sublinear(count: f64, enabled: bool) -> f64 {
    if enabled && count > 0.0 {
        1.0 + count.ln()
    } else {
        count
    }
}

/// The inverse document frequency of a term occurring in `df` of `n_docs`
/// documents.
///
/// A term only enters the learned vocabulary once it occurs in at least one
/// document, so `df >= 1` and none of these formulas can reach `ln(0)`.
fn inverse_document_frequency(weighting: IDFWeighting, n_docs: f64, df: f64) -> f64 {
    match weighting {
        IDFWeighting::Unsmoothed => (n_docs / df).ln(),
        IDFWeighting::Smoothed => ((1.0 + n_docs) / (1.0 + df)).ln() + 1.0,
        IDFWeighting::Probabilistic => ((n_docs - df + 0.5) / (df + 0.5)).ln(),
    }
}

/// The sum of the counted terms in every document, over `columns`.
///
/// `columns` holds the counts after any `sublinear_tf` rescaling, so this is
/// the denominator of [`TfWeighting::TermFrequency`]; terms outside the
/// vocabulary do not occur in the count matrix and therefore do not
/// contribute to it.
fn document_totals(columns: &[Vec<f64>], n_rows: usize) -> Vec<f64> {
    let mut totals = vec![0.0; n_rows];
    for values in columns {
        for (total, value) in totals.iter_mut().zip(values) {
            *total += *value;
        }
    }
    totals
}

/// The term frequency of a scaled count, per the configured weighting.
fn term_frequency(count: f64, total_terms: f64, weighting: TfWeighting) -> f64 {
    match weighting {
        TfWeighting::RawCount => count,
        TfWeighting::TermFrequency => {
            if total_terms > 0.0 {
                count / total_terms
            } else {
                0.0
            }
        }
        TfWeighting::Binary => {
            if count > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        TfWeighting::LogTf => {
            if count > 0.0 {
                1.0 + count.ln()
            } else {
                0.0
            }
        }
    }
}

/// Divide every row of the column-major `columns` by its L1 or L2 norm.
///
/// All-zero rows are left untouched rather than divided by zero, so a document
/// with no counted terms stays a zero row instead of becoming `NaN`.
fn normalize_rows(columns: &mut [Vec<f64>], norm: NormMethod) {
    if norm == NormMethod::None {
        return;
    }
    let n_rows = columns.first().map_or(0, Vec::len);
    for row in 0..n_rows {
        let length: f64 = columns
            .iter()
            .map(|values| match norm {
                NormMethod::L1 => values[row].abs(),
                _ => values[row] * values[row],
            })
            .sum();
        let length = if norm == NormMethod::L2 {
            length.sqrt()
        } else {
            length
        };
        if length > 0.0 {
            for values in columns.iter_mut() {
                values[row] /= length;
            }
        }
    }
}

impl Fit<DataFrame> for TFIDFVectorizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale state.
        self.reset();

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "TFIDFVectorizer.fit received an empty DataFrame. \
                 Provide at least one row and one column."
                    .into(),
            ));
        }

        let mut vectorizer = self.build_count_vectorizer();
        vectorizer.fit(x.clone()).map_err(relabel)?;
        // Counting the training documents with the same vectorizer keeps the
        // document frequencies in step with the vocabulary it just learned.
        let counts = vectorizer.transform(x).map_err(relabel)?;

        let n_docs = counts.height() as f64;
        let mut idf = Vec::with_capacity(counts.width());
        for column in counts.columns() {
            let ca = column
                .as_materialized_series()
                .f64()
                .map_err(|e| Error::Computation(format!("TFIDFVectorizer.fit: {}", e)))?;
            let document_frequency = ca
                .iter()
                .filter(|count| count.is_some_and(|count| count != 0.0))
                .count() as f64;
            idf.push(inverse_document_frequency(
                self.idf_weighting,
                n_docs,
                document_frequency,
            ));
        }

        self.count_vectorizer = Some(vectorizer);
        self.idf = Some(idf);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for TFIDFVectorizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let not_fitted = || {
            Error::NotFitted(
                "TFIDFVectorizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        };
        if !self.fitted {
            return Err(not_fitted());
        }
        let vectorizer = self.count_vectorizer.as_ref().ok_or_else(not_fitted)?;
        let idf = self.idf.as_ref().ok_or_else(not_fitted)?;

        // One Float64 count column per learned term, in vocabulary-index
        // order, one row per document.
        let counts = vectorizer.transform(x).map_err(relabel)?;
        if counts.width() == 0 {
            // polars ignores `height` for a frame without columns, so an empty
            // vocabulary can only be represented by a 0x0 frame.
            return Ok(DataFrame::empty());
        }
        if idf.len() != counts.width() {
            return Err(Error::Computation(format!(
                "TFIDFVectorizer: {} inverse document frequencies for a vocabulary of {} terms.",
                idf.len(),
                counts.width()
            )));
        }

        let n_rows = counts.height();
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(counts.width());
        for column in counts.columns() {
            let ca = column
                .as_materialized_series()
                .f64()
                .map_err(|e| Error::Computation(format!("TFIDFVectorizer.transform: {}", e)))?;
            let mut values = Vec::with_capacity(n_rows);
            for count in ca.iter() {
                values.push(sublinear(count.unwrap_or(0.0), self.sublinear_tf));
            }
            columns.push(values);
        }

        let totals = document_totals(&columns, n_rows);
        for (index, values) in columns.iter_mut().enumerate() {
            for (row, value) in values.iter_mut().enumerate() {
                *value = term_frequency(*value, totals[row], self.tf_weighting) * idf[index];
            }
        }
        normalize_rows(&mut columns, self.norm);

        // The count frame's columns are the vocabulary's terms in index order,
        // so its names line up with the weight columns; the terms are unique,
        // so the emitted names cannot collide.
        let out_cols: Vec<Column> = counts
            .get_column_names()
            .iter()
            .zip(columns)
            .map(|(name, values)| Column::from(Series::new(name.as_str().into(), &values)))
            .collect();

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_df(docs: &[&str]) -> DataFrame {
        let col = Column::from(Series::new("text".into(), docs));
        DataFrame::new(docs.len(), vec![col]).unwrap()
    }

    fn make_opt_df(docs: &[Option<&str>]) -> DataFrame {
        let ca: StringChunked = docs.iter().copied().collect();
        DataFrame::new(
            docs.len(),
            vec![Column::from(ca.into_series().with_name("text".into()))],
        )
        .unwrap()
    }

    /// Fit `v` on `docs`, returning the fitted vectorizer and the frame itself.
    fn fit_on(docs: &[&str], mut v: TFIDFVectorizer) -> (TFIDFVectorizer, DataFrame) {
        let df = make_df(docs);
        v.fit(df.clone()).unwrap();
        (v, df)
    }

    fn names(df: &DataFrame) -> Vec<&str> {
        df.get_column_names().iter().map(|n| n.as_str()).collect()
    }

    fn values(df: &DataFrame, name: &str) -> Vec<f64> {
        let ca = df
            .column(name)
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .clone();
        (0..df.height()).map(|i| ca.get(i).unwrap()).collect()
    }

    fn assert_values(df: &DataFrame, name: &str, expected: &[f64]) {
        let actual = values(df, name);
        assert_eq!(actual.len(), expected.len(), "length of column '{name}'");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_relative_eq!(*a, *e, max_relative = 1e-12);
            assert!(a.is_finite(), "column '{name}' row {i} is {a}");
        }
    }

    /// `assert_values` for every column of a small corpus at once.
    fn assert_matrix(df: &DataFrame, expected: &[(&str, &[f64])]) {
        assert_eq!(
            names(df),
            expected.iter().map(|(n, _)| *n).collect::<Vec<_>>()
        );
        for (name, column) in expected {
            assert_values(df, name, column);
        }
    }

    /// The issue's three-document corpus, hand-computed against the
    /// scikit-learn formula: `tf = count / terms_in_document`,
    /// `idf = ln((1 + N) / (1 + df)) + 1`, then L2 normalization.
    ///
    /// `N = 3`; `cat`/`dog`/`the` occur in two documents, so their weights are
    /// `ln(4 / 3) + 1 = 1.2876820724517808`, and `and`/`ran`/`sat` occur in one,
    /// so theirs are `ln(4 / 2) + 1 = 1.6931471805599454`.
    #[test]
    fn test_manual_tfidf_matches_the_sklearn_formula() {
        let (v, df) = fit_on(
            &["the cat sat", "the dog ran", "cat and dog"],
            TFIDFVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();

        // Raw weights of document 0: cat 0.42922735748392693, the
        // 0.42922735748392693, sat 0.5643823935199818, divided by the row's
        // Euclidean length 0.8288544715090902.
        assert_matrix(
            &out,
            &[
                ("cat", &[0.5178561161676974, 0.0, 0.5178561161676974]),
                ("dog", &[0.0, 0.5178561161676974, 0.5178561161676974]),
                ("the", &[0.5178561161676974, 0.5178561161676974, 0.0]),
                ("and", &[0.0, 0.0, 0.680918560398684]),
                ("ran", &[0.0, 0.680918560398684, 0.0]),
                ("sat", &[0.680918560398684, 0.0, 0.0]),
            ],
        );
        assert_eq!(out.height(), 3);
        assert_eq!(
            out.column("cat").unwrap().dtype(),
            &DataType::Float64,
            "the weights are emitted as Float64"
        );
    }

    /// A term occurring in every document gets a smoothed idf of exactly one:
    /// `ln((1 + N) / (1 + N)) + 1`.
    #[test]
    fn test_term_in_every_document_has_unit_idf() {
        let (v, df) = fit_on(
            &["a b", "a c"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();

        // "a" is in both documents, "b" and "c" in one each:
        // ln((1 + 2) / (1 + 1)) + 1 = 1.4054651081081644.
        assert_matrix(
            &out,
            &[
                ("a", &[1.0, 1.0]),
                ("b", &[1.4054651081081644, 0.0]),
                ("c", &[0.0, 1.4054651081081644]),
            ],
        );
    }

    /// A term occurring in exactly one of `N` documents gets
    /// `ln((1 + N) / 2) + 1`.
    #[test]
    fn test_single_document_term_uses_the_smoothed_formula() {
        let (v, df) = fit_on(
            &["a b", "a c", "a"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();

        // N = 3, so ln((1 + 3) / (1 + 1)) + 1 = ln(2) + 1.
        assert_values(&out, "b", &[1.6931471805599454, 0.0, 0.0]);
        assert_values(&out, "c", &[0.0, 1.6931471805599454, 0.0]);
    }

    /// `Unsmoothed` and `Probabilistic` idf follow their formulas.
    #[test]
    fn test_unsmoothed_and_probabilistic_idf() {
        let corpus = ["a b", "a c", "a"];

        let (v, df) = fit_on(
            &corpus,
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .idf_weighting(IDFWeighting::Unsmoothed)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();
        // "a" is in all three documents: ln(3 / 3) = 0. "b": ln(3 / 1).
        assert_values(&out, "a", &[0.0, 0.0, 0.0]);
        assert_values(&out, "b", &[1.0986122886681098, 0.0, 0.0]);

        let (v, df) = fit_on(
            &corpus,
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .idf_weighting(IDFWeighting::Probabilistic)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();
        // "a": ln((3 - 3 + 0.5) / (3 + 0.5)); "b": ln((3 - 1 + 0.5) / (1 + 0.5)).
        assert_values(
            &out,
            "a",
            &[
                -1.9459101490553135,
                -1.9459101490553135,
                -1.9459101490553135,
            ],
        );
        assert_values(&out, "b", &[0.5108256237659907, 0.0, 0.0]);
    }

    /// L2 normalization gives every non-zero row a unit Euclidean length.
    #[test]
    fn test_l2_rows_have_unit_length() {
        let (v, df) = fit_on(
            &["the cat sat", "the dog ran", "cat and dog"],
            TFIDFVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();
        for row in 0..out.height() {
            let length: f64 = names(&out)
                .iter()
                .map(|name| values(&out, name)[row].powi(2))
                .sum::<f64>()
                .sqrt();
            assert_relative_eq!(length, 1.0, max_relative = 1e-12);
        }
    }

    /// L1 normalization gives every non-zero row weights summing to one.
    #[test]
    fn test_l1_rows_sum_to_one() {
        let (v, df) = fit_on(
            &["the cat sat", "the dog ran", "cat and dog"],
            TFIDFVectorizer::new().column("text").norm(NormMethod::L1),
        );
        let out = v.transform(df).unwrap();
        for row in 0..out.height() {
            let total: f64 = names(&out)
                .iter()
                .map(|name| values(&out, name)[row].abs())
                .sum();
            assert_relative_eq!(total, 1.0, max_relative = 1e-12);
        }
        // Document 0 with L1 norm: cat 0.42922735748392693 / 1.4228371084878355.
        assert_values(&out, "cat", &[0.3016700611218256, 0.0, 0.3016700611218256]);
    }

    /// `TermFrequency` divides by the document's total after `sublinear_tf`,
    /// so the denominator is the sum of the rescaled counts.
    #[test]
    fn test_sublinear_tf_feeds_the_term_frequency_denominator() {
        let (v, df) = fit_on(
            &["a a b", "b"],
            TFIDFVectorizer::new()
                .column("text")
                .sublinear_tf(true)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();

        // Document 0 rescaled: "a" = 1 + ln(2), "b" = 1, so the row totals
        // 2.6931471805599454. "a" occurs in one of the two documents, "b" in
        // both.
        assert_matrix(
            &out,
            &[
                ("a", &[0.8835979341737835, 0.0]),
                ("b", &[0.37131279241563214, 1.0]),
            ],
        );
    }

    /// The weights are learned at fit and reused unchanged, however different
    /// the transform frame is: its own term occurrences and height play no
    /// part.
    #[test]
    fn test_weights_are_learned_at_fit_and_reused() {
        let (v, _) = fit_on(
            &["a b", "a c", "a"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .norm(NormMethod::None),
        );

        // "b" occurred in one of the three fitted documents, so its weight
        // stays ln((1 + 3) / (1 + 1)) + 1 even though it is the only term of
        // this frame and the only document.
        let out = v.transform(make_df(&["b"])).unwrap();
        assert_matrix(
            &out,
            &[("a", &[0.0]), ("b", &[1.6931471805599454]), ("c", &[0.0])],
        );
    }

    /// `sublinear_tf` maps a count of 5 to `1 + ln(5)` before the weighting.
    #[test]
    fn test_sublinear_tf_maps_counts_to_one_plus_ln() {
        let (v, df) = fit_on(
            &["a a a a a", "b"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .sublinear_tf(true)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();

        // "a" (five occurrences) and "b" (one) both occur in a single
        // document, so both have idf ln((1 + 2) / (1 + 1)) + 1 =
        // 1.4054651081081644, and "a" is rescaled to 1 + ln(5) first.
        assert_matrix(
            &out,
            &[
                ("a", &[3.667473937700736, 0.0]),
                ("b", &[0.0, 1.4054651081081644]),
            ],
        );
    }

    /// `LogTf` is the same transform as `RawCount` with `sublinear_tf`.
    #[test]
    fn test_log_tf_equals_raw_count_with_sublinear_tf() {
        let corpus = ["a a a a a", "b b"];

        let (v, df) = fit_on(
            &corpus,
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .sublinear_tf(true)
                .norm(NormMethod::None),
        );
        let logged = v.transform(df).unwrap();

        let (v, df) = fit_on(
            &corpus,
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::LogTf)
                .norm(NormMethod::None),
        );
        let log_tf = v.transform(df).unwrap();

        for name in names(&logged) {
            assert_values(&logged, name, &values(&log_tf, name));
        }
        // "b" counts twice in its only document and occurs in one of the two
        // documents, so its weight is (1 + ln(2)) * (ln(3 / 2) + 1).
        assert_values(&logged, "b", &[0.0, 2.3796592851687173]);
    }

    /// `Binary` collapses counts to presence and still multiplies by the idf.
    #[test]
    fn test_binary_tf_collapses_counts() {
        let (v, df) = fit_on(
            &["a a b", "a"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::Binary)
                .norm(NormMethod::None),
        );
        let out = v.transform(df).unwrap();

        // Three occurrences of "a" become 1, and "a" is in both documents
        // (idf 1); "b" occurs once, in one document only.
        assert_matrix(
            &out,
            &[("a", &[1.0, 1.0]), ("b", &[1.4054651081081644, 0.0])],
        );
    }

    /// Without normalization the weights are left large.
    #[test]
    fn test_norm_none_keeps_unscaled_weights() {
        let (v, df) = fit_on(
            &["a a a a a"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .norm(NormMethod::None),
        );
        // The only term is in the only document, so its idf is 1.
        assert_relative_eq!(values(&v.transform(df.clone()).unwrap(), "a")[0], 5.0);

        let (v, df) = fit_on(
            &["a a a a a"],
            TFIDFVectorizer::new()
                .column("text")
                .tf_weighting(TfWeighting::RawCount)
                .norm(NormMethod::L2),
        );
        // The same weights, scaled to a unit row.
        assert_relative_eq!(values(&v.transform(df).unwrap(), "a")[0], 1.0);
    }

    /// Null and empty documents become all-zero, non-null rows.
    #[test]
    fn test_empty_and_null_documents_are_zero_rows() {
        let df = make_opt_df(&[Some("cat"), Some(""), None, Some("cat dog")]);
        let mut v = TFIDFVectorizer::new().column("text");
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        // N = 4 documents; "cat" is in two of them, "dog" in one.
        assert_matrix(
            &out,
            &[
                ("cat", &[1.0, 0.0, 0.0, 0.6191302964899972]),
                ("dog", &[0.0, 0.0, 0.0, 0.7852882757103967]),
            ],
        );
        for name in names(&out) {
            let ca = out
                .column(name)
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .clone();
            assert_eq!(ca.get(1), Some(0.0), "empty document is not a zero row");
            assert_eq!(ca.get(2), Some(0.0), "null document is not a zero row");
        }
    }

    /// A one-term vocabulary keeps its single column at weight one.
    #[test]
    fn test_single_term_vocabulary() {
        let (v, df) = fit_on(&["a", "a"], TFIDFVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();

        // df == N == 2, so the smoothed idf is ln((1 + 2) / (1 + 2)) + 1 = 1.
        assert_matrix(&out, &[("a", &[1.0, 1.0])]);
    }

    /// Identical documents produce identical rows.
    #[test]
    fn test_identical_documents_produce_identical_rows() {
        let (v, df) = fit_on(&["a b", "a b"], TFIDFVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();

        // Both terms are in both documents (idf 1), so each row holds
        // 1 / sqrt(2) twice.
        assert_values(&out, "a", &[0.7071067811865475, 0.7071067811865475]);
        assert_values(&out, "b", &values(&out, "a"));
    }

    /// Terms outside the learned vocabulary are dropped at transform time.
    #[test]
    fn test_unknown_terms_are_dropped_at_transform() {
        let (v, _) = fit_on(&["hello world"], TFIDFVectorizer::new().column("text"));
        let out = v.transform(make_df(&["goodbye world"])).unwrap();

        // "world" is the only in-vocabulary term, so it is the document's
        // entire counted content: tf 1, and idf 1 from the single fit
        // document.
        assert_matrix(&out, &[("hello", &[0.0]), ("world", &[1.0])]);
    }

    /// `max_df(1.0)` filters nothing; a lower bound drops terms in many docs.
    #[test]
    fn test_max_df_one_point_zero_keeps_every_term() {
        let (v, df) = fit_on(
            &["a b", "a c"],
            TFIDFVectorizer::new().column("text").max_df(1.0),
        );
        assert_eq!(names(&v.transform(df).unwrap()), vec!["a", "b", "c"]);

        // "a" is in both documents, so 0.5 filters it out and keeps the
        // single-document terms.
        let (v, df) = fit_on(
            &["a b", "a c"],
            TFIDFVectorizer::new().column("text").max_df(0.5),
        );
        assert_eq!(names(&v.transform(df).unwrap()), vec!["b", "c"]);
    }

    /// An empty vocabulary transforms to a 0x0 frame.
    #[test]
    fn test_empty_vocabulary_yields_an_empty_frame() {
        let (v, df) = fit_on(&["", "  \t"], TFIDFVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);

        let (v, df) = fit_on(
            &["cat dog"],
            TFIDFVectorizer::new().column("text").min_df(2),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);
    }

    /// A zero-row transform input keeps the vocabulary width.
    #[test]
    fn test_transform_zero_row_frame() {
        let (v, _) = fit_on(&["cat dog"], TFIDFVectorizer::new().column("text"));
        let no_rows: StringChunked = Vec::<Option<&str>>::new().into_iter().collect();
        let df = DataFrame::new(
            0,
            vec![Column::from(no_rows.into_series().with_name("text".into()))],
        )
        .unwrap();
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 0);
    }

    /// Stop words are removed before n-grams are formed, exactly as in
    /// `CountVectorizer`.
    #[test]
    fn test_stop_words_are_removed_before_ngrams() {
        let (v, _) = fit_on(
            &["big the cat"],
            TFIDFVectorizer::new()
                .column("text")
                .stop_words(vec!["the".to_string()])
                .ngram_range(1, 2),
        );
        let out = v.transform(make_df(&["big the cat"])).unwrap();
        assert_eq!(names(&out), vec!["big", "big cat", "cat"]);
        // All three terms occur in the single training document, so each has
        // idf 1 and tf 1/3 before the row is scaled to unit length.
        assert_values(&out, "big cat", &[0.5773502691896257]);
        assert_values(&out, "big", &values(&out, "big cat"));
    }

    /// Every builder option reaches the fitted vectorizer.
    #[test]
    fn test_every_builder_option_is_honored() {
        let (v, df) = fit_on(
            &["The Cat", "the dog"],
            TFIDFVectorizer::new()
                .column("text")
                .tokenizer(Tokenizer::Whitespace)
                .ngram_range(1, 2)
                .max_features(4)
                .min_df(1)
                .max_df(1.0)
                .stop_words(Vec::new())
                .lowercase(true)
                .tf_weighting(TfWeighting::Binary)
                .idf_weighting(IDFWeighting::Smoothed)
                .norm(NormMethod::L1)
                .sublinear_tf(false),
        );
        let out = v.transform(df).unwrap();

        // Lowercased unigrams and bigrams of both documents, ranked by corpus
        // frequency ("the" wins) and truncated to the four most frequent.
        assert_eq!(names(&out), vec!["the", "cat", "dog", "the cat"]);
        for row in 0..out.height() {
            let total: f64 = names(&out)
                .iter()
                .map(|name| values(&out, name)[row].abs())
                .sum();
            assert_relative_eq!(total, 1.0, max_relative = 1e-12);
        }
    }

    /// `transform` before `fit` reports `NotFitted`, and a failed re-fit clears
    /// the state learned by an earlier successful fit.
    #[test]
    fn test_not_fitted_and_failed_refit_resets_state() {
        let df = make_df(&["cat dog"]);
        let mut v = TFIDFVectorizer::new().column("text");
        let err = v.transform(df.clone()).unwrap_err().to_string();
        assert!(err.contains("not fitted"), "{err}");

        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df.clone()).unwrap().width(), 2);

        v.column = Some("missing".into());
        assert!(v.fit(df.clone()).is_err());
        let err = v.transform(df).unwrap_err().to_string();
        assert!(
            err.contains("not fitted"),
            "stale state after failed refit: {err}"
        );
    }

    /// Every configuration change invalidates the learned weights.
    #[test]
    fn test_setter_invalidates_fitted_state() {
        let df = make_df(&["cat dog"]);
        let mut v = TFIDFVectorizer::new().column("text");
        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df.clone()).unwrap().width(), 2);

        let v = v.norm(NormMethod::L1);
        let err = v.transform(df).unwrap_err().to_string();
        assert!(err.contains("not fitted"), "{err}");
    }

    /// Every invalid configuration surfaces as `Error::InvalidInput` at fit,
    /// named after this transformer rather than the composed count vectorizer.
    #[test]
    fn test_fit_validation_errors() {
        let df = make_df(&["cat dog"]);

        let cases = vec![
            (TFIDFVectorizer::new(), "no column configured".to_string()),
            (
                TFIDFVectorizer::new().column("missing"),
                "not found".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").ngram_range(0, 3),
                "ngram_range".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").ngram_range(3, 2),
                "ngram_range".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").min_df(0),
                "min_df".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").max_features(0),
                "max_features".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").max_df(0.0),
                "max_df".to_string(),
            ),
            (
                TFIDFVectorizer::new().column("text").max_df(1.5),
                "max_df".to_string(),
            ),
            (
                TFIDFVectorizer::new()
                    .column("text")
                    .tokenizer(Tokenizer::WordRegex("(".into())),
                "invalid regex".to_string(),
            ),
        ];
        for (mut v, expected) in cases {
            let err = v.fit(df.clone()).unwrap_err().to_string();
            assert!(err.contains(&expected), "{expected}: {err}");
            assert!(err.contains("TFIDFVectorizer"), "{expected}: {err}");
        }

        let numeric = DataFrame::new(
            3,
            vec![Column::from(Series::new("text".into(), &[1i32, 2, 3]))],
        )
        .unwrap();
        let err = TFIDFVectorizer::new()
            .column("text")
            .fit(numeric)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected String"), "{err}");

        let no_rows: StringChunked = Vec::<Option<&str>>::new().into_iter().collect();
        let empty_rows = DataFrame::new(
            0,
            vec![Column::from(no_rows.into_series().with_name("text".into()))],
        )
        .unwrap();
        for empty in [DataFrame::empty(), empty_rows] {
            let err = TFIDFVectorizer::new()
                .column("text")
                .fit(empty)
                .unwrap_err()
                .to_string();
            assert!(err.contains("empty DataFrame"), "{err}");
        }
    }

    /// `transform` validates its own input column too.
    #[test]
    fn test_transform_validation_errors() {
        let (v, _) = fit_on(&["cat dog"], TFIDFVectorizer::new().column("text"));

        let other =
            DataFrame::new(1, vec![Column::from(Series::new("other".into(), &["cat"]))]).unwrap();
        let err = v.transform(other).unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
        assert!(err.contains("TFIDFVectorizer"), "{err}");

        let numeric =
            DataFrame::new(1, vec![Column::from(Series::new("text".into(), &[1i32]))]).unwrap();
        let err = v.transform(numeric).unwrap_err().to_string();
        assert!(err.contains("expected String"), "{err}");
    }
}
