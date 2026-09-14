//! Bag-of-words vectorization.
//!
//! [`CountVectorizer`] turns a `String` column into a term-count matrix: each
//! row is a document, each column is a vocabulary term, and each cell is the
//! number of times that term occurs in that document. It is the word-level
//! counterpart of [`character n-grams`](crate::preprocessing::text::char_ngram_vectorizer),
//! which slice a document into characters instead of tokens.
//!
//! The vocabulary is learned by `fit` from the training column and reused by
//! `transform`; terms that were never seen during `fit` are silently dropped.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use regex::Regex;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

/// How a document is split into tokens.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Tokenizer {
    /// Split on Unicode whitespace, keeping every non-empty run as a token
    /// (`str::split_whitespace`). Single-character tokens are kept, unlike
    /// scikit-learn's default `token_pattern`.
    #[default]
    Whitespace,
    /// Keep every non-empty match of a user-supplied Rust `regex` pattern as a
    /// token, in the order the matches appear. The token is the whole match, so
    /// a capture group does not narrow it (use `find_iter`, not
    /// `captures_iter`, semantics); empty matches are skipped. Use this to keep
    /// punctuation out of the vocabulary (`\w+`), to restrict tokens to letters
    /// (`[a-z]+`), or to pull tokens out of text that splitting on whitespace
    /// cannot separate (e.g. `hi,there` under `\w+`). Note that the pattern
    /// *selects* tokens rather than splitting on separators, so a separator
    /// pattern would put the separators themselves into the vocabulary.
    WordRegex(String),
}

/// Convert a text column into bag-of-words term counts.
///
/// `fit` tokenizes every document of the configured column, expands each
/// document into the n-grams of `ngram_range`, and learns a vocabulary from
/// the terms it sees. `min_df` and `max_df` filter terms by document frequency
/// and `max_features` then keeps the most frequent ones. `transform` counts,
/// per document, how often each learned term occurs and returns a new
/// `DataFrame` with one `Float64` column per term, named after the term
/// itself — bigrams contain the joining space, e.g. `"the cat"`. Terms missing
/// from the vocabulary are dropped, so the output width is the vocabulary size
/// and the input column is not carried over.
///
/// A vocabulary that ends up empty is returned as a 0x0 frame (polars cannot
/// represent columns-less frames with a row count), so a downstream pipeline
/// step receives an empty frame rather than one row per document. `null` cells
/// contribute no terms but still count as documents in the `min_df`/`max_df`
/// denominators.
///
/// The order of the emitted columns is part of the contract: terms are ranked
/// by corpus frequency (total occurrences, not document frequency), ties are
/// broken alphabetically, and `max_features` truncates that ranking, so a
/// vocabulary index — and therefore the column order — is deterministic across
/// runs. This is a deliberate deviation from scikit-learn, whose `vocabulary_`
/// enumerates the same terms alphabetically. `min_df == 0`, `max_features(0)`,
/// an `ngram_range` that is not `1 <= lo <= hi`, a `max_df` outside
/// `(0.0, 1.0]`, and an invalid [`Tokenizer::WordRegex`] pattern are all
/// rejected by `fit`.
///
/// # Pipelines
///
/// - `lowercase` is applied to the whole document before tokenization, so the
///   default vocabulary folds case (`"The"` and `"the"` are one term).
/// - `stop_words` are matched against the token exactly as it comes out of the
///   tokenizer (i.e. after lowercasing), and are removed *before* n-grams are
///   formed, so no term ever contains a stop word. Removing a stop word makes
///   its neighbours adjacent, so they can form a new n-gram: `"big the cat"`
///   with `"the"` removed yields the bigram `"big cat"`, as in scikit-learn.
///   Pass already-lowercased stop words when `lowercase` is on.
/// - With a large vocabulary (`V > 10000`) the output is a dense `V`-column
///   frame of `Float64`s, which is usually the wrong representation — cap
///   `max_features`, raise `min_df`, or reach for a hashing transformer when
///   the term space is unbounded.
/// - `strip_accents` is deliberately not implemented: it would need a Unicode
///   normalization crate, which the text preprocessing in this crate avoids.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::text::count_vectorizer::CountVectorizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new(
///     "text".into(),
///     &["the cat sat", "the dog ran", "cat and dog"],
/// ));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut v = CountVectorizer::new().column("text");
/// v.fit(df.clone())?;
/// let counts = v.transform(df)?;
///
/// // "cat", "dog" and "the" all occur twice; the tie is alphabetical.
/// let names: Vec<String> = counts
///     .get_column_names()
///     .iter()
///     .map(|n| n.to_string())
///     .collect();
/// assert_eq!(names, vec!["cat", "dog", "the", "and", "ran", "sat"]);
/// assert_eq!(counts.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CountVectorizer {
    fitted: bool,
    column: Option<String>,
    tokenizer: Tokenizer,
    ngram_range: (usize, usize),
    max_features: Option<usize>,
    min_df: usize,
    max_df: f64,
    stop_words: Option<Vec<String>>,
    lowercase: bool,
    vocabulary: Option<HashMap<String, u32>>,
    /// Terms in vocabulary-index order (`terms[i]` is the term whose vocabulary
    /// index is `i`), so `transform` emits columns deterministically instead of
    /// iterating the `HashMap`.
    terms: Vec<String>,
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings.
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
    ///
    /// An n-gram is the `n` adjacent tokens joined with a single space, e.g.
    /// `"the cat"` under `ngram_range(2, 2)`.
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

    /// Discard the learned vocabulary.
    ///
    /// Every configuration change invalidates the vocabulary it was learned
    /// from, so a setter called after `fit` cannot leave `transform` applying a
    /// stale vocabulary to the new configuration.
    fn reset(&mut self) {
        self.fitted = false;
        self.vocabulary = None;
        self.terms.clear();
    }
}

impl Default for CountVectorizer {
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
            vocabulary: None,
            terms: Vec::new(),
        }
    }
}

/// Compile the tokenizer's pattern, if any.
///
/// `None` means the whitespace tokenizer, which needs no regex. Compiling up
/// front turns an invalid pattern into an `InvalidInput` error instead of a
/// panic, and lets a whole fit/transform run reuse one compiled pattern.
fn compile_tokenizer(tokenizer: &Tokenizer) -> Result<Option<Regex>> {
    match tokenizer {
        Tokenizer::Whitespace => Ok(None),
        Tokenizer::WordRegex(pattern) => Regex::new(pattern).map(Some).map_err(|error| {
            Error::InvalidInput(format!(
                "CountVectorizer: invalid regex pattern '{}': {}. \
                 Replace Tokenizer::WordRegex with a valid Rust regex, \
                 or use Tokenizer::Whitespace.",
                pattern, error
            ))
        }),
    }
}

/// Call `f` with every term generated from `text`.
///
/// `text` is tokenized with `regex` when one is given and on whitespace
/// otherwise, the `stop_words` are removed, and the remaining tokens are
/// expanded into n-grams of length `lo..=hi` — each one the `n` adjacent
/// tokens joined with a single space. The join buffer is reused across terms.
fn for_each_term(
    text: &str,
    regex: Option<&Regex>,
    stop_words: &HashSet<&str>,
    lo: usize,
    hi: usize,
    mut f: impl FnMut(&str),
) {
    let tokens: Vec<&str> = match regex {
        None => text
            .split_whitespace()
            .filter(|token| !stop_words.contains(*token))
            .collect(),
        Some(re) => re
            .find_iter(text)
            .map(|m| m.as_str())
            .filter(|token| !token.is_empty() && !stop_words.contains(*token))
            .collect(),
    };

    let hi = hi.min(tokens.len());
    let mut term = String::new();
    // `lo..=hi` is empty when `lo > hi`, so short documents yield nothing.
    for n in lo..=hi {
        for window in tokens.windows(n) {
            term.clear();
            for (i, token) in window.iter().enumerate() {
                if i > 0 {
                    term.push(' ');
                }
                term.push_str(token);
            }
            f(&term);
        }
    }
}

impl Fit<DataFrame> for CountVectorizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale state.
        self.reset();

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "CountVectorizer.fit received an empty DataFrame. \
                 Provide at least one row and one column."
                    .into(),
            ));
        }

        let column = self.column.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "CountVectorizer.fit: no column configured. \
                 Call .column(\"name\") before .fit()."
                    .into(),
            )
        })?;
        let s = x
            .column(column.as_str())
            .map_err(|e| {
                Error::InvalidInput(format!(
                    "CountVectorizer.fit: column '{}' not found. {}",
                    column, e
                ))
            })?
            .as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "CountVectorizer.fit: column '{}' has dtype {}; expected String. {}",
                column,
                s.dtype(),
                e
            ))
        })?;

        let (lo, hi) = self.ngram_range;
        if lo < 1 || lo > hi {
            return Err(Error::InvalidInput(format!(
                "CountVectorizer: invalid ngram_range ({}, {}); \
                 require 1 <= lower <= upper.",
                lo, hi
            )));
        }
        if self.min_df == 0 {
            return Err(Error::InvalidInput(
                "CountVectorizer: min_df must be >= 1.".into(),
            ));
        }
        if self.max_features == Some(0) {
            return Err(Error::InvalidInput(
                "CountVectorizer: max_features must be >= 1.".into(),
            ));
        }
        if !(self.max_df > 0.0 && self.max_df <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "CountVectorizer: max_df must be in (0.0, 1.0], got {}.",
                self.max_df
            )));
        }
        let regex = compile_tokenizer(&self.tokenizer)?;
        let stop_words: HashSet<&str> = self
            .stop_words
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(String::as_str)
            .collect();

        let n_docs = x.height() as f64;
        // One entry per distinct term: (document frequency, total occurrences).
        let mut stats: HashMap<String, (u64, u64)> = HashMap::new();
        for doc in ca.iter() {
            let Some(doc) = doc else { continue };
            let text: Cow<str> = if self.lowercase {
                Cow::Owned(doc.to_lowercase())
            } else {
                Cow::Borrowed(doc)
            };
            let mut counts: HashMap<String, u64> = HashMap::new();
            for_each_term(
                &text,
                regex.as_ref(),
                &stop_words,
                lo,
                hi,
                |term| match counts.get_mut(term) {
                    Some(count) => *count += 1,
                    None => {
                        counts.insert(term.to_owned(), 1);
                    }
                },
            );
            for (term, count) in counts {
                let entry = stats.entry(term).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += count;
            }
        }

        let mut kept: Vec<(String, u64)> = stats
            .into_iter()
            .filter(|(_, (doc_freq, _))| {
                *doc_freq >= self.min_df as u64 && *doc_freq as f64 <= self.max_df * n_docs
            })
            .map(|(term, (_, total_freq))| (term, total_freq))
            .collect();
        // Deterministic ranking: most frequent first, alphabetically on ties.
        kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some(max) = self.max_features {
            kept.truncate(max);
        }

        let mut vocabulary = HashMap::with_capacity(kept.len());
        for (idx, (term, _)) in kept.into_iter().enumerate() {
            vocabulary.insert(term.clone(), idx as u32);
            self.terms.push(term);
        }
        self.vocabulary = Some(vocabulary);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CountVectorizer {
    type Output = DataFrame;

    /// Count the fitted vocabulary's terms per document.
    ///
    /// Returns a frame of `Float64` counts (one column per term, one row per
    /// document). When the vocabulary is empty the result is a 0x0 frame, since
    /// polars drops the height of a frame without columns — downstream steps
    /// receive an empty frame rather than one row per document.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let not_fitted = || {
            Error::NotFitted(
                "CountVectorizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        };
        if !self.fitted {
            return Err(not_fitted());
        }
        let vocabulary = self.vocabulary.as_ref().ok_or_else(not_fitted)?;

        let column = self.column.as_ref().ok_or_else(not_fitted)?;
        let s = x
            .column(column.as_str())
            .map_err(|e| {
                Error::InvalidInput(format!(
                    "CountVectorizer.transform: column '{}' not found. {}",
                    column, e
                ))
            })?
            .as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "CountVectorizer.transform: column '{}' has dtype {}; \
                 expected String. {}",
                column,
                s.dtype(),
                e
            ))
        })?;

        if vocabulary.is_empty() {
            // polars ignores `height` for a frame without columns, so an empty
            // vocabulary can only be represented by a 0x0 frame.
            return Ok(DataFrame::empty());
        }

        let regex = compile_tokenizer(&self.tokenizer)?;
        let stop_words: HashSet<&str> = self
            .stop_words
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(String::as_str)
            .collect();

        let n_rows = x.height();
        let (lo, hi) = self.ngram_range;
        let mut counts = vec![vec![0.0f64; n_rows]; self.terms.len()];
        for (row, doc) in ca.iter().enumerate() {
            let Some(doc) = doc else { continue };
            let text: Cow<str> = if self.lowercase {
                Cow::Owned(doc.to_lowercase())
            } else {
                Cow::Borrowed(doc)
            };
            for_each_term(&text, regex.as_ref(), &stop_words, lo, hi, |term| {
                if let Some(idx) = vocabulary.get(term) {
                    counts[*idx as usize][row] += 1.0;
                }
            });
        }

        // Terms are the vocabulary's unique keys, so the emitted column names
        // cannot collide; the frame is built from scratch rather than by
        // appending a column at a time, so no collision guard is needed.
        let out_cols: Vec<Column> = self
            .terms
            .iter()
            .enumerate()
            .map(|(idx, term)| Column::from(Series::new(term.as_str().into(), &counts[idx])))
            .collect();

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df(docs: &[&str]) -> DataFrame {
        let col = Column::from(Series::new("text".into(), docs));
        DataFrame::new(docs.len(), vec![col]).unwrap()
    }

    /// Fit `v` on `docs`, returning the fitted vectorizer and the frame itself.
    fn fit_on(docs: &[&str], mut v: CountVectorizer) -> (CountVectorizer, DataFrame) {
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

    /// The issue's corpus: six terms, ranked by corpus frequency with
    /// alphabetical ties, and the exact per-document counts.
    #[test]
    fn test_small_corpus_counts_and_vocabulary_order() {
        let (v, df) = fit_on(
            &["the cat sat", "the dog ran", "cat and dog"],
            CountVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();

        assert_eq!(out.height(), 3);
        assert_eq!(out.width(), 6);
        // "cat", "dog" and "the" each occur twice; the tie is alphabetical.
        assert_eq!(names(&out), vec!["cat", "dog", "the", "and", "ran", "sat"]);
        assert_eq!(values(&out, "the"), vec![1.0, 1.0, 0.0]);
        assert_eq!(values(&out, "cat"), vec![1.0, 0.0, 1.0]);
        assert_eq!(values(&out, "sat"), vec![1.0, 0.0, 0.0]);
        assert_eq!(values(&out, "dog"), vec![0.0, 1.0, 1.0]);
        assert_eq!(values(&out, "ran"), vec![0.0, 1.0, 0.0]);
        assert_eq!(values(&out, "and"), vec![0.0, 0.0, 1.0]);
    }

    /// `transform` returns only vocabulary columns, not the input text column.
    #[test]
    fn test_text_column_is_not_carried_over() {
        let (v, df) = fit_on(&["the cat sat"], CountVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert!(out.column("text").is_err());
        assert_eq!(out.width(), 3);
    }

    /// `min_df` filters on document frequency.
    #[test]
    fn test_min_df_filters_rare_terms() {
        let corpus = ["the cat sat", "the dog ran", "cat and dog"];

        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text").min_df(2));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat", "dog", "the"]);

        // A threshold above the number of documents keeps nothing.
        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text").min_df(4));
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);
    }

    /// `max_df` filters on the fraction of documents a term appears in; a term
    /// in exactly `max_df` of the documents is kept (`<=`).
    #[test]
    fn test_max_df_filters_common_terms() {
        let corpus = ["the cat sat", "the dog ran", "cat and dog"];

        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text").max_df(0.5));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["and", "ran", "sat"]);

        // The default 1.0 filters nothing.
        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text").max_df(1.0));
        assert_eq!(v.transform(df).unwrap().width(), 6);

        // A threshold below every observed frequency keeps nothing.
        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text").max_df(0.01));
        assert_eq!(v.transform(df).unwrap().width(), 0);

        // Two documents, a term in both: kept at 1.0, dropped at 0.5.
        let pair = ["cat dog", "cat"];
        let (v, df) = fit_on(&pair, CountVectorizer::new().column("text").max_df(1.0));
        assert_eq!(v.transform(df).unwrap().width(), 2);
        let (v, df) = fit_on(&pair, CountVectorizer::new().column("text").max_df(0.5));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["dog"]);
    }

    /// `max_features` ranks by corpus frequency and breaks ties alphabetically.
    #[test]
    fn test_max_features_keeps_most_frequent_terms() {
        let corpus = ["the cat sat", "the dog ran", "cat and dog"];
        let (v, df) = fit_on(
            &corpus,
            CountVectorizer::new().column("text").max_features(2),
        );
        let out = v.transform(df).unwrap();
        // Three terms tie at two occurrences; the first two alphabetically win.
        assert_eq!(names(&out), vec!["cat", "dog"]);
        assert_eq!(values(&out, "cat"), vec![1.0, 0.0, 1.0]);
        assert_eq!(values(&out, "dog"), vec![0.0, 1.0, 1.0]);
    }

    /// `ngram_range(1, 2)` emits unigrams and bigrams; the default is unigrams.
    #[test]
    fn test_ngram_range_adds_bigrams() {
        let (v, df) = fit_on(
            &["the cat sat"],
            CountVectorizer::new().column("text").ngram_range(1, 2),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat", "cat sat", "sat", "the", "the cat"]);
        assert_eq!(values(&out, "the cat"), vec![1.0]);
        assert_eq!(values(&out, "cat sat"), vec![1.0]);

        let (v, df) = fit_on(&["the cat sat"], CountVectorizer::new().column("text"));
        assert_eq!(names(&v.transform(df).unwrap()), vec!["cat", "sat", "the"]);
    }

    /// A document shorter than the lower n-gram bound contributes no term, so
    /// it becomes an all-zero row.
    #[test]
    fn test_document_shorter_than_ngram_range() {
        let (v, df) = fit_on(
            &["a b", "a"],
            CountVectorizer::new().column("text").ngram_range(2, 2),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["a b"]);
        assert_eq!(values(&out, "a b"), vec![1.0, 0.0]);
    }

    /// A zero-row transform input keeps the vocabulary width.
    #[test]
    fn test_transform_zero_row_frame() {
        let (v, _) = fit_on(&["cat dog"], CountVectorizer::new().column("text"));
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

    /// Stop words are removed before n-grams are formed, so a term never
    /// contains a stop word: the tokens on either side become adjacent and can
    /// form a new n-gram.
    #[test]
    fn test_stop_words_are_removed_before_ngram_formation() {
        let (v, df) = fit_on(
            &["the cat sat"],
            CountVectorizer::new()
                .column("text")
                .stop_words(vec!["the".to_string()])
                .ngram_range(1, 2),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat", "cat sat", "sat"]);
        assert!(out.column("the").is_err());
        assert!(out.column("the cat").is_err());

        // Removing "the" joins its neighbours, so "big" and "cat" become a
        // bigram that the unfiltered corpus would not produce.
        let (v, _) = fit_on(
            &["big the cat"],
            CountVectorizer::new()
                .column("text")
                .stop_words(vec!["the".to_string()])
                .ngram_range(1, 2),
        );
        let out = v.transform(make_df(&["big the cat"])).unwrap();
        assert_eq!(names(&out), vec!["big", "big cat", "cat"]);
        assert_eq!(values(&out, "big cat"), vec![1.0]);
    }

    /// Repeats inside one document are counted, not collapsed.
    #[test]
    fn test_repeated_terms_are_counted_per_document() {
        let (v, df) = fit_on(
            &["cat cat cat", "cat"],
            CountVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat"]);
        assert_eq!(values(&out, "cat"), vec![3.0, 1.0]);
    }

    /// Terms absent from the vocabulary are dropped at transform time.
    #[test]
    fn test_unknown_terms_are_dropped_at_transform() {
        let (v, _) = fit_on(&["hello world"], CountVectorizer::new().column("text"));
        let out = v.transform(make_df(&["goodbye world"])).unwrap();
        assert_eq!(names(&out), vec!["hello", "world"]);
        assert_eq!(values(&out, "hello"), vec![0.0]);
        assert_eq!(values(&out, "world"), vec![1.0]);
    }

    /// `transform` accepts a frame with a different number of rows than `fit`.
    #[test]
    fn test_transform_accepts_a_different_frame() {
        let (v, _) = fit_on(&["cat dog"], CountVectorizer::new().column("text"));
        let out = v.transform(make_df(&["dog", "cat cat"])).unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(values(&out, "cat"), vec![0.0, 2.0]);
        assert_eq!(values(&out, "dog"), vec![1.0, 0.0]);
    }

    /// Empty and whitespace-only documents produce all-zero rows.
    #[test]
    fn test_empty_and_whitespace_documents_are_zero_rows() {
        let (v, df) = fit_on(
            &["cat", "", "   ", "dog cat"],
            CountVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat", "dog"]);
        assert_eq!(values(&out, "cat"), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(values(&out, "dog"), vec![0.0, 0.0, 0.0, 1.0]);
    }

    /// A corpus with no tokens at all still fits, and transforms to 0x0.
    #[test]
    fn test_all_empty_corpus_yields_empty_frame() {
        let (v, df) = fit_on(&["", "  \t"], CountVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);

        // The input column is still validated when the vocabulary is empty.
        let other =
            DataFrame::new(1, vec![Column::from(Series::new("other".into(), &[""]))]).unwrap();
        let err = v.transform(other).unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }

    /// A single-document corpus gives every term a document frequency of 1.
    #[test]
    fn test_single_document() {
        let (v, df) = fit_on(&["cat dog cat"], CountVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["cat", "dog"]);
        assert_eq!(values(&out, "cat"), vec![2.0]);
        assert_eq!(values(&out, "dog"), vec![1.0]);

        let (v, df) = fit_on(
            &["cat dog cat"],
            CountVectorizer::new().column("text").min_df(2),
        );
        assert_eq!(v.transform(df).unwrap().width(), 0);
    }

    /// `Tokenizer::WordRegex` keeps each non-empty pattern match as a token,
    /// instead of splitting on whitespace.
    #[test]
    fn test_word_regex_tokenizer() {
        let corpus = ["hi,there", "hi there"];
        let (v, df) = fit_on(
            &corpus,
            CountVectorizer::new()
                .column("text")
                .tokenizer(Tokenizer::WordRegex(r"\w+".into())),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["hi", "there"]);
        assert_eq!(values(&out, "hi"), vec![1.0, 1.0]);
        assert_eq!(values(&out, "there"), vec![1.0, 1.0]);

        // The whitespace tokenizer keeps "hi,there" as a single token.
        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["hi", "hi,there", "there"]);

        // Text the pattern does not match is skipped entirely.
        let (v, df) = fit_on(
            &["abc-123"],
            CountVectorizer::new()
                .column("text")
                .tokenizer(Tokenizer::WordRegex(r"[a-z]+".into())),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["abc"]);
        assert_eq!(values(&out, "abc"), vec![1.0]);
    }

    /// `lowercase` (on by default) is applied before tokenizing.
    #[test]
    fn test_lowercase_option() {
        let corpus = ["Hello World", "hello world"];

        let (v, df) = fit_on(&corpus, CountVectorizer::new().column("text"));
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["hello", "world"]);
        assert_eq!(values(&out, "hello"), vec![1.0, 1.0]);

        let (v, df) = fit_on(
            &corpus,
            CountVectorizer::new().column("text").lowercase(false),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["Hello", "World", "hello", "world"]);
        assert_eq!(values(&out, "Hello"), vec![1.0, 0.0]);
        assert_eq!(values(&out, "hello"), vec![0.0, 1.0]);
    }

    /// Lowercasing is Unicode-aware, so `É` and `é` fold to one term.
    #[test]
    fn test_unicode_lowercasing() {
        let (v, df) = fit_on(
            &["Café CAFÉ", "café"],
            CountVectorizer::new().column("text"),
        );
        let out = v.transform(df).unwrap();
        assert_eq!(names(&out), vec!["café"]);
        assert_eq!(values(&out, "café"), vec![2.0, 1.0]);
    }

    /// Null cells yield no tokens but still count in the `min_df` denominator.
    #[test]
    fn test_null_documents_are_empty_but_still_count_as_documents() {
        let ca: StringChunked = [Some("cat dog"), None, Some("cat")].into_iter().collect();
        let df = DataFrame::new(
            3,
            vec![Column::from(ca.into_series().with_name("text".into()))],
        )
        .unwrap();

        let mut v = CountVectorizer::new().column("text").min_df(2);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();
        // "cat" is in 2 of the 3 documents; the null row still counts as one.
        assert_eq!(names(&out), vec!["cat"]);
        assert_eq!(values(&out, "cat"), vec![1.0, 0.0, 1.0]);
    }

    /// `transform` before `fit` reports `NotFitted`, and a failed re-fit clears
    /// the state learned by an earlier successful fit.
    #[test]
    fn test_not_fitted_and_failed_refit_resets_state() {
        let df = make_df(&["cat dog"]);
        let mut v = CountVectorizer::new().column("text");
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

    /// Every configuration change invalidates the learned vocabulary.
    #[test]
    fn test_setter_invalidates_fitted_state() {
        let df = make_df(&["cat dog"]);
        let mut v = CountVectorizer::new().column("text");
        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df.clone()).unwrap().width(), 2);

        let v = v.max_features(1);
        let err = v.transform(df).unwrap_err().to_string();
        assert!(err.contains("not fitted"), "{err}");
    }

    /// Every invalid configuration surfaces as `Error::InvalidInput` at fit.
    #[test]
    fn test_fit_validation_errors() {
        let df = make_df(&["cat dog"]);

        let err = CountVectorizer::new().fit(df.clone()).unwrap_err();
        assert!(err.to_string().contains("no column configured"), "{err}");

        let err = CountVectorizer::new()
            .column("missing")
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");

        let numeric = DataFrame::new(
            3,
            vec![Column::from(Series::new("text".into(), &[1i32, 2, 3]))],
        )
        .unwrap();
        let err = CountVectorizer::new()
            .column("text")
            .fit(numeric)
            .unwrap_err();
        assert!(err.to_string().contains("expected String"), "{err}");

        let no_rows: StringChunked = Vec::<Option<&str>>::new().into_iter().collect();
        let empty_rows = DataFrame::new(0, vec![Column::from(no_rows.into_series())]).unwrap();
        let err = CountVectorizer::new()
            .column("text")
            .fit(empty_rows)
            .unwrap_err();
        assert!(err.to_string().contains("empty DataFrame"), "{err}");
        let err = CountVectorizer::new()
            .column("text")
            .fit(DataFrame::empty())
            .unwrap_err();
        assert!(err.to_string().contains("empty DataFrame"), "{err}");

        for (lo, hi) in [(0usize, 3usize), (3, 2)] {
            let err = CountVectorizer::new()
                .column("text")
                .ngram_range(lo, hi)
                .fit(df.clone())
                .unwrap_err();
            assert!(
                err.to_string().contains("ngram_range"),
                "({lo}, {hi}): {err}"
            );
        }

        let err = CountVectorizer::new()
            .column("text")
            .min_df(0)
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("min_df"), "{err}");

        let err = CountVectorizer::new()
            .column("text")
            .max_features(0)
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("max_features"), "{err}");

        for p in [0.0f64, 1.5, f64::NAN] {
            let err = CountVectorizer::new()
                .column("text")
                .max_df(p)
                .fit(df.clone())
                .unwrap_err();
            assert!(err.to_string().contains("max_df"), "max_df({p}): {err}");
        }

        for pattern in ["(", "[a-"] {
            let err = CountVectorizer::new()
                .column("text")
                .tokenizer(Tokenizer::WordRegex(pattern.into()))
                .fit(df.clone())
                .unwrap_err();
            assert!(
                err.to_string().contains("invalid regex"),
                "{pattern}: {err}"
            );
        }
    }

    /// `transform` validates its own input column too.
    #[test]
    fn test_transform_validation_errors() {
        let (v, _) = fit_on(&["cat dog"], CountVectorizer::new().column("text"));

        let other =
            DataFrame::new(1, vec![Column::from(Series::new("other".into(), &["cat"]))]).unwrap();
        let err = v.transform(other).unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");

        let numeric =
            DataFrame::new(1, vec![Column::from(Series::new("text".into(), &[1i32]))]).unwrap();
        let err = v.transform(numeric).unwrap_err();
        assert!(err.to_string().contains("expected String"), "{err}");
    }
}
