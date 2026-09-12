//! Character n-gram vectorization.
//!
//! [`CharacterNGramVectorizer`] turns a `String` column into character-level
//! n-gram counts. Unlike word-level vectorization, the analyzer never splits on
//! word boundaries (unless `Analyzer::CharWb` is selected), so the features
//! capture morphology, typo tolerance, and sub-word structure.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

/// Which characters the n-grams are extracted from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Analyzer {
    /// Extract n-grams over the entire string, spaces included.
    #[default]
    CharOnly,
    /// Extract n-grams within word boundaries: each whitespace-separated word
    /// is padded with a leading and trailing space and analyzed on its own, so
    /// n-grams never span two words.
    CharWb,
}

/// Convert a text column into character n-gram term counts.
///
/// `fit` learns a vocabulary of character n-grams from the configured column
/// (`min_df`/`max_df` filter by document frequency, `max_features` then keeps
/// the most frequent terms). `transform` counts, per document, how often each
/// learned n-gram occurs and returns a new `DataFrame` with one `Float64`
/// column per term, named after the n-gram itself — a term can be whitespace
/// only, e.g. `" "` under `ngram_range(1, 1)`. Unknown n-grams are dropped. The
/// input column is not carried over, so the output width equals the vocabulary
/// size. A vocabulary that ends up empty is returned as a 0x0 frame (polars
/// cannot represent columns-less frames with a row count), so a downstream
/// pipeline step receives an empty frame rather than one row per document.
/// `null` cells contribute no n-grams but still count as documents for the
/// `min_df`/`max_df` denominators.
///
/// Ties in the `max_features` ranking (equal corpus frequency) are broken
/// alphabetically, so the emitted column order is deterministic across runs.
/// `min_df == 0` and `max_features(0)` are rejected at fit time.
///
/// `char`-level runs of whitespace are collapsed to a single space under
/// [`Analyzer::CharOnly`], as scikit-learn's `char` analyzer does, so `"a␣␣b"`
/// and `"a␣b"` produce the same n-grams. Under [`Analyzer::CharWb`] a word whose
/// padded length — its character count plus the two boundary spaces — is below
/// the lower bound of `ngram_range` produces no n-grams at all (scikit-learn
/// instead emits the padded word once); this keeps the guarantee that every
/// generated term is exactly as long as its n, and matches the "text shorter
/// than the minimum n yields an all-zero row" contract. Note that a one-word
/// document is therefore measured with its two padding spaces.
///
/// The vocabulary — and therefore the output width — grows quickly with the
/// upper bound of `ngram_range`: the number of distinct n-grams of a document
/// grows roughly with the square of its length. N-grams are streamed into the
/// counting map rather than collected per document first, but the vocabulary
/// itself is still bounded only by the distinct substrings seen, so cap it with
/// `max_features` when using wide ranges — the output is one `Float64` column
/// per term.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::text::char_ngram_vectorizer::CharacterNGramVectorizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("text".into(), &["hello"]));
/// let df = DataFrame::new(1, vec![col])?;
///
/// let mut v = CharacterNGramVectorizer::new().column("text").ngram_range(2, 2);
/// v.fit(df.clone())?;
/// let gram_counts = v.transform(df)?;
///
/// assert_eq!(gram_counts.width(), 4);
/// let names: Vec<String> = gram_counts
///     .get_column_names()
///     .iter()
///     .map(|n| n.to_string())
///     .collect();
/// assert_eq!(names, vec!["el", "he", "ll", "lo"]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CharacterNGramVectorizer {
    fitted: bool,
    column: Option<String>,
    analyzer: Analyzer,
    ngram_range: (usize, usize),
    max_features: Option<usize>,
    min_df: usize,
    max_df: f64,
    lowercase: bool,
    vocabulary: Option<HashMap<String, u32>>,
    /// Terms in vocabulary-index order (`terms[i]` is the term whose vocabulary
    /// index is `i`), so `transform` emits columns deterministically instead of
    /// iterating the `HashMap`.
    terms: Vec<String>,
}

impl CharacterNGramVectorizer {
    /// Create a new `CharacterNGramVectorizer` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name of the `String` column to vectorize.
    pub fn column(mut self, c: &str) -> Self {
        self.column = Some(c.to_string());
        self.reset();
        self
    }

    /// Set the analyzer (default [`Analyzer::CharOnly`]).
    pub fn analyzer(mut self, a: Analyzer) -> Self {
        self.analyzer = a;
        self.reset();
        self
    }

    /// Set the inclusive n-gram length range (default `(2, 4)`).
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

    /// Lowercase the text before extracting n-grams (default `true`).
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

impl Default for CharacterNGramVectorizer {
    fn default() -> Self {
        Self {
            fitted: false,
            column: None,
            analyzer: Analyzer::CharOnly,
            ngram_range: (2, 4),
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            lowercase: true,
            vocabulary: None,
            terms: Vec::new(),
        }
    }
}

/// Collapse each run of two or more whitespace characters into a single space.
///
/// This mirrors the whitespace normalization scikit-learn's `char` analyzer
/// applies before slicing n-grams; a lone whitespace character is left as-is.
/// `Analyzer::CharWb` needs no normalization because splitting on whitespace
/// already discards the difference between one and several separators.
fn collapse_whitespace_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() && chars.peek().is_some_and(|next| next.is_whitespace()) {
            out.push(' ');
            while chars.peek().is_some_and(|next| next.is_whitespace()) {
                chars.next();
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Call `f` with every character n-gram of length `lo..=hi` found in `chars`.
///
/// N-grams are passed as slices so callers can count or look them up without
/// materializing a `String` per occurrence.
fn for_each_ngram(chars: &[char], lo: usize, hi: usize, mut f: impl FnMut(&[char])) {
    let len = chars.len();
    let hi = hi.min(len);
    // `lo..=hi` is empty when `lo > hi`, so short strings simply yield nothing.
    for n in lo..=hi {
        for start in 0..=(len - n) {
            f(&chars[start..start + n]);
        }
    }
}

/// Feed every character n-gram of length `lo..=hi` in `text` to `f`.
///
/// `Analyzer::CharOnly` walks the whole string; `Analyzer::CharWb` pads each
/// whitespace-separated word with a space on both sides and analyzes the words
/// independently, so no n-gram spans two words. Both paths are Unicode-aware:
/// n-grams are built from `char` scalar values, never from UTF-8 bytes.
fn for_each_ngram_in_text(
    text: &str,
    analyzer: Analyzer,
    lo: usize,
    hi: usize,
    mut f: impl FnMut(&[char]),
) {
    match analyzer {
        Analyzer::CharOnly => {
            let normalized = collapse_whitespace_runs(text);
            let chars: Vec<char> = normalized.chars().collect();
            for_each_ngram(&chars, lo, hi, f);
        }
        Analyzer::CharWb => {
            for word in text.split_whitespace() {
                let mut padded = Vec::with_capacity(word.chars().count() + 2);
                padded.push(' ');
                padded.extend(word.chars());
                padded.push(' ');
                for_each_ngram(&padded, lo, hi, &mut f);
            }
        }
    }
}

impl Fit<DataFrame> for CharacterNGramVectorizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset state first so a failed re-fit cannot leave stale state.
        self.reset();

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "CharacterNGramVectorizer.fit received an empty DataFrame. \
                 Provide at least one row and one column."
                    .into(),
            ));
        }

        let column = self.column.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "CharacterNGramVectorizer.fit: no column configured. \
                 Call .column(\"name\") before .fit()."
                    .into(),
            )
        })?;
        let s = x
            .column(column.as_str())
            .map_err(|e| {
                Error::InvalidInput(format!(
                    "CharacterNGramVectorizer.fit: column '{}' not found. {}",
                    column, e
                ))
            })?
            .as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "CharacterNGramVectorizer.fit: column '{}' has dtype {}; expected String. {}",
                column,
                s.dtype(),
                e
            ))
        })?;

        let (lo, hi) = self.ngram_range;
        if lo < 1 || lo > hi {
            return Err(Error::InvalidInput(format!(
                "CharacterNGramVectorizer: invalid ngram_range ({}, {}); \
                 require 1 <= lower <= upper.",
                lo, hi
            )));
        }
        if self.min_df == 0 {
            return Err(Error::InvalidInput(
                "CharacterNGramVectorizer: min_df must be >= 1.".into(),
            ));
        }
        if self.max_features == Some(0) {
            return Err(Error::InvalidInput(
                "CharacterNGramVectorizer: max_features must be >= 1.".into(),
            ));
        }
        if !(self.max_df > 0.0 && self.max_df <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "CharacterNGramVectorizer: max_df must be in (0.0, 1.0], got {}.",
                self.max_df
            )));
        }

        let n_docs = x.height() as f64;
        // One entry per distinct n-gram: (document frequency, total occurrences).
        let mut stats: HashMap<Vec<char>, (u64, u64)> = HashMap::new();
        for doc in ca.iter() {
            let Some(doc) = doc else { continue };
            let text: Cow<str> = if self.lowercase {
                Cow::Owned(doc.to_lowercase())
            } else {
                Cow::Borrowed(doc)
            };
            let mut counts: HashMap<Vec<char>, u64> = HashMap::new();
            for_each_ngram_in_text(&text, self.analyzer, lo, hi, |gram| {
                match counts.get_mut(gram) {
                    Some(count) => *count += 1,
                    None => {
                        counts.insert(gram.to_vec(), 1);
                    }
                }
            });
            for (gram, count) in counts {
                let entry = stats.entry(gram).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += count;
            }
        }

        let mut kept: Vec<(Vec<char>, u64)> = stats
            .into_iter()
            .filter(|(_, (doc_freq, _))| {
                *doc_freq >= self.min_df as u64 && *doc_freq as f64 / n_docs <= self.max_df
            })
            .map(|(gram, (_, total_freq))| (gram, total_freq))
            .collect();
        // Deterministic ranking: most frequent first, alphabetically on ties.
        kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some(max) = self.max_features {
            kept.truncate(max);
        }

        let mut vocabulary = HashMap::with_capacity(kept.len());
        for (idx, (gram, _)) in kept.into_iter().enumerate() {
            let term: String = gram.iter().collect();
            vocabulary.insert(term.clone(), idx as u32);
            self.terms.push(term);
        }
        self.vocabulary = Some(vocabulary);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CharacterNGramVectorizer {
    type Output = DataFrame;

    /// Count the fitted vocabulary's n-grams per document.
    ///
    /// Returns a frame of `Float64` counts (one column per term, one row per
    /// document). When the vocabulary is empty the result is a 0x0 frame, since
    /// polars drops the height of a frame without columns — downstream steps
    /// receive an empty frame rather than one row per document.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let not_fitted = || {
            Error::NotFitted(
                "CharacterNGramVectorizer has not been fitted. \
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
                    "CharacterNGramVectorizer.transform: column '{}' not found. {}",
                    column, e
                ))
            })?
            .as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "CharacterNGramVectorizer.transform: column '{}' has dtype {}; \
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

        let n_rows = x.height();
        let (lo, hi) = self.ngram_range;
        // Look terms up by character slice so counting allocates nothing per
        // occurrence (only once per vocabulary entry, here).
        let lookup: HashMap<Vec<char>, u32> = vocabulary
            .iter()
            .map(|(term, idx)| (term.chars().collect(), *idx))
            .collect();
        let mut counts = vec![vec![0.0f64; n_rows]; self.terms.len()];
        for (row, doc) in ca.iter().enumerate() {
            let Some(doc) = doc else { continue };
            let text: Cow<str> = if self.lowercase {
                Cow::Owned(doc.to_lowercase())
            } else {
                Cow::Borrowed(doc)
            };
            for_each_ngram_in_text(&text, self.analyzer, lo, hi, |gram| {
                if let Some(idx) = lookup.get(gram) {
                    counts[*idx as usize][row] += 1.0;
                }
            });
        }

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

    /// Sum of one output column, or `0.0` if the term is not in the vocabulary.
    fn column_sum(df: &DataFrame, name: &str) -> f64 {
        match df.column(name) {
            Ok(c) => c
                .as_materialized_series()
                .f64()
                .unwrap()
                .sum()
                .unwrap_or(0.0),
            Err(_) => 0.0,
        }
    }

    /// `"hello"` with `(2, 3)` yields exactly the 7 expected n-grams; columns
    /// are emitted in vocabulary-index order (frequency desc, then alphabetical).
    #[test]
    fn test_hello_char_only_bigrams_and_trigrams() {
        let df = make_df(&["hello"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 3);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.width(), 7);
        assert_eq!(out.height(), 1);
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["el", "ell", "he", "hel", "ll", "llo", "lo"]);
    }

    /// `Analyzer::CharWb` analyzes each word on its own, padded with boundary
    /// spaces, so no n-gram spans two words; `CharOnly` on the same input does
    /// produce the spanning n-gram.
    #[test]
    fn test_char_wb_respects_word_boundaries() {
        let df = make_df(&["hello world"]);
        let mut wb = CharacterNGramVectorizer::new()
            .column("text")
            .analyzer(Analyzer::CharWb)
            .ngram_range(2, 3);
        wb.fit(df.clone()).unwrap();
        let out = wb.transform(df.clone()).unwrap();

        // 7 chars in " hello " and " world " each: 6 + 6 bigrams, 5 + 5 trigrams.
        assert_eq!(out.width(), 22);
        assert_eq!(column_sum(&out, " h"), 1.0);
        assert_eq!(column_sum(&out, "o "), 1.0);
        assert_eq!(column_sum(&out, " w"), 1.0);
        assert_eq!(column_sum(&out, "d "), 1.0);
        // The word-spanning n-gram only exists under CharOnly.
        assert!(out.column("o w").is_err(), "CharWb must not span words");

        let mut co = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 3);
        co.fit(df.clone()).unwrap();
        let co_out = co.transform(df).unwrap();
        assert_eq!(column_sum(&co_out, "o w"), 1.0);
    }

    /// `min_df`/`max_df` filter by document frequency; `max_features` keeps the
    /// most frequent terms; and the column order is deterministic, breaking
    /// frequency ties alphabetically.
    #[test]
    fn test_df_filters_max_features_and_deterministic_order() {
        // "ab" appears in 2 of 3 documents, "ac" in 1.
        let df = make_df(&["ab", "ab", "ac"]);

        let mut min_df = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2)
            .min_df(2);
        min_df.fit(df.clone()).unwrap();
        let out = min_df.transform(df.clone()).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "ab");

        let mut max_df = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2)
            .max_df(0.5);
        max_df.fit(df.clone()).unwrap();
        let out = max_df.transform(df.clone()).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "ac");

        // "ab" occurs twice, "ac" once, so the single kept term is the more frequent one.
        let mut capped = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2)
            .max_features(1);
        capped.fit(df.clone()).unwrap();
        let out = capped.transform(df).unwrap();
        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "ab");

        // Equal document frequencies: the tie is broken alphabetically, and two
        // independent fits emit the same column order.
        let df = make_df(&["hello", "world"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df.clone()).unwrap();
        let first = v.transform(df.clone()).unwrap();
        let names: Vec<&str> = first
            .get_column_names()
            .iter()
            .map(|n| n.as_str())
            .collect();
        assert_eq!(names, vec!["el", "he", "ld", "ll", "lo", "or", "rl", "wo"]);

        let mut again = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        again.fit(df.clone()).unwrap();
        let second = again.transform(df).unwrap();
        assert_eq!(
            second.get_column_names(),
            first.get_column_names(),
            "column order must be deterministic across fits"
        );
    }

    /// n-grams are built from Unicode scalar values, so a multi-byte `é` or
    /// emoji stays a single character for n-gram purposes.
    #[test]
    fn test_unicode_char_ngrams() {
        let df = make_df(&["héllo"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 3);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.width(), 7);
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        // `é` sorts after ASCII, so the `é…` terms come last.
        assert_eq!(names, vec!["hé", "hél", "ll", "llo", "lo", "él", "éll"]);
        assert_eq!(column_sum(&out, "hé"), 1.0);
        assert_eq!(column_sum(&out, "hél"), 1.0);

        // A 4-byte emoji is still one character.
        let df = make_df(&["a😀b"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(1, 1);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 3);
        assert_eq!(column_sum(&out, "😀"), 1.0);
    }

    /// Overlapping repeated n-grams are counted once per occurrence.
    #[test]
    fn test_repeated_characters_counted() {
        let df = make_df(&["aaa"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.width(), 1);
        assert_eq!(out.get_column_names()[0].as_str(), "aa");
        assert_eq!(column_sum(&out, "aa"), 2.0);
    }

    /// Empty and shorter-than-`lo` documents produce all-zero rows.
    #[test]
    fn test_empty_and_short_documents_are_all_zero() {
        let df = make_df(&["", "a", "hello"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 3);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.width(), 7);
        assert_eq!(out.height(), 3);
        for name in out.get_column_names() {
            let ca = out
                .column(name.as_str())
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap();
            assert_eq!(ca.get(0), Some(0.0), "empty document row for {name}");
            assert_eq!(ca.get(1), Some(0.0), "one-char document row for {name}");
        }
        assert_eq!(column_sum(&out, "ll"), 1.0);
    }

    /// Null cells count as empty documents.
    #[test]
    fn test_null_documents_are_empty() {
        let ca: StringChunked = [Some("aa"), None, Some("aa")].into_iter().collect();
        let s = ca.into_series().with_name("text".into());
        let df = DataFrame::new(3, vec![Column::from(s)]).unwrap();
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 3);
        let ca = out
            .column("aa")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();
        assert_eq!(ca.get(0), Some(1.0));
        assert_eq!(ca.get(1), Some(0.0));
        assert_eq!(ca.get(2), Some(1.0));
    }

    /// Counts are per document, and the row count follows the input frame.
    #[test]
    fn test_counts_per_document_and_row_count() {
        let df = make_df(&["aa", "aa aa", ""]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.height(), 3);
        let ca = out
            .column("aa")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap();
        assert_eq!(ca.get(0), Some(1.0));
        assert_eq!(ca.get(1), Some(2.0));
        assert_eq!(ca.get(2), Some(0.0));
    }

    /// Unigrams are single characters, ranked by corpus frequency.
    #[test]
    fn test_unigram_range_ranks_by_frequency() {
        let df = make_df(&["abc", "cc", "", "ab"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(1, 1);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        assert_eq!(out.height(), 4);
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        // "c" occurs 3 times, "a" and "b" twice each (alphabetical tie-break).
        assert_eq!(names, vec!["c", "a", "b"]);
        assert_eq!(column_sum(&out, "c"), 3.0);
    }

    /// n-grams absent from the vocabulary at fit time are dropped.
    #[test]
    fn test_unknown_ngrams_dropped() {
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(make_df(&["hello"])).unwrap();
        let out = v.transform(make_df(&["xyz"])).unwrap();

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 1);
        for name in out.get_column_names() {
            assert_eq!(column_sum(&out, name.as_str()), 0.0, "term {name}");
        }
    }

    /// `lowercase(false)` keeps the text as-is, so the vocabulary is
    /// case-sensitive; the default folds case and merges the terms.
    #[test]
    fn test_lowercase_option() {
        let df = make_df(&["Hello", "hello"]);

        let mut folded = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        folded.fit(df.clone()).unwrap();
        let out = folded.transform(df.clone()).unwrap();
        assert_eq!(out.width(), 4);
        assert_eq!(column_sum(&out, "ll"), 2.0);

        let mut kept = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2)
            .lowercase(false);
        kept.fit(df.clone()).unwrap();
        let out = kept.transform(df).unwrap();
        assert_eq!(out.width(), 5);
        assert_eq!(column_sum(&out, "He"), 1.0);
        assert_eq!(column_sum(&out, "he"), 1.0);
        assert_eq!(column_sum(&out, "el"), 2.0);
    }

    /// `transform` before `fit` reports `NotFitted`, and a failed re-fit clears
    /// the state learned by an earlier successful fit.
    #[test]
    fn test_not_fitted_and_failed_refit_resets_state() {
        let df = make_df(&["hello"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        let err = v.transform(df.clone()).unwrap_err().to_string();
        assert!(err.contains("not fitted"), "{err}");

        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df.clone()).unwrap().width(), 4);

        v.column = Some("missing".into());
        assert!(v.fit(df.clone()).is_err());
        let err = v.transform(df).unwrap_err().to_string();
        assert!(
            err.contains("not fitted"),
            "stale state after failed refit: {err}"
        );
    }

    /// An empty vocabulary (no document yields an n-gram, or every term is
    /// filtered out) is returned as a 0x0 frame.
    #[test]
    fn test_empty_vocabulary_returns_empty_frame() {
        let df = make_df(&["", "", "a"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(3, 3);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);

        // The column is still validated even when the vocabulary is empty.
        let other =
            DataFrame::new(1, vec![Column::from(Series::new("other".into(), &[""]))]).unwrap();
        assert!(
            v.transform(other)
                .unwrap_err()
                .to_string()
                .contains("not found")
        );

        // `max_df` can also filter the vocabulary down to nothing.
        let df = make_df(&["ab", "ab"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2)
            .max_df(0.5);
        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df).unwrap().width(), 0);
    }

    /// Every invalid configuration surfaces as `Error::InvalidInput` at fit.
    #[test]
    fn test_fit_validation_errors() {
        let df = make_df(&["hello"]);

        let err = CharacterNGramVectorizer::new().fit(df.clone()).unwrap_err();
        assert!(err.to_string().contains("no column configured"), "{err}");

        let err = CharacterNGramVectorizer::new()
            .column("missing")
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");

        let numeric = DataFrame::new(
            3,
            vec![Column::from(Series::new("text".into(), &[1i32, 2, 3]))],
        )
        .unwrap();
        let err = CharacterNGramVectorizer::new()
            .column("text")
            .fit(numeric)
            .unwrap_err();
        assert!(err.to_string().contains("expected String"), "{err}");

        let no_rows: StringChunked = Vec::<Option<&str>>::new().into_iter().collect();
        let empty_rows = DataFrame::new(0, vec![Column::from(no_rows.into_series())]).unwrap();
        let err = CharacterNGramVectorizer::new()
            .column("text")
            .fit(empty_rows)
            .unwrap_err();
        assert!(err.to_string().contains("empty DataFrame"), "{err}");
        let err = CharacterNGramVectorizer::new()
            .column("text")
            .fit(DataFrame::empty())
            .unwrap_err();
        assert!(err.to_string().contains("empty DataFrame"), "{err}");

        for (lo, hi) in [(0usize, 3usize), (3, 2)] {
            let err = CharacterNGramVectorizer::new()
                .column("text")
                .ngram_range(lo, hi)
                .fit(df.clone())
                .unwrap_err();
            assert!(
                err.to_string().contains("ngram_range"),
                "({lo}, {hi}): {err}"
            );
        }

        let err = CharacterNGramVectorizer::new()
            .column("text")
            .min_df(0)
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("min_df"), "{err}");

        let err = CharacterNGramVectorizer::new()
            .column("text")
            .max_features(0)
            .fit(df.clone())
            .unwrap_err();
        assert!(err.to_string().contains("max_features"), "{err}");

        for p in [0.0f64, 1.5, f64::NAN] {
            let err = CharacterNGramVectorizer::new()
                .column("text")
                .max_df(p)
                .fit(df.clone())
                .unwrap_err();
            assert!(err.to_string().contains("max_df"), "max_df({p}): {err}");
        }
    }

    /// `transform` validates its own input column too.
    #[test]
    fn test_transform_validation_errors() {
        let df = make_df(&["hello"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df).unwrap();

        let err = v
            .transform(
                DataFrame::new(
                    1,
                    vec![Column::from(Series::new("other".into(), &["hello"]))],
                )
                .unwrap(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");

        let numeric =
            DataFrame::new(1, vec![Column::from(Series::new("text".into(), &[1i32]))]).unwrap();
        let err = v.transform(numeric).unwrap_err();
        assert!(err.to_string().contains("expected String"), "{err}");
    }

    /// `CharOnly` collapses runs of whitespace into one space, so repeated
    /// separators do not add terms; a lone whitespace character is kept as-is.
    #[test]
    fn test_char_only_collapses_whitespace_runs() {
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 3);
        v.fit(make_df(&["a  b"])).unwrap();
        let out = v.transform(make_df(&["a  b"])).unwrap();
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec![" b", "a ", "a b"]);
        assert_eq!(column_sum(&out, "a "), 1.0);

        // A run of any whitespace characters collapses the same way.
        let out = v.transform(make_df(&["a\t\nb"])).unwrap();
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec![" b", "a ", "a b"]);

        // A single whitespace character is not a run and is preserved.
        let mut single = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        single.fit(make_df(&["a\tb"])).unwrap();
        let out = single.transform(make_df(&["a\tb"])).unwrap();
        assert_eq!(out.width(), 2);
        assert_eq!(column_sum(&out, "a\t"), 1.0);
    }

    /// Under `CharWb` a word shorter than the lower n-gram bound contributes no
    /// terms, so every emitted term has a length inside `ngram_range`.
    #[test]
    fn test_char_wb_short_word_yields_no_terms() {
        let df = make_df(&["i am here"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .analyzer(Analyzer::CharWb)
            .ngram_range(5, 5);
        v.fit(df.clone()).unwrap();
        let out = v.transform(df).unwrap();

        // " i " (3 chars) and " am " (4 chars) are too short for a 5-gram; only
        // " here " (6 chars) is long enough: " here" and "here ".
        let names: Vec<&str> = out.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec![" here", "here "]);
    }

    /// Changing any configuration invalidates the learned vocabulary.
    #[test]
    fn test_setter_invalidates_fitted_state() {
        let df = make_df(&["hello"]);
        let mut v = CharacterNGramVectorizer::new()
            .column("text")
            .ngram_range(2, 2);
        v.fit(df.clone()).unwrap();
        assert_eq!(v.transform(df.clone()).unwrap().width(), 4);

        let v = v.ngram_range(2, 3);
        let err = v.transform(df).unwrap_err().to_string();
        assert!(err.contains("not fitted"), "{err}");
    }
}
