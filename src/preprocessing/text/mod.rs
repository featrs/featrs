//! Text feature extraction.
//!
//! Transformers in this module convert `String` columns into numeric feature
//! matrices. They follow the crate's usual contract: `fit` learns a vocabulary
//! from a text column of the training frame, and `transform` emits a new
//! `DataFrame` holding one column per learned term (the input text column is
//! not carried over). Terms that were not seen during `fit` are dropped.
//!
//! [`char_ngram_vectorizer::CharacterNGramVectorizer`] extracts overlapping
//! character n-grams, which capture sub-word structure and are robust to typos.

pub mod char_ngram_vectorizer;
