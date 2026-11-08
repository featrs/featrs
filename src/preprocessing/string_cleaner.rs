//! Configurable cleaning for string columns.
//!
//! [`StringCleaner`] applies a deterministic sequence of operations to selected
//! `String` columns: trim leading and trailing whitespace, collapse internal
//! whitespace, normalize case, then apply regular-expression replacements in
//! declaration order. Non-selected columns and null values are preserved.

use polars::prelude::*;
use regex::Regex;

use crate::traits::{Error, Fit, Result, Transform};

/// Case normalization applied to each non-null string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CaseStyle {
    /// Preserve the original casing.
    #[default]
    Original,
    /// Convert all characters to lowercase using Unicode case conversion.
    Lower,
    /// Convert all characters to uppercase using Unicode case conversion.
    Upper,
    /// Capitalize each whitespace-separated word and join words with one space.
    Title,
    /// Capitalize each whitespace-separated word and join the words directly.
    ///
    /// This produces PascalCase (upper camel case): for example,
    /// `"hello world"` becomes `"HelloWorld"`.
    Camel,
}

/// A regular-expression replacement applied by [`StringCleaner`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringReplacement {
    /// Rust `regex` pattern to match.
    pub pattern: String,
    /// Replacement text; capture references such as `$1` are supported.
    pub replacement: String,
}

#[derive(Clone, Debug)]
struct CompiledReplacement {
    regex: Regex,
    replacement: String,
}

/// Normalize values in selected `String` columns.
///
/// Operations run in this order: trim, collapse internal whitespace, case
/// normalization, then configured regex replacements in declaration order.
/// Regexes are validated and compiled once during [`fit`](Fit::fit).
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::string_cleaner::{CaseStyle, StringCleaner};
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let text = Column::from(Series::new("text".into(), &["  Hello   WORLD  "]));
/// let df = DataFrame::new(1, vec![text])?;
/// let mut cleaner = StringCleaner::new()
///     .columns(&["text"])
///     .trim()
///     .collapse_internal_ws()
///     .case_style(CaseStyle::Lower);
/// cleaner.fit(df.clone())?;
/// let cleaned = cleaner.transform(df)?;
/// assert_eq!(cleaned.column("text")?.str()?.get(0), Some("hello world"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StringCleaner {
    fitted: bool,
    columns: Vec<String>,
    trim: bool,
    collapse_internal_ws: bool,
    case_style: CaseStyle,
    replacements: Vec<StringReplacement>,
    collapse_ws_regex: Option<Regex>,
    compiled_replacements: Vec<CompiledReplacement>,
}

impl StringCleaner {
    /// Create a no-op cleaner with no columns selected.
    ///
    /// Select at least one column with [`columns`](Self::columns) before
    /// fitting. Cleaning operations are opt-in.
    pub fn new() -> Self {
        Self {
            fitted: false,
            columns: Vec::new(),
            trim: false,
            collapse_internal_ws: false,
            case_style: CaseStyle::Original,
            replacements: Vec::new(),
            collapse_ws_regex: None,
            compiled_replacements: Vec::new(),
        }
    }

    /// Select the `String` columns to clean.
    ///
    /// Duplicate names are ignored after their first occurrence so each
    /// selected column is cleaned exactly once.
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.invalidate_fit();
        self.columns.clear();
        for column in cols {
            if !self.columns.iter().any(|selected| selected == column) {
                self.columns.push((*column).to_owned());
            }
        }
        self
    }

    /// Trim leading and trailing Unicode whitespace.
    pub fn trim(mut self) -> Self {
        self.invalidate_fit();
        self.trim = true;
        self
    }

    /// Collapse each run of internal Unicode whitespace to one ASCII space.
    pub fn collapse_internal_ws(mut self) -> Self {
        self.invalidate_fit();
        self.collapse_internal_ws = true;
        self
    }

    /// Set the case normalization style.
    pub fn case_style(mut self, case_style: CaseStyle) -> Self {
        self.invalidate_fit();
        self.case_style = case_style;
        self
    }

    /// Append a regex replacement.
    ///
    /// Replacements run in declaration order after whitespace and case
    /// operations. The pattern and its capture references are validated when
    /// the cleaner is fitted.
    pub fn replace(mut self, pattern: &str, replacement: &str) -> Self {
        self.invalidate_fit();
        self.replacements.push(StringReplacement {
            pattern: pattern.to_owned(),
            replacement: replacement.to_owned(),
        });
        self
    }

    fn invalidate_fit(&mut self) {
        self.fitted = false;
        self.collapse_ws_regex = None;
        self.compiled_replacements.clear();
    }

    fn clean_value(&self, value: &str) -> String {
        let mut cleaned = if self.trim {
            value.trim().to_owned()
        } else {
            value.to_owned()
        };

        if let Some(regex) = &self.collapse_ws_regex {
            cleaned = regex.replace_all(&cleaned, " ").into_owned();
        }

        cleaned = match self.case_style {
            CaseStyle::Original => cleaned,
            CaseStyle::Lower => cleaned.to_lowercase(),
            CaseStyle::Upper => cleaned.to_uppercase(),
            CaseStyle::Title => title_case(&cleaned, " "),
            CaseStyle::Camel => title_case(&cleaned, ""),
        };

        for replacement in &self.compiled_replacements {
            cleaned = replacement
                .regex
                .replace_all(&cleaned, replacement.replacement.as_str())
                .into_owned();
        }

        cleaned
    }
}

impl Default for StringCleaner {
    fn default() -> Self {
        Self::new()
    }
}

fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };

    first
        .to_uppercase()
        .chain(chars.flat_map(char::to_lowercase))
        .collect()
}

fn title_case(value: &str, separator: &str) -> String {
    value
        .split_whitespace()
        .map(capitalize_word)
        .collect::<Vec<_>>()
        .join(separator)
}

/// Reject replacement templates whose capture references cannot be expanded.
///
/// The `regex` crate expands an unknown reference to the empty string, which
/// silently blanks every matched value. Fit is the documented validation
/// point, so each `$index`, `$name` and `${...}` reference is checked against
/// the compiled pattern before it can be used by `transform`.
fn validate_replacement(regex: &Regex, pattern: &str, replacement: &str) -> Result<()> {
    let bytes = replacement.as_bytes();
    let mut index = 0;
    // Once a `${` has no closing brace, no later `${` can have one either.
    // Remember the failed search so a replacement full of unclosed braces
    // stays linear instead of rescanning the remaining template each time.
    let mut unclosed_brace = false;
    while index < bytes.len() {
        if bytes[index] != b'$' {
            index += 1;
            continue;
        }
        if bytes.get(index + 1) == Some(&b'$') {
            index += 2;
            continue;
        }

        let reference_start = index + 1;
        let (reference, next) = if bytes.get(reference_start) == Some(&b'{') {
            if unclosed_brace {
                index += 1;
                continue;
            }
            match replacement[reference_start + 1..].find('}') {
                Some(offset) => (
                    &replacement[reference_start + 1..reference_start + 1 + offset],
                    reference_start + 2 + offset,
                ),
                // An unclosed brace is not a capture reference; the regex
                // crate copies the dollar sign literally.
                None => {
                    unclosed_brace = true;
                    index += 1;
                    continue;
                }
            }
        } else {
            let end = replacement[reference_start..]
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .map(|offset| reference_start + offset)
                .unwrap_or(replacement.len());
            if end == reference_start {
                index = reference_start;
                continue;
            }
            (&replacement[reference_start..end], end)
        };
        index = next;

        // The `regex` crate parses a reference as a group index first and
        // falls back to a capture name, so numeric forms that its integer
        // parser accepts (such as `${+1}`) are valid when the index exists.
        let resolves = match reference.parse::<usize>() {
            Ok(number) => number < regex.captures_len(),
            Err(_) => regex
                .capture_names()
                .flatten()
                .any(|name| name == reference),
        };
        if !resolves {
            return Err(Error::InvalidInput(format!(
                "StringCleaner.fit: replacement '{replacement}' for pattern '{pattern}' \
                 references capture '{reference}', which the pattern does not define. \
                 Use a group index below {}, a name declared with (?P<name>...), or escape \
                 a literal dollar sign with '$$'.",
                regex.captures_len()
            )));
        }
    }
    Ok(())
}

impl Fit<DataFrame> for StringCleaner {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        self.invalidate_fit();

        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "StringCleaner.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "StringCleaner.fit has no selected columns. \
                 Select at least one String column with .columns(&[...])."
                    .into(),
            ));
        }

        for name in &self.columns {
            let column = x.column(name.as_str()).map_err(|error| {
                Error::InvalidInput(format!(
                    "StringCleaner.fit: column '{name}' was not found. {error}"
                ))
            })?;
            if column.dtype() != &DataType::String {
                return Err(Error::InvalidInput(format!(
                    "StringCleaner.fit: column '{name}' has dtype {}; expected String.",
                    column.dtype()
                )));
            }
        }

        if self.collapse_internal_ws {
            self.collapse_ws_regex = Some(Regex::new(r"\s+").map_err(|error| {
                Error::Computation(format!(
                    "StringCleaner.fit could not compile its whitespace regex. {error}"
                ))
            })?);
        }

        let mut compiled = Vec::with_capacity(self.replacements.len());
        for replacement in &self.replacements {
            let regex = Regex::new(&replacement.pattern).map_err(|error| {
                Error::InvalidInput(format!(
                    "StringCleaner.fit: invalid regex pattern '{}': {}. \
                     Remove this pattern or replace it with a valid Rust regex.",
                    replacement.pattern, error
                ))
            })?;
            validate_replacement(&regex, &replacement.pattern, &replacement.replacement)?;
            compiled.push(CompiledReplacement {
                regex,
                replacement: replacement.replacement.clone(),
            });
        }

        self.compiled_replacements = compiled;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for StringCleaner {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "StringCleaner has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut output = x;
        for name in &self.columns {
            let column = output.column(name.as_str()).map_err(|error| {
                Error::InvalidInput(format!(
                    "StringCleaner.transform: fitted column '{name}' was not found. {error}"
                ))
            })?;
            let strings = column.as_materialized_series().str().map_err(|error| {
                Error::InvalidInput(format!(
                    "StringCleaner.transform: column '{name}' has dtype {}; expected String. {}",
                    column.dtype(),
                    error
                ))
            })?;

            let mut cleaned: StringChunked = strings
                .iter()
                .map(|value| value.map(|value| self.clean_value(value)))
                .collect();
            cleaned.rename(name.as_str().into());
            output
                .with_column(Column::from(cleaned.into_series()))
                .map_err(|error| {
                    Error::Computation(format!(
                        "StringCleaner.transform: could not replace column '{name}'. {error}"
                    ))
                })?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(values: &[Option<&str>]) -> DataFrame {
        let text = Column::from(Series::new("text".into(), values));
        let numbers = vec![0_i64; values.len()];
        let number = Column::from(Series::new("number".into(), numbers));
        DataFrame::new(values.len(), vec![text, number]).unwrap()
    }

    fn values(frame: &DataFrame) -> Vec<Option<String>> {
        frame
            .column("text")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|value| value.map(str::to_owned))
            .collect()
    }

    #[test]
    fn test_trim() {
        let df = frame(&[Some(" hello "), Some("\tworld\n"), Some("")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).trim();
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("hello".into()), Some("world".into()), Some("".into())]
        );
    }

    #[test]
    fn test_collapse_internal_whitespace() {
        let df = frame(&[Some("a  b   c"), Some("a\tb\nc"), Some("")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .collapse_internal_ws();
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("a b c".into()), Some("a b c".into()), Some("".into())]
        );
    }

    #[test]
    fn test_lower_and_upper_case_are_unicode_aware() {
        let df = frame(&[Some("Äpfel"), Some("Straße"), Some("ÉLAN")]);
        let mut lower = StringCleaner::new()
            .columns(&["text"])
            .case_style(CaseStyle::Lower);
        lower.fit(df.clone()).unwrap();
        assert_eq!(
            values(&lower.transform(df.clone()).unwrap()),
            vec![
                Some("äpfel".into()),
                Some("straße".into()),
                Some("élan".into())
            ]
        );

        let mut upper = StringCleaner::new()
            .columns(&["text"])
            .case_style(CaseStyle::Upper);
        upper.fit(df.clone()).unwrap();
        assert_eq!(
            values(&upper.transform(df).unwrap()),
            vec![
                Some("ÄPFEL".into()),
                Some("STRASSE".into()),
                Some("ÉLAN".into())
            ]
        );
    }

    #[test]
    fn test_title_case() {
        let df = frame(&[
            Some("hello world"),
            Some("mIXED   cASE"),
            Some("élan VITAL"),
        ]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .case_style(CaseStyle::Title);
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![
                Some("Hello World".into()),
                Some("Mixed Case".into()),
                Some("Élan Vital".into())
            ]
        );
    }

    #[test]
    fn test_camel_case() {
        let df = frame(&[Some("hello world"), Some("mIXED case"), Some("")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .case_style(CaseStyle::Camel);
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![
                Some("HelloWorld".into()),
                Some("MixedCase".into()),
                Some("".into())
            ]
        );
    }

    #[test]
    fn test_regex_replacement_and_capture_groups() {
        let df = frame(&[Some("abc123"), Some("2026-08"), Some("none")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace(r"[0-9]+", "#")
            .replace(r"([a-z]+)(#)", "${2}-${1}");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![
                Some("#-abc".into()),
                Some("#-#".into()),
                Some("none".into())
            ]
        );
    }

    #[test]
    fn test_multiple_replacements_run_in_order() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("a", "b")
            .replace("b", "c");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("c".into()), Some("c".into()), Some("c".into())]
        );
    }

    #[test]
    fn test_operations_run_in_documented_order() {
        let df = frame(&[Some("  HELLO   WORLD  "), Some(" X "), Some("")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .trim()
            .collapse_internal_ws()
            .case_style(CaseStyle::Lower)
            .replace("hello world", "done");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("done".into()), Some("x".into()), Some("".into())]
        );
    }

    #[test]
    fn test_null_and_empty_strings_are_preserved() {
        let df = frame(&[Some(" "), None, Some("")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).trim();
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("".into()), None, Some("".into())]
        );
    }

    #[test]
    fn test_noop_preserves_schema_and_values() {
        let df = frame(&[Some(" A "), None, Some("b")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]);
        cleaner.fit(df.clone()).unwrap();
        let result = cleaner.transform(df.clone()).unwrap();

        assert!(result.equals_missing(&df));
        assert_eq!(result.get_column_names(), df.get_column_names());
    }

    #[test]
    fn test_only_selected_columns_are_cleaned() {
        let first = Column::from(Series::new("first".into(), &[" a ", " b "]));
        let second = Column::from(Series::new("second".into(), &[" x ", " y "]));
        let df = DataFrame::new(2, vec![first, second]).unwrap();
        let mut cleaner = StringCleaner::new().columns(&["first"]).trim();
        cleaner.fit(df.clone()).unwrap();
        let result = cleaner.transform(df).unwrap();

        assert_eq!(
            result.column("first").unwrap().str().unwrap().get(0),
            Some("a")
        );
        assert_eq!(
            result.column("second").unwrap().str().unwrap().get(0),
            Some(" x ")
        );
        assert_eq!(result.get_column_names()[0].as_str(), "first");
        assert_eq!(result.get_column_names()[1].as_str(), "second");
    }

    #[test]
    fn test_duplicate_column_names_are_cleaned_once() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text", "text"])
            .replace("a", "aa");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("aa".into()), Some("b".into()), Some("c".into())]
        );
    }

    #[test]
    fn test_invalid_regex_errors_at_fit_with_actionable_message() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).replace("(", "x");
        let error = cleaner.fit(df).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(error.to_string().contains("Remove this pattern"));
    }

    #[test]
    fn test_undefined_numeric_capture_errors_at_fit() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("[a-z]", "$1");
        let error = cleaner.fit(df).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        let message = error.to_string();
        assert!(message.contains("references capture '1'"));
        assert!(message.contains("(?P<name>..."));
    }

    #[test]
    fn test_undefined_named_capture_errors_at_fit() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("(?P<word>[a-z]+)", "$letter");
        let error = cleaner.fit(df).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(error.to_string().contains("references capture 'letter'"));
    }

    #[test]
    fn test_braced_undefined_capture_errors_at_fit() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("([a-z]+)", "${2}");
        let error = cleaner.fit(df).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(error.to_string().contains("references capture '2'"));
    }

    #[test]
    fn test_named_capture_reference_is_expanded() {
        let df = frame(&[Some("Hello"), Some("world")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("(?P<word>[A-Za-z]+)", "<$word>");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("<Hello>".into()), Some("<world>".into())]
        );
    }

    #[test]
    fn test_unbraced_whole_match_reference_is_expanded() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("(abc)", "[$0]");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("[abc]".into())]
        );
    }

    #[test]
    fn test_unbraced_numeric_capture_reference_is_expanded() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("(abc)", "[$1]");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("[abc]".into())]
        );
    }

    #[test]
    fn test_braced_signed_numeric_capture_reference_is_expanded() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("(abc)", "[${+1}]");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("[abc]".into())]
        );
    }

    #[test]
    fn test_unclosed_brace_replacement_is_literal() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("abc", "${x${y");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("${x${y".into())]
        );
    }

    #[test]
    fn test_unclosed_brace_does_not_hide_later_unbraced_reference() {
        let df = frame(&[Some("abc")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("[a-z]", "${oops $1");
        let error = cleaner.fit(df).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(error.to_string().contains("references capture '1'"));
    }

    #[test]
    fn test_escaped_dollar_is_not_a_capture_reference() {
        let df = frame(&[Some("5"), Some("x")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).replace("x", "$$1");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![Some("5".into()), Some("$1".into())]
        );
    }

    #[test]
    fn test_not_fitted_errors() {
        let cleaner = StringCleaner::new().columns(&["text"]);
        let error = cleaner
            .transform(frame(&[Some("a"), Some("b"), Some("c")]))
            .unwrap_err();
        assert!(matches!(error, Error::NotFitted(_)));
    }

    #[test]
    fn test_empty_input_errors() {
        let mut cleaner = StringCleaner::new().columns(&["text"]);
        let empty = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        assert!(matches!(cleaner.fit(empty), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_empty_columns_errors() {
        let mut cleaner = StringCleaner::new();
        let error = cleaner
            .fit(frame(&[Some("a"), Some("b"), Some("c")]))
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    #[test]
    fn test_missing_and_non_string_fit_columns_error() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut missing = StringCleaner::new().columns(&["missing"]);
        assert!(matches!(
            missing.fit(df.clone()),
            Err(Error::InvalidInput(_))
        ));

        let mut numeric = StringCleaner::new().columns(&["number"]);
        assert!(matches!(numeric.fit(df), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_transform_schema_drift_errors() {
        let train = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).trim();
        cleaner.fit(train).unwrap();

        let missing = DataFrame::new(
            3,
            vec![Column::from(Series::new("number".into(), &[1_i64, 2, 3]))],
        )
        .unwrap();
        assert!(matches!(
            cleaner.transform(missing),
            Err(Error::InvalidInput(_))
        ));

        let wrong_type = DataFrame::new(
            3,
            vec![Column::from(Series::new("text".into(), &[1_i64, 2, 3]))],
        )
        .unwrap();
        assert!(matches!(
            cleaner.transform(wrong_type),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_failed_refit_clears_fitted_state() {
        let df = frame(&[Some("a"), Some("b"), Some("c")]);
        let mut cleaner = StringCleaner::new().columns(&["text"]).trim();
        cleaner.fit(df.clone()).unwrap();

        let invalid = DataFrame::new(
            3,
            vec![Column::from(Series::new("text".into(), &[1_i64, 2, 3]))],
        )
        .unwrap();
        assert!(matches!(cleaner.fit(invalid), Err(Error::InvalidInput(_))));
        assert!(matches!(cleaner.transform(df), Err(Error::NotFitted(_))));
    }
}
