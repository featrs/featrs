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
    /// Replacement text; capture references such as `$1` are supported and
    /// must resolve against [`pattern`](Self::pattern), as checked at
    /// [`fit`](Fit::fit).
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
/// Regexes are compiled and replacement templates are validated once during
/// [`fit`](Fit::fit).
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
    /// operations. The pattern and the replacement template are validated
    /// when the cleaner is fitted: a template that references a capture
    /// group the pattern does not define (for example `$1` on a pattern
    /// without capture groups) is rejected with [`Error::InvalidInput`].
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

/// A capture-group reference parsed out of a replacement template.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TemplateRef {
    /// A numeric reference such as `$1` or `${2}`.
    Number(usize),
    /// A named reference such as `$word` or `${word}`.
    Named(String),
}

/// Returns true for the ASCII characters allowed in an unbraced capture
/// reference (`$ref`), mirroring the `regex` crate's expansion syntax.
fn is_valid_cap_letter(byte: u8) -> bool {
    matches!(byte, b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_')
}

/// Parse the capture reference at the start of `text`, which must begin with
/// `$`. Returns the reference and the byte offset just past it, or `None`
/// when `text` starts with no reference — a lone `$`, a `$` followed by a
/// non-reference character, or an unclosed `${...`. The `regex` crate
/// expands those literally instead of as references.
fn parse_cap_ref(text: &[u8]) -> Option<(TemplateRef, usize)> {
    if text.len() <= 1 || text[0] != b'$' {
        return None;
    }
    if text[1] == b'{' {
        let start = 2;
        let mut end = start;
        while end < text.len() && text[end] != b'}' {
            end += 1;
        }
        if end == text.len() {
            return None;
        }
        let reference = std::str::from_utf8(&text[start..end]).ok()?;
        return Some((template_ref(reference), end + 1));
    }
    let start = 1;
    let mut end = start;
    while end < text.len() && is_valid_cap_letter(text[end]) {
        end += 1;
    }
    if end == start {
        return None;
    }
    // The scan above only accepts ASCII, so the slice is valid UTF-8.
    let reference = std::str::from_utf8(&text[start..end]).ok()?;
    Some((template_ref(reference), end))
}

/// Classify a reference body: all digits mean a group index, anything else
/// is a group name.
fn template_ref(reference: &str) -> TemplateRef {
    match reference.parse::<usize>() {
        Ok(index) => TemplateRef::Number(index),
        Err(_) => TemplateRef::Named(reference.to_owned()),
    }
}

/// Collect every capture-group reference in `template`, following the
/// expansion syntax of the `regex` crate: `$1`, `${1}`, `$name` and
/// `${name}`, with `$$` as an escape for a literal dollar sign.
fn capture_refs(template: &str) -> Vec<TemplateRef> {
    let mut refs = Vec::new();
    let mut rest = template;
    while let Some(offset) = rest.find('$') {
        rest = &rest[offset..];
        if rest.as_bytes().get(1) == Some(&b'$') {
            rest = &rest[2..];
            continue;
        }
        match parse_cap_ref(rest.as_bytes()) {
            Some((reference, end)) => {
                refs.push(reference);
                rest = &rest[end..];
            }
            None => rest = &rest[1..],
        }
    }
    refs
}

/// Check every capture-group reference in `template` against the pattern it
/// will run on, returning a description of the first unresolvable one.
///
/// The `regex` crate expands a reference to a group that does not exist into
/// an empty string, so a template like `$1` on a pattern without capture
/// groups would silently blank every matched cell during `transform`.
fn validate_replacement_template(regex: &Regex, template: &str) -> Option<String> {
    let captures_len = regex.captures_len();
    let named: Vec<&str> = regex.capture_names().flatten().collect();
    for reference in capture_refs(template) {
        match reference {
            TemplateRef::Number(index) => {
                if index >= captures_len {
                    let group_count = captures_len - 1;
                    let noun = if group_count == 1 {
                        "capture group"
                    } else {
                        "capture groups"
                    };
                    return Some(format!(
                        "it references capture group {index}, but the pattern defines \
                         {group_count} {noun}"
                    ));
                }
            }
            TemplateRef::Named(name) => {
                if !named.contains(&name.as_str()) {
                    return Some(format!(
                        "it references the capture group named '{name}', which the pattern \
                         does not define"
                    ));
                }
            }
        }
    }
    None
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
            if let Some(detail) = validate_replacement_template(&regex, &replacement.replacement) {
                return Err(Error::InvalidInput(format!(
                    "StringCleaner.fit: invalid replacement template '{}' for pattern '{}': {}. \
                     Remove this replacement or fix the template to reference an existing \
                     capture group ($0 always refers to the whole match).",
                    replacement.replacement, replacement.pattern, detail
                )));
            }
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
    fn test_invalid_replacement_template_errors_at_fit_with_actionable_message() {
        let df = frame(&[Some("abc123")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace("[a-z]", "$1");
        let error = cleaner.fit(df.clone()).unwrap_err();

        assert!(matches!(error, Error::InvalidInput(_)));
        let message = error.to_string();
        assert!(
            message.contains("$1"),
            "message names the template: {message}"
        );
        assert!(
            message.contains("[a-z]"),
            "message names the pattern: {message}"
        );
        assert!(
            message.contains("Remove this replacement"),
            "message gives actionable advice: {message}"
        );
        // The failed fit must leave the cleaner unfitted rather than silently
        // blanking matched cells during transform.
        assert!(matches!(cleaner.transform(df), Err(Error::NotFitted(_))));
    }

    #[test]
    fn test_valid_replacement_templates_fit_and_expand() {
        let df = frame(&[Some("ab-12")]);

        // Numbered references ($1, $2) against a pattern that defines them.
        let mut numbered = StringCleaner::new()
            .columns(&["text"])
            .replace(r"([a-z]+)-([0-9]+)", "$2:$1");
        numbered.fit(df.clone()).unwrap();
        assert_eq!(
            values(&numbered.transform(df.clone()).unwrap()),
            vec![Some("12:ab".into())]
        );

        // Braced numbered reference (${1}).
        let mut braced = StringCleaner::new()
            .columns(&["text"])
            .replace(r"([a-z]+)", "<${1}>");
        braced.fit(df.clone()).unwrap();
        assert_eq!(
            values(&braced.transform(df.clone()).unwrap()),
            vec![Some("<ab>-12".into())]
        );

        // `$$` escapes a literal dollar sign, so `$$1` is the text "$1" and
        // not a group reference: it must fit even with no capture groups.
        let mut escaped = StringCleaner::new()
            .columns(&["text"])
            .replace("[0-9]+", "$$1");
        escaped.fit(df.clone()).unwrap();
        assert_eq!(
            values(&escaped.transform(df).unwrap()),
            vec![Some("ab-$1".into())]
        );
    }

    #[test]
    fn test_named_capture_reference_validated_at_fit() {
        let df = frame(&[Some("ab-12")]);

        let mut valid = StringCleaner::new()
            .columns(&["text"])
            .replace(r"(?<word>[a-z]+)", "[${word}]");
        valid.fit(df.clone()).unwrap();
        assert_eq!(
            values(&valid.transform(df.clone()).unwrap()),
            vec![Some("[ab]-12".into())]
        );

        let mut missing = StringCleaner::new()
            .columns(&["text"])
            .replace(r"[a-z]+", "${missing}");
        let error = missing.fit(df).unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
        assert!(
            error.to_string().contains("${missing}"),
            "message names the template: {error}"
        );
    }

    #[test]
    fn test_transform_output_unchanged_for_valid_templates() {
        let df = frame(&[Some("abc123"), Some("2026-08"), Some("none")]);
        let mut cleaner = StringCleaner::new()
            .columns(&["text"])
            .replace(r"([a-z]+)([0-9]+)", "${2}-${1}")
            .replace(r"[0-9]+", "[$0]");
        cleaner.fit(df.clone()).unwrap();

        assert_eq!(
            values(&cleaner.transform(df).unwrap()),
            vec![
                Some("[123]-abc".into()),
                Some("[2026]-[08]".into()),
                Some("none".into())
            ]
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
