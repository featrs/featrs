//! Shared helpers for transformers operating on `Float64` columns.
//!
//! These utilities exist to keep the per-transformer code free of duplicated
//! column-discovery, validation, and in-place replacement boilerplate.

use polars::prelude::{ChunkedArray, DataFrame, DataType, Float64Type, IntoSeries, Series};

use crate::traits::{Error, Result};

/// Return the names of all `Float64` columns in `df`, in frame order.
///
/// Non-`Float64` columns are silently skipped. Transformers that need to
/// surface this as an error should use [`require_f64_columns`] instead.
pub fn numeric_f64_columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter_map(|name| {
            df.column(name)
                .ok()
                .filter(|s| s.dtype() == &DataType::Float64)
                .map(|_| name.to_string())
        })
        .collect()
}

/// Return the names of all `Float64` columns, or an `InvalidInput` error that
/// lists every column and its dtype when none are `Float64`.
///
/// `who` is the transformer name used in the error message
/// (e.g. `"StandardScaler"`), so failures are easy to trace back to the
/// transformer that produced them.
pub fn require_f64_columns(df: &DataFrame, who: &str) -> Result<Vec<String>> {
    let cols = numeric_f64_columns(df);
    if cols.is_empty() {
        let all_types: Vec<String> = df
            .get_column_names()
            .iter()
            .filter_map(|n| df.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
            .collect();
        return Err(Error::InvalidInput(format!(
            "{who}: no Float64 columns found. This transformer only operates on f64 columns. \
             Available columns: [{}]. Cast non-f64 columns before fitting.",
            all_types.join(", ")
        )));
    }
    Ok(cols)
}

/// Apply a per-element f64 transform to a single named column of `df`,
/// replacing the column in place.
///
/// `f` maps each non-null `f64` value to its replacement; nulls are preserved.
/// `who` names the calling transformer for error context.
///
/// This collapses the `column(name).unwrap()` → `f64().unwrap()` →
/// `replace(...).unwrap()` boilerplate that would otherwise be duplicated in
/// every f64-based transformer's `transform` implementation.
pub fn replace_f64_column<F>(df: &mut DataFrame, name: &str, who: &str, f: F) -> Result<()>
where
    F: Fn(f64) -> f64,
{
    replace_f64_column_opt(df, name, who, move |v| Some(f(v)))
}

/// Apply a per-element optional f64 transform to a single named column of `df`,
/// replacing the column in place.
///
/// Unlike [`replace_f64_column`], `f` may return `None` to emit a null,
/// letting callers map sentinel values such as `NaN` to missing data instead
/// of a computed value. Input nulls never reach `f` and stay null.
/// `who` names the calling transformer for error context.
pub fn replace_f64_column_opt<F>(df: &mut DataFrame, name: &str, who: &str, f: F) -> Result<()>
where
    F: Fn(f64) -> Option<f64>,
{
    let s = df.column(name).map_err(|e| {
        Error::InvalidInput(format!("{who}.transform: column '{name}' not found. {e}"))
    })?;
    let ca = s.f64().map_err(|e| {
        Error::InvalidInput(format!(
            "{who}.transform: column '{name}' has dtype {}; expected Float64. {e}",
            s.dtype()
        ))
    })?;
    let mapped: ChunkedArray<Float64Type> = ca.iter().map(|opt| opt.and_then(&f)).collect();
    df.replace(name, mapped.into_series().into()).map_err(|e| {
        Error::Computation(format!(
            "{who}.transform: failed to replace column '{name}'. {e}"
        ))
    })?;
    Ok(())
}

/// Raise every non-null `f64` element of `s` to the integer power `exp`.
///
/// `who` names the calling transformer so failures are easy to trace.
/// Nulls are preserved; the returned series keeps the input's name.
pub(crate) fn series_pow(s: &Series, exp: usize, who: &str) -> Result<Series> {
    let ca = s.f64().map_err(|e| {
        Error::Computation(format!(
            "{who}: expected f64 series for column '{}'. {e}",
            s.name()
        ))
    })?;
    let exp_f64 = exp as f64;
    let result: ChunkedArray<Float64Type> =
        ca.iter().map(|opt| opt.map(|v| v.powf(exp_f64))).collect();
    Ok(result.into_series())
}

/// Element-wise product of two `f64` series.
///
/// `who` names the calling transformer for error context. The returned
/// series has no name set yet (callers typically rename it). If either
/// input is null at a given row, the output is null at that row; NaN
/// values are propagated by the underlying `f64` multiplication.
pub(crate) fn series_mul(a: &Series, b: &Series, who: &str) -> Result<Series> {
    let ca_a = a.f64().map_err(|e| {
        Error::Computation(format!(
            "{who}: expected f64 series for column '{}'. {e}",
            a.name()
        ))
    })?;
    let ca_b = b.f64().map_err(|e| {
        Error::Computation(format!(
            "{who}: expected f64 series for column '{}'. {e}",
            b.name()
        ))
    })?;
    let result: ChunkedArray<Float64Type> = ca_a
        .iter()
        .zip(ca_b.iter())
        .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
            (Some(va), Some(vb)) => Some(va * vb),
            _ => None,
        })
        .collect();
    Ok(result.into_series())
}

/// Element-wise ratio of two `f64` series with an additive divisor floor.
///
/// Computes `a / (b + epsilon)` element-wise. The `epsilon` floor prevents
/// division-by-zero producing `NaN`/`Inf` when the divisor is exactly `0.0`;
/// set it to `0.0` to disable the floor (divisions by zero then yield
/// `±Inf`/`NaN` per IEEE-754 semantics).
///
/// The floor is sign-aware: `epsilon` is applied with the divisor's sign
/// (`b + copysign(epsilon, b)`), so a small negative divisor is moved
/// further from zero and keeps its sign — it is never flipped positive. As a
/// consequence the floor cannot be cancelled by a divisor of exactly
/// `-epsilon`.
///
/// Precondition: `epsilon` must be finite and non-negative (callers are
/// expected to validate this; a `NaN` epsilon would produce an all-`NaN`
/// output).
///
/// `who` names the calling transformer for error context. The returned
/// series has no name set yet (callers typically rename it). If either
/// input is null at a given row, the output is null at that row; NaN
/// values are propagated by the underlying `f64` division.
pub(crate) fn series_div(a: &Series, b: &Series, epsilon: f64, who: &str) -> Result<Series> {
    let ca_a = a.f64().map_err(|e| {
        Error::Computation(format!(
            "{who}: expected f64 series for column '{}'. {e}",
            a.name()
        ))
    })?;
    let ca_b = b.f64().map_err(|e| {
        Error::Computation(format!(
            "{who}: expected f64 series for column '{}'. {e}",
            b.name()
        ))
    })?;
    let result: ChunkedArray<Float64Type> = ca_a
        .iter()
        .zip(ca_b.iter())
        .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
            (Some(va), Some(vb)) => Some(va / (vb + epsilon.copysign(vb))),
            _ => None,
        })
        .collect();
    Ok(result.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{Column, NamedFrom};

    #[test]
    fn test_replace_f64_column_opt_maps_none_to_null() {
        let x = Column::from(Series::new(
            "x".into(),
            &[Some(1.0), Some(f64::NAN), Some(-3.0)],
        ));
        let mut df = DataFrame::new(3, vec![x]).unwrap();

        replace_f64_column_opt(&mut df, "x", "Test", |v| {
            if v.is_nan() { None } else { Some(v) }
        })
        .unwrap();

        let vals: Vec<Option<f64>> = df.column("x").unwrap().f64().unwrap().iter().collect();
        assert_eq!(vals[0], Some(1.0));
        assert!(vals[1].is_none(), "None from the closure must become null");
        assert_eq!(vals[2], Some(-3.0));
    }

    #[test]
    fn test_replace_f64_column_opt_missing_column_errors() {
        let mut df = DataFrame::new(0, Vec::<Column>::new()).unwrap();
        let err = replace_f64_column_opt(&mut df, "nope", "Test", Some).unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "error should name the missing column"
        );
    }

    #[test]
    fn test_replace_f64_column_opt_wrong_dtype_errors() {
        let s = Column::from(Series::new("s".into(), &["a".to_string(), "b".to_string()]));
        let mut df = DataFrame::new(2, vec![s]).unwrap();
        let err = replace_f64_column_opt(&mut df, "s", "Test", Some).unwrap_err();
        assert!(
            err.to_string().contains("expected Float64"),
            "error should mention the dtype requirement"
        );
    }
}
