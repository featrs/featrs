use featrs::time_series::cyclical::CyclicalEncoder;
use featrs::time_series::diff::Difference;
use featrs::time_series::lag::Lagger;
use featrs::time_series::rolling::{RollingAggregator, RollingFn};
use featrs::traits::{Error, Fit, Transform};
use polars::prelude::{Column, DataFrame, NamedFrom, Series};

fn sales_df() -> DataFrame {
    let col = Column::from(Series::new("sales".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
    DataFrame::new(5, vec![col]).unwrap()
}

#[test]
fn test_lagger_integration() {
    let mut lagger = Lagger::new(&["sales"], &[1]);
    let df = sales_df();

    lagger.fit(df.clone()).unwrap();
    let result = lagger.transform(df).unwrap();

    let names: Vec<String> = result
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("lag")),
        "expected a lag column, got {names:?}"
    );
    assert_eq!(result.height(), 5);
}

#[test]
fn test_rolling_aggregator_integration() {
    let mut rolling = RollingAggregator::new(&["sales"], 3, RollingFn::Mean);
    let df = sales_df();

    rolling.fit(df.clone()).unwrap();
    let result = rolling.transform(df).unwrap();

    assert_eq!(result.height(), 5);
}

#[test]
fn test_difference_integration() {
    let mut diff = Difference::diff(&["sales"], 1);
    let df = sales_df();

    diff.fit(df.clone()).unwrap();
    let result = diff.transform(df).unwrap();

    assert_eq!(result.height(), 5);
    let names: Vec<String> = result
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("diff")),
        "expected a diff column, got {names:?}"
    );
}

#[test]
fn test_cyclical_encoder_integration() {
    let mut enc = CyclicalEncoder::new(&["sales"], 7);
    let df = sales_df();

    enc.fit(df.clone()).unwrap();
    let result = enc.transform(df).unwrap();

    // CyclicalEncoder adds sin + cos columns for each input column.
    assert!(result.width() >= 2);
    assert_eq!(result.height(), 5);
}

#[test]
fn test_rolling_fit_rejects_non_float64_columns() {
    // A valid column first proves the check reaches every configured column,
    // not just the first one.
    let good = Column::from(Series::new("good".into(), &[1.0_f64, 2.0, 3.0]));
    let bad = Column::from(Series::new("bad".into(), &["a", "b", "c"]));
    let df = DataFrame::new(3, vec![good, bad]).unwrap();

    let mut rolling = RollingAggregator::new(&["good", "bad"], 2, RollingFn::Mean);
    let err = rolling.fit(df).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(ref msg)
            if msg.contains("column 'bad'") && msg.contains("expected Float64")),
        "expected InvalidInput naming the String column, got {err:?}"
    );
}

#[test]
fn test_difference_fit_rejects_non_float64_columns() {
    // Difference::fit had the same gap: existence checked, dtype not,
    // so transform failed with InvalidInput for types visible at fit.
    let good = Column::from(Series::new("good".into(), &[1.0_f64, 2.0, 3.0]));
    let bad = Column::from(Series::new("bad".into(), &[1_i64, 2, 3]));
    let df = DataFrame::new(3, vec![good, bad]).unwrap();

    let mut d = Difference::diff(&["good", "bad"], 1);
    let err = d.fit(df).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(ref msg)
            if msg.contains("column 'bad'") && msg.contains("expected Float64")),
        "expected InvalidInput naming the Int64 column, got {err:?}"
    );
}
