use featrs::time_series::cyclical::CyclicalEncoder;
use featrs::time_series::diff::Difference;
use featrs::time_series::lag::Lagger;
use featrs::time_series::rolling::{RollingAggregator, RollingFn};
use featrs::traits::{Fit, Transform};
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
fn test_rolling_all_nan_window_matches_mean_and_expanding() {
    // An all-NaN window must yield NaN for Min and Max, not the former
    // INFINITY/NEG_INFINITY fold seeds (see #251), matching Mean/Sum/Std
    // and ExpandingAggregator on identical input.
    let col = Column::from(Series::new(
        "x".into(),
        &[f64::NAN, f64::NAN, f64::NAN, f64::NAN],
    ));
    let df = DataFrame::new(4, vec![col]).unwrap();

    for (fn_, name) in [
        (RollingFn::Min, "x_min_2"),
        (RollingFn::Max, "x_max_2"),
        (RollingFn::Mean, "x_mean_2"),
    ] {
        let mut rolling = RollingAggregator::new(&["x"], 2, fn_);
        rolling.fit(df.clone()).unwrap();
        let out = rolling.transform(df.clone()).unwrap();
        let v = out
            .column(name)
            .unwrap()
            .f64()
            .unwrap()
            .get(2)
            .unwrap_or_default();
        assert!(
            v.is_nan(),
            "expected NaN for {name} on an all-NaN window, got {v}"
        );
    }
}

#[test]
fn test_rolling_min_max_ignores_nan_in_mixed_windows() {
    // A mixed window containing NaN keeps its previous semantics: NaN is
    // ignored and the minimum/maximum of the remaining values is returned.
    let col = Column::from(Series::new(
        "x".into(),
        &[1.0_f64, f64::NAN, 2.0, 3.0, f64::NAN, 5.0],
    ));
    let df = DataFrame::new(6, vec![col]).unwrap();

    let mut rolling = RollingAggregator::new(&["x"], 2, RollingFn::Min);
    rolling.fit(df.clone()).unwrap();
    let out = rolling.transform(df).unwrap();

    // Mixed windows: index 0 -> None (window not complete);
    // index 1 window [1.0, NaN] -> 1.0 (NaN ignored); index 2 [NaN, 2.0] -> 2.0;
    // index 3 [2.0, 3.0] -> 2.0; index 4 [3.0, NaN] -> 3.0; index 5 [NaN, 5.0] -> 5.0.
    let vals: Vec<Option<f64>> = out
        .column("x_min_2")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(vals[0], None);
    assert_eq!(vals[1], Some(1.0));
    assert_eq!(vals[2], Some(2.0));
    assert_eq!(vals[3], Some(2.0));
    assert_eq!(vals[4], Some(3.0));
    assert_eq!(vals[5], Some(5.0));
}
