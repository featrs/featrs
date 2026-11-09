use featrs::time_series::cyclical::CyclicalEncoder;
use featrs::time_series::diff::Difference;
use featrs::time_series::expanding::{ExpandingAggregator, ExpandingFn};
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
fn test_rolling_all_nan_window_matches_expanding() {
    // An all-NaN window must yield NaN for Min and Max, not the former
    // INFINITY/NEG_INFINITY fold seeds (see #251), and must agree with
    // ExpandingAggregator for every function and window size.
    let col = Column::from(Series::new("x".into(), &[f64::NAN; 4]));
    let df = DataFrame::new(4, vec![col]).unwrap();

    for w in [2, 3] {
        for (rolling_fn, expanding_fn, suffix) in [
            (RollingFn::Min, ExpandingFn::Min, "min"),
            (RollingFn::Max, ExpandingFn::Max, "max"),
            (RollingFn::Mean, ExpandingFn::Mean, "mean"),
            (RollingFn::Sum, ExpandingFn::Sum, "sum"),
            (RollingFn::Std, ExpandingFn::Std, "std"),
        ] {
            let mut rolling = RollingAggregator::new(&["x"], w, rolling_fn);
            rolling.fit(df.clone()).unwrap();
            let rolled = rolling.transform(df.clone()).unwrap();
            let rolled = rolled
                .column(format!("x_{suffix}_{w}").as_str())
                .unwrap()
                .f64()
                .unwrap();

            let mut expanding = ExpandingAggregator::new(&["x"], expanding_fn).min_periods(w);
            expanding.fit(df.clone()).unwrap();
            let expanded = expanding.transform(df.clone()).unwrap();
            let expanded = expanded
                .column(format!("x_expanding_{suffix}").as_str())
                .unwrap()
                .f64()
                .unwrap();

            for i in (w - 1)..df.height() {
                let r = rolled.get(i);
                let e = expanded.get(i);
                assert!(
                    r.is_some_and(|v| v.is_nan()),
                    "rolling {suffix} w={w} index {i}: {r:?}"
                );
                assert!(
                    e.is_some_and(|v| v.is_nan()),
                    "expanding {suffix} w={w} index {i}: {e:?}"
                );
            }
        }
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

    for (fn_, name, expected) in [
        (
            RollingFn::Min,
            "x_min_2",
            [
                None,
                Some(1.0f64),
                Some(2.0),
                Some(2.0),
                Some(3.0),
                Some(5.0),
            ],
        ),
        (
            RollingFn::Max,
            "x_max_2",
            [
                None,
                Some(1.0f64),
                Some(2.0),
                Some(3.0),
                Some(3.0),
                Some(5.0),
            ],
        ),
    ] {
        let mut rolling = RollingAggregator::new(&["x"], 2, fn_);
        rolling.fit(df.clone()).unwrap();
        let out = rolling.transform(df.clone()).unwrap();
        let vals: Vec<Option<f64>> = out.column(name).unwrap().f64().unwrap().iter().collect();
        assert_eq!(vals, expected.to_vec(), "{name}");
    }
}
