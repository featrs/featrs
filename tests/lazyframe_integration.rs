//! Integration tests: lazy transforms must collect to the eager results.

use featrs::pipeline::{DataFrameTransformer, LazyPipeline, LazyTransformer, Pipeline};
use featrs::prelude::{Binarizer, LogMethod, LogTransformer, MinMaxScaler, StandardScaler};
use featrs::traits::{Error, FeatureNamesOut, Fit, FitLazy, Transform, TransformLazy};
use polars::prelude::{Column, DataFrame, DataType, IntoLazy, NamedFrom, Series, col, lit};

const TOL: f64 = 1e-12;

fn frame(columns: Vec<Column>) -> DataFrame {
    let height = columns[0].len();
    DataFrame::new(height, columns).unwrap()
}

fn sample_frame() -> DataFrame {
    frame(vec![
        Column::from(Series::new("a".into(), &[1.0_f64, 2.5, -3.0, 4.75, 10.0])),
        Column::from(Series::new("b".into(), &[0.1_f64, 0.2, 0.3, 0.4, 0.5])),
        Column::from(Series::new("n".into(), &[1_i64, 2, 3, 4, 5])),
    ])
}

/// Compare names, order, dtypes, nulls and values. Floats must agree within `tol`.
fn assert_frames_close(actual: &DataFrame, expected: &DataFrame, tol: f64) {
    assert_eq!(
        actual.get_column_names(),
        expected.get_column_names(),
        "column names or order differ"
    );
    assert_eq!(actual.height(), expected.height(), "row counts differ");
    for name in expected.get_column_names() {
        let a = actual.column(name.as_str()).unwrap();
        let e = expected.column(name.as_str()).unwrap();
        assert_eq!(a.dtype(), e.dtype(), "dtype of '{name}' differs");
        if e.dtype() == &DataType::Float64 {
            let av = a.f64().unwrap();
            let ev = e.f64().unwrap();
            for (i, (x, y)) in av.iter().zip(ev.iter()).enumerate() {
                match (x, y) {
                    (None, None) => {}
                    (Some(x), Some(y)) if x.is_nan() && y.is_nan() => {}
                    (Some(x), Some(y)) => {
                        assert!((x - y).abs() <= tol, "'{name}' row {i}: {x} vs {y}");
                    }
                    _ => panic!("'{name}' row {i}: null mismatch {x:?} vs {y:?}"),
                }
            }
        } else {
            assert!(a.equals_missing(e), "'{name}' differs");
        }
    }
}

#[test]
fn standard_scaler_lazy_matches_eager() {
    let df = sample_frame();
    let mut scaler = StandardScaler::new();
    scaler.fit(df.clone()).unwrap();
    let eager = scaler.transform(df.clone()).unwrap();
    let lazy = scaler.transform_lazy(df.lazy()).unwrap().collect().unwrap();
    assert_frames_close(&lazy, &eager, TOL);
    assert_eq!(lazy.column("n").unwrap().dtype(), &DataType::Int64);
}

#[test]
fn min_max_scaler_lazy_matches_eager() {
    let df = sample_frame();
    let mut scaler = MinMaxScaler::new().feature_range((-1.0, 1.0));
    scaler.fit(df.clone()).unwrap();
    let eager = scaler.transform(df.clone()).unwrap();
    let lazy = scaler.transform_lazy(df.lazy()).unwrap().collect().unwrap();
    assert_frames_close(&lazy, &eager, TOL);
    assert_eq!(lazy.column("n").unwrap().dtype(), &DataType::Int64);
}

#[test]
fn binarizer_lazy_matches_eager_with_nulls_and_nan() {
    let df = frame(vec![
        Column::from(Series::new(
            "x".into(),
            &[Some(0.5_f64), None, Some(f64::NAN), Some(2.0), Some(-1.0)],
        )),
        Column::from(Series::new("y".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0])),
    ]);
    let mut binarizer = Binarizer::new(1.0);
    binarizer.fit(df.clone()).unwrap();
    let eager = binarizer.transform(df.clone()).unwrap();
    let lazy = binarizer
        .transform_lazy(df.lazy())
        .unwrap()
        .collect()
        .unwrap();

    assert_frames_close(&lazy, &eager, TOL);
    for out in [&lazy, &eager] {
        let x = out.column("x").unwrap();
        assert_eq!(x.dtype(), &DataType::Float64);
        let values: Vec<Option<f64>> = x.f64().unwrap().iter().collect();
        assert_eq!(values, vec![Some(0.0), None, None, Some(1.0), Some(0.0)]);
    }
}

#[test]
fn log_transformer_lazy_matches_eager_all_methods() {
    let df = frame(vec![
        Column::from(Series::new("a".into(), &[0.5_f64, 1.0, 2.0, 10.0, 100.0])),
        Column::from(Series::new("b".into(), &[0.25_f64, 3.0, 7.5, 42.0, 1e3])),
    ]);
    let methods = [
        LogMethod::Ln,
        LogMethod::Log1p,
        LogMethod::Log2,
        LogMethod::Log10,
        LogMethod::LogBase(3.0),
    ];
    for method in methods {
        let mut transformer = LogTransformer::new().method(method);
        transformer.fit(df.clone()).unwrap();
        let eager = transformer.transform(df.clone()).unwrap();
        let lazy = transformer
            .transform_lazy(df.clone().lazy())
            .unwrap()
            .collect()
            .unwrap();
        assert_frames_close(&lazy, &eager, TOL);
    }
}

#[test]
fn lazy_pipeline_matches_eager_pipeline() {
    let df = sample_frame();

    let eager_steps: Vec<(String, Box<dyn DataFrameTransformer>)> = vec![
        ("scale".into(), Box::new(StandardScaler::new())),
        ("bin".into(), Box::new(Binarizer::new(0.0))),
    ];
    let mut eager = Pipeline::new(eager_steps).unwrap();
    eager.fit(df.clone()).unwrap();
    let expected = eager.transform(df.clone()).unwrap();

    let lazy_steps: Vec<(String, Box<dyn LazyTransformer>)> = vec![
        ("scale".into(), Box::new(StandardScaler::new())),
        ("bin".into(), Box::new(Binarizer::new(0.0))),
    ];
    let mut lazy_pipeline = LazyPipeline::new(lazy_steps).unwrap();
    lazy_pipeline.fit_lazy(df.clone().lazy()).unwrap();
    let actual = lazy_pipeline
        .transform_lazy(df.clone().lazy())
        .unwrap()
        .collect()
        .unwrap();
    assert_frames_close(&actual, &expected, TOL);

    let input_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .collect();
    assert_eq!(
        lazy_pipeline.feature_names_out(&input_names).unwrap(),
        eager.feature_names_out(&input_names).unwrap()
    );
}

#[test]
fn fit_lazy_equals_fit() {
    let df = sample_frame();

    let mut standard_fit = StandardScaler::new();
    standard_fit.fit(df.clone()).unwrap();
    let mut standard_lazy = StandardScaler::new();
    standard_lazy.fit_lazy(df.clone().lazy()).unwrap();
    let a = standard_fit
        .transform_lazy(df.clone().lazy())
        .unwrap()
        .collect()
        .unwrap();
    let b = standard_lazy
        .transform_lazy(df.clone().lazy())
        .unwrap()
        .collect()
        .unwrap();
    assert_frames_close(&a, &b, TOL);

    let mut minmax_fit = MinMaxScaler::new();
    minmax_fit.fit(df.clone()).unwrap();
    let mut minmax_lazy = MinMaxScaler::new();
    minmax_lazy.fit_lazy(df.clone().lazy()).unwrap();
    let a = minmax_fit
        .transform_lazy(df.clone().lazy())
        .unwrap()
        .collect()
        .unwrap();
    let b = minmax_lazy
        .transform_lazy(df.clone().lazy())
        .unwrap()
        .collect()
        .unwrap();
    assert_frames_close(&a, &b, TOL);
}

#[test]
fn not_fitted_errors() {
    let df = sample_frame();

    assert!(matches!(
        StandardScaler::new().transform_lazy(df.clone().lazy()),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        MinMaxScaler::new().transform_lazy(df.clone().lazy()),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        Binarizer::new(0.0).transform_lazy(df.clone().lazy()),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        LogTransformer::new()
            .method(LogMethod::Ln)
            .transform_lazy(df.clone().lazy()),
        Err(Error::NotFitted(_))
    ));

    let steps: Vec<(String, Box<dyn LazyTransformer>)> =
        vec![("scale".into(), Box::new(StandardScaler::new()))];
    let pipeline = LazyPipeline::new(steps).unwrap();
    assert!(matches!(
        pipeline.transform_lazy(df.lazy()),
        Err(Error::NotFitted(_))
    ));
}

#[test]
fn predicate_composes_with_lazy_transform() {
    let df = sample_frame();
    let mut scaler = StandardScaler::new();
    scaler.fit(df.clone()).unwrap();

    let predicate = col("a").gt(lit(2.0));
    let lazy = scaler
        .transform_lazy(df.clone().lazy().filter(predicate.clone()))
        .unwrap()
        .collect()
        .unwrap();

    let filtered = df.clone().lazy().filter(predicate).collect().unwrap();
    assert!(filtered.height() < df.height());
    let expected = scaler.transform(filtered).unwrap();
    assert_frames_close(&lazy, &expected, TOL);
}

fn lazy_err(result: Result<polars::prelude::LazyFrame, Error>) -> Error {
    match result {
        Ok(_) => panic!("expected an error, got a plan"),
        Err(e) => e,
    }
}

fn assert_rejects_int64_a(err: Error) {
    let text = err.to_string();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {text}"
    );
    assert!(
        text.contains("column 'a' has dtype i64; expected Float64"),
        "{text}"
    );
}

#[test]
fn lazy_transforms_reject_non_float_columns() {
    let df = sample_frame();
    let ints = frame(vec![Column::from(Series::new(
        "a".into(),
        &[1_i64, 2, 3, 4, 5],
    ))]);

    let mut standard = StandardScaler::new();
    standard.fit(df.clone()).unwrap();
    assert!(standard.transform(ints.clone()).is_err());
    assert_rejects_int64_a(lazy_err(standard.transform_lazy(ints.clone().lazy())));

    let mut min_max = MinMaxScaler::new();
    min_max.fit(df.clone()).unwrap();
    assert!(min_max.transform(ints.clone()).is_err());
    assert_rejects_int64_a(lazy_err(min_max.transform_lazy(ints.clone().lazy())));

    let positives = frame(vec![Column::from(Series::new(
        "a".into(),
        &[0.5_f64, 1.0, 2.0],
    ))]);
    let mut log = LogTransformer::new();
    log.fit(positives).unwrap();
    assert!(log.transform(ints.clone()).is_err());
    assert_rejects_int64_a(lazy_err(log.transform_lazy(ints.clone().lazy())));

    let mut pipeline =
        LazyPipeline::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
    pipeline.fit(df).unwrap();
    assert_rejects_int64_a(lazy_err(pipeline.transform_lazy(ints.lazy())));
}

#[test]
fn failed_fit_lazy_leaves_transformer_unfitted() {
    let df = sample_frame();
    let missing = df.clone().lazy().select([col("no_such_column")]);

    let mut scaler = StandardScaler::new();
    scaler.fit(df.clone()).unwrap();
    assert!(
        scaler.fit_lazy(missing.clone()).is_err(),
        "collect must fail"
    );
    assert!(matches!(
        scaler.transform(df.clone()),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        scaler.transform_lazy(df.clone().lazy()),
        Err(Error::NotFitted(_))
    ));

    let steps: Vec<(String, Box<dyn LazyTransformer>)> =
        vec![("scale".into(), Box::new(StandardScaler::new()))];
    let mut pipeline = LazyPipeline::new(steps).unwrap();
    pipeline.fit(df.clone()).unwrap();
    assert!(pipeline.fit_lazy(missing).is_err(), "collect must fail");
    assert!(matches!(
        pipeline.transform_lazy(df.lazy()),
        Err(Error::NotFitted(_))
    ));
}
