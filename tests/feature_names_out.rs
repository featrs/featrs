//! `FeatureNamesOut` describes the exact schema `transform` emits.
//!
//! Every transformer in the crate is exercised through the same helper: fit,
//! call `feature_names_out` with the input column names, then compare the
//! prediction with the real `transform` output. A transformer whose plan drifts
//! from its transform fails here.

use featrs::pipeline::conditional_pipeline::ColumnPredicate;
use featrs::prelude::*;
use featrs::time_series::expanding::ExpandingFn;
use featrs::time_series::rolling::RollingFn;
use polars::prelude::*;

fn names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn f64_col(name: &str, vals: &[f64]) -> Column {
    Column::from(Series::new(name.into(), vals))
}

fn opt_f64_col(name: &str, vals: &[Option<f64>]) -> Column {
    Column::from(Series::new(name.into(), vals))
}

fn str_col(name: &str, vals: &[&str]) -> Column {
    Column::from(Series::new(name.into(), vals))
}

fn date_col(name: &str, vals: &[i32]) -> Column {
    Series::new(name.into(), vals)
        .cast(&DataType::Date)
        .unwrap()
        .into()
}

fn frame(height: usize, cols: Vec<Column>) -> DataFrame {
    DataFrame::new(height, cols).unwrap()
}

fn numeric_df() -> DataFrame {
    frame(
        6,
        vec![
            f64_col("a", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            f64_col("b", &[2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
            f64_col("c", &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]),
        ],
    )
}

fn categorical_df() -> DataFrame {
    frame(
        4,
        vec![
            f64_col("num", &[1.0, 2.0, 3.0, 4.0]),
            str_col("cat", &["red", "blue", "red", "green"]),
        ],
    )
}

fn string_df() -> DataFrame {
    frame(
        4,
        vec![
            str_col("cat", &["red", "blue", "red", "green"]),
            str_col("kind", &["x", "y", "x", "y"]),
        ],
    )
}

fn date_df() -> DataFrame {
    frame(
        4,
        vec![
            date_col("date", &[19_723, 19_724, 19_725, 19_726]),
            f64_col("num", &[1.0, 2.0, 3.0, 4.0]),
        ],
    )
}

fn text_df() -> DataFrame {
    frame(
        3,
        vec![str_col(
            "text",
            &["the cat sat", "the dog sat", "a cat ran"],
        )],
    )
}

fn y_regression() -> DataFrame {
    frame(6, vec![f64_col("y", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
}

fn y_binary() -> DataFrame {
    frame(4, vec![f64_col("y", &[0.0, 1.0, 0.0, 1.0])])
}

fn assert_names_match<T>(mut t: T, df: DataFrame)
where
    T: Fit<DataFrame, Output = ()> + Transform<DataFrame, Output = DataFrame> + FeatureNamesOut,
{
    t.fit(df.clone()).expect("fit should succeed");
    let predicted = t
        .feature_names_out(&names(&df))
        .expect("feature_names_out should succeed");
    let output = t.transform(df).expect("transform should succeed");
    assert_eq!(predicted, names(&output));
}

fn assert_names_match_supervised<T>(mut t: T, df: DataFrame, y: DataFrame)
where
    T: FitSupervised<DataFrame, DataFrame>
        + Transform<DataFrame, Output = DataFrame>
        + FeatureNamesOut,
{
    t.fit(df.clone(), y).expect("fit should succeed");
    let predicted = t
        .feature_names_out(&names(&df))
        .expect("feature_names_out should succeed");
    let output = t.transform(df).expect("transform should succeed");
    assert_eq!(predicted, names(&output));
}

#[test]
fn scalers_and_elementwise_transforms_match() {
    assert_names_match(StandardScaler::new(), numeric_df());
    assert_names_match(MinMaxScaler::new(), numeric_df());
    assert_names_match(RobustScaler::new(), numeric_df());
    assert_names_match(MaxAbsScaler::new(), numeric_df());
    assert_names_match(Binarizer::new(0.5), numeric_df());
    assert_names_match(Normalizer::l2(), numeric_df());
    assert_names_match(PowerTransformer::new(), numeric_df());
    assert_names_match(QuantileTransformer::new(), numeric_df());
    assert_names_match(OutlierClipper::new(), numeric_df());
    assert_names_match(Winsorizer::new(), numeric_df());
    assert_names_match(
        SimpleImputer::mean(),
        frame(
            4,
            vec![
                opt_f64_col("a", &[Some(1.0), None, Some(3.0), Some(4.0)]),
                opt_f64_col("b", &[Some(2.0), Some(3.0), None, Some(5.0)]),
            ],
        ),
    );
    assert_names_match(LogTransformer::new(), numeric_df());
    assert_names_match(
        StringCleaner::new().columns(&["cat"]).trim(),
        categorical_df(),
    );
    assert_names_match(
        RareCategoryGrouper::new()
            .columns(&["cat"])
            .threshold(Threshold::MinCount(1)),
        categorical_df(),
    );
}

#[test]
fn encoders_match() {
    assert_names_match(OneHotEncoder::new(), string_df());
    assert_names_match(
        LabelEncoder::new(),
        frame(4, vec![str_col("cat", &["b", "a", "c", "a"])]),
    );
    assert_names_match(OrdinalEncoder::new(), string_df());
    assert_names_match(CountEncoder::new(), categorical_df());
    assert_names_match(FrequencyEncoder::new(), categorical_df());
    assert_names_match(BinaryEncoder::new(), categorical_df());
    assert_names_match(FeatureHasher::new(&["cat"], 4), categorical_df());
    assert_names_match(HolidayEncoder::new(), date_df());

    assert_names_match_supervised(LeaveOneOutEncoder::new(), categorical_df(), y_binary());
    assert_names_match_supervised(TargetEncoder::new(), categorical_df(), y_binary());
}

#[test]
fn feature_generators_match() {
    assert_names_match(InteractionFeatures::new(), numeric_df());
    assert_names_match(RatioFeatures::new(), numeric_df());
    assert_names_match(
        PolynomialFeatures::new(2).unwrap().include_bias(true),
        frame(
            4,
            vec![
                f64_col("a", &[1.0, 2.0, 3.0, 4.0]),
                f64_col("b", &[2.0, 3.0, 4.0, 5.0]),
            ],
        ),
    );
    assert_names_match(KBinsDiscretizer::new().n_bins(2), numeric_df());
    assert_names_match(
        KBinsDiscretizer::new()
            .n_bins(2)
            .encode(EncodeMode::OneHotDropFirst),
        numeric_df(),
    );
    assert_names_match(
        MissingIndicator::new(&["a"]),
        frame(
            4,
            vec![
                opt_f64_col("a", &[Some(1.0), None, Some(3.0), None]),
                f64_col("b", &[1.0, 2.0, 3.0, 4.0]),
            ],
        ),
    );
    assert_names_match(DatetimeFeatures::new(), date_df());
    assert_names_match(TimeSince::new(), date_df());
    assert_names_match(Lagger::new(&["a"], &[1, 2]), numeric_df());
    assert_names_match(
        RollingAggregator::new(&["a"], 2, RollingFn::Mean),
        numeric_df(),
    );
    assert_names_match(
        ExpandingAggregator::new(&["a"], ExpandingFn::Mean),
        numeric_df(),
    );
    assert_names_match(
        ExponentiallyWeightedMovingAverage::new(&["a"], EWMASmoothing::Alpha(0.5)),
        numeric_df(),
    );
    assert_names_match(Difference::diff(&["a"], 1), numeric_df());
    assert_names_match(CyclicalEncoder::new(&["a"], 24), numeric_df());
}

#[test]
fn text_vectorizers_match() {
    assert_names_match(CountVectorizer::new().column("text"), text_df());
    assert_names_match(TFIDFVectorizer::new().column("text"), text_df());
    assert_names_match(CharacterNGramVectorizer::new().column("text"), text_df());
}

#[test]
fn column_removers_match() {
    assert_names_match(
        ConstantColumnRemover::new(),
        frame(
            4,
            vec![
                f64_col("constant", &[1.0, 1.0, 1.0, 1.0]),
                f64_col("varying", &[1.0, 2.0, 3.0, 4.0]),
            ],
        ),
    );
    assert_names_match(
        DuplicateColumnRemover::new(),
        frame(
            4,
            vec![
                f64_col("a", &[1.0, 2.0, 3.0, 4.0]),
                f64_col("b", &[1.0, 2.0, 3.0, 4.0]),
                f64_col("c", &[4.0, 3.0, 2.0, 1.0]),
            ],
        ),
    );
}

#[test]
fn feature_selectors_match() {
    assert_names_match(VarianceThreshold::new(0.0), numeric_df());
    assert_names_match(CorrelationThreshold::new().threshold(0.9), numeric_df());
    assert_names_match_supervised(
        SelectKBest::new(1, Box::new(FRegression::new())),
        numeric_df(),
        y_regression(),
    );
    assert_names_match_supervised(
        SelectPercentile::new()
            .percentile(50.0)
            .score_func(Box::new(FRegression::new())),
        numeric_df(),
        y_regression(),
    );
    assert_names_match_supervised(
        MutualInformationSelector::new()
            .k(1)
            .task(MITask::Regression),
        numeric_df(),
        y_regression(),
    );
}

#[test]
fn auto_transformers_match() {
    assert_names_match(AutoScaler::new(), numeric_df());
    assert_names_match(
        AutoImputer::new(),
        frame(
            4,
            vec![
                opt_f64_col("a", &[Some(1.0), None, Some(3.0), Some(4.0)]),
                opt_f64_col("b", &[Some(2.0), Some(3.0), None, Some(5.0)]),
            ],
        ),
    );
    assert_names_match(
        AutoTypeDetector::new().cat_threshold(3).hash_buckets(4),
        frame(
            4,
            vec![
                f64_col("num", &[1.0, 2.0, 3.0, 4.0]),
                str_col("cat", &["a", "b", "a", "b"]),
                str_col("high", &["w", "x", "y", "z"]),
            ],
        ),
    );
}

#[test]
fn pipelines_match() {
    let pipeline = Pipeline::new(vec![
        ("scale".into(), Box::new(StandardScaler::new())),
        ("lag".into(), Box::new(Lagger::new(&["a"], &[1]))),
    ])
    .unwrap();
    assert_names_match(pipeline, numeric_df());

    let column_transformer = ColumnTransformer::new(
        vec![(
            "scale".into(),
            Box::new(StandardScaler::new()),
            vec!["a".into(), "b".into()],
        )],
        Remainder::Passthrough,
    );
    assert_names_match(column_transformer, numeric_df());

    let union = FeatureUnion::new(vec![
        ("scale".into(), Box::new(StandardScaler::new())),
        (
            "count".into(),
            Box::new(CountVectorizer::new().column("text")),
        ),
    ])
    .unwrap();
    assert_names_match(
        union,
        frame(
            3,
            vec![
                f64_col("num", &[1.0, 2.0, 3.0]),
                str_col("text", &["the cat sat", "the dog sat", "a cat ran"]),
            ],
        ),
    );

    let conditional = ConditionalPipeline::new()
        .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()))
        .rule(ColumnPredicate::IsString, Box::new(OneHotEncoder::new()));
    assert_names_match(conditional, categorical_df());

    let cached = CachedPipeline::new(
        Pipeline::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap(),
    );
    assert_names_match(cached, numeric_df());
}

#[test]
fn unfitted_transformers_report_not_fitted() {
    let input = vec!["a".to_string()];
    assert!(matches!(
        StandardScaler::new().feature_names_out(&input),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        OneHotEncoder::new().feature_names_out(&input),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        VarianceThreshold::new(0.0).feature_names_out(&input),
        Err(Error::NotFitted(_))
    ));
    assert!(matches!(
        Pipeline::new(vec![("scale".into(), Box::new(StandardScaler::new()))])
            .unwrap()
            .feature_names_out(&input),
        Err(Error::NotFitted(_))
    ));
}

#[test]
fn pipeline_threads_names_through_every_step() {
    let df = numeric_df();
    let mut pipeline = Pipeline::new(vec![
        ("poly".into(), Box::new(PolynomialFeatures::new(2).unwrap())),
        ("lag".into(), Box::new(Lagger::new(&["a"], &[1]))),
    ])
    .unwrap();
    pipeline.fit(df.clone()).unwrap();
    let predicted = pipeline.feature_names_out(&names(&df)).unwrap();
    let output = pipeline.transform(df).unwrap();
    assert_eq!(predicted, names(&output));
}

#[test]
fn column_transformer_names_include_the_passthrough_remainder() {
    let df = numeric_df();
    let mut ct = ColumnTransformer::new(
        vec![(
            "scale".into(),
            Box::new(StandardScaler::new()),
            vec!["b".into()],
        )],
        Remainder::Passthrough,
    );
    ct.fit(df.clone()).unwrap();
    let predicted = ct.feature_names_out(&names(&df)).unwrap();
    assert_eq!(predicted, vec!["b", "a", "c"]);
    let output = ct.transform(df).unwrap();
    assert_eq!(predicted, names(&output));
}

#[test]
fn missing_fitted_columns_are_invalid_input() {
    let mut scaler = StandardScaler::new();
    scaler.fit(numeric_df()).unwrap();
    let err = scaler.feature_names_out(&["a".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    let mut vt = VarianceThreshold::new(0.0);
    vt.fit(numeric_df()).unwrap();
    let err = vt.feature_names_out(&["nope".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    let mut lagger = Lagger::new(&["a"], &[1]);
    lagger.fit(numeric_df()).unwrap();
    let err = lagger.feature_names_out(&["b".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    let mut target = TargetEncoder::new();
    target.fit(categorical_df(), y_binary()).unwrap();
    let err = target.feature_names_out(&["num".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn generated_name_collisions_are_invalid_input() {
    let mut dt = DatetimeFeatures::new();
    dt.fit(date_df()).unwrap();
    let colliding = vec!["date".to_string(), "date_year".to_string()];
    let err = dt.feature_names_out(&colliding).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    let frame_df = frame(
        4,
        vec![
            opt_f64_col("a", &[Some(1.0), None, Some(3.0), None]),
            f64_col("b", &[1.0, 2.0, 3.0, 4.0]),
        ],
    );
    let mut mi = MissingIndicator::new(&["a"]);
    mi.fit(frame_df).unwrap();
    let colliding = vec!["a".to_string(), "b".to_string(), "a_missing".to_string()];
    let err = mi.feature_names_out(&colliding).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    // The same colliding frames are rejected by `transform`.
    let colliding_df = frame(
        4,
        vec![
            date_col("date", &[19_723, 19_724, 19_725, 19_726]),
            f64_col("date_year", &[2024.0, 2024.0, 2024.0, 2024.0]),
        ],
    );
    let mut dt = DatetimeFeatures::new();
    dt.fit(date_df()).unwrap();
    assert!(matches!(
        dt.transform(colliding_df),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn leave_one_out_rejects_a_different_column_set() {
    let mut enc = LeaveOneOutEncoder::new();
    enc.fit(categorical_df(), y_binary()).unwrap();
    let err = enc
        .feature_names_out(&["cat".to_string(), "extra".to_string()])
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
    let err = enc.feature_names_out(&["cat".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn binary_encoder_rejects_missing_fitted_columns() {
    let mut enc = BinaryEncoder::new();
    enc.fit(string_df()).unwrap();
    let ok = enc.feature_names_out(&names(&string_df())).unwrap();
    assert!(
        ok.iter()
            .all(|n| n.contains("_bit_") || n == "cat" || n == "kind")
    );
    let err = enc.feature_names_out(&["cat".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

/// A test-only transformer whose schema prediction always fails, so the
/// composition layers' error handling can be checked.
struct AlwaysInvalid;

impl Fit<DataFrame> for AlwaysInvalid {
    type Output = ();

    fn fit(&mut self, _x: DataFrame) -> Result<()> {
        Ok(())
    }
}

impl Transform<DataFrame> for AlwaysInvalid {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        Ok(x)
    }
}

impl FeatureNamesOut for AlwaysInvalid {
    fn feature_names_out(&self, _input_columns: &[String]) -> Result<Vec<String>> {
        Err(Error::InvalidInput("boom".into()))
    }
}

#[test]
fn column_transformer_preserves_child_error_variants() {
    let df = numeric_df();
    let mut ct = ColumnTransformer::new(
        vec![(
            "bad".into(),
            Box::new(AlwaysInvalid) as Box<dyn DataFrameTransformer>,
            vec!["a".into()],
        )],
        Remainder::Drop,
    );
    ct.fit(df.clone()).unwrap();
    let err = ct.feature_names_out(&names(&df)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected the child's InvalidInput to survive, got: {err:?}"
    );
}

#[test]
fn conditional_pipeline_rejects_missing_routed_columns() {
    let mut cp = ConditionalPipeline::new()
        .rule(ColumnPredicate::IsFloat64, Box::new(StandardScaler::new()))
        .rule(ColumnPredicate::IsString, Box::new(OneHotEncoder::new()));
    cp.fit(categorical_df()).unwrap();
    let err = cp.feature_names_out(&["num".to_string()]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}
