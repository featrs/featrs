use polars::prelude::*;

/// End-to-end test: build a mini sklearn-like pipeline and verify it works.
#[test]
fn test_end_to_end_pipeline() {
    // Create sample data
    let a = Column::from(Series::new("feat_a".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
    let b = Column::from(Series::new(
        "feat_b".into(),
        &[10.0f64, 20.0, 30.0, 40.0, 50.0],
    ));
    let t = Column::from(Series::new("target".into(), &[0.0f64, 0.0, 0.0, 1.0, 1.0]));
    let df = DataFrame::new(5, vec![a, b, t]).unwrap();
    let features = df.select(["feat_a", "feat_b"]).unwrap();

    // StandardScaler
    use featrs::traits::{Fit, Transform};
    let mut scaler = featrs::preprocessing::scaler::StandardScaler::new();
    scaler.fit(features.clone()).unwrap();
    let scaled = scaler.transform(features.clone()).unwrap();
    assert_eq!(scaled.width(), 2);
    assert_eq!(scaled.height(), 5);

    // MinMaxScaler
    let mut minmax = featrs::preprocessing::scaler::MinMaxScaler::new();
    minmax.fit(features.clone()).unwrap();
    let mm_scaled = minmax.transform(features.clone()).unwrap();
    assert_eq!(mm_scaled.width(), 2);

    // Binarizer
    let mut binarizer = featrs::preprocessing::binarizer::Binarizer::new(2.5);
    binarizer.fit(features.clone()).unwrap();
    let bin = binarizer.transform(features.clone()).unwrap();
    assert_eq!(bin.width(), 2);

    // Normalizer
    let mut normalizer = featrs::preprocessing::normalizer::Normalizer::l2();
    normalizer.fit(features.clone()).unwrap();
    let norm = normalizer.transform(features.clone()).unwrap();
    assert_eq!(norm.width(), 2);

    // PolynomialFeatures
    let mut poly = featrs::preprocessing::polynomial_features::PolynomialFeatures::new(2).unwrap();
    poly.fit(features.clone()).unwrap();
    let pf = poly.transform(features).unwrap();
    // bias + feat_a + feat_b + feat_a^2 + feat_a*feat_b + feat_b^2
    assert_eq!(pf.width(), 6); // bias(1) + degree1(2) + degree2(3)
    assert_eq!(pf.height(), 5);
}

#[test]
fn test_end_to_end_feature_selection() {
    use featrs::traits::{Fit, FitSupervised, Transform};

    let a = Column::from(Series::new("const".into(), &[1.0f64, 1.0, 1.0, 1.0, 1.0]));
    let b = Column::from(Series::new(
        "signal".into(),
        &[0.0f64, 1.0, 2.0, 10.0, 11.0],
    ));
    let t = Column::from(Series::new("target".into(), &[0.0f64, 0.0, 0.0, 1.0, 1.0]));
    let df = DataFrame::new(5, vec![a, b, t.clone()]).unwrap();
    let features = df.select(["const", "signal"]).unwrap();
    let target = DataFrame::new(5, vec![t]).unwrap();

    // VarianceThreshold
    let mut vt = featrs::feature_selection::VarianceThreshold::new(0.1);
    vt.fit(features.clone()).unwrap();
    let filtered = vt.transform(features.clone()).unwrap();
    assert_eq!(filtered.width(), 1);
    assert_eq!(filtered.get_column_names()[0].as_str(), "signal");

    // SelectKBest
    let mut skb = featrs::feature_selection::SelectKBest::new(
        1,
        Box::new(featrs::feature_selection::select_kbest::FClassif::new()),
    );
    skb.fit(features, target).unwrap();
    let selected = skb.transform(filtered).unwrap();
    assert_eq!(selected.width(), 1);
}

#[test]
fn test_end_to_end_encoders() {
    use featrs::traits::{Fit, Transform};

    let c = Column::from(Series::new(
        "color".into(),
        &["red", "blue", "red", "green"],
    ));
    let s = Column::from(Series::new("size".into(), &["S", "M", "L", "S"]));
    let df = DataFrame::new(4, vec![c, s]).unwrap();

    // OneHotEncoder
    let mut ohe = featrs::preprocessing::encoder::OneHotEncoder::new();
    ohe.fit(df.clone()).unwrap();
    let encoded = ohe.transform(df.clone()).unwrap();
    assert_eq!(encoded.width(), 6); // 3 + 3 categories

    // LabelEncoder
    let colors = df.select(["color"]).unwrap();
    let mut le = featrs::preprocessing::encoder::LabelEncoder::new();
    le.fit(colors.clone()).unwrap();
    let labeled = le.transform(colors).unwrap();
    assert_eq!(labeled.width(), 1);

    // OrdinalEncoder
    let mut oe = featrs::preprocessing::encoder::OrdinalEncoder::new();
    oe.fit(df.clone()).unwrap();
    let ordinal = oe.transform(df.clone()).unwrap();
    assert_eq!(ordinal.width(), 2);

    // CountEncoder
    let mut ce = featrs::preprocessing::encoder::CountEncoder::new();
    ce.fit(df.clone()).unwrap();
    let counts = ce.transform(df).unwrap();
    assert_eq!(counts.width(), 2);
    // color: red x2, blue x1, green x1
    let color_counts: Vec<u32> = counts
        .column("color")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .flatten()
        .collect();
    assert_eq!(color_counts, vec![2, 1, 2, 1]);
}

#[test]
fn test_end_to_end_imputer() {
    use featrs::traits::{Fit, Transform};

    let a = Column::from(Series::new(
        "x".into(),
        &[Some(1.0f64), None, Some(3.0), None],
    ));
    let df = DataFrame::new(4, vec![a]).unwrap();

    let mut imp = featrs::preprocessing::imputer::SimpleImputer::mean();
    imp.fit(df.clone()).unwrap();
    let filled = imp.transform(df).unwrap();

    let vals: Vec<f64> = filled
        .column("x")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .flatten()
        .collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 2.0]); // mean of [1, 3] = 2
}

#[test]
fn failed_refit_clears_transformer_state() {
    use featrs::traits::{Fit, Transform};

    let numeric = DataFrame::new(
        3,
        vec![Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0]))],
    )
    .unwrap();
    let empty = DataFrame::empty();

    let mut standard = featrs::preprocessing::scaler::StandardScaler::new();
    standard.fit(numeric.clone()).unwrap();
    assert!(standard.fit(empty.clone()).is_err());
    assert!(standard.transform(numeric.clone()).is_err());

    let mut minmax = featrs::preprocessing::scaler::MinMaxScaler::new();
    minmax.fit(numeric.clone()).unwrap();
    assert!(minmax.fit(empty.clone()).is_err());
    assert!(minmax.transform(numeric.clone()).is_err());

    let mut robust = featrs::preprocessing::scaler::RobustScaler::new();
    robust.fit(numeric.clone()).unwrap();
    assert!(robust.fit(empty.clone()).is_err());
    assert!(robust.transform(numeric.clone()).is_err());

    let mut normalizer = featrs::preprocessing::normalizer::Normalizer::l2();
    normalizer.fit(numeric.clone()).unwrap();
    assert!(normalizer.fit(empty.clone()).is_err());
    assert!(normalizer.transform(numeric.clone()).is_err());

    let mut quantile = featrs::preprocessing::quantile_transformer::QuantileTransformer::new();
    quantile.fit(numeric.clone()).unwrap();
    assert!(quantile.fit(empty.clone()).is_err());
    assert!(quantile.transform(numeric.clone()).is_err());

    let mut imputer = featrs::preprocessing::imputer::SimpleImputer::mean();
    imputer.fit(numeric.clone()).unwrap();
    assert!(imputer.fit(empty.clone()).is_err());
    assert!(imputer.transform(numeric.clone()).is_err());

    let mut power = featrs::preprocessing::power_transformer::PowerTransformer::new();
    power.fit(numeric.clone()).unwrap();
    assert!(power.fit(empty.clone()).is_err());
    assert!(power.transform(numeric.clone()).is_err());

    let mut interaction = featrs::preprocessing::interaction_features::InteractionFeatures::new();
    interaction.fit(numeric.clone()).unwrap();
    assert!(interaction.fit(empty.clone()).is_err());
    assert!(interaction.transform(numeric.clone()).is_err());

    let strings = DataFrame::new(
        3,
        vec![Column::from(Series::new("token".into(), &["a", "b", "c"]))],
    )
    .unwrap();
    let mut hasher = featrs::preprocessing::feature_hasher::FeatureHasher::new(&["token"], 8);
    hasher.fit(strings.clone()).unwrap();
    assert!(hasher.fit(empty).is_err());
    assert!(hasher.transform(strings).is_err());
}
