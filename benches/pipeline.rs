//! Five-step Pipeline fit and transform at 10k x 50.

mod common;

use common::{Profile, bench_fit_transform, numeric_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::pipeline::DataFrameTransformer;
use featrs::prelude::*;

fn pipeline(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(10_000);
    let frame = numeric_frame(rows, 50);
    let shape = format!("{rows}x50");

    let mut group = c.benchmark_group("pipeline");
    bench_fit_transform(&mut group, "five-step", &shape, &frame, || {
        let steps: Vec<(String, Box<dyn DataFrameTransformer>)> = vec![
            ("impute".to_string(), Box::new(SimpleImputer::mean())),
            ("standard".to_string(), Box::new(StandardScaler::new())),
            ("winsorize".to_string(), Box::new(Winsorizer::new())),
            ("robust".to_string(), Box::new(RobustScaler::new())),
            ("normalize".to_string(), Box::new(Normalizer::l2())),
        ];
        Pipeline::new(steps).expect("five steps")
    });
    group.finish();
}

criterion_group!(benches, pipeline);
criterion_main!(benches);
