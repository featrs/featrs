//! RollingAggregator transform with a 1000-row window over 100k rows.
//!
//! The aggregator is O(n * window), so the group runs ten samples per
//! function instead of Criterion's default hundred.

mod common;

use common::{Profile, numeric_frame};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use featrs::prelude::*;
use featrs::time_series::rolling::RollingFn;
use std::hint::black_box;

fn time_series(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(100_000);
    let frame = numeric_frame(rows, 1);
    let shape = format!("{rows}rows");

    let mut group = c.benchmark_group("time_series");
    group.sample_size(10);

    let functions = [
        ("mean", RollingFn::Mean),
        ("std", RollingFn::Std),
        ("min", RollingFn::Min),
        ("max", RollingFn::Max),
        ("sum", RollingFn::Sum),
    ];

    for (name, function) in functions {
        let mut aggregator = RollingAggregator::new(&["f0"], 1000, function);
        aggregator.fit(frame.clone()).expect("fit");
        group.bench_with_input(BenchmarkId::new(name, &shape), &frame, |bencher, frame| {
            bencher.iter(|| black_box(aggregator.transform(frame.clone()).expect("transform")))
        });
    }

    group.finish();
}

criterion_group!(benches, time_series);
criterion_main!(benches);
