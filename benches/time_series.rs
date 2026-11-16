//! RollingAggregator fit and transform with a 1000-row window over 100k rows.
//!
//! The transform is O(n * window), so the group runs ten samples per function
//! instead of Criterion's default hundred.

mod common;

use common::{Profile, bench_fit_transform, numeric_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;
use featrs::time_series::rolling::RollingFn;

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
        bench_fit_transform(&mut group, name, &shape, &frame, || {
            RollingAggregator::new(&["f0"], 1000, function)
        });
    }

    group.finish();
}

criterion_group!(benches, time_series);
criterion_main!(benches);
