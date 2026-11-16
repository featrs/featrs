//! Normalizer L2 fit and transform at 10k x 100.

mod common;

use common::{Profile, bench_fit_transform, numeric_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;

fn normalizer(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(10_000);
    let frame = numeric_frame(rows, 100);
    let shape = format!("{rows}x100");

    let mut group = c.benchmark_group("normalizer");
    bench_fit_transform(&mut group, "l2", &shape, &frame, Normalizer::l2);
    group.finish();
}

criterion_group!(benches, normalizer);
criterion_main!(benches);
