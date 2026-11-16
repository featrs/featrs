//! OneHotEncoder fit and transform over 50 categories and 10k rows.

mod common;

use common::{Profile, bench_fit_transform, categorical_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;

fn encoder(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(10_000);
    let categories = 50;
    let frame = categorical_frame(rows, categories);
    let shape = format!("{rows}x{categories}cat");

    let mut group = c.benchmark_group("encoder");
    bench_fit_transform(&mut group, "onehot", &shape, &frame, OneHotEncoder::new);
    group.finish();
}

criterion_group!(benches, encoder);
criterion_main!(benches);
