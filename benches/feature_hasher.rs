//! FeatureHasher fit and transform over 100 string columns and 10k rows.

mod common;

use common::{Profile, bench_fit_transform, text_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;

fn feature_hasher(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(10_000);
    let cols = 100;
    let frame = text_frame(rows, cols);
    let shape = format!("{rows}x{cols}");

    let names: Vec<String> = (0..cols).map(|index| format!("c{index}")).collect();
    let columns: Vec<&str> = names.iter().map(String::as_str).collect();

    let mut group = c.benchmark_group("feature_hasher");
    bench_fit_transform(&mut group, "1024", &shape, &frame, || {
        FeatureHasher::new(&columns, 1024)
    });
    group.finish();
}

criterion_group!(benches, feature_hasher);
criterion_main!(benches);
