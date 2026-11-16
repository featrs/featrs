//! Scaler fit and transform at both Phase 0 shapes.

mod common;

use common::{Profile, bench_fit_transform, numeric_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;

fn scalers(c: &mut Criterion) {
    let profile = Profile::from_env();
    let mut group = c.benchmark_group("scalers");

    for (rows, cols) in [(10_000, 100), (100_000, 10)] {
        let rows = profile.rows(rows);
        let frame = numeric_frame(rows, cols);
        let shape = format!("{rows}x{cols}");

        bench_fit_transform(&mut group, "standard", &shape, &frame, StandardScaler::new);
        bench_fit_transform(&mut group, "minmax", &shape, &frame, MinMaxScaler::new);
        bench_fit_transform(&mut group, "robust", &shape, &frame, RobustScaler::new);
    }

    group.finish();
}

criterion_group!(benches, scalers);
criterion_main!(benches);
