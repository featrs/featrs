//! PolynomialFeatures fit and transform over 8 input columns.
//!
//! Full degree 3 is rejected by `fit`, because output names do not encode
//! exponents and `f0^` from `f0^2` collides with `f0^` from `f0^3`. The
//! degree-3 run uses `interaction_only(true)`, the representable degree-3
//! path at 92 output columns. Degree 2 runs with the defaults at 45.

mod common;

use common::{Profile, bench_fit_transform, numeric_frame};
use criterion::{Criterion, criterion_group, criterion_main};
use featrs::prelude::*;

fn polynomial(c: &mut Criterion) {
    let profile = Profile::from_env();
    let rows = profile.rows(10_000);
    let frame = numeric_frame(rows, 8);
    let shape = format!("{rows}x8");

    let mut group = c.benchmark_group("polynomial");
    bench_fit_transform(&mut group, "degree2", &shape, &frame, || {
        PolynomialFeatures::new(2).expect("degree 2 is valid")
    });
    bench_fit_transform(&mut group, "degree3-interactions", &shape, &frame, || {
        PolynomialFeatures::new(3)
            .expect("degree 3 is valid")
            .interaction_only(true)
    });
    group.finish();
}

criterion_group!(benches, polynomial);
criterion_main!(benches);
