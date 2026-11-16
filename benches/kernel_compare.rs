//! Same statistics through the crate's hand loops and through Polars kernels.
//!
//! The hand side mirrors what the scalers do today: two-pass population
//! variance, a full sort plus the linear percentile from
//! `percentile_sorted` in `src/preprocessing/scaler.rs`, and a fold for min
//! and max. Polars picks quickselect for a single quantile, so the median and
//! quantile rows compare different algorithms on purpose. Phase 1 reads the
//! gap between each pair.

mod common;

use common::{Profile, Rng};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use std::hint::black_box;

fn values(count: usize) -> Vec<f64> {
    let mut rng = Rng::new(0xBEEF_0001);
    (0..count).map(|_| 50.0 + 10.0 * rng.normal()).collect()
}

fn hand_var(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / values.len() as f64
}

fn hand_percentile(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = (p / 100.0) * (values.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    let fraction = index - low as f64;
    values[low] * (1.0 - fraction) + values[high] * fraction
}

fn hand_min(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).expect("non-empty")
}

fn hand_max(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).expect("non-empty")
}

fn kernel_compare(c: &mut Criterion) {
    let profile = Profile::from_env();
    let count = profile.rows(1_000_000);
    let values = values(count);
    let series = Series::new("value".into(), values.clone());
    let column = series.f64().expect("Float64 series");

    let mut group = c.benchmark_group("kernel_compare");

    group.bench_with_input(
        BenchmarkId::new("var", "hand"),
        &values,
        |bencher, values| bencher.iter(|| black_box(hand_var(values))),
    );
    group.bench_with_input(
        BenchmarkId::new("var", "polars"),
        &column,
        |bencher, column| bencher.iter(|| black_box(column.var(0))),
    );

    group.bench_with_input(
        BenchmarkId::new("median", "hand"),
        &values,
        |bencher, values| bencher.iter(|| black_box(hand_percentile(values.clone(), 50.0))),
    );
    group.bench_with_input(
        BenchmarkId::new("median", "polars"),
        &column,
        |bencher, column| bencher.iter(|| black_box(column.median())),
    );

    group.bench_with_input(
        BenchmarkId::new("quantile25", "hand"),
        &values,
        |bencher, values| bencher.iter(|| black_box(hand_percentile(values.clone(), 25.0))),
    );
    group.bench_with_input(
        BenchmarkId::new("quantile25", "polars"),
        &column,
        |bencher, column| {
            bencher.iter(|| {
                black_box(
                    column
                        .quantile(0.25, QuantileMethod::Linear)
                        .expect("quantile in range"),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("min", "hand"),
        &values,
        |bencher, values| bencher.iter(|| black_box(hand_min(values))),
    );
    group.bench_with_input(
        BenchmarkId::new("min", "polars"),
        &column,
        |bencher, column| bencher.iter(|| black_box(column.min())),
    );

    group.bench_with_input(
        BenchmarkId::new("max", "hand"),
        &values,
        |bencher, values| bencher.iter(|| black_box(hand_max(values))),
    );
    group.bench_with_input(
        BenchmarkId::new("max", "polars"),
        &column,
        |bencher, column| bencher.iter(|| black_box(column.max())),
    );

    group.finish();
}

criterion_group!(benches, kernel_compare);
criterion_main!(benches);
