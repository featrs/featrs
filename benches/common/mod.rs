//! Shared frames and the fit/transform harness for the bench suites.
//!
//! `FEATRS_BENCH_PROFILE` sets the sizes. The default `ci` profile divides
//! every row count by ten. The `full` profile uses the sizes from issue #21.
//! Bench names carry the shape, so a CI run never compares against a
//! full-size baseline.
//!
//! Every timed closure clones the input frame first. The clone costs a few
//! pointer bumps against work measured in milliseconds, and it keeps
//! `transform` unchanged across iterations.

#![allow(dead_code)]

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId};
use featrs::traits::{Fit, Transform};
use polars::prelude::*;
use std::hint::black_box;

/// Row sizes for one bench run.
#[derive(Clone, Copy)]
pub enum Profile {
    /// Default. One tenth of the full row counts, for shared machines.
    Ci,
    /// The sizes documented in issue #21, for local runs.
    Full,
}

impl Profile {
    /// Read `FEATRS_BENCH_PROFILE`.
    ///
    /// An unknown value aborts the run, so a typo cannot mix shapes into an
    /// existing baseline.
    pub fn from_env() -> Self {
        match std::env::var("FEATRS_BENCH_PROFILE") {
            Ok(value) if value == "full" => Self::Full,
            Ok(value) if value == "ci" => Self::Ci,
            Ok(value) => panic!("FEATRS_BENCH_PROFILE must be 'ci' or 'full', got '{value}'"),
            Err(std::env::VarError::NotPresent) => Self::Ci,
            Err(error) => panic!("FEATRS_BENCH_PROFILE is not valid unicode: {error}"),
        }
    }

    /// Rows to use for a suite whose full size is `rows`.
    pub fn rows(self, rows: usize) -> usize {
        match self {
            Self::Full => rows,
            Self::Ci => (rows / 10).max(2),
        }
    }
}

/// xorshift64* generator. Fixed seeds keep every run's frames identical.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform value in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Roughly standard normal, by summing twelve uniform values.
    pub fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }

    /// Uniform index in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// `rows` x `cols` Float64 frame of roughly standard-normal values.
pub fn numeric_frame(rows: usize, cols: usize) -> DataFrame {
    let mut rng = Rng::new(0x5EED_1234);
    let columns: Vec<Column> = (0..cols)
        .map(|index| {
            let values: Vec<f64> = (0..rows).map(|_| 50.0 + 10.0 * rng.normal()).collect();
            Column::from(Series::new(format!("f{index}").into(), values))
        })
        .collect();
    DataFrame::new(rows, columns).expect("frame height matches its columns")
}

/// One String column cycling through `categories` values, `rows` rows.
pub fn categorical_frame(rows: usize, categories: usize) -> DataFrame {
    let mut rng = Rng::new(0xCAFE_0001);
    let values: Vec<String> = (0..rows)
        .map(|_| format!("cat_{:02}", rng.below(categories)))
        .collect();
    let column = Column::from(Series::new("category".into(), values));
    DataFrame::new(rows, vec![column]).expect("frame height matches its column")
}

/// `rows` x `cols` String frame with 500 distinct values per column.
pub fn text_frame(rows: usize, cols: usize) -> DataFrame {
    let mut rng = Rng::new(0xFACE_0001);
    let columns: Vec<Column> = (0..cols)
        .map(|index| {
            let values: Vec<String> = (0..rows)
                .map(|_| format!("value_{:03}", rng.below(500)))
                .collect();
            Column::from(Series::new(format!("c{index}").into(), values))
        })
        .collect();
    DataFrame::new(rows, columns).expect("frame height matches its columns")
}

/// Bench `fit` and `transform` for a transformer that `make` builds fresh.
///
/// `fit` runs on one instance and `transform` on a second, already fitted
/// instance, so the transform numbers exclude the fit cost.
pub fn bench_fit_transform<T, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    shape: &str,
    frame: &DataFrame,
    make: F,
) where
    T: Fit<DataFrame, Output = ()> + Transform<DataFrame, Output = DataFrame>,
    F: Fn() -> T,
{
    let mut fitter = make();
    group.bench_with_input(
        BenchmarkId::new(format!("{label}/fit"), shape),
        frame,
        |bencher, frame| bencher.iter(|| fitter.fit(frame.clone()).expect("fit")),
    );

    let mut fitted = make();
    fitted.fit(frame.clone()).expect("fit");
    group.bench_with_input(
        BenchmarkId::new(format!("{label}/transform"), shape),
        frame,
        |bencher, frame| {
            bencher.iter(|| black_box(fitted.transform(frame.clone()).expect("transform")))
        },
    );
}
