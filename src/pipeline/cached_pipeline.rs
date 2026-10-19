//! Caching wrapper for [`Pipeline`].
//!
//! [`CachedPipeline`] wraps a [`Pipeline`] and remembers the output of every
//! step for an input it has already seen, so transforming that same input again
//! replays the stored frames instead of re-running the steps. It pays off where
//! an unchanged early-stage frame goes through the same pipeline many times:
//! debugging a downstream step, repeated evaluation, or a search over the later
//! stages of a pipeline. The reuse only holds when those repeated inputs really
//! are identical — the cache key is the schema and row count, so a *different*
//! frame of the same shape hits an existing entry; see the cache-key note on
//! [`CachedPipeline`].
//!
//! - [`CachedPipeline`] — cache each step's output, bounded to
//!   [`max_cache_entries`](CachedPipeline::max_cache_entries) inputs with FIFO
//!   eviction.
//! - [`CachedPipeline::cache_size`] and [`CachedPipeline::clear_cache`] —
//!   inspect and drop the cache.

use crate::pipeline::{Pipeline, ensure_non_empty_rows};
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// Number of cached inputs a [`CachedPipeline`] starts with.
const DEFAULT_MAX_CACHE_ENTRIES: usize = 16;

/// A [`Pipeline`] that caches the output of each of its steps.
///
/// The first `transform` of an input runs every step and stores each step's
/// output; a later `transform` of an input with the same cache key returns the
/// stored final output without touching the transformers. A cache miss — a
/// different key, or an entry that was evicted — simply recomputes, so the
/// output is always the one the wrapped pipeline would have produced for the
/// input the key was computed from.
///
/// # Cache key
///
/// The key is a fingerprint of the input **schema** — its column names and
/// dtypes — plus its row count, hashed with [`DefaultHasher`]. Reading a schema
/// is cheap, which is what makes the check worth doing on every call.
///
/// The trade-off is that the key says nothing about the **values**: *any* frame
/// with the same column names, dtypes and height — a different cross-validation
/// fold, a fixed-size batch, an edited copy — hits the existing entry and
/// silently returns the output computed from the earlier frame. Only reuse a
/// `CachedPipeline` across calls whose inputs are known to be identical, and
/// call [`clear_cache`](CachedPipeline::clear_cache) before transforming a
/// different frame that shares the schema and height. Hashing the values
/// instead would close the gap at the cost of hashing every input on every
/// call, which is deliberately not done here.
///
/// # Eviction and limits
///
/// At most [`max_cache_entries`](CachedPipeline::max_cache_entries) inputs are
/// remembered (16 by default). Once the cache is full the oldest entry is
/// dropped to make room, so it cannot grow without bound, and an entry is never
/// partially kept. `max_cache_entries(0)` disables caching altogether: nothing
/// is stored, `cache_size` stays `0`, and every transform recomputes.
///
/// Memory is bounded by the *number* of entries, not by their size — cached
/// frames are held in full.
///
/// # Fitted state
///
/// `fit` fits the wrapped pipeline, clears the cache (a re-fit changes what the
/// steps compute, so every stored output is stale) and records that the wrapper
/// is fitted. `transform` before a successful `fit` returns
/// [`Error::NotFitted`]. A 0-row frame is rejected by both phases.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::cached_pipeline::CachedPipeline;
/// use featrs::pipeline::Pipeline;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let inner = Pipeline::new(vec![("scale".into(), Box::new(StandardScaler::new()))])?;
/// let mut cached = CachedPipeline::new(inner);
/// cached.fit(df.clone())?;
///
/// let first = cached.transform(df.clone())?;
/// let second = cached.transform(df)?;
/// assert_eq!(cached.cache_size(), 1);
/// assert!(second.equals_missing(&first)); // replayed, not recomputed
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CachedPipeline {
    /// The wrapped pipeline; `None` only in the [`Default`] value, which has no
    /// pipeline to wrap yet.
    inner: Option<Pipeline>,
    /// Per-input cache: key → the output of every step, in step order.
    ///
    /// [`Transform::transform`] takes `&self`, so recording an entry needs
    /// interior mutability. Borrows are confined to a single look-up or insert
    /// and no transformer runs while the cache is borrowed, so the `RefCell`
    /// cannot panic. Sharing one cache across threads is out of scope, and the
    /// wrapper is neither `Send` nor `Sync` anyway: the boxes in the wrapped
    /// [`Pipeline`] carry no `Send` bound.
    cache: RefCell<Cache>,
    /// Upper bound on cached inputs; `0` disables caching.
    max_cache_entries: usize,
    /// Whether the wrapped pipeline has been fitted.
    fitted: bool,
}

impl CachedPipeline {
    /// Wrap `inner` in a cache with the default limit of 16 inputs.
    pub fn new(inner: Pipeline) -> Self {
        Self {
            inner: Some(inner),
            cache: RefCell::new(Cache::default()),
            max_cache_entries: DEFAULT_MAX_CACHE_ENTRIES,
            fitted: false,
        }
    }

    /// Set how many inputs to remember, evicting the oldest entries first.
    ///
    /// `0` disables the cache: nothing is stored and any existing entries are
    /// dropped.
    pub fn max_cache_entries(mut self, n: usize) -> Self {
        self.max_cache_entries = n;
        self.cache.borrow_mut().truncate_to(n);
        self
    }

    /// Number of inputs currently cached.
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().entries.len()
    }

    /// Drop every cached entry.
    ///
    /// Call this before transforming a different frame that happens to share the
    /// cached key — same column names, dtypes and row count — whether it was
    /// produced by an in-place edit or is simply another frame of that shape:
    /// the key cannot see value changes, so the old entry would keep answering.
    pub fn clear_cache(&mut self) {
        self.cache.borrow_mut().clear();
    }

    /// The wrapped pipeline, or [`Error::InvalidInput`] for a [`Default`] value.
    fn pipeline(&self) -> Result<&Pipeline> {
        self.inner.as_ref().ok_or_else(no_pipeline_error)
    }

    /// Validate that `x` may be transformed and return its cache key.
    fn prepare(&self, x: &DataFrame) -> Result<u64> {
        self.pipeline()?;
        if !self.fitted {
            return Err(Error::NotFitted(
                "CachedPipeline has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        ensure_non_empty_rows(x, "CachedPipeline.transform")?;
        Ok(cache_key(x))
    }

    /// Run every step on `x`, collecting each step's output.
    ///
    /// This walks the wrapped pipeline's steps directly: going through
    /// `Pipeline::transform` would discard the intermediates the cache exists to
    /// store.
    fn run_steps(&self, x: DataFrame) -> Result<Vec<DataFrame>> {
        let steps = self.pipeline()?.steps();
        let mut outputs = Vec::with_capacity(steps.len());
        let mut current = x;
        for (i, (name, step)) in steps.iter().enumerate() {
            current = step
                .transform(current)
                .map_err(|e| wrap_step_error(e, i, name))?;
            outputs.push(current.clone());
        }
        Ok(outputs)
    }
}

impl Default for CachedPipeline {
    /// Create a cache with no wrapped pipeline.
    ///
    /// Like [`FeatureUnion::default`](crate::pipeline::FeatureUnion), the value
    /// is not usable as-is: both `fit` and `transform` reject it until a
    /// pipeline is wrapped with [`CachedPipeline::new`].
    fn default() -> Self {
        Self {
            inner: None,
            cache: RefCell::new(Cache::default()),
            max_cache_entries: DEFAULT_MAX_CACHE_ENTRIES,
            fitted: false,
        }
    }
}

/// Cache key: column names, their dtypes, and the row count.
///
/// Cheap by design — see the type-level docs for the value-change trade-off.
fn cache_key(x: &DataFrame) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.height().hash(&mut hasher);
    for col in x.columns() {
        col.name().hash(&mut hasher);
        // `DataType`'s own `Hash` covers only the variant discriminant, so
        // `Datetime(µs)` and `Datetime(ns)` — or `List(Int64)` and
        // `List(String)` — would share a key. The `Debug` form spells out those
        // parameters, which is what the key needs for everyday columns.
        // `Enum` and `Object` are the exception: their `Debug` is opaque
        // (`Enum([...])`, `Object`), so columns using them share the
        // variant-level key and can only be told apart by the value check that
        // this key deliberately skips.
        format!("{:?}", col.dtype()).hash(&mut hasher);
    }
    hasher.finish()
}

/// Single source of the "no wrapped pipeline" message; [`CachedPipeline::default`]
/// is the only way to reach it.
fn no_pipeline_error() -> Error {
    Error::InvalidInput(
        "CachedPipeline has no wrapped pipeline. Build it with \
         CachedPipeline::new(pipeline); Default only creates the empty cache."
            .into(),
    )
}

fn wrap_step_error(e: Error, i: usize, name: &str) -> Error {
    let msg = format!(
        "CachedPipeline: step {} ('{}') failed during transform: {}",
        i, name, e
    );
    match e {
        Error::NotFitted(_) => Error::NotFitted(msg),
        Error::InvalidInput(_) => Error::InvalidInput(msg),
        Error::Computation(_) => Error::Computation(msg),
    }
}

impl Fit<DataFrame> for CachedPipeline {
    type Output = ();

    /// Fit the wrapped pipeline and start from an empty cache.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first: a failed re-fit must leave neither a stale fitted flag
        // nor entries computed by the previous fit.
        self.fitted = false;
        self.cache.borrow_mut().clear();
        // The missing pipeline is reported before the frame is validated, so a
        // `Default` value reports what is actually wrong with it.
        let inner = self.inner.as_mut().ok_or_else(no_pipeline_error)?;
        ensure_non_empty_rows(&x, "CachedPipeline.fit")?;
        inner.fit(x)?;
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for CachedPipeline {
    type Output = DataFrame;

    /// Transform `x`, replaying a cached input when one matches its key.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let key = self.prepare(&x)?;
        if let Some(final_output) = self.cache.borrow().final_output(key) {
            return Ok(final_output);
        }

        let outputs = self.run_steps(x)?;
        // `Pipeline::new` rejects an empty step list, so the walk is never empty.
        let output = match outputs.last() {
            Some(last) => last.clone(),
            None => {
                return Err(Error::Computation(
                    "CachedPipeline: the wrapped pipeline produced no output.".into(),
                ));
            }
        };
        self.cache
            .borrow_mut()
            .insert(key, outputs, self.max_cache_entries);
        Ok(output)
    }
}

/// Map from cache key to per-step outputs, bounded by a FIFO eviction queue.
#[derive(Default)]
struct Cache {
    entries: HashMap<u64, Vec<DataFrame>>,
    /// Insertion order, oldest first.
    // ponytail: FIFO, matching the issue's v1; swap for an LRU list if hit
    // rates turn out to depend on recency rather than insertion.
    order: VecDeque<u64>,
}

impl Cache {
    /// Final output of the entry for `key`, if it is cached.
    fn final_output(&self, key: u64) -> Option<DataFrame> {
        self.entries
            .get(&key)
            .and_then(|outputs| outputs.last())
            .cloned()
    }

    /// Store one input's step outputs, evicting the oldest entries beyond
    /// `max_entries`. A limit of `0` stores nothing.
    fn insert(&mut self, key: u64, outputs: Vec<DataFrame>, max_entries: usize) {
        if max_entries == 0 {
            return;
        }
        if self.entries.insert(key, outputs).is_none() {
            self.order.push_back(key);
        }
        self.truncate_to(max_entries);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Evict oldest entries until at most `max_entries` remain.
    fn truncate_to(&mut self, max_entries: usize) {
        while self.entries.len() > max_entries {
            match self.order.pop_front() {
                Some(oldest) => {
                    self.entries.remove(&oldest);
                }
                // Unreachable: every entry is queued when it is inserted.
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::DataFrameTransformer;
    use crate::preprocessing::binarizer::Binarizer;
    use crate::preprocessing::scaler::StandardScaler;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts the transforms that reached it, so a test can tell a cache hit
    /// (no extra run) from a recomputation.
    struct CountingStep {
        runs: Arc<AtomicUsize>,
    }

    impl Fit<DataFrame> for CountingStep {
        type Output = ();

        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for CountingStep {
        type Output = DataFrame;

        fn transform(&self, x: DataFrame) -> Result<DataFrame> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            Ok(x)
        }
    }

    fn f64_frame(values: &[f64]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), values));
        DataFrame::new(values.len(), vec![col]).unwrap()
    }

    fn i64_frame(values: &[i64]) -> DataFrame {
        let col = Column::from(Series::new("x".into(), values));
        DataFrame::new(values.len(), vec![col]).unwrap()
    }

    /// A one-step pipeline whose step counts how often it runs.
    fn counting_pipeline() -> (CachedPipeline, Arc<AtomicUsize>) {
        let runs = Arc::new(AtomicUsize::new(0));
        let step = Box::new(CountingStep { runs: runs.clone() }) as Box<dyn DataFrameTransformer>;
        let inner = Pipeline::new(vec![("count".into(), step)]).unwrap();
        (CachedPipeline::new(inner), runs)
    }

    fn values(df: &DataFrame) -> Vec<f64> {
        df.column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    fn assert_frames_equal(a: &DataFrame, b: &DataFrame) {
        assert_eq!(a.height(), b.height(), "row counts differ");
        let (left, right) = (a.columns(), b.columns());
        assert_eq!(left.len(), right.len(), "column counts differ");
        for (l, r) in left.iter().zip(right) {
            assert_eq!(l.name(), r.name(), "column names differ");
            assert!(l.equals_missing(r), "column '{}' differs", l.name());
        }
    }

    #[test]
    fn test_repeated_input_is_served_from_the_cache() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        let first = cp.transform(df.clone()).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 1, "first call must compute");
        assert_eq!(cp.cache_size(), 1);

        let second = cp.transform(df).unwrap();
        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "a repeated input must not re-run the step"
        );
        assert_eq!(cp.cache_size(), 1);
        assert_frames_equal(&first, &second);
    }

    #[test]
    fn test_different_row_count_misses() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        cp.transform(df).unwrap();
        let other = cp.transform(f64_frame(&[1.0, 2.0, 3.0, 4.0])).unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a new height must recompute"
        );
        assert_eq!(cp.cache_size(), 2);
        assert_eq!(values(&other), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_different_dtype_misses() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        cp.transform(df).unwrap();
        cp.transform(i64_frame(&[1, 2, 3])).unwrap();

        assert_eq!(runs.load(Ordering::SeqCst), 2, "a new dtype must recompute");
        assert_eq!(cp.cache_size(), 2);
    }

    #[test]
    fn test_content_change_still_hits_until_the_cache_is_cleared() {
        let (mut cp, runs) = counting_pipeline();
        let first_input = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(first_input.clone()).unwrap();

        let first = cp.transform(first_input).unwrap();
        assert_eq!(values(&first), vec![1.0, 2.0, 3.0]);

        // Same schema and height, different values: the schema fingerprint
        // cannot tell them apart, so this is the documented stale hit.
        let changed = f64_frame(&[4.0, 5.0, 6.0]);
        let stale = cp.transform(changed.clone()).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 1, "key ignored the values");
        assert_eq!(values(&stale), vec![1.0, 2.0, 3.0], "output is the old one");

        // clear_cache is the escape hatch after an in-place change.
        cp.clear_cache();
        assert_eq!(cp.cache_size(), 0);
        let fresh = cp.transform(changed).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert_eq!(values(&fresh), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_clear_cache_between_identical_calls_misses() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        cp.transform(df.clone()).unwrap();
        cp.clear_cache();
        let again = cp.transform(df).unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a cleared cache must recompute"
        );
        assert_eq!(cp.cache_size(), 1);
        assert_eq!(values(&again), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fifo_eviction_drops_the_oldest_entry() {
        let (cp, runs) = counting_pipeline();
        let mut cp = cp.max_cache_entries(2);
        let frames = [
            f64_frame(&[1.0, 2.0, 3.0]),
            f64_frame(&[1.0, 2.0, 3.0, 4.0]),
            f64_frame(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        ];
        cp.fit(frames[0].clone()).unwrap();

        for frame in &frames {
            cp.transform(frame.clone()).unwrap();
        }
        assert_eq!(runs.load(Ordering::SeqCst), 3);
        assert_eq!(cp.cache_size(), 2, "a limit of 2 keeps the newest two");

        // The first input was evicted, so it recomputes.
        cp.transform(frames[0].clone()).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 4, "oldest entry was evicted");
        assert_eq!(cp.cache_size(), 2);
    }

    #[test]
    fn test_zero_max_entries_disables_the_cache() {
        let (cp, runs) = counting_pipeline();
        let mut cp = cp.max_cache_entries(0);
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        let first = cp.transform(df.clone()).unwrap();
        let second = cp.transform(df).unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a disabled cache never hits"
        );
        assert_eq!(cp.cache_size(), 0);
        assert_frames_equal(&first, &second);
    }

    #[test]
    fn test_reducing_the_limit_evicts_existing_entries() {
        let (cp, _runs) = counting_pipeline();
        let mut cp = cp.max_cache_entries(3);
        let frames = [
            f64_frame(&[1.0, 2.0, 3.0]),
            f64_frame(&[1.0, 2.0, 3.0, 4.0]),
            f64_frame(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        ];
        cp.fit(frames[0].clone()).unwrap();
        for frame in &frames {
            cp.transform(frame.clone()).unwrap();
        }
        assert_eq!(cp.cache_size(), 3);

        let cp = cp.max_cache_entries(1);
        assert_eq!(cp.cache_size(), 1, "lowering the limit trims the cache");
    }

    #[test]
    fn test_cache_holds_each_steps_own_output() {
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        let inner = Pipeline::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("binarize".into(), Box::new(Binarizer::new(0.0))),
        ])
        .unwrap();
        let mut cp = CachedPipeline::new(inner);
        cp.fit(df.clone()).unwrap();
        let result = cp.transform(df.clone()).unwrap();

        let cache = cp.cache.borrow();
        let steps = cache.entries.values().next().unwrap();
        assert_eq!(steps.len(), 2, "one cached frame per step");
        assert!(
            !steps[0].equals_missing(&steps[1]),
            "step 1 and step 2 outputs must not be conflated"
        );
        assert!(
            steps[1].equals_missing(&result),
            "the last cached frame is the returned output"
        );

        // Step 1's cached frame is what a single-step scaling pipeline produces.
        let mut scale_only =
            Pipeline::new(vec![("scale".into(), Box::new(StandardScaler::new()))]).unwrap();
        scale_only.fit(df.clone()).unwrap();
        assert!(steps[0].equals_missing(&scale_only.transform(df).unwrap()));

        // Both steps preserved the schema, so the cached frames share it.
        assert_eq!(steps[0].get_column_names(), steps[1].get_column_names());
    }

    #[test]
    fn test_cached_output_matches_the_plain_pipeline() {
        let df = f64_frame(&[1.0, 2.0, 3.0]);

        let mut plain = Pipeline::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("binarize".into(), Box::new(Binarizer::new(0.0))),
        ])
        .unwrap();
        plain.fit(df.clone()).unwrap();
        let expected = plain.transform(df.clone()).unwrap();

        let inner = Pipeline::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("binarize".into(), Box::new(Binarizer::new(0.0))),
        ])
        .unwrap();
        let mut cached = CachedPipeline::new(inner);
        cached.fit(df.clone()).unwrap();

        let first = cached.transform(df.clone()).unwrap();
        let replay = cached.transform(df).unwrap();

        assert_frames_equal(&first, &expected);
        assert_frames_equal(&replay, &expected);
    }

    #[test]
    fn test_single_step_pipeline_caches() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        cp.transform(df.clone()).unwrap();
        cp.transform(df).unwrap();

        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(cp.cache_size(), 1);
    }

    #[test]
    fn test_dtype_parameters_distinguish_the_key() {
        // `DataType::hash` hashes only the variant discriminant, so a key built
        // from it would treat a microsecond and a nanosecond datetime column as
        // the same input.
        let micros = {
            let s = Series::new("ts".into(), &[1_i64, 2, 3])
                .into_datetime(TimeUnit::Microseconds, None);
            DataFrame::new(3, vec![Column::from(s)]).unwrap()
        };
        let nanos = {
            let s =
                Series::new("ts".into(), &[1_i64, 2, 3]).into_datetime(TimeUnit::Nanoseconds, None);
            DataFrame::new(3, vec![Column::from(s)]).unwrap()
        };
        assert_ne!(
            micros.columns()[0].dtype(),
            nanos.columns()[0].dtype(),
            "the two frames must differ in dtype for this test to mean anything"
        );

        let (mut cp, runs) = counting_pipeline();
        cp.fit(micros.clone()).unwrap();
        cp.transform(micros).unwrap();
        cp.transform(nanos).unwrap();

        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a differing dtype parameter must recompute"
        );
        assert_eq!(cp.cache_size(), 2);
    }

    #[test]
    fn test_transform_before_fit_is_not_fitted() {
        let (cp, _runs) = counting_pipeline();
        let err = cp.transform(f64_frame(&[1.0, 2.0, 3.0])).unwrap_err();

        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
        assert_eq!(cp.cache_size(), 0);
    }

    #[test]
    fn test_empty_frame_is_rejected_by_both_phases() {
        let (mut cp, _runs) = counting_pipeline();
        let empty = f64_frame(&[]);

        let err = cp.fit(empty.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");

        cp.fit(f64_frame(&[1.0, 2.0, 3.0])).unwrap();
        let err = cp.transform(empty).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
        assert!(err.to_string().contains("0 rows"), "{err}");
    }

    #[test]
    fn test_failed_refit_resets_fitted_and_clears_the_cache() {
        let (mut cp, runs) = counting_pipeline();
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();
        cp.transform(df.clone()).unwrap();
        assert_eq!(cp.cache_size(), 1);

        let err = cp.fit(f64_frame(&[])).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
        assert_eq!(cp.cache_size(), 0, "a failed fit must not keep entries");

        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_default_has_no_pipeline_to_wrap() {
        let mut cp = CachedPipeline::default();
        let df = f64_frame(&[1.0, 2.0, 3.0]);

        let err = cp.fit(df.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
        assert!(
            err.to_string().contains("no wrapped pipeline"),
            "a Default value must report the missing pipeline, not the frame: {err}"
        );
        let err = cp.transform(df).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
        assert_eq!(cp.cache_size(), 0);
    }

    #[test]
    fn test_multi_step_pipeline_caches() {
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        let inner = Pipeline::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("binarize".into(), Box::new(Binarizer::new(0.0))),
        ])
        .unwrap();
        let mut cp = CachedPipeline::new(inner);
        cp.fit(df.clone()).unwrap();

        let first = cp.transform(df.clone()).unwrap();
        let replay = cp.transform(df).unwrap();

        assert_eq!(cp.cache_size(), 1, "an N-step pipeline caches one input");
        assert_frames_equal(&first, &replay);
    }

    /// Fails at transform time, so a step error can be exercised.
    struct FailingStep;

    impl Fit<DataFrame> for FailingStep {
        type Output = ();

        fn fit(&mut self, _x: DataFrame) -> Result<()> {
            Ok(())
        }
    }

    impl Transform<DataFrame> for FailingStep {
        type Output = DataFrame;

        fn transform(&self, _x: DataFrame) -> Result<DataFrame> {
            Err(Error::Computation("boom".into()))
        }
    }

    #[test]
    fn test_step_errors_name_the_failing_step() {
        let inner = Pipeline::new(vec![
            ("scale".into(), Box::new(StandardScaler::new())),
            ("boom".into(), Box::new(FailingStep)),
        ])
        .unwrap();
        let mut cp = CachedPipeline::new(inner);
        let df = f64_frame(&[1.0, 2.0, 3.0]);
        cp.fit(df.clone()).unwrap();

        let err = cp.transform(df).unwrap_err();
        assert!(
            matches!(err, Error::Computation(_)),
            "expected Computation, got {err:?}"
        );
        assert!(
            err.to_string().contains("CachedPipeline: step 1 ('boom')"),
            "{err}"
        );
        assert_eq!(cp.cache_size(), 0, "a failed step must not be cached");
    }
}
