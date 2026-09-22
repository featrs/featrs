//! Country-specific holiday indicator features.
//!
//! [`HolidayEncoder`] marks whether each row's date falls on a statutory
//! holiday of a chosen country and appends a binary `Float64` column
//! (`1.0` for a holiday, `0.0` otherwise) named `{column}_is_holiday` for every
//! configured `Date`/`Datetime` column. Time-series models (sales, traffic,
//! energy demand) can then pick holiday-driven seasonality up directly.
//!
//! # How the calendars are built
//!
//! Calendars are computed from explicit per-country **rules** in pure Rust —
//! no bundled data file, no new dependency, no network access — so they are
//! deterministic, reviewable, and valid for any year:
//!
//! - **fixed** dates (e.g. US 07-04 Independence Day, IN 08-15 Independence Day);
//! - **nth-weekday-of-month** dates, including the last such weekday
//!   (e.g. US Thanksgiving = 4th Thursday of November, GB Summer bank holiday =
//!   last Monday of August, JP Coming of Age Day = 2nd Monday of January);
//! - **Easter-relative** dates from the anonymous Gregorian computus
//!   (Meeus/Jones/Butcher), e.g. Good Friday = Easter − 2 days, Ascension =
//!   Easter + 39 days.
//!
//! Every holiday that is not derivable from one of those three rule kinds is
//! deliberately *not* claimed: lunar festivals (Diwali, Holi, Eid, Dussehra)
//! and Japan's astronomical equinox days have no closed-form Gregorian date and
//! are omitted rather than guessed. See [`HolidayEncoder`] for the exact
//! per-country coverage and exclusions.
//!
//! # Example
//!
//! ```rust
//! use featrs::preprocessing::holiday_encoder::{HolidayCountry, HolidayEncoder};
//! use featrs::traits::{Fit, Transform};
//! use polars::prelude::{Column, DataFrame, DataType, NamedFrom, Series};
//!
//! // 2024-01-01 (New Year's Day) and 2024-01-02, as days since the Unix epoch.
//! let s = Series::new("d".into(), &[Some(19_723i32), Some(19_724)])
//!     .cast(&DataType::Date)?;
//! let df = DataFrame::new(2, vec![Column::from(s)])?;
//!
//! let mut encoder = HolidayEncoder::new()
//!     .columns(&["d"])
//!     .country(HolidayCountry::US);
//! encoder.fit(df.clone())?;
//! let out = encoder.transform(df)?;
//! assert_eq!(out.column("d_is_holiday")?.f64()?.get(0), Some(1.0));
//! assert_eq!(out.column("d_is_holiday")?.f64()?.get(1), Some(0.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{HashMap, HashSet};

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// A country whose statutory holiday calendar is supported.
///
/// The variants are the ISO 3166-1 alpha-2 country codes named by the issue
/// this encoder was specified from, so an unknown country code is
/// unrepresentable — there is no runtime "invalid country" check to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HolidayCountry {
    /// United States: the eleven federal holidays.
    US,
    /// United Kingdom: the bank holidays of England and Wales.
    GB,
    /// Germany: the nine nationwide (bundesweit) holidays.
    DE,
    /// France: the eleven national (jours fériés) holidays.
    FR,
    /// Japan: the sixteen national holidays of the current era (not all of
    /// which are Gregorian-derivable — see [`HolidayEncoder`]).
    JP,
    /// India: five central-government gazetted holidays.
    IN,
}

/// Compute holiday indicators for one country's statutory calendar.
///
/// Each configured column produces one appended `Float64` column named
/// `{column}_is_holiday` holding `1.0` when the row's date is a holiday of
/// [`country`](HolidayEncoder::country) and `0.0` otherwise. Input columns are
/// preserved; new columns are appended.
///
/// # Calendar coverage
///
/// Nationwide/statutory holidays only. Regional and devolved calendars are out
/// of scope: German state holidays (Reformation Day, Epiphany, Corpus Christi,
/// International Women's Day in Berlin, …), the Scottish/Northern Irish bank
/// holidays, French Alsace-Moselle (which is the only French locale where Good
/// Friday is a holiday) and Indian state holidays are all **not** included.
///
/// | Country | Holidays | Notes |
/// |---------|----------|-------|
/// | [`US`](HolidayCountry::US) | New Year's Day; MLK Jr. Day (3rd Mon Jan, since 1986); Washington's Birthday (3rd Mon Feb); Memorial Day (last Mon May); Juneteenth (Jun 19, since 2021); Independence Day (Jul 4); Labor Day (1st Mon Sep); Columbus Day (2nd Mon Oct); Veterans Day (Nov 11); Thanksgiving (4th Thu Nov); Christmas (Dec 25) | US federal holidays. |
/// | [`GB`](HolidayCountry::GB) | New Year's Day (since 1974); Good Friday; Easter Monday; Early May bank holiday (1st Mon May, since 1978); Spring bank holiday (last Mon May); Summer bank holiday (last Mon Aug); Christmas (Dec 25); Boxing Day (Dec 26) | England and Wales. |
/// | [`DE`](HolidayCountry::DE) | Neujahr (Jan 1); Karfreitag; Ostermontag; Tag der Arbeit (May 1); Christi Himmelfahrt (Easter + 39); Pfingstmontag (Easter + 50); Tag der Deutschen Einheit (Oct 3, since 1990); 1./2. Weihnachtstag (Dec 25/26) | Nationwide only. |
/// | [`FR`](HolidayCountry::FR) | Jour de l'An (Jan 1); Lundi de Pâques; Fête du Travail (May 1); Victoire 1945 (May 8); Ascension (Easter + 39); Lundi de Pentecôte (Easter + 50); Fête Nationale (Jul 14); Assomption (Aug 15); Toussaint (Nov 1); Armistice (Nov 11); Noël (Dec 25) | Good Friday is *not* a French holiday. |
/// | [`JP`](HolidayCountry::JP) | 元日 (Jan 1); Coming of Age Day (2nd Mon Jan); National Foundation Day (Feb 11); Emperor's Birthday (Feb 23 since 2020, Dec 23 1989–2018); Showa Day (Apr 29); Constitution Memorial Day (May 3); Greenery Day (May 4); Children's Day (May 5); Marine Day (3rd Mon Jul); Mountain Day (Aug 11, since 2016); Respect for the Aged Day (3rd Mon Sep); Sports Day (2nd Mon Oct); Culture Day (Nov 3); Labor Thanksgiving Day (Nov 23) | Current era. |
/// | [`IN`](HolidayCountry::IN) | Republic Day (Jan 26); Good Friday; Independence Day (Aug 15); Gandhi Jayanti (Oct 2); Christmas (Dec 25) | Central-government gazetted. |
///
/// # Exclusions and known limits
///
/// - **Lunar and astronomical holidays are omitted entirely** — they cannot be
///   derived from Gregorian rules and are not guessed. That covers Diwali,
///   Holi, Eid, Dussehra and Guru Nanak Jayanti (India) and Vernal/Autumnal
///   Equinox Day (Japan, astronomical).
/// - **No observed/substitute days.** A holiday is marked on its own calendar
///   day only: when Christmas falls on a Saturday the 25th is `1.0` and the
///   substituted day off (Dec 24 in the US, Dec 27 in the UK) is `0.0`. Japan's
///   substitute holidays for Sunday holidays and its Jan 2/3 New Year bank
///   holidays are likewise not modeled.
/// - **Modern-era calendars.** Rules are applied to every year unless a rule
///   itself carries a start/end year. Where a holiday *date* was moved by an
///   earlier reform — US Memorial Day/Columbus Day/Washington's Birthday,
///   Japan's Coming of Age, Marine, Respect for the Aged and Sports Days — the
///   current-era rule is used for all years, so pre-reform years are
///   approximate. The `Date` dtype itself starts at 1970-01-01, which bounds
///   how far back this matters in practice.
/// - **One-off government moves are not modeled.** The GB Early May bank
///   holiday was moved to 2020-05-08 for the 75th anniversary of VE Day, and
///   GB had one-off holidays on 2022-06-02/06-03 (Platinum Jubilee),
///   2022-09-19 (State Funeral of Queen Elizabeth II) and 2023-05-08
///   (Coronation); Japan moved Marine Day and Sports Day for the 2020/2021
///   Olympics and had extra one-off holidays around the 2019 enthronement.
///   Those single-year exceptions are not represented — the rules give the
///   ordinary date. The 2019 gap in Japan's Emperor's Birthday *is* handled, by
///   the 1989–2018 / since-2020 bounds on the two era-specific rules.
/// - **Timezones.** A `Datetime` column is truncated to the calendar day of its
///   stored instant; timezone metadata is ignored (an instant is treated as
///   UTC-naive), so a timezone-aware column must already carry the instants
///   you want the day boundary drawn at.
/// - **Holidays on a weekend are still marked** as holidays — this is separate
///   from [`DatetimeFeatures`](crate::preprocessing::datetime_features::DatetimeFeatures)
///   `IsWeekend`, and the two can both be `1.0` on the same row.
/// - **Bridging holidays** are marked on the whole calendar day (a holiday's
///   midnight-spanning period is not split across two days).
///
/// # Fitted state and out-of-range years
///
/// `fit` discovers the first and last year present in the configured columns,
/// materialises that year span's holiday days into a `HashSet<i32>` of days
/// since the Unix epoch (the fast path), and records the span's day bounds.
/// Because [`Transform::transform`] takes `&self`, a row whose date falls
/// outside the fitted span is answered by evaluating the pure rules for that
/// row's own year instead of silently reporting `0.0`, so transforming a later
/// (or earlier) frame with a fitted encoder still yields correct indicators.
/// A fitted span wider than a thousand years (only reachable with corrupt
/// `Date` values) is not cached at all, so every row takes that rule path.
///
/// # Name collisions
///
/// `DataFrame::with_column` silently replaces a same-named column, so a
/// generated name that matches an input column (or another generated name)
/// makes both `fit` and `transform` return [`Error::InvalidInput`].
#[derive(Debug)]
pub struct HolidayEncoder {
    /// Whether `fit` completed; `transform` refuses to run when `false`.
    fitted: bool,
    /// The holiday calendar to apply.
    country: HolidayCountry,
    /// The user-supplied column configuration (`None` = auto-discover).
    column_config: Option<Vec<String>>,
    /// Resolved at fit time: the deduplicated columns to encode.
    columns: Vec<String>,
    /// Holiday days (days since the Unix epoch) covering `day_range`.
    holiday_days: HashSet<i32>,
    /// Inclusive day bounds of the fitted data, or `None` when nothing was
    /// cached (every configured column was all-null, or the span was wider than
    /// [`MAX_CACHED_YEARS`]).
    day_range: Option<(i32, i32)>,
}

impl HolidayEncoder {
    /// Create a new `HolidayEncoder` with auto-discovered columns and the
    /// [`US`](HolidayCountry::US) calendar.
    pub fn new() -> Self {
        Self {
            fitted: false,
            country: HolidayCountry::US,
            column_config: None,
            columns: vec![],
            holiday_days: HashSet::new(),
            day_range: None,
        }
    }

    /// Restrict the transformation to the named columns.
    ///
    /// When omitted (or passed an empty list), all `Date`/`Datetime` columns
    /// present at fit time are auto-discovered on every fit. Each column must
    /// exist at fit time with dtype `Date` or `Datetime`; otherwise `fit`
    /// returns [`Error::InvalidInput`]. Duplicates are collapsed, so
    /// `.columns(&["a", "a"])` emits a single `a_is_holiday` column.
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.column_config = Some(cols.iter().map(|s| s.to_string()).collect());
        self.fitted = false;
        self
    }

    /// Set the holiday calendar to apply (default:
    /// [`US`](HolidayCountry::US)).
    pub fn country(mut self, c: HolidayCountry) -> Self {
        self.country = c;
        self.fitted = false;
        self
    }
}

impl Default for HolidayEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// True for dtypes this transformer operates on.
fn is_datetime_dtype(dt: &DataType) -> bool {
    matches!(dt, DataType::Date | DataType::Datetime(..))
}

/// Deduplicate while preserving first-seen order.
fn dedup_preserve_order<T: Clone + Eq + std::hash::Hash>(items: &[T]) -> Vec<T> {
    let mut seen = HashSet::new();
    items.iter().filter(|i| seen.insert(*i)).cloned().collect()
}

/// Days since the Unix epoch (`1970-01-01`) for a proleptic-Gregorian civil
/// date.
///
/// Howard Hinnant's `days_from_civil`, integer-only and exact for the whole
/// `i32` day range. The intermediates are computed in `i64` because a corrupt
/// `Date` value can describe a year ~5.9 million, whose `era * 146_097` product
/// would overflow `i32`.
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = i64::from(y) - i64::from(m <= 2);
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u32;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + i64::from(doe) - 719_468) as i32
}

/// The inverse of [`days_from_civil`]: the `(year, month, day)` a
/// days-since-epoch value falls on.
fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = i64::from(z) + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    ((if m <= 2 { y + 1 } else { y }) as i32, m, d)
}

/// Monday-based weekday of a days-since-epoch value (`0` = Monday).
///
/// 1970-01-01 was a Thursday, which is index `3` in a Monday-based week.
fn weekday_from_days(z: i32) -> u32 {
    (i64::from(z) + 3).rem_euclid(7) as u32
}

/// True for a proleptic-Gregorian leap year.
fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Widest year span materialised into the cached holiday-day set at fit time.
///
/// A `Date` column is an `i32` day count, so a corrupt one can span ~±5.9
/// million years; caching that would allocate an enormous set. Beyond this
/// width the encoder skips the cache and answers every row from the rules,
/// which is slower per row but still correct.
const MAX_CACHED_YEARS: i32 = 1_000;

/// Days in `month` of `year` (`0` for an out-of-range month).
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since the epoch of the `n`-th `weekday` (`0` = Monday) of `month`.
fn nth_weekday_days(year: i32, month: u32, weekday: u32, n: u32) -> i32 {
    let first = days_from_civil(year, month, 1);
    let offset = (weekday + 7 - weekday_from_days(first)) % 7;
    (i64::from(first) + i64::from(offset) + (i64::from(n) - 1) * 7) as i32
}

/// Days since the epoch of the last `weekday` (`0` = Monday) of `month`.
fn last_weekday_days(year: i32, month: u32, weekday: u32) -> i32 {
    let first = days_from_civil(year, month, 1);
    let offset = (weekday + 7 - weekday_from_days(first)) % 7;
    let last = i64::from(first) + i64::from(offset) + 28;
    let month_end = i64::from(days_from_civil(year, month, days_in_month(year, month)));
    if last > month_end {
        (last - 7) as i32
    } else {
        last as i32
    }
}

/// Easter Sunday (Gregorian computus, Meeus/Jones/Butcher), as days since the
/// epoch. Exact for 1583 and later.
fn easter_days(year: i32) -> i32 {
    let a = year.rem_euclid(19);
    let b = year.div_euclid(100);
    let c = year.rem_euclid(100);
    let d = b.div_euclid(4);
    let e = b.rem_euclid(4);
    let f = (b + 8).div_euclid(25);
    let g = (b - f + 1).div_euclid(3);
    let h = (19 * a + b - d - g + 15).rem_euclid(30);
    let i = c.div_euclid(4);
    let k = c.rem_euclid(4);
    let l = (32 + 2 * e + 2 * i - h - k).rem_euclid(7);
    let m = (a + 11 * h + 22 * l).div_euclid(451);
    let month = (h + l - 7 * m + 114).div_euclid(31) as u32;
    let day = ((h + l - 7 * m + 114).rem_euclid(31) + 1) as u32;
    days_from_civil(year, month, day)
}

/// How a rule picks its date within a year.
#[derive(Debug, Clone, Copy)]
enum RuleKind {
    /// A fixed `month`/`day` in every year.
    Fixed { month: u32, day: u32 },
    /// An nth occurrence of a weekday in a month.
    NthWeekday {
        month: u32,
        /// `0` = Monday … `6` = Sunday.
        weekday: u32,
        nth: Nth,
    },
    /// `days` after Easter Sunday (negative for days before it).
    EasterOffset { days: i32 },
}

/// Which occurrence of a weekday a [`RuleKind::NthWeekday`] matches.
#[derive(Debug, Clone, Copy)]
enum Nth {
    /// The 1st, 2nd, … occurrence counted from the start of the month.
    FromStart(u32),
    /// The final occurrence in the month.
    Last,
}

/// A holiday rule plus the years over which it applies.
#[derive(Debug, Clone, Copy)]
struct HolidayRule {
    kind: RuleKind,
    /// First year (inclusive) the holiday existed, `None` = all years.
    since: Option<i32>,
    /// Last year (inclusive) the holiday existed, `None` = ongoing.
    until: Option<i32>,
}

impl HolidayRule {
    /// A rule valid in every year.
    const fn always(kind: RuleKind) -> Self {
        Self {
            kind,
            since: None,
            until: None,
        }
    }

    /// A rule that only starts in `year`.
    const fn since(kind: RuleKind, year: i32) -> Self {
        Self {
            kind,
            since: Some(year),
            until: None,
        }
    }

    /// A rule that applies from `from` through `to`, inclusive.
    const fn between(kind: RuleKind, from: i32, to: i32) -> Self {
        Self {
            kind,
            since: Some(from),
            until: Some(to),
        }
    }

    /// True when this rule applies to `year`.
    fn applies(self, year: i32) -> bool {
        self.since.is_none_or(|s| year >= s) && self.until.is_none_or(|u| year <= u)
    }

    /// The days-since-epoch value this rule resolves to in `year`.
    fn resolve(self, year: i32) -> i32 {
        match self.kind {
            RuleKind::Fixed { month, day } => days_from_civil(year, month, day),
            RuleKind::NthWeekday {
                month,
                weekday,
                nth,
            } => match nth {
                Nth::FromStart(n) => nth_weekday_days(year, month, weekday, n),
                Nth::Last => last_weekday_days(year, month, weekday),
            },
            RuleKind::EasterOffset { days } => easter_days(year) + days,
        }
    }
}

/// A fixed date in every year.
const fn fixed(month: u32, day: u32) -> RuleKind {
    RuleKind::Fixed { month, day }
}

/// The `n`-th `weekday` of `month` (`0` = Monday).
const fn nth(month: u32, weekday: u32, n: u32) -> RuleKind {
    RuleKind::NthWeekday {
        month,
        weekday,
        nth: Nth::FromStart(n),
    }
}

/// The last `weekday` of `month` (`0` = Monday).
const fn last(month: u32, weekday: u32) -> RuleKind {
    RuleKind::NthWeekday {
        month,
        weekday,
        nth: Nth::Last,
    }
}

/// A date `days` after Easter Sunday.
const fn easter(days: i32) -> RuleKind {
    RuleKind::EasterOffset { days }
}

/// US federal holidays.
const US_RULES: &[HolidayRule] = &[
    HolidayRule::always(fixed(1, 1)),       // New Year's Day
    HolidayRule::since(nth(1, 0, 3), 1986), // Martin Luther King Jr. Day
    HolidayRule::always(nth(2, 0, 3)),      // Washington's Birthday
    HolidayRule::always(last(5, 0)),        // Memorial Day
    HolidayRule::since(fixed(6, 19), 2021), // Juneteenth
    HolidayRule::always(fixed(7, 4)),       // Independence Day
    HolidayRule::always(nth(9, 0, 1)),      // Labor Day
    HolidayRule::always(nth(10, 0, 2)),     // Columbus Day
    HolidayRule::always(fixed(11, 11)),     // Veterans Day
    HolidayRule::always(nth(11, 3, 4)),     // Thanksgiving
    HolidayRule::always(fixed(12, 25)),     // Christmas Day
];

/// Bank holidays of England and Wales.
const GB_RULES: &[HolidayRule] = &[
    HolidayRule::since(fixed(1, 1), 1974),  // New Year's Day
    HolidayRule::always(easter(-2)),        // Good Friday
    HolidayRule::always(easter(1)),         // Easter Monday
    HolidayRule::since(nth(5, 0, 1), 1978), // Early May bank holiday
    HolidayRule::always(last(5, 0)),        // Spring bank holiday
    HolidayRule::always(last(8, 0)),        // Summer bank holiday
    HolidayRule::always(fixed(12, 25)),     // Christmas Day
    HolidayRule::always(fixed(12, 26)),     // Boxing Day
];

/// Nationwide German holidays.
const DE_RULES: &[HolidayRule] = &[
    HolidayRule::always(fixed(1, 1)),       // Neujahr
    HolidayRule::always(easter(-2)),        // Karfreitag
    HolidayRule::always(easter(1)),         // Ostermontag
    HolidayRule::always(fixed(5, 1)),       // Tag der Arbeit
    HolidayRule::always(easter(39)),        // Christi Himmelfahrt
    HolidayRule::always(easter(50)),        // Pfingstmontag
    HolidayRule::since(fixed(10, 3), 1990), // Tag der Deutschen Einheit
    HolidayRule::always(fixed(12, 25)),     // 1. Weihnachtstag
    HolidayRule::always(fixed(12, 26)),     // 2. Weihnachtstag
];

/// French national holidays.
const FR_RULES: &[HolidayRule] = &[
    HolidayRule::always(fixed(1, 1)),   // Jour de l'An
    HolidayRule::always(easter(1)),     // Lundi de Pâques
    HolidayRule::always(fixed(5, 1)),   // Fête du Travail
    HolidayRule::always(fixed(5, 8)),   // Victoire 1945
    HolidayRule::always(easter(39)),    // Ascension
    HolidayRule::always(easter(50)),    // Lundi de Pentecôte
    HolidayRule::always(fixed(7, 14)),  // Fête Nationale
    HolidayRule::always(fixed(8, 15)),  // Assomption
    HolidayRule::always(fixed(11, 1)),  // Toussaint
    HolidayRule::always(fixed(11, 11)), // Armistice 1918
    HolidayRule::always(fixed(12, 25)), // Noël
];

/// Japanese national holidays of the current era.
const JP_RULES: &[HolidayRule] = &[
    HolidayRule::always(fixed(1, 1)),                // 元日
    HolidayRule::always(nth(1, 0, 2)),               // Coming of Age Day
    HolidayRule::always(fixed(2, 11)),               // National Foundation Day
    HolidayRule::since(fixed(2, 23), 2020),          // Emperor's Birthday (Reiwa)
    HolidayRule::always(fixed(4, 29)),               // Showa Day
    HolidayRule::always(fixed(5, 3)),                // Constitution Memorial Day
    HolidayRule::always(fixed(5, 4)),                // Greenery Day
    HolidayRule::always(fixed(5, 5)),                // Children's Day
    HolidayRule::always(nth(7, 0, 3)),               // Marine Day
    HolidayRule::since(fixed(8, 11), 2016),          // Mountain Day
    HolidayRule::always(nth(9, 0, 3)),               // Respect for the Aged Day
    HolidayRule::always(nth(10, 0, 2)),              // Sports Day
    HolidayRule::always(fixed(11, 3)),               // Culture Day
    HolidayRule::always(fixed(11, 23)),              // Labor Thanksgiving Day
    HolidayRule::between(fixed(12, 23), 1989, 2018), // Emperor's Birthday (Heisei)
];

/// Central-government gazetted holidays of India.
const IN_RULES: &[HolidayRule] = &[
    HolidayRule::always(fixed(1, 26)),  // Republic Day
    HolidayRule::always(easter(-2)),    // Good Friday
    HolidayRule::always(fixed(8, 15)),  // Independence Day
    HolidayRule::always(fixed(10, 2)),  // Gandhi Jayanti
    HolidayRule::always(fixed(12, 25)), // Christmas Day
];

/// The rule table for `country`.
fn rules(country: HolidayCountry) -> &'static [HolidayRule] {
    match country {
        HolidayCountry::US => US_RULES,
        HolidayCountry::GB => GB_RULES,
        HolidayCountry::DE => DE_RULES,
        HolidayCountry::FR => FR_RULES,
        HolidayCountry::JP => JP_RULES,
        HolidayCountry::IN => IN_RULES,
    }
}

/// Every holiday of `country` in `year`, as days since the epoch.
///
/// All three rule kinds resolve within `year`, so one year's rules are enough
/// to answer a lookup for any day of that year.
fn holidays_in_year(country: HolidayCountry, year: i32) -> Vec<i32> {
    rules(country)
        .iter()
        .filter(|rule| rule.applies(year))
        .map(|rule| rule.resolve(year))
        .collect()
}

/// A column of days since the Unix epoch, with nulls preserved.
///
/// `Date` values are already day counts; `Datetime` values are truncated to
/// their calendar day (dropping the time and timezone portion).
fn day_values(s: &Series, col: &str, context: &str) -> Result<Int32Chunked> {
    let c_err = |e: PolarsError| Error::Computation(format!("{context}: column '{col}': {e}"));
    let as_date = s.cast(&DataType::Date).map_err(c_err)?;
    Ok(as_date.to_physical_repr().i32().map_err(c_err)?.clone())
}

impl Fit<DataFrame> for HolidayEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        // Reset first so a failed re-fit cannot leave stale fitted state, and
        // drop previously resolved columns so auto-discovery re-runs on every
        // fit instead of reusing the previous schema's columns.
        self.fitted = false;
        self.columns = vec![];
        self.holiday_days = HashSet::new();
        self.day_range = None;

        if x.width() == 0 || x.height() == 0 {
            return Err(Error::InvalidInput(
                "HolidayEncoder.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide at least one row and one column."
                    .into(),
            ));
        }

        match self.column_config.as_deref() {
            // Auto-discovery: no explicit configuration (or an empty list).
            None | Some([]) => {
                let discovered: Vec<String> = x
                    .get_column_names()
                    .iter()
                    .filter(|n| {
                        x.column(n)
                            .map(|c| is_datetime_dtype(c.dtype()))
                            .unwrap_or(false)
                    })
                    .map(|n| n.to_string())
                    .collect();
                if discovered.is_empty() {
                    let all_types: Vec<String> = x
                        .get_column_names()
                        .iter()
                        .filter_map(|n| x.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
                        .collect();
                    return Err(Error::InvalidInput(format!(
                        "HolidayEncoder: no Date or Datetime columns found. This transformer only \
                         operates on Date/Datetime columns. Available columns: [{}]. Cast non-date \
                         columns before fitting.",
                        all_types.join(", ")
                    )));
                }
                self.columns = discovered;
            }
            Some(cfg) => {
                for col in cfg {
                    let c = x.column(col.as_str()).map_err(|e| {
                        Error::InvalidInput(format!(
                            "HolidayEncoder.fit: column '{col}' not found. {e}"
                        ))
                    })?;
                    if !is_datetime_dtype(c.dtype()) {
                        return Err(Error::InvalidInput(format!(
                            "HolidayEncoder.fit: column '{col}' has dtype {}; expected Date or \
                             Datetime.",
                            c.dtype()
                        )));
                    }
                }
                self.columns = dedup_preserve_order(cfg);
            }
        }

        // Reject name collisions up front: `with_column` silently replaces a
        // same-named column, so a generated name matching an input column (or
        // another generated name) would silently overwrite data.
        let mut seen: HashSet<String> = x
            .get_column_names()
            .iter()
            .map(|n| n.as_str().to_string())
            .collect();
        for col in &self.columns {
            let out_name = format!("{col}_is_holiday");
            if !seen.insert(out_name.clone()) {
                return Err(Error::InvalidInput(format!(
                    "HolidayEncoder: generated column '{out_name}' collides with an existing \
                     input column or another generated column. Rename the conflicting input \
                     column."
                )));
            }
        }

        // Scope the holiday computation to the years actually present.
        let mut bounds: Option<(i32, i32)> = None;
        for col in &self.columns {
            let s = x
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "HolidayEncoder.fit: column '{col}' not found. {e}"
                    ))
                })?
                .as_materialized_series();
            let days = day_values(s, col, "HolidayEncoder.fit")?;
            for day in days.iter().flatten() {
                bounds = Some(match bounds {
                    Some((lo, hi)) => (lo.min(day), hi.max(day)),
                    None => (day, day),
                });
            }
        }

        if let Some((lo, hi)) = bounds {
            let (first_year, _, _) = civil_from_days(lo);
            let (last_year, _, _) = civil_from_days(hi);
            // Only cache a span of ordinary width; see `MAX_CACHED_YEARS`.
            if last_year - first_year <= MAX_CACHED_YEARS {
                let mut holiday_days = HashSet::new();
                for year in first_year..=last_year {
                    holiday_days.extend(holidays_in_year(self.country, year));
                }
                self.holiday_days = holiday_days;
                self.day_range = Some((lo, hi));
            }
        }

        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for HolidayEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "HolidayEncoder has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let mut out = x.clone();

        // The transform input may contain columns absent at fit time; guard
        // against silently overwriting them, mirroring the fit-time check.
        for col in &self.columns {
            let out_name = format!("{col}_is_holiday");
            if out.column(out_name.as_str()).is_ok() {
                return Err(Error::InvalidInput(format!(
                    "HolidayEncoder.transform: input already contains column '{out_name}', which \
                     would be overwritten by a generated feature. Rename the conflicting column."
                )));
            }
        }

        for col in &self.columns {
            let s = out
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "HolidayEncoder.transform: column '{col}' not found. The transformer was \
                         fitted on columns: {:?}. {e}",
                        self.columns
                    ))
                })?
                .as_materialized_series()
                .clone();

            if !is_datetime_dtype(s.dtype()) {
                return Err(Error::InvalidInput(format!(
                    "HolidayEncoder.transform: column '{col}' has dtype {}; expected the Date or \
                     Datetime dtype it was fitted with. Re-fit on this data before transforming.",
                    s.dtype()
                )));
            }

            let days = day_values(&s, col, "HolidayEncoder.transform")?;
            // The cached set covers the fitted span; anything outside it is
            // resolved from the rules directly so out-of-range years are
            // answered, not zeroed. Those rows usually share a year, so the
            // year's holiday days are built once and reused.
            let mut outside_years: HashMap<i32, Vec<i32>> = HashMap::new();
            let flags: ChunkedArray<Float64Type> = days
                .iter()
                .map(|day| {
                    day.map(|day| {
                        let is_holiday = match self.day_range {
                            Some((lo, hi)) if day >= lo && day <= hi => {
                                self.holiday_days.contains(&day)
                            }
                            _ => {
                                let (year, _, _) = civil_from_days(day);
                                outside_years
                                    .entry(year)
                                    .or_insert_with(|| holidays_in_year(self.country, year))
                                    .contains(&day)
                            }
                        };
                        if is_holiday { 1.0 } else { 0.0 }
                    })
                })
                .collect();

            let out_name = format!("{col}_is_holiday");
            out.with_column(
                flags
                    .into_series()
                    .with_name(out_name.as_str().into())
                    .into(),
            )
            .map_err(|e| Error::Computation(format!("HolidayEncoder.transform: {e}")))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use polars::prelude::{TimeUnit, TimeZone};

    /// Days since the Unix epoch of 2024-01-01 (a Monday).
    const D0: i32 = 19_723;
    /// Days since the Unix epoch of 2024-07-04 (US Independence Day).
    const JULY4: i32 = 19_908;
    /// Microseconds since the Unix epoch of 2024-01-01T00:00:00Z.
    const T0: i64 = 1_704_067_200_000_000;
    /// Microseconds in one day.
    const DAY_US: i64 = 86_400_000_000;

    /// Days since the Unix epoch of the given civil date.
    fn d(y: i32, m: u32, day: u32) -> i32 {
        days_from_civil(y, m, day)
    }

    fn date_col(name: &str, vals: &[Option<i32>]) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Date)
            .unwrap()
            .into()
    }

    fn datetime_col(name: &str, vals: &[Option<i64>], tz: Option<TimeZone>) -> Column {
        Series::new(name.into(), vals)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, tz))
            .unwrap()
            .into()
    }

    /// Fit `encoder` on one `Date` column named `d` holding `vals`.
    fn fit_on(encoder: &mut HolidayEncoder, vals: &[Option<i32>]) -> DataFrame {
        let df = DataFrame::new(vals.len(), vec![date_col("d", vals)]).unwrap();
        encoder.fit(df.clone()).unwrap();
        df
    }

    /// Assert `out["{col}_is_holiday"]` is `Float64` and equals `expected`.
    fn assert_flags(out: &DataFrame, col: &str, expected: &[Option<f64>]) {
        let name = format!("{col}_is_holiday");
        let s = out.column(name.as_str()).unwrap();
        assert_eq!(
            s.dtype(),
            &DataType::Float64,
            "wrong output dtype for {name}"
        );
        let ca = s.f64().unwrap();
        assert_eq!(ca.len(), expected.len(), "wrong length for {name}");
        for (i, want) in expected.iter().enumerate() {
            match (ca.get(i), want) {
                (Some(got), Some(want)) => assert_relative_eq!(got, *want),
                (None, None) => {}
                (got, want) => panic!("row {i} of {name}: got {got:?}, want {want:?}"),
            }
        }
    }

    // --- calendar arithmetic -------------------------------------------------

    #[test]
    fn test_days_from_civil_anchors() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2024, 1, 1), D0);
        assert_eq!(days_from_civil(2000, 1, 1), 10_957);
        assert_eq!(days_from_civil(2024, 7, 4), JULY4);
        // Leap day exists in 2024 and shifts every later day by one.
        assert_eq!(days_from_civil(2024, 2, 29), 19_782);
        assert_eq!(days_from_civil(2024, 3, 1), 19_783);
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2100, 2), 28); // 2100 is not a leap year
        assert_eq!(days_in_month(2000, 2), 29); // 2000 is
    }

    #[test]
    fn test_weekday_from_days() {
        // 1970-01-01 was a Thursday (index 3 of a Monday-based week).
        assert_eq!(weekday_from_days(0), 3);
        assert_eq!(weekday_from_days(D0), 0); // 2024-01-01, Monday
        assert_eq!(weekday_from_days(JULY4), 3); // 2024-07-04, Thursday
        assert_eq!(weekday_from_days(d(2024, 12, 25)), 2); // Wednesday
    }

    #[test]
    fn test_civil_from_days_round_trip() {
        for (y, m, day) in [
            (1970, 1, 1),
            (1969, 12, 31),
            (2024, 2, 29),
            (1999, 12, 31),
            (2100, 3, 1),
        ] {
            assert_eq!(civil_from_days(days_from_civil(y, m, day)), (y, m, day));
        }
    }

    #[test]
    fn test_easter_dates() {
        // Published Easter Sundays.
        assert_eq!(easter_days(1970), d(1970, 3, 29));
        assert_eq!(easter_days(1985), d(1985, 4, 7));
        assert_eq!(easter_days(2000), d(2000, 4, 23));
        assert_eq!(easter_days(2023), d(2023, 4, 9));
        assert_eq!(easter_days(2024), d(2024, 3, 31));
        assert_eq!(easter_days(2025), d(2025, 4, 20));
        assert_eq!(easter_days(2026), d(2026, 4, 5));
        assert_eq!(easter_days(2038), d(2038, 4, 25));
    }

    #[test]
    fn test_nth_and_last_weekday_resolution() {
        // 3rd Monday of January 2024, 4th Thursday of November 2024,
        // 1st Monday of May 2024, last Monday of August 2024.
        assert_eq!(nth_weekday_days(2024, 1, 0, 3), d(2024, 1, 15));
        assert_eq!(nth_weekday_days(2024, 11, 3, 4), d(2024, 11, 28));
        assert_eq!(nth_weekday_days(2024, 5, 0, 1), d(2024, 5, 6));
        assert_eq!(last_weekday_days(2024, 8, 0), d(2024, 8, 26));
        assert_eq!(last_weekday_days(2025, 5, 0), d(2025, 5, 26));
        // A month whose 1st is the target weekday, and a 5-occurrence month.
        assert_eq!(nth_weekday_days(2024, 7, 0, 1), d(2024, 7, 1));
        assert_eq!(nth_weekday_days(2024, 4, 1, 5), d(2024, 4, 30));
    }

    // --- issue acceptance cases ---------------------------------------------

    #[test]
    fn test_us_new_year_and_plain_day() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[Some(D0), Some(D0 + 1)]);
        let out = enc.transform(df).unwrap();
        // 2024-01-01 is New Year's Day; 2024-01-02 is not a holiday.
        assert_flags(&out, "d", &[Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_us_independence_day() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[Some(JULY4), Some(JULY4 + 1)]);
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_gb_christmas() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::GB);
        let df = fit_on(&mut enc, &[Some(d(2024, 12, 25)), Some(d(2024, 12, 24))]);
        let out = enc.transform(df).unwrap();
        // Christmas Day is a holiday, Christmas Eve is not.
        assert_flags(&out, "d", &[Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_non_holiday_weekday() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        // 2024-03-13 and 2024-03-14 are ordinary weekdays.
        let df = fit_on(&mut enc, &[Some(d(2024, 3, 13)), Some(d(2024, 3, 14))]);
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(0.0), Some(0.0)]);
    }

    #[test]
    fn test_null_dates_are_preserved() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[Some(D0), None, Some(D0 + 1)]);
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(1.0), None, Some(0.0)]);
    }

    #[test]
    fn test_countries_differ_on_july_fourth() {
        let vals = [Some(JULY4)];
        let df = DataFrame::new(1, vec![date_col("d", &vals)]).unwrap();

        let mut us = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        us.fit(df.clone()).unwrap();
        assert_flags(&us.transform(df.clone()).unwrap(), "d", &[Some(1.0)]);

        // July 4th is not a Japanese holiday.
        let mut jp = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::JP);
        jp.fit(df.clone()).unwrap();
        assert_flags(&jp.transform(df).unwrap(), "d", &[Some(0.0)]);
    }

    #[test]
    fn test_us_thanksgiving_multi_year() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2024, 11, 28)),
                Some(d(2025, 11, 27)),
                Some(d(2026, 11, 26)),
                Some(d(2025, 11, 28)), // the day after, not a holiday
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(1.0), Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_not_fitted_error() {
        let df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        let enc = HolidayEncoder::new().columns(&["d"]);
        let err = enc.transform(df).unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
    }

    // --- calendar contents ---------------------------------------------------

    #[test]
    fn test_easter_relative_holidays() {
        // Easter 2024 was 2024-03-31.
        let vals = [
            Some(d(2024, 3, 29)), // Good Friday
            Some(d(2024, 4, 1)),  // Easter Monday
            Some(d(2024, 5, 9)),  // Ascension (Easter + 39)
            Some(d(2024, 5, 20)), // Whit Monday (Easter + 50)
        ];
        let df = DataFrame::new(vals.len(), vec![date_col("d", &vals)]).unwrap();

        let mut gb = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::GB);
        gb.fit(df.clone()).unwrap();
        assert_flags(
            &gb.transform(df.clone()).unwrap(),
            "d",
            &[Some(1.0), Some(1.0), Some(0.0), Some(0.0)],
        );

        let mut de = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::DE);
        de.fit(df.clone()).unwrap();
        assert_flags(
            &de.transform(df.clone()).unwrap(),
            "d",
            &[Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
        );

        let mut fr = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::FR);
        fr.fit(df.clone()).unwrap();
        // Good Friday is not a French national holiday.
        assert_flags(
            &fr.transform(df.clone()).unwrap(),
            "d",
            &[Some(0.0), Some(1.0), Some(1.0), Some(1.0)],
        );

        let mut us = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        us.fit(df.clone()).unwrap();
        assert_flags(
            &us.transform(df).unwrap(),
            "d",
            &[Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
        );
    }

    #[test]
    fn test_us_nth_weekday_holidays() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2024, 1, 15)),  // MLK Day, 3rd Mon Jan
                Some(d(2024, 2, 19)),  // Washington's Birthday, 3rd Mon Feb
                Some(d(2024, 5, 27)),  // Memorial Day, last Mon May
                Some(d(2024, 9, 2)),   // Labor Day, 1st Mon Sep
                Some(d(2024, 10, 14)), // Columbus Day, 2nd Mon Oct
                Some(d(2024, 1, 8)),   // 2nd Mon Jan: not a US holiday
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(
            &out,
            "d",
            &[
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
            ],
        );
    }

    #[test]
    fn test_gb_bank_holidays() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::GB);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2024, 1, 1)),   // New Year's Day
                Some(d(2024, 5, 6)),   // Early May bank holiday, 1st Mon May
                Some(d(2024, 5, 27)),  // Spring bank holiday, last Mon May
                Some(d(2024, 8, 26)),  // Summer bank holiday, last Mon Aug
                Some(d(2024, 12, 26)), // Boxing Day
                Some(d(2024, 8, 19)),  // 3rd Mon Aug: not a holiday
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(
            &out,
            "d",
            &[
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
            ],
        );
    }

    #[test]
    fn test_japanese_holidays() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::JP);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2024, 1, 8)),   // Coming of Age Day, 2nd Mon Jan
                Some(d(2024, 5, 5)),   // Children's Day
                Some(d(2024, 7, 15)),  // Marine Day, 3rd Mon Jul
                Some(d(2024, 9, 16)),  // Respect for the Aged Day, 3rd Mon Sep
                Some(d(2024, 10, 14)), // Sports Day, 2nd Mon Oct
                Some(d(2024, 11, 23)), // Labor Thanksgiving Day
                Some(d(2024, 7, 4)),   // US Independence Day, no JP holiday
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(
            &out,
            "d",
            &[
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
            ],
        );
    }

    #[test]
    fn test_indian_fixed_holidays() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::IN);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2024, 1, 26)), // Republic Day
                Some(d(2024, 8, 15)), // Independence Day
                Some(d(2024, 10, 2)), // Gandhi Jayanti
                Some(d(2024, 8, 14)),
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(1.0), Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_juneteenth_only_from_2021() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2020, 6, 19)),
                Some(d(2021, 6, 19)),
                Some(d(2024, 6, 19)),
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(0.0), Some(1.0), Some(1.0)]);
    }

    #[test]
    fn test_german_unity_day_only_from_1990() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::DE);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(1985, 10, 3)),
                Some(d(1990, 10, 3)),
                Some(d(2024, 10, 3)),
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(0.0), Some(1.0), Some(1.0)]);
    }

    #[test]
    fn test_japanese_emperors_birthday_era_bounds() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::JP);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(1985, 12, 23)), // Showa era: Dec 23 was an ordinary day
                Some(d(1989, 12, 23)), // first Heisei-era celebration
                Some(d(2015, 12, 23)), // Heisei era Emperor's Birthday
                Some(d(2018, 12, 23)), // last Heisei-era occurrence
                Some(d(2024, 12, 23)), // no longer a holiday
                Some(d(2015, 2, 23)),  // not yet a holiday
                Some(d(2024, 2, 23)),  // Reiwa era Emperor's Birthday
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(
            &out,
            "d",
            &[
                Some(0.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
            ],
        );
    }

    #[test]
    fn test_weekend_holiday_marked_without_substitute_day() {
        // 2021-12-25 was a Saturday and 2022-01-01 a Saturday: both are marked
        // on their own calendar day, and the US observed days (2021-12-24,
        // 2021-12-31) are not holidays.
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(
            &mut enc,
            &[
                Some(d(2021, 12, 25)),
                Some(d(2021, 12, 24)),
                Some(d(2022, 1, 1)),
                Some(d(2021, 12, 31)),
            ],
        );
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(0.0), Some(1.0), Some(0.0)]);
    }

    // --- dtype handling ------------------------------------------------------

    #[test]
    fn test_datetime_with_time_resolves_to_calendar_day() {
        // 2024-07-04T13:30Z is still Independence Day; 2024-07-03T23:30Z is not.
        let vals = [
            Some(T0 + 185 * DAY_US + 13 * 3_600_000_000),
            Some(T0 + 184 * DAY_US - 1_800_000_000),
        ];
        let df = DataFrame::new(2, vec![datetime_col("t", &vals, None)]).unwrap();

        let mut enc = HolidayEncoder::new()
            .columns(&["t"])
            .country(HolidayCountry::US);
        enc.fit(df.clone()).unwrap();
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "t", &[Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_timezone_aware_datetime_uses_utc_naive_day() {
        let vals = [Some(T0 + 185 * DAY_US + 3_600_000_000)];
        let df = DataFrame::new(1, vec![datetime_col("t", &vals, Some(TimeZone::UTC))]).unwrap();

        let mut enc = HolidayEncoder::new()
            .columns(&["t"])
            .country(HolidayCountry::US);
        enc.fit(df.clone()).unwrap();
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "t", &[Some(1.0)]);
    }

    #[test]
    fn test_multiple_columns() {
        let a = date_col("a", &[Some(D0), Some(JULY4)]);
        let b = date_col("b", &[Some(JULY4 + 1), Some(D0)]);
        let df = DataFrame::new(2, vec![a, b]).unwrap();

        let mut enc = HolidayEncoder::new()
            .columns(&["a", "b"])
            .country(HolidayCountry::US);
        enc.fit(df.clone()).unwrap();
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "a", &[Some(1.0), Some(1.0)]);
        assert_flags(&out, "b", &[Some(0.0), Some(1.0)]);
        assert_eq!(out.width(), 4);
    }

    // --- out-of-range years --------------------------------------------------

    #[test]
    fn test_year_outside_fitted_range_uses_rules() {
        // Fit on 2024 only, then transform a 2030 frame: the row is outside the
        // cached day range and must be answered from the rules.
        let fit_df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        let later = DataFrame::new(
            3,
            vec![date_col(
                "d",
                &[
                    Some(d(2030, 1, 1)),   // New Year's Day
                    Some(d(2030, 1, 2)),   // not a holiday
                    Some(d(2030, 11, 28)), // Thanksgiving
                ],
            )],
        )
        .unwrap();

        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        enc.fit(fit_df).unwrap();
        let out = enc.transform(later).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(0.0), Some(1.0)]);
    }

    #[test]
    fn test_year_before_fitted_range_uses_rules() {
        let fit_df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        let earlier = DataFrame::new(
            2,
            vec![date_col(
                "d",
                &[Some(d(1999, 12, 25)), Some(d(1999, 12, 24))],
            )],
        )
        .unwrap();

        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::DE);
        enc.fit(fit_df).unwrap();
        let out = enc.transform(earlier).unwrap();
        assert_flags(&out, "d", &[Some(1.0), Some(0.0)]);
    }

    // --- fit-time validation -------------------------------------------------

    #[test]
    fn test_empty_dataframe_rejected() {
        let mut enc = HolidayEncoder::new().columns(&["d"]);
        let err = enc.fit(DataFrame::empty()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_non_date_column_rejected() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut enc = HolidayEncoder::new().columns(&["x"]);
        match enc.fit(df).unwrap_err() {
            Error::InvalidInput(msg) => {
                // polars displays Float64 as `f64`.
                assert!(msg.contains("f64"), "dtype not named: {msg}")
            }
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_string_column_rejected() {
        let s = Column::from(Series::new("s".into(), &["2024-01-01"]));
        let df = DataFrame::new(1, vec![s]).unwrap();
        let mut enc = HolidayEncoder::new().columns(&["s"]);
        match enc.fit(df).unwrap_err() {
            Error::InvalidInput(msg) => {
                // polars displays String as `str`.
                assert!(msg.contains("str"), "dtype not named: {msg}")
            }
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_column_rejected() {
        let df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        let mut enc = HolidayEncoder::new().columns(&["nope"]);
        assert!(matches!(enc.fit(df).unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn test_auto_discovery_requires_a_date_column() {
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![f]).unwrap();
        let mut enc = HolidayEncoder::new();
        assert!(matches!(enc.fit(df).unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn test_auto_discovery_uses_date_columns_only() {
        let d = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let f = Column::from(Series::new("x".into(), &[1.0_f64, 2.0]));
        let df = DataFrame::new(2, vec![d, f]).unwrap();

        let mut enc = HolidayEncoder::new();
        enc.fit(df.clone()).unwrap();
        let out = enc.transform(df).unwrap();
        assert!(out.column("d_is_holiday").is_ok());
        assert!(out.column("x_is_holiday").is_err());
        assert_eq!(out.width(), 3);
    }

    #[test]
    fn test_extreme_date_values_do_not_panic() {
        // A corrupt `Date` column can hold any i32 day count, including the
        // extremes, whose years are ~±5.9 million. Neither fit nor transform
        // may overflow or try to materialise that span.
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[Some(i32::MIN), Some(i32::MAX), Some(0)]);
        let out = enc.transform(df).unwrap();
        // 1970-01-01 is New Year's Day; the absurd days match no rule.
        assert_flags(&out, "d", &[Some(0.0), Some(0.0), Some(1.0)]);
    }

    #[test]
    fn test_all_null_column_is_not_an_error() {
        // Nothing to pre-compute, but the column is still encodable (all nulls
        // in, all nulls out).
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[None, None]);
        let out = enc.transform(df).unwrap();
        assert_flags(&out, "d", &[None, None]);
    }

    #[test]
    fn test_duplicate_columns_deduped() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d", "d"])
            .country(HolidayCountry::US);
        let df = fit_on(&mut enc, &[Some(D0), Some(D0 + 1)]);
        let out = enc.transform(df).unwrap();
        // one input column + one generated column
        assert_eq!(out.width(), 2);
        assert_flags(&out, "d", &[Some(1.0), Some(0.0)]);
    }

    #[test]
    fn test_generated_name_collision_rejected() {
        let d = date_col("d", &[Some(D0), Some(D0 + 1)]);
        let clash = date_col("d_is_holiday", &[Some(D0), Some(D0 + 1)]);
        let df = DataFrame::new(2, vec![d, clash]).unwrap();

        let mut enc = HolidayEncoder::new().columns(&["d"]);
        match enc.fit(df).unwrap_err() {
            Error::InvalidInput(msg) => {
                assert!(msg.contains("d_is_holiday"), "name not named: {msg}")
            }
            other => panic!("expected Error::InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_input_collision_rejected() {
        let mut enc = HolidayEncoder::new().columns(&["d"]);
        let fit_df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        enc.fit(fit_df).unwrap();

        let d = date_col("d", &[Some(D0)]);
        let clash = date_col("d_is_holiday", &[Some(D0)]);
        let transform_df = DataFrame::new(1, vec![d, clash]).unwrap();
        assert!(matches!(
            enc.transform(transform_df).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn test_transform_rejects_changed_dtype() {
        let mut enc = HolidayEncoder::new().columns(&["d"]);
        let fit_df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();
        enc.fit(fit_df).unwrap();

        let s = Column::from(Series::new("d".into(), &[1.0_f64]));
        let transform_df = DataFrame::new(1, vec![s]).unwrap();
        assert!(matches!(
            enc.transform(transform_df).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }

    // --- builders and state --------------------------------------------------

    #[test]
    fn test_default_equals_new() {
        let def = HolidayEncoder::default();
        let new = HolidayEncoder::new();
        assert_eq!(def.country, new.country);
        assert_eq!(def.country, HolidayCountry::US);
        assert_eq!(def.fitted, new.fitted);
        assert!(!def.fitted);
        assert!(def.columns.is_empty());
        assert!(def.column_config.is_none());
        assert!(def.holiday_days.is_empty());
        assert!(def.day_range.is_none());
    }

    #[test]
    fn test_country_builder_selects_calendar() {
        let vals = [Some(JULY4)];
        let df = DataFrame::new(1, vec![date_col("d", &vals)]).unwrap();

        let mut us = HolidayEncoder::new();
        us = us.columns(&["d"]).country(HolidayCountry::US);
        us.fit(df.clone()).unwrap();
        assert_flags(&us.transform(df.clone()).unwrap(), "d", &[Some(1.0)]);

        let mut jp = HolidayEncoder::new();
        jp = jp.columns(&["d"]).country(HolidayCountry::JP);
        jp.fit(df.clone()).unwrap();
        assert_flags(&jp.transform(df).unwrap(), "d", &[Some(0.0)]);
    }

    #[test]
    fn test_failed_refit_resets_fitted_state() {
        let mut enc = HolidayEncoder::new()
            .columns(&["d"])
            .country(HolidayCountry::US);
        let df1 = fit_on(&mut enc, &[Some(D0)]);
        assert!(enc.transform(df1).is_ok());

        // Re-fit on a frame that has no Date/Datetime column named "d".
        let bad =
            DataFrame::new(1, vec![Column::from(Series::new("x".into(), &[1.0_f64]))]).unwrap();
        assert!(enc.fit(bad).is_err());

        let err = enc
            .transform(DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap())
            .unwrap_err();
        assert!(matches!(err, Error::NotFitted(_)), "got {err:?}");
    }

    #[test]
    fn test_setter_after_fit_invalidates() {
        let df = DataFrame::new(1, vec![date_col("d", &[Some(D0)])]).unwrap();

        let mut enc = HolidayEncoder::new().columns(&["d"]);
        enc.fit(df.clone()).unwrap();
        assert!(enc.transform(df.clone()).is_ok());

        let mut enc = enc.country(HolidayCountry::GB);
        assert!(matches!(
            enc.transform(df.clone()).unwrap_err(),
            Error::NotFitted(_)
        ));

        enc.fit(df.clone()).unwrap();
        let enc = enc.columns(&["d"]);
        assert!(matches!(
            enc.transform(df).unwrap_err(),
            Error::NotFitted(_)
        ));
    }
}
