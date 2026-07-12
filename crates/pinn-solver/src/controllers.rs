// ─── Convergence tracker ─────────────────────────────────────────────────────

/// Half-window for plateau comparison (recent vs older max). Must stay at 20 for the
/// cascade mechanism to fire correctly — the older window retains the pre-AMR@7000 peak.
/// 20 readings × 200-step interval = 4 000 steps per half.
const PLATEAU_WINDOW: usize = 20;
/// Readings required for `is_kt_converged`. Smaller than `PLATEAU_WINDOW` to exit sooner
/// after the cascade drives K_t above target (saves ~1 600 steps vs window=20).
/// 12 readings × 200-step interval = 2 400 steps.
const CONVERGENCE_WINDOW: usize = 12;
/// Max K_t must improve by at least this much per `PLATEAU_WINDOW` (4 000 steps) or a
/// plateau restart fires.
const PLATEAU_EPSILON: f64 = 0.05;
/// Restart budgets, each tracked independently so a string of crashes can't exhaust the
/// plateau-restart budget and vice versa (the two failure modes are unrelated).
const MAX_PLATEAU_RESTARTS: usize = 4;
const MAX_CRASH_RESTARTS: usize = 4;
/// `lam_h_cap`/`lam_d_cap` floor and per-restart decay factor (cascade: 50 → 30 → 18 → ...).
const LAM_CAP_DECAY_FACTOR: f64 = 0.6;
const LAM_CAP_FLOOR: f64 = 15.0;
const LAM_CAP_INITIAL: f64 = 50.0;
/// Consecutive missed-reading threshold. At the existing 200-step probe cadence, 4 misses in
/// a row means the network has been fully NaN-diverged for ~800 steps with zero recovery
/// signal reaching the cascade — presumed stuck. `pub(crate)` so callers (`headless.rs`) can
/// reference it in restart log messages rather than re-hardcoding the number.
pub(crate) const STUCK_NONE_THRESHOLD: usize = 4;
/// K_t crash detection: fires when current K_t drops below this fraction of the recent
/// peak, provided that peak exceeds `CRASH_MIN_PEAK_KT` (so noise near zero doesn't trigger).
const CRASH_DROP_FRACTION: f64 = 0.5;
const CRASH_MIN_PEAK_KT: f64 = 1.5;
/// K_t convergence band: mean must exceed this fraction of target, with stdev under this
/// fraction of target, over `CONVERGENCE_WINDOW` readings.
const CONVERGED_MEAN_FRACTION: f64 = 0.98;
const CONVERGED_STDDEV_FRACTION: f64 = 0.015;

/// Which direction of `ConvergenceTracker`'s tracked metric counts as "better" — K_t wants
/// `LargerIsBetter` (target 3.0, improving means increasing), pin-lug's interface-gap RMS
/// wants `SmallerIsBetter` (target 0.0, improving means decreasing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricDirection {
    LargerIsBetter,
    SmallerIsBetter,
}

/// Internal metric-scale mode. `KtLegacy` reproduces today's K_t-only behavior byte-for-byte
/// (absolute epsilon/fraction thresholds tuned specifically for K_t's O(1..3) scale);
/// `Relative` generalizes to any metric whose absolute scale is problem-configuration-
/// dependent (e.g. pin-lug's interface-gap RMS, unit meters) by expressing every threshold
/// as a dimensionless fraction of the tracker's own recent history, except
/// `significant_floor` — an absolute "below this the metric is noise" cutoff the caller
/// derives from the problem's own physical reference scale.
#[derive(Clone, Copy)]
enum MetricMode {
    KtLegacy {
        plateau_epsilon: f64,
        crash_min_peak: f64,
        crash_drop_fraction: f64,
    },
    Relative {
        direction: MetricDirection,
        plateau_rel_eps: f64,
        crash_spike_factor: f64,
        significant_floor: f64,
    },
}

/// Which condition fired inside `ConvergenceTracker::check()` — crash always takes priority
/// over plateau when a single reading independently satisfies both (see `check()`'s doc
/// comment). Carries the same `new_lam_h_cap` value the corresponding individual method
/// (`check_kt_crash`/`check_plateau`) would itself have returned, so callers that migrate to
/// `check()` in the future get identical cap-cascade behavior, just disambiguated by variant
/// instead of by which of two calls returned `Some`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RestartReason {
    Crash(f64),
    Plateau(f64),
}

/// Detects K_t plateau in Phase 2 and triggers warm restarts to escape local attractors.
///
/// Plateau = max K_t in the recent window hasn't improved by `PLATEAU_EPSILON` over the
/// previous window. This is robust to AMR-induced K_t dips (±0.1–0.2 per event) that
/// would fool a stdev-based check into thinking training is still active.
/// Each restart resets LR + Adam and tightens `lam_h_cap` (50 → 30 → 18 → 15) to
/// increase kirsch gradient dominance over hole traction penalty.
pub struct ConvergenceTracker {
    kt_history: std::collections::VecDeque<f64>,
    pub plateau_restarts: usize,
    pub crash_restarts: usize,
    lam_h_cap: f64,
    mode: MetricMode,
    /// Consecutive `note_missed_reading()` calls since the last `note_reading_received()`
    /// (or since construction). Reset to 0 on any real reading or once a stuck-restart fires.
    consecutive_none: usize,
}

impl ConvergenceTracker {
    pub fn new() -> Self {
        Self {
            kt_history: std::collections::VecDeque::with_capacity(200),
            plateau_restarts: 0,
            crash_restarts: 0,
            lam_h_cap: LAM_CAP_INITIAL,
            mode: MetricMode::KtLegacy {
                plateau_epsilon: PLATEAU_EPSILON,
                crash_min_peak: CRASH_MIN_PEAK_KT,
                crash_drop_fraction: CRASH_DROP_FRACTION,
            },
            consecutive_none: 0,
        }
    }

    /// Construct a tracker for a non-K_t metric (e.g. pin-lug's interface-gap RMS), whose
    /// absolute scale is problem-configuration-dependent — every threshold below is a
    /// dimensionless fraction of the tracker's own recent history except
    /// `significant_floor`.
    ///
    /// `plateau_rel_eps`: recent-window best must improve by at least this fraction of the
    /// older window's best, or a plateau restart fires.
    /// `crash_spike_factor`: metric must spike to at least this multiple of its recent best
    /// (and that best must exceed `significant_floor`) for a crash restart to fire.
    /// `significant_floor`: absolute cutoff below which the metric is considered noise, not
    /// signal (mirrors `CRASH_MIN_PEAK_KT`'s role, but derived from the problem, not a
    /// hand-rolled literal).
    pub fn for_metric(
        direction: MetricDirection,
        plateau_rel_eps: f64,
        crash_spike_factor: f64,
        significant_floor: f64,
    ) -> Self {
        Self {
            kt_history: std::collections::VecDeque::with_capacity(200),
            plateau_restarts: 0,
            crash_restarts: 0,
            lam_h_cap: LAM_CAP_INITIAL,
            mode: MetricMode::Relative {
                direction,
                plateau_rel_eps,
                crash_spike_factor,
                significant_floor,
            },
            consecutive_none: 0,
        }
    }

    /// Total restarts across both budgets — used only for display (e.g. "[WARM RESTART #N]").
    pub fn total_restarts(&self) -> usize {
        self.plateau_restarts + self.crash_restarts
    }

    pub fn push(&mut self, kt: f64) {
        self.kt_history.push_back(kt);
        while self.kt_history.len() > 200 { self.kt_history.pop_front(); }
    }

    /// True when K_t has been within 2% of `target` and stable for the last `CONVERGENCE_WINDOW` readings.
    pub fn is_kt_converged(&self, target: f64) -> bool {
        if target <= 0.0 || self.kt_history.len() < CONVERGENCE_WINDOW { return false; }
        let recent: Vec<f64> = self.kt_history.iter().rev().take(CONVERGENCE_WINDOW).cloned().collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let var  = recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        mean > target * CONVERGED_MEAN_FRACTION && var.sqrt() < CONVERGED_STDDEV_FRACTION * target
    }

    /// Tighten `lam_h_cap` on the second+ restart overall (plateau or crash combined) —
    /// preserves the single progressive cascade (50 → 30 → 18 → ...) regardless of which
    /// budget the restart drew from.
    fn step_down_cap(&mut self) -> f64 {
        if self.total_restarts() >= 1 {
            self.lam_h_cap = (self.lam_h_cap * LAM_CAP_DECAY_FACTOR).max(LAM_CAP_FLOOR);
        }
        self.lam_h_cap
    }

    /// If plateau detected and the plateau-restart budget remains, returns `Some(new_lam_h_cap)`.
    /// Caller is responsible for resetting LR and Adam.
    pub fn check_plateau(&mut self) -> Option<f64> {
        // Need 2×window readings: recent window vs previous window
        if self.kt_history.len() < PLATEAU_WINDOW * 2 { return None; }
        if self.plateau_restarts >= MAX_PLATEAU_RESTARTS { return None; }

        let recent: Vec<f64> = self.kt_history.iter().rev().take(PLATEAU_WINDOW).cloned().collect();
        let older: Vec<f64>  = self.kt_history.iter().rev()
            .skip(PLATEAU_WINDOW).take(PLATEAU_WINDOW).cloned().collect();

        let is_plateau = match self.mode {
            MetricMode::KtLegacy { plateau_epsilon, .. } => {
                let max_recent = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let max_older  = older.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // Plateau: best K_t in recent 4 000 steps didn't improve by PLATEAU_EPSILON
                max_recent - max_older < plateau_epsilon
            }
            MetricMode::Relative { direction, plateau_rel_eps, .. } => match direction {
                MetricDirection::SmallerIsBetter => {
                    let min_recent = recent.iter().cloned().fold(f64::INFINITY, f64::min);
                    let min_older  = older.iter().cloned().fold(f64::INFINITY, f64::min);
                    min_older > 0.0 && (min_older - min_recent) / min_older < plateau_rel_eps
                }
                MetricDirection::LargerIsBetter => {
                    let max_recent = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let max_older  = older.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    max_older > 0.0 && (max_recent - max_older) / max_older < plateau_rel_eps
                }
            },
        };

        if is_plateau {
            let new_cap = self.step_down_cap();
            self.plateau_restarts += 1;
            Some(new_cap)
        } else {
            None
        }
    }

    /// Detects a catastrophic metric collapse/spike relative to the recent peak/best.
    ///
    /// `KtLegacy`: fires when K_t < `CRASH_DROP_FRACTION` × max_prev5 and
    /// max_prev5 > `CRASH_MIN_PEAK_KT`, indicating the network has escaped the converged
    /// basin. `Relative`/`SmallerIsBetter` mirrors this: fires when `current` spikes to at
    /// least `crash_spike_factor` × the recent best (min_prev5), provided that best exceeds
    /// `significant_floor`. Caller should clear kt_history after triggering to prevent
    /// cascade detections while the metric is recovering.
    pub fn check_kt_crash(&mut self, current_kt: f64) -> Option<f64> {
        if self.crash_restarts >= MAX_CRASH_RESTARTS { return None; }

        let is_crash = match self.mode {
            MetricMode::KtLegacy { crash_min_peak, crash_drop_fraction, .. } => {
                // Need ≥6 readings so we have 5 prior readings before the current push
                // (Kirsch's convention: caller does `tracker.push(kt); check_kt_crash(kt)`,
                // so the just-pushed current reading is `skip(1)`-ed past).
                if self.kt_history.len() < 6 { return None; }
                let max_prev5: f64 = self.kt_history.iter().rev()
                    .skip(1).take(5)
                    .cloned().fold(f64::NEG_INFINITY, f64::max);
                max_prev5 > crash_min_peak && current_kt < max_prev5 * crash_drop_fraction
            }
            MetricMode::Relative { direction, crash_spike_factor, significant_floor, .. } => {
                // 5 readings suffice — the "recent best" window, not requiring the current
                // reading to have already been pushed (unlike KtLegacy's convention above).
                if self.kt_history.len() < 5 { return None; }
                match direction {
                    MetricDirection::SmallerIsBetter => {
                        let min_prev5: f64 = self.kt_history.iter().rev()
                            .take(5)
                            .cloned().fold(f64::INFINITY, f64::min);
                        min_prev5 > significant_floor && current_kt > crash_spike_factor * min_prev5
                    }
                    MetricDirection::LargerIsBetter => {
                        let max_prev5: f64 = self.kt_history.iter().rev()
                            .take(5)
                            .cloned().fold(f64::NEG_INFINITY, f64::max);
                        max_prev5 > significant_floor && current_kt < max_prev5 / crash_spike_factor
                    }
                }
            }
        };

        if is_crash {
            let new_cap = self.step_down_cap();
            self.crash_restarts += 1;
            Some(new_cap)
        } else {
            None
        }
    }

    /// Single entry point that internally enforces crash-before-plateau priority, making the
    /// ordering an invariant of `ConvergenceTracker` itself rather than a call-site
    /// convention every future caller must independently get right (issue #15a: previously,
    /// nothing stopped a careless call site from invoking `check_kt_crash`/`check_plateau`
    /// unconditionally instead of chaining them with `else if`, double-firing both restart
    /// budgets from one reading — see the `characterization_calling_both_checks_
    /// unconditionally_double_fires_today` test for that failure mode reproduced against the
    /// two-method API).
    ///
    /// Behaviorally identical to the pre-existing `if let Some(..) = check_kt_crash(current)
    /// { .. } else if let Some(..) = check_plateau() { .. }` pattern `run_training` (runner.rs),
    /// `run_headless`, and `run_headless_pinlug_inner` (headless.rs) all use today: checks
    /// crash first, and only consults plateau (via `or_else`, which is lazy — `check_plateau`'s
    /// side effects on `plateau_restarts` only run when this closure is actually invoked) when
    /// crash did not fire, so a reading satisfying both conditions can only ever consume the
    /// crash budget.
    ///
    /// Additive, not a replacement: `check_kt_crash`/`check_plateau` remain public and
    /// unchanged, and are still what every existing call site (`runner.rs`, `headless.rs`)
    /// uses — no call site is migrated onto `check()` in this change.
    pub fn check(&mut self, current: f64) -> Option<RestartReason> {
        self.check_kt_crash(current)
            .map(RestartReason::Crash)
            .or_else(|| self.check_plateau().map(RestartReason::Plateau))
    }

    /// Clears K_t history (call after crash recovery to avoid cascade detections).
    pub fn clear_history(&mut self) {
        self.kt_history.clear();
    }

    /// Call when the per-step metric probe returns `None`. Returns `Some(new_lam_h_cap)` —
    /// treated identically to `check_kt_crash`'s effect (same cap cascade, same
    /// `crash_restarts` budget) — once `STUCK_NONE_THRESHOLD` consecutive misses have been
    /// observed. Without this, a fully NaN-diverged network (whose metric probe returns `None`
    /// forever) would silently stop the crash/plateau cascade from ever firing again.
    pub fn note_missed_reading(&mut self) -> Option<f64> {
        self.consecutive_none += 1;
        if self.consecutive_none < STUCK_NONE_THRESHOLD { return None; }
        self.consecutive_none = 0;
        if self.crash_restarts >= MAX_CRASH_RESTARTS { return None; }
        let new_cap = self.step_down_cap();
        self.crash_restarts += 1;
        Some(new_cap)
    }

    /// Call whenever a real (non-`None`) reading is obtained — resets the missed-reading streak.
    pub fn note_reading_received(&mut self) {
        self.consecutive_none = 0;
    }

    /// Force-consumes one crash-budget restart unconditionally (no data-driven condition to
    /// check — the caller already has direct proof of Param-level corruption). Identical cap
    /// cascade / budget to `check_kt_crash`, just skipping its data-threshold test.
    pub fn force_crash_restart(&mut self) -> Option<f64> {
        if self.crash_restarts >= MAX_CRASH_RESTARTS { return None; }
        let new_cap = self.step_down_cap();
        self.crash_restarts += 1;
        Some(new_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a constant K_t reading enough times to satisfy `check_plateau`'s window
    /// requirement without ever satisfying `check_kt_crash` (no peak > CRASH_MIN_PEAK_KT
    /// relative to a drop, since every reading is identical).
    fn fill_plateau(t: &mut ConvergenceTracker, value: f64, n: usize) {
        for _ in 0..n { t.push(value); }
    }

    #[test]
    fn plateau_cap_cascades_and_caps_at_max_restarts() {
        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, 1.0, PLATEAU_WINDOW * 2);

        // Constant readings never improve -> every call plateaus, cascading the cap.
        assert_eq!(t.check_plateau(), Some(LAM_CAP_INITIAL)); // 1st restart: no decay yet
        assert_eq!(t.check_plateau(), Some(30.0));            // 2nd: 50*0.6
        assert_eq!(t.check_plateau(), Some(18.0));            // 3rd: 30*0.6
        assert_eq!(t.check_plateau(), Some(LAM_CAP_FLOOR));   // 4th: 18*0.6=10.8 -> floored to 15
        assert_eq!(t.plateau_restarts, MAX_PLATEAU_RESTARTS);

        // Budget exhausted -> no further restarts even though it's still plateauing.
        assert_eq!(t.check_plateau(), None);
        assert_eq!(t.plateau_restarts, MAX_PLATEAU_RESTARTS);
    }

    #[test]
    fn crash_and_plateau_budgets_are_independent() {
        let mut t = ConvergenceTracker::new();

        // Exhaust the crash budget (4 restarts) without ever calling check_plateau.
        for _ in 0..MAX_CRASH_RESTARTS {
            t.clear_history();
            for _ in 0..5 { t.push(2.0); } // peak > CRASH_MIN_PEAK_KT
            t.push(0.5);                   // drop below CRASH_DROP_FRACTION * peak
            assert!(t.check_kt_crash(0.5).is_some());
        }
        assert_eq!(t.crash_restarts, MAX_CRASH_RESTARTS);
        assert_eq!(t.plateau_restarts, 0, "crash restarts must not consume the plateau budget");

        // Crash budget is exhausted...
        t.clear_history();
        for _ in 0..5 { t.push(2.0); }
        t.push(0.5);
        assert_eq!(t.check_kt_crash(0.5), None);

        // ...but the plateau budget is untouched and still usable.
        fill_plateau(&mut t, 1.0, PLATEAU_WINDOW * 2);
        assert!(t.check_plateau().is_some());
        assert_eq!(t.plateau_restarts, 1);
    }

    // ─── Stuck-NaN missed-reading recovery ─────────────────────────────────────────────

    #[test]
    fn note_missed_reading_does_not_fire_below_threshold() {
        let mut t = ConvergenceTracker::new();
        for i in 0..3 {
            assert_eq!(t.note_missed_reading(), None, "miss #{} (below threshold) must not fire", i + 1);
        }
    }

    #[test]
    fn note_missed_reading_fires_on_reaching_threshold_with_initial_uncapped_lam() {
        let mut t = ConvergenceTracker::new();
        for _ in 0..3 { assert_eq!(t.note_missed_reading(), None); }
        let fired = t.note_missed_reading();
        assert_eq!(fired, Some(50.0), "first-ever restart (any kind) must return LAM_CAP_INITIAL=50.0 unchanged, got {fired:?}");
        assert_eq!(t.crash_restarts, 1, "note_missed_reading must increment the SAME crash_restarts budget check_kt_crash uses");
    }

    #[test]
    fn note_reading_received_resets_the_missed_streak() {
        let mut t = ConvergenceTracker::new();
        for _ in 0..3 { assert_eq!(t.note_missed_reading(), None); }
        t.note_reading_received();
        for _ in 0..3 { assert_eq!(t.note_missed_reading(), None, "streak must have been reset to 0, not resumed from 3"); }
        assert!(t.note_missed_reading().is_some(), "4th miss of the FRESH streak fires");
    }

    #[test]
    fn note_missed_reading_shares_crash_restart_budget_and_stops_at_max() {
        let mut t = ConvergenceTracker::new();
        // Exhaust MAX_CRASH_RESTARTS purely via note_missed_reading (4 misses per restart):
        for _ in 0..4 {
            for _ in 0..3 { assert_eq!(t.note_missed_reading(), None); }
            assert!(t.note_missed_reading().is_some());
        }
        assert_eq!(t.crash_restarts, 4);
        for _ in 0..3 { assert_eq!(t.note_missed_reading(), None); }
        assert_eq!(t.note_missed_reading(), None, "budget exhausted: must not fire even at a fresh 4th miss");
    }

    #[test]
    fn is_kt_converged_requires_enough_stable_readings() {
        let mut t = ConvergenceTracker::new();
        assert!(!t.is_kt_converged(3.0), "no readings yet");

        fill_plateau(&mut t, 3.0, CONVERGENCE_WINDOW);
        assert!(t.is_kt_converged(3.0));
        assert!(!t.is_kt_converged(0.0), "target <= 0 is never considered converged");
    }

    // ─── Relative-mode (pin-lug) generalization ────────────────────────────────────────

    #[test]
    fn for_metric_smaller_is_better_plateau_fires_on_insufficient_relative_improvement() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW * 2 { t.push(1e-3); }
        assert_eq!(t.check_plateau(), Some(LAM_CAP_INITIAL));
        assert_eq!(t.check_plateau(), Some(30.0));
        assert_eq!(t.check_plateau(), Some(18.0));
        assert_eq!(t.check_plateau(), Some(LAM_CAP_FLOOR));
        assert_eq!(t.plateau_restarts, MAX_PLATEAU_RESTARTS);
        assert_eq!(t.check_plateau(), None);
    }

    #[test]
    fn for_metric_smaller_is_better_plateau_does_not_fire_on_sufficient_relative_improvement() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(0.90); }
        assert_eq!(t.check_plateau(), None);
        assert_eq!(t.plateau_restarts, 0);
    }

    #[test]
    fn for_metric_smaller_is_better_plateau_boundary_at_exactly_epsilon_does_not_fire() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(0.95); } // exactly 5.0% improvement: strict < means this must NOT fire
        assert_eq!(t.check_plateau(), None);
    }

    #[test]
    fn for_metric_smaller_is_better_plateau_just_inside_boundary_fires() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(0.951); } // 4.9% improvement < 5%
        assert!(t.check_plateau().is_some());
    }

    #[test]
    fn for_metric_smaller_is_better_crash_fires_on_spike_above_floor() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(1e-4); }
        assert!(t.check_kt_crash(2.5e-4).is_some());
        assert_eq!(t.crash_restarts, 1);
    }

    #[test]
    fn for_metric_smaller_is_better_crash_does_not_fire_on_sub_spike_ratio() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(1e-4); }
        assert_eq!(t.check_kt_crash(1.5e-4), None);
    }

    #[test]
    fn for_metric_smaller_is_better_crash_suppressed_below_significant_floor() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(1e-8); }
        assert_eq!(t.check_kt_crash(1e-7), None);
        assert_eq!(t.crash_restarts, 0);
    }

    #[test]
    fn for_metric_crash_and_plateau_budgets_independent_in_relative_mode() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..MAX_CRASH_RESTARTS {
            t.clear_history();
            for _ in 0..5 { t.push(1e-4); }
            assert!(t.check_kt_crash(3e-4).is_some());
        }
        assert_eq!(t.crash_restarts, MAX_CRASH_RESTARTS);
        assert_eq!(t.plateau_restarts, 0);
        t.clear_history();
        for _ in 0..5 { t.push(1e-4); }
        assert_eq!(t.check_kt_crash(3e-4), None);
        for _ in 0..PLATEAU_WINDOW { t.push(1e-4); }
        for _ in 0..PLATEAU_WINDOW { t.push(1e-4); }
        assert!(t.check_plateau().is_some());
    }

    #[test]
    fn new_kt_legacy_behavior_is_unaffected_by_the_generalization() {
        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, 1.0, PLATEAU_WINDOW * 2);
        assert_eq!(t.check_plateau(), Some(LAM_CAP_INITIAL));
    }

    // ─── Relative-mode LargerIsBetter (unused by pin-lug today, but public API surface —
    // must be exercised so a flipped comparison/direction bug can't silently ship) ─────────

    #[test]
    fn for_metric_larger_is_better_plateau_fires_on_insufficient_relative_improvement() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(1.03); } // 3% improvement < 5% epsilon -> plateau
        assert!(t.check_plateau().is_some());
    }

    #[test]
    fn for_metric_larger_is_better_plateau_does_not_fire_on_sufficient_relative_improvement() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(1.10); } // 10% improvement >= 5% epsilon -> no plateau
        assert_eq!(t.check_plateau(), None);
    }

    #[test]
    fn for_metric_larger_is_better_plateau_boundary_at_exactly_epsilon_does_not_fire() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(1.05); } // exactly 5.0% improvement: strict < means this must NOT fire
        assert_eq!(t.check_plateau(), None);
    }

    #[test]
    fn for_metric_larger_is_better_plateau_just_inside_boundary_fires() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW { t.push(1.0); }
        for _ in 0..PLATEAU_WINDOW { t.push(1.049); } // 4.9% improvement < 5%
        assert!(t.check_plateau().is_some());
    }

    #[test]
    fn for_metric_larger_is_better_crash_fires_on_drop_below_floor_ratio() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(2.0); } // max_prev5 = 2.0, well above significant_floor
        // current < max_prev5 / crash_spike_factor = 1.0 -> crash
        assert!(t.check_kt_crash(0.9).is_some());
        assert_eq!(t.crash_restarts, 1);
    }

    #[test]
    fn for_metric_larger_is_better_crash_does_not_fire_above_drop_ratio() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(2.0); }
        assert_eq!(t.check_kt_crash(1.5), None); // 1.5 > 1.0 threshold -> no crash
    }

    #[test]
    fn for_metric_larger_is_better_crash_suppressed_below_significant_floor() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::LargerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..5 { t.push(1e-8); } // max_prev5 below significant_floor=1e-6
        assert_eq!(t.check_kt_crash(1e-9), None);
        assert_eq!(t.crash_restarts, 0);
    }

    // ─── Crash-before-plateau call-site priority ordering ────────────────────────────────
    //
    // `ConvergenceTracker` itself does not enforce that crash is checked before plateau —
    // both `run_headless` (K_t) and `run_headless_pinlug_inner` (interface-gap RMS) rely on
    // an `if let Some(..) = check_kt_crash(..) { .. } else if let Some(..) = check_plateau()
    // { .. }` call-site pattern. These tests pin down that pattern's actual behavior: when a
    // single history snapshot independently satisfies BOTH conditions, crash must win and
    // the plateau budget must stay untouched — proving an "else if" accidentally weakened to
    // two independent "if"s (which would double-fire and consume both budgets from one
    // reading) would be caught.

    #[test]
    fn relative_mode_call_site_pattern_crash_takes_priority_over_plateau_when_both_conditions_true() {
        let mk = || ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);

        // Adversarial pre-check: prove check_plateau() would ALSO independently fire from
        // this exact push history, so this isn't a vacuous "crash always wins because
        // plateau could never have fired anyway" test.
        let mut t_plateau_only = mk();
        for _ in 0..PLATEAU_WINDOW * 2 { t_plateau_only.push(1e-4); }
        assert!(t_plateau_only.check_plateau().is_some(),
            "test setup invariant broken: plateau must independently be true for this to be a real priority test");

        // Now exercise the actual call-site pattern on a fresh tracker with the same history.
        let mut t = mk();
        for _ in 0..PLATEAU_WINDOW * 2 { t.push(1e-4); }
        let current = 3e-4; // >= crash_spike_factor(2.0) * min_prev5(1e-4) -> crash ALSO true

        if t.check_kt_crash(current).is_some() {
            // crash path — matches production's ordering exactly.
        } else if t.check_plateau().is_some() {
            panic!("plateau must not fire: crash's else-if must short-circuit it when crash already fired");
        }
        assert_eq!(t.crash_restarts, 1, "crash must have fired");
        assert_eq!(t.plateau_restarts, 0,
            "plateau budget must be untouched because crash fired first (else-if ordering)");
    }

    #[test]
    fn kt_legacy_call_site_pattern_crash_takes_priority_over_plateau_when_both_conditions_true() {
        // Same priority-ordering proof as the Relative-mode test above, but for Kirsch's
        // pre-existing KtLegacy mode — `run_headless`'s call site uses the identical
        // `if check_kt_crash { .. } else if check_plateau { .. }` pattern. A uniform history
        // above CRASH_MIN_PEAK_KT (1.5) satisfies both conditions simultaneously: plateau
        // fires on ANY uniform value (max_recent - max_older == 0 < PLATEAU_EPSILON,
        // regardless of magnitude — the same mechanism `fill_plateau`'s existing 1.0-valued
        // tests rely on), while a uniform 1.6 also clears CRASH_MIN_PEAK_KT so a sufficiently
        // low `current` triggers the crash-drop condition too.
        const PEAK: f64 = 1.6;
        const CURRENT: f64 = 0.5; // < CRASH_DROP_FRACTION(0.5) * PEAK(1.6) = 0.8 -> crash fires

        // Adversarial pre-check: prove check_plateau() would ALSO independently fire from
        // this exact push history, so this isn't a vacuous "crash always wins because
        // plateau could never have fired anyway" test.
        let mut t_plateau_only = ConvergenceTracker::new();
        fill_plateau(&mut t_plateau_only, PEAK, PLATEAU_WINDOW * 2);
        assert!(t_plateau_only.check_plateau().is_some(),
            "test setup invariant broken: plateau must independently be true for this to be a real priority test");

        // Now exercise the actual call-site pattern on a fresh tracker with the same history.
        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, PEAK, PLATEAU_WINDOW * 2);
        if t.check_kt_crash(CURRENT).is_some() {
            // crash path — matches production's ordering exactly.
        } else if t.check_plateau().is_some() {
            panic!("plateau must not fire: crash's else-if must short-circuit it when crash already fired");
        }
        assert_eq!(t.crash_restarts, 1, "crash must have fired");
        assert_eq!(t.plateau_restarts, 0,
            "plateau budget must be untouched because crash fired first (else-if ordering)");
    }

    // ─── CHARACTERIZATION: today's ordering is a call-site convention only (issue #15a) ──
    //
    // The two tests above prove the *correctly written* `if check_kt_crash {..} else if
    // check_plateau {..}` call-site pattern behaves as intended. They do NOT prove the
    // ordering is enforced by `ConvergenceTracker` itself — nothing stops a future call site
    // from invoking both checks unconditionally (e.g. two separate `if let Some(..) = ..`
    // statements instead of an `else if` chain). This test demonstrates that exact failure
    // mode against TODAY'S API: from a single reading that independently satisfies both the
    // crash and plateau conditions, calling both checks unconditionally (no `else`) consumes
    // BOTH restart budgets, even though only one restart should have been counted. This is
    // the characterization/safety-net test for `ConvergenceTracker::check()`, which closes
    // this gap by construction.

    #[test]
    fn characterization_calling_both_checks_unconditionally_double_fires_today() {
        let mut t = ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);
        for _ in 0..PLATEAU_WINDOW * 2 { t.push(1e-4); }
        let current = 3e-4; // >= crash_spike_factor(2.0) * min_prev5(1e-4) -> crash ALSO true

        // NOT an else-if: both checks are invoked unconditionally, as a future careless call
        // site might. `ConvergenceTracker`'s public API today provides nothing to prevent this.
        let crash_fired = t.check_kt_crash(current).is_some();
        let plateau_fired = t.check_plateau().is_some();

        assert!(crash_fired, "test setup invariant broken: crash must independently fire");
        assert!(plateau_fired, "test setup invariant broken: plateau must independently fire too \
            (that's what makes double-firing possible from a single reading)");
        assert_eq!(t.crash_restarts, 1, "crash budget consumed");
        assert_eq!(t.plateau_restarts, 1,
            "BUG (today's behavior): plateau budget was ALSO consumed from the same single \
            reading that already triggered a crash restart — nothing in the API stops this");
    }

    // ─── `ConvergenceTracker::check()` — ordering enforced by construction (issue #15a fix) ──

    #[test]
    fn check_returns_none_and_consumes_no_budget_when_neither_condition_holds() {
        let mut t = ConvergenceTracker::new();
        // Too few readings for either check_plateau (needs 2*PLATEAU_WINDOW) or
        // check_kt_crash (needs >= 6) to do anything but return None.
        t.push(1.0);
        assert_eq!(t.check(1.0), None);
        assert_eq!(t.crash_restarts, 0);
        assert_eq!(t.plateau_restarts, 0);
    }

    #[test]
    fn check_returns_crash_variant_when_only_crash_condition_holds() {
        let mut t = ConvergenceTracker::new();
        for _ in 0..5 { t.push(2.0); } // 5 prior readings, peak > CRASH_MIN_PEAK_KT
        let current = 0.5; // < CRASH_DROP_FRACTION * peak -> crash
        // KtLegacy convention (mirrors crash_and_plateau_budgets_are_independent above):
        // caller pushes the current reading, then checks with that same value — check_kt_crash
        // needs >= 6 total readings (5 prior + the just-pushed current) to do anything.
        t.push(current);
        assert_eq!(t.check(current), Some(RestartReason::Crash(LAM_CAP_INITIAL)));
        assert_eq!(t.crash_restarts, 1);
        assert_eq!(t.plateau_restarts, 0);
    }

    #[test]
    fn check_returns_plateau_variant_when_only_plateau_condition_holds() {
        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, 1.0, PLATEAU_WINDOW * 2); // constant 1.0 never crashes (< CRASH_MIN_PEAK_KT)
        assert_eq!(t.check(1.0), Some(RestartReason::Plateau(LAM_CAP_INITIAL)));
        assert_eq!(t.crash_restarts, 0);
        assert_eq!(t.plateau_restarts, 1);
    }

    #[test]
    fn check_cascades_the_cap_identically_to_check_plateau_across_repeated_calls() {
        // Proves check() reuses the exact same step_down_cap cascade as the two-method API —
        // byte-identical cap sequence (50 -> 30 -> 18 -> 15), just reached through one call.
        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, 1.0, PLATEAU_WINDOW * 2);
        assert_eq!(t.check(1.0), Some(RestartReason::Plateau(LAM_CAP_INITIAL)));
        assert_eq!(t.check(1.0), Some(RestartReason::Plateau(30.0)));
        assert_eq!(t.check(1.0), Some(RestartReason::Plateau(18.0)));
        assert_eq!(t.check(1.0), Some(RestartReason::Plateau(LAM_CAP_FLOOR)));
        assert_eq!(t.plateau_restarts, MAX_PLATEAU_RESTARTS);
        assert_eq!(t.check(1.0), None, "plateau budget exhausted");
    }

    /// The test the acceptance criteria calls for by name: construct a state where BOTH a
    /// crash condition and a plateau condition would independently fire, call `check()`, and
    /// assert the returned `RestartReason` is the crash variant regardless of what order a
    /// hypothetical caller might otherwise have tried the two checks in — proving the
    /// ordering is now a property of `ConvergenceTracker` itself, not callable-order-dependent.
    #[test]
    fn check_prioritizes_crash_over_plateau_when_both_conditions_true_relative_mode() {
        let mk = || ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, 2.0, 1e-6);

        // Adversarial pre-check: prove check_plateau() would ALSO independently fire from this
        // exact push history (mirrors the pre-existing call-site-pattern test's precondition).
        let mut t_plateau_only = mk();
        for _ in 0..PLATEAU_WINDOW * 2 { t_plateau_only.push(1e-4); }
        assert!(t_plateau_only.check_plateau().is_some(),
            "test setup invariant broken: plateau must independently be true for this to be a real priority test");

        let mut t = mk();
        for _ in 0..PLATEAU_WINDOW * 2 { t.push(1e-4); }
        let current = 3e-4; // >= crash_spike_factor(2.0) * min_prev5(1e-4) -> crash ALSO true

        let reason = t.check(current);
        assert_eq!(reason, Some(RestartReason::Crash(LAM_CAP_INITIAL)),
            "crash must win when both conditions are true, regardless of hypothetical call order, got {reason:?}");
        assert_eq!(t.crash_restarts, 1, "crash must have fired");
        assert_eq!(t.plateau_restarts, 0, "check() must not double-fire — plateau budget must stay untouched");
    }

    #[test]
    fn check_prioritizes_crash_over_plateau_when_both_conditions_true_kt_legacy_mode() {
        const PEAK: f64 = 1.6;
        const CURRENT: f64 = 0.5; // < CRASH_DROP_FRACTION(0.5) * PEAK(1.6) = 0.8 -> crash fires

        let mut t_plateau_only = ConvergenceTracker::new();
        fill_plateau(&mut t_plateau_only, PEAK, PLATEAU_WINDOW * 2);
        assert!(t_plateau_only.check_plateau().is_some(),
            "test setup invariant broken: plateau must independently be true for this to be a real priority test");

        let mut t = ConvergenceTracker::new();
        fill_plateau(&mut t, PEAK, PLATEAU_WINDOW * 2);

        let reason = t.check(CURRENT);
        assert_eq!(reason, Some(RestartReason::Crash(LAM_CAP_INITIAL)),
            "crash must win when both conditions are true, regardless of hypothetical call order, got {reason:?}");
        assert_eq!(t.crash_restarts, 1, "crash must have fired");
        assert_eq!(t.plateau_restarts, 0, "check() must not double-fire — plateau budget must stay untouched");
    }

    #[test]
    fn force_crash_restart_fires_unconditionally_and_returns_lam_cap_initial_first_time() {
        let mut t = ConvergenceTracker::new();
        assert_eq!(t.force_crash_restart(), Some(LAM_CAP_INITIAL));
        assert_eq!(t.crash_restarts, 1);
        assert_eq!(t.plateau_restarts, 0, "must not touch the plateau budget");
    }

    #[test]
    fn force_crash_restart_cascades_the_cap_across_repeated_calls() {
        let mut t = ConvergenceTracker::new();
        assert_eq!(t.force_crash_restart(), Some(50.0));
        assert_eq!(t.force_crash_restart(), Some(30.0));
        assert_eq!(t.force_crash_restart(), Some(18.0));
        assert_eq!(t.force_crash_restart(), Some(LAM_CAP_FLOOR));
        assert_eq!(t.crash_restarts, MAX_CRASH_RESTARTS);
    }

    #[test]
    fn force_crash_restart_returns_none_once_crash_budget_exhausted() {
        let mut t = ConvergenceTracker::new();
        for _ in 0..MAX_CRASH_RESTARTS { assert!(t.force_crash_restart().is_some()); }
        assert_eq!(t.force_crash_restart(), None);
        assert_eq!(t.crash_restarts, MAX_CRASH_RESTARTS);
    }

    #[test]
    fn force_crash_restart_shares_the_same_budget_as_check_kt_crash() {
        let mut t = ConvergenceTracker::new();
        assert!(t.force_crash_restart().is_some());
        for _ in 0..5 { t.push(2.0); }
        t.push(0.5);
        assert!(t.check_kt_crash(0.5).is_some());
        assert_eq!(t.crash_restarts, 2, "both call paths must increment the SAME counter");
    }
}
