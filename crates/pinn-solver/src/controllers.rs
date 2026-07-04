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
/// K_t crash detection: fires when current K_t drops below this fraction of the recent
/// peak, provided that peak exceeds `CRASH_MIN_PEAK_KT` (so noise near zero doesn't trigger).
const CRASH_DROP_FRACTION: f64 = 0.5;
const CRASH_MIN_PEAK_KT: f64 = 1.5;
/// K_t convergence band: mean must exceed this fraction of target, with stdev under this
/// fraction of target, over `CONVERGENCE_WINDOW` readings.
const CONVERGED_MEAN_FRACTION: f64 = 0.98;
const CONVERGED_STDDEV_FRACTION: f64 = 0.015;

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
}

impl ConvergenceTracker {
    pub fn new() -> Self {
        Self {
            kt_history: std::collections::VecDeque::with_capacity(200),
            plateau_restarts: 0,
            crash_restarts: 0,
            lam_h_cap: LAM_CAP_INITIAL,
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

        let max_recent = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_older  = older.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Plateau: best K_t in recent 4 000 steps didn't improve by PLATEAU_EPSILON
        if max_recent - max_older < PLATEAU_EPSILON {
            let new_cap = self.step_down_cap();
            self.plateau_restarts += 1;
            Some(new_cap)
        } else {
            None
        }
    }

    /// Detects a catastrophic K_t collapse (≥50% drop from recent peak).
    ///
    /// Fires when K_t < `CRASH_DROP_FRACTION` × max_prev5 and max_prev5 > `CRASH_MIN_PEAK_KT`,
    /// indicating the network has escaped the converged basin. Caller should clear
    /// kt_history after triggering to prevent cascade detections while K_t is recovering.
    pub fn check_kt_crash(&mut self, current_kt: f64) -> Option<f64> {
        if self.crash_restarts >= MAX_CRASH_RESTARTS { return None; }
        // Need ≥6 readings so we have 5 prior readings before the current push.
        if self.kt_history.len() < 6 { return None; }
        let max_prev5: f64 = self.kt_history.iter().rev()
            .skip(1).take(5)
            .cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_prev5 > CRASH_MIN_PEAK_KT && current_kt < max_prev5 * CRASH_DROP_FRACTION {
            let new_cap = self.step_down_cap();
            self.crash_restarts += 1;
            Some(new_cap)
        } else {
            None
        }
    }

    /// Clears K_t history (call after crash recovery to avoid cascade detections).
    pub fn clear_history(&mut self) {
        self.kt_history.clear();
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

    #[test]
    fn is_kt_converged_requires_enough_stable_readings() {
        let mut t = ConvergenceTracker::new();
        assert!(!t.is_kt_converged(3.0), "no readings yet");

        fill_plateau(&mut t, 3.0, CONVERGENCE_WINDOW);
        assert!(t.is_kt_converged(3.0));
        assert!(!t.is_kt_converged(0.0), "target <= 0 is never considered converged");
    }
}
