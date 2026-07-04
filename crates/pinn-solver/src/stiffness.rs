/// Stiffness-coupled SAW-BRDR / PirateNet-gate accelerator (opt-in, disabled by default).
///
/// Reuses the same real-time gradient-conflict cosine-similarity metric that drives
/// [`crate::decision_maker::PinnDecisionMaker`], but as a fully independent subsystem:
/// a run can enable either, both, or neither. When both are enabled, the caller computes
/// `GradientConflict` at most once per step (whichever subsystem's `advance()` fires) and
/// feeds the same value to both.
///
/// Conflict → `stiffness_factor` → two effects:
///   1. SAW-BRDR's physics-loss weight (lam_e, lam_eq, lam_const) is boosted externally,
///      after `SawBrdr::update()` returns — mirroring how `dynamic_lam_h_cap`/
///      `dynamic_lam_d_cap` are already applied as external clamps in `step_physics`.
///   2. PirateNet gate scalars get an accelerated learning rate, so a real gradient
///      conflict "wakes up" adaptive-residual capacity faster.
///
/// Architecture invariant: like `PinnDecisionMaker`, this module never reads K_t,
/// `ConvergenceTracker` history, or any analytical baseline — `update()` takes only a
/// `GradientConflict` (cosine similarity + gradient norms).
use pinn_core::messages::StiffnessConfig;

use crate::decision_maker::GradientConflict;

/// Maps `cosine_sim` to the standard PCGrad/gradient-surgery conflict score:
/// `cosine_sim = 1` (aligned) → 0; `cosine_sim = -1` (max conflict) → 1.
fn raw_stiffness(cosine_sim: f32) -> f32 {
    ((1.0 - cosine_sim.clamp(-1.0, 1.0)) / 2.0).clamp(0.0, 1.0)
}

/// EMA-smoothed stiffness controller. `current_factor` is held constant between checks
/// and updated at most once per `check_interval` steps.
pub struct StiffnessController {
    config: StiffnessConfig,
    pub current_factor: f32,
    step_counter: usize,
}

impl StiffnessController {
    pub fn new(config: StiffnessConfig) -> Self {
        Self { config, current_factor: 0.0, step_counter: 0 }
    }

    /// Advance internal step counter. Returns `true` when the caller should compute the
    /// gradient conflict metric and call `update()`.
    pub fn advance(&mut self) -> bool {
        self.step_counter += 1;
        self.config.enabled && (self.step_counter % self.config.check_interval == 0)
    }

    /// Update the held stiffness factor from a fresh conflict reading via EMA smoothing.
    /// Returns the new `current_factor`.
    pub fn update(&mut self, conflict: &GradientConflict) -> f32 {
        let raw = raw_stiffness(conflict.cosine_sim);
        let beta = self.config.ema_beta;
        self.current_factor = beta * self.current_factor + (1.0 - beta) * raw;
        self.current_factor
    }

    /// SAW-BRDR physics-loss boost: `1 + gain * factor`, hard-clamped to `[1, 4]`.
    /// Returns `1.0` (no-op) when disabled.
    pub fn physics_boost(&self) -> f64 {
        if !self.config.enabled {
            return 1.0;
        }
        let boost = 1.0 + self.config.physics_boost_gain * self.current_factor;
        boost.clamp(1.0, 4.0) as f64
    }

    /// PirateNet gate-LR multiplier: `1 + gain * factor`, hard-clamped to `[1, 5]`.
    /// Returns `1.0` (no-op) when disabled.
    pub fn alpha_lr_mult(&self) -> f64 {
        if !self.config.enabled {
            return 1.0;
        }
        let mult = 1.0 + self.config.alpha_accel_gain * self.current_factor;
        mult.clamp(1.0, 5.0) as f64
    }

    /// Gate magnitude above which a PirateNet block is "awake" (config passthrough).
    pub fn gate_awake_epsilon(&self) -> f32 {
        self.config.gate_awake_epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conflict(cosine_sim: f32) -> GradientConflict {
        GradientConflict { cosine_sim, g_pde_norm: 1.0, g_bc_norm: 1.0 }
    }

    fn enabled_config() -> StiffnessConfig {
        StiffnessConfig {
            enabled: true,
            check_interval: 50,
            ema_beta: 0.7,
            physics_boost_gain: 1.0,
            alpha_accel_gain: 2.0,
            gate_awake_epsilon: 1e-4,
        }
    }

    #[test]
    fn max_conflict_clamps_factor_to_one() {
        let mut ctrl = StiffnessController::new(enabled_config());
        // Repeated updates at cosine_sim=-1 converge the EMA to 1.0.
        for _ in 0..50 {
            ctrl.update(&conflict(-1.0));
        }
        assert!((ctrl.current_factor - 1.0).abs() < 1e-3);
        assert!((ctrl.physics_boost() - 2.0).abs() < 1e-3); // 1 + 1.0*1.0
        assert!((ctrl.alpha_lr_mult() - 3.0).abs() < 1e-3); // 1 + 2.0*1.0
    }

    #[test]
    fn perfect_alignment_keeps_factor_at_zero() {
        let mut ctrl = StiffnessController::new(enabled_config());
        ctrl.update(&conflict(1.0));
        assert_eq!(ctrl.current_factor, 0.0);
        assert_eq!(ctrl.physics_boost(), 1.0);
        assert_eq!(ctrl.alpha_lr_mult(), 1.0);
    }

    #[test]
    fn ema_does_not_snap_to_raw_value() {
        let mut ctrl = StiffnessController::new(enabled_config());
        let after_one = ctrl.update(&conflict(-1.0));
        // beta=0.7: factor = 0.7*0 + 0.3*1.0 = 0.3, not 1.0.
        assert!((after_one - 0.3).abs() < 1e-6);
    }

    #[test]
    fn disabled_is_a_no_op() {
        let mut ctrl = StiffnessController::new(StiffnessConfig { enabled: false, ..enabled_config() });
        assert!(!ctrl.advance());
        ctrl.update(&conflict(-1.0)); // still updates internal factor if called directly
        assert_eq!(ctrl.physics_boost(), 1.0);
        assert_eq!(ctrl.alpha_lr_mult(), 1.0);
    }

    #[test]
    fn boost_and_mult_are_hard_clamped_regardless_of_gain() {
        let mut ctrl = StiffnessController::new(StiffnessConfig {
            physics_boost_gain: 100.0,
            alpha_accel_gain: 100.0,
            ..enabled_config()
        });
        for _ in 0..50 {
            ctrl.update(&conflict(-1.0));
        }
        assert_eq!(ctrl.physics_boost(), 4.0);
        assert_eq!(ctrl.alpha_lr_mult(), 5.0);
    }
}
