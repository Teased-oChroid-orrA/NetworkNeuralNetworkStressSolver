//! Issue #61 EPIC P2-08: executable verification ladder.
//!
//! A declared sequence of increasing-confidence checks (L0-L5), each backed by real,
//! executable code already built by earlier epics in this remediation plan, plus the one
//! MANDATORY new check this epic itself introduces: [`run_affine_amplitude_test`]. Issue #61
//! §1.3's own rule ("no claims without tests+integration+proof of live use") applies here as
//! much as anywhere — a "verification ladder" that is only a diagram, with no executable rungs,
//! is exactly the kind of dead abstraction the issue forbids.
//!
//! - **L0 — Analytic single-DOF sanity.** [`run_affine_amplitude_test`]: a single trainable
//!   scalar `a` (NOT a neural network — the network is not even involved yet), the ansatz
//!   `u=a*x, v=-nu*a*y`, optimized via real gradient descent through the measure-aware
//!   variational functional (`measure_integral::domain_integral_tensor`/`boundary_integral_
//!   tensor`, P2-04's differentiable variants), must recover `a_exact = sigma0/E` to high
//!   precision. Issue #61's own text: "mandatory... before neural-optimization debugging."
//! - **L1 — Differential operator cross-validation.** `differential_operator`'s own tests
//!   (P2-02): AD/FD backends agree with each other and with hand-derived analytic derivatives
//!   on a manufactured field.
//! - **L2 — Field-source / mixed-formulation enforcement.** `field_graph::check_mixed_stress_
//!   source_compatibility` (P2-03), asserted live on every training step in `step_physics`/
//!   `step_physics_multi`.
//! - **L3 — Measure-aware integral correctness.** `measure_integral`'s own tests (P2-04):
//!   constant-field exactness, and convergence under nonuniform/AMR-biased sampling.
//! - **L4 — No-hole neural training health.** [`no_hole_health_check`]: a full neural network
//!   (not a single scalar) trained on a real, hole-free plate must show finite, non-collapsed
//!   energy balance and non-trivial displacement. Issue #61's own text: "MANDATORY no-hole
//!   neural gate before Kt/hole results accepted." This level reports PASS/FAIL against
//!   documented (not yet hard-numeric-calibrated) criteria — the actual hard-threshold
//!   protocol distinguishing finite/infinite-domain references is P2-14's own job (issue #61's
//!   own epic list places "benchmark protocol with numeric thresholds" as a LATER, separate
//!   epic) — see this function's own doc comment for what is and isn't decided yet.
//! - **L5 — Hole/Kt acceptance.** Only meaningful once L4 has PASSED for a companion no-hole
//!   run of the same material/load — the actual gating enforcement (requiring a stored,
//!   verified L4 pass before a hole run's Kt is reported as accepted) is P2-14's job (it needs
//!   cross-run provenance, P2-13, to know which no-hole run a given hole run corresponds to).
//!   Declared here as the ladder's final rung so the full pipeline is visible in one place, not
//!   built here.

use pinn_core::material::MaterialProps;

use crate::measure_integral::{boundary_integral_tensor, domain_integral_tensor, plate_domain_area};
use crate::training_core::{BDevice, B};

/// See this module's own "L0" doc comment. `u = a*x`, `v = -nu*a*y` is the EXACT solution for
/// a rectangular plate (no hole) under uniform far-field uniaxial tension `sigma0` in x, `a_
/// exact = sigma0/E` — this test verifies the measure-aware variational functional (`Pi = U -
/// W_ext`, computed via the SAME `domain_integral_tensor`/`boundary_integral_tensor` a live
/// training consumer would use) recovers it via real gradient descent on a single scalar `a`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineAmplitudeResult {
    pub a_recovered: f64,
    pub a_exact: f64,
    pub relative_error: f64,
    pub steps_run: usize,
    /// True iff `relative_error` is below the tolerance passed to
    /// [`run_affine_amplitude_test`].
    pub passed: bool,
}

/// Runs the L0 affine-amplitude test for the given material/load/geometry. `steps` and
/// `tolerance` are caller-supplied (not hardcoded) so this can be used both as a fast unit-test
/// gate (few steps, loose tolerance) and a stricter startup gate (more steps, tight tolerance).
///
/// Learning rate is sized from the QUADRATIC coefficient of `Pi(a)` (`Pi(a) = C*a^2 - D*a` for
/// this exact ansatz - a closed, known fact about this specific test problem's shape, not the
/// unknown answer `a_exact` itself), so gradient descent converges reliably regardless of the
/// material's absolute magnitude (steel vs. aluminum vs. a synthetic test material) without
/// per-call tuning. This is standard optimizer practice (sizing a step from a known Lipschitz/
/// curvature bound), not a shortcut around actually running the optimization.
pub fn run_affine_amplitude_test(
    material: &MaterialProps,
    sigma0: f64,
    half_w: f64,
    half_h: f64,
    thickness: f64,
    steps: usize,
    tolerance: f64,
) -> AffineAmplitudeResult {
    let device = BDevice::default();
    let area = plate_domain_area(half_w, half_h, &[]);
    let e = material.e;
    let nu = material.nu;

    // Pi(a) = C*a^2 - D*a (see this module's own derivation in the P2-08 manifest entry):
    // C = 0.5*E*area*thickness (from U(a) = domain_integral_tensor(area, thickness, 0.5*E*a^2)),
    // D = sigma0*thickness*area (from W_ext(a) = boundary_integral_tensor(...) = D*a).
    // Newton-optimal step is 1/(2C); half that step converges geometrically over `steps`
    // iterations without overshoot risk from floating-point/formula mismatches.
    let c_coeff = 0.5 * e * area * thickness;
    let lr = 0.5 / (2.0 * c_coeff);

    let a_exact = sigma0 / e;
    let mut a_val: f64 = 0.0;

    for _ in 0..steps {
        let a = burn::tensor::Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![a_val as f32], vec![1]), &device,
        ).require_grad();

        let exx = a.clone();
        let eyy = a.clone().mul_scalar(-nu);
        let exy = burn::tensor::Tensor::<B, 1>::zeros([1], &device);
        let energy_density = crate::energy::dem_energy_per_point::<B>(exx, eyy, exy, material);
        let u_energy = domain_integral_tensor::<B>(area, thickness, energy_density);

        // Traction*displacement per outer edge (right/left carry the load; top/bottom carry
        // zero traction since py=0) - see this module's own W_ext(a) derivation above.
        let right = a.clone().mul_scalar(sigma0 * half_w);
        let left = a.clone().mul_scalar(sigma0 * half_w);
        let top = burn::tensor::Tensor::<B, 1>::zeros([1], &device);
        let bottom = burn::tensor::Tensor::<B, 1>::zeros([1], &device);
        let values = burn::tensor::Tensor::cat(vec![right, left, top, bottom], 0);
        let ds_per_point = [2.0 * half_h, 2.0 * half_h, 2.0 * half_w, 2.0 * half_w];
        let w_ext = boundary_integral_tensor::<B>(values, &ds_per_point, thickness);

        let pi = u_energy - w_ext;
        let grads = pi.backward();
        let grad_a = a.grad(&grads).expect("run_affine_amplitude_test: a must have a gradient");
        let grad_v: f32 = grad_a.into_data().to_vec::<f32>().unwrap()[0];

        a_val -= lr * grad_v as f64;
    }

    let relative_error = if a_exact.abs() > 1e-30 {
        (a_val - a_exact).abs() / a_exact.abs()
    } else {
        (a_val - a_exact).abs()
    };

    AffineAmplitudeResult {
        a_recovered: a_val,
        a_exact,
        relative_error,
        steps_run: steps,
        passed: relative_error < tolerance,
    }
}

/// See this module's own "L4" doc comment. Documented, NOT-yet-hard-calibrated health check
/// for a just-trained no-hole model - `energy_balance_error` finite and below a generous
/// sanity bound, and displacement genuinely non-trivial (not a collapsed near-zero solution -
/// directly motivated by the real `Debug_runs/stress_solver_report-with-hole.json` evidence
/// this remediation plan was opened against: a collapsed solution accepted as "converged" by
/// premature `auto_stop_on_plateau`). Final, tuned numeric thresholds are P2-14's job
/// ("benchmark protocol with numeric thresholds") - the bounds here are deliberately loose
/// sanity checks, not the calibrated acceptance criteria.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoHoleHealthCheck {
    pub energy_balance_error: f64,
    pub max_abs_displacement: f64,
    pub passed: bool,
    pub failure_reason: Option<&'static str>,
}

pub fn no_hole_health_check(energy_balance: &pinn_core::messages::EnergyBalance, max_abs_displacement: f64) -> NoHoleHealthCheck {
    let energy_balance_error = energy_balance.energy_balance_error;
    if !energy_balance_error.is_finite() {
        return NoHoleHealthCheck { energy_balance_error, max_abs_displacement, passed: false, failure_reason: Some("energy_balance_error is not finite") };
    }
    if !max_abs_displacement.is_finite() {
        return NoHoleHealthCheck { energy_balance_error, max_abs_displacement, passed: false, failure_reason: Some("max_abs_displacement is not finite") };
    }
    // A generous sanity bound, not a calibrated acceptance threshold (see doc comment).
    if energy_balance_error > 0.5 {
        return NoHoleHealthCheck { energy_balance_error, max_abs_displacement, passed: false, failure_reason: Some("energy_balance_error exceeds the 50% sanity bound") };
    }
    if max_abs_displacement < 1e-12 {
        return NoHoleHealthCheck { energy_balance_error, max_abs_displacement, passed: false, failure_reason: Some("max_abs_displacement is effectively zero - a collapsed/trivial solution") };
    }
    NoHoleHealthCheck { energy_balance_error, max_abs_displacement, passed: true, failure_reason: None }
}

/// Issue #62 PH3-03: the three states a production no-hole run's gate SHALL resolve to - never
/// left as a `null`/`None`/"not evaluated" stand-in once L0 and L4 have actually run (issue #62
/// §7's own literal wording). `Invalid` is for the caller to use when a run terminated before
/// reaching the point where L4 could even be computed (e.g. a report exported before the first
/// vis-cadence tick) - [`evaluate_no_hole_operational_gate`] itself is total given two REAL,
/// already-executed results, and never returns `Invalid` on its own; that state exists for the
/// caller to report honestly instead of calling this function with fabricated inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalStatus {
    Pass,
    Fail,
    Invalid,
}

/// See this module's own "PH3-03" doc comment above [`OperationalStatus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationalGateResult {
    pub status: OperationalStatus,
    /// Which rung failed, when `status == Fail` - `"L0"` or `"L4"`. `None` for `Pass` (and for
    /// `Invalid`, which [`evaluate_no_hole_operational_gate`] itself never produces - see that
    /// type's doc comment).
    pub failed_rung: Option<&'static str>,
    pub l0_passed: bool,
    pub l4_passed: bool,
}

/// Combines the REAL, already-executed L0 ([`run_affine_amplitude_test`], P2-08 - mandatory,
/// run before any neural optimization begins) and L4 (`user_problem::run_no_hole_benchmark`,
/// P2-14 - the hard numeric-threshold no-hole gate) results into one machine-readable verdict.
///
/// L1 (differential-operator cross-validation)/L2 (mixed-stress-source enforcement)/L3
/// (measure-aware integral correctness) are deliberately NOT re-evaluated here. L1/L3 are
/// structural code invariants proven once, by this crate's own test suite
/// (`differential_operator`'s/`measure_integral`'s own tests) against manufactured reference
/// solutions - properties of the CODE, not of any one run's model/config, so there is nothing
/// for a per-run gate to recompute (re-deriving them here would need a manufactured solution
/// this run doesn't have, or would just re-invoke the same fixed unit tests a training run has
/// no mechanism to call). L2 is a live `assert!` inside `step_physics`/`step_physics_multi`
/// (`field_graph::check_mixed_stress_source_compatibility`) that would already have PANICKED
/// this exact run had it been violated - a completed run is proof L2 held, by construction, not
/// something to separately query after the fact.
pub fn evaluate_no_hole_operational_gate(
    l0: &AffineAmplitudeResult,
    l4: &crate::user_problem::NoHoleBenchmarkResult,
) -> OperationalGateResult {
    let (status, failed_rung) = if !l0.passed {
        (OperationalStatus::Fail, Some("L0"))
    } else if !l4.passed {
        (OperationalStatus::Fail, Some("L4"))
    } else {
        (OperationalStatus::Pass, None)
    };
    OperationalGateResult { status, failed_rung, l0_passed: l0.passed, l4_passed: l4.passed }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_amplitude_test_recovers_a_exact_for_aluminum_under_uniaxial_tension() {
        let material = MaterialProps::al7075_t6();
        let sigma0 = 69e6_f64;
        let result = run_affine_amplitude_test(&material, sigma0, 0.1, 0.1, 0.005, 50, 1e-6);
        assert!(result.passed, "{result:?}");
        assert!((result.a_exact - sigma0 / material.e).abs() < 1e-15);
        assert!(result.relative_error < 1e-6, "{result:?}");
    }

    #[test]
    fn affine_amplitude_test_recovers_a_exact_for_steel_under_a_different_load_and_geometry() {
        // Different material, load, and geometry - proves this isn't tuned to one specific
        // numeric case.
        let material = MaterialProps::steel_4340();
        let sigma0 = 150e6_f64;
        let result = run_affine_amplitude_test(&material, sigma0, 0.25, 0.08, 0.01, 50, 1e-6);
        assert!(result.passed, "{result:?}");
        assert!(result.relative_error < 1e-6, "{result:?}");
    }

    #[test]
    fn no_hole_health_check_passes_for_healthy_metrics() {
        let eb = pinn_core::messages::EnergyBalance { internal_energy: 1.0, external_work: 1.0001, energy_balance_error: 0.0001 };
        let check = no_hole_health_check(&eb, 1e-5);
        assert!(check.passed, "{check:?}");
    }

    #[test]
    fn no_hole_health_check_fails_on_collapsed_near_zero_displacement() {
        // Directly reproduces the real Debug_runs evidence this remediation plan was opened
        // against: a near-zero-amplitude collapsed solution.
        let eb = pinn_core::messages::EnergyBalance { internal_energy: 0.0, external_work: 0.0, energy_balance_error: 0.001 };
        let check = no_hole_health_check(&eb, 1e-14);
        assert!(!check.passed);
        assert_eq!(check.failure_reason, Some("max_abs_displacement is effectively zero - a collapsed/trivial solution"));
    }

    #[test]
    fn no_hole_health_check_fails_on_non_finite_energy_balance_error() {
        let eb = pinn_core::messages::EnergyBalance { internal_energy: f64::NAN, external_work: 1.0, energy_balance_error: f64::NAN };
        let check = no_hole_health_check(&eb, 1e-5);
        assert!(!check.passed);
        assert_eq!(check.failure_reason, Some("energy_balance_error is not finite"));
    }

    // ─── Issue #62 PH3-03: machine-enforced operational gate ───────────────────────────────

    fn passing_l0() -> AffineAmplitudeResult {
        AffineAmplitudeResult { a_recovered: 1.0, a_exact: 1.0, relative_error: 0.0, steps_run: 100, passed: true }
    }
    fn failing_l0() -> AffineAmplitudeResult {
        AffineAmplitudeResult { a_recovered: 0.5, a_exact: 1.0, relative_error: 0.5, steps_run: 100, passed: false }
    }
    fn passing_l4() -> crate::user_problem::NoHoleBenchmarkResult {
        crate::user_problem::NoHoleBenchmarkResult {
            sigma_xx_relative_error: 0.0, sigma_yy_over_ref: 0.0, sigma_xy_over_ref: 0.0,
            traction_rms_over_ref: 0.0, load_transfer_ratio: 1.0, passed: true, failures: Vec::new(),
        }
    }
    fn failing_l4() -> crate::user_problem::NoHoleBenchmarkResult {
        crate::user_problem::NoHoleBenchmarkResult {
            sigma_xx_relative_error: 0.5, sigma_yy_over_ref: 0.5, sigma_xy_over_ref: 0.5,
            traction_rms_over_ref: 0.5, load_transfer_ratio: 0.1, passed: false,
            failures: vec!["sigma_xx_relative_error"],
        }
    }

    #[test]
    fn operational_gate_passes_only_when_both_l0_and_l4_pass() {
        let g = evaluate_no_hole_operational_gate(&passing_l0(), &passing_l4());
        assert_eq!(g.status, OperationalStatus::Pass);
        assert_eq!(g.failed_rung, None);
        assert!(g.l0_passed && g.l4_passed);
    }

    #[test]
    fn operational_gate_fails_at_l0_when_l0_fails_even_if_l4_would_pass() {
        // L0 failing is a formulation-level defect - reported as the failed rung even though
        // L4's own numbers (fabricated as "passing" here) look fine; in real operation L0
        // failing panics training before L4 could ever be computed at all (see this module's
        // own doc comment), but the pure function itself must still resolve this combination
        // honestly if ever called with it directly.
        let g = evaluate_no_hole_operational_gate(&failing_l0(), &passing_l4());
        assert_eq!(g.status, OperationalStatus::Fail);
        assert_eq!(g.failed_rung, Some("L0"));
    }

    #[test]
    fn operational_gate_fails_at_l4_when_only_l4_fails() {
        let g = evaluate_no_hole_operational_gate(&passing_l0(), &failing_l4());
        assert_eq!(g.status, OperationalStatus::Fail);
        assert_eq!(g.failed_rung, Some("L4"));
        assert!(g.l0_passed && !g.l4_passed);
    }
}
