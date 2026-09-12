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

/// Issue #62 PH3-10: convergence evidence beyond `step == max_steps`.
///
/// Plan text (verbatim): "The current run reached: 1999 / 2000 steps, final gradient norm ~
/// 0.484. This does not by itself prove optimization convergence... A run SHALL NOT be
/// declared converged merely because step == max_steps or because a loss plateau detector
/// stopped it." This module is the machine-readable answer: given the real per-tick trend of
/// three independent signals collected over a run (loss, gradient norm, boundary-condition
/// residual - the same `bc_residual_rms` already computed at the vis cadence), classify each
/// signal's own trajectory and combine them into one verdict, instead of relying on step count
/// alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Second half of the series meaningfully better than the first half.
    Improving,
    /// No meaningful change between halves (within `TREND_RELATIVE_THRESHOLD`).
    Plateaued,
    /// Second half meaningfully WORSE than the first half - the exact signature PH3-08's own
    /// investigation would have caught immediately had this existed at the time.
    Worsening,
    /// Fewer than `MIN_TREND_SAMPLES` points were collected - too little evidence to classify
    /// either way. Distinct from `Plateaued` (which is a real, evidenced verdict) so a caller
    /// can never mistake "we didn't look" for "we looked and it's flat".
    InsufficientData,
}

/// Below this many samples, `classify_trend` refuses to guess - see `TrendDirection::
/// InsufficientData`.
const MIN_TREND_SAMPLES: usize = 4;

/// Relative change between the first-half and second-half mean needed to call a real trend
/// rather than noise. 5% is a deliberately loose bar - this is a coarse, always-available
/// convergence SIGNAL, not a precision statistical test.
const TREND_RELATIVE_THRESHOLD: f64 = 0.05;

/// Classifies a real time series (in collection order - NOT sorted) by comparing the mean of
/// its first half against its second half. `lower_is_better` is `true` for loss/gradient-norm/
/// residual-style metrics (smaller = better) and `false` for a metric where growth is the good
/// direction.
pub fn classify_trend(values: &[f64], lower_is_better: bool) -> TrendDirection {
    if values.len() < MIN_TREND_SAMPLES {
        return TrendDirection::InsufficientData;
    }
    let half = values.len() / 2;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let first_half_mean = mean(&values[..half]);
    let second_half_mean = mean(&values[half..]);
    let denom = first_half_mean.abs().max(1e-300);
    let relative_change = (second_half_mean - first_half_mean) / denom;
    let improved = if lower_is_better { relative_change < -TREND_RELATIVE_THRESHOLD }
                    else { relative_change > TREND_RELATIVE_THRESHOLD };
    let worsened = if lower_is_better { relative_change > TREND_RELATIVE_THRESHOLD }
                   else { relative_change < -TREND_RELATIVE_THRESHOLD };
    if improved { TrendDirection::Improving }
    else if worsened { TrendDirection::Worsening }
    else { TrendDirection::Plateaued }
}

/// The combined, multi-signal convergence verdict for one run - PH3-10's own deliverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunConvergenceEvidence {
    pub n_samples: usize,
    pub loss_trend: TrendDirection,
    pub grad_norm_trend: TrendDirection,
    pub bc_residual_trend: TrendDirection,
    /// `true` iff NEITHER `loss_trend` NOR `bc_residual_trend` is `Worsening` - the two
    /// signals that directly reflect physical solution quality. `grad_norm_trend` is
    /// deliberately excluded from this gate: gradient norm can legitimately oscillate/rise
    /// near a saddle or during a SAW-BRDR reweighting event without the solution itself
    /// getting worse (see PH3-08/PH3-09's own real evidence of exactly this kind of
    /// non-monotonic-loss-but-improving-benchmark behavior), so treating it as a hard veto
    /// would produce false negatives on runs this project has already proven are fine.
    pub plausibly_converged: bool,
}

/// Combines three real per-tick series (loss, gradient norm, BC residual RMS - all "lower is
/// better") into one `RunConvergenceEvidence`. This is what a caller runs INSTEAD of trusting
/// `step == max_steps` alone.
pub fn assess_convergence(loss: &[f64], grad_norm: &[f64], bc_residual: &[f64]) -> RunConvergenceEvidence {
    let loss_trend = classify_trend(loss, true);
    let grad_norm_trend = classify_trend(grad_norm, true);
    let bc_residual_trend = classify_trend(bc_residual, true);
    let plausibly_converged = loss_trend != TrendDirection::Worsening && bc_residual_trend != TrendDirection::Worsening;
    RunConvergenceEvidence {
        n_samples: loss.len().min(grad_norm.len()).min(bc_residual.len()),
        loss_trend, grad_norm_trend, bc_residual_trend, plausibly_converged,
    }
}

/// Issue #62 PH3-15: the hole/Kt activation gate - "the hole benchmark SHALL NOT be considered
/// operational merely because Kt can be computed." Eligibility requires ALL FIVE of the plan's
/// own named preconditions to hold; this function makes that combination a real, machine-
/// checkable value instead of an implicit assumption. Every input is a plain `bool` the CALLER
/// must supply from real, dated evidence (this function has no way to independently verify a
/// claim) - see `crates::runner::tests::real_ph3_15_hole_activation_gate_reflects_the_actual_
/// current_phase_3_evidence` for the one place this project's OWN real findings are plugged in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoleActivationGateResult {
    pub eligible: bool,
    pub no_hole_benchmark_passed: bool,
    pub variational_path_passed: bool,
    pub measure_aware_path_passed: bool,
    pub authoritative_field_audit_passed: bool,
    pub derivative_path_audit_passed: bool,
}

impl HoleActivationGateResult {
    /// Names of every failed precondition, in the plan's own listed order - empty iff
    /// `eligible`.
    pub fn failed_conditions(&self) -> Vec<&'static str> {
        let mut failed = Vec::new();
        if !self.no_hole_benchmark_passed { failed.push("no_hole_benchmark"); }
        if !self.variational_path_passed { failed.push("variational_path"); }
        if !self.measure_aware_path_passed { failed.push("measure_aware_path"); }
        if !self.authoritative_field_audit_passed { failed.push("authoritative_field_audit"); }
        if !self.derivative_path_audit_passed { failed.push("derivative_path_audit"); }
        failed
    }
}

/// Combines the plan's own 5 named preconditions (issue #62 §19) into one eligibility verdict.
/// `eligible` is `true` only when every single one holds - a pure AND, no partial credit, no
/// threshold-tuning knob (that would be exactly the kind of "benchmark-specific hack" issue #62
/// §3.1 forbids for a gate whose entire purpose is refusing to rubber-stamp readiness).
pub fn evaluate_hole_activation_gate(
    no_hole_benchmark_passed: bool,
    variational_path_passed: bool,
    measure_aware_path_passed: bool,
    authoritative_field_audit_passed: bool,
    derivative_path_audit_passed: bool,
) -> HoleActivationGateResult {
    HoleActivationGateResult {
        eligible: no_hole_benchmark_passed && variational_path_passed && measure_aware_path_passed
            && authoritative_field_audit_passed && derivative_path_audit_passed,
        no_hole_benchmark_passed,
        variational_path_passed,
        measure_aware_path_passed,
        authoritative_field_audit_passed,
        derivative_path_audit_passed,
    }
}

/// The current, dated, real Phase 3 evidence behind each of `evaluate_hole_activation_gate`'s 5
/// inputs - a SINGLE source of truth shared by `runner::tests::real_ph3_15_hole_activation_
/// gate_reflects_the_actual_current_phase_3_evidence` (which asserts against these) and
/// `app-egui`'s own Hole Stress Analysis card (which reads these to render an honest "not yet
/// operational" banner on any holed run) - see each constant's own doc comment for the exact
/// manifest entry establishing it. Update these ONLY when a NEW, real, dated verification
/// result changes one of them - never to "make the gate pass".
pub mod current_phase_3_evidence {
    /// PH3-09: `Debug_run/baseline_legacy_no_hole/` resumed to 2800 steps, `traction_rms_
    /// over_ref` dropped under 1%, full P2-14 hard benchmark PASS.
    pub const NO_HOLE_BENCHMARK_PASSED: bool = true;
    /// PH3-14: a 16000-step run of the shipped `variational_no_hole_plate.toml` config
    /// DIVERGES past step ~7000 (`sigma_xx_relative_error` reaching 539%), not merely under-
    /// converges.
    pub const VARIATIONAL_PATH_PASSED: bool = false;
    /// PH3-15's own isolating test: measure-aware training under the SAME Hybrid formulation
    /// PH3-09 proved converges still FAILS all 5 hard thresholds at the identical 2800-step
    /// budget.
    pub const MEASURE_AWARE_PATH_PASSED: bool = false;
    /// PH3-07: `field_graph::consumer_field_report`, a real, tested registry cross-validated
    /// against the Kt investigation's own written findings.
    pub const AUTHORITATIVE_FIELD_AUDIT_PASSED: bool = true;
    /// PH3-06: `differential_operator::ad_fd_strain_agreement`, real, live AD-vs-FD cross-
    /// validation during actual training, agreement well within tolerance.
    pub const DERIVATIVE_PATH_AUDIT_PASSED: bool = true;
}

/// Convenience wrapper: evaluates the gate against this project's own current, dated evidence
/// (`current_phase_3_evidence`) rather than requiring every caller to spell out all 5 booleans.
pub fn evaluate_hole_activation_gate_for_this_project() -> HoleActivationGateResult {
    use current_phase_3_evidence::*;
    evaluate_hole_activation_gate(
        NO_HOLE_BENCHMARK_PASSED, VARIATIONAL_PATH_PASSED, MEASURE_AWARE_PATH_PASSED,
        AUTHORITATIVE_FIELD_AUDIT_PASSED, DERIVATIVE_PATH_AUDIT_PASSED,
    )
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

    #[test]
    fn classify_trend_reports_insufficient_data_below_the_minimum_sample_count() {
        assert_eq!(classify_trend(&[1.0, 0.5, 0.1], true), TrendDirection::InsufficientData);
        assert_eq!(classify_trend(&[], true), TrendDirection::InsufficientData);
    }

    #[test]
    fn classify_trend_detects_improving_when_lower_is_better_and_values_drop() {
        assert_eq!(classify_trend(&[1.0, 1.0, 0.1, 0.1], true), TrendDirection::Improving);
    }

    #[test]
    fn classify_trend_detects_worsening_when_lower_is_better_and_values_rise() {
        assert_eq!(classify_trend(&[0.1, 0.1, 1.0, 1.0], true), TrendDirection::Worsening);
    }

    #[test]
    fn classify_trend_reports_plateaued_when_the_two_halves_are_nearly_equal() {
        assert_eq!(classify_trend(&[1.0, 1.01, 0.99, 1.0], true), TrendDirection::Plateaued);
    }

    #[test]
    fn classify_trend_direction_flips_correctly_when_higher_is_better() {
        assert_eq!(classify_trend(&[0.1, 0.1, 1.0, 1.0], false), TrendDirection::Improving);
        assert_eq!(classify_trend(&[1.0, 1.0, 0.1, 0.1], false), TrendDirection::Worsening);
    }

    #[test]
    fn assess_convergence_is_plausibly_converged_when_loss_and_bc_residual_both_improve_even_if_grad_norm_is_noisy() {
        // Real PH3-08/PH3-09 evidence shape: loss/BC residual genuinely improve while gradient
        // norm itself doesn't monotonically fall - `plausibly_converged` must not be vetoed by
        // grad_norm alone (see `RunConvergenceEvidence::plausibly_converged`'s own doc comment).
        let loss = vec![1.0, 0.8, 0.3, 0.2];
        let grad_norm = vec![0.5, 0.9, 0.4, 0.8]; // noisy, no clean trend either way
        let bc_residual = vec![0.02, 0.018, 0.008, 0.007];
        let evidence = assess_convergence(&loss, &grad_norm, &bc_residual);
        assert_eq!(evidence.loss_trend, TrendDirection::Improving);
        assert_eq!(evidence.bc_residual_trend, TrendDirection::Improving);
        assert!(evidence.plausibly_converged, "{evidence:?}");
    }

    #[test]
    fn assess_convergence_is_not_plausibly_converged_when_bc_residual_worsens_even_though_loss_improves() {
        // The exact failure mode PH3-10's own plan text warns against: a run that merely
        // reached `step == max_steps` with a falling loss can still be physically diverging at
        // the boundary - this must be caught, not hidden behind a falling loss curve.
        let loss = vec![1.0, 0.8, 0.5, 0.3];
        let grad_norm = vec![0.5, 0.4, 0.3, 0.2];
        let bc_residual = vec![0.01, 0.012, 0.05, 0.09];
        let evidence = assess_convergence(&loss, &grad_norm, &bc_residual);
        assert_eq!(evidence.bc_residual_trend, TrendDirection::Worsening);
        assert!(!evidence.plausibly_converged, "{evidence:?}");
    }

    #[test]
    fn assess_convergence_reports_the_true_min_sample_count_across_uneven_length_series() {
        let evidence = assess_convergence(&[1.0, 1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_eq!(evidence.n_samples, 3);
    }

    #[test]
    fn hole_activation_gate_is_eligible_only_when_all_five_conditions_hold() {
        let g = evaluate_hole_activation_gate(true, true, true, true, true);
        assert!(g.eligible);
        assert!(g.failed_conditions().is_empty());
    }

    #[test]
    fn hole_activation_gate_reports_every_failed_condition_by_name_not_just_the_first() {
        let g = evaluate_hole_activation_gate(true, false, false, true, true);
        assert!(!g.eligible);
        assert_eq!(g.failed_conditions(), vec!["variational_path", "measure_aware_path"]);
    }

    #[test]
    fn hole_activation_gate_is_not_eligible_when_only_one_condition_fails() {
        // No partial credit - a single failed precondition still blocks eligibility entirely,
        // per this function's own "pure AND, no threshold-tuning knob" doc comment.
        let g = evaluate_hole_activation_gate(true, true, true, true, false);
        assert!(!g.eligible);
        assert_eq!(g.failed_conditions(), vec!["derivative_path_audit"]);
    }
}
