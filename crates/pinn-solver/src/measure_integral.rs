//! Issue #61 EPIC P2-04: measure-aware integration.
//!
//! `pinn_core::amr::DensitySample`'s own doc comment (Priority 8 of the prior General-PINN
//! pass) already names the real gap this epic closes: "every existing Monte-Carlo
//! domain-integral loss term in this codebase currently [uses] a plain, UNWEIGHTED mean ...
//! which becomes false the moment AMR refines the grid unevenly" — naming `InteriorEnergyTerm`,
//! `ExternalWorkTerm`, and `probe_energy_balance`'s own internal-energy integral explicitly.
//! `probe_energy_balance` (`user_problem.rs`) already computes its internal-energy term
//! CORRECTLY per issue #61's own literal formula (`mean_density * area * thickness` — exactly
//! `Integral_Omega(f) ≈ |Omega|*mean(f)`, generalized by the thickness factor) and its
//! external-work term correctly (`Σ f_i * ds_i * thickness`, an exact per-point arc-length
//! sum) — but both formulas were inlined ad hoc in that one function, not a reusable, tested
//! abstraction, and neither is compensated for nonuniform sampling density.
//!
//! This module extracts and generalizes both patterns into real, standalone, unit-tested
//! functions, adds the missing nonuniform-sampling-aware variant, and refactors
//! `probe_energy_balance` to call through them (real proof of live use per issue #61 §1.3 —
//! this is a refactor of an already-live function, not a new function nobody calls).
//!
//! Deliberately NOT done in this epic (see the P2-04 manifest entry's "Known limitations"):
//! migrating `InteriorEnergyTerm`/`ExternalWorkTerm` — the actual TRAINING loss terms, as
//! opposed to `probe_energy_balance`'s read-only diagnostic — onto this abstraction. Doing so
//! would change the numeric scale of the live optimization objective, which needs the
//! mandatory P2-08 verification ladder (not yet built) to validate before it can land safely;
//! per issue #61 §4's own "no Kt weight tuning SHALL substitute for P2-04 through P2-08"
//! ordering and §1.4's "no destructive refactor" rule, that substitution is P2-15's job.

use pinn_core::amr::{compensation_weights, DensitySample};

/// `Integral_Omega(f) ≈ |Omega| * mean(f)` — issue #61's own literal formula, generalized by a
/// `thickness` factor (this codebase's plates are always finite-thickness, matching
/// `probe_energy_balance`'s existing convention). `measure` is the domain's total area (m²).
/// Empty `samples` returns `0.0` (an empty point set represents no evaluated volume, not a
/// division-by-zero NaN).
pub fn domain_integral(measure: f64, thickness: f64, samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean: f64 = samples.iter().map(|&v| v as f64).sum::<f64>() / samples.len() as f64;
    mean * measure * thickness
}

/// Same as [`domain_integral`], but applies Priority 8's AMR density-compensation weights
/// (`compensation_weights` — per-point multiplier with `mean(w_i) == 1.0` by construction)
/// before averaging, so the estimate stays correct under nonuniform/adaptive sampling instead
/// of only uniform sampling. `density_samples[i]` must correspond to `samples[i]`'s point
/// (same order, same length); a length mismatch (no compensation info available) falls back to
/// [`domain_integral`]'s plain uniform-sampling estimate rather than silently misaligning data.
pub fn domain_integral_weighted(
    measure: f64,
    thickness: f64,
    samples: &[f32],
    density_samples: &[DensitySample],
) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    if density_samples.len() != samples.len() {
        return domain_integral(measure, thickness, samples);
    }
    let weights = compensation_weights(density_samples);
    let n = samples.len() as f64;
    let weighted_sum: f64 = samples.iter().zip(weights.iter()).map(|(&v, &w)| v as f64 * w).sum();
    (weighted_sum / n) * measure * thickness
}

/// Boundary/interface integral: `∮ f ds ≈ Σ f_i * ds_i * thickness` — an exact per-point
/// arc-length-weighted sum, matching `probe_energy_balance`/`probe_reaction_force`'s existing
/// per-edge `ds` treatment (each of a rectangular plate's 4 outer edges has its own `ds`
/// depending on which edge a point lies on, so a single scalar `measure` like
/// [`domain_integral`] takes doesn't fit this shape — the per-point `ds_per_point` already
/// carries the real local measure). `samples` and `ds_per_point` must be the same length
/// (paired per point); mismatched lengths panic rather than silently truncating or padding —
/// silently misaligning a physical measure to the wrong point is exactly the kind of unchecked
/// error issue #61 §1.2 is about, so this fails loudly instead.
pub fn boundary_integral(samples: &[f32], ds_per_point: &[f64], thickness: f64) -> f64 {
    assert_eq!(
        samples.len(), ds_per_point.len(),
        "boundary_integral: samples ({}) and ds_per_point ({}) must be the same length",
        samples.len(), ds_per_point.len(),
    );
    samples.iter().zip(ds_per_point.iter()).map(|(&v, &ds)| v as f64 * ds * thickness).sum()
}

/// Real plate domain area (m²): outer rectangle minus every hole's circular area. Matches
/// `probe_energy_balance`'s pre-existing inline formula exactly (that call site is refactored
/// to call this function instead of repeating the arithmetic — see this module's own tests).
pub fn plate_domain_area(half_w: f64, half_h: f64, hole_radii: &[f64]) -> f64 {
    let plate = 4.0 * half_w * half_h;
    let holes: f64 = hole_radii.iter().map(|&r| std::f64::consts::PI * r * r).sum();
    (plate - holes).max(0.0)
}

/// Real outer-boundary perimeter (m) of a rectangular plate.
pub fn plate_outer_perimeter(half_w: f64, half_h: f64) -> f64 {
    2.0 * (2.0 * half_w + 2.0 * half_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_integral_of_constant_field_equals_measure_times_thickness_times_value() {
        let samples = vec![3.0_f32; 500];
        let result = domain_integral(2.0, 0.01, &samples);
        assert!((result - 3.0 * 2.0 * 0.01).abs() < 1e-12);
    }

    #[test]
    fn domain_integral_matches_hand_computed_mean_times_area_times_thickness() {
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0]; // mean = 2.5
        let result = domain_integral(10.0, 0.5, &samples);
        assert!((result - (2.5 * 10.0 * 0.5)).abs() < 1e-9);
    }

    #[test]
    fn domain_integral_empty_samples_is_zero_not_nan() {
        let result = domain_integral(10.0, 0.5, &[]);
        assert_eq!(result, 0.0);
    }

    /// Issue #61 P2-04's own acceptance wording: "same analytic integral must converge under
    /// uniform/nonuniform/adaptive sampling." Reproduces exactly the failure mode `DensitySample`'s
    /// own doc comment describes: an AMR-style nonuniform point set that over-samples a small
    /// region (biasing the PLAIN mean away from the true area-weighted average) for a KNOWN
    /// linear field `f(x) = x` over `x in [-1, 1]` (true area-weighted mean = 0.0, by symmetry).
    /// Two "leaf" regions of very different sizes are sampled at DIFFERENT densities (small
    /// region over-sampled, matching real AMR refinement behavior) — proves the unweighted
    /// estimator is measurably biased AND that `domain_integral_weighted` recovers the true
    /// value.
    #[test]
    fn domain_integral_weighted_recovers_the_true_average_under_amr_biased_nonuniform_sampling() {
        // Big leaf: x in [-1, -0.5], area = 0.5*2 = 1.0 (using domain height 2 for a 2D leaf
        // area analogue) - sparsely sampled (5 points, all at x=-0.75).
        // Small leaf: x in [0.9, 1.0], area = 0.1*2 = 0.2 - densely oversampled (45 points, all
        // at x=0.95), exactly the "smaller cells get more points" AMR bias pattern.
        // True area-weighted mean = ((-0.75)*1.0 + 0.95*0.2) / (1.0 + 0.2) = -0.4667
        // `DensitySample::leaf_area` is the PER-POINT share of its leaf's total area (the
        // leaf's total area divided by how many points were drawn from it), matching
        // `pinn_core::amr`'s own `compensated_mean_recovers_true_area_weighted_average_that_
        // naive_mean_misses` test convention exactly.
        let mut samples: Vec<f32> = Vec::new();
        let mut density_samples: Vec<DensitySample> = Vec::new();
        for _ in 0..5 {
            samples.push(-0.75);
            density_samples.push(DensitySample { point: [-0.75, 0.0], leaf_area: 1.0 / 5.0 });
        }
        for _ in 0..45 {
            samples.push(0.95);
            density_samples.push(DensitySample { point: [0.95, 0.0], leaf_area: 0.2 / 45.0 });
        }
        let true_area_weighted_mean = ((-0.75_f64) * 1.0 + 0.95 * 0.2) / (1.0 + 0.2);

        let unweighted = domain_integral(1.0, 1.0, &samples);
        let weighted = domain_integral_weighted(1.0, 1.0, &samples, &density_samples);

        // The plain mean is measurably biased toward the oversampled small leaf's value.
        assert!(
            (unweighted - true_area_weighted_mean).abs() > 0.1,
            "test setup sanity check: plain mean ({unweighted}) should be visibly biased away \
             from the true area-weighted mean ({true_area_weighted_mean}) for this to be a real \
             regression guard",
        );
        // The weighted estimator recovers the true value (f32-sample precision, not f64).
        assert!(
            (weighted - true_area_weighted_mean).abs() < 1e-6,
            "weighted={weighted} should match true_area_weighted_mean={true_area_weighted_mean}",
        );
    }

    #[test]
    fn domain_integral_weighted_matches_unweighted_when_all_leaf_areas_are_equal() {
        // Uniform sampling (every leaf the same size) - the weighted and unweighted estimators
        // must agree exactly, since compensation_weights degenerates to all-1.0 (real
        // convergence-under-uniform-sampling requirement).
        let samples = vec![1.0_f32, 5.0, 3.0, 7.0, 2.0];
        let density_samples: Vec<DensitySample> = samples.iter()
            .map(|_| DensitySample { point: [0.0, 0.0], leaf_area: 1.0 })
            .collect();
        let unweighted = domain_integral(4.0, 1.0, &samples);
        let weighted = domain_integral_weighted(4.0, 1.0, &samples, &density_samples);
        assert!((unweighted - weighted).abs() < 1e-9);
    }

    #[test]
    fn domain_integral_weighted_falls_back_to_unweighted_on_length_mismatch() {
        let samples = vec![1.0_f32, 2.0, 3.0];
        let density_samples = vec![DensitySample { point: [0.0, 0.0], leaf_area: 1.0 }]; // wrong length
        let fallback = domain_integral_weighted(2.0, 1.0, &samples, &density_samples);
        let plain = domain_integral(2.0, 1.0, &samples);
        assert_eq!(fallback, plain);
    }

    #[test]
    fn boundary_integral_matches_hand_computed_line_integral() {
        let samples = vec![2.0_f32, 4.0, 6.0];
        let ds = vec![0.1, 0.2, 0.3];
        let thickness = 0.5;
        let expected = 2.0 * 0.1 * 0.5 + 4.0 * 0.2 * 0.5 + 6.0 * 0.3 * 0.5;
        assert!((boundary_integral(&samples, &ds, thickness) - expected).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "must be the same length")]
    fn boundary_integral_panics_on_length_mismatch() {
        boundary_integral(&[1.0, 2.0], &[0.1], 1.0);
    }

    #[test]
    fn plate_domain_area_subtracts_hole_areas_from_the_outer_rectangle() {
        let half_w = 0.1;
        let half_h = 0.1;
        let no_hole = plate_domain_area(half_w, half_h, &[]);
        assert!((no_hole - (4.0 * half_w * half_h)).abs() < 1e-12);

        let radius = 0.02;
        let with_hole = plate_domain_area(half_w, half_h, &[radius]);
        let expected = 4.0 * half_w * half_h - std::f64::consts::PI * radius * radius;
        assert!((with_hole - expected).abs() < 1e-12);
    }

    #[test]
    fn plate_outer_perimeter_matches_hand_computed_value() {
        let perimeter = plate_outer_perimeter(0.1, 0.05);
        assert!((perimeter - (2.0 * (0.2 + 0.1))).abs() < 1e-12);
    }
}
