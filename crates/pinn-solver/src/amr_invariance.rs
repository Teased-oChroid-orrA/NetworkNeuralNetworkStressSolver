//! Issue #61 EPIC P2-11: adaptive sampling invariance.
//!
//! AMR (`pinn_core::amr::AdaptiveGrid`) refines WHERE it samples, based on observed residuals -
//! it is an ESTIMATOR REFINEMENT (concentrating samples where the integrand is estimated to
//! vary most, for a tighter estimate at a given point budget), not, on its own, PROOF that a
//! domain integral has converged to the true value (issue #61 §3's own explicit concern,
//! echoed for the Kt QoI in P2-10's own `kt_convergence_check`). A refinement step that
//! silently introduced a NEW bias (e.g. via an unnormalized point-density compensation) would
//! be invisible without a real check against a KNOWN answer.
//!
//! [`check_analytic_integral_invariance_under_amr_refinement`] runs a real `AdaptiveGrid`
//! refinement cycle (real residual injection -> real `adapt()` call, changing the actual point
//! cloud) on a KNOWN analytic field, and confirms `measure_integral::domain_integral_weighted`
//! (P2-04's AMR-aware estimator) stays within tolerance of the true value BOTH before and
//! after refinement - proving refinement preserves correctness rather than merely proving it
//! doesn't obviously explode.

use pinn_core::amr::{AdaptiveGrid, AmrDomain};

use crate::measure_integral::domain_integral_weighted;

/// See this module's own doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmrInvarianceReport {
    pub estimate_before: f64,
    pub estimate_after: f64,
    pub true_value: f64,
    pub relative_error_before: f64,
    pub relative_error_after: f64,
    /// True iff BOTH `relative_error_before` and `relative_error_after` are under `tolerance` -
    /// refinement changed the point cloud without breaking correctness.
    pub invariant: bool,
}

/// Runs one real refinement cycle on `grid` (residuals = `|analytic_fn(x,y)|` at each currently-
/// sampled point - a real, physically-motivated refinement driver: cells where the integrand's
/// own magnitude is largest get refined, exactly how a real training-residual-driven refinement
/// would behave), and confirms the measure-aware estimate of `analytic_fn`'s domain integral
/// stays within `tolerance` of `true_value` both before and after.
pub fn check_analytic_integral_invariance_under_amr_refinement<G: AmrDomain + Clone>(
    grid: &mut AdaptiveGrid<G>,
    measure: f64,
    thickness: f64,
    analytic_fn: impl Fn(f64, f64) -> f64,
    true_value: f64,
    tolerance: f64,
) -> AmrInvarianceReport {
    let samples_before = grid.sample_points_with_density();
    let values_before: Vec<f32> = samples_before.iter().map(|s| analytic_fn(s.point[0], s.point[1]) as f32).collect();
    let estimate_before = domain_integral_weighted(measure, thickness, &values_before, &samples_before);

    let residuals: Vec<f32> = values_before.iter().map(|v| v.abs()).collect();
    grid.update_residuals(&residuals);
    grid.adapt();

    let samples_after = grid.sample_points_with_density();
    let values_after: Vec<f32> = samples_after.iter().map(|s| analytic_fn(s.point[0], s.point[1]) as f32).collect();
    let estimate_after = domain_integral_weighted(measure, thickness, &values_after, &samples_after);

    let rel = |estimate: f64| if true_value.abs() > 1e-30 {
        (estimate - true_value).abs() / true_value.abs()
    } else {
        (estimate - true_value).abs()
    };
    let relative_error_before = rel(estimate_before);
    let relative_error_after = rel(estimate_after);
    let invariant = relative_error_before < tolerance && relative_error_after < tolerance;

    AmrInvarianceReport { estimate_before, estimate_after, true_value, relative_error_before, relative_error_after, invariant }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::amr::AmrtConfig;
    use pinn_core::geometry::{GeometryConfig, HoleType, SymmetryMode};

    fn no_hole_full_square() -> GeometryConfig {
        GeometryConfig { half_w: 1.0, half_h: 1.0, thickness: 1.0, hole: HoleType::None, symmetry: SymmetryMode::Full }
    }

    #[test]
    fn constant_field_integral_is_invariant_under_amr_refinement() {
        let geom = no_hole_full_square();
        let mut grid = AdaptiveGrid::new(&geom, AmrtConfig::default());
        // f(x,y) = 5.0 (constant) - true domain integral = 5.0 * area(4.0) * thickness(1.0) = 20.0.
        let true_value = 5.0 * 4.0 * 1.0;
        let report = check_analytic_integral_invariance_under_amr_refinement(
            &mut grid, 4.0, 1.0, |_x, _y| 5.0, true_value, 0.05,
        );
        assert!(report.invariant, "{report:?}");
        assert!(report.relative_error_before < 0.05, "{report:?}");
        assert!(report.relative_error_after < 0.05, "{report:?}");
    }

    /// A genuinely non-trivial analytic field with a hand-derivable exact integral:
    /// `mean(x^2+y^2)` over `[-1,1]x[-1,1]` = `mean(x^2) + mean(y^2)` = `1/3 + 1/3` = `2/3`
    /// (since `mean(x^2)` over `[-1,1]` = `(integral of x^2 dx from -1 to 1) / 2` = `(2/3)/2` =
    /// `1/3`). True domain integral = `(2/3) * area(4.0) * thickness(1.0)` = `8/3`.
    #[test]
    fn quadratic_field_integral_is_invariant_under_amr_refinement() {
        let geom = no_hole_full_square();
        let mut grid = AdaptiveGrid::new(&geom, AmrtConfig::default());
        let true_value = (2.0 / 3.0) * 4.0 * 1.0;
        let report = check_analytic_integral_invariance_under_amr_refinement(
            &mut grid, 4.0, 1.0, |x, y| x * x + y * y, true_value, 0.15,
        );
        assert!(report.invariant, "{report:?}");
    }

    #[test]
    fn refinement_actually_changes_the_point_cloud_not_a_vacuous_check() {
        // Sanity check that this test genuinely exercises `adapt()`, not a no-op - the point
        // count must differ (or at minimum the grid's own adapt_count must increment),
        // otherwise "invariant under refinement" would be trivially true for the wrong reason.
        let geom = no_hole_full_square();
        let mut grid = AdaptiveGrid::new(&geom, AmrtConfig::default());
        let n_before = grid.sample_points_with_density().len();
        let _ = check_analytic_integral_invariance_under_amr_refinement(
            &mut grid, 4.0, 1.0, |x, y| x * x + y * y, 8.0 / 3.0, 0.15,
        );
        assert_eq!(grid.adapt_count(), 1, "the grid must have actually been adapted once");
        let n_after = grid.sample_points_with_density().len();
        assert!(n_after != n_before || grid.max_depth() > 0, "refinement should change something observable");
    }
}
