//! Issue #61 EPIC P2-07: gauge/nullspace handling for pure-Neumann rigid-body modes.
//!
//! A linear-elasticity BVP with ONLY traction (Neumann) boundary conditions and no essential
//! (Dirichlet) constraint anywhere is only determined up to an additive rigid-body motion (in
//! 2D: 2 translational + 1 rotational degree of freedom) — strain energy, traction residuals,
//! and equilibrium are all invariant under a constant displacement offset (and, for the
//! rotational mode, under a small rigid rotation). This codebase's real examples hit exactly
//! this case: `no_hole_plate.toml` (no holes at all) and `single_hole_plate.toml` (its one
//! hole set to `HoleBc::Free`) are BOTH pure-Neumann — see [`pinn_core::user_geometry::
//! UserGeometry::is_pure_neumann`]. Kirsch (`DisplacementAnchorTerm`) and pin-lug
//! (`LugShankAnchorTerm`) never hit this gap because they always have an essential anchor.
//!
//! This module closes the TRANSLATIONAL half of that gap: [`TranslationGaugeTerm`] (in
//! `user_problem.rs`, since it needs `USER_DOMAIN`/`DomainForwardOutputs` from that module)
//! penalizes the mean interior displacement (`mean(u)^2 + mean(v)^2`) toward zero — a standard
//! "mean-field constraint" gauge-fixing technique, one of the three named in issue #61 P2-07's
//! own acceptance text ("mean-field constraints, point anchors, nullspace projection"). It is
//! registered ONLY when [`UserGeometry::is_pure_neumann`] is true — never for a problem that
//! already has a real Dirichlet anchor, where it would be redundant, and never silently
//! altering an already-well-posed problem.
//!
//! The ROTATIONAL rigid-body mode (fixing `mean(x*v - y*u)` toward zero — the standard 2D
//! infinitesimal-rotation nullspace projection) is NOT implemented here: it needs each
//! collocation point's physical `(x, y)` coordinates, which `DomainForwardOutputs` does not
//! currently carry (only `raw_out`/`strains`/`normals`/`shifted_stress`/`hessian`). Adding a
//! coordinates field would mean touching `compute_domain_forwards` and every existing
//! `DomainForwardOutputs`/`Computed` construction site across the codebase — the kind of
//! broad, structural plumbing change issue #61 §1.4 reserves for P2-15's migration step, not a
//! single new term. Tracked as a known limitation in the P2-07 manifest entry, not silently
//! dropped: for this codebase's real symmetric-loading examples (uniaxial tension, no applied
//! moment), the true solution has zero net rotation regardless, so the missing rotational
//! gauge-fix is a real but low-impact gap for the configurations that exist today.

use crate::problem::BoundaryValueProblem;

/// Convenience re-export of the geometry-level check this module's registration logic uses -
/// see [`pinn_core::user_geometry::UserGeometry::is_pure_neumann`] for the real definition.
/// Operates on the problem's OWN declared essential-BC terms (`boundary_kind() == Some(
/// Dirichlet)`) rather than re-deriving from geometry, so it stays correct for any future
/// problem type that gains a different essential-constraint mechanism, not just holes.
pub fn has_any_essential_constraint(problem: &dyn BoundaryValueProblem) -> bool {
    problem.loss_terms().iter()
        .any(|t| matches!(t.boundary_kind(), Some(crate::problem::BoundaryOperatorKind::Dirichlet)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kirsch_problem::KirschProblem;
    use pinn_core::material::MaterialProps;

    #[test]
    fn kirsch_has_an_essential_constraint_via_displacement_anchor() {
        let problem = KirschProblem::new(MaterialProps::al7075_t6(), 5, 4000, 3.0);
        assert!(has_any_essential_constraint(&problem), "Kirsch's DisplacementAnchorTerm is Dirichlet");
    }
}
