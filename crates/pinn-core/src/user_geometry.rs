//! Geometry representation for user-defined problems (the "design a joint from scratch"
//! ingestion path) — deliberately **independent** of [`crate::geometry::GeometryConfig`],
//! which hardcodes exactly one hole (`hole: HoleType`, singular) and is load-bearing for the
//! frozen Kirsch/pin-lug paths. Supporting N holes through the shared struct would risk
//! those paths for no benefit; this type exists so user-defined problems can have an
//! arbitrary hole count without touching `GeometryConfig` at all.
//!
//! A [`UserGeometry`]'s real geometry is read directly by
//! `pinn_solver::user_problem::UserSamplingStrategy` — it does NOT get encoded into a
//! `GeometryConfig`. The [`DomainSamplingStrategy`](crate::problem::DomainSamplingStrategy)
//! trait's methods take a `&GeometryConfig` parameter that concrete strategies are free to
//! ignore in favor of their own captured state (an established, precedented pattern — see
//! `FakeInterfaceSampling` in `pinn-core/src/problem.rs`'s own tests). [`to_placeholder`]
//! builds the inert `GeometryConfig` `DomainSpec.geometry` still requires as a field, sized
//! to match this geometry's real bounding box (so any code that innocently reads
//! `geom.x_range()`/`y_range()`/`half_w`/`half_h` for pure bounding-box purposes — e.g.
//! normalization — still gets a correct answer), with `hole: HoleType::None` (there is no
//! single hole to encode) and `symmetry: SymmetryMode::Full` (v1 user-defined problems don't
//! support symmetry reduction).

use serde::{Deserialize, Serialize};

use crate::geometry::{GeometryConfig, HoleType, SymmetryMode};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HoleBc {
    /// Traction-free hole boundary (natural/Neumann, zero prescribed traction).
    Free,
    /// Zero-displacement hole boundary (soft Dirichlet penalty — see
    /// `pinn_solver::user_problem`'s module doc for why this is soft, not a hard ansatz).
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HoleSpec {
    /// Hole center, physical coordinates [m], relative to the plate's own center.
    pub center: [f64; 2],
    /// Hole radius [m].
    pub radius: f64,
    pub bc: HoleBc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserGeometry {
    /// Plate half-width in x [m] — full width = 2*half_w.
    pub half_w: f64,
    /// Plate half-height in y [m] — full height = 2*half_h.
    pub half_h: f64,
    /// Thickness [m] (for plane-stress scaling).
    pub thickness: f64,
    pub holes: Vec<HoleSpec>,
}

impl UserGeometry {
    /// True if `(x, y)` is inside the plate's rectangular bound and outside every hole.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        if x < -self.half_w || x > self.half_w || y < -self.half_h || y > self.half_h {
            return false;
        }
        for hole in &self.holes {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            if dx * dx + dy * dy < hole.radius * hole.radius {
                return false;
            }
        }
        true
    }

    /// Returns a clone with every hole's radius inflated by `margin_m` — used ONLY to build
    /// an `AdaptiveGrid<UserGeometry>`'s own containment gate (via `amr::AmrDomain::contains`)
    /// for collocation purposes, NEVER for physics (hole boundary-condition terms), display,
    /// or the real `contains`/masking semantics above. AMR's leaf-cell-center containment
    /// check has no margin of its own, and quadtree cells aren't boundary-aligned, so without
    /// this a hole-zone-refined cell's center can legitimately satisfy `contains()` (r >
    /// radius) while still being close enough to the true edge that an FD stencil there
    /// crosses back inside the hole — silently corrupting the interior-energy/constitutive-
    /// consistency signal exactly where AMR concentrates the most collocation density. See
    /// `powershell_tool/CLAUDE.md`'s Kt investigation for the real, measured margin-vs-cell-
    /// size comparison that confirmed this as a genuine (not merely theoretical) gap.
    pub fn inflated_for_collocation(&self, margin_m: f64) -> Self {
        let mut inflated = self.clone();
        for hole in &mut inflated.holes {
            hole.radius += margin_m;
        }
        inflated
    }

    /// Positional-Fourier-feature count the network's input should use for this geometry.
    /// Currently always `0` (raw x,y,z, no embedding) - see below for why, despite a real
    /// attempt to enable it.
    ///
    /// This mechanism exists (and every consumer downstream of it is real, generalized
    /// infrastructure, not dead code) because of a genuine, tested hypothesis: Kirsch's own
    /// problem uses positional-Fourier embedding specifically for its hole
    /// (`pinn_solver::engine::EngineParams::analyze`: `let n_fourier = if has_hole { 8_usize }
    /// else { 0 };`, commented "corrects spectral bias near hole") and achieves real Kt
    /// convergence; the generalized N-hole path never had it
    /// (`training_core::compute_domain_forwards` hardcoded `n_fourier = 0` unconditionally,
    /// with a comment explaining only why it skips the *hard Dirichlet ansatz* Fourier
    /// embedding was originally paired with there - not a deliberate decision to omit it for
    /// holed geometries). After ruling out weighting (capping `dynamic_lam_h_cap`, matching
    /// `constitutive_consistency_weight` to it) and sampling density (a 2.47x guaranteed
    /// hole-zone collocation increase) as the cause of a plate-with-hole run's Kt staying ~0,
    /// porting Kirsch's own Fourier fix was the next well-motivated, evidence-driven step.
    ///
    /// Measured result, not assumed: enabling `n_fourier = 8` for a holed geometry made the
    /// interior PDE (constitutive-consistency) residual RMS **~28x WORSE** (2.80e7 Pa vs.
    /// 9.85e5 Pa without it, same 8000-step config) while total_loss converged FASTER and Kt
    /// stayed at ~0 either way. Interpretation: the higher-frequency basis let the network fit
    /// the boundary/traction collocation points more precisely while oscillating wildly
    /// between them - a well-known Fourier-feature pitfall when embedding frequency outstrips
    /// collocation density, and a straightforward port of Kirsch's own working fix without
    /// re-deriving whether its frequency/density balance holds for a different point-sampling
    /// scheme. A real negative result, kept disabled (not deleted) so a future attempt (e.g.
    /// paired with denser boundary-adjacent sampling, or a lower `n_fourier`) doesn't have to
    /// re-build this plumbing from scratch or re-discover this pitfall blind.
    pub fn n_fourier(&self) -> usize {
        let _ = &self.holes; // kept as a parameter for when this is revisited - see doc comment
        0
    }

    /// Network input dimension implied by [`Self::n_fourier`] — `3` (raw x,y,z) when there's
    /// no Fourier embedding, `4 * n_fourier` when there is. Mirrors `pinn_solver::engine::
    /// EngineParams::net_input_dim`'s identical formula.
    pub fn net_input_dim(&self) -> usize {
        let nf = self.n_fourier();
        if nf > 0 { 4 * nf } else { 3 }
    }

    /// Inert placeholder `GeometryConfig` sized to this geometry's real bounding box — see
    /// the module doc comment for why this is safe and what it's actually used for.
    pub fn to_placeholder(&self) -> GeometryConfig {
        GeometryConfig {
            half_w: self.half_w,
            half_h: self.half_h,
            thickness: self.thickness,
            hole: HoleType::None,
            symmetry: SymmetryMode::Full,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_hole_geometry() -> UserGeometry {
        UserGeometry {
            half_w: 1.0,
            half_h: 1.0,
            thickness: 0.1,
            holes: vec![
                HoleSpec { center: [-0.5, 0.0], radius: 0.1, bc: HoleBc::Free },
                HoleSpec { center: [0.5, 0.0], radius: 0.1, bc: HoleBc::Fixed },
            ],
        }
    }

    #[test]
    fn contains_rejects_points_outside_the_rectangle() {
        let geom = two_hole_geometry();
        assert!(!geom.contains(1.5, 0.0));
        assert!(!geom.contains(0.0, -1.5));
    }

    #[test]
    fn contains_rejects_points_inside_either_hole() {
        let geom = two_hole_geometry();
        assert!(!geom.contains(-0.5, 0.0)); // center of first hole
        assert!(!geom.contains(0.55, 0.02)); // inside second hole, off-center
    }

    #[test]
    fn contains_accepts_a_point_in_the_plate_between_holes() {
        let geom = two_hole_geometry();
        assert!(geom.contains(0.0, 0.0));
    }

    #[test]
    fn to_placeholder_preserves_bounding_box_and_has_no_hole() {
        let geom = two_hole_geometry();
        let placeholder = geom.to_placeholder();
        assert_eq!(placeholder.half_w, geom.half_w);
        assert_eq!(placeholder.half_h, geom.half_h);
        assert_eq!(placeholder.thickness, geom.thickness);
        assert_eq!(placeholder.hole, HoleType::None);
        assert_eq!(placeholder.symmetry, SymmetryMode::Full);
    }

    #[test]
    fn inflated_for_collocation_grows_every_hole_radius_by_the_margin_and_nothing_else() {
        let geom = two_hole_geometry();
        let margin = 0.003;
        let inflated = geom.inflated_for_collocation(margin);
        assert_eq!(inflated.half_w, geom.half_w);
        assert_eq!(inflated.half_h, geom.half_h);
        assert_eq!(inflated.thickness, geom.thickness);
        assert_eq!(inflated.holes.len(), geom.holes.len());
        for (orig, grown) in geom.holes.iter().zip(inflated.holes.iter()) {
            assert_eq!(grown.center, orig.center);
            assert_eq!(grown.bc, orig.bc);
            assert!((grown.radius - (orig.radius + margin)).abs() < 1e-15);
        }
    }

    #[test]
    fn inflated_for_collocation_rejects_points_the_original_geometry_would_accept() {
        let geom = two_hole_geometry();
        let margin = 0.02; // deliberately large relative to this fixture's 0.1 m radius
        let inflated = geom.inflated_for_collocation(margin);
        let hole = geom.holes[0];
        let just_outside_true_radius = (hole.center[0] + hole.radius + margin * 0.5, hole.center[1]);
        assert!(geom.contains(just_outside_true_radius.0, just_outside_true_radius.1));
        assert!(!inflated.contains(just_outside_true_radius.0, just_outside_true_radius.1));
    }
}
