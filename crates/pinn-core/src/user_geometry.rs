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

/// Issue #61 EPIC P2-06: identifies one boundary component of a [`UserGeometry`] - an outer
/// rectangle edge, or a specific hole by index. See [`UserGeometry::nearest_boundary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryRef {
    OuterLeft,
    OuterRight,
    OuterTop,
    OuterBottom,
    /// Index into [`UserGeometry::holes`].
    Hole(usize),
}

/// Issue #61 EPIC P2-06: per-direction validity of a 5-point central-difference FD stencil
/// centered at some point - see [`UserGeometry::valid_stencil`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilValidity {
    pub center_valid: bool,
    pub x_plus_valid: bool,
    pub x_minus_valid: bool,
    pub y_plus_valid: bool,
    pub y_minus_valid: bool,
}

impl StencilValidity {
    /// True iff the center AND all 4 shifted neighbors are valid - a real, usable FD stencil.
    pub fn all_valid(&self) -> bool {
        self.center_valid && self.x_plus_valid && self.x_minus_valid && self.y_plus_valid && self.y_minus_valid
    }
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

    /// Issue #61 EPIC P2-06: signed distance to the domain boundary (positive = inside the
    /// valid domain - inside the outer rectangle AND outside every hole; negative = outside).
    /// `min(rect_sdf, hole_sdfs...)` is an approximate (not exact) SDF for a rectangle-minus-
    /// circles domain - exact everywhere except where a hole boundary and the outer rectangle
    /// boundary are close enough that their influence regions overlap (this codebase's real
    /// hole radii are always small relative to the plate, so this never matters in practice,
    /// but is not claimed exact in general). Generic over any number of holes - not hardcoded
    /// to one, unlike `crate::geometry::GeometryConfig`'s single `HoleType`.
    pub fn signed_distance(&self, x: f64, y: f64) -> f64 {
        let rect_sdf = (self.half_w - x.abs()).min(self.half_h - y.abs());
        self.holes.iter().fold(rect_sdf, |sdf, hole| {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            sdf.min((dx * dx + dy * dy).sqrt() - hole.radius)
        })
    }

    /// Which boundary component (an outer edge or a specific hole) is closest to `(x, y)` -
    /// the generic identifier [`Self::boundary_normal`]/[`Self::boundary_tangent`]/
    /// [`Self::boundary_measure`] key off, instead of each re-deriving "which edge" ad hoc.
    pub fn nearest_boundary(&self, x: f64, y: f64) -> BoundaryRef {
        let mut best = BoundaryRef::OuterLeft;
        let mut best_dist = (x + self.half_w).abs();
        for (candidate, dist) in [
            (BoundaryRef::OuterRight, (self.half_w - x).abs()),
            (BoundaryRef::OuterTop, (self.half_h - y).abs()),
            (BoundaryRef::OuterBottom, (y + self.half_h).abs()),
        ] {
            if dist < best_dist {
                best_dist = dist;
                best = candidate;
            }
        }
        for (i, hole) in self.holes.iter().enumerate() {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            let dist = ((dx * dx + dy * dy).sqrt() - hole.radius).abs();
            if dist < best_dist {
                best_dist = dist;
                best = BoundaryRef::Hole(i);
            }
        }
        best
    }

    /// Outward unit normal for the given boundary component, evaluated at `(x, y)` - constant
    /// per outer edge (axis-aligned), radial (from the hole's own center through `(x,y)`) for
    /// a hole. `(x, y)` need not lie exactly ON the boundary (the radial direction is still
    /// well-defined for any point other than a hole's exact center).
    pub fn boundary_normal_for(&self, boundary: BoundaryRef, x: f64, y: f64) -> (f64, f64) {
        match boundary {
            BoundaryRef::OuterLeft => (-1.0, 0.0),
            BoundaryRef::OuterRight => (1.0, 0.0),
            BoundaryRef::OuterTop => (0.0, 1.0),
            BoundaryRef::OuterBottom => (0.0, -1.0),
            BoundaryRef::Hole(i) => {
                let hole = &self.holes[i];
                let dx = x - hole.center[0];
                let dy = y - hole.center[1];
                let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                (dx / len, dy / len)
            }
        }
    }

    /// [`Self::boundary_normal_for`] at whichever boundary [`Self::nearest_boundary`] finds
    /// closest to `(x, y)` - the common case of "what's the normal AT this point".
    pub fn boundary_normal(&self, x: f64, y: f64) -> (f64, f64) {
        self.boundary_normal_for(self.nearest_boundary(x, y), x, y)
    }

    /// Unit tangent (90° counter-clockwise rotation of the outward normal) at `(x, y)`'s
    /// nearest boundary.
    pub fn boundary_tangent(&self, x: f64, y: f64) -> (f64, f64) {
        let (nx, ny) = self.boundary_normal(x, y);
        (-ny, nx)
    }

    /// Total arc length / edge length of the given boundary component (m) - the real
    /// geometric measure [`crate`]-level `BoundaryIntegral`-style consumers (see
    /// `pinn_solver::measure_integral`) need for a specific component, generalizing
    /// `pinn_solver::measure_integral::plate_outer_perimeter`'s outer-only formula to also
    /// cover individual holes.
    pub fn boundary_measure(&self, boundary: BoundaryRef) -> f64 {
        match boundary {
            BoundaryRef::OuterLeft | BoundaryRef::OuterRight => 2.0 * self.half_h,
            BoundaryRef::OuterTop | BoundaryRef::OuterBottom => 2.0 * self.half_w,
            BoundaryRef::Hole(i) => 2.0 * std::f64::consts::PI * self.holes[i].radius,
        }
    }

    /// Whether a 5-point central-difference FD stencil centered at `(x, y)` with half-steps
    /// `(hx, hy)` stays entirely within the valid domain (issue #61 EPIC P2-06's own "stencils
    /// avoid invalid points with recorded fallback/quality diagnostics" - this is the
    /// diagnostic; [`StencilValidity::all_valid`] is the yes/no answer, the per-direction
    /// fields are the "which neighbor(s) failed" detail a fallback strategy would need).
    /// Generalizes the ad hoc margin-based checks this codebase's real sampling strategies
    /// already perform (e.g. `pinn_solver::user_problem::UserSamplingStrategy::contains_for_
    /// collocation`) into a declared, reusable, geometry-level primitive.
    pub fn valid_stencil(&self, x: f64, y: f64, hx: f64, hy: f64) -> StencilValidity {
        StencilValidity {
            center_valid: self.contains(x, y),
            x_plus_valid: self.contains(x + hx, y),
            x_minus_valid: self.contains(x - hx, y),
            y_plus_valid: self.contains(x, y + hy),
            y_minus_valid: self.contains(x, y - hy),
        }
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
    fn signed_distance_matches_hand_computed_values() {
        let geom = two_hole_geometry();
        // Interior point midway between holes: rect_sdf=1.0, both hole sdfs=0.5-0.1=0.4.
        assert!((geom.signed_distance(0.0, 0.0) - 0.4).abs() < 1e-12);
        // Exactly at hole 1's center: inside the hole, sdf = 0 - radius = -0.1 (the min term).
        assert!((geom.signed_distance(-0.5, 0.0) - (-0.1)).abs() < 1e-12);
        // Outside the outer rectangle: rect_sdf = 1.0 - 1.5 = -0.5, dominates the min.
        assert!((geom.signed_distance(1.5, 0.0) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn nearest_boundary_identifies_the_closest_outer_edge_or_hole() {
        let geom = two_hole_geometry();
        assert_eq!(geom.nearest_boundary(0.99, 0.0), BoundaryRef::OuterRight);
        assert_eq!(geom.nearest_boundary(-0.99, 0.0), BoundaryRef::OuterLeft);
        assert_eq!(geom.nearest_boundary(0.0, 0.99), BoundaryRef::OuterTop);
        assert_eq!(geom.nearest_boundary(0.0, -0.99), BoundaryRef::OuterBottom);
        // Just outside hole 1's boundary (radius 0.1, center -0.5) - much closer to that hole
        // than to any outer edge.
        assert_eq!(geom.nearest_boundary(-0.39, 0.0), BoundaryRef::Hole(0));
        assert_eq!(geom.nearest_boundary(0.39, 0.0), BoundaryRef::Hole(1));
    }

    #[test]
    fn boundary_normal_is_axis_aligned_on_outer_edges_and_radial_on_holes() {
        let geom = two_hole_geometry();
        assert_eq!(geom.boundary_normal(0.99, 0.0), (1.0, 0.0));
        assert_eq!(geom.boundary_normal(-0.99, 0.0), (-1.0, 0.0));
        assert_eq!(geom.boundary_normal(0.0, 0.99), (0.0, 1.0));
        assert_eq!(geom.boundary_normal(0.0, -0.99), (0.0, -1.0));
        // Point just outside hole 1, to its right - radial direction points away from the
        // hole's center, i.e. in +x.
        let (nx, ny) = geom.boundary_normal(-0.39, 0.0);
        assert!((nx - 1.0).abs() < 1e-9, "{nx}");
        assert!(ny.abs() < 1e-9, "{ny}");
    }

    #[test]
    fn boundary_tangent_is_perpendicular_to_the_normal() {
        let geom = two_hole_geometry();
        assert_eq!(geom.boundary_tangent(0.99, 0.0), (0.0, 1.0));
        let (nx, ny) = geom.boundary_normal(0.99, 0.0);
        let (tx, ty) = geom.boundary_tangent(0.99, 0.0);
        assert!((nx * tx + ny * ty).abs() < 1e-12, "normal and tangent must be perpendicular");
    }

    #[test]
    fn boundary_measure_matches_hand_computed_lengths() {
        let geom = two_hole_geometry();
        assert!((geom.boundary_measure(BoundaryRef::OuterLeft) - 2.0).abs() < 1e-12);
        assert!((geom.boundary_measure(BoundaryRef::OuterTop) - 2.0).abs() < 1e-12);
        let expected_circumference = 2.0 * std::f64::consts::PI * 0.1;
        assert!((geom.boundary_measure(BoundaryRef::Hole(0)) - expected_circumference).abs() < 1e-12);
    }

    #[test]
    fn valid_stencil_is_fully_valid_far_from_any_boundary() {
        let geom = two_hole_geometry();
        let v = geom.valid_stencil(0.0, 0.0, 0.05, 0.05);
        assert!(v.all_valid());
    }

    #[test]
    fn valid_stencil_flags_exactly_the_direction_that_crosses_into_a_hole() {
        let geom = two_hole_geometry();
        // (-0.65, 0) is outside hole 1 (dist to center 0.15 > radius 0.1) - a valid center.
        // Shifting +hx=0.1 lands at (-0.55, 0): dist to hole 1 center = 0.05 < radius 0.1 -
        // INSIDE the hole. Shifting -hx lands at (-0.75, 0): still well outside. y-shifts stay
        // at x=-0.65, also well outside either hole.
        let v = geom.valid_stencil(-0.65, 0.0, 0.1, 0.1);
        assert!(v.center_valid);
        assert!(!v.x_plus_valid, "x+ neighbor crosses into hole 1, must be flagged invalid");
        assert!(v.x_minus_valid);
        assert!(v.y_plus_valid);
        assert!(v.y_minus_valid);
        assert!(!v.all_valid());
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
