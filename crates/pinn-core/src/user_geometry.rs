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
}
