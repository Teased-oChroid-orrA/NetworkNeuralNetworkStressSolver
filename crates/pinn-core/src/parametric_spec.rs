//! Parametric problem specification - the "train once across a parameter range, then infer
//! instantly at any point in that range" workflow (`enhancement.txt` items 7/8/17: "the real
//! 'instant PINN solver' vision"). Distinct from [`crate::problem_spec::ProblemSpec`], which
//! trains a network for exactly ONE fixed material/load (see `crate::inference_envelope`'s
//! doc comment for why that distinction matters) - deliberately a NEW sibling type, not a
//! modification of `ProblemSpec`, so the existing single-instance training path stays
//! completely untouched.
//!
//! v1 scope, stated explicitly rather than left implicit: **only `material.e`, `material.nu`,
//! and `load.px` are parametric.** Geometry (plate extents, hole count/position/radius) is
//! FIXED, matching `enhancement.txt`'s own item 13 risk ranking ("geometry parameters... much
//! more dangerous... changing these may fundamentally alter the solution space" and "topology
//! changes... immediate retraining") - making geometry parametric too is a real, much larger
//! follow-up (it would require the network to condition on a variable-size hole list, not
//! just three scalars), not something to fold in silently here. `material.density`/
//! `material.ultimate_strength_pa` and `load.py` are also fixed (no current loss term reads
//! density in this problem family; `ultimate_strength_pa` only matters when
//! `use_ultimate_strength_scaling` is set, which this v1 path doesn't use; `py` fixed at 0 -
//! matches every shipped example problem's own uniaxial convention).

use serde::{Deserialize, Serialize};

use crate::{material::MaterialProps, problem_spec::{NetworkSpec, TrainingSpec}, user_geometry::UserGeometry};

/// A closed interval `[min, max]` a parameter is trained across. `min <= max` is the caller's
/// responsibility (mirrors every other range-like config in this codebase, e.g.
/// `AmrtConfig`'s fields - not independently validated here, since the failure mode of a
/// degenerate/reversed range is caught immediately and obviously by `sample`/`normalize`
/// producing nonsensical values, not a silent wrong answer).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: f64,
    pub max: f64,
}

impl ParamRange {
    pub fn new(min: f64, max: f64) -> Self { Self { min, max } }

    /// Maps a uniform `u` in `[0, 1)` to a physical value in `[min, max]`.
    pub fn sample(&self, u: f64) -> f64 { self.min + (self.max - self.min) * u }

    /// Maps a physical value to `[-1, 1]` - the network-input-friendly normalization every
    /// other normalized quantity in this codebase already uses (matches the `[-1,1]^2`
    /// spatial convention `UserSamplingStrategy`/`evaluate_user_vis_grid` use). Degenerate
    /// (zero-width) range normalizes to `0.0` rather than dividing by zero.
    pub fn normalize(&self, v: f64) -> f64 {
        let width = self.max - self.min;
        if width.abs() < 1e-300 { 0.0 } else { 2.0 * (v - self.min) / width - 1.0 }
    }

    /// The midpoint - used as a sensible single representative value (e.g. for computing
    /// fixed normalization reference scales once for the whole training run).
    pub fn mid(&self) -> f64 { 0.5 * (self.min + self.max) }

    /// The value of largest magnitude in the range - used where a single WORST-CASE
    /// reference scale is wanted (e.g. `u_ref`/`ref_stress2`, which must stay a valid physical
    /// scale across the entire trained range, not just at one point in it).
    pub fn max_abs(&self) -> f64 { self.min.abs().max(self.max.abs()) }

    pub fn contains(&self, v: f64) -> bool { v >= self.min && v <= self.max }
}

/// The complete parametric problem: fixed geometry, three parametric ranges
/// (`material.e`/`material.nu`/`load.px` - see this module's doc comment for why only these
/// three), fixed secondary material properties, network size, and training schedule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricProblemSpec {
    pub geometry: UserGeometry,
    pub e_range: ParamRange,
    pub nu_range: ParamRange,
    pub load_range: ParamRange,
    /// Fixed (not parametric in v1) - see this module's doc comment.
    pub density: f64,
    pub ultimate_strength_pa: f64,
    #[serde(default)]
    pub network: NetworkSpec,
    #[serde(default)]
    pub training: TrainingSpec,
}

impl ParametricProblemSpec {
    /// Builds the `MaterialProps` for a concrete `(e, nu)` sample within this spec's ranges -
    /// `density`/`ultimate_strength_pa` come from the fixed fields above.
    pub fn material_at(&self, e: f64, nu: f64) -> MaterialProps {
        MaterialProps { e, nu, density: self.density, ultimate_strength_pa: self.ultimate_strength_pa }
    }

    /// `true` if `(e, nu, px)` all fall within this spec's trained ranges - the Phase
    /// 17/18-style "is this a safe inference request" check for the parametric path (min/max
    /// range only, matching `enhancement.txt` item 10's own explicit note that a real
    /// distance-to-training-distribution metric is a future refinement, not required for a
    /// first, honest version).
    pub fn in_range(&self, e: f64, nu: f64, px: f64) -> bool {
        self.e_range.contains(e) && self.nu_range.contains(nu) && self.load_range.contains(px)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_geometry::HoleSpec;

    fn sample_spec() -> ParametricProblemSpec {
        ParametricProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.05, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.01, bc: crate::user_geometry::HoleBc::Free }],
            },
            e_range: ParamRange::new(50e9, 100e9),
            nu_range: ParamRange::new(0.25, 0.35),
            load_range: ParamRange::new(40e6, 80e6),
            density: 2810.0,
            ultimate_strength_pa: 503e6,
            network: NetworkSpec::default(),
            training: TrainingSpec::default(),
        }
    }

    #[test]
    fn param_range_sample_at_zero_and_one_hits_the_endpoints() {
        let r = ParamRange::new(5.0, 10.0);
        assert_eq!(r.sample(0.0), 5.0);
        assert_eq!(r.sample(1.0), 10.0);
        assert_eq!(r.sample(0.5), 7.5);
    }

    #[test]
    fn param_range_normalize_round_trips_sample() {
        let r = ParamRange::new(5.0, 10.0);
        assert!((r.normalize(r.sample(0.0)) - (-1.0)).abs() < 1e-9);
        assert!((r.normalize(r.sample(1.0)) - 1.0).abs() < 1e-9);
        assert!((r.normalize(r.sample(0.5)) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn param_range_normalize_degenerate_zero_width_is_zero_not_nan() {
        let r = ParamRange::new(7.0, 7.0);
        assert_eq!(r.normalize(7.0), 0.0);
    }

    #[test]
    fn param_range_max_abs_picks_larger_magnitude_endpoint() {
        assert_eq!(ParamRange::new(-10.0, 3.0).max_abs(), 10.0);
        assert_eq!(ParamRange::new(-2.0, 8.0).max_abs(), 8.0);
    }

    #[test]
    fn spec_material_at_uses_fixed_density_and_ultimate_strength() {
        let spec = sample_spec();
        let m = spec.material_at(71.7e9, 0.33);
        assert_eq!(m.e, 71.7e9);
        assert_eq!(m.nu, 0.33);
        assert_eq!(m.density, spec.density);
        assert_eq!(m.ultimate_strength_pa, spec.ultimate_strength_pa);
    }

    #[test]
    fn spec_in_range_rejects_any_single_out_of_range_parameter() {
        let spec = sample_spec();
        assert!(spec.in_range(71.7e9, 0.33, 6.9e7));
        assert!(!spec.in_range(200e9, 0.33, 6.9e7), "E outside range must fail");
        assert!(!spec.in_range(71.7e9, 0.5, 6.9e7), "nu outside range must fail");
        assert!(!spec.in_range(71.7e9, 0.33, 5e8), "load outside range must fail");
    }

    #[test]
    fn parametric_problem_spec_round_trips_through_toml() {
        let spec = sample_spec();
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: ParametricProblemSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed, spec);
    }

    /// Regression guard mirroring `problem_spec::tests::shipped_plate_example_specs_parse` -
    /// the shipped parametric example must stay loadable by this exact struct.
    #[test]
    fn shipped_parametric_example_spec_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems/parametric_single_hole_plate.toml");
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
        let spec: ParametricProblemSpec = toml::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        assert_eq!(spec.geometry.holes.len(), 1);
        assert!(spec.e_range.contains(71.7e9), "Al 7075-T6's real E should fall inside the trained range");
    }

    /// A plain (non-parametric) `ProblemSpec` example must NOT accidentally parse as
    /// `ParametricProblemSpec` (missing e_range/nu_range/load_range) - this is exactly what
    /// `app-egui::stress_solver::load_spec`'s try-Parametric-before-Plate dispatch relies on.
    #[test]
    fn plain_problem_spec_example_does_not_parse_as_parametric() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems/single_hole_plate.toml");
        let contents = std::fs::read_to_string(&path).expect("read single_hole_plate.toml");
        assert!(toml::from_str::<ParametricProblemSpec>(&contents).is_err());
    }
}
