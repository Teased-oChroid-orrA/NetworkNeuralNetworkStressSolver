//! User-facing problem specification — the full "design a joint from scratch" input,
//! deserialized from a TOML file (`pinn_solver::user_runner::run_headless_user_problem`
//! consumes this; loading/parsing happens at the `pinn-app` CLI boundary since this crate
//! stays a plain data-type crate with no file-I/O concerns of its own).

use serde::{Deserialize, Serialize};

use crate::{loading::LoadConfig, material::MaterialProps, user_geometry::UserGeometry};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NetworkSpec {
    pub hidden_dim: usize,
    pub n_hidden: usize,
}

impl Default for NetworkSpec {
    fn default() -> Self {
        Self { hidden_dim: 64, n_hidden: 3 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingSpec {
    pub max_steps: usize,
    pub n_interior: usize,
    pub n_boundary: usize,
    /// Finite-difference step in normalized [-1,1]^2 coordinates.
    pub fd_h: f32,
    pub lr: f64,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self { max_steps: 2000, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3 }
    }
}

/// The complete user-defined problem: geometry (rectangular plate + N holes), material,
/// far-field load, network size, and training schedule. See `examples/problems/` for a
/// documented, ready-to-edit template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProblemSpec {
    pub geometry: UserGeometry,
    pub material: MaterialProps,
    pub load: LoadConfig,
    #[serde(default)]
    pub network: NetworkSpec,
    #[serde(default)]
    pub training: TrainingSpec,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_geometry::{HoleBc, HoleSpec};

    fn sample_spec() -> ProblemSpec {
        ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1,
                half_h: 0.05,
                thickness: 0.005,
                holes: vec![
                    HoleSpec { center: [-0.03, 0.0], radius: 0.01, bc: HoleBc::Free },
                    HoleSpec { center: [0.03, 0.0], radius: 0.008, bc: HoleBc::Fixed },
                ],
            },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec::default(),
            training: TrainingSpec::default(),
        }
    }

    #[test]
    fn problem_spec_round_trips_through_toml() {
        let spec = sample_spec();
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: ProblemSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed, spec);
    }

    #[test]
    fn network_and_training_specs_default_when_omitted_from_toml() {
        let toml_str = r#"
            [geometry]
            half_w = 0.1
            half_h = 0.05
            thickness = 0.005
            holes = []

            [material]
            e = 71.7e9
            nu = 0.33
            density = 2810.0
            ultimate_strength_pa = 503e6

            [load]
            px = 6.9e7
            py = 0.0
        "#;
        let parsed: ProblemSpec = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.network, NetworkSpec::default());
        assert_eq!(parsed.training, TrainingSpec::default());
    }
}
