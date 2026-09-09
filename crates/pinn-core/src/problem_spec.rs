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
    /// Smart adaptive architecture master switch (default off - every existing TOML spec
    /// keeps parsing/behaving unchanged). When true, the training loop internally builds the
    /// network with PirateNet-style gated residual blocks (regardless of any other setting)
    /// since that structure is what makes safe depth growth/shrink possible, and runs an
    /// `architecture_controller::ArchitectureController` alongside training.
    #[serde(default)]
    pub adaptive: bool,
    /// Width growth ceiling when `adaptive`. `None` = no cap beyond hardware limits.
    #[serde(default)]
    pub max_hidden_dim: Option<usize>,
    /// Depth growth ceiling when `adaptive`. `None` = no cap beyond hardware limits.
    #[serde(default)]
    pub max_n_hidden: Option<usize>,
    /// Auto-stop when training plateaus (no further real progress) - defaults ON, matching
    /// this toolbox's prior behavior on the Kirsch path (`headless.rs`'s own `ConvergenceTracker`-
    /// driven cascade), which the plate/user-defined-problem path never had until this field.
    /// Unlike Kirsch's warm-restart cascade (LR/Adam reset, tightened `lam_h_cap`, tuned
    /// specifically for K_t dynamics), this triggers a plain graceful stop - the same path
    /// clicking Stop already takes - not a restart.
    #[serde(default = "default_auto_stop_on_plateau")]
    pub auto_stop_on_plateau: bool,
}

fn default_auto_stop_on_plateau() -> bool {
    true
}

impl Default for NetworkSpec {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            n_hidden: 3,
            adaptive: false,
            max_hidden_dim: None,
            max_n_hidden: None,
            auto_stop_on_plateau: true,
        }
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

    /// Regression guard: every shipped `examples/problems/*.toml` plate spec must stay
    /// loadable by this exact struct — a silent field-rename here would otherwise only be
    /// caught by a human manually running each example file.
    #[test]
    fn shipped_plate_example_specs_parse() {
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems");
        for name in ["notched_plate.toml", "single_hole_plate.toml", "triple_hole_plate.toml", "biaxial_steel_plate.toml"] {
            let path = examples_dir.join(name);
            let contents = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
            toml::from_str::<ProblemSpec>(&contents).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        }
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
