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
    /// Issue #62 PH3-04: opt-in switch for the AMR-density-compensated, measure-aware
    /// variational functional (`measure_integral::domain_integral_weighted_tensor`/
    /// `boundary_integral_tensor`, already proven by L0/L3) in place of the legacy plain
    /// `.mean()` `InteriorEnergyTerm`/`ExternalWorkTerm` computation. `#[serde(default)]` so
    /// every existing TOML spec (which predates this field) keeps parsing unchanged and keeps
    /// training on the EXACT legacy path - per issue #62 §3.3's "legacy paths SHALL remain
    /// available behind an explicit compatibility switch until validated" rule, this defaults
    /// to `false`, never silently opting a pre-existing config into different training math.
    #[serde(default)]
    pub measure_aware_training: bool,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self { max_steps: 2000, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3, measure_aware_training: false }
    }
}

/// Explicit formulation selection - Phase 2 remediation (issue #61, epic P2-01, "Explicit
/// formulation model"). This is a REAL gate on which loss terms `UserDefinedProblem::
/// loss_terms()` returns, not a classification label (see `pinn_solver::problem::
/// LossTerm::formulation_kind()`, which only classifies terms that already exist in the
/// returned list - this type controls what's IN that list in the first place).
///
/// Base terms are `interior_energy` (U), `equilibrium` (strong-form ∇·σ=0 residual),
/// `outer_traction` (strong-form Neumann residual), `external_work` (W_ext, the DEM natural-BC
/// counterpart to `outer_traction`). Every hole's essential (Dirichlet, `HoleBc::Fixed`)
/// constraint is ALWAYS active regardless of formulation - an essential constraint is required
/// in every formulation, not a formulation-specific choice (issue #61's own text lists
/// "essential BC/gauge constraints separately declared" as a fixed member of the variational
/// minimum, not an optional one). Each hole's NATURAL boundary (`HoleBc::Free`,
/// traction-free) is where formulations genuinely differ - see each variant's own doc comment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FormulationSelection {
    /// `Pi = U - W_ext` ONLY (+ essential constraints). Natural boundaries (`outer_traction`'s
    /// strong-form penalty, and each `HoleBc::Free` hole's traction-free penalty) are
    /// EXCLUDED from the optimization objective by construction - satisfied automatically by
    /// the variational principle itself (a correctly-posed `W_ext` already encodes the
    /// natural BC; a separate penalty term would double-enforce it - issue #61 §1.2's
    /// "duplicate natural Neumann enforcement" prohibition). `equilibrium`/`outer_traction`
    /// remain available as post-hoc DIAGNOSTICS (`training_core::probe_interior_energy_
    /// residuals`/`user_problem::probe_boundary_residuals` - neither depends on `loss_terms()`
    /// at all), just not as optimization-objective terms.
    Variational,
    /// Only strong-form PDE/BC residuals (`equilibrium`, `outer_traction`, plus each hole's
    /// `HoleBc::Free` traction-free penalty) - NO energy-functional terms (`interior_energy`,
    /// `external_work` excluded).
    Strong,
    /// An explicit, named list of BASE terms (a subset of `interior_energy`/`equilibrium`/
    /// `outer_traction`/`external_work`) - "Hybrid requires an explicit term list" (issue #61
    /// P2-01 acceptance criterion): there is no implicit "everything" fallback, every entry
    /// must be named. Every hole's terms (both essential `hole_fixed` and natural `hole_free`)
    /// are always included for `Hybrid`, matching this codebase's own pre-remediation
    /// behavior when reproduced via `default_formulation()`'s literal 4-name list below.
    /// Unknown names panic at `loss_terms()` time (same "loud, not silent" failure mode
    /// `base_weight`'s own `panic!("unknown loss term")` already established for this file).
    Hybrid(Vec<String>),
}

/// The exact pre-remediation behavior (`UserDefinedProblem::loss_terms()` before issue #61):
/// every base term active, unconditionally. Kept as the `#[serde(default)]` so every existing
/// shipped/user TOML spec keeps parsing AND behaving identically (issue #61 §1.2/P2-15's
/// "no silent formulation change" migration requirement) - formulation is now explicit
/// (a real, named `Hybrid` selection), even though the DEFAULT explicit choice reproduces old
/// behavior byte-for-byte.
pub fn default_formulation() -> FormulationSelection {
    FormulationSelection::Hybrid(vec![
        "interior_energy".to_string(),
        "equilibrium".to_string(),
        "outer_traction".to_string(),
        "external_work".to_string(),
    ])
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
    /// Issue #61 P2-01. Defaults to `default_formulation()` (the exact pre-remediation
    /// behavior) so every existing spec file keeps parsing and training identically.
    #[serde(default = "default_formulation")]
    pub formulation: FormulationSelection,
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
            formulation: default_formulation(),
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

    /// Issue #62 PH3-05: the shipped pure-Variational + measure-aware production benchmark
    /// config must parse with EXACTLY the formulation/switch this epic's own manifest entry
    /// claims - a real regression guard, not just "the file exists".
    #[test]
    fn shipped_variational_no_hole_example_spec_parses_with_expected_formulation_and_switch() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems/variational_no_hole_plate.toml");
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
        let spec: ProblemSpec = toml::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        assert_eq!(spec.formulation, FormulationSelection::Variational);
        assert!(spec.training.measure_aware_training);
        assert!(spec.geometry.holes.is_empty(), "this is the no-hole benchmark configuration");
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
        assert_eq!(parsed.formulation, default_formulation(), "omitting [formulation] must reproduce pre-remediation behavior exactly");
    }

    #[test]
    fn default_formulation_is_the_literal_pre_remediation_hybrid_list() {
        assert_eq!(default_formulation(), FormulationSelection::Hybrid(vec![
            "interior_energy".to_string(), "equilibrium".to_string(),
            "outer_traction".to_string(), "external_work".to_string(),
        ]));
    }

    #[test]
    fn formulation_variational_round_trips_through_toml() {
        let mut spec = sample_spec();
        spec.formulation = FormulationSelection::Variational;
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: ProblemSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.formulation, FormulationSelection::Variational);
    }

    #[test]
    fn formulation_strong_round_trips_through_toml() {
        let mut spec = sample_spec();
        spec.formulation = FormulationSelection::Strong;
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: ProblemSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.formulation, FormulationSelection::Strong);
    }

    #[test]
    fn formulation_hybrid_with_explicit_subset_round_trips_through_toml() {
        let mut spec = sample_spec();
        spec.formulation = FormulationSelection::Hybrid(vec!["interior_energy".to_string()]);
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: ProblemSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.formulation, FormulationSelection::Hybrid(vec!["interior_energy".to_string()]));
    }
}
