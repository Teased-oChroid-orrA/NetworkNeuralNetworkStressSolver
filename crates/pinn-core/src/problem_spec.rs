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
    /// Issue #62 PH3-11: the seed passed to `Backend::seed` immediately before network weight
    /// initialization - see `pinn_solver::network::ElasticityNetConfig::init`'s call site for
    /// where this is actually consumed, and `provenance::RunProvenance::model_init_seeded`'s
    /// doc comment for the real, previously-negative finding this closes (model init was
    /// NEVER seeded anywhere in this codebase before this field existed). Every existing TOML
    /// spec keeps parsing (this is `#[serde(default)]`), but note this IS a real, deliberate
    /// behavior change versus every run before this field existed: those runs' initial weights
    /// were drawn from whatever RNG state the process happened to be in at that point (already
    /// effectively random, never reproducible) - defaulting to a fixed constant here makes
    /// runs reproducible by default rather than "randomly random, and now differently random".
    #[serde(default = "default_model_init_seed")]
    pub model_init_seed: u64,
}

fn default_auto_stop_on_plateau() -> bool {
    true
}

/// Arbitrary but fixed - one more than `user_problem::SEED_INTERIOR` (90_210) purely as a
/// naming convention tying the two seeds together, not a derived/meaningful value.
fn default_model_init_seed() -> u64 {
    90_211
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
            model_init_seed: default_model_init_seed(),
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
    /// Issue #62 PH3-06: opt-in live AD-vs-FD strain cross-validation diagnostic
    /// (`differential_operator::ad_fd_strain_agreement`) - real extra cost (an independent
    /// forward+backward pass through the SAME model weights, on top of the normal training
    /// step), never on by default. See that function's own doc comment for why this is a
    /// DIAGNOSTIC only, never a live training-loss backend substitution (burn-autodiff 0.21 has
    /// no nested/higher-order autodiff, so an AD-retrieved gradient is structurally incapable of
    /// staying connected to the model-weight autodiff graph `LossTerm::compute()` needs).
    #[serde(default)]
    pub derivative_operator_diagnostic: bool,
    /// Issue #62 PH3-12: real on/off switch for the periodic AMR sweep (`AdaptiveGrid::adapt`
    /// + resample), so the plan's own mandated "fixed sampling vs AMR, same training budget"
    /// controlled comparison is actually possible - before this field, AMR fired unconditionally
    /// on the plate path with no way to run the "fixed sampling" control arm at all.
    /// `#[serde(default = "default_amr_enabled")]` = `true`, matching the exact PRE-EXISTING
    /// unconditional behavior (every existing TOML spec keeps training exactly as before -
    /// per issue #62 §3.3, the new "fixed sampling" behavior is the one that must be explicitly
    /// opted into, since AMR-on was already the shipped default, not the other way around).
    #[serde(default = "default_amr_enabled")]
    pub amr_enabled: bool,
}

fn default_amr_enabled() -> bool {
    true
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            max_steps: 2000, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3,
            measure_aware_training: false, derivative_operator_diagnostic: false,
            amr_enabled: true,
        }
    }
}

/// Explicit formulation selection - Phase 2 remediation (issue #61, epic P2-01, "Explicit
/// formulation model"). This is a REAL gate on which loss terms `UserDefinedProblem::
/// loss_terms()` returns, not a classification label (see `pinn_solver::problem::
/// LossTerm::formulation_kind()`, which only classifies terms that already exist in the
/// returned list - this type controls what's IN that list in the first place).
///
/// Base terms are `physical_potential` (atomic `U-W_ext`), `interior_energy` (legacy hybrid
/// U), `equilibrium` (strong-form ∇·σ=0 residual), `outer_traction` (strong-form Neumann
/// residual), and `external_work` (legacy hybrid W_ext). Every hole's essential (Dirichlet, `HoleBc::Fixed`)
/// constraint is ALWAYS active regardless of formulation - an essential constraint is required
/// in every formulation, not a formulation-specific choice (issue #61's own text lists
/// "essential BC/gauge constraints separately declared" as a fixed member of the variational
/// minimum, not an optional one). Each hole's NATURAL boundary (`HoleBc::Free`,
/// traction-free) is where formulations genuinely differ - see each variant's own doc comment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FormulationSelection {
    /// Atomic `Pi = U - W_ext` ONLY (+ essential constraints). Natural boundaries (`outer_traction`'s
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

/// Issue #77 PH4-45: which of the three architectures PH4-41..44 corrected and measured to
/// converge within 0.67-1.23% of FEM (see `PHASE_4_IMPLEMENTATION_MANIFEST.md`) trains an
/// annular-decomposition-eligible spec's hole domain — orthogonal to `hard_constraint_ansatz`/
/// `coordinate_embedding` below (both apply to whichever procedure is selected).
/// `#[serde(default)]` = `Joint`, the exact pre-existing `AnnularDecompositionProblem`
/// single-stage behavior — no existing spec's dispatch changes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum TrainingProcedure {
    /// The exact pre-#77 dispatch: `AnnularDecompositionProblem`'s single joint optimization
    /// when the geometry qualifies (`AnnularDecompositionProblem::supports`), the plain
    /// single-domain `UserDefinedProblem` path otherwise. `hard_constraint_ansatz`/
    /// `coordinate_embedding` apply to whichever branch this naturally selects.
    #[default]
    Joint,
    /// Forces the single-domain `UserDefinedProblem` path (Phase 1/PH4-42's own real tested
    /// configuration) EVEN when the geometry would otherwise also qualify for annular
    /// decomposition - which every "L5" shape this investigation used does. Without this,
    /// `hard_constraint_ansatz=true` on an L5-shaped spec would silently apply to the
    /// TWO-domain annular path's annulus model instead of Phase 1's actual single-domain
    /// model - a different, untested architecture from the one PH4-42 verified converges to
    /// 0.67% of FEM. `coordinate_embedding` must stay `Cartesian` here (`UserDefinedProblem`
    /// has no log-polar wiring - that's Phase 3's, annular-only).
    SingleDomain,
    /// Phase 2's frozen-anchor sequential two-stage procedure
    /// (`user_runner::run_annular_decomposition_training_sequential`): train the outer domain
    /// alone for `stage_a_steps`, freeze it, then train the annulus domain alone for
    /// `stage_b_steps` against the frozen outer model's own interface trace.
    SequentialTwoStage { stage_a_steps: usize, stage_b_steps: usize },
}

fn default_training_procedure() -> TrainingProcedure {
    TrainingProcedure::Joint
}

/// Issue #77 PH4-45: selects between the annulus domain's two tested coordinate
/// representations - `Cartesian` (the pre-existing `SingleHoleChart`/raw embedding, chosen by
/// `UserGeometry::coordinate_embedding()`) or Phase 3's `LogPolar` reparameterization
/// (`ξ=ln(r/a), cosθ, sinθ`, `UserGeometry::log_polar_embedding()`). `#[serde(default)]` =
/// `Cartesian`, byte-identical to every existing spec.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CoordinateEmbeddingSelection {
    Cartesian,
    LogPolar,
}

impl Default for CoordinateEmbeddingSelection {
    fn default() -> Self {
        Self::Cartesian
    }
}

/// Issue #77 PH4-45: opt-in selection of the three architectures PH4-41..44 proved converge to
/// the FEM reference within 0.67-1.23% (`PHASE_4_IMPLEMENTATION_MANIFEST.md`), replacing the
/// prior test-only construction (`UserDefinedProblem::new_with_hard_constraint_ansatz`,
/// `AnnularDecompositionProblem::new_with_log_polar_embedding`,
/// `run_annular_decomposition_training_sequential`) with a TOML-reachable one. Every field
/// `#[serde(default)]`s to the exact pre-existing behavior, so no existing spec's training
/// changes. `hard_constraint_ansatz`/`hole_bias_fraction` require
/// `pinn_solver::user_problem::decomposition_applicable` (one centered, traction-free hole) -
/// the same scope `UserDefinedProblem::new_with_hard_constraint_ansatz` already asserts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ArchitectureSpec {
    /// Phase 1/PH4-35's exact, closed-form hard-constraint hole ansatz
    /// (`kirsch_hole_correction::HoleTractionFreeAnsatz`) in place of the soft `hole_free`
    /// penalty. Applies to the hole domain of whichever `training_procedure` is selected
    /// (the single domain under `Joint` without annular decomposition, the annulus domain
    /// under `Joint` with it, or Stage B's annulus domain under `SequentialTwoStage`).
    #[serde(default)]
    pub hard_constraint_ansatz: bool,
    /// Phase 1's near-hole stratified-sampling bias fraction (`UserSamplingStrategy::
    /// with_hole_bias`), quadrature-compensated via `hole_bias_quadrature_weights` (PH4-41
    /// finding 2's fix) so the energy integral stays correct under the resulting non-uniform
    /// density. `0.0` (default) reproduces the exact pre-existing uniform-density sampling.
    #[serde(default)]
    pub hole_bias_fraction: f64,
    #[serde(default)]
    pub coordinate_embedding: CoordinateEmbeddingSelection,
    #[serde(default = "default_training_procedure")]
    pub training_procedure: TrainingProcedure,
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
    /// Issue #77 PH4-45. Defaults to `ArchitectureSpec::default()` (plain `Identity` ansatz,
    /// no sampling bias, Cartesian embedding, `Joint` procedure) - byte-identical to every
    /// existing spec file's dispatch.
    #[serde(default)]
    pub architecture: ArchitectureSpec,
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
            architecture: ArchitectureSpec::default(),
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

    /// Issue #77 PH4-45: the shipped example demonstrating `[architecture]` - a real regression
    /// guard that TOML deserialization of `hard_constraint_ansatz`/`hole_bias_fraction` and the
    /// commented-out `SequentialTwoStage` syntax documented in the file's own comment are both
    /// exactly what this session's real, reproducing PH4-42 result used.
    #[test]
    fn shipped_issue_77_hard_constraint_example_parses_with_expected_architecture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems/issue_77_l5_hard_constraint.toml");
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
        let spec: ProblemSpec = toml::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        assert!(spec.architecture.hard_constraint_ansatz);
        assert_eq!(spec.architecture.hole_bias_fraction, 0.5);
        assert_eq!(spec.architecture.coordinate_embedding, CoordinateEmbeddingSelection::Cartesian);
        assert_eq!(spec.architecture.training_procedure, TrainingProcedure::SingleDomain);
        assert_eq!(spec.geometry.holes.len(), 1);
        assert_eq!(spec.geometry.holes[0].radius, 0.005);
    }

    /// Issue #77 PH4-45: the commented `[architecture.training_procedure]` syntax the example
    /// file's own comment documents must actually parse - written here as its own standalone
    /// TOML fragment (not uncommented in the shipped file, since PH4-43's real verified config
    /// used a bare spec without `hole_bias_fraction`/log-polar) so the documented syntax itself
    /// is regression-proven, not just eyeballed.
    #[test]
    fn documented_sequential_two_stage_toml_syntax_round_trips() {
        let toml_str = r#"
            [architecture.training_procedure]
            SequentialTwoStage = { stage_a_steps = 1500, stage_b_steps = 1500 }

            [geometry]
            half_w = 0.1
            half_h = 0.1
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
        let parsed: ProblemSpec = toml::from_str(toml_str).expect("documented syntax must parse");
        assert_eq!(
            parsed.architecture.training_procedure,
            TrainingProcedure::SequentialTwoStage { stage_a_steps: 1500, stage_b_steps: 1500 },
        );
    }

    /// Issue #62 PH3-17's own real decision record, enforced: after running every "new path"
    /// this Phase 3 built a compatibility switch for through real, generous-budget evidence
    /// (PH3-05/09/12/14/15), NONE of them qualified for promotion to default - `formulation`
    /// defaults to Hybrid, not Variational (PH3-14: pure Variational DIVERGES past step ~7000);
    /// `measure_aware_training` defaults to `false` (PH3-15: measure-aware training under the
    /// SAME Hybrid formulation that otherwise converges still fails every hard threshold);
    /// `amr_enabled` defaults to `true` (PH3-12: fixed sampling beat AMR on the no-hole
    /// geometry, but that single-geometry result doesn't generalize to the holed geometries
    /// AMR was actually designed for, so the legacy "AMR always on" default is kept). Per
    /// issue #62 §21 ("deletion is the final step, not the implementation strategy"), the
    /// correct, evidence-driven outcome of running that full process was "keep every legacy
    /// default and remove nothing" - this test guards against an accidental future default
    /// flip being mistaken for a deliberate, evidence-backed one. If a REAL new finding
    /// justifies changing one of these defaults, update this test AND cite the manifest entry
    /// that justifies it - never flip a default silently.
    #[test]
    fn ph3_17_decision_record_no_legacy_default_was_changed_without_new_proof() {
        assert_eq!(default_formulation(), FormulationSelection::Hybrid(vec![
            "interior_energy".to_string(), "equilibrium".to_string(),
            "outer_traction".to_string(), "external_work".to_string(),
        ]), "formulation must stay the legacy Hybrid baseline - PH3-14 found pure Variational diverges");
        assert!(!TrainingSpec::default().measure_aware_training, "measure_aware_training must stay off by default - PH3-15 found it fails even under the converging Hybrid formulation");
        assert!(TrainingSpec::default().amr_enabled, "amr_enabled must stay on by default - PH3-12's fixed-sampling result was no-hole-only, not shown to generalize to holed geometries");
        assert!(NetworkSpec::default().auto_stop_on_plateau, "auto_stop_on_plateau must stay on by default - PH3-14's premature-stop finding was Variational-specific, not shown to be wrong for the default Hybrid path");
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
        assert_eq!(parsed.architecture, ArchitectureSpec::default(), "omitting [architecture] must reproduce pre-#77-Phase-1..3 dispatch exactly");
    }

    /// Issue #77 PH4-45: `ArchitectureSpec::default()` must be the exact pre-existing dispatch
    /// - plain `Identity` ansatz (`hard_constraint_ansatz=false`), no sampling bias, Cartesian
    /// embedding, `Joint` (single-stage) training procedure. A future default change here would
    /// silently re-route every existing spec's training through a different architecture.
    #[test]
    fn architecture_spec_default_is_the_pre_existing_dispatch() {
        let a = ArchitectureSpec::default();
        assert!(!a.hard_constraint_ansatz);
        assert_eq!(a.hole_bias_fraction, 0.0);
        assert_eq!(a.coordinate_embedding, CoordinateEmbeddingSelection::Cartesian);
        assert_eq!(a.training_procedure, TrainingProcedure::Joint);
    }

    #[test]
    fn architecture_spec_round_trips_through_toml_for_every_variant() {
        let mut spec = sample_spec();
        for architecture in [
            ArchitectureSpec::default(),
            ArchitectureSpec { hard_constraint_ansatz: true, hole_bias_fraction: 0.5, ..Default::default() },
            ArchitectureSpec { coordinate_embedding: CoordinateEmbeddingSelection::LogPolar, ..Default::default() },
            ArchitectureSpec { training_procedure: TrainingProcedure::SingleDomain, ..Default::default() },
            ArchitectureSpec {
                training_procedure: TrainingProcedure::SequentialTwoStage { stage_a_steps: 1500, stage_b_steps: 1500 },
                ..Default::default()
            },
            ArchitectureSpec {
                hard_constraint_ansatz: true, hole_bias_fraction: 0.3,
                coordinate_embedding: CoordinateEmbeddingSelection::LogPolar,
                training_procedure: TrainingProcedure::SequentialTwoStage { stage_a_steps: 500, stage_b_steps: 2500 },
            },
        ] {
            spec.architecture = architecture;
            let toml_str = toml::to_string(&spec).expect("serialize");
            let parsed: ProblemSpec = toml::from_str(&toml_str).expect("deserialize");
            assert_eq!(parsed.architecture, architecture, "round-trip must preserve every field exactly");
        }
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
