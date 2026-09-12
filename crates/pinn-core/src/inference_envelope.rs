//! Phases 17-18 ("Training Once / Instant New-Input Evaluation" and "Inference Guardrails")
//! of the "Neural-Network-Wide Adaptive Collocation" epic.
//!
//! # Phase 0 finding this module is built on (verified, not assumed)
//!
//! `pinn_solver::network::ElasticityNet`'s input is `net_input_dim()` — either `3` (raw
//! `x, y, z=0`) or `4 * n_fourier` (a positional-Fourier embedding of `x, y`) — confirmed by
//! reading `pinn_solver::engine::EngineParams::net_input_dim` directly. **No geometry,
//! material, or load value is ever part of the network's input.** Every `ProblemSpec` field
//! other than the query point itself is baked into training only: geometry shapes the
//! collocation sampling domain and the hole boundary-condition loss terms
//! (`user_problem::HoleBcTerm`), material shapes the constitutive law used to normalize and
//! interpret the network's direct stress-column outputs (`ref_energy`, the
//! `constitutive_consistency` term), and load shapes the traction boundary targets and the
//! stress-column physical scale (`ref_stress2`, `px_pa`).
//!
//! The direct consequence: **a trained `UserDefinedProblem` model is a solution to exactly
//! one `ProblemSpec`, not a parametric surrogate over a family of them.** There is currently
//! no model checkpoint save/load in this codebase at all (verified: no `Recorder`/`.mpk`
//! usage anywhere in `pinn-solver`) — every trained model lives and dies within one training
//! run's process lifetime. This module does not change that; it defines the classification
//! and guardrail logic so that if/when checkpointing is added, "can I reuse this trained
//! model for a changed input" has an honest, evidence-based answer already in place, and so
//! that in-process re-evaluation of a live model against a hypothetically-changed spec (e.g.
//! a "what if" comparison during the same run) is guarded correctly today.
//!
//! Per the epic's own rule ("do not claim `Train once, change anything instantly`"), this
//! module's classification is deliberately conservative: every physical parameter is
//! `RequiresRetraining` (or a more specific retraining variant) except the query point
//! itself, which is the ONLY quantity the network was actually trained to generalize over.

use crate::problem_spec::ProblemSpec;
use crate::user_geometry::UserGeometry;

/// Phase 17: how a `ProblemSpec` field behaves if changed *without* retraining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterClass {
    /// The trained network was actually trained to generalize over this — currently only the
    /// spatial query point `(x, y)` within the trained domain.
    SafeForInference,
    /// The network's weights are still shape-compatible, but the physical law/normalization
    /// they encode no longer matches — retraining (same architecture) is required.
    RequiresRetraining,
    /// The domain shape itself changed (hole count/center/radius, plate extents) — collocation
    /// sampling and hole boundary terms must be rebuilt from scratch, not just re-fit.
    RequiresGeometryRegeneration,
    /// Only a hole's boundary condition (`Free`/`Fixed`) changed — geometry sampling is
    /// unaffected, but the loss term attached to that hole's ring changes.
    RequiresNewBoundaryConditionTraining,
    /// The network architecture itself changed — a differently-shaped model, not the same
    /// weights re-fit. There is nothing to "retrain"; a new model must be built.
    Unsupported,
}

/// Static, per-field description of one `ProblemSpec`-reachable parameter's inference
/// behavior — the Phase 17 "PARAMETER classification" table, determined from the actual
/// solver (see module doc), not assumed.
#[derive(Debug, Clone, Copy)]
pub struct ParameterEnvelope {
    pub name: &'static str,
    pub class: ParameterClass,
    pub reason: &'static str,
}

/// The full Phase 17 classification table for `UserDefinedProblem`. Every `ProblemSpec` field
/// reachable from a running/trained model appears exactly once. `training.*` is deliberately
/// absent — it has no meaning at inference time (it only shapes how training itself runs).
pub fn user_problem_parameter_envelope() -> Vec<ParameterEnvelope> {
    vec![
        ParameterEnvelope {
            name: "query point (x, y)",
            class: ParameterClass::SafeForInference,
            reason: "the only quantity the network's input layer actually takes — the trained \
                     model can be evaluated at any (x, y); accuracy outside the trained \
                     bounding box is unvalidated (see classify_inference's Yellow case), but \
                     the operation itself needs no retraining.",
        },
        ParameterEnvelope {
            name: "geometry.half_w / geometry.half_h / geometry.thickness",
            class: ParameterClass::RequiresGeometryRegeneration,
            reason: "the plate extents define the interior/boundary collocation sampling \
                     domain (UserSamplingStrategy::sample_interior/sample_boundary) — changing \
                     them changes which points the network was ever trained against.",
        },
        ParameterEnvelope {
            name: "geometry.holes (count, center, radius)",
            class: ParameterClass::RequiresGeometryRegeneration,
            reason: "each hole gets its own named collocation ring (UserSamplingStrategy::\
                     named_point_sets) and its own HoleBcTerm; adding, moving, or resizing a \
                     hole changes the domain and the loss terms the network was fit to, not \
                     just an input value.",
        },
        ParameterEnvelope {
            name: "geometry.holes[i].bc (Free/Fixed)",
            class: ParameterClass::RequiresNewBoundaryConditionTraining,
            reason: "HoleBcTerm's loss expression itself changes (hole_traction_loss_direct vs \
                     a soft zero-displacement anchor) — the domain/sampling is untouched, only \
                     what was trained to hold at that hole's boundary.",
        },
        ParameterEnvelope {
            name: "material.e / material.nu",
            class: ParameterClass::RequiresRetraining,
            reason: "bakes into ref_energy (InteriorEnergyTerm's normalization) and the \
                     constitutive law the constitutive_consistency term trains sigma_net \
                     against — the network's direct stress-column outputs were fit to satisfy \
                     Hooke's law for THIS E/nu, not a general one.",
        },
        ParameterEnvelope {
            name: "material.density / material.ultimate_strength_pa",
            class: ParameterClass::RequiresRetraining,
            reason: "ultimate_strength_pa can additionally change the stress normalization \
                     reference when use_ultimate_strength_scaling is enabled; density is \
                     currently unused by UserDefinedProblem's static loss terms but is part of \
                     the same MaterialProps the trained model is fit against.",
        },
        ParameterEnvelope {
            name: "load.px / load.py",
            class: ParameterClass::RequiresRetraining,
            reason: "sets ref_stress2 (stress-column physical scale) and the Neumann/hole \
                     traction boundary targets directly — the network's raw stress outputs are \
                     only meaningful multiplied by the px_pa scale they were trained under.",
        },
        ParameterEnvelope {
            name: "network.hidden_dim / network.n_hidden",
            class: ParameterClass::Unsupported,
            reason: "changes the weight tensor shapes themselves — there is no trained weight \
                     to reuse or fine-tune, this is a structurally different model that must \
                     be built and trained from nothing.",
        },
    ]
}

/// Phase 18: the outcome of checking one candidate inference request (a `requested` spec,
/// optionally at a specific query point) against the spec a model was actually `trained` on.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceClass {
    /// Requested spec matches the trained spec in every field this module checks, and (if a
    /// query point was given) that point lies inside the trained geometry's validated region.
    Green,
    /// Spec matches, but the query point is outside the trained domain (extrapolation) — the
    /// model was never shown data there; treat the result as unvalidated, not wrong.
    Yellow(String),
    /// Spec differs from the trained spec in a way `user_problem_parameter_envelope`
    /// classifies as anything other than `SafeForInference` — retraining (of some kind) is
    /// required before this request's output can be trusted.
    Red(String),
}

/// Classify a candidate inference request against the `ProblemSpec` a model was trained on.
/// Checks fields in the same order as `user_problem_parameter_envelope` (network → material →
/// load → geometry) so the FIRST reported mismatch is the most structurally severe one.
/// `training` fields are never compared — see this module's doc comment.
pub fn classify_inference(
    trained: &ProblemSpec,
    requested: &ProblemSpec,
    query_point: Option<(f64, f64)>,
) -> InferenceClass {
    if requested.network != trained.network {
        return InferenceClass::Red(format!(
            "network architecture changed (hidden_dim {}->{}, n_hidden {}->{}): {:?} — {}",
            trained.network.hidden_dim, requested.network.hidden_dim,
            trained.network.n_hidden, requested.network.n_hidden,
            ParameterClass::Unsupported,
            "a new model must be built and trained; no weights can be reused",
        ));
    }
    if requested.material.e != trained.material.e || requested.material.nu != trained.material.nu {
        return InferenceClass::Red(format!(
            "material.e/nu changed ({:.4e}/{:.4}->{:.4e}/{:.4}): {:?} — the trained stress \
             outputs no longer satisfy this material's constitutive law",
            trained.material.e, trained.material.nu, requested.material.e, requested.material.nu,
            ParameterClass::RequiresRetraining,
        ));
    }
    if requested.material.ultimate_strength_pa != trained.material.ultimate_strength_pa
        || requested.material.density != trained.material.density
    {
        return InferenceClass::Red(format!(
            "material.density/ultimate_strength_pa changed: {:?} — retraining required",
            ParameterClass::RequiresRetraining,
        ));
    }
    if requested.load.px != trained.load.px || requested.load.py != trained.load.py {
        return InferenceClass::Red(format!(
            "load changed (px {:.4e}->{:.4e}, py {:.4e}->{:.4e}): {:?} — boundary targets and \
             stress scale no longer match what the network was trained against",
            trained.load.px, requested.load.px, trained.load.py, requested.load.py,
            ParameterClass::RequiresRetraining,
        ));
    }
    if let Some(class) = geometry_mismatch_class(&trained.geometry, &requested.geometry) {
        let reason = match class {
            ParameterClass::RequiresGeometryRegeneration =>
                "hole count/center/radius or plate extents changed — the collocation domain \
                 itself differs from what was trained",
            ParameterClass::RequiresNewBoundaryConditionTraining =>
                "only a hole's boundary condition changed — domain is unchanged, but that \
                 hole's trained boundary behavior is not",
            _ => unreachable!("geometry_mismatch_class only returns the two geometry variants"),
        };
        return InferenceClass::Red(format!("{class:?} — {reason}"));
    }

    if let Some((x, y)) = query_point {
        if !trained.geometry.contains(x, y) {
            return InferenceClass::Yellow(format!(
                "query point ({x}, {y}) lies outside the trained geometry's validated region \
                 (outside the plate bound, or inside an excluded hole) — result is \
                 extrapolated, not validated"
            ));
        }
    }

    InferenceClass::Green
}

/// `None` if `a`/`b` describe the same domain; otherwise the most specific
/// `ParameterClass` describing what changed (geometry regeneration takes priority over a
/// pure boundary-condition change when both differ).
fn geometry_mismatch_class(a: &UserGeometry, b: &UserGeometry) -> Option<ParameterClass> {
    if a.half_w != b.half_w || a.half_h != b.half_h || a.thickness != b.thickness
        || a.holes.len() != b.holes.len()
    {
        return Some(ParameterClass::RequiresGeometryRegeneration);
    }
    let mut bc_only_diff = false;
    for (ha, hb) in a.holes.iter().zip(b.holes.iter()) {
        if ha.center != hb.center || ha.radius != hb.radius {
            return Some(ParameterClass::RequiresGeometryRegeneration);
        }
        if ha.bc != hb.bc {
            bc_only_diff = true;
        }
    }
    if bc_only_diff { Some(ParameterClass::RequiresNewBoundaryConditionTraining) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{loading::LoadConfig, material::MaterialProps, problem_spec::{NetworkSpec, TrainingSpec}, user_geometry::{HoleBc, HoleSpec}};

    fn base_spec() -> ProblemSpec {
        ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.05, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.01, bc: HoleBc::Free }],
            },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec::default(),
            training: TrainingSpec::default(),
            formulation: crate::problem_spec::default_formulation(),
        }
    }

    #[test]
    fn envelope_classifies_query_point_as_the_only_safe_parameter() {
        let table = user_problem_parameter_envelope();
        let safe: Vec<_> = table.iter().filter(|p| p.class == ParameterClass::SafeForInference).collect();
        assert_eq!(safe.len(), 1, "exactly one parameter should be SafeForInference: the query point");
        assert_eq!(safe[0].name, "query point (x, y)");
    }

    #[test]
    fn envelope_never_leaves_a_problem_spec_field_family_unclassified() {
        let table = user_problem_parameter_envelope();
        let names: Vec<&str> = table.iter().map(|p| p.name).collect();
        for expect in ["geometry.half_w", "geometry.holes", "material.e", "load.px", "network.hidden_dim"] {
            assert!(names.iter().any(|n| n.contains(expect)), "missing classification for {expect}");
        }
    }

    #[test]
    fn classify_inference_identical_spec_and_point_inside_domain_is_green() {
        let spec = base_spec();
        assert_eq!(classify_inference(&spec, &spec, Some((0.05, 0.02))), InferenceClass::Green);
    }

    #[test]
    fn classify_inference_identical_spec_no_point_is_green() {
        let spec = base_spec();
        assert_eq!(classify_inference(&spec, &spec, None), InferenceClass::Green);
    }

    #[test]
    fn classify_inference_material_change_is_red_requires_retraining() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.material.e *= 1.5;
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("RequiresRetraining")),
            other => panic!("expected Red, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_load_change_is_red_requires_retraining() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.load.px *= 2.0;
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("RequiresRetraining")),
            other => panic!("expected Red, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_network_change_is_red_unsupported() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.network.hidden_dim += 32;
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("Unsupported")),
            other => panic!("expected Red, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_hole_added_is_red_requires_geometry_regeneration() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.geometry.holes.push(HoleSpec { center: [0.03, 0.0], radius: 0.005, bc: HoleBc::Free });
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("RequiresGeometryRegeneration")),
            other => panic!("expected Red, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_hole_bc_only_change_is_red_requires_new_bc_training() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.geometry.holes[0].bc = HoleBc::Fixed;
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("RequiresNewBoundaryConditionTraining")),
            other => panic!("expected Red, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_point_outside_bounding_box_is_yellow() {
        let spec = base_spec();
        match classify_inference(&spec, &spec, Some((5.0, 5.0))) {
            InferenceClass::Yellow(msg) => assert!(msg.contains("outside")),
            other => panic!("expected Yellow, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_point_inside_hole_is_yellow() {
        let spec = base_spec();
        match classify_inference(&spec, &spec, Some((0.0, 0.0))) {
            InferenceClass::Yellow(_) => {}
            other => panic!("expected Yellow, got {other:?}"),
        }
    }

    #[test]
    fn classify_inference_checks_network_before_material_reports_most_severe_first() {
        let trained = base_spec();
        let mut requested = trained.clone();
        requested.network.hidden_dim += 1;
        requested.material.e *= 2.0;
        match classify_inference(&trained, &requested, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("Unsupported"), "network mismatch should win: {msg}"),
            other => panic!("expected Red, got {other:?}"),
        }
    }
}
