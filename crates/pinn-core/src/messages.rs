use ndarray::Array2;
use crate::geometry::GeometryConfig;
use crate::loading::LoadConfig;
use crate::material::MaterialProps;

/// Per-AMR-sweep effectiveness report (Phase 11, "Neural-Network-Wide Adaptive Collocation"
/// epic) — before/after residual + point count + timing for ONE sweep event on one domain.
/// Attached to whichever `TrainingUpdate`/`PinLugTrainingUpdate` corresponds to the step the
/// sweep fired on; absent (`None`/empty) on every other step. This is the concrete data
/// behind "did this sweep actually help, and what did it cost" — not just "the collocation
/// count changed".
#[derive(Debug, Clone)]
pub struct AmrSweepReport {
    /// Which domain this sweep ran on — e.g. `"interior"` (single-domain `UserDefinedProblem`)
    /// or `"pin"`/`"lug"` (pin-in-lug's two independently-gridded domains).
    pub domain_label: &'static str,
    pub step: usize,
    pub points_before: usize,
    pub points_after: usize,
    pub residual_rms_before: f64,
    pub residual_max_before: f64,
    pub residual_rms_after: f64,
    pub residual_max_after: f64,
    pub sweep_duration_ms: f64,
    /// Phase 8 ("Plate-With-Hole Physics Validation") diagnostic: mean local point density
    /// (see `amr::AdaptiveGrid::lock_zone_density`) at this domain's hole zone(s), before vs.
    /// after this sweep — `0.0`/`0.0` for a geometry with no lock zones (nothing to
    /// concentrate near, not a missing value). Answers "did AMR actually identify the hole
    /// region as needing resolution" with a real number, not just "the point count changed".
    pub hole_zone_density_before: f64,
    pub hole_zone_density_after: f64,
    /// Domain-wide mean point density (`points / domain_area`) at the same two instants —
    /// the denominator `hole_zone_density_*` is meaningfully compared against. A ratio well
    /// above 1.0 (hole zone denser than the domain average) is the concrete, numeric answer
    /// to "is AMR targeting the hole", not a plausible-looking visualization.
    pub domain_mean_density_before: f64,
    pub domain_mean_density_after: f64,
}

/// `enhancement.md` Phase 9 ("Force Equilibrium Validation") — result of `pinn_solver::
/// user_problem::probe_reaction_force`/`parametric_problem::reaction_force_stats`. Lives here
/// (not in `pinn-solver`) for the same reason `HoleBoundaryPoint`/`VisFields` do: `pinn-core`
/// owns the SHAPE of any solver-computed value that needs to travel inside a `TrainingUpdate`,
/// even though only `pinn-solver` knows how to produce one (`pinn-core` never depends on
/// `pinn-solver`).
///
/// `net_fx`/`net_fy` are the model's PREDICTED net force [N] integrated over the entire outer
/// boundary — this should be close to zero for a converged solution. The applied far-field
/// traction target is, by construction, self-canceling around the whole closed rectangle (it
/// pulls one edge one way and the opposite edge the other way), so there is no separate
/// nonzero "applied resultant" to compare against; a nonzero PREDICTED net force is itself the
/// meaningful inconsistency. `reference_force` [N] is the nominal one-edge load magnitude used
/// to normalize `equilibrium_error` into a scale-free ratio.
#[derive(Debug, Clone, Copy)]
pub struct ReactionForce {
    pub net_fx: f64,
    pub net_fy: f64,
    pub reference_force: f64,
    pub equilibrium_error: f64,
}

/// `enhancement.md` Phase 10 ("Energy Validation") — result of `pinn_solver::user_problem::
/// probe_energy_balance`/`parametric_problem::energy_balance_stats`. Lives here for the same
/// reason `ReactionForce` does (see that type's doc comment).
///
/// **Distinct from `TrainingUpdate::energy_loss`/`ParametricTrainingUpdate::energy_loss`**,
/// which are the raw, un-integrated, SAW-BRDR-weighted optimizer LOSS TERM value — NOT a
/// physical energy in joules (`enhancement.md` Phase 10's own explicit warning: "do not label
/// a quantity 'energy error' if it is merely the training loss"). `internal_energy` here is a
/// genuine Monte-Carlo domain integral of the per-point strain energy density over the
/// plate's real area×thickness; `external_work` is a genuine `∮ t·u ds` integral over the
/// loaded boundary. For a converged linear-elastic solution under pure traction loading (no
/// body force), the work-energy theorem requires these to be equal — `energy_balance_error`
/// is how far apart they are, normalized by `external_work`'s own magnitude.
#[derive(Debug, Clone, Copy)]
pub struct EnergyBalance {
    pub internal_energy: f64,
    pub external_work: f64,
    pub energy_balance_error: f64,
}

/// Stage I ("live network-evolution visualization", the user's own explicit follow-up ask) —
/// a per-layer snapshot of `pinn_solver::network::ElasticityNet`'s weight tensors, read at the
/// same vis cadence `VisFields` already uses (never inside the per-step hot loop — see
/// `pinn_solver::network::network_snapshot`'s doc comment for why this adds zero training-loop
/// cost). `layer_mean_abs_weight`/`layer_max_abs_weight` are parallel, one entry per hidden
/// layer (input layer first). `awake_mask` is empty whenever PirateNet gating is disabled —
/// true for every problem type this toolbox currently trains (`NetworkSpec` has no
/// `use_piratenet` field), kept here anyway so this type stays correct if that ever changes.
///
/// `layer_weights` carries the REAL end-to-end weight matrices (approved neuron-and-edge
/// diagram, not the earlier per-layer bar-chart version) — one entry per `Linear` layer
/// INCLUDING the final output projection (`ElasticityNet::all_weight_matrices`), shape
/// `[d_input, d_output]` each, unlike `layer_mean_abs_weight`/`awake_mask` which deliberately
/// exclude the output layer to mirror `awake_mask`'s own scope — a wiring diagram needs the
/// complete input-to-output path to be meaningful. Payload is small (a 64×64 hidden layer is
/// 16 KB; the input/output layers are far smaller) and sent only at the existing vis cadence.
#[derive(Debug, Clone)]
pub struct NetworkSnapshot {
    pub layer_mean_abs_weight: Vec<f32>,
    pub layer_max_abs_weight: Vec<f32>,
    pub awake_mask: Vec<bool>,
    pub layer_weights: Vec<Array2<f32>>,
}

/// Smart adaptive architecture — sent exactly once on the step a
/// `pinn_solver::architecture_controller::ArchitectureController` action was actually applied
/// to the live model (never every step, and never on a step nothing happened — the UI treats
/// `Some` as a one-shot event, e.g. a loss-chart marker, not an ongoing status). `NetworkSpec::
/// adaptive` gates whether this can ever be populated; `None` on every non-adaptive run
/// (including Kirsch/pin-lug, which have no `adaptive` field at all).
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureEvent {
    pub step: usize,
    /// Human-readable summary of what happened, e.g. "Grew width 64 -> 96",
    /// "Removed dormant layer 2", "Pruned 4 neurons from layer 1", "Reverted last change
    /// (no improvement)" — built by the caller from the `ArchAction` it just applied.
    pub description: String,
    pub hidden_dim_before: usize,
    pub hidden_dim_after: usize,
    pub n_hidden_before: usize,
    pub n_hidden_after: usize,
}

/// Command sent from GUI thread → solver thread
pub enum ControlMsg {
    Stop,
    Pause,
    Resume,
    /// Trigger a warm-start with new configuration
    WarmStart { config: SolverConfig, geometry_changed: bool },
    /// Request that the pin-in-lug training loop export the current contact-pressure
    /// profile to CSV (see `pinn_solver::contact_export`). No-op (treated as `Continue`)
    /// on the single-domain Kirsch path — there is nothing to export there.
    ExportContactPressure,
    /// Parametric PINN "instant inference" request (`enhancement.txt` items 7/8/17 - "train
    /// once, change parameters, get a new solution instantly") - query the just-trained
    /// parametric model at a NEW `(e, nu, px)` without retraining. Only meaningful after
    /// `TrainingMsg::ParametricReady`; a no-op before that (nothing trained yet to query) or
    /// on any non-parametric training path.
    ParametricInfer { e: f64, nu: f64, px: f64 },
    /// Stage H (model checkpoint save/load) - save the current model's weights + a metadata
    /// sidecar to disk. Only meaningful once the model has reached a stable, queryable state
    /// (after `TrainingMsg::Done`/`ParametricReady` - the solver thread stays alive to serve
    /// exactly this, mirroring the existing `ParametricInfer` post-training serving loop). A
    /// no-op on any path that doesn't yet implement a save handler for its own control loop.
    /// `saved_at_unix` is stamped by the UI thread (which naturally has clock access for a
    /// real user-triggered action) rather than read inside the solver thread, keeping every
    /// solver-side probe/computation in this codebase a pure function of its arguments.
    SaveCheckpoint { path: std::path::PathBuf, saved_at_unix: u64 },
}

/// Data sent from solver thread → GUI thread (bounded channel capacity=1)
pub enum TrainingMsg {
    Update(Box<TrainingUpdate>),
    /// Pin-in-lug analogue of `Update` — carries both domains' visualization fields and a
    /// generic convergence metric instead of Kirsch's K_t.
    PinLugUpdate(Box<PinLugTrainingUpdate>),
    /// `toy_beam` analogue of `Update` — a 1D beam has no spatial field/heatmap concept and
    /// no separate energy/BC loss split (one combined potential-energy scalar), so it
    /// carries its own shape (a single loss value plus the network-vs-exact comparison
    /// points) instead of being force-fit into `TrainingUpdate`.
    BeamUpdate(Box<BeamTrainingUpdate>),
    /// `ParametricProblemSpec` analogue of `Update` — see `ParametricTrainingUpdate`'s doc
    /// comment for why this needs its own shape (a per-step sampled `(e, nu, px)` triple,
    /// no K_t, no AMR in v1).
    ParametricUpdate(Box<ParametricTrainingUpdate>),
    /// Sent exactly once, when the parametric training loop reaches `training.max_steps` (or
    /// is stopped early) and transitions from "training" to "serving instant-inference
    /// requests" - the model is NOT dropped after this; the training thread stays alive
    /// blocked on `ControlMsg::ParametricInfer`/`Stop` (see `pinn_solver::parametric_problem`'s
    /// module doc for why this is the chosen way to keep a trained model queryable without a
    /// checkpoint-persistence mechanism, which this codebase doesn't have - see
    /// `pinn_core::inference_envelope`'s doc comment).
    ParametricReady,
    /// Result of a `ControlMsg::ParametricInfer` request - the model evaluated at the
    /// requested `(e, nu, px)`, plus whether that request fell inside the trained ranges
    /// (`ParametricProblemSpec::in_range`, a min/max check - see that method's doc comment for
    /// why a real distance-to-training-distribution metric is a stated future refinement, not
    /// this v1's scope).
    ParametricInferResult(Box<ParametricInferenceResult>),
    Done,
    Error(String),
    /// Contact-pressure CSV export finished successfully; carries the written file path.
    ExportComplete(String),
    /// Result of a `ControlMsg::SaveCheckpoint` request - `Ok(path)` on success (the exact
    /// weights-file path actually written, which may differ slightly from the requested path
    /// once the recorder's own extension is appended), `Err(message)` on I/O/serialization
    /// failure. Never silently swallowed - the UI surfaces either outcome as a toast.
    CheckpointSaved(Result<String, String>),
}

/// `ParametricProblemSpec` analogue of `TrainingUpdate` — see `TrainingMsg::ParametricUpdate`.
pub struct ParametricTrainingUpdate {
    pub step: usize,
    pub total_loss: f32,
    pub energy_loss: f32,
    /// Sum of every boundary-condition term this step (outer traction + every hole term) —
    /// mirrors `PinLugTrainingUpdate::neumann_loss`'s own "documented approximation: sum of
    /// all non-energy BC-term scalars" convention.
    pub boundary_loss: f32,
    pub lr: f32,
    /// The `(E, nu, Px)` triple THIS step's forward/backward pass was actually trained
    /// against - real-time evidence the network is seeing the full range over the course of
    /// training, not evidence of convergence at any single point in it.
    pub e_this_step: f64,
    pub nu_this_step: f64,
    pub load_this_step: f64,
    /// `enhancement.txt` item B ("Gradient Norm") — L2 norm of every weight gradient this
    /// step.
    pub grad_norm: f32,
    /// `enhancement.txt` items 4/C ("BC residual RMS/max") — real per-point traction/
    /// displacement residual at the outer boundary and every hole ring this step, combined.
    /// Distinct from `VisFields::pde_residual` (the interior constitutive-consistency
    /// residual) — see that field's doc comment.
    pub bc_residual_rms: f64,
    pub bc_residual_max: f64,
    /// `enhancement.md` Phase 9 — see `ReactionForce`'s doc comment. Computed only on the same
    /// cadence `vis` is (`None` otherwise — a real absence, not a `0.0` sentinel).
    pub reaction_force: Option<ReactionForce>,
    /// `enhancement.md` Phase 10 — see `EnergyBalance`'s doc comment. Same "only on the vis
    /// cadence" convention as `reaction_force` above.
    pub energy_balance: Option<EnergyBalance>,
    /// Stage I — see `NetworkSnapshot`'s doc comment. Same "only on the vis cadence"
    /// convention as every other `Option` field above.
    pub network_snapshot: Option<NetworkSnapshot>,
    /// Smart adaptive architecture — see `ArchitectureEvent`'s doc comment. `Some` only on the
    /// exact step an action was applied.
    pub architecture_event: Option<ArchitectureEvent>,
    /// Visualization at `e_this_step`/`nu_this_step`/`load_this_step` — sent on the same
    /// periodic cadence `VisFields` uses elsewhere, not every step.
    pub vis: Option<VisFields>,
    pub hole_analyses: Vec<HoleAnalysis>,
}

/// Result of one `ControlMsg::ParametricInfer` request — see `TrainingMsg::
/// ParametricInferResult`'s doc comment.
#[derive(Debug, Clone)]
pub struct ParametricInferenceResult {
    pub e: f64,
    pub nu: f64,
    pub px: f64,
    pub in_range: bool,
    /// `enhancement.txt` items 11/12 ("physics-based safety check", GREEN/YELLOW/RED) — the
    /// real BC residual RMS/max AT THIS QUERY POINT, computed without retraining. `vis.
    /// pde_residual` carries the equivalent interior-residual field; this is its
    /// boundary-only, scalar analogue. The UI combines this with `in_range` to classify the
    /// result, rather than trusting the parameter-range check alone.
    pub bc_residual_rms: f64,
    pub bc_residual_max: f64,
    /// `enhancement.md` Phase 9 — see `ReactionForce`'s doc comment. Always `Some` for this
    /// result type (every inference query computes it, unlike the periodic training update).
    pub reaction_force: ReactionForce,
    /// `enhancement.md` Phase 10 — see `EnergyBalance`'s doc comment. Always computed for
    /// this result type (every inference query computes it).
    pub energy_balance: EnergyBalance,
    /// `enhancement.md` Phase 21 ("Do Not Rely Only on Min/Max") — this query's nearest-
    /// neighbor distance (in normalized `[-1,1]^3` `(e_n,nu_n,p_n)` space) to the closest
    /// `(E,nu,Px)` triple actually drawn during training, from a bounded recent-sample
    /// reservoir (see `pinn_solver::parametric_problem::run_training_parametric`'s doc
    /// comment). `f64::INFINITY` if the reservoir was empty (no coverage information yet).
    pub nearest_sample_distance: f64,
    /// The reservoir's own median pairwise nearest-neighbor spacing (`pinn_core::
    /// param_distance::median_nn_spacing`) — the self-baseline `nearest_sample_distance`
    /// should be compared against as a ratio, not an arbitrary absolute constant. `0.0` if
    /// the reservoir had fewer than 2 samples.
    pub typical_sample_spacing: f64,
    pub vis: VisFields,
    pub hole_analyses: Vec<HoleAnalysis>,
}

/// `toy_beam` analogue of `TrainingUpdate` — see `TrainingMsg::BeamUpdate`.
pub struct BeamTrainingUpdate {
    pub step: usize,
    pub max_steps: usize,
    pub loss: f32,
    pub max_abs_error: f64,
    pub max_abs_deflection: f64,
    /// `(x, w_net(x), w_exact(x))` at each evaluation-grid point — mirrors
    /// `pinn_solver::toy_beam::ToyBeamResult::eval_points` exactly.
    pub eval_points: Vec<(f64, f64, f64)>,
}

/// One angular sample around a hole's circumference (Phase 10, "Neural-Network-Wide Adaptive
/// Collocation" epic) — see `pinn_solver::user_problem::probe_hole_boundary_profile`'s doc
/// comment for how this is computed. Lives here (not in `pinn-solver`) for the same reason
/// `VisFields` does: `pinn-core` defines the data SHAPE a solver-computed message carries,
/// even though only `pinn-solver` knows how to produce one — `pinn-core` never depends on
/// `pinn-solver`, so a type traveling inside a `TrainingUpdate` can't live on the solver side.
#[derive(Debug, Clone, Copy)]
pub struct HoleBoundaryPoint {
    pub theta_deg: f64,
    pub x: f64,
    pub y: f64,
    pub ux: f32,
    pub uy: f32,
    /// Tensor-convention shear strain — see `pinn_solver::user_problem`'s own doc comment
    /// on this field for the Phase 9 Von Mises pipeline audit that verified this convention.
    pub eps_xx: f32,
    pub eps_yy: f32,
    pub eps_xy: f32,
    pub sxx: f32,
    pub syy: f32,
    pub sxy: f32,
    pub von_mises: f32,
}

/// Stress-concentration summary derived from a `HoleBoundaryPoint` profile — see
/// `pinn_solver::user_problem::stress_concentration_from_profile`'s doc comment. Deliberately
/// NOT compared against a hardcoded Kt=3 anywhere this travels.
#[derive(Debug, Clone, Copy)]
pub struct StressConcentration {
    pub nominal_stress: f64,
    pub max_von_mises: f64,
    pub max_theta_deg: f64,
    pub kt: f64,
}

/// One hole's full stress analysis, bundled for transport in a `TrainingUpdate` (Phase 16,
/// "Final Results Dashboard", of the "Neural-Network-Wide Adaptive Collocation" epic).
#[derive(Debug, Clone)]
pub struct HoleAnalysis {
    /// Index into the originating `UserGeometry::holes` — lets a UI label "Hole 1"/"Hole 2"
    /// consistently across updates without needing the geometry itself in scope.
    pub hole_index: usize,
    pub profile: Vec<HoleBoundaryPoint>,
    pub concentration: StressConcentration,
}

/// Transport-side mirror of `pinn_solver::training_core::GradientShareReport` — a separate
/// type (not the solver's own) so `pinn-core` never needs to depend on `pinn-solver`, same
/// pattern `HoleBoundaryPoint`/`StressConcentration` already established for this exact reason
/// (see their own doc comments). General-PINN architecture recommendations §15/§30: which
/// active loss term's gradient dominates optimization, and which are functionally inert.
#[derive(Debug, Clone)]
pub struct GradientShareSummary {
    /// `(term_name, share)` pairs, `share = ||grad_i|| / Σ_j ||grad_j||`, summing to ~1.0.
    pub shares: Vec<(&'static str, f32)>,
    pub inert: Vec<&'static str>,
    pub dominant: Option<&'static str>,
}

/// Transport-side mirror of `pinn_solver::training_core::GradientConflictReport` — same
/// "pinn-core never depends on pinn-solver" split `GradientShareSummary` already established.
/// General-PINN architecture recommendations §17: pairwise gradient cosine similarity between
/// every two active loss terms - `most_conflicting` is the strongest active disagreement this
/// step (`None` when no pair is negative).
#[derive(Debug, Clone)]
pub struct GradientConflictSummary {
    /// `(term_a, term_b, cosine_similarity)` triples, one per distinct pair of active terms.
    pub pairs: Vec<(&'static str, &'static str, f32)>,
    pub most_conflicting: Option<(&'static str, &'static str, f32)>,
}

pub struct TrainingUpdate {
    pub step: usize,
    pub total_loss:   f32,
    pub energy_loss:  f32,
    pub neumann_loss: f32,
    pub lr:           f32,
    pub lam_energy:   f32,
    pub lam_neumann:  f32,
    pub n_colloc:     usize,
    pub kt_estimate:  Option<f32>,
    pub vis: Option<VisFields>,
    /// Set only on the step an AMR sweep actually fired (see `AmrSweepReport`'s doc comment).
    pub amr_sweep: Option<AmrSweepReport>,
    /// Per-hole stress analysis (Phase 16, "Final Results Dashboard") — populated on the
    /// same cadence as `vis` (empty otherwise), one entry per hole in the originating
    /// geometry. Empty for problems with no user-defined geometry (Kirsch's own path).
    pub hole_analyses: Vec<HoleAnalysis>,
    /// `enhancement.txt` item B ("Gradient Norm") — mirrors `training_core::StepOutput::
    /// grad_norm`'s own doc comment for what this is and why it's always real (not gated
    /// behind a diagnostics flag).
    pub grad_norm: Option<f32>,
    /// `enhancement.txt` items 4/C ("BC residual RMS/max") — real per-point traction/
    /// displacement residual at the outer boundary and every hole ring, sent on the same
    /// cadence as `vis` (`0.0`/`0.0` otherwise, not a meaningful "no data" sentinel since a
    /// genuine zero residual is also a valid value — check alongside `vis.is_some()`).
    pub bc_residual_rms: f64,
    pub bc_residual_max: f64,
    /// `enhancement.md` Phase 9 — see `ReactionForce`'s doc comment. Computed only on the same
    /// cadence `vis` is (`None` otherwise). `None` on Kirsch's own path too (deliberately not
    /// wired there — see `powershell_tool/CLAUDE.md`'s note on why BC residual was likewise
    /// only added to the `UserDefinedProblem`/parametric paths, not Kirsch/pin-lug).
    pub reaction_force: Option<ReactionForce>,
    /// `enhancement.md` Phase 10 — see `EnergyBalance`'s doc comment. Same treatment as
    /// `reaction_force` above (Kirsch's own path leaves this `None`).
    pub energy_balance: Option<EnergyBalance>,
    /// Stage I ("live network-evolution visualization") — see `NetworkSnapshot`'s doc comment.
    /// Same "only on the vis cadence" convention as every other `Option` field above.
    pub network_snapshot: Option<NetworkSnapshot>,
    /// Smart adaptive architecture — see `ArchitectureEvent`'s doc comment. `Some` only on the
    /// exact step an action was applied, `None` on every other step (not a vis-cadence field).
    pub architecture_event: Option<ArchitectureEvent>,
    /// General-PINN architecture recommendations §15/§30, generalized from this session's own
    /// ad-hoc `term_grad_norms` diagnostic — see `GradientShareSummary`'s doc comment. `Some`
    /// only on the same vis cadence `hole_analyses`/`bc_residual_rms` already use (an extra
    /// backward pass per active term is real cost, never paid every step).
    pub gradient_share_report: Option<GradientShareSummary>,
    /// General-PINN architecture recommendations §17 (Priority 4, "gradient conflict
    /// diagnostics") - see `GradientConflictSummary`'s doc comment. Same vis-cadence gate as
    /// `gradient_share_report` (built from the SAME per-term backward pass, no extra cost).
    pub gradient_conflict_report: Option<GradientConflictSummary>,
    /// General-PINN architecture recommendations §4 (Priority 1, "physics dependency graph"),
    /// narrowed to the one edge this codebase's own real bugs were about - see
    /// `pinn_solver::problem::StressSource`'s doc comment. `(term_name, "Direct"/"Derived"/
    /// "Both")` pairs - plain strings, not the solver's own enum, same "pinn-core never
    /// depends on pinn-solver" rule `GradientShareSummary` already established. Static per
    /// problem (doesn't change step to step) - genuinely free to compute every update, unlike
    /// `gradient_share_report`, so this is never gated/`None`, just possibly empty (Kirsch's
    /// own path, which doesn't drive its per-step computation through a `BoundaryValueProblem`
    /// trait object at all - see that path's own doc comment).
    pub stress_source_report: Vec<(&'static str, &'static str)>,
    /// General-PINN architecture recommendations §13 (Priority 5, "generic boundary operator
    /// system") - which classical PDE boundary-condition family each active term enforces, if
    /// any. `(term_name, "Dirichlet"/"Neumann"/"Robin"/"Periodic"/"Symmetry"/"Interface")`
    /// pairs - plain strings, not the solver's own enum, same "pinn-core never depends on
    /// pinn-solver" rule `stress_source_report` already established. Same "static per problem,
    /// never gated" treatment as `stress_source_report` too.
    pub boundary_operator_report: Vec<(&'static str, &'static str)>,
    /// General-PINN architecture recommendations §10 (Priority 6, "generic derivative
    /// backend") - which order of spatial derivative each active term needs, if any.
    /// `(term_name, "First"/"Second")` pairs - plain strings, not the solver's own enum, same
    /// "pinn-core never depends on pinn-solver" rule `stress_source_report`/`boundary_
    /// operator_report` already established. Same "static per problem, never gated" treatment.
    pub derivative_order_report: Vec<(&'static str, &'static str)>,
}

/// Pin-in-lug analogue of `TrainingUpdate` — one entry per domain's visualization fields,
/// plus a generic (non-K_t) convergence metric.
pub struct PinLugTrainingUpdate {
    pub step: usize,
    pub total_loss:   f32,
    /// Documented approximation: sum of both domains' interior-energy scalars.
    pub energy_loss:  f32,
    /// Documented approximation: sum of all non-energy BC-term scalars.
    pub neumann_loss: f32,
    pub lr:           f32,
    pub lam_energy:   f32,
    pub lam_neumann:  f32,
    /// Pin + lug interior point counts, summed.
    pub n_colloc:     usize,
    /// Interface-gap RMS (see `PinLugProblem::convergence_metric`) — deliberately NOT named
    /// `kt_estimate`; pin-in-lug has no closed-form K_t.
    pub convergence_metric: Option<f32>,
    pub vis: Option<PinLugVisFields>,
    /// 0, 1, or 2 entries (pin and/or lug) - populated only on the step an AMR sweep
    /// actually fired for that domain. See `AmrSweepReport`'s doc comment.
    pub amr_sweep: Vec<AmrSweepReport>,
    /// `enhancement.txt` item B ("Gradient Norm") — mirrors `training_core::StepOutput::
    /// grad_norm`. No BC-residual equivalent here (unlike `TrainingUpdate`) — pin-lug's
    /// boundary condition is Signorini contact (penetration/non-tension KKT terms), not a
    /// simple prescribed traction, so "BC residual" isn't the same well-defined quantity;
    /// giving it one would mean inventing a new metric definition, not surfacing an existing
    /// one, and this pass didn't do that.
    pub grad_norm: Option<f32>,
}

/// Visualization fields — sent every 10 steps (not every step, to keep channel fast).
///
/// Phase 14 ("Spatial Diagnostic Visualization") of the "Neural-Network-Wide Adaptive
/// Collocation" epic added the six fields below the original stress/displacement set — all
/// the same `(ny, nx)` shape, NaN-masked outside the domain exactly like the original six.
#[derive(Debug, Clone)]
pub struct VisFields {
    pub von_mises: Array2<f32>,
    pub sigma_xx:  Array2<f32>,
    pub sigma_yy:  Array2<f32>,
    pub sigma_xy:  Array2<f32>,
    pub disp_u:    Array2<f32>,
    pub disp_v:    Array2<f32>,
    pub eps_xx: Array2<f32>,
    pub eps_yy: Array2<f32>,
    pub eps_xy: Array2<f32>,
    /// Constitutive-consistency residual magnitude `|sigma_net - C:eps_fd|` (mDEM domains,
    /// where sigma is a direct, independently-learned network output) — the real,
    /// already-trained-against quantity `training_core::step_physics_multi`'s
    /// `constitutive_consistency` term penalizes, now surfaced for display. Exactly `0.0`
    /// (not NaN) inside the domain for ansatzes where stress is analytically derived from
    /// strain (Kirsch's `QuarterSymmAnsatz`) — there is no independent network stress output
    /// to disagree with strain there, so the residual is trivially and correctly zero, not
    /// missing data.
    pub pde_residual: Array2<f32>,
    /// `|dem_energy_per_point|` — the actual signal `AdaptiveGrid`'s residual-driven
    /// refine/coarsen decision uses (see `training_core::probe_interior_energy_residuals`),
    /// evaluated on the visualization grid instead of at collocation points. This is "what
    /// AMR is looking at", not a separate invented indicator.
    pub amr_score: Array2<f32>,
    /// Collocation point count per grid cell — a genuine 2D histogram of the domain's
    /// current collocation set, binned into this same `(ny, nx)` grid. Raw counts, not
    /// normalized; the UI clips/scales for display the same way it already does for every
    /// other field.
    pub collocation_density: Array2<f32>,
}

/// Per-domain visualization fields for the pin-in-lug 2-domain problem.
pub struct PinLugVisFields {
    pub pin: VisFields,
    pub lug: VisFields,
}

/// Which `BoundaryValueProblem` a `SolverConfig`/GUI session is driving. Single source of
/// truth shared by `pinn-app`'s CLI parsing and `pinn-gui`'s problem selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProblemKind {
    #[default]
    Kirsch,
    PinLug,
}

/// Which execution strategy the training loop should use for host-side work (resampling
/// today; a future CPU-parallel/GPU-dispatch executor later — see `pinn_solver::execution`).
/// Phase 1 of the hardware-adaptive-execution epic: `Serial` is what every code path already
/// does, and `Auto` currently resolves to `Serial` unconditionally
/// (`pinn_solver::execution::ExecutionPlanner` is a deliberate stub until a real workload-aware
/// decision is warranted by profiling data — see that module's own doc comment). No
/// `CpuParallel`/`Gpu` variants yet: adding them with nothing behind them would invite dead-code
/// noise and a false impression of capability that doesn't exist yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Auto,
    Serial,
}

/// Coarse hardware/resource-usage target (Eco/Balanced/Performance/Maximum), independent of
/// `ExecutionMode` (mode picks *how* work executes; profile is meant to eventually cap *how
/// much* — batch size, thread count). Phase 1 only accepts, validates, and threads this value
/// through `SolverConfig`/`pinn.env`; nothing reads it yet to change behavior (see
/// `pinn_solver::execution`'s module doc for why: profiling-driven optimization, not a guess,
/// decides what `Eco`/`Performance`/`Maximum` should each concretely do). Never alters the
/// mathematical formulation being solved — only ever execution strategy, once wired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PerformanceProfile {
    Eco,
    #[default]
    Balanced,
    Performance,
    Maximum,
}

/// `ExecutionMode` + `PerformanceProfile` bundled onto `SolverConfig`, following the same
/// opt-in-subsystem-config shape as `DecisionMakerConfig`/`StiffnessConfig`/`WidthGrowthConfig`.
/// Phase 1: read from `pinn.env`'s `EXEC_MODE`/`EXEC_PROFILE` keys (`pinn-app/src/main.rs`'s
/// `apply_env`), round-tripped and validated, not yet load-bearing on any executed code path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ExecutionConfig {
    pub mode: ExecutionMode,
    pub profile: PerformanceProfile,
}

/// Per-step profiling instrumentation (hardware-adaptive-execution epic, Phase 2). Opt-in,
/// disabled by default - when `enabled = false`, `training_core::step_physics` performs zero
/// extra `Instant::now()`/device-sync calls (see `pinn_solver::diagnostics`'s module doc for
/// why a device sync is unavoidable for honest GPU timing, and why it's therefore gated
/// behind this flag rather than always-on). Wall time only in Phase 2 - memory/numerical
/// diagnostics are a later phase's scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DiagnosticsConfig {
    pub enabled: bool,
}

/// Configuration for the meta-optimizer decision maker (opt-in, disabled by default).
///
/// When `enabled = false`, the existing training loop runs unchanged (SOAP-Muon for
/// weights, AdamW for biases, for the entire run). Set `enabled = true` to activate
/// the three-tier optimizer state machine.
///
/// Architecture invariant: K_t is a post-hoc verification metric. It is **never** used
/// as a transition trigger here — all gates are pure gradient-signal metrics so the
/// decision maker works identically on problems with no closed-form analytical solution.
#[derive(Clone, Debug)]
pub struct DecisionMakerConfig {
    /// Enable the three-tier optimizer state machine (default: false — opt-in).
    pub enabled: bool,
    /// Steps between gradient conflict evaluations (default: 50).
    pub check_interval: usize,
    /// Cosine similarity below which Explore → Align (default: 0.60).
    pub conflict_threshold: f32,
    /// Cosine similarity above which Align → Explore (hysteresis, default: 0.25).
    pub alignment_threshold: f32,
    /// Minimum cosine similarity to enter Converge tier (default: 0.75).
    ///
    /// Kirsch-derived; reused verbatim for pin-lug (`PinnDecisionMaker::new`'s `allow_converge`
    /// arm, see CLAUDE.md's Multi-domain Converge-tier L-BFGS section) with no problem-specific
    /// derivation (issue #42, untuned). Pin-lug's Signorini KKT complementarity terms
    /// (`interface_penetration`/`interface_non_tension`) have discontinuous curvature at the
    /// active-set boundary — a structurally different gradient-conflict regime from Kirsch's
    /// smooth energy landscape — so this threshold may gate Converge entry too early or too
    /// late for pin-lug specifically. Retuning requires a real multi-thousand-step pin-lug run
    /// (tracked in issue #42), not a code-only change.
    pub converge_cosine_min: f32,
    /// g_total_norm (g_pde_norm + g_bc_norm) threshold for Converge entry (default: 5.0).
    /// Dimensionless, relative to O(1)-normalized losses on a 25% collocation subset.
    ///
    /// Same Kirsch-derived-but-untuned-for-pin-lug caveat as `converge_cosine_min` above
    /// (issue #42) — shared verbatim across both problems' `DecisionMakerConfig`.
    pub converge_grad_threshold: f32,
    /// Use exact dual-pass cosine similarity; if false, use cheap proxy ratio (default: true).
    /// Converge tier (L-BFGS) is only entered when this is true.
    pub use_exact_cosine: bool,
    /// Minimum steps to spend in any tier before allowing a transition (default: 50).
    pub min_dwell_steps: usize,
    /// L-BFGS max inner iterations per outer step (default: 5).
    pub lbfgs_max_iter: usize,
}

impl Default for DecisionMakerConfig {
    fn default() -> Self {
        Self {
            enabled:                 false,
            check_interval:          50,
            conflict_threshold:      0.60,
            alignment_threshold:     0.25,
            converge_cosine_min:     0.75,
            converge_grad_threshold: 5.0,
            use_exact_cosine:        true,
            min_dwell_steps:         50,
            lbfgs_max_iter:          5,
        }
    }
}

/// Configuration for the stiffness-coupled SAW-BRDR / PirateNet-gate accelerator
/// (opt-in, disabled by default). When `enabled = false`, no extra gradient-conflict
/// computation is scheduled by this subsystem and `step_physics()` receives
/// `physics_boost = 1.0`, `alpha_lr_mult = 1.0` (both no-ops).
///
/// Architecture invariant: like [`DecisionMakerConfig`], this is driven purely by the
/// real-time gradient-conflict cosine-similarity metric — K_t is never read here.
#[derive(Clone, Debug)]
pub struct StiffnessConfig {
    /// Enable the stiffness controller (default: false — opt-in).
    pub enabled: bool,
    /// Steps between gradient-conflict evaluations (default: 50).
    pub check_interval: usize,
    /// EMA smoothing factor for the held stiffness value (default: 0.7).
    pub ema_beta: f32,
    /// Gain for the SAW-BRDR physics-loss boost; boost = `1 + gain * stiffness`,
    /// hard-clamped to `[1, 4]` regardless of this value (default: 1.0).
    pub physics_boost_gain: f32,
    /// Gain for the PirateNet gate-LR multiplier; mult = `1 + gain * stiffness`,
    /// hard-clamped to `[1, 5]` regardless of this value (default: 2.0).
    pub alpha_accel_gain: f32,
    /// Gate magnitude above which a PirateNet block is considered "awake" and its
    /// weights are included in the SOAP-Muon optimizer step (default: 1e-4).
    pub gate_awake_epsilon: f32,
}

impl Default for StiffnessConfig {
    fn default() -> Self {
        Self {
            enabled:            false,
            check_interval:     50,
            ema_beta:           0.7,
            physics_boost_gain: 1.0,
            alpha_accel_gain:   2.0,
            gate_awake_epsilon: 1e-4,
        }
    }
}

/// Configuration for one-shot, fixed-step-count, function-preserving network width growth
/// (Net2WiderNet-style — see `pinn_solver::network::ElasticityNet::grow_width`), opt-in and
/// disabled by default. Unlike `DecisionMakerConfig`/`StiffnessConfig`, growth is triggered by
/// a plain step-count comparison (`step == trigger_step`), not a plateau/gradient-conflict
/// signal — a deliberate v1 scope cut (see the issue #50 design doc), not an oversight.
///
/// v1 scope: only read by the Kirsch headless path (`pinn_solver::headless::run_headless`) —
/// `run_headless_pinlug_inner`, `runner.rs`'s GUI-driving paths, and pin-lug entirely do not
/// read this field yet, mirroring `use_piratenet_compute_skip`'s existing GUI-absence
/// precedent. Present on both `default_kirsch()`/`default_pinlug()` regardless (matching this
/// struct's existing flag-uniformity convention) so `SolverConfig` stays a single shared shape
/// across both problems even though only one problem's training loop currently acts on it.
#[derive(Clone, Default)]
pub struct WidthGrowthConfig {
    /// Enable the one-shot width-growth event (default: false — opt-in).
    pub enabled: bool,
    /// The training step at which growth fires (compared with plain integer equality against
    /// the training loop's own step counter — zero GPU sync). Default: 0.
    pub trigger_step: usize,
    /// `hidden_dim` to grow to. Must be strictly greater than `SolverConfig::hidden_dim` when
    /// `enabled = true` (`ElasticityNet::grow_width` panics otherwise). Default: 0.
    pub target_hidden_dim: usize,
}

/// Complete solver configuration (passed when spawning the solver thread)
#[derive(Clone)]
pub struct SolverConfig {
    pub material: MaterialProps,
    pub geometry: GeometryConfig,
    pub load:     LoadConfig,
    pub n_interior: usize,
    pub n_boundary: usize,
    pub max_steps:  usize,
    pub vis_grid:   [usize; 2],   // [Nx, Ny]
    pub hidden_dim: usize,
    pub n_hidden:   usize,
    /// FD step size in normalized coordinates [−1,1]²
    pub fd_h: f32,
    /// If true (default), 2D weight matrices are trained with the SOAP-Muon hybrid
    /// optimizer. If false, falls back to plain AdamW for all parameters — kept as an
    /// escape hatch in case the hybrid proves unstable on a given configuration.
    pub use_soap_muon: bool,
    /// Meta-optimizer decision maker configuration (disabled by default).
    pub decision_maker: DecisionMakerConfig,
    /// Opt-in PirateNet adaptive-residual gating (disabled by default). See
    /// [`ElasticityNetConfig::use_piratenet`] in `pinn-solver`.
    pub use_piratenet: bool,
    /// Stiffness-coupled SAW-BRDR / gate-LR accelerator configuration (disabled by
    /// default).
    pub stiffness: StiffnessConfig,
    /// If true, `training_core::compute_reference_scales` (and `PinLugProblem::new`'s
    /// internal equivalent) normalizes stress by `config.material.ultimate_strength_pa`
    /// instead of the applied load (`config.load.px`). Disabled by default — the applied-
    /// load normalization is what the K_t=3.0 Kirsch validation and pin-lug's tuned
    /// SAW-BRDR/LR/ConvergenceTracker thresholds were established against; switching the
    /// stress reference changes every loss term's O(1) magnitude by roughly
    /// `(Px/ultimate_strength_pa)^2` and must be an explicit, informed choice.
    pub use_ultimate_strength_scaling: bool,
    /// Skip forward/backward compute (not just SOAP-Muon preconditioning) for PirateNet
    /// hidden blocks whose gate magnitude is below `stiffness.gate_awake_epsilon`. No-op
    /// when `use_piratenet=false` (gates are empty). Default false — opt-in, matching
    /// `use_ultimate_strength_scaling`'s convention: a numerically-provable-lossless
    /// optimization (see network.rs's `dormant_block_gradient_is_exactly_zero`) that still
    /// ships behind a kill-switch because it changes autodiff-graph structure per step.
    pub use_piratenet_compute_skip: bool,
    /// One-shot function-preserving network width growth (issue #50). Disabled by default —
    /// see [`WidthGrowthConfig`]'s doc comment for scope (Kirsch headless only in v1).
    pub width_growth: WidthGrowthConfig,
    /// Execution-mode/performance-profile selection (hardware-adaptive-execution epic, Phase
    /// 1). See [`ExecutionConfig`]'s doc comment — accepted/validated, not yet load-bearing.
    pub execution: ExecutionConfig,
    /// Per-step profiling instrumentation (hardware-adaptive-execution epic, Phase 2). See
    /// [`DiagnosticsConfig`]'s doc comment — disabled by default, zero cost when off.
    pub diagnostics: DiagnosticsConfig,
}

impl SolverConfig {
    pub fn default_kirsch() -> Self {
        Self {
            material:   MaterialProps::al7075_t6(),
            geometry:   GeometryConfig::kirsch_plate_inches(),
            load:       LoadConfig::default_10ksi(),
            n_interior: 4096,
            n_boundary: 1024,
            max_steps:  28000,
            vis_grid:   [64, 64],
            hidden_dim: 128,
            n_hidden:   5,
            fd_h:       1e-3,
            use_soap_muon:  true,
            decision_maker: DecisionMakerConfig::default(),
            use_piratenet:  false,
            stiffness:      StiffnessConfig::default(),
            use_ultimate_strength_scaling: false,
            use_piratenet_compute_skip: false,
            width_growth: WidthGrowthConfig::default(),
            execution: ExecutionConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
        }
    }

    /// Pin-in-lug contact problem defaults. This single-domain `SolverConfig` shape can't
    /// carry two domains' geometry/material — it's populated here with the LUG domain's
    /// values (the driven/output-of-interest domain) purely so CLI/env plumbing that reads
    /// `config.geometry`/`config.material`/`config.load` for display (see
    /// `pinn-app/src/main.rs`, `headless.rs`'s startup banner) has *something* sensible to
    /// show; the actual two-domain geometry/material/load setup used for training lives in
    /// `pinn_solver::pinlug_problem::PinLugProblem::new`, which is the single source of
    /// truth for both domains.
    ///
    /// Force→traction conversion for the driving load (see `PinLugProblem::new`'s doc
    /// comment for the full derivation): `load.px` here is set to the SAME equivalent
    /// traction magnitude used for the pin's driving boundary condition, expressed as a
    /// far-field-style stress purely for display consistency with `default_kirsch()`.
    pub fn default_pinlug() -> Self {
        use crate::units::{IN_TO_M, LBF_TO_N};
        let pin_radius = 0.5 * IN_TO_M;
        let thickness = 0.4 * IN_TO_M;
        // P = 20,000 lbf total axial force / (projected diametral contact area = 2*r*t).
        // See PinLugProblem::new doc comment for why diametral projection is the right
        // denominator (Hertzian/pin-bearing convention: the resultant force is reacted by
        // the pressure distribution's projection onto the loading axis, whose max extent is
        // the pin diameter times thickness).
        let total_force_lbf = 20_000.0;
        let total_force_n = total_force_lbf * LBF_TO_N;
        let projected_area_m2 = 2.0 * pin_radius * thickness;
        let equivalent_traction_pa = total_force_n / projected_area_m2;
        Self {
            material:   MaterialProps::steel_4340(),
            geometry:   GeometryConfig::pinlug_lug_inches(),
            load:       LoadConfig::uniaxial_x(equivalent_traction_pa),
            n_interior: 2048,
            n_boundary: 512,
            max_steps:  20000,
            vis_grid:   [64, 64],
            hidden_dim: 128,
            n_hidden:   5,
            fd_h:       1e-3,
            use_soap_muon:  true,
            decision_maker: DecisionMakerConfig::default(),
            use_piratenet:  false,
            stiffness:      StiffnessConfig::default(),
            use_ultimate_strength_scaling: false,
            use_piratenet_compute_skip: false,
            width_growth: WidthGrowthConfig::default(),
            execution: ExecutionConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kirsch_has_ultimate_strength_scaling_disabled() {
        assert!(!SolverConfig::default_kirsch().use_ultimate_strength_scaling);
    }

    #[test]
    fn default_pinlug_has_ultimate_strength_scaling_disabled() {
        assert!(!SolverConfig::default_pinlug().use_ultimate_strength_scaling);
    }

    #[test]
    fn solver_config_use_piratenet_compute_skip_defaults_to_false() {
        assert!(!SolverConfig::default_kirsch().use_piratenet_compute_skip);
    }
}
