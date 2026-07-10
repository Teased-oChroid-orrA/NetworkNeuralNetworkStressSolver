# PINN Structural Stress Solver

A physics-informed neural network (PINN) solver for structural boundary-value problems, built
on `burn` (ML framework) + `egui`/`wgpu` (GUI). Ships two problems: **Kirsch** (a plate with a
circular hole under remote tension; validation target K_t = 3.0 at the hole boundary) and
**pin-in-lug** (a two-domain pin/lug contact-mechanics problem with a Signorini contact
interface; `--problem pinlug` headless, or select "Pin-in-Lug" in the GUI's problem-kind radio).

Workspace crates: `pinn-core` (geometry/material/sampling, no ML deps), `pinn-solver`
(training loop, optimizer, losses), `pinn-gui` (egui panels), `pinn-app` (binary; `--headless`
for terminal-only training, no GUI).

## Pluggable boundary-value problems

`pinn_solver::problem::BoundaryValueProblem` is the trait a new problem implements to reuse the
generic training/optimizer/convergence machinery without touching it: `domains()` (one
`DomainSpec` per physical domain — Kirsch has 1, pin-lug has 2), `sampling_strategy()`/
`ansatz()` per domain, `loss_terms()` (a `Vec<Box<dyn LossTerm>>` — the stable order is the
SAW-BRDR component vector), `convergence_metric()` (a plain `Option<f64>`; Kirsch returns K_t,
pin-lug returns interface-gap RMS — `ConvergenceTracker` neither knows nor cares which). A
`LossTerm` declares which domain(s) and named point-set(s) it needs (`domains()`/
`point_sets()`); cross-domain terms (e.g. pin-lug's Signorini penetration/non-tension
penalties, `pinn_solver::signorini`) declare more than one domain and receive both domains'
forward-pass outputs in `compute()`. `compute()`'s returned tensor must stay connected to
`inputs`'s live autodiff graph end-to-end — the Signorini terms once detached by reading
`raw_out` to a host `Vec`, running the penalty math in plain `f64`, and rebuilding a fresh leaf
via `Tensor::from_data`, which silently supplied zero gradient despite the term's scalar value
looking correct in logging/SAW-BRDR bookkeeping (the bug `compute()`'s own doc comment now warns
against). `pinn_solver::signorini`'s pure `f64` functions remain as a CPU-math oracle for tests,
not production call sites.

**Two step-driver functions, not one, by design.** `training_core::step_physics` is the frozen,
byte-proven single-domain path Kirsch runs through — it predates the trait, is regression-tested
bit-for-bit against the pre-trait hardcoded implementation, and must never be refactored into a
thin wrapper around the newer path (that would remove the independent code path the regression
test relies on to catch a future mistake). `training_core::step_physics_multi` is the additive,
N-domain generalization pin-in-lug uses; a `step_physics_multi_single_domain_matches_
step_physics_kirsch` test proves the two agree for N=1. Both preserve the same invariant: every
loss term is summed into one SAW-BRDR-weighted scalar, `.backward()` is called exactly once, and
gradients are split back to each domain's own optimizer via `GradientsParams::from_params` keyed
by that domain's own `ParamId`s (globally unique per `burn` `Param`, so domain iteration order
during the split is irrelevant — see `gradient_split_attributes_domain_b_step_only_to_domain_b_
params` for the adversarial proof this can't cross-contaminate).

Every `LossTerm`/reference-scale field must be normalized to O(1) before SAW-BRDR weighting
(divide by a computed `ref_energy`/`ref_stress2`/similar physical scale) — a term left at raw
Pa/Pa²/m² magnitude will dominate the SAW-BRDR total by many orders of magnitude and silently
starve every correctly-normalized term of gradient signal. `pinn.env`'s generic
`MATERIAL_*`/`LOAD_*`/`GEOM_*` overrides are Kirsch-tuned and must not be applied to problems
whose material/geometry/load are fixed as part of the problem definition (see `apply_env`'s
`skip_problem_specific` parameter in `pinn-app/src/main.rs`).

## Units

Internal storage is always SI: stress/modulus in **Pa**, length in **m**. The UI and console
output display **US Customary** units instead — psi/ksi/Msi for stress and modulus, inches
for length. `pinn_core::units` (`IN_TO_M`, `PSI_TO_PA`, `KSI_TO_PA`, `MSI_TO_PA`) is the single
conversion source of truth; every display-layer conversion should go through it rather than
hand-rolled literals. `Px`/`Py` (`pinn_core::loading::LoadConfig`) are far-field *stress*
(traction) boundary conditions — not forces — which is why they're in Pa/ksi like any other
stress quantity, not N/lbf.

## Reference-scale normalization

Every loss term is normalized to O(1) before SAW-BRDR weighting (see `compute_reference_scales`
in `pinn-solver::training_core`) by dividing by a physical reference stress `P` squared
(`ref_stress2 = P²`) and its energy-scale derivative (`ref_energy = 0.5*P²/E`); the length
reference is always `config.geometry.half_w` — that domain's own `GeometryConfig`, never a
shared/hardcoded constant, since a different problem's domain has a different characteristic
length. By default `P` is the applied far-field load (`config.load.px` for Kirsch,
`equivalent_traction_pa` for pin-lug) — the convention this codebase's K_t=3.0 validation and
pin-lug's tuned SAW-BRDR/LR/`ConvergenceTracker` thresholds were established against.

Setting `SolverConfig::use_ultimate_strength_scaling = true` (default `false`, opt-in) switches
`P` to `config.material.ultimate_strength_pa` (`PinLugProblem` takes the equivalent
`PinLugScalingMode::UltimateStrength` instead, since it has no `SolverConfig` of its own) — the
material's ultimate tensile strength (`MaterialProps::al7075_t6()`: 83,000 psi, ASTM B209
minimum spec; `MaterialProps::steel_4340()`: 200,000 psi, a representative quenched-and-tempered
condition — actual 4340 UTS spans 125,000–287,000 psi by temper, unlike `e`/`nu` which are
treatment-invariant). This makes the reference stress **itself** normalize to exactly 1.0 by
construction (it's the denominator: `P²/P² = 1`) — it is **not** a claim that the solved stress
field will reach or approach the material's ultimate strength. Changing this flag changes every
normalized loss term's magnitude by roughly `(P_load/F_c)²` and has not been validated against
the existing training hyperparameters — treat it as an experimental alternate scaling, not a
drop-in improvement.

`MaterialProps::dimensionless_modulus(f_c) -> f64` (`E/f_c`) is a pure diagnostic helper only —
it is never wired into `energy.rs`'s constitutive law (`compute_stress`/`compute_strains`).
Nondimensionalization lives only at the reference-scale/loss-normalization layer described
above; the constitutive law itself always operates on physical SI values.

## Multi-domain Converge-tier L-BFGS

`pinn_solver::decision_maker::PinnDecisionMaker` (Explore/Align/Converge, gated by
`SolverConfig::decision_maker.enabled`, default `false`) is opt-in for pin-in-lug just as it is
for Kirsch — wired into `run_headless_pinlug` behind the same flag, byte-identical to the
pre-wiring trajectory when disabled (`run_headless_pinlug_with_decision_maker_disabled_
matches_pre_change_trajectory`). Kirsch's `Align → Converge` transition is gated on
`phase2_active` (Kirsch has a BC-only Phase 1 that must finish first); pin-in-lug has no such
curriculum split, so `PinnDecisionMaker::new`'s third parameter, `allow_converge`, unlocks an
additional `(Align, phase2_active=false)` → `Converge` arm (and Converge's real exit logic,
rather than an unconditional demote) specifically for it — every Kirsch call site passes
`allow_converge=false`, leaving its own six-arm transition table untouched. **Only
`run_headless_pinlug_inner` has this wired in** — `runner.rs::run_training_pinlug` (the
GUI-driving path) does not yet construct a `PinnDecisionMaker` or `ConvergenceTracker` at all,
a deliberate, tracked scope cut (see the pin-lug GUI cascade follow-up issue), not an oversight.
Converge-entry code on both problems must snapshot L-BFGS's frozen loss weights from the live,
SAW/cap-adapted `StepOutput.lam_by_name` of the immediately preceding step — never from
`problem.base_weight()`'s static seed, which would silently discard both SAW adaptation and any
cap cascade the instant Converge is entered. `training_core::TwoDomainModels<B>`
(`#[derive(Module, Debug)]`
on a named 2-field struct, `pin`/`lug`) is the wrapper that makes pin-lug's Converge-tier L-BFGS
step possible: `burn` 0.21's `LBFGS::step` requires a single `AutodiffModule<B>`, and no blanket
`Module` impl exists for tuples in burn-core — a bare `(ElasticityNet<B>, ElasticityNet<B>)`
does not satisfy that bound, so a named wrapper (not a tuple) is required, not merely preferred.
`step_lbfgs_multi`/`compute_gradient_conflict_multi` are the N-domain generalizations of the
single-model `step_lbfgs`/`compute_gradient_conflict`, reusing `ctx.problem.loss_terms()` (never
a second hand-rolled loss assembly) and partitioning terms by the new `LossTerm::conflict_group()`
(`Physics` — interior energy/equilibrium; `Bc` — every boundary or interface condition,
including Signorini contact terms, which constrain state *at* a boundary rather than *throughout*
a domain's interior, the same classification logic that puts Kirsch's boundary/Neumann terms in
the `Bc` group). `FrozenMultiStepCtx` is the multi-domain analogue of `LbfgsCtxScalars` — frozen
fresh on every Converge-tier *entry* (not just the first) and cleared on exit, since pin-lug
resamples every step unconditionally, unlike Kirsch's AMR-gated resampling.

## GUI

`pinn_core::messages::ProblemKind` (`Kirsch`/`PinLug`) is the single shared selector — `pinn-app`'s
CLI parsing and `pinn-gui`'s problem-kind radio both read/write the same enum, not independently
drifting copies. `pinn_solver::run_training` (Kirsch) and `run_training_pinlug` (pin-lug) are two
separate GUI-driving functions, not one branching function, mirroring the `step_physics`/
`step_physics_multi` precedent: forcing two structurally different training loops through a
shared abstraction increases regression risk on the proven Kirsch path for no benefit. Pin-lug's
two domains are visualized via `PinLugVisFields { pin: VisFields, lug: VisFields }`, sent as a
dedicated `TrainingMsg::PinLugUpdate` variant (not an extension of the existing `Update`/
`VisFields`, which stay exactly as they were — zero regression risk on the Kirsch GUI path).
Pin-lug's contact-pressure CSV export is triggered via `ControlMsg::ExportContactPressure`
(solver-side write, confirmed back to the GUI via `TrainingMsg::ExportComplete`) rather than
sending the trained model over the channel. Pin-lug's `WarmStart` handling is a deliberate scope
cut in this slice: only scalar config fields are honored (no full two-domain resample) — the
GUI disables the warm-start button entirely when `ProblemKind::PinLug` is selected rather than
silently doing a partial warm-start.

## Optimizer

Weight matrices (2D) in `ElasticityNet`'s `Linear` layers are trained with a custom
**SOAP-Muon hybrid** optimizer (`pinn_solver::optim::SoapMuon`); biases (1D) go through plain
`AdamW`. `pinn_solver::optim::WeightOptim` is the runtime-selectable wrapper — set
`SolverConfig::use_soap_muon = false` to fall back to AdamW-only training for every parameter
if the hybrid proves unstable on a given configuration.

The algorithm (ported from `github.com/nikhilvyas/SOAP` and `github.com/nikhilvyas/SOAP_MUON`,
per "Improving SOAP Using Iterative Whitening and Muon", Vyas et al.) is **not** a per-dimension
split between the two optimizers. Each step: SOAP maintains Shampoo-style per-dimension
preconditioners, projects the gradient into their eigenbasis, runs a standard Adam update in
that rotated space, and projects back — then Muon's Newton-Schulz orthogonalization is applied
to the resulting update as a refinement pass. The eigenbasis is refreshed via a full
`nalgebra` eigendecomposition every `precondition_frequency` steps (simplified from the
original's power-iteration+QR approximation, which exists there to amortize cost on
LLM-scale matrices — this network's weights are at most `hidden_dim`×`hidden_dim`, where a
full eigh is microseconds).

`burn`'s native `Muon` optimizer (added in burn 0.21) is not used directly — it's a
self-contained optimizer, not a composable building block — but its Newton-Schulz defaults
(`ns_coefficients`, `ns_steps`) are mirrored for consistency.

## Convergence cascade

`pinn_solver::controllers::ConvergenceTracker` drives warm restarts when a problem's
`convergence_metric()` (a plain `f64` — K_t for Kirsch, interface-gap RMS for pin-lug) plateaus
or crashes. `ConvergenceTracker::new()` (K_t, `MetricMode::KtLegacy`) uses absolute thresholds
tuned to K_t's fixed 0–3.0 scale; `ConvergenceTracker::for_metric(direction, plateau_rel_eps,
crash_spike_factor, significant_floor)` (`MetricMode::Relative`, used by pin-lug) generalizes to
any metric whose absolute scale is problem-configuration-dependent by expressing every
threshold as a dimensionless fraction of the tracker's own recent history, except
`significant_floor` — an absolute noise-floor cutoff the caller derives from the problem's own
physical reference scale (pin-lug: `5.0 * u_ref`). `push`/`check_plateau`/`check_kt_crash`/
`is_kt_converged` keep the same names/signatures across both modes so every pre-existing
K_t-mode test and call site is unaffected; only `check_plateau`/`check_kt_crash`'s bodies branch
on the mode. The plateau-comparison window (`PLATEAU_WINDOW = 20`) is coupled to Kirsch's
training schedule (AMR sweep interval, "retains the pre-AMR@7000 peak") and must not be changed
independently for Kirsch — see the comment at its definition; this rationale does not transfer
to pin-lug (no AMR), which reuses the same window constant for now as an untuned first pass.
Plateau and crash restarts draw from separate 4-restart budgets (`MAX_PLATEAU_RESTARTS`,
`MAX_CRASH_RESTARTS`, independent regardless of mode); both feed the same `lam_h_cap`/
`lam_d_cap`-equivalent decay cascade (50 → 30 → 18 → 15) regardless of which budget fired. For
pin-lug, `MultiStepCtx.phase2_active` is hardcoded `true` (independent of the decision-maker's
own `PHASE2_ACTIVE` constant, which stays `false`) purely to activate `step_physics_multi`'s
pre-existing `lug_free_edge_traction`/`lug_shank_anchor` cap-dispatch arms — the two booleans
are unrelated axes that happen to share a name.
