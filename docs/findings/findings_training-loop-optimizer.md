# Multi-domain Converge-tier L-BFGS, SOAP-Muon optimizer, tensor backend, convergence cascade

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. How `step_physics`/`step_physics_multi` drive training, the SOAP-Muon hybrid optimizer, the `BInner`/`Wgpu`/`NdArray` backend alias, and `ConvergenceTracker`'s warm-restart cascade.

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

## Tensor backend

`training_core::BInner` is the canonical single-source-of-truth backend alias (`B =
Autodiff<BInner>`, `BDevice = <BInner as BackendTypes>::Device`) — every other module
(`headless`/`runner`/`pinlug_problem`/`kirsch_problem`/`contact_export`) imports `B`/`BDevice`
from `training_core` rather than redeclaring its own. The default build is `BInner =
burn::backend::Wgpu`, byte-identical to before this alias existed. The opt-in Cargo feature
`ndarray-backend` on `pinn-solver` (passed through by `pinn-app`'s own `ndarray-backend`
feature, `["pinn-solver/ndarray-backend"]`) swaps `BInner` to `burn::backend::NdArray` at
compile time — a CPU-only path useful on machines without a working Wgpu device. `network.rs`/
`energy.rs`/`soap_muon.rs` each keep their own `TB` test-oracle alias hardcoded to `Wgpu`
regardless of this feature, by design — they're self-contained test fixtures, not part of the
training-loop backend selection this alias controls.

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

