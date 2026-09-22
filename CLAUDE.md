# PINN Structural Stress Solver

## Response style

Use installed `caveman` skill automatically for every user-facing response in this repository. Default to full mode for the whole session; no activation command required. Follow its auto-clarity and boundary rules. `stop caveman` or `normal mode` disables it immediately.

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

## Hardware-adaptive execution (Phase 1-4)

`pinn_solver::execution` (`Executor` trait, `SerialExecutor`, `ExecutionPlanner`) and
`pinn_core::messages::{ExecutionMode, ExecutionConfig, PerformanceProfile}` are Phase 1 of a
multi-phase hardware-adaptive-execution effort (see the "Epic Addendum" this reconciles against
the actual codebase before implementing). **Deliberately narrow scope — do not expand without
re-reading this section:**

- `Executor` wraps ONLY the free-function resample calls
  (`pinn_core::sampling::sample_interior`/`sample_boundary`) that Kirsch's pre-trait, frozen
  `runner.rs`/`headless.rs` call sites use directly. It does **not** wrap
  `pinn_core::problem::DomainSamplingStrategy` (the existing per-*domain* sampling abstraction
  `KirschSamplingStrategy`/`PinLugSamplingStrategy` implement — an orthogonal axis, per-domain
  behavior vs. per-hardware execution strategy) — pin-lug's every-step resample is untouched.
  It does **not** wrap `training_core::step_physics`/`step_physics_multi` — those are single
  batched `burn` tensor graphs with nothing embarrassingly parallel inside them, and the
  step_physics-must-not-become-a-thin-wrapper warning earlier in this file applies with full
  force to any temptation to route it through an `Executor` too.
- No `TensorBackend` trait exists or should be added — `burn::tensor::backend::Backend`
  (already threaded through everything via `training_core::B`/`BInner`) already is that layer.
- `ExecutionPlanner::plan` is a deliberate stub: it always returns `SerialExecutor` regardless
  of `ExecutionMode`/`PerformanceProfile`. `EXEC_MODE`/`EXEC_PROFILE` (`pinn.env`) are accepted,
  validated, and threaded through `SolverConfig`, but change no executed code path yet — do not
  assume setting `EXEC_PROFILE=eco` does anything at runtime until a later phase says otherwise.
- Rayon is intentionally NOT a dependency of this crate's own code (criterion's dev-dependency
  pulls it in transitively for benchmarking only). The one real per-step CPU-bound host loop,
  `pinn_core::sampling::sample_interior`, is **RNG-order-sensitive** (seeded LCG; a near-hole-
  guarantee ring is explicitly prepended before truncation) — naively `par_iter`-ing it would
  silently change point streams. Parallelizing it requires deterministic RNG-stream
  partitioning as its own reviewed change, informed by real profiling data, not assumed.

`crates/pinn-solver/benches/training_step.rs` (criterion, `cargo bench -p pinn-solver`)
benchmarks `step_physics` at 5 tiers (tiny/small/medium/large/very_large; `medium` matches
`SolverConfig::default_kirsch()`'s real shipped config exactly). It constructs its own
`StepCtx` using already-`pub` production setup functions (`EngineParams::analyze`,
`build_gathered_boundary_tensors`, etc. — the same ones `headless::run_headless` calls) rather
than touching `training_core.rs`'s private test-fixture helpers, which are `#[cfg(test)]`-gated
and invisible to an external bench crate. **First real measurement already surfaced something
worth knowing**: a short `cargo bench -- --sample-size 10` run showed near-identical wall time
(~550-900ms) across all 5 tiers on the default Wgpu backend, despite a 64x difference in
`n_interior` and 16x in `hidden_dim` between `tiny` and `very_large`. This is consistent with
Wgpu/cubecl-fusion kernel-compile/dispatch overhead dominating at these problem sizes on this
hardware, not the actual tensor compute — exactly the kind of profiling evidence the epic's own
"tiny tensor operation → avoid GPU transfer" guidance is about. Don't assume this bench shows
clean O(n) scaling without re-running it with a real sample size and warm-up first; a longer,
statistically-sound run (not the quick `--sample-size 10` smoke check) is what any future
Rayon/backend-comparison decision should be based on.

### Phase 2: per-step profiling instrumentation

`pinn_solver::diagnostics` (`StepTimer`, `StepTiming`) and `pinn_core::messages::
DiagnosticsConfig` (`SolverConfig::diagnostics`, `pinn.env`'s `DIAGNOSTICS_ENABLED`, default
`false`) instrument `training_core::step_physics` with three wall-clock buckets: forward pass +
SAW-BRDR loss assembly, the single `.backward()` call, and gradient-extraction + all three
optimizer `.step()` calls. Populated into the new `StepOutput.timing: Option<StepTiming>` field
(every other `StepOutput` construction site — `step_physics_multi`, both GUI/headless
Converge-tier synthetic outputs, the `old_hardcoded_step_physics` test oracle — sets this to
`None`; **only `step_physics` itself is instrumented in this pass**, a deliberate smaller slice
rather than doing `step_physics_multi` in the same edit).

**A device sync is required for honest numbers, and that's a real, disclosed, opt-in cost.**
`burn`'s tensor ops on a GPU backend are queued, not executed synchronously — timing without a
sync at each boundary would measure "time to enqueue," not "time to actually compute" (this is
exactly what Phase 1's bench's suspiciously-flat tiny-vs-very_large timing turned out to be
evidence of). `training_core::sync_device` wraps `<B as Backend>::sync(device)`
("ensure all computation are finished," `burn-backend`'s own doc comment) and is called at
every `StepTimer` checkpoint — but ONLY when `diagnostics.enabled == true`. When disabled (the
default), `step_physics` performs zero extra `Instant::now()` calls and zero extra syncs -
genuinely zero-cost, not just cheap. **Regression-safety was the primary risk here** (this
inserts new code into the frozen byte-exact `step_physics` path) — proven by
`step_physics_diagnostics_enabled_matches_disabled_and_populates_timing`
(`training_core.rs`), which asserts enabling diagnostics changes `StepOutput.timing` and
nothing else (`total_scalar`/`e_scalar`/`lam_e`/`lam_h` all byte-identical), and by the full
216-test suite passing unchanged both before and after this instrumentation landed.

Real end-to-end confirmation (a 5-step headless Kirsch run, `DIAGNOSTICS_ENABLED=true`):
step 0 showed forward=264ms/backward=256ms, step 4 showed forward=35ms/backward=117ms - the
same one-time GPU kernel-compile-overhead pattern Phase 1's bench finding predicted, now
visible per-step rather than only as a flat aggregate. `headless::run_headless` prints an
indented `[diagnostics] forward=...us backward=...us optimizer=...us` line under each printed
step row when `out.timing` is `Some` - this is the one place the data is actually surfaced to a
human, not just populated silently into a struct nobody reads.

### Phase 3: measured, then deliberately did NOT add Rayon

Phase 3's own plan gated adding Rayon on Phase 2's profiling actually identifying a bottleneck
at pin-lug's every-step resample (the one live candidate — `sample_interior`'s RNG-order-
sensitivity is what makes it risky to parallelize, so it was never going to be worth doing
without real evidence first). **The measurement said no**: `crates/pinn-solver/benches/
resample.rs` (criterion, `cargo bench -p pinn-solver --bench resample`) benchmarks pin-lug's
real per-step resample (`PinLugSamplingStrategy::sample_interior`/`sample_boundary` for both
domains, at `SolverConfig::default_pinlug()`'s real shipped sizes — n_interior=2048,
n_boundary=512) at **~21.7us per step** (release profile, 30-sample criterion run). Compare
against Phase 2's real headless measurement of `step_physics_multi`'s actual tensor-step cost —
tens of milliseconds even in steady state. Resample is roughly **0.05-0.1% of step cost**.

**Rayon was NOT added.** Parallelizing a ~22-microsecond operation would add thread-
spawn/synchronization overhead exceeding the work itself, and would introduce the real,
already-documented RNG-stream-determinism engineering risk (`sample_interior`'s seeded LCG,
near-hole-guarantee-ring prepend-then-truncate ordering) for zero measured benefit. This is
exactly the outcome the epic's own acceptance criteria explicitly calls acceptable ("A
benchmark should be allowed to demonstrate that serial execution is faster... that is an
acceptable and expected outcome") — Phase 3 is complete as a measurement-and-conclusion phase,
not a not-yet-implemented one. **Do not add Rayon here without new evidence**: if a future
change genuinely increases resample cost (much larger `n_interior`, a more expensive rejection
test, a different problem's sampling strategy), re-run `benches/resample.rs` first and let a
new real number — not this note's memory of an old one — justify revisiting this.

No code changes to `pinn_solver::execution`/`ExecutionMode` were needed for this conclusion —
`CpuParallel` remains unimplemented (Phase 1's stub still applies), which is the correct state
given nothing yet justifies building it.

### Phase 4: `PerformanceProfile` gets a real effect (Eco sampling reduction + rayon thread cap)

Two additions to `pinn_solver::execution`, both derived purely from `SolverConfig.execution.
profile` — no new `pinn.env` key needed:

- `apply_performance_profile(&mut SolverConfig)`: under `PerformanceProfile::Eco`, divides
  `n_interior`/`n_boundary` by `ECO_SAMPLING_DIVISOR` (4), floored at `ECO_MIN_INTERIOR`(256)/
  `ECO_MIN_BOUNDARY`(64). Every other profile leaves them untouched — this function only ever
  *reduces*, never increases, sampling density.
- `cpu_thread_count(PerformanceProfile) -> Option<usize>`: `Eco` → `Some(1)`, `Balanced` →
  `None` (don't touch rayon's global pool), `Performance`/`Maximum` → all available cores. The
  caller (`pinn-app::main()`) feeds this into
  `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()`, capping whatever CPU-side
  parallelism `burn-ndarray` uses internally (confirmed via its own `parallel.rs`:
  `rayon::scope`) when `--features ndarray-backend` is active. `rayon` is now a direct
  dependency of `pinn-app` only (not `pinn-solver`) — configuring a dependency's global thread
  pool is an app-entry-point concern, matching Phase 3's "no rayon dependency of pinn_solver's
  own code" precedent.

**Call-site ordering is load-bearing and caused a real bug during verification.** Kirsch's
`headless::run_headless_inner` calls `EngineParams::analyze(&config)` then
`engine.apply_to(&mut config)`, and `apply_to` **unconditionally overwrites**
`config.n_interior`/`n_boundary` from geometry-derived analysis — regardless of what
`pinn.env`/CLI configured them to. The first implementation called `apply_performance_profile`
in `pinn-app::main()`, *before* `run_headless` — so `apply_to` silently clobbered the Eco
reduction every time, and a verification run showed the banner printing values *larger* than
the configured `N_INTERIOR`, not smaller. **Fix**: `apply_performance_profile` is now called
inside each headless entry point, at the point where it actually sticks — right after
`engine.apply_to(&mut config)` in `run_headless_inner` (Kirsch), and at the top of
`run_headless_pinlug_inner` (pin-lug, which has no `EngineParams::analyze`/`apply_to` call at
all, so nothing to run after). `pinn-app::main()` no longer calls it — only the rayon
thread-pool setup remains there, since that's a true one-time process-global action independent
of any one problem's config path. **If a similar profile-application function is added later,
check whether `EngineParams::apply_to` (or an equivalent geometry-driven recompute) runs between
`apply_env` and the point you're relying on the value — `main()` is not the right call site for
anything that touches `n_interior`/`n_boundary` in the Kirsch path.**

Verified end-to-end (`EXEC_PROFILE=eco`, `N_INTERIOR=4096`, `N_BOUNDARY=1024`, `MAX_STEPS=1`,
release build): banner printed `Interior: 2048  Boundary: 512` — exactly 1/4 of `Balanced`'s
real engine-analyzed defaults (8192/2048; note `EngineParams::analyze` derives these from
geometry, not from the configured `N_INTERIOR`/`N_BOUNDARY` values directly — a pre-existing,
Eco-unrelated behavior, confirmed by reproducing the same 8192/2048 base under `EXEC_PROFILE=
balanced`). Full workspace suite: 222 passed, 0 failed, unchanged by this phase.

## `toy_beam`: fast, standalone 1D methodology sanity check

`pinn_solver::toy_beam` (`crates/pinn-solver/src/toy_beam.rs`) exists because the real
Kirsch/pin-lug default config takes ~70-90 minutes per run (`MAX_STEPS=28000` at ~150-200ms/
step, real measured cost — not a bug), too slow to quickly sanity-check the core training
methodology itself: does the network actually minimize the physics residual, or can it
collapse into a trivial/near-zero deflection that happens to satisfy the boundary conditions
cheaply? Solves the 1D Euler-Bernoulli beam `d⁴w/dx⁴ = q` (`E=I=q=1`) for both cantilever
(clamped-free) and simply-supported (pinned-pinned) boundary conditions, both with known
closed-form solutions used as the pass/fail oracle.

**Deliberately decoupled from `BoundaryValueProblem`/`DomainSamplingStrategy`/`SolverConfig`
— do not try to fit a 1D problem through that stack.** Investigated first: `GeometryConfig`/
`LoadConfig`/`BoundaryPoint` are irreducibly "2D plate with optional hole" shaped, and
`--headless` dispatch only routes `ProblemKind::{Kirsch,PinLug}` through concrete
`SolverConfig`s with no generic `Box<dyn BoundaryValueProblem>` entry point. Forcing a scalar
1D field through that stack would mean dead y/hole/traction fields and a new enum arm for a
component explicitly meant to be fast and disposable-feeling, not a third production problem.
Only two things are reused: `ElasticityNetConfig`/`ElasticityNet` (genuinely output-width-
agnostic — `ElasticityNetConfig::new().with_input_dim(1).with_output_dim(1)` works as-is, since
only the *optional* Fourier embedding assumes 3 input columns, and it's skipped when
`n_fourier=0`), and plain `burn::optim::AdamW` directly (not `pinn_solver::optim`'s SOAP-Muon
weight/bias split, which exists for elasticity's own rationale and is irrelevant here).

**Formulation is energy minimization (DEM), not a strong-form residual — this is the key
design choice, not an implementation detail.** A strong-form residual `(d⁴w/dx⁴ - q)²` would
need a novel 4th-derivative stencil; nothing above 1st-derivative (strain) exists anywhere else
in this codebase. The weak (variational) form only needs the 2nd derivative (curvature `w''`),
because integration by parts drops the order by 2 — and critically, **natural boundary
conditions (moment/shear at a free end) are automatically satisfied at the energy minimum and
need no explicit loss term at all**, only essential BCs (displacement/slope) need enforcing.
Essential BCs are hard-enforced via the same scale-factor pattern `pinn_core::problem::
DirichletAnsatz` already establishes: `w(x) = p(x) * NN(x)` for a polynomial `p` chosen so the
essential BCs hold at every `x` (`p(x)=x²` for cantilever, `p(x)=x(1-x)` for simply-supported —
see `BeamBc::essential_factor`), never penalized/approximate. `w''` comes from a 3-point
central FD stencil (`h=1e-2`), mirroring `fd_stencil::assemble_stencil`'s "stack all offset
points into one batched forward pass, then combine" pattern — 1D/3-point here instead of
2D/5-point. Loss = discretized potential energy over a fixed midpoint-rule grid (no RNG
needed): `Π[w] ≈ Σ [0.5·w''(x_i)² − q·w(x_i)] · Δx`.

**Verified end-to-end** (`cargo run -p pinn-solver --example toy_beam --release`, 3000 steps,
`hidden_dim=48`, `n_hidden=3`, 32 collocation points, ~15s/case after the one-time compile):
cantilever converged to max_abs_error=7.6e-4 against max_abs_deflection=0.126 (~0.6% relative);
simply-supported to max_abs_error=5.4e-5 against max_abs_deflection=0.013 (~0.4% relative).
Both far exceed their regression-test tolerances (`toy_beam::tests`) with wide margin —
confirms the training methodology genuinely minimizes the residual rather than collapsing to a
trivial solution. Two entry points: `cargo test -p pinn-solver toy_beam:: -- --ignored` (fast,
silent pass/fail, ~40s in debug per case) and the example above (slower to build once,
human-readable comparison table). No existing file's behavior changed — purely additive.

**The two training-loop tests are `#[ignore]`d, not run by default `cargo test --workspace`.**
Two separate full-workspace runs after adding `toy_beam` each failed a *different*
`training_core` byte-exact oracle test (`compute_gradient_conflict_bc_group_includes_w_neumann_
again`, then `step_physics_trait_driven_matches_independently_reimplemented_old_formula`), each
with a tiny (~0.01-0.1%) relative-error mismatch, each passing cleanly when re-run standalone.
Consistent with this project's already-known Wgpu weight-init non-determinism, tipped into an
occasional failure by `toy_beam`'s two ~15s concurrent GPU-training tests adding contention on
top of `cargo test`'s default parallelism — not a logic bug in `toy_beam` or in the oracle
tests themselves. Run the training-loop tests explicitly (`-- --ignored`) when you want to
verify `toy_beam` itself; don't remove the `#[ignore]` without re-confirming full-workspace
stability across at least two runs first.

## User-defined problem ingestion (`--problem-spec`)

`pinn_core::user_geometry`/`problem_spec` + `pinn_solver::user_problem`/`user_runner` let a
user design a NEW 2D elasticity joint (rectangular plate, N circular holes, each either
`Free` or `Fixed`) from a TOML file, instead of writing new Rust — the missing piece
`--headless` dispatch didn't have (it only ever routed `ProblemKind::{Kirsch,PinLug}`
through concrete `SolverConfig`s). Run: `cargo run -p pinn-app --release -- --headless
--problem-spec <path.toml>` (see `examples/problems/notched_plate.toml` for a documented
template). **Absent this flag, `main()`'s existing Kirsch/pin-lug dispatch runs completely
unchanged** — the flag is checked and `main()` returns before any of that code executes.

Drives the SAME `BoundaryValueProblem`/`LossTerm`/`DomainSamplingStrategy` trait family
Kirsch/pin-lug use, through the already-generic `step_physics_multi` driver
(`training_core.rs`) — this is genuinely its first live production user (Kirsch's own trait
impl is test-fixture-only, per the hardware-adaptive-execution epic's earlier investigation).
No new physics math: every loss term is a near-verbatim copy of an existing Kirsch/pin-lug
term's shape (`InteriorEnergyTerm`→`dem_energy_loss`, `OuterTractionTerm`→`neumann_loss`,
`HoleBcTerm::Free`→`hole_traction_loss_direct`, `HoleBcTerm::Fixed`→`mean(u²+v²)` anchor,
mirroring `LugShankAnchorTerm`). Essential BCs are soft-penalty (`IdentityAnsatz`, reused
directly from `pinlug_problem.rs`), not a hard ansatz — there's no closed-form scale factor
that zeroes displacement on an arbitrary circle at an arbitrary position, the same reason
pin-lug itself doesn't hard-enforce its shank anchor.

**Deliberately separate geometry type, zero changes to any existing file.**
`GeometryConfig` hardcodes exactly one hole and is load-bearing for Kirsch/pin-lug;
`UserGeometry`/`HoleSpec` (N holes) is a fully independent type. `UserSamplingStrategy`
ignores the `&GeometryConfig` parameter every `DomainSamplingStrategy` method takes, using
its own captured `UserGeometry` instead — an established, precedented pattern (see
`FakeInterfaceSampling` in `pinn-core/src/problem.rs`'s own tests). `DomainSpec.geometry`
gets `UserGeometry::to_placeholder()`, a `GeometryConfig` sized to the real bounding box
(`hole: HoleType::None`, `symmetry: Full`) — safe because nothing generic reads a domain's
`GeometryConfig` except bounding-box math (`x_range`/`y_range`), confirmed by tracing
`compute_domain_forwards`; the only `SolverConfig` field that function reads at all is
`config.load.px` (mDEM stress-column physical-unit scaling) — `user_runner` builds its
`SolverConfig` from `SolverConfig::default_kirsch()` with only `.load` overridden.

**v1 scope, deliberately**: single domain (not full N-domain/contact generality — no product
need yet, and pin-lug's 2-domain coupling isn't generalized), headless-only (no GUI wiring),
no curriculum/AMR/decision-maker/SAW-BRDR-tiering (plain constant/scheduled-LR AdamW loop via
`step_physics_multi` directly, matching `toy_beam`'s own "prove the formulation converges
before adding curriculum machinery" discipline). No closed-form convergence oracle exists for
an arbitrary user geometry (unlike Kirsch's K_t) — verification is qualitative: loss should
decrease, and `max|displacement|` should be non-negligible (a real trivial-collapse check,
printed by `user_runner`).

**Verified end-to-end** (`examples/problems/notched_plate.toml` — 0.2×0.1 m plate, one Free
hole, one Fixed hole, ~10 ksi far-field traction, `hidden_dim=64`, 2000 steps): loss decreased
monotonically 5.5 → 0.47, `max|displacement|~9.86e-5 m` — matches the back-of-envelope
estimate `(px/E)·half_w ≈ 9.6e-5 m` almost exactly, a strong physical sanity check that this
is real elasticity, not noise. Confirmed the no-flag default (`--headless`, no
`--problem-spec`) still produces byte-for-byte the same Kirsch banner/output as before this
feature. Full workspace suite: 229 passed, 0 failed, 2 ignored (the pre-existing `toy_beam`
ignores) — purely additive, zero regressions.

## GUI wiring for user-defined problems + redesigned training panel

`pinn-gui` now supports a third, GUI-driven mode for the same `ProblemSpec`/
`UserDefinedProblem` path above: a "User-Defined" radio option in `params.rs` with a spec
path field + "Load" button, streaming live progress via a new `runner::
run_training_user_problem` into the heatmap/training panels.

**Deliberately did NOT add a variant to `pinn_core::messages::ProblemKind`.** Doing so would
ripple required-arm additions through every exhaustive match on it in the frozen Kirsch/
pin-lug paths for no benefit. Instead `pinn-gui`'s own `StressSolverApp` gained separate
fields (`user_defined_active: bool`, `user_spec_path/spec/error`) that override
`problem_kind`-driven dispatch only where needed — every existing `match problem_kind {
Kirsch, PinLug }` site is byte-for-byte unchanged.

**`run_training_user_problem` reuses `TrainingMsg::Update(Box<TrainingUpdate>)` — not a new
message variant.** `TrainingUpdate`'s fields are already generically named (`energy_loss`/
`neumann_loss`, not Kirsch-specific names), and this problem is single-domain like Kirsch,
not two-domain like pin-lug. `energy_loss = out.e_scalar` (this problem's energy term is
literally named `"interior_energy"`, matching `StepOutput::e_scalar`'s lookup for every
problem); `neumann_loss = out.total_scalar - out.e_scalar`, mirroring `run_training_pinlug`'s
own exact convention for aggregating an arbitrary number of differently-named BC terms into
one number without needing this problem's own term names to match any hardcoded accessor.

**`evaluate_user_vis_grid`** (`user_problem.rs`) mirrors `runner.rs`'s private
`evaluate_vis_grid_mdem` (same mDEM direct-column read — no FD stencil needed for
visualization — same von Mises formula), adapted for `UserGeometry`'s N-hole containment
check instead of `GeometryConfig`'s single-hole one.

**Heatmap N-hole overlay**: `heatmap::draw_overlays` gained a `user_holes: Option<&[HoleSpec]>`
parameter — when `Some`, draws each hole as its own full circle (no `QuarterSymm` assumption)
color-coded by `HoleBc::Free` (green) / `Fixed` (red), instead of the single-hole Kirsch/
pin-lug arc. The pixel/texture/colorbar core (`field_to_pixels`, `select_field`) needed zero
changes — already field-agnostic.

**Training panel redesign** (`training.rs`): added a row of 4 stat cards (total loss, energy
term, boundary term, learning rate — `egui::Frame` with a colored left accent bar) above the
existing `egui_plot` curves, restyled to a shared accent palette (teal/blue/amber/violet).
This panel is shared — Kirsch and pin-lug get the same visual upgrade with zero per-mode
branching, since it only ever reads `TrainingState`'s already-generic fields. Palette/layout
were prototyped first as an HTML design-reference artifact (dark "instrument panel" aesthetic)
before writing any egui code — see the artifact link in that session's conversation; egui's
real capability (immediate-mode 2D, no CSS effects) means the in-app result approximates that
reference's palette/layout, not a pixel-perfect port.

Verified: `cargo build --workspace` clean, full suite 229 passed/0 failed/2 ignored (same as
above — purely additive), release binary launches without crashing. Full interactive
click-through (load a spec, Start, confirm the heatmap/stat cards update) was NOT
independently verified end-to-end in this pass — the training math itself was already proven
correct via the headless path's own verification above, and this GUI runner is a thin
streaming wrapper around the identical `UserDefinedProblem`/`step_physics_multi` call, but an
actual mouse-driven session is worth a real check before relying on this heavily.

## Issue #77 PH4-45: GUI/headless wiring for the three PH4-41..44-corrected architectures

PH4-41..44 fixed the Kt-measurement bug and proved three architectures (single-domain
hard-constraint ansatz, sequential two-stage, log-polar embedding) converge to FEM within
0.67-1.23% — but none of that was reachable outside test code, and the GUI heatmap for any
`decomposition_applicable` single-centered-Free-hole spec had the SAME field-reconstruction bug
PH4-41 fixed in the diagnostic probe (`evaluate_user_vis_grid` did a bare forward pass, no
ansatz, no affine background). This closes both gaps.

**`evaluate_user_vis_grid` fix — a real, currently-shipping bug, not just a missing feature.**
Gained `ansatz: &dyn DirichletAnsatz`/`affine_strain_pair: Option<(f64,f64)>` parameters and now
routes through `training_core::stencil_forward_with_ansatz` (the same shared helper the
corrected probe uses), adding the affine background strain/displacement exactly like
`probe_hole_boundary_profile_derived`. Every one of its ~11 call sites was fixed the same way
the probe's were — the compiler enumerates them exhaustively, same discipline as PH4-41.
Proven by two closed-form tests mirroring the probe's own proofs
(`evaluate_user_vis_grid_adds_affine_strain_exactly`,
`evaluate_user_vis_grid_reflects_hard_constraint_ansatz_near_hole_boundary`).

**`ProblemSpec.architecture: ArchitectureSpec`** (`pinn-core/src/problem_spec.rs`) — the
first TOML-reachable selector for the three architectures, every field `#[serde(default)]` to
the exact pre-#77 dispatch: `hard_constraint_ansatz: bool`, `hole_bias_fraction: f64`,
`coordinate_embedding: CoordinateEmbeddingSelection` (`Cartesian`/`LogPolar`),
`training_procedure: TrainingProcedure` (`Joint`/`SingleDomain`/`SequentialTwoStage{stage_a_
steps, stage_b_steps}`). See `examples/problems/issue_77_l5_hard_constraint.toml` — PH4-42's
own exact verified config, now runnable via `--headless --problem-spec`, not just a test.

**`TrainingProcedure::SingleDomain` exists because of a real dispatch ambiguity the first
draft of this wiring got wrong.** Every "L5" shape this whole investigation used (single
centered Free hole, enough margin for `annular_partition()`) qualifies for BOTH the plain
single-domain `UserDefinedProblem` path AND `AnnularDecompositionProblem`'s two-domain Joint
path. `hard_constraint_ansatz=true` alone doesn't disambiguate which model it should apply
to — Phase 1's real PH4-42 result used `UserDefinedProblem::new_with_hard_constraint_ansatz`
(single-domain), a DIFFERENT architecture from Phase 3's `AnnularDecompositionProblem::
new_with_log_polar_embedding` (two-domain), even though both can carry the same ansatz flag.
Without `SingleDomain` forcing the single-domain branch, `run_training_user_problem`/
`run_headless_user_problem`'s pre-existing dispatch order (check `AnnularDecompositionProblem::
supports` first) would silently route Phase 1's own example config through the two-domain
architecture instead — reachable, but not the one PH4-42 verified. `Joint` (default) keeps the
exact pre-#77 dispatch; `SingleDomain` is required whenever a spec wants the genuinely
single-domain architecture on a geometry that also happens to qualify for annular
decomposition.

**Live Kt + a correct spliced heatmap for the annular paths** — previously hardcoded
`vis: None`/`kt_estimate: None` ("no correct two-model field evaluator existed"). Fixed via:
- `run_annular_decomposition_training_inner`'s `diagnostics: &mut Vec<AnnularL5Diagnostic>`
  parameter became `on_diagnostic: &mut dyn FnMut(AnnularL5Diagnostic, &annulus_model,
  &outer_model)` — a sink, not an accumulator, so a live caller gets both the diagnostic AND
  the two models at the exact checkpoint. All 10 existing wrapper functions (`run_annular_
  decomposition_training_with_diagnostics_and_*`) and the bare `run_annular_decomposition_
  training` were mechanically updated to pass `&mut |d, _a, _o| diagnostics.push(d)` — same
  values, same order, same cadence, purely a delivery-mechanism change. Same additive-sink
  pattern applied to `run_annular_decomposition_training_sequential` (new trailing
  `on_diagnostic` parameter; the 3 existing test callers pass `&mut |_d, _a, _o| {}` and keep
  reading the returned `Vec` exactly as before).
- `evaluate_annular_vis_grid` (`user_problem.rs`) — the two-domain analogue of `evaluate_user_
  vis_grid`: evaluates both models over the full grid with their own real training-time
  ansatz, then splices per-cell by physical distance from the hole center vs.
  `interface_radius` (annulus model inside, outer model outside). Proven by
  `evaluate_annular_vis_grid_splices_at_the_interface_radius_not_one_model_everywhere` (two
  models with different seeds, asserting each region matches ONLY its own model's field).
- `runner::run_training_annular_decomposition`/the new `run_training_annular_decomposition_
  sequential` compute this at the same "every 10th step/last step" cadence the single-domain
  GUI path already uses, sharing the result with `on_step` via a `Rc<RefCell<Option<...>>>`
  (safe — diagnostic and `on_step` run sequentially within the same training step, never
  concurrently, so the `RefCell` never double-borrows).

**A second real, pre-existing bug this session's own headless smoke test caught** (not
introduced by this work): `run_headless_user_problem`'s "not a trivial collapse" diagnostic
called `model.forward()` directly on a bare 3-column `[xn,yn,0.0]` tensor, bypassing the
embedding transform entirely — panicked on ANY single-hole geometry (the model is built with
`net_input_dim()`=10 for `SingleHoleChart`, not 3). Never caught before because every real
PH4-24..44 run trained through a different function
(`run_user_problem_training_with_diagnostics`/`run_annular_decomposition_training*`), never
this exact plain-headless-against-a-real-hole path. Fixed by routing through `fwd_embedded`
(the same embedding-aware forward every other real call site already uses); regression-proven
by `run_headless_user_problem_completes_on_a_real_single_hole_geometry_without_panicking`.

**GUI surfacing** (`pinn-gui`): `TrainingState` gained `hole_analyses: Vec<HoleAnalysis>`
(populated from `TrainingUpdate.hole_analyses` in `app.rs`'s existing `Update` handler — no new
message variant) since the plain single-domain path's real Kt lives there
(`kt_estimate` stays `None` on that path; the annular paths populate `kt_estimate` directly,
reusing the exact mechanism Kirsch's own Kt readout already used). `params.rs`'s User-Defined
panel shows: a read-only architecture summary (only when non-default — no interactive
widgets, TOML stays the single source of truth for User-Defined mode, matching every other
field there) and both Kt sources (`kt_estimate` and/or per-hole `hole_analyses`), no hardcoded
"theory: 3.000" comparison (no closed-form Kt exists for an arbitrary user geometry).

**Verified end-to-end via real release-binary headless runs** (not just unit tests) for all
three architectures plus the untouched default: `SingleDomain` hard-constraint (dispatches
correctly to the single-domain path despite the L5 geometry also qualifying for annular
decomposition — confirmed by banner text, absent `hole_free` term, present `rotation_gauge`/
`translation_gauge`), `SequentialTwoStage` (`stage_a`/`stage_b` both print, Kt printed at the
end), `Joint` + `LogPolar` + hard-constraint together (annular dispatch, Kt printed), and a
pre-existing `[architecture]`-free example (byte-identical dispatch/output shape to before).
Full workspace regression after every change in this pass: 537+ passed, 0 regressions (same
1 pre-existing, unrelated failure as PH4-41..44 — `compute_loss_for_lbfgs_panics_on_lams_
missing_a_real_term_key`, confirmed failing on unmodified `d9f38fe` via `git stash`).

**Not done in this pass**: a real mouse-driven GUI click-through (load a spec with
`[architecture]` set, click Start, watch the heatmap/Kt update) — same disclosed gap the
original GUI-wiring section above already flagged for its own work, for the same reason
(headless verification of the identical underlying training/diagnostic code already
establishes correctness; the GUI runner is a thin streaming wrapper around it). Worth a real
check before relying on this heavily for a live demo.

## Issue #64/#66/#73: sampling-resampling regression, its GUI-path twin, and the fix

`UserSamplingStrategy::sample_interior`/`sample_boundary` (`user_problem.rs`) draw a genuinely
different jittered-stratified point set on every call, seeded from a per-instance `AtomicU64`
call counter (`interior_calls`/`boundary_calls`) — **not** the fixed-seed pure functions they
were before issue #64. Before that fix, both no-hole methods returned the byte-identical point
cloud on every call (deterministic grid, and a rejection fallback that reseeded from the same
constant and was never reached with zero holes to reject), which silently defeated the training
loop's own intended "resample every step" and let the network overfit that one frozen finite
quadrature-node set — `InteriorEnergyTerm`/`PhysicalPotentialEnergyTerm`'s `U` and
`ExternalWorkTerm`'s `W_ext` are both plain `mean(f(x_i))` Monte-Carlo estimators, unbiased only
if the `x_i` genuinely vary across the optimization trajectory. Fixed by making both methods
draw a real per-call jittered sample; still fully reproducible run-to-run (same base seed
constant ⇒ same full sequence of per-call point sets).

**Any caller of `sample_interior`/`sample_boundary` that caches the result across multiple
steps now gets stale, non-varying points again — the exact bug class this fix exists to
prevent.** This bit `runner::run_user_problem_training_from` (the GUI-streaming path) directly:
it cached `sample_boundary`'s output once before its training loop (a real, correct-at-the-time
perf optimization from before issue #64, whose own doc comment said so and was true then) and
was never updated, silently re-freezing the boundary point cloud for the entire run. Caught
only because sub-issue #66 strengthened a benchmark assertion on that exact path. If you add a
new plate-problem training loop or diagnostic that resamples, resample every call it's meant to
vary on — never cache `sample_interior`/`sample_boundary`'s output across more than one call.

**AMR defaults to `amr_enabled=true` but the canonical no-hole benchmark needs it off.** Issue
#63's own PH4-09 policy states "AMR = OFF, uniform sampling = ON until the baseline is
mathematically correct" — AMR has no dedicated evidence yet (tracked separately). The shipped
`examples/problems/variational_no_hole_plate.toml` now sets `amr_enabled = false` explicitly to
actually honor that policy; omitting it silently pulls in the default and reintroduces
borderline-variance behavior on the GUI-streaming path specifically (headless never implements
AMR at all, so it was unaffected either way).

**Single source of truth (issue #73)**: `user_problem::resample_plate_step_data`/
`plate_multi_step_ctx` are now the ONLY implementation of the plate problem's per-step
resampling and `MultiStepCtx` construction — both `user_runner::run_headless_user_problem`
(headless CLI) and `runner::run_user_problem_training_from` (GUI-streaming) call them instead
of independently duplicating this logic, which is exactly how the boundary-caching bug above
happened (one copy got updated for issue #64's fix, the other didn't). Regression-proven by
`runner::tests::gui_streaming_step_zero_matches_independent_shared_function_computation`: the
GUI-streaming path's real, captured step-0 `total_loss` is asserted to exactly equal an
independent reference computed by calling the same two shared functions directly — genuine
unification, not just two call sites that happen to both compile.

**Deliberately NOT unified with Kirsch/pin-lug.** `headless::run_headless`/`runner::
run_training` (Kirsch) and `headless::run_headless_pinlug`/`runner::run_training_pinlug`
(pin-lug) remain two separate, independently-maintained implementations each — this is the
same precedent as `step_physics`/`step_physics_multi` (frozen, byte-exact-tested paths must
never be refactored into a thin wrapper around something else) and was never in scope for #73:
they've never exhibited this bug class, and forcing them through a shared abstraction would
carry real regression risk against their own frozen paths for no evidenced benefit. Only the
user-defined plate problem's headless-vs-GUI-streaming pair was unified.

**`parametric_problem.rs`'s own `FixedPoints`/`build_fixed_points`** (a *different* per-step
caching pattern, for the separate "instant inference over varying E/nu/P" feature) was
investigated during the same session and found to have the identical "sampled once, reused
every step" shape — but parametric problems never claimed per-step resampling and are not
regressed by issue #64 (their pre- and post-#64 behavior is the same: one static point draw for
the whole run). Left alone, out of scope for #64/#66/#73.

## Issue #63 Phase 4 close-out: what's operational, what isn't, and why L5 is blocked

Phase 4 (issues #64-#73, epic #63) reached VERIFIED-or-BLOCKED-with-evidence on every item once
but epic #63 was reopened — per this project's own standard, `BLOCKED` documented with evidence
is not the same as complete, and #74/#70/#71's remaining gaps are real work, not paperwork.
Full detail lives in
`docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`, `docs/FORMULATION_SUPPORT_MATRIX.md`, and
`docs/GENERAL_SOLVER_OPERATIONAL_STATUS.md` — this section is the short pointer for a future
session, not a duplicate of those documents.

**Operational**: Variational no-hole L4, both square and non-square geometries (real headless
runs pass all five P2-14 hard thresholds with real margin). **Not yet operational**: hole/Kt
numerical accuracy (L5) — real, evidenced, and explicitly not hidden behind a passing benchmark.

**Issue #74 (AMR+Variational autodiff crash) is FIXED and verified — via a DIFFERENT root cause
than first diagnosed.** See PH4-23 in the manifest for the full, corrected record; PH4-09's own
entry documents the original (real but incomplete) diagnosis for history. One-line summary: the
`probe_interior_energy_residuals`-through-`BInner` fix (routing the AMR residual probe through
the non-autodiff backend) is a genuine, valid improvement but was **not** the actual cause of
this crash — the crash fires on a fresh model's very first `.backward()` call, before AMR's own
warmup period even ends. The REAL cause is a confirmed, currently-unreleased upstream burn-
autodiff bug (`tracel-ai/burn` issue #5573 - a concurrent `backward()` can free another thread's
still-being-registered graph node via the library's own process-global post-backward cleanup
sweep; fixed by burn PR #5647, not yet in any published release), triggered in OUR OWN test
harnesses by two compounding causes: spawning a separate OS thread per comparison arm, and
building one `initial_model` then handing `.clone()` to one arm while moving the original into
the other - **a burn `Module`/`Tensor` clone is cheap and shares the same underlying autodiff
`NodeId`, it does not mint a fresh leaf**, so the second arm ends up reusing an identity the
first arm already drove through thousands of real backward passes. Fixed in
`ph4_09_controlled_comparison_...`/`ph3_12_controlled_comparison_...`/`gui_streaming_step_zero_
...` by (a) calling the training function synchronously in the test's own thread instead of
spawning one per arm, and (b) building independently-`.init()`'d models from the same seed
instead of clone/move. This is also the likely real explanation for `gui_streaming_step_zero_
...`'s long-documented "contention-flaky" behavior this whole epic attributed to vague floating-
point nondeterminism. AMR still stays OFF for the *canonical no-hole* benchmark specifically —
that was always a separate, independently-measured quality-regression finding, unrelated to
either crash mechanism.

**A `cargo test` run WITHOUT `--test-threads=1` (cargo's own default) can still occasionally
show `gui_streaming_step_zero_...` fail** - this is the SAME upstream bug at the cross-test-
function level (some unrelated, concurrently-running test's own training thread racing with
this one), not a new or unfixed issue. **CI is unaffected**: `.github/workflows/rust.yml`
already runs every job with `--test-threads=1` for an unrelated, pre-existing GPU/lavapipe-
contention reason. Use `--test-threads=1` for a reliable full-suite run locally too.

**Durable lesson**: a burn `Module`/`Tensor` `.clone()` is a cheap, identity-sharing clone, not
a deep copy that mints a fresh autodiff leaf. Any future test wanting two genuinely independent
training runs from "the same starting weights" must build them via two separate `.init()` calls
under the same seed, never via `.clone()`/move of one shared instance.

**Why L5 doesn't pass (issue #70), and why AMR alone doesn't fix it either**: a small hole
(radius/half-width ratio ~0.05) gets almost no collocation density near its own boundary under
uniform Monte-Carlo sampling — confirmed by a zero-cost sampling-only check (no training): only
~0.55% of interior points land within 2 hole-radii of the boundary. A real 3000-step, 4096-point
run gets `Kt=1.008` against the theoretical `3.0` (66% relative error). With #74 fixed, the same
config was re-run WITH AMR enabled (`issue_70_real_l5_with_amr_enabled_after_issue_74_fix`,
`user_problem.rs`, `#[ignore]`d) — AMR genuinely re-densified the point set across 3 sweeps
(`4096→706→1381→2281`, confirmed working by a companion zero-cost fixture proving the refinement
mechanism itself is sound) but Kt barely moved: `1.0088`, essentially unchanged. **Don't assume
"fix #74, enable AMR" solves L5 — it doesn't, at least not with AMR's current residual-driven
refinement strategy.** Working hypothesis (NOT verified): AMR's signal reflects where the
CURRENT network's residual is large, which may not correlate with the true stress concentration
early in training, so refinement may be concentrating in the wrong place, or too late in the
budget, to help. A future attempt should test this hypothesis directly (e.g. earlier/more
frequent sweeps, or a structural initial-density bias seeded from hole geometry rather than
waiting on the network's own residual) rather than assuming more of the same AMR config will
eventually converge.

**A pre-existing CI-only flake, found and fixed during this close-out**: CI (not local `wgpu`)
uses a software/different Wgpu backend than a local machine, and
`network::tests::coordinate_skip_represents_affine_displacement_and_leaves_stress_mlp_only` used
to assert exact `f32` equality on a hand-computed matmul+bias result — failing on CI with a
*different* mismatched value each run (`1.1000001` vs `1.1`, `-0.49999997` vs `-0.5`), proving
float rounding noise, not a logic bug. Now uses an epsilon-tolerant comparison. If a similarly
CI-only-flaky test turns up again, check for exact `assert_eq!` on raw `f32`/`Vec<f32>` values
first before assuming a real regression.

## Issue #78 Stage 1: multi-hole hardening — machinery vs. accuracy, and why they're separate

Full plan/progress: `docs/multi-hole-and-boundary-shape-epic.md`. Short version for anyone
touching a multi-hole (N≥2 circular `HoleSpec`) spec on the default (non-`hard_constraint_
ansatz`) `UserDefinedProblem` path:

**Machinery is genuinely proven for N holes** — containment, signed distance, boundary
normals/measures, valid-stencil, interior/boundary sampling (including narrow-ligament
rejection via `UserSamplingStrategy::contains_for_collocation`'s per-hole union), AMR lock
zones, one `HoleBcTerm` per hole, and per-hole Kt reporting in both headless and GUI default
paths all loop over `holes.iter()` correctly. Real headless runs (both Wgpu and NdArray
backends) on 2-hole and 3-hole configs complete cleanly — no panic, no NaN, finite per-hole Kt,
sane equilibrium error.

**Kt numerical accuracy has NO correction path for N≥2, and that's a structural fact, not an
oversight to file a follow-up for.** Every accuracy mechanism this codebase built for the
single-hole case is explicitly gated to exactly one centered `Free` hole:
`decomposition_applicable` (`user_problem.rs`), `hole_bias_fraction`'s `holes.len()==1` guard,
and `coordinate_embedding`'s `[hole] = self.holes.as_slice() else { return Raw }` pattern all
fall back to the plain raw-coordinate model for N≠1. A multi-hole run therefore has strictly
LESS accuracy machinery than the single-hole default case, which itself only reaches Kt≈1.0-1.5
against a true ≈2.4-3.0 (see the PH4-4x sections above). **Do not assume the single-hole
hard-constraint-ansatz fix "just needs generalizing" to N holes** — it's a real, separate,
harder problem (the ansatz's closed-form correction and the sampling bias are both derived
relative to ONE hole's own center/radius; a multi-hole generalization needs its own kinematic
decomposition scheme, not a loop over the existing one) and is explicitly out of this epic's
scope.

**A real, previously-unfixed bug closed alongside this**: `HoleBcTerm::name()` used to return
the CONSTANT `"hole_free"`/`"hole_fixed"` regardless of which hole a term belonged to. Two holes
sharing a BC (`triple_hole_plate.toml`'s own two `Free` holes, a real shipped example) collided
in `training_core.rs`'s `lam_by_name`/`raw_scalar_by_name`/`term_grad_norms` `HashMap<&str, _>`s
— the second hole's entry silently overwrote the first, so both ended up weighted by whichever
hole's SAW-adapted lambda was computed last. Both holes still received real gradients and
contributed to the training objective throughout — this was a diagnostics/weighting-fidelity
bug, not a dropped-physics one. Fixed via `hole_bc_term_name` (`user_problem.rs`): the first
hole of a given BC keeps the exact pre-fix unsuffixed name (every single-hole or mixed-BC, e.g.
one `Free` + one `Fixed`, spec's term names/diagnostics stay byte-identical); only the 2nd+ hole
sharing that BC gets a numeric suffix (`"hole_free_1"`, `"hole_free_2"`, ...). `UserDefinedProblem::
base_weight` matches by prefix (`starts_with("hole_free_")`) rather than needing a second exact
arm per suffix.

**Also new**: `UserGeometry::validate()` — a real, previously-nonexistent check (overlap and
out-of-plate holes were silently accepted before this) wired into both `pinn-app`'s
`--problem-spec` headless path and `pinn-gui`'s spec-Load button. Deliberately a pure geometric
check (in-bounds, non-overlapping), NOT an FD-stencil-safety-margin check — `pinn-core` has no
dependency on `pinn-solver`, where the real per-run margin formula (a function of
`training.fd_h`) lives, so "too close for a numerically safe FD stencil at this run's specific
`fd_h`" stays a real, narrower, still-open gap.

**Also new**: hole placement by edge-referenced distance (`from_left`/`from_right`/`from_top`/
`from_bottom` in `[[geometry.holes]]`, alongside the existing `center = [x, y]`) - pure
TOML-input-layer sugar via a custom `Deserialize` impl on `UserGeometry`, zero solver/physics
change. See `examples/problems/hole_by_edge_reference.toml`.

## Issue #78 Stage 2: N-hole hard-constraint ansatz — a real Kt-accuracy fix, and two real bugs
## found chasing it

Stage 1 (above) closed with "multi-hole Kt accuracy has no correction path yet." This stage
built one, following the multi-hole-fem-ground-truth-investigation's own approved plan
(`docs/multi-hole-fem-ground-truth-investigation.md` — the authoritative record; this section
is the short pointer). One-line summary: real, substantial improvement (trained Kt for every
`HoleBc::Free` hole went from off by 6-600x to within 12-20% of real FEM ground truth), via one
real methodology extension plus two real, previously-undiscovered bugs found investigating why
the first attempt didn't work.

**`AnnulusAnsatz::MultiHoleHardConstraint(Vec<HoleTractionFreeAnsatz>)`**
(`kirsch_hole_correction.rs`) generalizes the single-hole `HardConstraint` ansatz to N `HoleBc::
Free` holes, any position — `decomposition_applicable`'s centered/single-hole restriction was
confirmed (by reading its own code) to be exactly what it always said: a deliberate scope
narrowing, not a mathematical constraint. Product of envelopes (multiplicative suppression
stays EXACTLY zero at every hole's own boundary for any N), sum of closed-form corrections
(additive baseline is only APPROXIMATELY traction-free for N>1 — the same 0.1-4.4% interaction
error the investigation's own Phase B measured, now also present in the trained field, not just
the closed form). N=1 reduces byte-identically to the pre-existing path — proven, not assumed.
`UserDefinedProblem::new_with_hard_constraint_ansatz` builds one `HoleTractionFreeAnsatz` per
Free hole; `loss_terms()`'s existing single-hole `hole_free`-suppression gate generalizes to
"any Free hole under an active hard constraint"; `UserSamplingStrategy`'s hole-biased sampling
and `hole_bias_quadrature_weights` generalize to bias every Free hole, splitting the budget
evenly (loud assertion, not silent mishandling, if bias disks would overlap).

**Real bug #1, found because the first real N-hole training run collapsed to Kt≈0.004-0.014
(trivial-solution collapse, not "just needs more steps"):** `UserDefinedProblem::loss_terms()`'s
`affine_strain_pair` (the constant far-field background strain added to the network's own
learning target under kinematic decomposition) was gated ONLY on `decomposition_applicable`
(the single-centered-hole case) — never on the new N-hole `MultiHoleHardConstraint` case, even
though this constant's own value is independent of hole count/position by construction. Every
off-center/multi-hole hard-constraint spec therefore trained with the network having to learn
the ENTIRE affine far-field background from scratch on top of refining the hole correction —
precisely the gradient-competition failure mode issue #77's original decomposition fix existed
to eliminate, silently reintroduced for exactly the geometry class this stage targets. Fixed by
generalizing the gate to `decomposed || self.hard_constraint_active()` (`UserDefinedProblem::
hard_constraint_active` made `pub(crate)` for this) — a strict OR, byte-identical for every
pre-existing case (`decomposed=true` already implied `Some` before; the new case this adds is
`decomposed=false, hard_constraint_active()=true`). Alone, this moved trained Kt from ≈0.004 to
≈0.5 once the spec's own `formulation`/`measure_aware_training` were also corrected to match
(see next paragraph).

**A real methodology trap this session fell into and corrected**: the shipped
`triple_hole_plate.toml`/`notched_plate.toml` examples have NO `[formulation]` override, so
they use `default_formulation()` — the literal PRE-#77 Hybrid trajectory, which never got issue
#77's own Kt fix at all. PH4-42's real verified L5 result used `formulation = "Variational"` +
`training.measure_aware_training = true` explicitly. Real headless verification MUST use that
same formulation on any hard-constraint spec — `variational_triple_hole_smoke.toml`/
`variational_notched_smoke.toml` (already-shipped examples with the right formulation) are the
correct base to build a hard-constraint verification config from, not the plain production
examples.

**Real bug #2, found because a P2-09 "trivial-solution" warning (load transfer ratio 0.02-0.07)
persisted even after bug #1's fix, despite the model's own real Kt already reading correctly
(~2.5) via a DIFFERENT, correctly-wired diagnostic function.** `probe_load_transfer`/
`probe_reaction_force`/`probe_boundary_residuals` (`user_problem.rs`) had never been extended
to accept an `ansatz`/`affine_strain_pair` parameter at all — unlike `probe_hole_boundary_
profile_derived` (correctly fixed back in PH4-41), these three read the model via a bare
`fwd_embedded` forward, completely bypassing any active `AnnulusAnsatz`/affine background, for
ANY spec. This bug predates issue #78 entirely and would have affected the ORIGINAL single-hole
L5 hard-constraint case too — never noticed because that case's real PH4-42 verification went
through `run_user_problem_training_with_diagnostics`/`user_problem_l5_diagnostic` (correctly
ansatz-threaded from the start), never `run_headless_user_problem`'s own printed CLI
diagnostics. Fixed by threading `ansatz`/`affine_strain_pair` through all three functions
(same `stencil_forward_with_ansatz`-based pattern as `probe_hole_boundary_profile_derived`) and
updating every call site: the headless CLI diagnostic loop, the GUI-streaming vis-cadence block,
the GUI checkpoint-save serving loop (all three now use the model's own real training-time
`problem.ansatz(0)`), the GUI checkpoint-LOAD serving path (deliberately LEFT `IdentityAnsatz`-
only — checkpoint metadata doesn't carry `ProblemSpec.architecture` yet, a real, disclosed,
still-open gap, not silently glossed over), `run_no_hole_benchmark` (correctly stays
`IdentityAnsatz`-only — no hole, no ansatz concept), and 8 pure-logic tests (`IdentityAnsatz,
None` — every one already trained a plain `UserDefinedProblem::new(spec)`). This alone took
the load transfer ratio from 0.02-0.07 (spurious "collapsed solution") to 1.00-1.01 (genuinely
healthy) with ZERO change to the actual trained model weights — purely a diagnostic-
reconstruction fix, proving the earlier false alarm was a measurement bug, not a training bug.

**Real trained result** (both real shipped geometries, both bugs fixed, `formulation=
"Variational"`, `measure_aware_training=true`, 3000 steps, `hole_bias_fraction=0.4`):
`notched_plate.toml` (N=1 Free, off-center) hole0 Kt=2.694 (FEM=3.063, superposition=2.960);
`triple_hole_plate.toml` (N=2 Free, off-center) hole0/hole2 Kt=2.507/2.507 (FEM=3.133/3.058,
superposition=2.994/2.994); the Fixed middle hole read Kt=1.048, not directly comparable to the
investigation's own Free/Free/Free FEM numbers (a pre-existing, disclosed caveat — the FEM tool
has no Fixed-BC support). Load transfer ratio 1.00-1.01 and reaction-force equilibrium error
~1% on every run — genuinely healthy, non-collapsed solutions.

**Honestly NOT full PH4-42-level convergence (0.67-1.23%)** — 12-20% remaining error, for two
disclosed reasons: off-center/multi-hole specs never get the single-hole path's OTHER accuracy
machinery (log-polar embedding, `SequentialTwoStage`, still single-hole-gated by this stage's
own deliberate scope), and the additive baseline itself carries Phase B's own measured N>1
interaction residual into the trained field, not just the closed form.

**The flat-loss-plateau observation above has since been fully investigated in a direct
follow-up (same session) — real evidence, not left as an open flag.** Full detail lives in
`docs/multi-hole-fem-ground-truth-investigation.md`'s own "Follow-up" section; short version:

**A NEW real bug found and fixed while investigating the (separately real) BC-mismatch
caveat**: `tools/multi_hole_reference.py`'s FEM ground-truth tool never supported `Fixed`
holes at all — every prior FEM number in this stage (and Phase A/B before it) treated every
hole as traction-free, while the real PINN specs have one genuinely `Fixed` hole each. Fixed by
adding real Dirichlet (zero-displacement) BC support. Building and verifying that fix surfaced
a SECOND, more serious, pre-existing bug in the SAME tool: its rigid-body-motion pin scheme
(asymmetric by construction — full pin on the left edge, roller-only on the right) had always
been silently wrong, invisible only because every all-`Free` geometry this tool ever shipped
with made the pins carry zero reaction force regardless of asymmetric placement. The instant a
`Fixed` hole (a real internal support with a genuinely nonzero reaction) was added, this broke
physical mirror symmetry outright — two Free holes that MUST have identical Kt by symmetry
(`triple_hole_plate.toml`'s own real geometry) came out ~30-50% different, confirmed real (not
noise) via a geometry-mirroring cross-check. Fixed with a genuinely symmetric pin scheme;
verified the fix is a pure bug fix with ZERO effect on every existing all-`Free` number (direct
stress-field comparison, old vs. new pin choice, agrees to `~1.6e-10` relative precision). 6
new regression tests. Real, BC-corrected FEM ground truth is now available and materially
changes the comparison picture — see the investigation doc's own table.

**The flat-loss plateau itself: definitively diagnosed via a new live per-step Kt diagnostic**
(`user_runner.rs::run_headless_user_problem`, printed every checkpoint during training, not
just at the end — a genuinely new, permanent capability). Real finding: Kt reaches its final
value by ~step 300 and then GENUINELY does not move for the remaining ~2700 steps — the flat
total_loss is real convergence, not a metric hiding continued progress, and not a training bug.
Three independent hyperparameter experiments (hole_bias_fraction 0.4→1.0, hidden_dim 64→128,
peak lr 1e-3→5e-3), each a major axis change, each reproduced the IDENTICAL converged Kt to
3-4 significant figures — cleanly ruling out collocation density, network capacity, and
optimizer step size as the cause. Working conclusion: the remaining accuracy gap is a
structural/formulation limitation, not a tuning problem — the N-hole ansatz (this stage's own
work) gives the network a strictly LARGER, less-constrained learning target than the original
single-hole L5 case had, because L5's real 0.67-1.23% accuracy came from the ansatz AND
kinematic decomposition TOGETHER, and only the ansatz half was generalized to N holes this
stage. Full closure would mean generalizing kinematic decomposition itself to N holes too — a
real, substantial, separately-scoped future task, deliberately not attempted in this pass. This
is a DEFINITIVELY DIAGNOSED, disclosed open item (evidence in hand for what it is NOT), not a
silently-accepted ceiling — matches this project's own "BLOCKED documented with evidence is not
the same as complete" standard (Issue #63 Phase 4 close-out, above).

Full regression suite after every change in this stage (`cargo test -p pinn-solver -p pinn-core
--release -- --test-threads=1`): 546 passed, 1 failed — the same pre-existing, unrelated
`compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key` failure this project has
tracked since before this stage began. Zero regressions from the ansatz generalization, either
Rust-side diagnostic bug fix, the live-Kt diagnostic addition, or the FEM tool's own BC/pin
fixes (Python-side, verified by its own separate 16-test suite, `tools/test_multi_hole_
reference.py`).

## Issue #78 Stage 2 second follow-up: `CoordinateEmbedding::MultiHoleChart` closes a real gap, doesn't move Kt — five hypotheses now falsified

Re-examining the "generalize kinematic decomposition to N holes" framing above (before
implementing it) found it doesn't hold up: decomposition's only real effect for the hard-
constraint path (`affine_strain_pair`'s contribution to `PhysicalPotentialEnergyTerm`) was
ALREADY generalized to N holes by this stage's own first bug fix (`decomposed ||
hard_constraint_active()`). Nothing was left to generalize under that name — corrected here
rather than silently re-doing complete work. The real remaining single-hole-only accuracy
machinery was `UserGeometry::coordinate_embedding()`: a single-hole geometry gets 7 explicit
hole-relative features (`SingleHoleChart` — polar-like invariants relative to the hole's own
center/radius) baked into the network's own input; every multi-hole geometry fell back to bare
`Raw` (3 columns, zero geometric hole-awareness) — `SingleHoleChart`'s own doc comment already
flagged this precise gap ("intentionally deferred until it has an unambiguous benchmark" — this
stage's own repeated-identical-Kt finding is that benchmark).

**Closed**: `CoordinateEmbedding::MultiHoleChart` (`pinn-core/src/user_geometry.rs`) computes
the same 7 features independently for EVERY hole (not just `Free` — a `Fixed` hole needs
geometric awareness too) and concatenates them (`3 + 7*N` input width); `network::multi_chart_
embed` is the forward-pass counterpart, proven byte-identical to `SingleHoleChart`'s own
`chart_embed` at N=1 and correct per-hole at N=2 (3 new tests). `holes.len()==1` is untouched —
every single-hole spec's embedding stays exactly what it was. `CoordinateEmbedding` can no
longer be `Copy` (only `Clone`) now that it holds a runtime-length `Vec` — a real, disclosed,
mechanical ripple the compiler enumerated exhaustively, fixed with `.clone()` at each call site
(cheap, never a hot-path cost).

**A real, separate bug this exposed**: `probe_boundary_residuals`'s hole-ring loop used a bare
`geometry.coordinate_embedding()` instead of the model-aware `embedding_for_model(model,
geometry)` this file's every other diagnostic probe already uses — harmless while every multi-
hole geometry's "default" embedding was `Raw` (any model built for it was also 3-wide, so the
two always coincidentally agreed), but a real shape-mismatch panic the instant a wider
`MultiHoleChart` default no longer matched a deliberately-narrower test model
(`probe_load_transfer_handles_a_geometry_with_holes` caught it - an existing test, not a new
one). Fixed to match the established convention.

**Real result: closes a genuine architectural gap, but did NOT move the trained Kt.** A live
per-step check showed the IDENTICAL `hole0=2.507, hole2=2.507` with the wider, hole-aware
embedding as without it — the same value this stage's other three hyperparameter experiments
already landed on. A fifth experiment (extending `hole_bias_fraction`'s sampling bias to the
`Fixed` hole too, not just `Free` holes — a real, plausible, cheaply-testable hypothesis, since
the `Fixed` hole got zero extra near-hole collocation density) also reproduced the identical
result and was reverted (no evidence to justify shipping it). **Five independent structural/
hyperparameter axes, all cleanly falsified**: collocation density (2x), network capacity (2x),
learning rate (5x), coordinate embedding (Raw→N-hole-aware chart), and Fixed-hole sampling
bias. This is strong evidence AGAINST a representation/optimization-search explanation and FOR
a genuine property of the Π functional itself as currently formulated for this geometry — i.e.
Kt≈2.507 plausibly IS the true minimizer of the sampled, measure-aware Monte-Carlo Π estimate
this codebase computes, and the remaining ~12-20% gap versus FEM reflects a real difference
between that estimator and a direct-stiffness-matrix FEM solve for a multi-hole domain
specifically — not a training deficiency. **This is now a formulation-level question** (auditing
the Π estimator's own correctness for N holes — quadrature-weight area bookkeeping, FD stencil
step size relative to the smallest hole's radius, etc.), not a hyperparameter-search one, and
is out of scope for further blind iteration — a real, scoped, disclosed item for a future,
more rigorous mathematical audit, not a silently-accepted ceiling.

Full regression suite after this follow-up: same 546 passed / 1 pre-existing failure, zero
regressions, plus 3 new `multi_chart_embed` tests and 1 updated `pinn-core` test
(`coordinate_embedding_preserves_raw_no_hole_and_generalizes_multi_hole_inputs`, renamed from
its own pre-#78 name to reflect the real, intentional behavior change it now asserts).

## Issue #78 Stage 2 third follow-up: the actual root cause of the Kt gap - found and fixed

The "formulation-level question" framing above was too pessimistic. Full derivation in
`docs/multi-hole-fem-ground-truth-investigation.md`'s "Fourth pass" section; short version:

**The decisive proof**: a real trained model's own measured Kt (`2.5072`) matches the PURE
closed-form `MultiHoleHardConstraint` baseline (no network contribution at all) evaluated at
the SAME real FD-safety-margin radius (`2.5075`, computed independently) to four decimal
places. The network's own learned correction near the Free holes contributes essentially
nothing.

**Why**: Kt is necessarily measured at `r = hole.radius + margin` (never exactly at the
boundary, an FD-stencil-safety requirement). For this codebase's real shipped multi-hole
geometries, `margin` is only ~5-7% of the hole's own radius, and `traction_free_envelope`'s
own saturation rate (tuned so `phi(3·radius)≈0.98`) gives `phi ≈ 0.44%` at that tiny distance.
The network's own gradient-trainable output is multiplied by `phi` before being added to the
closed form - meaning the GRADIENT reaching the specific weights that would learn a local
correction is ALSO scaled by ~0.44%, a severe, real, precisely-quantified vanishing-gradient
bottleneck exactly where Kt is read. This is exactly why none of the five (then six, after
testing `LAM_HOLE_FIXED` 50→5 too) previously-falsified hyperparameter axes ever moved the
needle - none of them change what `phi` numerically IS at the margin.

**The fix**: `traction_free_envelope_scaled(x, y, a, saturation_scale)` - a faster-saturating
generalization (`saturation_scale=1.0` byte-identical to the original), still exactly `0` with
exactly zero derivative at the true boundary (the hard constraint itself is untouched). Applied
via a new `HoleTractionFreeAnsatz.saturation_scale` field - strictly `1.0` at N=1, load-bearing
since `new_with_hard_constraint_ansatz` is the SAME function PH4-42's own verified L5 result goes
through. Verified zero regression two ways: unit tests proving the N=1 gate, AND a real
end-to-end re-run of `issue_77_l5_hard_constraint.toml` (N=1) giving `Kt=2.4444` - matching
PH4-42's own documented converged value (`2.444127304`) to 4 significant figures.

**The scale itself is DERIVED from geometry, not hand-picked - this was a deliberate correction
mid-investigation, not the first design.** A first pass tried a flat `MULTI_HOLE_SATURATION_
SCALE` constant (`15.0`, then `30.0`) applied to every hole regardless of its own size. The user
asked whether this needed to be a fixed input at all, wanting the app to derive it instead of
requiring a hand-tuned magic number - it does now: `multi_hole_saturation_scale(hole_radius,
margin)` solves `traction_free_envelope_scaled`'s own formula for the `scale` that makes `phi`
reach a dimensionless target (`TARGET_PHI_AT_MARGIN = 0.9`) at the real Kt-measurement point
(`scale = sqrt(-ln(1-0.9)) / (margin/hole_radius)`, `margin` = the same `ring_anchor_margin_m`
the Kt diagnostic already uses). Only the dimensionless target remains a constant - a raw scale
magnitude no longer needs re-guessing per geometry, it's computed from each hole's own
radius/margin ratio.

**Real, measured improvement** (`triple_hole_plate.toml`, N=2 Free holes, `scale≈22.75`
computed automatically for this geometry): hole0/hole2 Kt went from `2.507`/`2.507` (no fix) to
`2.930`/`2.958` (FEM ≈2.98-3.06/2.97-3.02) - within a few percent of FEM, matching the quality
of the hand-picked `scale=30` trial (`2.973`/`2.956`) without hand-tuning either number to this
specific geometry. Training dynamics genuinely healthier: `grad_norm` stays real and nonzero
throughout instead of collapsing then oscillating, and per-step Kt visibly moves during training
instead of being bit-for-bit frozen. Full regression: unchanged count (553 passed, +4 new/
updated tests for the derivation), same 1 pre-existing unrelated failure, zero regressions.

**Honestly still open**: hole0's Kt convergence check still flags `NOT converged` (radial
Δ≈0.147) at this higher saturation rate, matching the same open caveat the `scale=30` hand-picked
trial already had - close to FEM but not yet a fully settled number; the constitutive-residual
max also rises alongside Kt accuracy, a real trade-off not yet characterized. `TARGET_PHI_AT_
MARGIN=0.9` itself is not swept (chosen as the dimensionless analogue of the best-measured hand-
picked trial). A genuinely trainable `saturation_scale` (a `burn` `Param` updated by gradient
descent, rather than solved in closed form) was considered and set aside - the closed-form
derivation already removes the hand-tuned magic number; making it co-trained with a highly
nonlinear envelope function is separate, larger engineering scope with its own risk.
