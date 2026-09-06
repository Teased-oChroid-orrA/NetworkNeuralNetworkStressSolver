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
