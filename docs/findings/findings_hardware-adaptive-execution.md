# Hardware-adaptive execution epic, Phase 1-4

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. The `Executor`/`ExecutionPlanner` scaffolding, per-step diagnostics instrumentation, the measured (and rejected) Rayon parallelization of resampling, and `PerformanceProfile`'s real Eco-mode sampling/thread-count effect.

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

