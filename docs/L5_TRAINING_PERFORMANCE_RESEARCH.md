# L5 annular-decomposition training-time research (issue #77 support work)

Status: **research only, no code changed**. Written while the extended
12000-step convergence-trend run (`issue_77_l5_extended_convergence_trend_trace`)
was in flight. Every candidate below is graded by real evidence already in hand;
final prioritization is explicitly **contingent on that run's result** (see
"Decision gate" at the end) — an optimization to make a still-diverging run finish
in the same wrong place faster is not the goal.

## Measured baseline

`issue_77_l5_annular_diagnostic_trace` (release, default Wgpu backend, L5 config:
`hidden_dim=64, n_hidden=8, n_interior=4096, n_boundary=4096`, two-domain annular
decomposition): **3000 steps in 2395.68 s = 0.7986 s/step**. This is the number
every candidate below is measured against.

For comparison, this codebase's own documented Kirsch (single-domain,
`SolverConfig::default_kirsch()`) baseline is ~150-200ms/step on the same Wgpu
backend (`CLAUDE.md`'s Hardware-adaptive-execution section, Phase 2's real
measurement). L5's ~800ms/step is ~4-5x that — plausible given two full
domains' forward/backward per step (roughly 2x) plus this config's larger
`n_interior`/`n_boundary` (4096/4096) than Kirsch's typical defaults.

## Already done — not new opportunities

Checked against the codebase before proposing anything, to avoid re-litigating
solved problems:

- **`burn`'s `fusion` feature is already enabled** (`Cargo.toml:24`:
  `features = ["std", "autodiff", "wgpu", "fusion"]`). The stale
  `perf/enable-burn-fusion` branch predates this — already landed on `main`.
- **Batched GPU→CPU scalar syncs already exist** (`training_core::t_scalars`,
  used in `step_physics`/`step_physics_multi`/`compute_gradient_conflict(_multi)`
  — issue #16, closed). A stale `perf/batch-gpu-cpu-syncs` branch proposed
  exactly this; it's already on `main` under a different implementation.
- **Phase 3's rayon-for-resampling investigation concluded resampling is
  ~0.05-0.1% of step cost** (`crates/pinn-solver/benches/resample.rs`,
  ~21.7us/step for pin-lug's real per-step resample vs. tens of ms for the
  actual tensor step) — not re-measured for the annular sampler specifically
  in this pass, but the mechanism (LCG-seeded host-side point generation) is
  the same class of cost and this precedent is strong enough not to prioritize
  re-checking without new evidence it's grown disproportionate.
- **`PerformanceProfile`/`ExecutionMode` (`EXEC_PROFILE=eco` etc.) only affects
  the `ndarray-backend` CPU path** (rayon thread cap + sampling-density
  reduction) — irrelevant to this run, which uses the default Wgpu backend.

## Real, evidenced candidates

### 1. The outer domain appears to converge almost immediately — asymmetric step budget

`physical_potential` (the OUTER domain's own energy term, post-Step-1
decomposition represents only the small residual correction to the exact
affine field) measured at all 4 diagnostic-trace checkpoints:

| step | physical_potential (outer) | annulus_potential (annulus) |
|---|---|---|
| 0 | 0.98418 | 0.015949 |
| 300 | 1.06676 | 0.010758 |
| 1500 | 1.10395 | 0.000657 |
| 2999 | 1.00996 | 0.004298 |

The outer term shows **no net progress across 3000 steps** — it fluctuates in a
tight 0.98-1.10 band from step 0 onward. This is exactly what Step 1's own
kinematic decomposition would predict if it worked as intended for the outer
domain (the outer network only has a small correction left to learn, once
`u_affine` is subtracted, and evidently finds it almost immediately) — but it
means the OUTER model's own forward/backward pass, roughly half of every
step's cost, may be mostly wasted compute for most of the run.

**Candidate**: give the outer domain a coarser update schedule (e.g., update
it every 2nd or 4th step, or with a smaller network) once its own term is
observed to have plateaued, while the annulus domain (the one that actually
needs to develop the concentration) keeps updating every step. This is a
per-domain step-frequency asymmetry, not currently a concept this codebase's
step driver has — would need new, carefully-scoped machinery in
`step_physics_multi`/`compute_domain_forwards`, and must preserve the
interface-continuity terms' correctness (they read BOTH domains' current
state every step they fire, so an outer-domain "stale" step needs explicit
handling, not just skipping its optimizer step).

**Risk**: the flat trend could also mean the outer term is *already at its own
local minimum's noise floor* rather than genuinely done — the diagnostic only
has 4 points; needs more resolution before trusting this as a skip signal (the
same lesson as the Kt-trend finding that triggered this whole research pass).

### 2. Backend choice: ndarray+SIMD+rayon vs. Wgpu for this specific small-network config

This codebase's own Phase 1 finding (`CLAUDE.md`, hardware-adaptive-execution
section): Wgpu's per-step wall time was **near-flat across a 64x range of
`n_interior` and 16x range of `hidden_dim`** — consistent with
kernel-compile/dispatch overhead dominating over actual FLOPs at these problem
sizes, not clean O(n) scaling. If dispatch overhead is the dominant cost and
is roughly *size-insensitive*, a CPU backend with no per-op GPU dispatch
(`ndarray-backend`, already SIMD+rayon-enabled per `pinn-solver/Cargo.toml`'s
own documented rationale) could plausibly be competitive or faster for this
specific `hidden_dim=64` two-domain config — the exact regime the fusion
feature and Wgpu's own kernel-compile overhead are least advantaged in.

**Candidate**: a real, short (not the full 3000/12000-step run — a few hundred
steps is enough to get past warmup and measure steady-state per-step cost, per
Phase 2's own finding that warmup dominates only the first handful of steps)
timing comparison, `--features ndarray-backend` vs. default Wgpu, on the exact
L5 annular config. Cheap to run (minutes, not hours) and directly answers the
question with this project's own established methodology instead of a
prediction from an analogous-but-different past benchmark.

**Risk**: Phase 1's bench measured `step_physics` (single-domain Kirsch path),
not `step_physics_multi`/the annular two-domain path — the flat-across-sizes
finding may not transfer exactly; must be re-measured on the real path, not
assumed.

**Real result (measured)**: `issue_77_backend_comparison_timing` (new test,
300 steps, 50-step warmup discarded), `--features ndarray-backend`:
**steady-state 0.9026 s/step** (225.643s over 250 steps) — **~13% SLOWER**
than the Wgpu reference (0.7986 s/step), not faster. The hypothesis is
**not supported** by this measurement.

Caveat, disclosed rather than hidden: this run executed concurrently with the
12000-step Wgpu extended-convergence run (both were intentionally run in
parallel — ndarray is CPU/rayon-bound, Wgpu is GPU-bound, expected not to
contend directly, but host-side CPU scheduling/orchestration overlap is a
real, undisclosed-magnitude confound on this specific number). Directionally
still informative (ndarray was not a dramatic win, and trended the wrong way)
but a clean, uncontended re-measurement would be needed before treating
"~13% slower" as a precise figure. Given the direction is already negative,
re-measuring cleanly is low priority unless a future finding makes backend
choice load-bearing again.

### 3. Interior sampling budget vs. measured SNR headroom

This session's own zero-cost check found the annulus-local energy estimator's
SNR is **~12.2** at `n_annulus=2048` (well above any reasonable detection
threshold — see the `PH4-24`/`PH4-25`-adjacent Step 3 redirect in
`PHASE_4_IMPLEMENTATION_MANIFEST.md`). SNR scales as `sqrt(n)`; reaching a
still-comfortable SNR~6-7 would need roughly `n_annulus~600-700` — a real
`>2x` reduction in the annulus domain's own per-step point count, and a
proportional reduction in per-step FLOPs/dispatch count for that domain's
forward/backward.

**Candidate**: reduce `n_annulus` (and possibly `n_outer`, which has not been
SNR-checked the same way — the outer domain's own signal is essentially the
already-small residual correction, likely an even *more* forgiving case) with
an explicit before/after Kt-trajectory comparison, not just a raw speed
number, since sample count also affects gradient noise/variance during
training, not only the final estimator's reported value.

**Risk**: this SNR calculation was for the *converged/analytic* Kirsch field,
not the network's own (still-training, currently-wrong) field — the estimator
noise during EARLY training, when the field is far from the true solution,
could behave differently. Needs to be checked against real training dynamics,
not just the closed-form target, before trusting it as a training-time lever
rather than only a post-hoc measurement lever.

### 4. Plateau-based early stopping (reuse existing machinery)

`pinn_solver::controllers::ConvergenceTracker::for_metric` (already built,
used by pin-lug's `MetricMode::Relative` path) is explicitly documented as
NOT wired into the plate/user-defined-problem path at all
(`CLAUDE.md`'s Convergence-cascade section: "no curriculum/AMR/decision-maker/
SAW-BRDR-tiering" is a deliberate v1 scope cut, not an oversight). If the
extended run's Kt trajectory turns out to be smooth and monotonic (not
noisy/crash-prone the way Kirsch's own K_t curve is), a plateau-based stop
using this existing tracker would let future runs stop exactly when they've
actually converged instead of guessing a fixed step count up front — this
doesn't speed up any SINGLE run, but avoids wasting wall-clock on runs whose
step count was set too high (or re-running because it was set too low).

**Risk**: needs a real, price-of-entry integration into
`run_annular_decomposition_training_inner`/`_with_diagnostics` (currently
neither exists there) — nontrivial, and the pin-lug precedent's plateau/crash
thresholds are pin-lug-tuned, not transferable without their own tuning pass
for this problem's own metric (Kt-derived-FD-VM) scale.

## Not proposed (explicitly out of scope)

- Reducing the mDEM stencil's 5-point-per-point forward pass — needed for FD
  accuracy, not a real lever without accepting worse derivatives.
- Reducing `hidden_dim`/`n_hidden` (network capacity) — a physics/accuracy
  lever, not a pure performance one; changing it changes what Kt the network
  *can* represent, conflating two different investigations.
- AMR — `amr_enabled: true` is set on this config but is **inert** for the
  `run_annular_decomposition_training_inner` code path (no AMR call exists
  there at all, per this session's own investigation) — it costs nothing to
  begin with, so there's nothing to remove or optimize.

## Decision gate — RESOLVED, and the answer changes the priority list

The extended run finished. Result (see `PHASE_4_IMPLEMENTATION_MANIFEST.md`'s
PH4-26 for the full writeup): Kt **peaks around step 3000 (1.287) then
declines** to 0.977 by step 11999 — training longer *hurt*, not helped.
Root cause: one shared `LrSchedule` (`ReduceLROnPlateau` on TOTAL loss)
governs both domains; `physical_potential` (outer) plateaus almost
immediately and drags LR down for the annulus domain too, right around
when Kt peaks.

This resolves the gate the opposite way from either branch originally
written above:

- **Candidate 1 (asymmetric outer-domain schedule) is now the real, primary
  lever** — but reframed from "the outer model is done early, so update it
  less" to the sharper, now-evidenced version: **give the annulus domain its
  own `LrSchedule`, decoupled from the outer domain's early-plateauing total
  loss.** This is a genuinely new mechanism (`run_annular_decomposition_
  training_inner` has one shared schedule for N domains today) — a training-
  *quality* fix, not merely a training-*speed* one; it's now the top
  candidate on this list.
- **Candidate 4 (plateau-based early stop)** is validated as a real,
  independently useful mitigation — `DualMetricStopAdvisor` (added this
  session, `controllers.rs`, 4 passing tests) would have signaled stop near
  the real Kt peak (~step 3000-4500) rather than running to 12000 and ending
  in a worse state. Still only a symptom mitigation (train less, don't decay
  past the peak), not the root-cause fix candidate 1 targets — do both,
  candidate 1 first if only one fits in a session.
- **Candidate 2 (backend comparison)**: measured. Ndarray+SIMD+rayon was
  **~13% slower** than Wgpu (0.9026 vs 0.7986 s/step), not faster — hypothesis
  not supported (see that section for the concurrent-run confound caveat).
  Deprioritized; not worth pursuing further without a new reason to revisit.
- **Candidate 3 (trim `n_annulus`)** is now lower priority than candidate 1 —
  its own SNR analysis was for the *value* estimator on the converged field,
  and the real bottleneck turned out to be a training-*dynamics* issue
  (LR collapse), not sampling variance. Revisit only after candidate 1 is
  tried and a new Kt-vs-FEM number exists to optimize against.
