# `toy_beam`: standalone 1D methodology sanity check

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. Why a fast 1D Euler-Bernoulli DEM formulation exists separately from the 2D plate stack, and its real convergence verification against closed-form beam deflection.

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

