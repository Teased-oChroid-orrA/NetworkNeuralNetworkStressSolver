# Numerical and TUI reconciliation — 2026-09-23

## Scope and repository state

Implementation branch: `feature/pinn-numerics-tui-audit-20260923`, based on
`e3a55a0f5f1be131d093027fe9222d6c05af8f87`. At branch creation, local `main`, cached
`origin/main`, and live GitHub `main` all matched that commit. Existing deleted and untracked
`Debug_runs` files were preserved and remain outside this work's commits.

Downstream `powershell_tool/app-egui` work is isolated on
`feature/pinn-numerics-tui-integration-20260923`, based on
`0500a6ad62c0923378769fbeb9fea4e8c1583919`. Its pre-existing uncommitted
`app-egui/src/stress_solver.rs` changes were recorded before this work and preserved.

GitHub reports no repository rulesets and returns `404 Branch not protected` for `main`.
Changing repository administration requires separate approval. This implementation never
targets `main`.

## Negative loss and gradient finding

The user observed negative `total_loss`, `energy_loss`, and `neumann` plus high `grad_norm` on
fresh TUI and GUI runs, including `notched_plate.toml` and no-hole plates. Both interfaces call
the same `runner::run_training_user_problem` path, so matching readings are expected.

The old scalar labels do not have one stable mathematical meaning across all paths:

- `total_loss` is the signed weighted objective. It may legitimately be negative when it
  contains potential energy `Pi = U - W_ext`.
- `energy_loss` is a raw normalized optimizer term. Under the atomic Variational path this can
  itself be the signed `physical_potential`; under legacy Hybrid it is normally nonnegative
  strain energy.
- On the generic plate path, `neumann_loss` was assigned as
  `total_loss - energy_loss`. It can therefore include external work, constraints,
  constitutive consistency, and every other active term. It is not a Neumann traction-residual
  invariant and may be negative.
- A true squared Neumann traction residual is nonnegative apart from negligible roundoff.
- `grad_norm` is the L2 norm of model-parameter gradients before the optimizer update. Its
  magnitude is scale- and objective-dependent. A high finite value alone is not a defect.

Authoritative telemetry now transports the solver's existing loss ledger: named normalized raw
values, effective weights, weighted contributions, total, reconstruction difference, optimizer
tier, and pre-optimizer gradient norm. The TUI persists this data per run and shows signed
values without clamping. Physical probes separately show U, full load-potential work, Pi,
force balance, boundary residuals, and per-hole Kt when available.

`EnergyBalance.external_work` historically stores proportional-loading work
`0.5 * integral(t dot u)`, whose equilibrium value matches U. The optimizer's potential uses
the full load work. `EnergyBalance::load_potential_work()` and `physical_potential()` make that
factor explicit without changing serialized data or training mathematics.

## Preserved mathematical baseline

- Variational training retains one atomic `physical_potential = U - W_ext` term.
- Measure-aware integration and per-step resampling remain unchanged.
- SAW-BRDR uses absolute magnitudes only for decay-rate adaptation; it does not make the signed
  physical functional positive.
- Documented no-hole Variational L4 evidence and single-hole hard-constraint Kt evidence around
  2.43–2.44 remain the regression standard.
- Multi-hole machinery remains operational, but trained Free-hole Kt remains roughly 10–20%
  below corrected FEM evidence, with a larger Fixed-hole gap. Existing controlled experiments
  reject blind density, width, learning-rate, embedding, and sampling changes as solutions.
  This capability remains experimental until formulation and local stress reconstruction pass
  an independent acceptance gate.

## Entry-point reconciliation

| Entry point | Training path | Status |
|---|---|---|
| User plate GUI | `runner::run_training_user_problem` | Authoritative streaming path |
| User plate TUI | same as GUI | Objective/configuration parity by construction |
| User plate diagnostic example | same as GUI/TUI | Persists bounded JSONL evidence |
| User plate plain headless | `user_runner::run_headless_user_problem` | Independent console path; shares core step machinery but has distinct dispatch/reporting |
| Built-in Kirsch GUI/TUI | `runner::run_training` | Frozen single-domain path |
| Built-in Kirsch headless | `headless::run_headless` | Independent regression path by design |
| Pin-lug GUI/TUI | `runner::run_training_pinlug` | Streaming multi-domain path |
| Pin-lug headless | `headless::run_headless_pinlug` | Decision-maker behavior differs and remains documented |

Independent headless paths are intentional regression or console paths. UI parity claims apply
to TUI versus GUI for the same problem and effective configuration.

## Correctness and lifecycle changes

- Missing or invalid active L-BFGS weights fail closed in debug and release for single- and
  multi-domain objectives. Explicit zero remains distinct from a missing key.
- CLI and `pinn.env` parsing reject unknown arguments/problems/keys, missing option values,
  duplicate keys, malformed/nonfinite values, invalid ranges, unreadable explicitly selected
  files, and invalid post-conversion geometry before training starts.
- TUI provides Overview, Loss/gradients, Physics/holes, Sampling/AMR, Run/backend, and Logs
  tabs; signed charts; bounded history; small-terminal fallback; scrolling/filtering;
  pause/resume requests; cancellation; worker-panic propagation; persisted diagnostics; and
  bounded data/control channels.
- Terminal state uses RAII cleanup after every setup stage. Unix stdout and stderr are sent to
  the run's `solver.log` during the alternate screen, then restored. Worker disconnection is a
  visible error.
- CI labels short training as runtime smoke only. Deterministic L0/L3 and affine-field fixtures
  are explicit physics gates. A scheduled/manual workflow runs expensive trained no-hole
  Variational L4 and single-hole hard-constraint L5 tests with numerical acceptance assertions.
- Burn remains on 0.21 because the confirmed concurrent-backward fix is not in a compatible
  stable release. Solver tests remain serialized; concurrent autodiff runs are unsupported.

## Cross-repository reproducibility

`app-egui` consumes the solver by relative path and sees uncommitted solver changes. Its
independent lockfile controls resolution. Companion GUI charts now preserve signed losses and
label the legacy aggregate accurately. CI still needs a reviewed solver revision contract: its
sibling checkout reproduces directory layout but does not pin a solver `ref`, so identical GUI
commits can resolve different solver revisions. This is a release/CI policy decision.

## Operational-status rule

Compilation, test success, process completion, finite loss, and a falling objective are not
physical acceptance. A run is operational for an engineering case only when applicable
analytical/reference thresholds, field checks, force/energy balance, boundary residuals,
convergence evidence, and provenance are present and pass. Missing evidence means not evaluated.
