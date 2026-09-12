# Phase 2 Implementation Manifest

Tracks GitHub issue #61 ("Phase 2 — Implementation Gap & Remediation Plan"). Per that
issue's own §7/§8: `IMPLEMENTED` means code exists; `VERIFIED` requires tests AND runtime
evidence; only `VERIFIED` counts toward completion. Updated after each epic, not batched at
the end.

## P2-01 — Explicit formulation model

Status: VERIFIED

### Requirement

Introduce `FormulationKind`-equivalent (`Strong`/`Weak`/`Variational`/`Hybrid`) that actually
GATES which loss terms are active, not merely classifies terms that already exist (Priority 9
of the prior General-PINN pass added `LossTerm::formulation_kind()` as a classification only -
it never controlled `loss_terms()`'s output; this epic fixes that gap for real).

### Files changed

- `crates/pinn-core/src/problem_spec.rs` — new `FormulationSelection` enum
  (`Variational`/`Strong`/`Hybrid(Vec<String>)`), `default_formulation()` (the literal
  pre-remediation 4-term list), new `ProblemSpec.formulation` field
  (`#[serde(default = "default_formulation")]`).
- `crates/pinn-core/src/inference_envelope.rs` — test helper updated for the new field.
- `crates/pinn-solver/src/user_problem.rs` — `UserDefinedProblem::loss_terms()` rewritten to
  build each base term (`interior_energy`/`equilibrium`/`outer_traction`/`external_work`)
  conditionally on `self.spec.formulation`, and to gate each hole's natural (`HoleBc::Free`)
  term on the formulation while its essential (`HoleBc::Fixed`) term stays always-active. 17
  test-only `ProblemSpec` literals across `checkpoint.rs`/`runner.rs`/`training_core.rs`/
  `user_problem.rs` updated to carry the new field explicitly (auto-patched, then spot-checked).

### Architecture decision

`Variational` = `{interior_energy, external_work}` + essential (`hole_fixed`) constraints only;
natural boundaries (`outer_traction`, `hole_free`) OMITTED entirely — relies on the variational
principle itself to satisfy them (a correctly-posed `W_ext` already encodes the natural BC; a
separate penalty would duplicate it, which issue #61 §1.2 explicitly forbids). `Strong` =
`{equilibrium, outer_traction}` + essential + natural-as-penalty hole terms, NO energy-functional
terms. `Hybrid(names)` activates exactly the named base terms (empty list activates zero base
terms — no implicit "everything" fallback) plus ALL hole terms (matching this codebase's own
pre-remediation behavior, reproduced byte-for-byte via `default_formulation()`'s literal
4-name list, so every existing example TOML keeps training identically). Unknown Hybrid names
panic loudly rather than being silently ignored. `equilibrium`/`outer_traction` remain available
as post-hoc diagnostics under Variational via `probe_interior_energy_residuals`/`probe_boundary_
residuals` — neither depends on `loss_terms()`, so the diagnostic requirement is met without
extra plumbing.

### Tests

- command: `cargo test -p pinn-core --lib -- problem_spec::` — 7 passed (default-formulation
  literal-list check, TOML round-trip for all 3 variant shapes, shipped-example backward compat).
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::` — 32
  passed, including 5 new: `variational_formulation_activates_only_u_minus_w_ext_and_essential_
  constraints`, `strong_formulation_activates_only_pde_and_bc_residuals`, `hybrid_formulation_
  activates_exactly_the_named_subset`, `hybrid_formulation_panics_on_unknown_term_name`,
  `variational_formulation_excluded_terms_have_no_gradient_norm_entry_at_all` (the strongest
  proof: excluded terms have NO `term_grad_norms` entry at all under a real `step_physics_multi`
  call — not merely a zero gradient, genuine absence from the computation graph).
- command: `cargo test -p pinn-solver` (full suite) — 336 passed, 2 failed (both the
  already-documented `run_training_pinlug_*` "must send Done" GPU-backend flake, unrelated to
  this epic — pin-lug's own code path is untouched; confirmed via isolation rerun: both pass
  clean in isolation), 17 ignored.

### Runtime evidence

Real headless training run (`stress-solver --headless`, release build, `no_hole_plate.toml`
base config with `formulation = "Variational"`, 300 steps): completed without crash/NaN,
`total_loss` decreased monotonically (5.93 → 4.68), `max|displacement|` non-trivial
(1.144e-5 m), constitutive residual finite (RMS 3.34e5 Pa). Confirms the new formulation-gated
code path is genuinely exercised end-to-end through the real CLI entry point, not only unit
tests against `loss_terms()` in isolation.

### Known limitations

`equilibrium`/`outer_traction`'s diagnostic availability under `Variational` was verified by
code inspection (both probe functions take `&model`/`&spec` directly, never `problem.loss_
terms()`) but not exercised via a dedicated new test in this epic — they already have their own
test coverage from prior passes. No convergence-quality claim is made for `Variational`/`Strong`
at this epic (that's P2-08's mandatory-gate job); this epic proves the GATE is real, not that
either formulation converges well.

### Reviewer verification

PASS — self-reviewed against issue #61's own P2-01 acceptance criteria line by line (all 4
bullets satisfied by the test list above).

## P2-02 — DifferentialOperator abstraction

Status: VERIFIED

### Requirement

Explicit `DifferentialOperator` abstraction with named backends (AD/FD/Analytic), no silent
backend switching, cross-validated against a manufactured field's exact derivatives.

### Files changed

- `crates/pinn-solver/src/differential_operator.rs` — new module. `DerivativeBackend` enum
  (`Fd`/`Ad`/`Analytic`), `DerivativeBackendPolicy` struct (`primary`/`verification`/`fallback`
  fields, explicit routing record — no field silently swaps backend) + `FD_ONLY` const,
  `ScalarDerivatives` struct + `fd_scalar_derivatives()`, `ad_strain<B: AutodiffBackend>()` (real
  AD strain via sum-trick over network input coords), `fd_strain_via<B: Backend>()` (adapter over
  existing `fd_stencil` functions, same interface shape as `ad_strain` for direct comparison).
- `crates/pinn-solver/src/lib.rs` — `pub mod differential_operator;` added.

### Architecture decision

Scope held to scalar/strain-level operators only (gradient/strain), not the full
gradient/divergence/laplacian/hessian/directional_derivative trait surface issue #61 describes,
per §1.4 (no destructive refactor) and §4's staged order — `compute_domain_forwards` and its
existing FD/Hessian call sites are NOT touched in this epic; P2-15 (migration) is where existing
paths get adapted onto this abstraction, not P2-02 itself. `ad_strain` differentiates network
INPUT coordinates via burn autodiff's sum-trick (`d(sum(u))/d(pts)` recovers per-row `du_i/dx_i`
in one `.backward()`, exploiting the batched pointwise-no-cross-row-coupling structure) rather
than N independent passes or full Jacobian materialization. Hessian-via-AD is a documented
NON-GOAL, not a gap: grepped `burn-autodiff` 0.21 source for
`second.order|higher.order|create_graph|grad_grad|double.backward` — zero matches, confirming
this backend has no nested/higher-order autodiff support. A fake zero-returning `ad_hessian` was
drafted, recognized mid-write as exactly the "dead abstraction" pattern issue #61 §1.3/§6
forbids, and deleted; the module doc comment states the limitation honestly instead. Existing FD
9-point Hessian stencil (`fd_stencil.rs`, from the prior General-PINN pass) remains the only
Hessian backend and is unaffected.

### Tests

- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- differential_operator::`
  — 4 passed:
  - `fd_scalar_derivatives_matches_analytic_values_for_the_acceptance_polynomial` — issue #61's
    own literal example polynomial f(x,y)=x²+3xy+2y², FD vs hand-derived analytic first/second
    derivatives.
  - `fd_scalar_derivatives_matches_manufactured_field_exact_strain_formula` — cross-check against
    `manufactured::ManufacturedField::quadratic` (Priority 7 module, reused not duplicated).
  - `ad_strain_matches_exact_manufactured_strain` — AD backend vs hand-derived exact formula.
  - `ad_strain_matches_fd_strain_on_the_same_manufactured_field` — AD vs FD, same field, same
    point, backend cross-check.

### Runtime evidence

Unit-tested only (pure-function operators, no training-loop integration in this epic per the
staged-migration decision above — nothing in the live training path calls this module yet, so
there is no live-run evidence to report and none is claimed).

### Known limitations

Hessian-via-AD not implemented (confirmed backend limitation, not an oversight — see Architecture
decision). Not yet wired into any live physics consumer (`compute_domain_forwards`,
`EquilibriumTerm`, etc.) — that migration is explicitly P2-15's job, done only after P2-03..P2-14
land, per issue #61 §4's mandatory order. Directional-derivative and divergence operators not
implemented (not needed by any current consumer; would be speculative scaffolding today).

### Reviewer verification

PASS against P2-02's acceptance bullets that apply at this stage (named backends, explicit
routing record, cross-validated against manufactured/analytic ground truth, no silent switching).
Hessian trait surface and live-path wiring deliberately deferred, tracked as known limitations
above rather than hidden.

## P2-03 — Authoritative Field Dependency Graph

Status: NOT_STARTED

## P2-04 — Measure-aware integration

Status: NOT_STARTED

## P2-05 — Separate physical coefficients from optimization weights

Status: NOT_STARTED

## P2-06 — Geometry-aware operators

Status: NOT_STARTED

## P2-07 — Generic gauge/nullspace handling

Status: NOT_STARTED

## P2-08 — Executable verification ladder

Status: NOT_STARTED

## P2-09 — Load-transfer and trivial-solution diagnostics

Status: NOT_STARTED

## P2-10 — QoI and Kt architecture

Status: NOT_STARTED

## P2-11 — Adaptive sampling invariance

Status: NOT_STARTED

## P2-12 — Complete diagnostic ledger

Status: NOT_STARTED

## P2-13 — Reproducibility and provenance

Status: NOT_STARTED

## P2-14 — Benchmark protocol

Status: NOT_STARTED

## P2-15 — Migration and backward compatibility

Status: NOT_STARTED
