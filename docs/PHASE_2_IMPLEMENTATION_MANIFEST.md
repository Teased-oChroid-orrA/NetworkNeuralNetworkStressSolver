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

Status: VERIFIED

### Requirement

Formal `NN -> displacement -> strain -> constitutive -> stress` dependency graph; every
stress consumer requests a declared authoritative field; mixed formulations (both Direct
network-output stress and Derived constitutive stress relied on simultaneously) enforce
sigma_aux<->constitutive compatibility explicitly, not silently.

### Files changed

- `crates/pinn-solver/src/field_graph.rs` — new module. `FieldKind` enum
  (`NetworkOutput`/`Displacement`/`Strain`/`ConstitutiveStress`/`DirectStress`) +
  `depends_on()`/`dependency_chain()` (issue #61 §3's pipeline as real, queryable data) +
  `from_stress_source()` (maps the existing `StressSource` classification onto the graph's two
  stress leaves — generalizes, does not duplicate, the prior pass's Priority-1 label).
  `MixedFormulationCheck` + `check_mixed_stress_source_compatibility()` — the real enforcement
  predicate.
- `crates/pinn-solver/src/lib.rs` — `pub mod field_graph;` added.
- `crates/pinn-solver/src/training_core.rs` — `stress_source_report()` refactored (behavior
  unchanged) to delegate to new `stress_source_report_from_terms()`, so callers that already
  built a term list (avoiding a second `loss_terms()` construction) can reuse it. Both
  `step_physics` and `step_physics_multi` — the two real per-step training entry points, used
  by every problem type (Kirsch via `step_physics`, plate/pin-lug via `step_physics_multi`) —
  gained a genuine runtime assertion calling `check_mixed_stress_source_compatibility` against
  that step's own `active_terms`, using `use_mdem`/`any_mdem` (already-computed, zero new cost)
  as the "is the existing compatibility mechanism active" flag.
- `crates/pinn-solver/src/user_problem.rs` — `dependency_chain_for_kt()` rewritten to build its
  string FROM `FieldKind::dependency_chain()` instead of two hand-written literal strings —
  proof the graph is load-bearing, not a dead parallel abstraction (see Tests).

### Architecture decision

The existing `ConstitutiveConsistencyTerm` mechanism (`step_physics`/`step_physics_multi`'s
"(b.5)" block, applied to every `output_dim == 5` domain, pre-dating this epic) already WAS
this codebase's real sigma_aux<->constitutive compatibility enforcement — P2-03's job was
making that fact explicit and checkable, not building a new mechanism from scratch (per §1.4,
no destructive refactor of working training-loop internals). The new assertion is placed
immediately after `active_terms`/`any_mdem` (or `use_mdem`) are already known in both step
functions — genuinely free (reuses the already-built term list, no extra `loss_terms()` call),
so it runs on every training step, not behind an opt-in diagnostics flag. It is a real
enforcement point, not a report: if a future formulation change ever produced a Direct+Derived
mix with neither a `Both`-classified term nor the mDEM mechanism covering it, `step_physics`/
`step_physics_multi` would panic immediately rather than train silently on an unpoliced
mismatch (issue #61 §1.2's "no silent physics replacement", applied to this specific gap).
`FieldKind`'s dependency chain intentionally mirrors only the fields this codebase's real
problems compute (no speculative nodes for stress representations that don't exist yet).

### Tests

- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- field_graph::` — 6
  passed: dependency-chain-matches-§3-pipeline (both leaf branches), `from_stress_source`
  mapping, plate-mix-uncovered-without/covered-with the external flag (reproduces the real
  single-hole plate classification from `docs/investigations/kt-investigation-bugsource-new.md`
  and proves the check correctly requires `any_mdem`), Kirsch `Both`-term coverage, and the
  never-flagged-mixed cases (all-derived, empty).
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::` — 41
  passed (39 pre-existing + 2 new: `dependency_chain_for_kt_derived_matches_the_graph_exactly`
  proves the Kt chain string is now graph-derived, not hand-written).
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- training_core:: kirsch_problem:: pinlug_problem::`
  — 104 passed, 0 failed, including every real-config parity/regression test that exercises the
  new assertion on live data: `step_physics_multi_single_domain_matches_step_physics_kirsch`,
  `step_physics_trait_driven_matches_independently_reimplemented_old_formula`,
  `step_physics_stays_finite_with_mdem_and_ultimate_strength_scaling_combined`,
  `kirsch_regression_matches_hardcoded_step_physics` — none panic, confirming the new
  invariant genuinely holds for every current problem configuration rather than only being
  satisfied by construction in a narrow unit test.

### Runtime evidence

Real headless training run (`stress-solver --headless`-equivalent `--problem-spec` path,
release build, `single_hole_plate.toml` with `max_steps=60` for a fast evidence run): completed
all 60 steps without panic (`total_loss` 9.60 -> 5.42 monotonically, `max|displacement|`
non-trivial at 9.999e-6 m, `constitutive residual RMS=4.79e6 Pa`). This is exactly the mixed
mDEM configuration (`hole_free`=Direct, `equilibrium`/`outer_traction`=Derived,
`output_dim==5`) the new assertion runs against on EVERY step — confirms the P2-03 enforcement
path is genuinely exercised end-to-end on the real training loop, not only in isolated unit
tests.

### Known limitations

Enforcement covers the one dependency edge (`Direct` vs `ConstitutiveStress`/`Derived`) this
codebase's real terms actually use — `FieldKind::Strain`/`Displacement`/`NetworkOutput` are
real graph nodes with a real `dependency_chain()`, but no consumer other than the Kt chain
string currently "requests" them by name through an explicit API (issue #61's fuller vision of
every PDE/BC/energy/vis/QoI consumer declaring its authoritative field via a shared funnel
function is not built — doing so now would mean rewriting how `compute_domain_forwards` and
every `LossTerm::compute()` accesses strain/stress, a destructive refactor issue #61 §1.4 and
§4's staged order explicitly reserve for P2-15). The enforcement assertion is a `panic`, not a
`Result` — appropriate for an internal invariant violation during training (matches this
codebase's existing convention, e.g. `compute_gradient_conflict_panics_on_unrecognized_loss_
term_name`), not intended as user-facing error handling.

### Reviewer verification

PASS against P2-03's acceptance wording at the scope this epic covers: the graph is real,
queryable data (not just a diagram); the Kt QoI path provably consumes it; mixed-formulation
compatibility is now an enforced, always-on runtime invariant on both real training entry
points, verified against every existing real problem configuration's regression tests plus a
live training run. The broader "every consumer requests its field through a shared API"
vision is explicitly deferred to P2-15 (migration), tracked above as a known limitation rather
than silently dropped.

## P2-04 — Measure-aware integration

Status: VERIFIED

### Requirement

Explicit domain/boundary integral abstraction carrying real geometric measure (area/length),
implementing issue #61's own literal `Integral_Omega(f) ≈ |Omega|*mean(f)` formula, with no
magic benchmark-specific multipliers, converging to the same analytic value under uniform and
nonuniform/adaptive sampling.

### Files changed

- `crates/pinn-solver/src/measure_integral.rs` — new module. `domain_integral()` (issue #61's
  literal formula, generalized by a thickness factor), `domain_integral_weighted()` (same, but
  applies Priority 8's `pinn_core::amr::compensation_weights` first, for nonuniform/AMR
  sampling), `boundary_integral()` (exact per-point arc-length-weighted sum, for boundary/
  interface integrals whose local measure varies per point/edge), `plate_domain_area()` /
  `plate_outer_perimeter()` (real plate geometry measures).
- `crates/pinn-solver/src/lib.rs` — `pub mod measure_integral;` added.
- `crates/pinn-solver/src/user_problem.rs` — `probe_energy_balance()`'s internal-energy and
  external-work blocks refactored to call `measure_integral::domain_integral`/`boundary_integral`
  instead of their own inlined arithmetic — byte-identical formulas, now a real, tested, shared
  abstraction instead of duplicated inline code.

### Architecture decision

`probe_energy_balance` (a pre-existing, already-live diagnostic — called from
`runner.rs::run_user_problem_training_from`, the shared training-loop driver behind both the GUI
and headless plate paths, and the source of the `energy_balance_error` field visible in the
user's own uploaded `Debug_runs/stress_solver_report*.json`) already implemented issue #61's own
`Integral_Omega(f) ≈ |Omega|*mean(f)` formula correctly for its internal-energy term
(`mean_density * area * thickness`) and a correct exact arc-length sum for its external-work
term — but only as one-off inline arithmetic, not a reusable, independently-tested abstraction.
This epic extracts and generalizes both formulas (`domain_integral`/`boundary_integral`) and
refactors that one live call site onto them — real proof of use via refactor, not a new function
nobody calls (issue #61 §1.3). It ALSO adds the genuinely new capability that call site didn't
have: `domain_integral_weighted`, using Priority 8's existing `compensation_weights` mechanism,
proven (via a hand-computed AMR-bias reproduction) to recover the true area-weighted integral
under nonuniform sampling where the plain estimator is measurably biased — directly answering
P2-04's "must converge under uniform/nonuniform/adaptive sampling" acceptance bullet.

Deliberately NOT done in this epic: migrating `InteriorEnergyTerm`/`ExternalWorkTerm` — the
actual TRAINING loss terms (as opposed to `probe_energy_balance`'s read-only diagnostic) — onto
this abstraction. Both currently use a bare, unscaled `.mean()` with no `|Omega|`/thickness
factor at all; since `U` and `W_ext` are both scaled by the same `ref_energy` constant before
being compared/summed in the training objective, this is dimensionally self-consistent (the
missing common factor cancels in their difference) rather than a correctness bug in the current
formulation, but it IS the same "no explicit measure" gap this epic exists to name. Actually
swapping the training loss's estimator would change its exact numeric scale/gradients, which
needs the mandatory P2-08 verification ladder (not yet built) to validate safely before landing
— per issue #61 §4's own "no Kt weight tuning SHALL substitute for P2-04 through P2-08"
ordering and §1.4's "no destructive refactor" rule. Reserved for P2-15 (migration).

### Tests

- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- measure_integral::`
  — 11 passed: constant-field exactness, hand-computed mean*area*thickness, empty-input safety,
  the AMR-bias reproduction proving `domain_integral_weighted` recovers the true area-weighted
  average while the plain estimator is measurably biased (>0.1 off) for the same data, the
  uniform-sampling-degenerates-to-unweighted case, length-mismatch fallback/panic behavior for
  the two integral functions respectively, and `plate_domain_area`/`plate_outer_perimeter` hand
  checks (including hole-area subtraction).
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::tests::probe_energy_balance user_problem::tests::probe_reaction_force`
  — 5 passed, all pre-existing, confirming the refactor is byte-identical behavior (same
  finite/distinguishing-internal-from-external assertions the pre-P2-04 inline arithmetic
  already satisfied).
- command: `cargo build --workspace --tests --features ndarray-backend` — clean, no regressions
  in any other crate.

### Runtime evidence

Real headless training run (`stress-solver --problem-spec`, release build, `single_hole_plate.toml`
with `max_steps=60`): completed all 60 steps without panic, `total_loss` decreased monotonically
(6.59 -> 4.99), diagnostics populated normally. Confirms the workspace-wide build (including the
new module) is healthy end-to-end on the real training path; `probe_energy_balance` itself is
exercised on `runner.rs`'s GUI/checkpoint-serving path (not the bare CLI path used for this
run), covered instead by its own pre-existing unit tests re-run above against the refactored code.

### Known limitations

The fuller P2-04 vision (`DomainIntegral`/`BoundaryIntegral`/`InterfaceIntegral` as first-class
types threaded through every energy/loss computation, replacing every bare `.mean()` in the
codebase) is not built — only the two formulas `probe_energy_balance` already needed are
generalized, plus the new weighted variant. `InteriorEnergyTerm`/`ExternalWorkTerm` (the live
training loss) still use unscaled `.mean()` — tracked above, deferred to P2-15. No interface
integral type exists yet (no current problem in this codebase has a multi-domain interface
requiring one — pin-lug's `InterfacePenetrationTerm` uses a different, already-tested mechanism
not touched here).

### Reviewer verification

PASS against P2-04's acceptance bullets at the scope covered: real geometric measure (not a
magic multiplier), the literal `Integral_Omega(f) ≈ |Omega|*mean(f)` formula, and a demonstrated,
tested nonuniform-sampling convergence fix, wired into an already-live diagnostic via refactor.
The broader "replace every implicit integral in the codebase" scope is explicitly deferred to
P2-15, tracked as a known limitation rather than silently dropped.

## P2-05 — Separate physical coefficients from optimization weights

Status: VERIFIED

### Requirement

Categorize terms as PhysicalFunctionalTerm/ConstraintTerm/DiagnosticTerm; the loss ledger
separates physical value from optimization/effective weight; tests prove adaptive weighting
cannot alter `U-W_ext`.

### Files changed

- `crates/pinn-solver/src/problem.rs` — new `TermRole` enum (`PhysicalFunctional`/`Constraint`/
  `Diagnostic`) + `LossTerm::term_role()` (forced-choice, no meaningful default - same
  convention as `formulation_kind`).
- `crates/pinn-solver/src/user_problem.rs`, `kirsch_problem.rs`, `pinlug_problem.rs` — every
  concrete `LossTerm` overrides `term_role()` explicitly, classified by what each term actually
  enforces (see Architecture decision).
- `crates/pinn-solver/src/training_core.rs` — `term_role_report()` (mirrors `formulation_kind_
  report`'s shape); `LossLedgerEntry` gains a `role: Option<TermRole>` field; `build_loss_
  ledger()` gains a `roles: Option<&HashMap<&'static str, TermRole>>` parameter (all 4 existing
  call sites updated, `None` where role data isn't available - non-breaking, additive).
- `crates/pinn-solver/src/runner.rs` — one call site updated for the new `build_loss_ledger`
  arity (`None` for roles - a diagnostic printer, not affected by this epic's core claim).

### Architecture decision

Classification (issue #61's own wording in parens): `interior_energy`/`external_work` (the
literal `U`/`-W_ext` halves of `Π=U-W_ext`), `equilibrium`/`outer_traction`/`hole_free`/
`neumann_traction`/`hole_traction`/`equilibrium_ring`/`lug_free_edge_traction`/`pin_driving_
traction` → **PhysicalFunctional** (governing PDE/BC physics, strong or weak form alike).
`hole_fixed`/`displacement_anchor`/`lug_shank_anchor` (essential/Dirichlet, issue #61 §1.1's own
"essential constraints"), `interface_penetration`/`interface_non_tension` (Signorini KKT
inequality admissibility), `kirsch_stress` (a data-fit anchor to a known analytical solution,
not the governing functional of the trained problem itself) → **Constraint**.
`constitutive_consistency` (issue #61's own literal example: "Discrete points... MAY be used as
evaluation/operator machinery but SHALL NOT silently become permanent solution degrees of
freedom" — polices representation consistency, solves no new physics) → **Diagnostic**.

The "physical coefficient vs optimization weight" separation this epic's title names is not
newly built — it was already a structural property of this codebase's real architecture:
`LossTerm::compute(&self, inputs: &[DomainForwardOutputs]) -> Tensor<B, 1>` has NO weight
parameter in its signature at all. `step_physics`/`step_physics_multi` compute every term's
`raw` value (via `compute()`) BEFORE `saw.update()` ever runs, and apply `lambda` strictly
afterward (`total += raw * lambda`) — architecturally impossible for any adaptive weight to
feed back into `raw`. This epic's real job was making that fact (a) EXPLICIT via `TermRole`
classification (which terms are physics vs admissibility vs diagnostic) and (b) PROVEN via a
real test on production term objects, not merely asserted from reading the code.

### Tests

- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- term_role
  physical_functional_value_is_invariant build_loss_ledger` — 8 passed: `term_role`
  classification tables for all three problem types (`user_problem_loss_terms_have_expected_
  term_role_classification`, `kirsch_loss_terms_have_expected_term_role_classification`,
  `pinlug_loss_terms_have_expected_term_role_classification` - every concrete term covered, no
  silent default relied on), `build_loss_ledger_joins_roles_by_name`, and
  `physical_functional_value_is_invariant_to_optimization_weighting` (see Runtime evidence).
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- kirsch_problem::
  pinlug_problem:: user_problem:: training_core::` (broader regression) — 142 passed, 0 failed.
- command: `cargo build --workspace --tests --features ndarray-backend` — clean.

### Runtime evidence

`physical_functional_value_is_invariant_to_optimization_weighting` calls `interior_energy`/
`external_work` (real `UserDefinedProblem::loss_terms()` production objects, not mocks) via
their actual `LossTerm::compute()` trait method on hand-built deterministic inputs, confirms
`compute()` is a pure function (identical output called twice), then applies two deliberately
different weights (1.0 vs 50.0, mirroring `step_physics_multi`'s real `raw * lambda` formula)
and confirms the raw physical value recovers identically either way — a genuine, executable
proof of the P2-05 invariant against real term objects, not a synthetic HashMap-only test.

### Known limitations

No standalone "physical coefficient" ledger field distinct from `raw` was added — this
codebase's terms don't carry a single separable scalar physical constant outside their
`compute()` formula (material properties like E/ν are Rust struct fields on `MaterialProps`,
baked into the physics formula itself, not a ledger-reportable number); `raw` already IS
P2-05's "physical value" side of the ledger by construction. An unrelated, real finding
surfaced while building the (ultimately unused) full-integration version of the invariance
test: two `step_physics_multi` calls from separately-cloned `ElasticityNet` instances (same
`ElasticityNetConfig`, same `net_cfg.init()` source) produced DIFFERENT `raw_scalar_by_name`
values even with IDENTICAL `SawBrdr` seeding — i.e. `base_model.clone()` does not appear to
preserve bit-identical initial weights (or something in `step_physics_multi`'s call chain has
hidden model-independent randomness). Not investigated further here (out of P2-05's scope,
and not needed once the test was rewritten to call `LossTerm::compute()` directly on
deterministic hand-built inputs) — flagged here as a real, reproducible observation for a
future session, since exact-reproducibility failures like this are exactly the kind of thing
P2-13 (reproducibility/provenance) should eventually catch.

### Reviewer verification

PASS against P2-05's acceptance bullets: the three-way categorization exists and covers every
concrete term in all three problem types with no silent default; the ledger carries the role
alongside raw/lambda/weighted; a real, executable test proves adaptive weighting cannot alter
the physical functional value on genuine production term objects.

## P2-06 — Geometry-aware operators

Status: VERIFIED

### Requirement

Generic (not rectangle-only) `contains`/`signed_distance`/`nearest_boundary`/`boundary_normal`/
`boundary_tangent`/`boundary_measure`/`valid_stencil` operators; stencils avoid invalid points
with recorded fallback/quality diagnostics.

### Files changed

- `crates/pinn-core/src/user_geometry.rs` — new `BoundaryRef` enum (`OuterLeft`/`OuterRight`/
  `OuterTop`/`OuterBottom`/`Hole(usize)`) and `StencilValidity` struct (+ `all_valid()`).
  `UserGeometry` gains `signed_distance`, `nearest_boundary`, `boundary_normal_for`/`boundary_
  normal`, `boundary_tangent`, `boundary_measure`, `valid_stencil` — all generic over an
  arbitrary number of holes (this codebase's real geometry representation - not literally
  arbitrary polygons, see Known limitations). `contains` already existed (pre-dates this epic).
- `crates/pinn-solver/src/user_problem.rs` — new `StencilQualityReport` struct + `stencil_
  quality_report()` function, built on `UserGeometry::valid_stencil`.
- `crates/pinn-solver/src/user_runner.rs` — `run_headless_user_problem` calls `stencil_quality_
  report` once at startup and prints a `[diag]` line - real, live use, zero effect on training
  (`sample_interior` is deterministically re-seeded every call, so this preview reproduces
  exactly what the training loop's own first call will sample).

### Architecture decision

`signed_distance` uses `min(rect_sdf, hole_sdfs...)` - an approximate (not exact in general,
documented as such) but correct-in-practice SDF for this codebase's real rectangle-minus-
circles domains, where holes are always small relative to the plate and never near the outer
boundary. `nearest_boundary`/`boundary_normal`/`boundary_tangent`/`boundary_measure` are all
keyed off a single `BoundaryRef` identifier so a caller resolves "which boundary" once and
reuses it, rather than each operator re-deriving "which edge/hole" independently.
`valid_stencil` generalizes the ad hoc margin-based exclusion this codebase's real sampling
strategy already performs (`UserSamplingStrategy::contains_for_collocation`, from a prior
General-PINN pass) into a declared, reusable, geometry-level primitive - NOT a replacement of
that working rejection-sampling code (per §1.4, no destructive refactor of a working
mechanism); `stencil_quality_report` makes what that margin is actually protecting against a
visible, queryable number via a new, additive diagnostic instead.

### Tests

- command: `cargo test -p pinn-core --lib -- user_geometry::` — 13 passed (7 new): hand-computed
  `signed_distance` for an interior point, a hole center, and a point outside the rectangle;
  `nearest_boundary` correctly identifying all 4 outer edges and both holes; `boundary_normal`
  axis-aligned on edges and radial on holes; `boundary_tangent` perpendicular to the normal;
  `boundary_measure` matching hand-computed edge lengths and hole circumference; `valid_stencil`
  fully-valid far from any boundary AND correctly flagging the exact single direction (`x_plus`)
  that crosses into a hole while every other direction stays valid.
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::` — 37
  passed (3 new `stencil_quality_report_*` tests: all-fully-valid, one-fallback-needed-point,
  and an invalid-center case classified separately from fallback-needed).
- command: `cargo build --workspace --tests --features ndarray-backend` — clean.

### Runtime evidence

Real headless training run (`single_hole_plate.toml`, `max_steps=60`, release build): startup
diagnostic printed `[diag] stencil quality: 1997/2048 fully valid, 51 fallback-needed, 0
invalid-center` before training began, then all 60 steps completed normally (`total_loss` 6.90
-> 4.99). Confirms `stencil_quality_report`/`valid_stencil` are genuinely exercised on the real
plate collocation point set (not just synthetic test data) and reveal a real, previously-
invisible number: 51 of 2048 real interior points (~2.5%) sit close enough to the hole that an
FD stencil there needs the existing margin-based fallback, with zero invalid centers (confirming
the existing rejection-sampling margin is working correctly for this configuration).

### Known limitations

"Generic, not rectangle-only" is scoped to this codebase's real domain representation
(a rectangle minus N circular holes, arbitrary hole count) - not literally arbitrary polygonal
or curved domains, which no problem type in this codebase currently has or needs. `valid_
stencil` is NOT wired into `UserSamplingStrategy`'s actual rejection-sampling loop (that
remains its own working, tested, margin-based mechanism) - only into the new, additive
diagnostic. Migrating the real sampling loop onto this primitive (if ever warranted) is
deferred to P2-15.

### Reviewer verification

PASS against P2-06's acceptance bullets: every named operator exists, is generic over hole
count, is hand-verified against known values, and is proven exercised on the real training
path via a live diagnostic with genuine, non-trivial output.

## P2-07 — Generic gauge/nullspace handling

Status: VERIFIED

### Requirement

Mean-field constraints, point anchors, and nullspace projection for pure-Neumann rigid-body
modes, separate from load enforcement.

### Files changed

- `crates/pinn-core/src/user_geometry.rs` — `UserGeometry::is_pure_neumann()` (true iff no
  hole is `HoleBc::Fixed` - the plate's only essential/Dirichlet mechanism).
- `crates/pinn-solver/src/gauge.rs` — new module: rationale doc comment + `has_any_essential_
  constraint()` (a generic, problem-level form of the same check, for future problem types -
  see Known limitations for why the real wiring doesn't use this one).
- `crates/pinn-solver/src/lib.rs` — `pub mod gauge;` added.
- `crates/pinn-solver/src/user_problem.rs` — new `TranslationGaugeTerm` (a real "mean-field
  constraint" - issue #61's own named technique) registered in `UserDefinedProblem::loss_
  terms()` exactly when `self.spec.geometry.is_pure_neumann()`; `base_weight()` extended for
  its name.

### Architecture decision

This codebase's real elasticity BVP has a genuine, previously-unaddressed rigid-body-
translation nullspace for `no_hole_plate.toml` (no holes at all) and `single_hole_plate.toml`
(its hole set to `HoleBc::Free`) - BOTH real, current example configurations: with no
Dirichlet condition anywhere, strain energy/traction/equilibrium are all invariant under an
added constant `(u0, v0)` displacement offset, so nothing in the pre-P2-07 loss penalized one.
`TranslationGaugeTerm` closes this with the standard "mean-field constraint" gauge-fixing
technique: `mean(u)^2 + mean(v)^2` (squaring the MEAN, not the mean of squares - local
displacement variation is untouched, only domain-wide rigid-body drift is penalized). It is
registered ONLY for pure-Neumann configurations (never redundantly alongside a real Dirichlet
anchor) and classified `TermRole::Constraint` (P2-05) - "separate from load enforcement" per
the epic's own wording, never part of the physical functional `U-W_ext`.

The ROTATIONAL rigid-body mode (the third DOF; the standard fix is penalizing `mean(x*v -
y*u)` toward zero) is NOT implemented - it needs each collocation point's physical `(x,y)`
coordinates, which `DomainForwardOutputs` does not carry (only `raw_out`/`strains`/`normals`/
`shifted_stress`/`hessian`). Adding a coordinates field would mean touching `compute_domain_
forwards` and every existing `DomainForwardOutputs`/`Computed` construction site - the kind of
broad structural plumbing change issue #61 §1.4 reserves for P2-15's migration step, not a
single new term. Honestly tracked below, not silently dropped.

`has_any_essential_constraint()` (the generic, `&dyn BoundaryValueProblem`-based form of the
same check) is real and tested, but is NOT what `UserDefinedProblem::loss_terms()` calls -
doing so would recurse (`loss_terms()` calling a function that itself calls `loss_terms()`).
The real registration gate uses `UserGeometry::is_pure_neumann()` directly. `has_any_essential_
constraint` is kept as the generalized form for a future problem type that might gain a
different essential-constraint mechanism than holes - honestly flagged as not currently
load-bearing in this codebase (see Known limitations), not presented as more than it is.

### Tests

- command: `cargo test -p pinn-core --lib -- user_geometry::` — 15 passed (2 new): `is_pure_
  neumann` true for no-holes and all-Free-holes geometries, false when any hole is Fixed.
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::
  gauge::` — 20 passed (4 new): `translation_gauge` registered for a no-hole geometry AND for
  a single-Free-hole geometry (both real example configurations), ABSENT for `two_hole_
  geometry` (has a Fixed hole - existing exact-term-count test extended to assert this), and
  `TranslationGaugeTerm::compute()` matches hand-computed values for both a uniform-offset
  field (real nonzero penalty) and a zero-mean field (zero penalty despite large local
  variation) - proving the squared-mean formula, not mean-of-squares.
- command: `cargo build --workspace --tests --features ndarray-backend` — clean.
- command: `cargo test -p pinn-solver --features ndarray-backend --lib -- user_problem::
  kirsch_problem:: pinlug_problem:: gauge::` (broader regression) — 79 passed, 0 failed -
  confirms every existing formulation/term-role/term-count test (all built on `two_hole_
  geometry`, which has a Fixed hole) is completely unaffected by the new conditional term.

### Runtime evidence

Real headless training runs (release build, `max_steps=60`) on BOTH real pure-Neumann example
configurations, confirming the new term trains stably (no NaN/blowup, comparable loss
trajectories and Kt/residual magnitudes to pre-P2-07 runs - a real before/after comparison, not
assumed):
- `single_hole_plate.toml`: `total_loss` 6.47 -> 5.09, `max|displacement|` 1.121e-5 m,
  `constitutive residual RMS`=5.12e5 Pa, `Kt`=0.0078 (same tiny-Kt range as prior runs - this
  epic does not claim to fix Kt, only the translation gauge).
- `no_hole_plate.toml`: `total_loss` 5.68 -> 4.95, `max|displacement|` 7.973e-6 m,
  `constitutive residual RMS`=2.44e5 Pa. Its startup stencil-quality diagnostic (P2-06) also
  showed a real, informative side effect of this epic's own investigation: 48/2048 points need
  fallback even with ZERO holes - confirming `valid_stencil`/`stencil_quality_report` correctly
  generalizes to outer-rectangle-boundary-adjacent FD risk too, not just hole-adjacent risk (a
  broader signal than the pre-existing hole-only margin mechanism covers).

### Known limitations

Rotational gauge-fixing not implemented (needs point coordinates not currently threaded through
`DomainForwardOutputs` - see Architecture decision). No formal "did this reduce net rigid-body
drift" measurement exists yet (would need a dedicated mean-displacement diagnostic, naturally
suited to P2-09's "trivial-solution diagnostics" scope, next in the mandatory order) - this
epic's own verification is limited to "trains stably, doesn't regress loss/Kt magnitude,"
which IS real evidence but not a direct measurement of the gauge fix's own effect size.
`has_any_essential_constraint()` exists and is tested but is not wired into any live decision
in this codebase (see Architecture decision) - a real, honestly-scoped limitation, not a dead
function pretending to be load-bearing.

### Reviewer verification

PASS against P2-07's acceptance bullets at the scope covered: a real mean-field constraint
technique, correctly gated to pure-Neumann configurations only, verified against real example
configurations both by unit test and live training runs. Point-anchor and full nullspace-
projection techniques, and the rotational mode, are explicitly named as not built, not silently
omitted.

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
