# PINN Structural Stress Solver

## Response style

Use installed `caveman` skill automatically for every user-facing response in this repository. Default to full mode for the whole session; no activation command required. Follow its auto-clarity and boundary rules. `stop caveman` or `normal mode` disables it immediately.

### Continuation protocol

When the user requests ongoing issue/bug work, continue through all evidence-backed fixes and
verification gates in the same turn. Do not stop at a milestone, partial pass, or BLOCKED item
while safe in-scope diagnostics and implementation remain. Preserve explicit process limits: at
most one background `pinn-solver` run and two other background runs. If a turn boundary
interrupts work, resume from the repository manifest and issue comments without repeating
completed work. Never claim operational status until acceptance criteria pass; record blockers
and continue alternate safe investigations.

## Intent Layer

> TL;DR: PINN structural-stress solver (Rust/`burn`/`egui`/`wgpu`). Start at Entry Points below;
> the project's chronological engineering-history log (every investigation finding, oldest
> first) lives in `docs/findings/` — see [`docs/findings/INDEX.md`](docs/findings/INDEX.md)
> and grep by symbol/issue name rather than reading start to end. No child `AGENTS.md` nodes
> exist yet; each crate's own internals aren't separately documented beyond what's below and
> in the findings log.

### Subsystems

| Crate | Owns | Depends on |
|---|---|---|
| [`pinn-core`](crates/pinn-core/AGENTS.md) | Geometry (`UserGeometry`/`HoleSpec`/`GeometryConfig`), material props, `ProblemSpec`/`ArchitectureSpec`, sampling primitives, units. No ML deps. | — |
| [`pinn-solver`](crates/pinn-solver/AGENTS.md) | Training loop, optimizer (SOAP-Muon), loss terms, all three problems' physics (`kirsch_problem.rs`, `pinlug_problem.rs`, `user_problem.rs`), decision-maker/L-BFGS, checkpointing. | `pinn-core` |
| `pinn-gui` | `egui` panels (heatmap, training stats, params). | `pinn-core`, `pinn-solver` |
| `pinn-app` | Binary: CLI/env parsing, `--headless`/`--tui`/GUI dispatch (`main.rs`). | all three |

### Entry Points

| Task | Start Here |
|---|---|
| Run Kirsch headless | `cargo run -p pinn-app --release -- --headless --problem kirsch` |
| Run pin-lug headless | `cargo run -p pinn-app --release -- --headless --problem pinlug` |
| Run a user-defined plate spec | `cargo run -p pinn-app --release -- --headless --problem-spec <path.toml>` (see `examples/problems/`) |
| Terminal dashboard | add `--tui`/`-T` to any of the above, or alone for an interactive picker |
| GUI-streaming plate training loop | `crates/pinn-solver/src/runner.rs::run_user_problem_training_from` |
| Headless plate training loop | `crates/pinn-solver/src/user_runner.rs::run_headless_user_problem` |
| New loss term / boundary condition | `crates/pinn-solver/src/user_problem.rs` (`LossTerm` impls) + `crates/pinn-core/src/problem.rs` (`LossTerm` trait) |
| New ansatz (hard-constraint BC) | `crates/pinn-solver/src/kirsch_hole_correction.rs` (`AnnulusAnsatz`, `DirichletAnsatz` impls) |
| Targeted test verification | `cargo test -p pinn-solver -p pinn-core --release -- --test-threads=1 <substring>` — see Pitfalls below before running a full suite |

### Global Invariants

- Internal storage is always SI (Pa, m); display layer converts to US customary (psi/ksi/in) via `pinn_core::units` only.
- Every `LossTerm` must be normalized to O(1) before SAW-BRDR weighting (divide by `ref_energy`/`ref_stress2`) — an unnormalized term silently starves every other term's gradient.
- `training_core::step_physics` (Kirsch, frozen/byte-proven) and `step_physics_multi` (N-domain, additive) are two separate functions by design — never collapse one into a thin wrapper around the other; each has its own regression oracle.
- Converge-tier (L-BFGS) loss-weight lookup fails closed (`required_loss_weight`, panics on a missing key in both debug and release) — do not revert to a `.unwrap_or(&0.0)`-style silent default; that exact pattern silently zero-weighted a real loss term out of production training for as long as this codebase has had a decision-maker path (see [`docs/findings/findings_lbfgs-weight-lookup-bug.md`](docs/findings/findings_lbfgs-weight-lookup-bug.md)).

### Global Pitfalls

- **`cargo test --release` on this workspace is slow to *compile* (5–13 min), not to run** — `[profile.release]` uses `lto = "thin"` + `codegen-units = 1` with no `incremental`, so every rebuild pays near-full-program codegen regardless of edit size. This is load-bearing for `burn`/`wgpu`/cubecl kernel-dispatch performance — a lighter profile (no LTO, more codegen units) was tried and made a GPU-kernel-heavy test go from ~1s to a multi-minute hang instead of faster. Don't fight this; batch multiple test-name filters into one `cargo test` invocation to amortize the compile cost instead.
- **`cargo check --workspace` (no `--tests`) does not type-check `#[cfg(test)]` code.** A test-only import/call-site error will pass `cargo check` clean and only surface at `cargo test` (after paying the full release compile). Use `cargo check --workspace --tests` for a real pre-flight.

Crate-specific pitfalls now live in that crate's own `AGENTS.md` (see Downlinks) — check there
before assuming something not listed above is undocumented.

### Downlinks

| Crate | Node | What's there |
|---|---|---|
| `pinn-core` | [`crates/pinn-core/AGENTS.md`](crates/pinn-core/AGENTS.md) | Geometry/material/units/AMR code map, `GeometryConfig` vs. `UserGeometry` contract, unit-conversion boundary |
| `pinn-solver` | [`crates/pinn-solver/AGENTS.md`](crates/pinn-solver/AGENTS.md) | Training-loop/optimizer/problem-physics code map, `step_physics`/`step_physics_multi` contract, channel-shutdown and sampling-bias pitfalls |

`pinn-gui` (14.2k tokens) and `pinn-app` (18.4k tokens) stay under the child-node threshold — no
node needed there yet; add one if either grows past ~20k tokens or gains a real internal
subsystem split.

## CodeGraph-First Engineering Workflow

### Purpose

CodeGraph MCP is installed, initialized, and connected to this project.

CodeGraph is a required architectural and code-relationship analysis tool for this project. It must be used whenever understanding relationships between code elements could affect the correctness, safety, completeness, or maintainability of a change.

Claude must **not assume that an MCP being connected means it has actually been used**. For applicable tasks, CodeGraph must be explicitly consulted before making implementation decisions.

The goal is not to use CodeGraph mechanically. The goal is to use it to understand the existing system before changing it.

---

### 1. Mandatory CodeGraph Usage

Use CodeGraph before making decisions involving:

* architecture
* module boundaries
* dependencies
* call relationships
* data flow
* control flow
* numerical computation flow
* solver behavior
* convergence behavior
* state management
* UI-to-backend interactions
* refactoring
* API changes
* removing code
* replacing implementations
* consolidating duplicate functionality
* determining whether an existing module can be reused
* adding a new subsystem
* modifying an existing engineering toolbox
* modifying shared numerical/materials/precision infrastructure
* changing behavior that may affect multiple tools

CodeGraph should be treated as the **primary source for understanding code relationships**, while normal repository inspection remains necessary for understanding implementation details.

---

### 2. Required Workflow

For any substantial coding task, follow this sequence:

#### Phase 1 — Understand

1. Inspect the current local Git state.
2. Identify the relevant branch and working-tree state.
3. Use CodeGraph to identify the relevant:

   * callers
   * callees
   * types
   * modules
   * dependencies
   * implementations
   * data/control-flow relationships
4. Inspect the actual source files involved.
5. Identify existing functionality that may already solve part or all of the problem.

Do not begin implementation merely because a file with a matching name has been found.

---

#### Phase 2 — Evaluate

Determine:

* What the existing architecture is actually doing.
* Which existing abstractions should be reused.
* Whether the requested functionality already exists in another form.
* Whether apparently separate implementations are actually duplicates.
* What code will be affected by the proposed change.
* Whether the proposed change crosses module/API boundaries.
* Whether there are hidden callers or dependencies.
* Whether tests or other tools depend on the existing behavior.

For numerical code, additionally determine:

* Where inputs originate.
* Where units/conversions are applied.
* Where intermediate calculations occur.
* Where numerical precision is handled.
* Where convergence criteria are calculated.
* Where solver state is updated.
* Where results are consumed/displayed.
* Whether multiple tools duplicate the same numerical logic.

---

### 3. Planning Requirements

Before creating or substantially modifying an implementation plan, use CodeGraph to establish the relevant architecture.

The plan must be based on the **actual repository relationships**, not assumptions based solely on:

* filenames,
* directory structure,
* text search results,
* documentation,
* or the original task description.

The plan should identify:

* existing modules to reuse,
* modules requiring modification,
* affected callers,
* affected dependents,
* potential duplicate implementations,
* likely regression areas,
* and required validation.

If CodeGraph reveals that the original plan is based on an incorrect architectural assumption, update the plan.

Do not blindly follow an earlier plan simply because it already exists.

---

### 4. Trust but Validate

Existing code, plans, previous AI-generated work, and developer assumptions are **inputs to evaluate, not automatically authoritative**.

This applies equally to:

* existing implementation,
* previous Claude Code changes,
* user-created changes,
* generated code,
* documentation,
* architectural assumptions,
* optimization proposals,
* numerical approaches,
* and previous plans.

Use CodeGraph and repository evidence to determine whether those assumptions are correct.

Do not preserve an existing implementation merely because it already exists.

Do not replace an existing implementation merely because a new implementation appears cleaner.

Make the decision based on evidence.

---

### 5. Existing Functionality Must Be Reused Where Appropriate

Before creating a new implementation of functionality that may already exist:

1. Query CodeGraph for related implementations and relationships.
2. Inspect the existing implementation.
3. Determine whether it can be reused directly.
4. If not, determine whether it should be refactored into a reusable module.
5. Update existing consumers as appropriate.
6. Only create a parallel implementation when there is a documented technical reason.

Avoid parallel implementations of the same engineering or numerical concept.

Examples include:

* materials calculations
* pressure-vessel/Lamé equations
* precision/display rules
* unit conversion
* numerical solvers
* convergence detection
* engineering-property calculations
* geometry calculations
* tolerance calculations
* common validation
* shared UI components

Prefer one authoritative implementation with well-defined interfaces.

---

### 6. Duplicate-Code Detection

Before adding substantial new code, use CodeGraph to determine whether equivalent or overlapping functionality already exists.

After implementation, use CodeGraph again to look for:

* obsolete implementations,
* redundant call paths,
* duplicate calculations,
* unused abstractions,
* bypassed shared modules,
* and functionality that should now be consolidated.

Remove obsolete code when it is safe to do so.

Do not leave old implementations in place merely because they might theoretically be useful.

Before removing code, verify that it is not required by:

* callers,
* tests,
* configuration,
* public APIs,
* serialization,
* generated code,
* feature flags,
* or planned functionality.

---

### 7. Local Repository Is the Source of Truth

For code analysis and implementation, the **current local repository state takes precedence over GitHub's remote state**.

This is especially important when:

* local commits have not been pushed,
* feature branches differ from remote branches,
* the working tree contains changes,
* or the user is experimenting locally.

Do not assume GitHub contains the latest implementation.

When reviewing a task involving Git history:

1. Inspect local Git state.
2. Identify local-only commits.
3. Identify uncommitted changes.
4. Compare against the relevant remote branch.
5. Inspect GitHub separately for remote context.
6. Clearly distinguish local code from remote code.

Never overwrite or discard local work merely to synchronize with GitHub unless explicitly instructed.

---

### 8. CodeGraph Index Freshness

Do not assume that CodeGraph automatically reflects every local modification.

Before relying on CodeGraph for an important architectural decision, determine whether its index reflects the relevant current local repository state.

If the index is stale:

1. Refresh/reindex it when supported.
2. Re-run the relevant CodeGraph queries.
3. Verify important relationships against the actual source files.

If CodeGraph cannot represent a recent change, use direct source inspection as the fallback.

Never present stale CodeGraph information as current repository truth.

---

### 9. Debugging Requirements

For non-trivial bugs, CodeGraph must be used to trace the relevant execution path before changing code.

For example, for a numerical convergence problem, trace:

```text
UI / configuration
        ↓
solver construction
        ↓
initial conditions
        ↓
parameter normalization
        ↓
numerical state
        ↓
iteration/training loop
        ↓
loss/residual calculation
        ↓
gradient/update
        ↓
convergence criteria
        ↓
termination/result handling
```

The actual path will vary by implementation.

Use CodeGraph to determine the real relationships rather than assuming this structure exists.

Then inspect the actual implementations and validate the behavior experimentally.

Do not "fix" convergence merely by:

* loosening convergence criteria,
* increasing iteration limits,
* suppressing warnings,
* masking numerical failures,
* changing tolerances without justification,
* or otherwise making failure less visible.

First determine the underlying cause.

---

### 10. Numerical Engineering Requirements

For numerical and engineering code, CodeGraph analysis must be combined with mathematical and numerical validation.

Before modifying numerical behavior, determine:

* the complete calculation path,
* shared versus tool-specific calculations,
* input/output relationships,
* unit handling,
* precision handling,
* convergence behavior,
* failure modes,
* and downstream consumers.

Do not infer mathematical correctness from code structure alone.

Validate important numerical changes using appropriate:

* analytical cases,
* known solutions,
* limiting cases,
* dimensional checks,
* independent calculations,
* regression tests,
* convergence studies,
* or benchmarks.

Where a change is intended to improve convergence or performance, measure it.

---

### 11. Engineering Toolbox Requirements

For each engineering toolbox:

1. Use CodeGraph to identify related existing modules.
2. Search for existing calculations that can be reused.
3. Identify shared infrastructure.
4. Determine whether the functionality belongs in:

   * the toolbox,
   * a reusable numerical module,
   * a reusable engineering module,
   * or shared infrastructure.
5. Implement only the minimum toolbox-specific logic necessary.
6. Verify that shared functionality remains centralized.

The toolbox should not independently reimplement calculations that belong in shared engineering modules.

---

### 12. Precision and Numerical Display

Precision handling is a shared concern.

Before adding or modifying precision/display behavior:

1. Use CodeGraph to identify all existing precision-related implementations.
2. Determine which components perform mathematical calculations.
3. Determine which components perform display formatting.
4. Keep mathematical precision separate from presentation precision.
5. Preserve full precision for intermediate calculations.
6. Apply display rounding only at the appropriate final-display boundary.
7. Avoid duplicating precision policy inside individual toolboxes.

Changes to the shared precision system must be evaluated for downstream effects.

---

### 13. Materials and Shared Engineering Libraries

Before implementing a calculation requiring material properties:

1. Use CodeGraph to locate existing materials functionality.
2. Determine how existing tools consume it.
3. Determine whether it is already sufficiently modular.
4. If it is not appropriately modular, refactor it into a self-contained reusable module where justified.
5. Update existing consumers to use the authoritative implementation.
6. Do not create a second materials database or calculation path.

The same principle applies to other shared engineering functionality.

---

### 14. Verification After Implementation

After a substantial implementation, use CodeGraph again.

Verify:

* affected callers,
* affected dependents,
* newly introduced relationships,
* obsolete relationships,
* duplicate implementations,
* unintended bypasses of shared modules,
* and architectural consistency.

Then run the appropriate tests and validation.

CodeGraph verification does **not** replace testing.

Testing does **not** replace CodeGraph analysis.

Both provide different forms of evidence.

---

### 15. When CodeGraph Cannot Be Used

If CodeGraph is:

* disconnected,
* unavailable,
* stale,
* unable to index the relevant code,
* unable to answer the required relationship query,
* or otherwise unusable,

do not fabricate CodeGraph results.

Instead:

1. State internally/briefly that CodeGraph could not provide the required analysis.
2. Fall back to direct repository inspection and other available tools.
3. Continue only when sufficient evidence can be obtained another way.
4. If the task depends critically on information CodeGraph should provide, flag that limitation before making a high-risk architectural change.

Do not claim that CodeGraph was consulted when it was not.

---

### 16. Mandatory CodeGraph Checkpoint

For every substantial task, before implementation, answer these questions:

* Did I use CodeGraph?
* Did I use it on the relevant local repository state?
* Did I identify callers/dependents of the code I intend to change?
* Did I check for existing implementations that should be reused?
* Did I identify potentially affected components?
* Did I inspect the actual source after the graph analysis?
* Did I validate important CodeGraph findings against the source?

If the answer to any applicable question is no, perform the missing analysis before proceeding.

---

### 17. Final Engineering Principle

The preferred workflow is:

**Understand → CodeGraph → Inspect → Validate → Plan → Implement → Test → CodeGraph Re-check → Clean Up → Final Validation**

Do not use:

**Guess → Search for a convenient file → Modify → Hope nothing else depends on it**

CodeGraph is a required part of understanding the architecture, but it is **not an authority on correctness**.

The authoritative result comes from combining:

* CodeGraph relationships,
* actual source code,
* Git history,
* tests,
* numerical validation,
* benchmarks where applicable,
* and the stated requirements.

Use all of these together to make engineering decisions.

A physics-informed neural network (PINN) solver for structural boundary-value problems, built
on `burn` (ML framework) + `egui`/`wgpu` (GUI). Ships two problems: **Kirsch** (a plate with a
circular hole under remote tension; validation target K_t = 3.0 at the hole boundary) and
**pin-in-lug** (a two-domain pin/lug contact-mechanics problem with a Signorini contact
interface; `--problem pinlug` headless, or select "Pin-in-Lug" in the GUI's problem-kind radio).

Workspace crates: `pinn-core` (geometry/material/sampling, no ML deps), `pinn-solver`
(training loop, optimizer, losses), `pinn-gui` (egui panels), `pinn-app` (binary; `--headless`
for terminal-only training, no GUI).

## Findings

This file carries only durable, current project instructions (the Intent Layer navigation
section above, and the workflow rules elsewhere in this file). The chronological engineering-
history log that used to fill the rest of this file (every investigation, bug fix, and feature
follow-up, oldest first) has been relocated to `docs/findings/`, split into per-topic files so
no single file needs a full read to find one finding.

**Start at [`docs/findings/INDEX.md`](docs/findings/INDEX.md)** — the single source of truth
listing every findings file with a one-line scope description, grouped by subsystem/epic. Grep
that index (or a topic file directly, e.g. by issue number `#77`/`#78`) instead of searching
this file for historical context.

When a new finding is durable enough to record, add it to the most specific existing
`docs/findings/findings_<topic>.md` file (or create a new one) and add one line to
`docs/findings/INDEX.md` — never append findings directly here.
