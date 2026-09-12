# Phase 3 — Operationalization, Live-Path Migration & 100% Verification Plan

Source: GitHub issue #62, `Teased-oChroid-orrA/NetworkNeuralNetworkStressSolver`. Copied verbatim
here as the durable, version-controlled record of this phase's mandate (issue text can be edited
or closed; this file is the source of truth once work begins per PH3-01).

## Purpose

This plan is the **post–Phase 2 operationalization epic** for:

`Teased-oChroid-orrA/NetworkNeuralNetworkStressSolver`

It is based on the current `main` branch, the completed Phase 2 implementation/manifest, and the newest `Debug_run` no-hole result.

The objective is not to add another layer of architecture.

The objective is to make the architecture already built in Phase 2 the **actual production execution path**, prove the no-hole solver is physically correct, and establish an auditable definition of "100% operational."

---

# 1. Current verified state

## 1.1 Phase 2 status

The repository reports:

- P2-01 through P2-14: `VERIFIED`
- P2-15: `PARTIALLY VERIFIED`

The remaining P2-15 work is explicitly identified as three live-path migrations:

1. `InteriorEnergyTerm` / `ExternalWorkTerm` onto the measure-aware integral machinery.
2. `EquilibriumTerm` and other derivative consumers onto `DifferentialOperator`.
3. Stress/strain consumers onto the `FieldKind` authoritative-field funnel.

Those migrations were deliberately deferred because they change the live training objective and are higher-risk.

**Phase 3 SHALL complete those migrations.**

---

# 2. Latest no-hole run assessment

The latest `Debug_run/stress_solver_report.json` is substantially better than the earlier collapsed solutions.

Reported:

```text
nominal applied stress       = 69.000 MPa
average von Mises            = 69.144 MPa
maximum von Mises             = 71.882 MPa
maximum principal stress     = 71.808 MPa

energy balance error         = 0.00080166  (~0.080%)
internal energy              = 6.66492 J
external work                = 6.65958 J

boundary residual RMS         = 0.9677 MPa  (~1.40% of nominal)
boundary residual maximum    = 3.0625 MPa

PDE residual RMS              = 0.1827 MPa  (~0.265% of nominal)
PDE residual P95              = 0.3181 MPa

AMR residual RMS:
    before                   = 0.167123 MPa
    after                    = 0.166409 MPa

max displacement              = 123.446 um

net whole-boundary Fx        = 57.17 N
net whole-boundary Fy        = 14.06 N
reference force               = 69,000 N

training steps                = 1,999 / 2,000
final gradient norm            = 0.4839
```

The current energy balance is very good and the stress magnitude has recovered dramatically.

However, this run SHALL NOT be declared "100% operational" yet.

## 2.1 Why the run is not yet accepted

### A. The persisted report does not contain the hard benchmark result

The report contains:

```json
"model_validity": null
```

and does not persist the hard P2-14 `NoHoleBenchmarkResult` / load-transfer ratio.

The benchmark machinery exists, but the saved engineering artifact does not prove the run passed it.

### B. The run used Hybrid, not the intended pure variational DEM path

The saved model metadata declares:

```text
Hybrid(
  interior_energy,
  equilibrium,
  outer_traction,
  external_work
)
```

Therefore this run does not prove that the **pure variational / Deep Energy Method** execution path is fully operational.

### C. Boundary acceptance remains above the Phase 2 target

The Phase 2 target was:

```text
traction RMS / sigma_ref < 1%
```

Current:

```text
0.9677 / 69 = ~1.40%
```

Therefore the current run fails that specific acceptance criterion.

### D. Displacement is not yet sufficiently validated

For the analytical no-hole field:

```text
u = (sigma/E) * x
v = -nu * (sigma/E) * y
```

the maximum corner displacement magnitude for the current material/load/geometry is approximately:

```text
101.3 um
```

The reported maximum is:

```text
123.45 um
```

This is approximately 22% higher than the analytical corner magnitude.

Stress is much closer than displacement, so this needs investigation rather than being dismissed as harmless.

### E. The live training path still contains legacy computation

Phase 2 intentionally left the live:

```text
InteriorEnergyTerm
ExternalWorkTerm
compute_domain_forwards / FD call sites
stress/strain consumer selection
```

on legacy execution paths.

Therefore Phase 2 created and verified the new architecture, but the production solver is not yet fully running through it.

### F. Reproducibility is still incomplete

Phase 2 correctly records the real negative finding:

```text
model_init_seeded = false
derivative_backend = FD
```

The new run therefore cannot yet be considered fully reproducible at the model-initialization level.

### G. AMR refinement is demonstrably active, but convergence is weak

The latest AMR sweep changed:

```text
points: 484 -> 772
residual RMS: 0.167123 MPa -> 0.166409 MPa
```

The improvement is only approximately 0.43%.

That does not prove AMR is wrong, but it does mean AMR is not currently strong evidence of meaningful convergence.

---

# 3. Non-negotiable Phase 3 rules

Claude Code SHALL obey these rules.

## 3.1 No benchmark hacks

Do not add:

- stress rescaling;
- displacement correction;
- Kt correction;
- geometry-specific weighting;
- hard-coded load-transfer compensation;
- special no-hole logic;
- "acceptable" threshold changes solely to make the current run pass.

## 3.2 No silent formulation change

Changing:

```text
Hybrid -> Variational
```

is a mathematical change.

It SHALL be explicit in configuration, provenance, logs, and test coverage.

## 3.3 Do not remove the legacy path prematurely

Legacy paths SHALL remain available behind an explicit compatibility switch until:

1. new path is validated;
2. same benchmark passes;
3. regression evidence exists;
4. the default is intentionally changed;
5. legacy removal is separately verified.

## 3.4 No "verified" from unit tests alone

Every live-path migration requires:

```text
unit test
+
integration test
+
real solver run
+
persisted report evidence
```

## 3.5 The NN remains the solution representation

No FEM/FDM replacement is permitted.

---

# 4. Phase 3 architecture target

The production execution path SHALL become:

```text
ProblemSpec
   |
   v
FormulationSelection
   |
   v
Neural Network
   |
   v
Primary Fields
   |
   v
FieldKind dependency graph
   |
   v
DifferentialOperator policy
   |
   v
Derived strain/stress
   |
   v
Measure-aware integration
   |
   +-----------------------------+
   |                             |
   v                             v
Physical functional         Constraints/diagnostics
   |                             |
   +-------------+---------------+
                 |
                 v
          Optimization
                 |
                 v
       Verification ladder
                 |
                 v
       Persisted benchmark
             evidence
```

There SHALL NOT be a second hidden path that bypasses these abstractions.

---

# 5. PH3-01 — Establish the production no-hole baseline

Before changing live numerical code:

1. freeze the current no-hole configuration;
2. record current `main` SHA;
3. run the benchmark with the current path;
4. persist:
   - full benchmark result;
   - load-transfer ratio;
   - all threshold results;
   - formulation;
   - derivative backend;
   - sampling/AMR state;
   - provenance.

Create a baseline artifact:

```text
Debug_run/baseline_legacy_no_hole/
```

The baseline SHALL remain immutable.

Acceptance:

- baseline can be reproduced from source/config;
- baseline report is complete;
- every metric is traceable to code/config.

---

# 6. PH3-02 — Persist the hard benchmark result

The solver SHALL persist the actual P2-14 benchmark result in the final report.

At minimum:

```json
"benchmark": {
  "level": "L4",
  "name": "no_hole",
  "passed": false,
  "sigma_xx_relative_error": ...,
  "sigma_yy_over_reference": ...,
  "sigma_xy_over_reference": ...,
  "traction_rms_over_reference": ...,
  "load_transfer_ratio": ...,
  "thresholds": {...},
  "failure_reasons": [...]
}
```

Do not allow:

```json
"model_validity": null
```

to represent an otherwise completed benchmark run.

The persisted result SHALL be the exact result used to decide PASS/FAIL.

---

# 7. PH3-03 — Add a machine-enforced operational gate

A production run SHALL finish in one of these states:

```text
PASS
FAIL
INVALID
```

Never:

```text
unknown
null
not evaluated
```

unless the run genuinely terminated before the benchmark could execute.

For a completed no-hole training run:

```text
L0 must pass
L1 must pass
L2 must pass
L3 must pass
L4 hard benchmark must pass
```

If any prerequisite fails, the final status SHALL identify the failed rung.

---

# 8. PH3-04 — Migrate measure-aware integration into the live loss

## Current gap

Phase 2 created and tested:

```text
domain_integral
domain_integral_tensor
boundary_integral_tensor
```

but the live `InteriorEnergyTerm` / `ExternalWorkTerm` still use legacy mean-based computation.

## Required migration

Introduce an explicit configuration switch:

```text
measure_aware_training = false
```

initially preserving old behavior.

Then implement:

```text
measure_aware_training = true
```

using the real geometry and sampling measures.

For the variational functional:

```text
Pi = U - W_ext
```

the live computation SHALL use the same measure-aware machinery already proven by L0/L3.

## Validation

Run the exact same no-hole problem:

```text
legacy
vs
measure_aware
```

using the same:

- problem;
- architecture;
- seed;
- sampling;
- optimizer;
- max steps.

Both reports must be saved.

## Acceptance

The measure-aware path becomes eligible for default use only when the hard no-hole benchmark passes.

---

# 9. PH3-05 — Run the no-hole problem as pure Variational DEM

Create an explicit production benchmark configuration:

```text
formulation = Variational
measure_aware_training = true
```

For this configuration the training objective SHALL contain:

```text
internal energy
-
external work
+
declared essential/gauge constraints
```

It SHALL NOT contain:

```text
equilibrium penalty
outer traction penalty
duplicate natural traction penalty
```

unless explicitly declared by a Hybrid formulation.

The existing equilibrium/traction probes SHALL remain available for diagnostics.

---

# 10. PH3-06 — Migrate DifferentialOperator into live derivative consumers

Route live derivative calls through the Phase 2 operator policy.

At minimum migrate:

```text
compute_domain_forwards
EquilibriumTerm
strain consumers
other derivative call sites identified by repository-wide call-site audit
```

The implementation SHALL preserve numerical behavior under the legacy-default policy before the backend is changed.

Acceptance:

- a code-level test proves the live consumer invoked the selected backend;
- provenance records the actual backend used;
- no direct bypass remains in migrated consumers.

---

# 11. PH3-07 — Complete the authoritative FieldKind funnel

Every live stress/strain consumer SHALL request its source through the field graph.

Audit:

```text
energy
equilibrium
traction
constitutive consistency
BC
visualization
reaction
engineering results
QoI
Kt
checkpoint/export
```

For every consumer record:

```text
requested field
resolved field
dependency chain
```

No consumer may directly choose between:

```text
network stress
derived stress
```

outside the field graph.

---

# 12. PH3-08 — Resolve displacement/stress discrepancy

The current stress field is close to target, but maximum displacement is significantly above the analytical no-hole result.

Investigate in this order:

1. displacement output scaling;
2. coordinate normalization/de-normalization;
3. boundary sign conventions;
4. gauge contribution;
5. stress-vs-displacement consistency;
6. Poisson contraction;
7. edge/corner evaluation;
8. FD approximation effects;
9. network approximation error.

Do not "fix" displacement by multiplying it by a correction factor.

## Acceptance

For the no-hole analytical solution:

```text
u(x,y) ≈ (sigma/E)x
v(x,y) ≈ -nu(sigma/E)y
```

report:

```text
max absolute u error
max absolute v error
RMS u error
RMS v error
corner displacement error
```

The final report SHALL prove displacement independently of stress.

---

# 13. PH3-09 — Close the no-hole boundary acceptance gap

Current:

```text
traction RMS ≈ 1.40% sigma_ref
```

Target:

```text
< 1%
```

Use diagnostics to determine whether the error is:

- model approximation;
- FD stencil error;
- boundary sampling error;
- measure/integration error;
- formulation error;
- optimizer convergence issue.

Do NOT change the threshold.

## Acceptance

At least one of these SHALL occur:

```text
traction RMS / sigma_ref < 1%
```

or a mathematically justified, configuration-dependent refinement mechanism demonstrably drives it below 1% without changing the acceptance threshold.

---

# 14. PH3-10 — Validate optimizer/convergence behavior

The current run reached:

```text
1999 / 2000 steps
final gradient norm ≈ 0.484
```

This does not by itself prove optimization convergence.

Add convergence evidence:

```text
loss slope
gradient norm trend
field/QoI trend
benchmark trend
energy balance trend
load-transfer trend
```

A run SHALL NOT be declared converged merely because:

```text
step == max_steps
```

or because a loss plateau detector stopped it.

---

# 15. PH3-11 — Strengthen reproducibility

Phase 2 discovered that model initialization is not seeded.

Phase 3 SHALL either:

1. make model initialization deterministic under the recorded seed, OR
2. explicitly classify training as non-reproducible and record that limitation.

Preferred implementation:

```text
problem seed
+
sampling seed
+
network initialization seed
+
optimizer state seed/state
```

must be captured.

Run the same configuration at least twice.

Acceptance:

```text
same seed + same config
=> numerically identical or documented deterministic tolerance
```

If exact determinism is impossible because of backend behavior, the report SHALL state:

- source of nondeterminism;
- expected tolerance;
- observed divergence.

---

# 16. PH3-12 — Validate AMR as a convergence accelerator, not a cosmetic feature

The current AMR sweep reduced residual RMS only slightly.

Add a controlled comparison:

```text
fixed sampling
vs
AMR
```

for the same training budget.

Compare:

```text
stress error
traction error
energy balance
load transfer
displacement error
final objective
```

AMR is beneficial only if it improves a physically relevant metric, not merely its own refinement indicator.

---

# 17. PH3-13 — Benchmark report must become authoritative

The final report SHALL contain all of:

```text
run identity
git SHA
dirty state
problem hash
config hash
seed(s)
formulation
derivative backend
integration mode
sampling mode
AMR state

L0 result
L1 result
L2 result
L3 result
L4 no-hole benchmark
L5 hole benchmark, when eligible

field accuracy
traction accuracy
load transfer
energy consistency
displacement accuracy
stress consistency
```

The GUI/headless runner, persisted JSON, and checkpoint metadata SHALL agree.

No duplicate reporting logic may produce contradictory statuses.

---

# 18. PH3-14 — Complete the variational acceptance test

The affine amplitude test already proves the mathematical `U-W` machinery in isolation.

Phase 3 SHALL now prove:

```text
NN
 -> displacement
 -> strain
 -> constitutive stress
 -> measure-aware U-W
 -> optimized NN
```

recovers the same analytical no-hole state.

This is the critical bridge between:

```text
L0 analytic functional
```

and:

```text
full NN solver
```

---

# 19. PH3-15 — Hole/Kt activation gate

The hole benchmark SHALL NOT be considered operational merely because Kt can be computed.

It becomes eligible only after:

```text
no-hole hard benchmark = PASS
AND
variational path = PASS
AND
measure-aware path = PASS
AND
authoritative field audit = PASS
AND
derivative-path audit = PASS
```

Then run the hole case.

Kt SHALL report:

```text
stress projection
reference stress
peak angle
angular refinement
radial offset refinement
finite/infinite-domain reference classification
```

---

# 20. PH3-16 — Cross-configuration regression matrix

At minimum test:

| Case | Strong | Variational | Hybrid |
|---|---:|---:|---:|
| No-hole | required | required | required |
| Hole | required where supported | required | required |
| Manufactured | required | required | supported |
| AMR | required | required | required |

Each configuration SHALL prove that only its declared objective terms are active.

---

# 21. PH3-17 — Remove legacy paths only after proof

After all migrations:

1. keep compatibility switch;
2. make the new path default;
3. run regression suite;
4. run benchmark suite;
5. verify persisted reports;
6. remove legacy path only after those results are stable.

Deletion is the final step, not the implementation strategy.

---

# 22. Mandatory files/artifacts

Create:

```text
docs/PHASE_3_OPERATIONALIZATION_PLAN.md
docs/PHASE_3_IMPLEMENTATION_MANIFEST.md
Debug_run/baseline_legacy_no_hole/
Debug_run/measure_aware_no_hole/
Debug_run/variational_no_hole/
Debug_run/final_no_hole/
```

Each run directory SHALL include:

```text
stress_solver_report.json
model.meta.json
configuration copy
```

and any required heatmaps/diagnostics.

---

# 23. Mandatory implementation manifest

For every PH3 item:

```markdown
## PH3-XX

Status: NOT_STARTED | PARTIAL | IMPLEMENTED | VERIFIED | BLOCKED

### Current evidence
...

### Required change
...

### Files changed
...

### Tests
...

### Runtime run
...

### Benchmark result
...

### Known limitations
...

### Reviewer verification
PASS | FAIL | NOT REVIEWED
```

Only `VERIFIED` counts.

---

# 24. Definition of 100% operational

The solver SHALL be called **100% operational** only when all conditions below are true.

## Architecture

- [ ] Phase 2 abstractions are the live execution path.
- [ ] no live energy term bypasses measure-aware integration when enabled.
- [ ] live derivative consumers use `DifferentialOperator`.
- [ ] live field consumers use the authoritative `FieldKind` path.
- [ ] formulation selection fully controls the training objective.

## Mathematical correctness

- [ ] L0 affine amplitude passes.
- [ ] no-hole NN variational solve passes.
- [ ] energy balance is within acceptance.
- [ ] traction RMS is below the hard threshold.
- [ ] load-transfer ratio passes.
- [ ] displacement field passes independent analytical verification.
- [ ] constitutive stress and derived stress agree.

## Numerical reliability

- [ ] convergence is demonstrated by field/QoI trends, not max-step count alone.
- [ ] AMR does not introduce estimator bias.
- [ ] backend choice is explicit.
- [ ] reproducibility status is honest and tested.

## Benchmarking

- [ ] no-hole result is persisted as explicit PASS/FAIL.
- [ ] hole/Kt cannot be accepted without a valid no-hole prerequisite.
- [ ] Kt convergence is demonstrated.
- [ ] finite/infinite-domain references are distinguished.

## Software quality

- [ ] workspace build passes.
- [ ] workspace tests pass except for explicitly documented unrelated flakes.
- [ ] clippy passes with no new warnings.
- [ ] no benchmark-specific hacks exist.
- [ ] no dead Phase 2 abstraction remains in the intended production path.
- [ ] manifests are complete.

---

# 25. Final directive to Claude Code

**Do not optimize the present debug number.**

The current run proves that the architecture has moved the solver from the old collapsed state toward the correct physical stress magnitude. It does **not** prove that the implementation is finished.

The remaining task is:

```text
Phase 2 abstractions
        ↓
actual live training path
        ↓
pure mathematical formulation
        ↓
measure-correct integration
        ↓
authoritative field graph
        ↓
verified derivatives
        ↓
converged neural solution
        ↓
machine-verifiable benchmark PASS
        ↓
only then hole/Kt
```

The final solver must be able to take a new problem and use the same architecture without relying on:

- plate-specific compensation;
- manually chosen magic multipliers;
- hidden loss terms;
- duplicate physics;
- alternate stress representations;
- undocumented backend choices;
- benchmark-specific corrections.

**The finish line is not "the stress looks right."
The finish line is a reproducible neural PDE solver whose mathematics, execution path, diagnostics, and benchmark acceptance all agree.**
