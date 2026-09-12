# Phase 3 Implementation Manifest

Tracks GitHub issue #62 ("Phase 3 — Operationalization, Live-Path Migration & 100% Verification
Plan"), governed by `docs/PHASE_3_OPERATIONALIZATION_PLAN.md`. Per that plan's own §23: only
`VERIFIED` counts. Updated after each PH3-XX item, not batched at the end. Strict order
PH3-01 -> PH3-17 per the plan's own mandate (mirrors issue #61's own P2-01 -> P2-15 ordering
discipline, which this repository already validated).

## PH3-01 — Establish the production no-hole baseline

Status: VERIFIED

### Current evidence
The repository already contained a real, freshly-uploaded legacy no-hole run at `Debug_run/`
(singular — the user's own upload, merged from `origin/main` at the start of this item) whose
`stress_solver_report.json`/`model.meta.json` match issue #62 §2's own reported numbers exactly
(69.144 MPa avg von Mises, 0.080% energy balance error, 123.446 um max displacement, Hybrid
formulation, `model_validity: null`). This IS the exact run the plan describes — no need to
re-run training to reproduce it.

### Required change
Freeze that run into an immutable `Debug_run/baseline_legacy_no_hole/` directory (plan §5/§22),
and — critically — actually COMPUTE the hard P2-14 benchmark against it, since the persisted
report's `model_validity: null` (plan §2.1.A) proves the benchmark was never run against this
checkpoint, only an unrelated parametric-surrogate "model validity" concept (`self.infer_result`
in `app-egui/src/stress_solver.rs` — confirmed by reading that code: `model_validity` there is a
`ValidityTier` GREEN/YELLOW/RED classification of a parametric inference QUERY, not the P2-14
`NoHoleBenchmarkResult` at all. A real, load-bearing architectural finding for PH3-02.

### Files changed
- `Debug_run/baseline_legacy_no_hole/` (new, immutable): `model.mpk.gz`, `model.meta.json`,
  `stress_solver_report.json`, `no_hole_plate.toml`, `VonMisses-Stress HeatMap.png`,
  `BASELINE_NOTES.md` (provenance/reproducibility notes), `benchmark_result.json` (real captured
  benchmark evidence, see below).
- `crates/pinn-solver/src/user_problem.rs`: new `#[ignore]`d test
  `ph3_01_baseline_legacy_no_hole_checkpoint_benchmark_result` — loads the frozen checkpoint via
  `checkpoint::load_checkpoint` and runs the real `run_no_hole_benchmark` against it.

### Tests
`cargo test --release -p pinn-solver --features ndarray-backend --lib ph3_01_baseline_legacy_no_hole -- --ignored --nocapture`
— 1 passed. `#[ignore]`d (reads a fixed repo-relative checkpoint path, not a self-contained unit
test), consistent with this codebase's existing convention for real-artifact-reading tests.

### Runtime run
The test above IS the runtime evidence — it loads the real, previously-trained checkpoint
(2000-step Hybrid run) and executes the real `run_no_hole_benchmark` function against it (no
mocking, no re-training). Captured output:
```
NoHoleBenchmarkResult {
    sigma_xx_relative_error: 0.019238057970042754,
    sigma_yy_over_ref: 0.008142509555433572,
    sigma_xy_over_ref: 0.004616288049931893,
    traction_rms_over_ref: 0.014040773330894896,
    load_transfer_ratio: 1.0027142837981733,
    passed: false,
    failures: ["sigma_xx_relative_error", "traction_rms_over_ref"],
}
```

### Benchmark result
`passed: false`. Two of five thresholds fail (all thresholds are 1%):
- `sigma_xx_relative_error = 1.924%` (> 1% max) — **a real failure the issue text itself did not
  call out** (issue #62 §2.1.C only names traction RMS); found here because this item actually
  ran the benchmark rather than trusting the narrative description.
- `traction_rms_over_ref = 1.404%` (> 1% max) — confirms issue #62 §2.1.C's own reported ~1.40%
  figure exactly.
- `sigma_yy_over_ref = 0.814%`, `sigma_xy_over_ref = 0.462%`, `load_transfer_ratio = 1.0027`
  (within [0.99, 1.01]) — all PASS.

Full result persisted to `Debug_run/baseline_legacy_no_hole/benchmark_result.json`.

### Known limitations
- The checkpoint's own recorded `provenance.git_sha` (`af92d02c9...`) predates and does not
  match the SHA this baseline was frozen at (`21a0e7306...`) — the run was produced by the
  separate `powershell_tool/app-egui` GUI binary's own working-directory `git rev-parse HEAD` at
  save time, not necessarily this solver repository's checkout at that instant. Documented in
  `BASELINE_NOTES.md` rather than silently picking one SHA as authoritative.
- Model weight initialization is not seeded (`model_init_seeded: false`, issue #61 P2-13's own
  finding, still real here) — this exact checkpoint is an immutable, byte-identical artifact, but
  a fresh re-run of `no_hole_plate.toml` would draw different initial weights. PH3-11's job.
- This baseline uses the legacy Hybrid formulation and legacy (non-measure-aware) integration —
  by design, this is what "legacy baseline" means; PH3-04/PH3-05 build the comparison points.

### Reviewer verification
PASS — baseline is real (not synthesized), immutable, reproducible from configuration, and its
hard benchmark result is now actually computed and persisted (closing the exact
`model_validity: null` gap PH3-02 still needs to fix everywhere else, e.g. in the GUI's own live
report-writing path).

## PH3-02 — Persist the hard benchmark result

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-03 — Machine-enforced operational gate (PASS/FAIL/INVALID)

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-04 — Migrate measure-aware integration into the live loss

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-05 — Run the no-hole problem as pure Variational DEM

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-06 — Migrate DifferentialOperator into live derivative consumers

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-07 — Complete the authoritative FieldKind funnel

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-08 — Resolve displacement/stress discrepancy

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-09 — Close the no-hole boundary acceptance gap

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-10 — Validate optimizer/convergence behavior

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-11 — Strengthen reproducibility

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-12 — Validate AMR as a convergence accelerator

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-13 — Benchmark report becomes authoritative

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-14 — Complete the variational acceptance test (NN bridge)

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-15 — Hole/Kt activation gate

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-16 — Cross-configuration regression matrix

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED

## PH3-17 — Remove legacy paths only after proof

Status: NOT_STARTED

### Current evidence
### Required change
### Files changed
### Tests
### Runtime run
### Benchmark result
### Known limitations
### Reviewer verification
NOT REVIEWED
