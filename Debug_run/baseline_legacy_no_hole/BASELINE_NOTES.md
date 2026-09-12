# PH3-01 baseline: legacy Hybrid no-hole run

Frozen for issue #62 PH3-01 ("Establish the production no-hole baseline"). This directory is
immutable evidence — do not overwrite its contents in later PH3-XX work; later runs go in their
own `Debug_run/measure_aware_no_hole/`, `Debug_run/variational_no_hole/`, `Debug_run/final_no_hole/`
directories per the plan's own §22.

## Provenance

- **Baseline frozen at main SHA**: `21a0e730634251f7d7a564d3f78550390bda69f8` (the commit that
  merged this real run's uploaded artifacts and PH3-00's scaffolding into `main`).
- **The run's own recorded `git_sha`** (in `model.meta.json`'s `provenance` block) is
  `af92d02c908444415ecb514a8685b5e55ef68506` — CONFIRMED (during PH3-02) to be the `powershell_
  tool` repo's own commit SHA immediately before this pass's PH3-02 commit (`git log` there
  showed `af92d02..ccd59e1` on push) — i.e. it's `git rev-parse HEAD` run inside the GUI's own
  `powershell_tool` working directory at save time (per `provenance.rs`'s own doc comment), NOT
  this `NeuralNetwork-Stress-Solver` repository's SHA. This is a real, structural provenance gap
  (not a bug in this specific run): a checkpoint saved by the GUI records the WRONG repo's
  commit for "which solver code produced these weights" whenever the two repos' histories
  diverge, since `NeuralNetwork-Stress-Solver` is consumed as a path dependency, not vendored.
  Not fixed in this pass (out of PH3-02's scope) - worth revisiting in a future provenance
  epic if exact solver-commit traceability from a checkpoint ever becomes load-bearing.
- **Formulation**: `Hybrid(interior_energy, equilibrium, outer_traction, external_work)` —
  confirmed directly from `model.meta.json`. This is the "legacy" path issue #62 §2.1.B
  identifies (not pure Variational DEM).
- **Derivative backend**: `FD` (finite difference) — confirmed from `provenance.derivative_backend`.
- **Model init seeded**: `false` — confirmed from `provenance.model_init_seeded` (issue #61
  P2-13's own real negative finding, still true here).
- **Interior sampling seed**: `90210` (`SEED_INTERIOR`).
- **Backend**: `NdArray`, `dtype: f32`.

## Files in this directory

- `model.mpk.gz` / `model.meta.json` — the exact trained weights + checkpoint metadata, loadable
  via `pinn_solver::checkpoint::load_checkpoint(Path::new(".../baseline_legacy_no_hole/model"), &device)`.
- `stress_solver_report.json` — the GUI-generated engineering report for this exact run (contains
  `"model_validity": null` — the literal gap issue #62 §2.1.A calls out).
- `no_hole_plate.toml` — the example problem spec used (`examples/problems/no_hole_plate.toml` at
  freeze time; copied here so the baseline is self-contained even if the example file changes
  later).
- `VonMisses-Stress HeatMap.png` — the GUI's rendered heatmap for this run.

## Hard benchmark result (the gap this baseline exists to document)

`crates/pinn-solver/src/user_problem.rs::tests::ph3_01_baseline_legacy_no_hole_checkpoint_benchmark_result`
loads this exact checkpoint and runs the real `run_no_hole_benchmark` (issue #61 P2-14) against
it — the actual hard-threshold PASS/FAIL judgment that `stress_solver_report.json`'s
`"model_validity": null` never captured. See `docs/PHASE_3_IMPLEMENTATION_MANIFEST.md`'s PH3-01
entry for the recorded result. Run it yourself with:

```sh
cargo test --release -p pinn-solver --features ndarray-backend ph3_01_baseline_legacy_no_hole -- --ignored --nocapture
```

## Reproducibility

This baseline is reproducible FROM CONFIGURATION (`no_hole_plate.toml`, same material/load/
network/training parameters) but NOT bit-for-bit from a fresh run, because:

1. Model weight initialization is not seeded (`model_init_seeded: false`) — a different random
   initialization will be drawn each time `cargo run --release -- --headless --problem-spec
   examples/problems/no_hole_plate.toml` is invoked (issue #61 P2-13's own recorded limitation,
   PH3-11's job to address).
2. Interior/boundary sampling IS seeded (`SEED_INTERIOR = 90_210`), so that part of a fresh run
   is deterministic.

The checkpoint file itself, however, is an exact, immutable, byte-identical artifact — the
benchmark result computed from it is fully reproducible by re-running the test above.
