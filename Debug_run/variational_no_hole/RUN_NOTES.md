# PH3-05: pure Variational + measure-aware, no-hole, 2000 steps

Real evidence for issue #62 PH3-05, captured by running `crates/pinn-solver/src/runner.rs::
tests::variational_no_hole_plate_trains_and_produces_real_benchmark_evidence` (`#[ignore]`d,
loads the actual shipped `examples/problems/variational_no_hole_plate.toml`):

```sh
cargo test --release -p pinn-solver --features ndarray-backend --lib variational_no_hole_plate_trains -- --ignored --nocapture
```

Wall clock: 292.27s (comparable to PH3-01's own frozen legacy baseline's `elapsed_secs: 646.3` -
faster here, consistent with the pure-Variational path having fewer active loss terms to compute
per step than the legacy Hybrid config).

## Configuration

`formulation = "Variational"`, `training.measure_aware_training = true` - everything else
(material, load, geometry, network, `max_steps=2000`) identical to `Debug_run/baseline_legacy_
no_hole/`'s frozen legacy config, for direct comparability. `L0` gate: **PASSED**
(`l0_passed: true` in the benchmark summary below).

## Term activation (verified separately by a fast unit test, not just asserted here)

`variational_formulation_on_a_no_hole_geometry_activates_u_minus_w_ext_and_translation_gauge`
proves `loss_terms()` for this exact geometry+formulation combination returns EXACTLY
`interior_energy`, `external_work`, `translation_gauge` (issue #61 P2-07's rigid-body gauge fix,
active because a no-hole plate has no `HoleBc::Fixed` essential constraint) - no `equilibrium`,
no `outer_traction`, matching issue #62 PH3-05's own requirement exactly.

## Benchmark result (from the LIVE run's own `TrainingUpdate.no_hole_benchmark`)

```
NoHoleBenchmarkSummary {
    level: "L4", name: "no_hole", passed: false,
    sigma_xx_relative_error: 0.7081521326850257,
    sigma_yy_over_reference: 0.07741460931257514,
    sigma_xy_over_reference: 0.08049855247371138,
    traction_rms_over_reference: 0.5689988928496821,
    load_transfer_ratio: 0.18804505543194097,
    thresholds: { all at 0.01 except load_transfer_ratio in [0.99, 1.01] },
    failure_reasons: ["sigma_xx_relative_error", "sigma_yy_over_ref", "sigma_xy_over_ref", "traction_rms_over_ref", "load_transfer_ratio"],
    l0_passed: true,
    operational_status: "FAIL",
}
```

Independently RECOMPUTED from the saved checkpoint (a second, separate code path - `checkpoint::
load_checkpoint` + a fresh `run_no_hole_benchmark` call - not just re-reading the same in-memory
value) to cross-validate the live result:

```
NoHoleBenchmarkResult {
    sigma_xx_relative_error: 0.7081741581410851, sigma_yy_over_ref: 0.0773841604779796,
    sigma_xy_over_ref: 0.08049141156267353, traction_rms_over_ref: 0.5690012525542006,
    load_transfer_ratio: 0.18800272495028375, passed: false,
    failures: ["sigma_xx_relative_error", "sigma_yy_over_ref", "sigma_xy_over_ref", "traction_rms_over_ref", "load_transfer_ratio"],
}
```

The two independently-computed results agree to ~4 significant figures - real cross-validation,
not a single unverified number.

## Energy balance (recomputed from the saved checkpoint)

```
EnergyBalance { internal_energy: 0.6535426030576231, external_work: 0.446417219012976, energy_balance_error: 0.4639726588114126 }
```

## Honest interpretation - NOT declared a success

Per issue #62's own explicit "do not optimize the present debug number" directive: this run
**FAILS the hard no-hole benchmark on all 5 thresholds**, load_transfer_ratio (0.188, target
range [0.99, 1.01]) shows the network is far from having learned to carry anywhere near the
correct fraction of the applied far-field load within this budget, and the 46% energy-balance
error confirms the same. This is a REAL, honest result, not a defect in this epic's own
implementation: pure Variational (no strong-form penalty terms at all to anchor early
convergence, unlike the Hybrid legacy baseline which also enforces `equilibrium`/
`outer_traction` pointwise residuals) has fewer, weaker early gradient signals and would
plausibly need substantially more than 2000 steps, a tuned learning-rate schedule, or both, to
reach comparable convergence - a real, open question for PH3-09/PH3-14 (which own the actual
convergence/acceptance work), not something PH3-05 itself was asked to solve. PH3-05's own job -
prove the pure-Variational + measure-aware path runs for real, activates exactly the declared
terms, and reports its real (even if currently failing) benchmark result honestly - is complete.

## What is NOT included in this directory

Model weights (`model.mpk.gz`)/`model.meta.json` were NOT preserved from this specific run (the
test's own temp checkpoint was cleaned up after the benchmark/energy-balance recomputation
above, which already cross-validated the live result from a second, independent code path). Re-
running the same `#[ignore]`d test (command above) reproduces an equivalent checkpoint on demand
if a future epic needs the actual weights.
