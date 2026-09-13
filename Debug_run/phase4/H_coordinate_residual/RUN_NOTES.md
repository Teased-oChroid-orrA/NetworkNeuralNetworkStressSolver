# Issue #64 H — coordinate-residual control

Purpose: compare trainable raw-coordinate displacement residual against completed F MLP-only
control without changing geometry, objective, sampling, seed, optimizer schedule, or benchmark.

Command:

```text
cargo run -p pinn-app --release --features ndarray-backend -- --headless --problem-spec Debug_run/phase4/H_coordinate_residual/problem.toml
```

Acceptance remains unchanged: all L4 hard thresholds, energy-health, and independent field
validation must pass. This artifact is diagnostic until then.

## Result

Completed 1,000 steps. The residual reduced optimization loss rapidly and L4 health passed
(`energy_balance_error=2.7854e-4`), but it did not recover the physical field. Hard L4 failed:
`sigma_xx_relative_error=0.1525`, `sigma_yy_over_ref=0.0410`,
`sigma_xy_over_ref=0.0239`, `traction_rms_over_ref=0.1231`, and
`load_transfer_ratio=0.8049`. The matched F MLP-only control had `load_transfer_ratio=0.8838`
and `sigma_xx_relative_error=0.1196`; coordinate residual is therefore not winning and must not
be extended. Full stdout is `solver.log`.
