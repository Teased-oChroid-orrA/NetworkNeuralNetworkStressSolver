Comprehensive progress / failure update

Implemented:

- `ElasticityNet` has a zero-initialized trainable raw-coordinate `[x_norm, y_norm, z] -> [u, v]` residual.
- Production `fwd` and `fwd_masked` retain raw coordinates alongside optional Fourier MLP input. Residual changes displacement only; direct stress remains MLP-only.
- Embedded-input `forward` remains compatible.
- Skip weight/bias participate in optimizer gradients. Width/depth growth and pruning preserve branch.
- New checkpoints persist branch; legacy records load with zero skip.

Code proof:

- `cargo test -p pinn-solver --features ndarray-backend checkpoint::tests -- --test-threads=1`: 3 passed.
- Focused interior sampling safety tests: 2 passed.
- `cargo check -p pinn-app --features ndarray-backend`: passed.
- WGPU-only network tests cannot run on host: no Metal adapter.

Controlled runtime H:

- Fixed seed / Variational `Pi=U-W_ext` / `fd_h=1e-2` / 512 interior / 256 boundary / AMR off / 1,000 steps.
- Health passes: `energy_balance_error=2.7854e-4`; hard L4 fails all five metrics.
- `sigma_xx_relative_error=0.1525`; `sigma_yy_over_ref=0.0410`; `sigma_xy_over_ref=0.0239`; `traction_rms_over_ref=0.1231`; `load_transfer_ratio=0.8049`.
- Earlier F MLP control: `sigma_xx_relative_error=0.1196`, `load_transfer_ratio=0.8838`; H is not winning and is not extended.

New diagnosis / experiment:

- H reaches normalized `Pi=-1.028`, below exact affine continuum `-1`, while independent field metrics worsen. Evidence points to sampled-functional underintegration / neural between-sample exploitation, not valid continuum solution.
- Replaced random base interior samples with geometry-generic deterministic stratification plus existing rejection fallback for holes. Focused hole-safety tests pass.
- Stratified rerun has emitted startup only; no `step 0` or final ledger. It is not evidence.

Open blockers:

1. No corrected Variational L4 pass.
2. No independent field-validation pass.
3. No AMR, single-hole/Kt, non-square, or multiple-hole acceptance runs; all gated by L4.
4. Do not close #64 or promote #63 operational status.

Artifacts: `Debug_run/phase4/H_coordinate_residual/`. Local record: `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`.
