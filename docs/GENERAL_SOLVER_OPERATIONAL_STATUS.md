# General Solver Operational Status

Current status: **NOT_OPERATIONAL** for generalized Variational/DEM support.

Reason: PH4 found and corrected an invalid independently weighted U/W objective. Corrected
Variational runtime evidence, L4, displacement/strain validation, FieldKind/DifferentialOperator
enforcement, L5 gating, generalized topology regressions, and complete objective provenance are
not yet verified.

Latest corrected-objective D run is finite but non-converged: L4 fails all five hard metrics,
load-transfer ratio is `0.1996`, and prescribed-work energy-balance error is `1.7482`. This is
evidence against calling the corrected Variational path operational, not a tuning target.

PH4 now persists physical `U`, full prescribed `W_ext`, `Pi`, normalized values, active terms,
constraints, base/adaptive weights, geometry measures, estimator, and optimizer policy through
the shared plate checkpoint/GUI report producer. FieldKind resolution and FD production
derivative routing are executable and tested. This implementation is not runtime-verified
until a corrected Variational checkpoint is produced.

Legacy Hybrid no-hole artifact is not generalized operational evidence. Its persisted L4 result
fails `sigma_xx_relative_error` and `traction_rms_over_ref` at the current 1% thresholds.

Latest controls: restricted neural capacity, `3e-3` peak LR, and fourfold uniform resolution
all fail L4. The 1,000-step F convergence run ended at step 100 before final evidence; partial
data is explicitly excluded from acceptance.

Independent no-hole field validator now has exact-affine and rigid-translation unit proof. It
remains acceptance-blocked until corrected Variational L4 runtime evidence exists.
