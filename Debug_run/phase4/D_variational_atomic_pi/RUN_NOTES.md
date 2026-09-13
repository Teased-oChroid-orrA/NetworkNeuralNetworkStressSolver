# PH4 controlled runtime: atomic, measure-aware Variational Pi

Command:

```text
cargo run -p pinn-app --features ndarray-backend -- --headless --problem-spec Debug_run/phase4/D_variational_atomic_pi/problem.toml
```

Result: completed 15/15 steps with finite loss, decreasing from `14.08062` to `13.76781`.
The mandatory affine L0 gate passed with relative error `1.039e-7`.

This is PH4-06 ladder stage 3 only: full NN, uniform sampling, AMR off, corrected atomic
measure-aware Pi. It is not an L4 attempt and is not accepted as a benchmark.

Observed diagnostic values after 15 steps:

```text
max displacement: 4.940e-5 m
energy balance error: 1.7482 (post-PH4-05 prescribed-traction probe)
load transfer ratio: 0.1996
L4: FAIL
sigma_xx error: 1.1882
sigma_yy/reference: 0.1726
sigma_xy/reference: 0.1413
traction RMS/reference: 0.8382
```

Command rerun after PH4-05: same command above, 2026-09-12. The corrected diagnostic shows
`energy_balance_error=1.7482`; every L4 metric still fails. No checkpoint/report/objective
snapshot exists for this headless diagnostic path. This absence is a PH4-05/PH4-19 evidence
gap, not a passing result or a substituted artifact.

PH4-06 controlled rerun after removing hidden auxiliary direct-stress consistency, 2026-09-12:

```text
step-0 loss: 3.616970e-1 (previously 1.408062e1)
step-14 loss: 3.593229e-1
energy balance error: 1.7463
load transfer ratio: 0.1990
L4: FAIL all five hard metrics
```

This changes only removal of the undeclared `constitutive_consistency` constraint. It proves
that constraint dominated reported loss but does not explain the remaining poor physical field
after 15 steps. It is not a benchmark result or optimizer-tuning justification.

PH4-07 rerun after reference-normalizing translation gauge, 2026-09-12:

```text
physical Pi gradient norm: 2.565057
translation gauge gradient norm: 1.196630
rotation gauge gradient norm: 4.467696e-9
translation raw: 1.589016e-1, effective lambda: 49.99318
```

The translation mode is now materially constrained without exceeding physical-Pi gradient
scale. This run is only gauge-scale evidence; L4 still fails all hard metrics.
