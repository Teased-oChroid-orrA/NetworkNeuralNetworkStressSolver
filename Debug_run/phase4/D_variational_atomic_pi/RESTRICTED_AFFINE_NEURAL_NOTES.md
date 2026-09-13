# PH4-06 ladder stage 2 — restricted neural representation

Command:

```text
cargo run -p pinn-app --features ndarray-backend -- --headless --problem-spec Debug_run/phase4/D_variational_atomic_pi/restricted_affine_neural.toml
```

Only difference from E is network capacity: two hidden units and one hidden layer. This remains
an existing neural-network/PINN representation; no affine solution or analytical correction
entered training. Atomic measure-aware Pi, normalized translation and rotation gauges, uniform
sampling, AMR off, seed, optimizer, learning-rate schedule, point counts, and 200 steps are
unchanged. Complete stdout is `restricted_affine_neural.log`.

Results:

```text
L0 affine: PASS, relative error 1.039e-7
loss: 9.865384e0 at step 0; 6.510069e-1 at step 199
U: 5.644865e-2 J
W_ext: 4.139742e-1 J
Pi: -3.575255e-1 J
normalized Pi: -5.384285e-2
physical-Pi gradient norm: 6.821348e-1
translation-gauge gradient norm: 3.410605e-1
rotation-gauge gradient norm: 1.738076e-9
load transfer: 0.0960
L4 health: FAIL (energy-balance error 7.2728e-1)
L4 hard benchmark: FAIL all five metrics
```

Conclusion: restricting existing neural capacity does not recover affine field behavior. It
does show physical Pi can become negative without an invalid U/W ratio. Translation-gauge
gradient is material but smaller than physical Pi; rotation remains negligible. This excludes
full-network capacity as sole cause and supplies evidence before any optimizer-contract change.
