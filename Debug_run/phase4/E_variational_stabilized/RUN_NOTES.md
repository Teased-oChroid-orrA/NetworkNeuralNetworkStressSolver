# PH4-06 controlled ladder E

Command:

```text
cargo run -p pinn-app --features ndarray-backend -- --headless --problem-spec Debug_run/phase4/E_variational_stabilized/problem.toml
```

Only change from D: training budget 15 to 200 steps. The code is the corrected atomic Pi path
without undeclared direct-stress consistency. AMR remains off and sampling remains uniform.
Full stdout is persisted in `solver.log`.

Results:

```text
L0 affine: PASS, relative error 1.039e-7
loss: 3.616970e-1 at step 0; -2.691467e-2 at step 199
U: 3.557104e-1 J
W_ext: 5.456002e-1 J
Pi: -1.898899e-1 J
normalized Pi: -2.859715e-2
physical-Pi gradient norm: 2.100251
translation-gauge gradient norm: 1.464989e-8
rotation-gauge gradient norm: 7.491606e-9
load transfer: 0.0699
L4 health: PASS only (energy-balance error 3.0392e-1)
L4 hard benchmark: FAIL all five metrics
```

Conclusion: declining Pi and a passing loose health check do not establish physical convergence.
Load transfer degraded from the 15-step D diagnostic; no optimizer change is justified until the
objective/field discrepancy is explained. Gauge gradients are negligible relative to physical
Pi, so this run rules out gauge-gradient dominance as the present cause.

## Gauge-scale correction rerun (2026-09-12)

The preceding result used a dimensional translation gauge. `TranslationGaugeTerm` now divides
the squared mean displacement by `u_ref^2`, so it is dimensionless like the normalized physical
potential. This rerun changes no sampler, network, objective, optimizer, or step budget.

```text
loss: 8.478048e0 at step 0; 1.771334e-1 at step 199
U: 2.799427e-1 J
W_ext: -7.325458e-1 J
Pi: 1.012488e0 J
normalized Pi: 1.524794e-1
physical-Pi gradient norm: 2.181287
translation-gauge raw/lambda/gradient: 4.480239e-4 / 4.858950e1 / 6.169782e-2
rotation-gauge raw/lambda/gradient: 1.078861e-10 / 5.077903e1 / 2.548643e-9
load transfer: 0.0780
L4 health: FAIL (energy-balance error 1.7643)
L4 hard benchmark: FAIL all five metrics
```

Conclusion: corrected translation scaling makes its gradient observable but remains about 35x
below the physical-potential gradient. It does not repair load transfer or the field benchmark.
The root cause is therefore neither a missing rotation constraint nor a dimensional translation
gauge alone. Do not tune optimizer parameters before the restricted-representation comparison.
