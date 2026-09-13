# PH4-08 controlled optimizer probe

Only change from E: peak learning rate `1e-3` to `3e-3`. Same full 16x2 neural network,
seed, atomic measure-aware potential, gauges, uniform 128-point sampling, AMR-off configuration,
and 200-step budget. Complete cadence ledger is in `optimizer_peak_lr_3e-3.log`.

```text
normalized Pi trend: +3.616969e-1 at step 0, +3.095275e-1 at 80,
                     -1.991536e-2 at 160, -1.767713e-1 at 199
physical weight: 1.000000 to 1.061891 (atomic whole-block scale only)
translation gauge: 1.623270e-1 to 4.686160e-6
final physical-Pi / translation / rotation gradient norms:
2.010310 / 6.311712e-3 / 2.324791e-9
U=1.654809e-1 J; W_ext=1.363868e0 J; Pi=-1.198387e0 J
probe normalized Pi=-1.804754e-1; load transfer=0.0868
L4 health: FAIL; hard L4: all five metrics FAIL
```

Optimizer suppresses translation without gauge dominance and decreases sampled Pi. Exact affine
no-hole elasticity has normalized continuum Pi minimum `-1`; final probe value `-0.1804754`
remains far above it. The run is therefore plainly non-stationary, not a valid converged field.
Next ladder stage increases only uniform collocation resolution; no benchmark threshold or
physical coefficient changes.
