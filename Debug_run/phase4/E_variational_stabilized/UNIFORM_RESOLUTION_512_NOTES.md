# PH4-06/08 uniform-resolution control

Relative to `optimizer_peak_lr_3e-3.toml`, only uniform quadrature resolution changes:
`n_interior=128,n_boundary=64` to `512,256`. All objective, network, optimizer, seed, gauges,
AMR-off state, and 200-step budget remain fixed. Full output: `uniform_resolution_512.log`.

```text
normalized Pi: +3.611552e-1 at step 0; -1.538412e-1 at step 199
physical-Pi / translation / rotation gradients: 2.127576 / 3.486059e-3 / 2.610272e-9
U=1.765495e-1 J; W_ext=1.222590e0 J; Pi=-1.046041e0 J
probe normalized Pi=-1.575323e-1
load transfer=0.0836; L4 health FAIL; five hard L4 failures
```

Result: fourfold uniform resolution does not repair load transfer or field metrics in the
fixed 200-step budget. It marginally changes Pi but leaves physical gradient large. This rules
out low quadrature count alone as root cause. AMR remains disabled because no mathematically
correct uniform baseline exists.
