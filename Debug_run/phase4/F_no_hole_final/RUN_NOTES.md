# PH4-10 convergence gate — extended

Configuration fixes corrected Variational Pi, uniform 512/256 sampling, AMR off, and peak
learning rate `3e-3`. Initial 1,000-step run completed: L4 health passed with
`energy_balance_error=3.2164e-3`, but hard L4 benchmark failed (`load_transfer=0.8838`,
`sigma_xx_relative_error=0.1196`). Extended 3,000-step run follows with unchanged physics and
sampling; acceptance waits for its final ledger.

The 32x3 capacity extension was terminated at step 300 after several minutes because its cost
was excessive and its trajectory matched the completed 1,000-step control. It is incomplete
diagnostic evidence only; no acceptance claim uses it.
