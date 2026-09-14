# General Solver Operational Status

Computed per issue #63's own explicit criteria (PH4-21), from real evidence collected across
sub-issues #64-#71. Status is scoped per capability — no single blanket claim, per issue #72's
own explicit instruction not to inflate partial coverage into a blanket OPERATIONAL claim.

## Corrected-Variational, no-hole (square)

**OPERATIONAL.**

Real headless CLI run (`examples/problems/variational_no_hole_plate.toml`, 3000 steps,
n_interior=n_boundary=4096): all five P2-14 hard thresholds pass with wide margin
(`sigma_xx_relative_error≈0.0009`, `load_transfer_ratio≈1.000`); independent field validation
(`validate_no_hole_fields`, a separate grid, not training points) also passes; normalized
`Pi≈-1.000` matches the exact continuum affine minimum. Root cause of the prior failure (a
frozen, non-varying collocation point cloud silently defeating the per-step resampling design
intent) is fixed and regression-tested (issue #64); the GUI-streaming training path has the same
fix applied and is proven equivalent to the headless path at step 0 (issue #66/#73). AMR stays
disabled for this configuration (see below) — this status covers the AMR-off, fixed-sampling
path, which is what the shipped example config actually runs.

## Corrected-Variational, no-hole (non-square)

**OPERATIONAL.**

Identical configuration and machinery, `half_w=0.15`/`half_h=0.08` (aspect ratio 1.875:1). Real
headless CLI run (issue #68): all five hard thresholds pass (`sigma_xx_relative_error=0.0016`,
`load_transfer_ratio=0.9971`) with real margin, slightly lower quality than the square case
consistent with a genuinely harder geometry, not a latent bug. No code changes were needed — the
measure-aware boundary integral already reads `geometry.half_w`/`half_h` independently.

## Corrected-Variational, single/multi-hole topology (machinery)

**PARTIALLY_OPERATIONAL** — sampling/stencil/FieldKind/gauge machinery is OPERATIONAL; Kt
numerical accuracy is NOT_OPERATIONAL.

Three real hole-bearing headless runs (single-hole, 2-hole mixed-BC non-square, 3-hole
asymmetric mixed-BC; issue #69) all run cleanly end-to-end: no crashes, finite diagnostics
throughout, correct `FieldKind`-resolved stress source, correct gauge-term activation
(`translation_gauge` correctly absent when a `Fixed` hole already anchors the geometry),
closed-boundary equilibrium error `0.3%-1.5%` of reference force in all three (a real,
closed-form-independent physical consistency check). These were deliberately short (800-step)
smoke budgets — the Kt values reported (0.5-0.6) are far below the physically-expected >1
stress-concentration range and are explicitly diagnostic-only, not a convergence claim.

## Corrected-Variational, hole/Kt accuracy (L5)

**NOT_OPERATIONAL.**

A real, much longer run (issue #70: 3000 steps, n_interior=n_boundary=4096, single hole
ratio=0.05, against a verified-passing no-hole companion) gives `kt=1.0076` against the
theoretical infinite-plate value of `3.0` — a 66.4% relative error. Reported exactly as
measured; no benchmark-specific correction was applied to force a pass (issue #63's own rule).

Root cause is diagnosed, not merely observed: a zero-cost sampling-only check shows only 0.55%
of interior collocation points land within 2 hole-radii of the boundary at this ratio — uniform
Monte-Carlo sampling structurally starves the sharp near-boundary stress-concentration region of
training signal. This is a real, understood limitation, not an unexplained failure. The known
fix (density biasing toward the hole boundary) is exactly what AMR already implements, but
AMR+Variational currently crashes past ~1200-2200 steps (issue #74) and is disabled for that
reason. **L5 is blocked on issue #74**, not on further tuning of the current approach.

## AMR + Variational

**NOT_OPERATIONAL** (deliberately disabled, not merely broken).

Two independent findings this epic (issue #67/#74): (1) a real interior-weight staleness bug was
found and fixed (stale AMR density-compensation weights applied to freshly-resampled, unrelated
points for up to 1000 steps) — fixed and regression-tested; (2) even after that fix, AMR
combined with the Variational formulation's term structure reproducibly crashes inside
`.backward()` with a burn-autodiff internal panic, somewhere between step 1200 and 2200 of a
2200-step run. A root-cause hypothesis exists (an AMR residual probe running forward passes
through the live autodiff graph without ever calling `.backward()` on them) but is not
independently confirmed, and the fix was deliberately deferred — it touches shared AMR
infrastructure also used by Kirsch/pin-lug and needs its own careful, independently-verified
change. AMR stays off (`amr_enabled=false`) for every shipped Variational config as the correct,
evidenced, policy-compliant state, not a workaround pending later cleanup.

## Strong / Hybrid / Weak `FormulationSelection` (user-defined problems)

**NOT_OPERATIONAL** (untested), except where noted.

`Strong` and `Hybrid` are wired (term-selection logic exists and is unit-tested — correct term
activation for a given formulation is verified) but have never been exercised in a real training
run in this epic; no runtime convergence evidence exists for either. `Weak` has no user-defined-
problem implementation at all. The legacy pre-trait Hybrid L4 artifact (Kirsch's own hardcoded
path, not this formulation-selection mechanism) is not generalized evidence and itself fails
current hard thresholds.

## Cross-cutting infrastructure

**OPERATIONAL** (evidenced, not merely implemented):

- `DifferentialOperator` production FD choke point (PH4-11): code-audited, single call site,
  no bypass, for every non-test production strain computation. Runtime-proven only for the
  Variational formulation (the only one exercised in a real run this epic).
- `FieldKind` executable resolver (PH4-12): both production consumers migrated, real hole
  runtime evidence exists (issue #69).
- Formulation-aware convergence cadence records (PH4-10): proven end-to-end by every passing
  real run above.
- Checkpoint/provenance round-trip of the full mathematical objective (PH4-19): real checkpoint
  artifact from corrected-Variational training, `Pi=U-W_ext` consistency proven on save/load.

## What this status does NOT claim

- No formulation other than Variational has real training-run evidence.
- Hole/Kt numerical accuracy is not operational for any formulation or geometry.
- AMR is not operational in combination with Variational for any geometry.
- No claim is made about GUI-driven pin-lug/Kirsch paths beyond what was already established
  pre-Phase-4 (out of this epic's scope).

This document reflects real evidence as of the #64-#71 chain (2026-09-14). It supersedes the
prior version of this file, which recorded the PRE-#64-fix broken state
(`load_transfer_ratio=0.1996`, all L4 metrics failing) and is now obsolete.
