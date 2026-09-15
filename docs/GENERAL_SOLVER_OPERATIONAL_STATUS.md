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

**NOT_OPERATIONAL** — real, understood, and NOT resolved by fixing the AMR crash (issue #74).

Two real attempts, both reported exactly as measured (no benchmark-specific correction, issue
#63's own rule): without AMR, `kt=1.0076` vs theoretical `3.0` (66.4% relative error); WITH AMR
enabled (after #74's fix made this safe to try), `kt=1.0088` (66.37% relative error) — a 0.13%
change, not a meaningful improvement, despite AMR genuinely re-densifying the point set across
three sweeps (`4096→706→1381→2281` points) and a companion zero-cost fixture confirming AMR's
refinement mechanism itself works correctly (>3x density increase in a synthetic near-hole
residual band).

Root cause of the underlying undersampling is diagnosed, not merely observed: a zero-cost
sampling-only check shows only 0.55% of interior collocation points land within 2 hole-radii of
the boundary at this ratio under uniform sampling. But AMR's residual-driven refinement did not
translate that fixed mechanism into Kt accuracy at this training budget — plausibly because its
signal (`|dem_energy_per_point|`) reflects where the CURRENT network's residual happens to be
large, not literally "distance to the hole," so early sweeps (before the network has learned
much) may not concentrate density where it will matter later. This is a hypothesis, not a
confirmed mechanism. **L5 is a real, open numerical-accuracy problem, not merely blocked on a
crash fix** — #74 landing removed one candidate blocker and the accuracy gap remained.

## AMR + Variational

**OPERATIONAL** (crash fixed, verified) for running AMR at all; still NOT sufficient on its own
to close the L5 accuracy gap above.

Three findings across this epic: (1) issue #67 — a real interior-weight staleness bug (stale
AMR density-compensation weights applied to freshly-resampled, unrelated points for up to 1000
steps), fixed and regression-tested; (2) issue #74 — AMR combined with the Variational
formulation's term structure used to reproducibly crash inside `.backward()` with a burn-
autodiff internal panic between step 1200 and 2200 of a 2200-step run, root-caused (confirmed,
not just hypothesized, by direct comparison against Kirsch's own already-working AMR sweep) to
`probe_interior_energy_residuals` running forward passes through the live autodiff graph without
ever calling `.backward()` on them, and fixed by routing the probe through the non-autodiff
`BInner` backend instead — verified by the exact reproduction test now completing all 2200 steps
cleanly, an independent adversarial review of the fix's soundness, and a clean isolated re-run of
pin-lug's own AMR test suite (no regression on shared infrastructure); (3) even with the crash
fixed, AMR-enabled training still does not meaningfully improve L5's Kt accuracy (see above) —
AMR stays off (`amr_enabled=false`) for the canonical no-hole benchmark specifically for an
UNRELATED, independently-measured quality-regression reason (three real AMR-on runs consistently
borderline-worse than AMR-off there), which this crash fix never addressed and was never meant
to.

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
