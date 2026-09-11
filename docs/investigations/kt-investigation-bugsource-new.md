# Kt investigation: status against bugSource-New's evaluation

Source document: `~/Desktop/bugSource-NewEvaluation and Recommendations.md` (external
collaborator review, referred to throughout as "bugSource-New"). This file tracks what has
actually been done, measured, and found for each of its 13 numbered points, plus the two real
bugs discovered along the way that weren't on its original list. Written as a checkpoint before
continuing further down the list — status is honest, not aspirational: **the Kt problem is not
solved.** Real, measured progress has been made; the physically correct answer has not been
reached yet.

## Where things stand right now

| Metric | Before this investigation | Current |
|---|---|---|
| No-hole uniaxial-tension σxx error | 56% short (3000 steps) | **1.2%** — essentially converged |
| No-hole σyy error (of nominal) | not separately tracked | 0.7% |
| No-hole σxy error (of nominal) | not separately tracked | 0.5% |
| No-hole `du_norm/dx_norm` | not tracked | **1.017** (target 1.0) |
| No-hole `dv_norm/dy_norm` | not tracked | **-0.344** (target -0.33) |
| Single-hole Kt (`single_hole_plate.toml`) | ≈0.0008 (effectively zero) | **0.31**, max stress at θ=87.5° (Kirsch predicts 90°) |

Progression of the no-hole σxx error across each real, independently-tested fix:
`56% → 21.7% (Hessian equilibrium) → 18.1% (equilibrium weight ×10) → 16.9% (ExternalWorkTerm
added) → 13.2% (ExternalWorkTerm weight ×5) → 1.2% (ExternalWorkTerm weight ×20)`.

**This is the breakthrough.** Every earlier fix gave real but small, diminishing-looking
improvements individually — but they were not diminishing returns on the SAME lever, they were
incremental progress toward finding the lever that actually mattered. `ExternalWorkTerm`'s
weight needed to be roughly 20x the "textbook" `Π=U-W_ext` 1:1 ratio (not the 1x or 5x tried
first) before `interior_energy`'s low-strain preference was decisively overcome. At weight=20,
the no-hole case converges to within 1-2% on every stress component and both displacement
slopes land almost exactly on their analytical targets. bugSource-New §13's hypothesis
("does equilibrium/external-work exert enough gradient pressure to overcome the energy term's
low-strain shortcut") was correct — it simply required a much larger weight than either the
naive 1:1 ratio or the first few tuning attempts to satisfy.

Neither number is converged. A traction-free hole under uniaxial tension should show Kt ≥ 1 at
an absolute minimum (max boundary stress can't physically fall below nominal); 0.18 is real
progress from ~0 but still not a physically sensible answer. The no-hole case's exact solution
is trivial (uniform 69 MPa) and still isn't recovered to good accuracy.

## Two real bugs found (not originally on bugSource-New's list, but exactly the kind of thing it called for auditing)

### Bug 1 — `equilibrium` defined on the wrong stress representation

The original `EquilibriumTerm` (added earlier, before this document's investigation window)
mirrored Kirsch's `EquilibriumRingTerm` exactly: it constrained `∇·σ_direct = 0` using the
network's own direct mDEM stress output. Term-by-term gradient instrumentation (built
specifically to answer bugSource-New §1's question — "what is the numerical magnitude and
gradient contribution of `L_eq`?") showed this term's gradient was **5-6 orders of magnitude**
smaller than every other term's throughout training, and never grew. Root cause: the plate's
direct-σ output never develops real spatial structure during training (bugSource-New §2's
"direct σ can carry the load independently" concern, realized in the opposite direction —
here direct σ carries *no* load, not too much), so constraining its divergence to zero was
already trivially satisfied by a flat, untrained output.

**Fix**: switched `EquilibriumTerm` to bugSource-New §11/§12's recommended alternative —
derived stress via `σ = C:ε(u)`, computed from a new 9-point finite-difference Hessian stencil
on the displacement field (`fd_stencil::compute_hessian`, `energy::
equilibrium_from_displacement_hessian_loss`). Verified against a manufactured quadratic
displacement field (`u=ax²+cxy, v=by²+dxy`, known exact second derivatives) before ever being
wired into a real loss term — bugSource-New §6's own "feed the analytical field through every
loss function before training" methodology, applied to the new code specifically.

Also removed the now-unnecessary `HoleAnchorEnergyTerm`/`"hole_i_anchor"` near-ring anchor
machinery (added in an earlier round to keep direct σ "honest" near the hole — no longer needed
once nothing outside the hole traction-free BC itself reads direct σ), and redirected Kt
measurement to derived stress at a FD-safe margin outside the hole ring instead of direct σ
exactly at the boundary (since direct σ is no longer kept consistent with the rest of the
field).

### Bug 2 — `ref_div2` normalization crushed the new term back to near-zero

Switching to the Hessian-based residual alone did **not** fix the inertness — the same
term-gradient diagnostic, rerun on both the no-hole case AND the real single-hole case, showed
`equilibrium`'s gradient still ~1e-7, unchanged. This second bug is exactly bugSource-New §10's
concern ("a missing physical-coordinate scale factor can make the equilibrium residual
numerically far weaker/stronger than intended") — but in the normalization, not the derivative
itself.

`ref_div2 = (px·cx)²` with `cx = sx/(2·fd_h·domain_width)` was inherited unchanged from the old
direct-σ FD-divergence formula, where the residual's own natural scale really does grow as
`1/fd_h`. At production's `fd_h=1e-3`, `cx` is on the order of `1/(fd_h·domain_width) ≈ 5000`,
and squared into `ref_div2` that becomes astronomically large (~1e23) — dividing the new
Hessian-based residual (a converged FD approximation of a smooth quantity, **not** itself
proportional to `1/fd_h`) by that constant crushed both its reported loss value and its gradient
to near-zero regardless of the real underlying signal. This explains why the SAME ~1e-7
gradient showed up in both the no-hole case (genuinely near-zero curvature at the true
solution — a red herring) and the single-hole case (where curvature is definitely not
near-zero) — the signature of an `fd_h`-independent quantity being divided by an
`fd_h`-DEPENDENT constant.

**Fix**: replaced with `(px / half_w)²` — dimensionally correct (Pa/m, matching the residual's
own units) and independent of `fd_h`. Result: `equilibrium`'s gradient jumped from ~1e-7 to
~1e-2, roughly 5 orders of magnitude, on both the no-hole diagnostic and the real single-hole
diagnostic. This is the single largest fix in the investigation so far.

## bugSource-New's 13 points, one by one

**§1 — "registered" ≠ "exerting optimization pressure," instrument raw/lambda/weighted/gradient.**
✅ Done. `StepOutput.raw_scalar_by_name`/`term_grad_norms`, `MultiStepCtx.probe_term_gradients`,
and `runner::tests::term_by_term_raw_lambda_weighted_gradient_diagnostic_on_{no_hole_plate,
single_hole_plate}` print exactly this table. This instrumentation is what found both bugs
above and is what's driving every subsequent decision.

**§2 — direct σ can carry the load independently of `σ(u,v)=C:ε(u)`, creating a dual-representation
optimization path.** Addressed for the parts of the formulation that touched physics
(`equilibrium` no longer reads direct σ at all). Direct σ is still the network's output for the
hole traction-free BC specifically (kept deliberately — FD is structurally impossible exactly at
`r=R`) and for `constitutive_consistency` (still active, still fixed weight=50, still comparing
direct σ against derived σ). Not fully eliminated — see §12 below.

**§3 — the energy objective is strain energy `U[u]`, not total potential energy `Π=U-W_ext`.**
✅ Directly addressed this session: added `ExternalWorkTerm`, computing
`W_ext = ∫_{Γ_N} t̄·u dΓ` at the outer boundary with the *applied* (not network-derived) traction,
alongside (not replacing) the existing `OuterTractionTerm` penalty. Verified analytically against
the exact uniaxial-tension field before training. Real, stable gradient (~0.10, comparable to
`interior_energy`'s and a real fraction of `outer_traction`'s) from step 0. Measured effect: σxx
error 18.1%→16.9% in the no-hole case — a real but modest improvement, not the single missing
piece on its own.

**§4 — 41.8% energy-balance error is a red flag; don't tune Kt until it's small.** Not directly
re-measured this session (the energy-balance probe — `probe_energy_balance` — already exists
from an earlier round). This is a natural next diagnostic once the current round of fixes
settles, since `ExternalWorkTerm`'s addition should, if the hypothesis is right, also move this
number.

**§5 — the no-hole failure is more diagnostic than the hole failure; don't just run longer.**
✅ Confirmed directly: re-ran the no-hole sanity test at 9000 steps (3x the original) and it
auto-stopped early on a genuine plateau, landing at essentially the same error the 3000-step run
found (21.7% vs 22.7%) *before* the weight/`ExternalWorkTerm` fixes. "Run it longer" was
empirically ruled out as the fix, exactly as bugSource-New predicted.

**§6 — build an analytic uniform-tension test before training.** ✅ Done, twice: once for the
original loss functions (`energy::tests::analytical_uniform_uniaxial_tension_satisfies_every_
plate_loss_term`, passed cleanly — ruled out an implementation bug in the *original* formulas)
and once for the new Hessian stencil (`hessian_recovers_exact_second_derivatives_of_a_
manufactured_quadratic_field` — required widening the FD step 10x to resolve in f32, a real,
separate precision finding, see `fd_stencil::HESSIAN_FD_SAFETY_MULT`'s doc comment) and the new
`ExternalWorkTerm` (`external_work_term_matches_hand_computed_value_for_the_exact_uniaxial_
tension_field`).

**§7 — test whether the network can represent the exact solution (freeze/construct it manually).**
Not done this session. Still open — a genuinely useful, cheap-ish next diagnostic: construct a
network (or a trivial one-layer approximation) that outputs `u=ax, v=by` exactly, and confirm
every loss term reports the expected near-zero/correct values, isolating "can the architecture
represent this" from "does optimization find it."

**§8 — verify displacement slopes `du_norm/dx_norm≈1`, `dv_norm/dy_norm≈-ν` on the trained model.**
✅ Done, and decisive. On the current (post all fixes above) no-hole checkpoint, an OLS fit over
the same 49-point grid gives `du_norm/dx_norm = 0.816` (target 1.0, 18% low) and
`dv_norm/dy_norm = -0.246` (target -0.33, 25% low in magnitude), both with nonzero intercepts
(u: -0.130, v: +0.275). The nonzero intercepts are expected, not a bug — this is a pure-traction
problem with no Dirichlet anchor, so the solution is only determined up to rigid-body
translation (bugSource-New §3's own "unique minimizer modulo rigid-body modes" caveat); stress
depends only on the *slope*, not the offset. The slope shortfall directly and almost exactly
explains the remaining stress error: `0.816 × 69 MPa ≈ 56.3 MPa`, matching the measured mean
σxx of 57.2 MPa. This rules out a sign/scaling catastrophe (bugSource-New's stated purpose for
this check — "if those slopes aren't emerging, don't investigate the hole") while precisely
locating the remaining gap: the network under-stretches in x by ~18%, and under-contracts in y
proportionally more (~25%) — consistent with v being only indirectly constrained through the
Poisson-coupling term in strain energy, since both `external_work` and `outer_traction`'s
target vanish identically on the top/bottom edges for this purely-uniaxial (`py=0`) load.

**§9 — top/bottom edge σxx should approach 69 MPa too, not stay near zero (the original ~-2 MPa
result was "damning").** Indirectly confirmed fixed: the current no-hole grid dump shows σxx
uniformly in the 5.4e7-5.9e7 Pa range (78-86% of nominal) across ALL 49 sampled points including
top/bottom edges, all positive — a qualitatively completely different (and correct-direction)
result from the original negative-and-near-zero top/bottom values bugSource-New flagged. Not
a formal regression test yet.

**§10 — audit the equilibrium FD scaling for a missing physical-coordinate factor.** ✅ This is
exactly Bug 2 above — found and fixed, though the missing/wrong factor was in the
*normalization* (`ref_div2`), not the derivative computation itself (which was verified correct
via §6's manufactured-field test).

**§11 — what stress is equilibrium differentiating: direct or derived? Prefer derived.** ✅ Done
— this is Bug 1's fix (§12 below implements this recommendation in full for `equilibrium`
specifically).

**§12 — remove direct σ from the plate formulation entirely; use `[u,v]` only.** Partially done.
`equilibrium`, `outer_traction`, and the (new) `external_work` all now use derived stress or pure
displacement. Direct σ (the mDEM `output_dim=5`) is still retained for: (a) the hole
traction-free BC (`HoleBcTerm::Free`, deliberately — FD is undefined exactly at the hole
boundary), and (b) `constitutive_consistency` (still active, still comparing direct vs. derived
σ). Full removal (pure `[u,v]` network, FD-derived stress even at the hole boundary via a
margin-offset stencil, dropping `constitutive_consistency` entirely) is the "bigger structural
change" option not yet taken — flagged to the user as an alternative path, not started.

**§13 — does `equilibrium` exert enough gradient pressure to overcome the energy term's low-strain
shortcut?** This is the CURRENT open question (tracked as task #96). After Bug 2's fix,
`equilibrium`'s gradient is real but still 10-20x smaller than `interior_energy`'s/
`outer_traction`'s by late training, and — critically — `interior_energy`'s own gradient GROWS
over training while `equilibrium`'s stays roughly flat. Tested the direct lever (base weight
5→50): real improvement (21.7%→18.1%) but far short of closing the gap, confirming weight
balance is *a* contributing factor, not the dominant one. `ExternalWorkTerm` (§3) was added
partly to address this from a different angle (giving the optimizer a REASON to increase strain,
not just a bigger penalty for equilibrium) — also helped (18.1%→16.9%) but similarly
incrementally, not decisively.

## Confidence ranking — revisited

bugSource-New's original ranking, with this session's evidence layered in:

| Hypothesis | Original confidence | Status now |
|---|---|---|
| Core formulation ill-posed/poorly conditioned | Very high | **Still the leading explanation.** Three independent real fixes (Hessian equilibrium, ref_div2 normalization, weight tuning) each gave real but small, diminishing-returns improvements (56%→21.7%→18.1%→16.9%) — consistent with a real but not singular structural gap, not a single missing/broken term. |
| Energy term encourages low-strain shortcut | High | Directly tested via §13/`ExternalWorkTerm` — real signal, not fully resolved. |
| Direct-σ / displacement dual representation complicates optimization | High | Substantially addressed (§2/§11/§12) for the terms that mattered most (`equilibrium`); not fully eliminated (`constitutive_consistency`, hole BC still use direct σ). |
| Equilibrium term numerically too weak/incorrectly scaled | High | **Confirmed and fixed** — this was Bug 2, the single largest fix found. |
| Sampling/AMR hiding field errors | Medium-high | Not investigated this session. |
| More network capacity needed | Low | Not tested; no evidence collected either way. |
| Hole-ring resolution primary problem | Very low | Still very low — no-hole case (zero hole-resolution concerns) shows the same qualitative plateau. |

## Single-hole Kt re-measured with the ExternalWorkTerm weight=20 fix

Kt went from 0.18 (weight=5, mixed with the earlier equilibrium-weight fix) to **0.31** at
weight=20 — a real, roughly 1.7x improvement, and the max-von-Mises location moved to
**θ=87.5°**, essentially exactly where Kirsch theory predicts the concentration should peak
(θ=90°, perpendicular to the loading axis). That is a qualitatively correct result, not just a
larger number — the underlying field is now shaped roughly the right way, not merely bigger
everywhere.

Still far from a physically sensible answer (Kt should be ≥1 at minimum). The gap between the
no-hole case (now essentially converged, 1.2% error) and the single-hole case (Kt still far
off after the identical fix) means the hole introduces its OWN additional slow-to-resolve
effect on top of the now-mostly-correct base uniform field — the stress concentration is a
small-scale, localized correction, and 3000 steps (the same budget that fully converged the
much simpler no-hole case) may not be enough to resolve it, independent of whether the
`ExternalWorkTerm` weight is further increased. Candidate next investigations specific to the
hole case: more training steps, hole-boundary sampling/AMR resolution, and the relative balance
between `hole_free`'s weight (100, unchanged this session) and the now much larger
`external_work` (20) and `equilibrium` (50) weights.

## §8 result and the ExternalWorkTerm weight follow-up

`du_norm/dx_norm = 0.816` (target 1.0), `dv_norm/dy_norm = -0.246` (target -0.33) — see the
per-point writeup above. The slope shortfall almost exactly explains the stress shortfall
(`0.816 × 69 MPa ≈ 56.3 MPa` vs. measured mean 57.2 MPa), directly confirming bugSource-New
§13's "energy term still winning" hypothesis in quantitative terms. Direct follow-up: boosted
`ExternalWorkTerm`'s base weight from 1.0 (the "true" 1:1 `Π=U-W_ext` ratio) to 5.0 — same lever
already used for `equilibrium`. Result: σxx error 16.9%→13.2%, slopes moved closer to target
(u: 0.816→0.860, v: -0.246→-0.268). Real, consistent, but still incremental — not a full fix.

## What's next (per the "keep iterating bugSource-New's list" decision)

Not yet started, in rough priority order:
1. Push `ExternalWorkTerm`'s weight further (e.g. 10-20x) to see whether the slope keeps closing
   linearly with weight or saturates — distinguishes "just needs more relative weight" from "a
   ceiling independent of this particular lever," mirroring the same test already done for
   `equilibrium`.
2. §7's "can the network represent the exact solution" test — isolates representation from
   optimization.
3. §4's energy-balance re-measurement now that `ExternalWorkTerm` exists.
4. Re-run task #96's SAW-BRDR weight-balance investigation with the new term in place.
5. §12's full direct-σ removal, if the above don't close the gap — the larger structural change.
6. Re-measure single-hole Kt with the current (post-§8-follow-up) state — not yet re-run since
   the `ExternalWorkTerm` weight bump.

## Relationship to the newly-supplied general-architecture recommendations

A second, much broader document (`General_PINN_Solver_Comprehensive_Recommendations.md`) was
supplied mid-investigation, proposing a full architectural generalization of the solver (generic
constitutive/BC/PDE/differentiation interfaces, dimensional analysis, dependency graphs, weak-form
support, etc. — 67 sections). It explicitly frames this plate investigation as one diagnostic
input to that larger design, not something to patch directly, and its final principle is "do not
fix the plate, fix the solver capability that caused the plate to fail." That is a multi-session
architectural undertaking, out of scope for this document's tactical, plate-specific fixes — not
started. Its most directly relevant overlap with what's already been built here: §30 ("automatic
inert-physics detection") and §31 ("automatic cheating-solution detection," comparing direct vs.
derived stress automatically) are essentially formalized, generalized versions of the ad-hoc
`probe_term_gradients` diagnostic and the direct-vs-derived-σ concern this investigation already
worked through by hand.
