# Issue #78 Stage 2 + three follow-ups: N-hole hard-constraint ansatz and the saturation_scale root-cause fix

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. The `MultiHoleHardConstraint`/`multi_hole_saturation_scale` ansatz generalization, the falsified per-hole coordinate-embedding hypothesis, and the real root cause (vanishing gradient at the Kt measurement margin) plus its derived-`TARGET_PHI_AT_MARGIN` fix.

## Issue #78 Stage 2: N-hole hard-constraint ansatz — a real Kt-accuracy fix, and two real bugs
## found chasing it

Stage 1 (above) closed with "multi-hole Kt accuracy has no correction path yet." This stage
built one, following the multi-hole-fem-ground-truth-investigation's own approved plan
(`docs/multi-hole-fem-ground-truth-investigation.md` — the authoritative record; this section
is the short pointer). One-line summary: real, substantial improvement (trained Kt for every
`HoleBc::Free` hole went from off by 6-600x to within 12-20% of real FEM ground truth), via one
real methodology extension plus two real, previously-undiscovered bugs found investigating why
the first attempt didn't work.

**`AnnulusAnsatz::MultiHoleHardConstraint(Vec<HoleTractionFreeAnsatz>)`**
(`kirsch_hole_correction.rs`) generalizes the single-hole `HardConstraint` ansatz to N `HoleBc::
Free` holes, any position — `decomposition_applicable`'s centered/single-hole restriction was
confirmed (by reading its own code) to be exactly what it always said: a deliberate scope
narrowing, not a mathematical constraint. Product of envelopes (multiplicative suppression
stays EXACTLY zero at every hole's own boundary for any N), sum of closed-form corrections
(additive baseline is only APPROXIMATELY traction-free for N>1 — the same 0.1-4.4% interaction
error the investigation's own Phase B measured, now also present in the trained field, not just
the closed form). N=1 reduces byte-identically to the pre-existing path — proven, not assumed.
`UserDefinedProblem::new_with_hard_constraint_ansatz` builds one `HoleTractionFreeAnsatz` per
Free hole; `loss_terms()`'s existing single-hole `hole_free`-suppression gate generalizes to
"any Free hole under an active hard constraint"; `UserSamplingStrategy`'s hole-biased sampling
and `hole_bias_quadrature_weights` generalize to bias every Free hole, splitting the budget
evenly (loud assertion, not silent mishandling, if bias disks would overlap).

**Real bug #1, found because the first real N-hole training run collapsed to Kt≈0.004-0.014
(trivial-solution collapse, not "just needs more steps"):** `UserDefinedProblem::loss_terms()`'s
`affine_strain_pair` (the constant far-field background strain added to the network's own
learning target under kinematic decomposition) was gated ONLY on `decomposition_applicable`
(the single-centered-hole case) — never on the new N-hole `MultiHoleHardConstraint` case, even
though this constant's own value is independent of hole count/position by construction. Every
off-center/multi-hole hard-constraint spec therefore trained with the network having to learn
the ENTIRE affine far-field background from scratch on top of refining the hole correction —
precisely the gradient-competition failure mode issue #77's original decomposition fix existed
to eliminate, silently reintroduced for exactly the geometry class this stage targets. Fixed by
generalizing the gate to `decomposed || self.hard_constraint_active()` (`UserDefinedProblem::
hard_constraint_active` made `pub(crate)` for this) — a strict OR, byte-identical for every
pre-existing case (`decomposed=true` already implied `Some` before; the new case this adds is
`decomposed=false, hard_constraint_active()=true`). Alone, this moved trained Kt from ≈0.004 to
≈0.5 once the spec's own `formulation`/`measure_aware_training` were also corrected to match
(see next paragraph).

**A real methodology trap this session fell into and corrected**: the shipped
`triple_hole_plate.toml`/`notched_plate.toml` examples have NO `[formulation]` override, so
they use `default_formulation()` — the literal PRE-#77 Hybrid trajectory, which never got issue
#77's own Kt fix at all. PH4-42's real verified L5 result used `formulation = "Variational"` +
`training.measure_aware_training = true` explicitly. Real headless verification MUST use that
same formulation on any hard-constraint spec — `variational_triple_hole_smoke.toml`/
`variational_notched_smoke.toml` (already-shipped examples with the right formulation) are the
correct base to build a hard-constraint verification config from, not the plain production
examples.

**Real bug #2, found because a P2-09 "trivial-solution" warning (load transfer ratio 0.02-0.07)
persisted even after bug #1's fix, despite the model's own real Kt already reading correctly
(~2.5) via a DIFFERENT, correctly-wired diagnostic function.** `probe_load_transfer`/
`probe_reaction_force`/`probe_boundary_residuals` (`user_problem.rs`) had never been extended
to accept an `ansatz`/`affine_strain_pair` parameter at all — unlike `probe_hole_boundary_
profile_derived` (correctly fixed back in PH4-41), these three read the model via a bare
`fwd_embedded` forward, completely bypassing any active `AnnulusAnsatz`/affine background, for
ANY spec. This bug predates issue #78 entirely and would have affected the ORIGINAL single-hole
L5 hard-constraint case too — never noticed because that case's real PH4-42 verification went
through `run_user_problem_training_with_diagnostics`/`user_problem_l5_diagnostic` (correctly
ansatz-threaded from the start), never `run_headless_user_problem`'s own printed CLI
diagnostics. Fixed by threading `ansatz`/`affine_strain_pair` through all three functions
(same `stencil_forward_with_ansatz`-based pattern as `probe_hole_boundary_profile_derived`) and
updating every call site: the headless CLI diagnostic loop, the GUI-streaming vis-cadence block,
the GUI checkpoint-save serving loop (all three now use the model's own real training-time
`problem.ansatz(0)`), the GUI checkpoint-LOAD serving path (deliberately LEFT `IdentityAnsatz`-
only — checkpoint metadata doesn't carry `ProblemSpec.architecture` yet, a real, disclosed,
still-open gap, not silently glossed over), `run_no_hole_benchmark` (correctly stays
`IdentityAnsatz`-only — no hole, no ansatz concept), and 8 pure-logic tests (`IdentityAnsatz,
None` — every one already trained a plain `UserDefinedProblem::new(spec)`). This alone took
the load transfer ratio from 0.02-0.07 (spurious "collapsed solution") to 1.00-1.01 (genuinely
healthy) with ZERO change to the actual trained model weights — purely a diagnostic-
reconstruction fix, proving the earlier false alarm was a measurement bug, not a training bug.

**Real trained result** (both real shipped geometries, both bugs fixed, `formulation=
"Variational"`, `measure_aware_training=true`, 3000 steps, `hole_bias_fraction=0.4`):
`notched_plate.toml` (N=1 Free, off-center) hole0 Kt=2.694 (FEM=3.063, superposition=2.960);
`triple_hole_plate.toml` (N=2 Free, off-center) hole0/hole2 Kt=2.507/2.507 (FEM=3.133/3.058,
superposition=2.994/2.994); the Fixed middle hole read Kt=1.048, not directly comparable to the
investigation's own Free/Free/Free FEM numbers (a pre-existing, disclosed caveat — the FEM tool
has no Fixed-BC support). Load transfer ratio 1.00-1.01 and reaction-force equilibrium error
~1% on every run — genuinely healthy, non-collapsed solutions.

**Honestly NOT full PH4-42-level convergence (0.67-1.23%)** — 12-20% remaining error, for two
disclosed reasons: off-center/multi-hole specs never get the single-hole path's OTHER accuracy
machinery (log-polar embedding, `SequentialTwoStage`, still single-hole-gated by this stage's
own deliberate scope), and the additive baseline itself carries Phase B's own measured N>1
interaction residual into the trained field, not just the closed form.

**The flat-loss-plateau observation above has since been fully investigated in a direct
follow-up (same session) — real evidence, not left as an open flag.** Full detail lives in
`docs/multi-hole-fem-ground-truth-investigation.md`'s own "Follow-up" section; short version:

**A NEW real bug found and fixed while investigating the (separately real) BC-mismatch
caveat**: `tools/multi_hole_reference.py`'s FEM ground-truth tool never supported `Fixed`
holes at all — every prior FEM number in this stage (and Phase A/B before it) treated every
hole as traction-free, while the real PINN specs have one genuinely `Fixed` hole each. Fixed by
adding real Dirichlet (zero-displacement) BC support. Building and verifying that fix surfaced
a SECOND, more serious, pre-existing bug in the SAME tool: its rigid-body-motion pin scheme
(asymmetric by construction — full pin on the left edge, roller-only on the right) had always
been silently wrong, invisible only because every all-`Free` geometry this tool ever shipped
with made the pins carry zero reaction force regardless of asymmetric placement. The instant a
`Fixed` hole (a real internal support with a genuinely nonzero reaction) was added, this broke
physical mirror symmetry outright — two Free holes that MUST have identical Kt by symmetry
(`triple_hole_plate.toml`'s own real geometry) came out ~30-50% different, confirmed real (not
noise) via a geometry-mirroring cross-check. Fixed with a genuinely symmetric pin scheme;
verified the fix is a pure bug fix with ZERO effect on every existing all-`Free` number (direct
stress-field comparison, old vs. new pin choice, agrees to `~1.6e-10` relative precision). 6
new regression tests. Real, BC-corrected FEM ground truth is now available and materially
changes the comparison picture — see the investigation doc's own table.

**The flat-loss plateau itself: definitively diagnosed via a new live per-step Kt diagnostic**
(`user_runner.rs::run_headless_user_problem`, printed every checkpoint during training, not
just at the end — a genuinely new, permanent capability). Real finding: Kt reaches its final
value by ~step 300 and then GENUINELY does not move for the remaining ~2700 steps — the flat
total_loss is real convergence, not a metric hiding continued progress, and not a training bug.
Three independent hyperparameter experiments (hole_bias_fraction 0.4→1.0, hidden_dim 64→128,
peak lr 1e-3→5e-3), each a major axis change, each reproduced the IDENTICAL converged Kt to
3-4 significant figures — cleanly ruling out collocation density, network capacity, and
optimizer step size as the cause. Working conclusion: the remaining accuracy gap is a
structural/formulation limitation, not a tuning problem — the N-hole ansatz (this stage's own
work) gives the network a strictly LARGER, less-constrained learning target than the original
single-hole L5 case had, because L5's real 0.67-1.23% accuracy came from the ansatz AND
kinematic decomposition TOGETHER, and only the ansatz half was generalized to N holes this
stage. Full closure would mean generalizing kinematic decomposition itself to N holes too — a
real, substantial, separately-scoped future task, deliberately not attempted in this pass. This
is a DEFINITIVELY DIAGNOSED, disclosed open item (evidence in hand for what it is NOT), not a
silently-accepted ceiling — matches this project's own "BLOCKED documented with evidence is not
the same as complete" standard (Issue #63 Phase 4 close-out, above).

Full regression suite after every change in this stage (`cargo test -p pinn-solver -p pinn-core
--release -- --test-threads=1`): 546 passed, 1 failed — the same pre-existing, unrelated
`compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key` failure this project has
tracked since before this stage began. Zero regressions from the ansatz generalization, either
Rust-side diagnostic bug fix, the live-Kt diagnostic addition, or the FEM tool's own BC/pin
fixes (Python-side, verified by its own separate 16-test suite, `tools/test_multi_hole_
reference.py`).

## Issue #78 Stage 2 second follow-up: `CoordinateEmbedding::MultiHoleChart` closes a real gap, doesn't move Kt — five hypotheses now falsified

Re-examining the "generalize kinematic decomposition to N holes" framing above (before
implementing it) found it doesn't hold up: decomposition's only real effect for the hard-
constraint path (`affine_strain_pair`'s contribution to `PhysicalPotentialEnergyTerm`) was
ALREADY generalized to N holes by this stage's own first bug fix (`decomposed ||
hard_constraint_active()`). Nothing was left to generalize under that name — corrected here
rather than silently re-doing complete work. The real remaining single-hole-only accuracy
machinery was `UserGeometry::coordinate_embedding()`: a single-hole geometry gets 7 explicit
hole-relative features (`SingleHoleChart` — polar-like invariants relative to the hole's own
center/radius) baked into the network's own input; every multi-hole geometry fell back to bare
`Raw` (3 columns, zero geometric hole-awareness) — `SingleHoleChart`'s own doc comment already
flagged this precise gap ("intentionally deferred until it has an unambiguous benchmark" — this
stage's own repeated-identical-Kt finding is that benchmark).

**Closed**: `CoordinateEmbedding::MultiHoleChart` (`pinn-core/src/user_geometry.rs`) computes
the same 7 features independently for EVERY hole (not just `Free` — a `Fixed` hole needs
geometric awareness too) and concatenates them (`3 + 7*N` input width); `network::multi_chart_
embed` is the forward-pass counterpart, proven byte-identical to `SingleHoleChart`'s own
`chart_embed` at N=1 and correct per-hole at N=2 (3 new tests). `holes.len()==1` is untouched —
every single-hole spec's embedding stays exactly what it was. `CoordinateEmbedding` can no
longer be `Copy` (only `Clone`) now that it holds a runtime-length `Vec` — a real, disclosed,
mechanical ripple the compiler enumerated exhaustively, fixed with `.clone()` at each call site
(cheap, never a hot-path cost).

**A real, separate bug this exposed**: `probe_boundary_residuals`'s hole-ring loop used a bare
`geometry.coordinate_embedding()` instead of the model-aware `embedding_for_model(model,
geometry)` this file's every other diagnostic probe already uses — harmless while every multi-
hole geometry's "default" embedding was `Raw` (any model built for it was also 3-wide, so the
two always coincidentally agreed), but a real shape-mismatch panic the instant a wider
`MultiHoleChart` default no longer matched a deliberately-narrower test model
(`probe_load_transfer_handles_a_geometry_with_holes` caught it - an existing test, not a new
one). Fixed to match the established convention.

**Real result: closes a genuine architectural gap, but did NOT move the trained Kt.** A live
per-step check showed the IDENTICAL `hole0=2.507, hole2=2.507` with the wider, hole-aware
embedding as without it — the same value this stage's other three hyperparameter experiments
already landed on. A fifth experiment (extending `hole_bias_fraction`'s sampling bias to the
`Fixed` hole too, not just `Free` holes — a real, plausible, cheaply-testable hypothesis, since
the `Fixed` hole got zero extra near-hole collocation density) also reproduced the identical
result and was reverted (no evidence to justify shipping it). **Five independent structural/
hyperparameter axes, all cleanly falsified**: collocation density (2x), network capacity (2x),
learning rate (5x), coordinate embedding (Raw→N-hole-aware chart), and Fixed-hole sampling
bias. This is strong evidence AGAINST a representation/optimization-search explanation and FOR
a genuine property of the Π functional itself as currently formulated for this geometry — i.e.
Kt≈2.507 plausibly IS the true minimizer of the sampled, measure-aware Monte-Carlo Π estimate
this codebase computes, and the remaining ~12-20% gap versus FEM reflects a real difference
between that estimator and a direct-stiffness-matrix FEM solve for a multi-hole domain
specifically — not a training deficiency. **This is now a formulation-level question** (auditing
the Π estimator's own correctness for N holes — quadrature-weight area bookkeeping, FD stencil
step size relative to the smallest hole's radius, etc.), not a hyperparameter-search one, and
is out of scope for further blind iteration — a real, scoped, disclosed item for a future,
more rigorous mathematical audit, not a silently-accepted ceiling.

Full regression suite after this follow-up: same 546 passed / 1 pre-existing failure, zero
regressions, plus 3 new `multi_chart_embed` tests and 1 updated `pinn-core` test
(`coordinate_embedding_preserves_raw_no_hole_and_generalizes_multi_hole_inputs`, renamed from
its own pre-#78 name to reflect the real, intentional behavior change it now asserts).

## Issue #78 Stage 2 third follow-up: the actual root cause of the Kt gap - found and fixed

The "formulation-level question" framing above was too pessimistic. Full derivation in
`docs/multi-hole-fem-ground-truth-investigation.md`'s "Fourth pass" section; short version:

**The decisive proof**: a real trained model's own measured Kt (`2.5072`) matches the PURE
closed-form `MultiHoleHardConstraint` baseline (no network contribution at all) evaluated at
the SAME real FD-safety-margin radius (`2.5075`, computed independently) to four decimal
places. The network's own learned correction near the Free holes contributes essentially
nothing.

**Why**: Kt is necessarily measured at `r = hole.radius + margin` (never exactly at the
boundary, an FD-stencil-safety requirement). For this codebase's real shipped multi-hole
geometries, `margin` is only ~5-7% of the hole's own radius, and `traction_free_envelope`'s
own saturation rate (tuned so `phi(3·radius)≈0.98`) gives `phi ≈ 0.44%` at that tiny distance.
The network's own gradient-trainable output is multiplied by `phi` before being added to the
closed form - meaning the GRADIENT reaching the specific weights that would learn a local
correction is ALSO scaled by ~0.44%, a severe, real, precisely-quantified vanishing-gradient
bottleneck exactly where Kt is read. This is exactly why none of the five (then six, after
testing `LAM_HOLE_FIXED` 50→5 too) previously-falsified hyperparameter axes ever moved the
needle - none of them change what `phi` numerically IS at the margin.

**The fix**: `traction_free_envelope_scaled(x, y, a, saturation_scale)` - a faster-saturating
generalization (`saturation_scale=1.0` byte-identical to the original), still exactly `0` with
exactly zero derivative at the true boundary (the hard constraint itself is untouched). Applied
via a new `HoleTractionFreeAnsatz.saturation_scale` field - strictly `1.0` at N=1, load-bearing
since `new_with_hard_constraint_ansatz` is the SAME function PH4-42's own verified L5 result goes
through. Verified zero regression two ways: unit tests proving the N=1 gate, AND a real
end-to-end re-run of `issue_77_l5_hard_constraint.toml` (N=1) giving `Kt=2.4444` - matching
PH4-42's own documented converged value (`2.444127304`) to 4 significant figures.

**The scale itself is DERIVED from geometry, not hand-picked - this was a deliberate correction
mid-investigation, not the first design.** A first pass tried a flat `MULTI_HOLE_SATURATION_
SCALE` constant (`15.0`, then `30.0`) applied to every hole regardless of its own size. The user
asked whether this needed to be a fixed input at all, wanting the app to derive it instead of
requiring a hand-tuned magic number - it does now: `multi_hole_saturation_scale(hole_radius,
margin)` solves `traction_free_envelope_scaled`'s own formula for the `scale` that makes `phi`
reach a dimensionless target (`TARGET_PHI_AT_MARGIN = 0.9`) at the real Kt-measurement point
(`scale = sqrt(-ln(1-0.9)) / (margin/hole_radius)`, `margin` = the same `ring_anchor_margin_m`
the Kt diagnostic already uses). Only the dimensionless target remains a constant - a raw scale
magnitude no longer needs re-guessing per geometry, it's computed from each hole's own
radius/margin ratio.

**Real, measured improvement** (`triple_hole_plate.toml`, N=2 Free holes, `scale≈22.75`
computed automatically for this geometry): hole0/hole2 Kt went from `2.507`/`2.507` (no fix) to
`2.930`/`2.958` (FEM ≈2.98-3.06/2.97-3.02) - within a few percent of FEM, matching the quality
of the hand-picked `scale=30` trial (`2.973`/`2.956`) without hand-tuning either number to this
specific geometry. Training dynamics genuinely healthier: `grad_norm` stays real and nonzero
throughout instead of collapsing then oscillating, and per-step Kt visibly moves during training
instead of being bit-for-bit frozen. Full regression: unchanged count (553 passed, +4 new/
updated tests for the derivation), same 1 pre-existing unrelated failure, zero regressions.

**Honestly still open**: hole0's Kt convergence check still flags `NOT converged` (radial
Δ≈0.147) at this higher saturation rate, matching the same open caveat the `scale=30` hand-picked
trial already had - close to FEM but not yet a fully settled number; the constitutive-residual
max also rises alongside Kt accuracy, a real trade-off not yet characterized. `TARGET_PHI_AT_
MARGIN=0.9` itself is not swept (chosen as the dimensionless analogue of the best-measured hand-
picked trial). A genuinely trainable `saturation_scale` (a `burn` `Param` updated by gradient
descent, rather than solved in closed form) was considered and set aside - the closed-form
derivation already removes the hand-tuned magic number; making it co-trained with a highly
nonlinear envelope function is separate, larger engineering scope with its own risk.

## Issue #78 fourth follow-up: `TARGET_PHI_AT_MARGIN` derived algorithmically, Kt convergence
## check decomposed into real vs. expected curvature - honest, not fully "fixed"

User asked whether `TARGET_PHI_AT_MARGIN` itself could be derived rather than a user/hand-picked
input, and to fix the "NOT converged" convergence-check flag. Both done, with a real mid-course
correction along the way (caught before shipping wrong code, not after).

**`TARGET_PHI_AT_MARGIN` is now derived, not hand-picked.** As `target→1`, the envelope's own
physical transition length `L = hole_radius/scale` shrinks; once `L` gets smaller than the real
FD stencil step, finite-difference strain stops resolving the envelope's own curvature
accurately - the actual mechanism behind the earlier session's "constitutive residual rose
alongside Kt accuracy" observation. Requiring `L ≥ ENVELOPE_FD_RESOLUTION_FACTOR·fd_step` and
solving for the tightest `target` satisfying it gives `target_phi_at_margin(margin, fd_step) =
1 - exp(-(margin/(ENVELOPE_FD_RESOLUTION_FACTOR·fd_step))²)` - closed-form, no per-geometry
hand-tuning. **A first attempt reused `RING_ANCHOR_SAFETY_FACTOR` for this new factor - checked
by hand BEFORE writing code and found to collapse the target to a fixed `1-exp(-1)≈0.632` for
every geometry, mathematically the WORST choice** (`dphi/dr` is MAXIMIZED near `target≈0.632`,
not minimized - `RING_ANCHOR_SAFETY_FACTOR` answers a different question, a category error
caught in time). Used a new, honestly-separate constant (`ENVELOPE_FD_RESOLUTION_FACTOR=2.0`)
instead. For `triple_hole_plate.toml`'s real geometry this derives `target≈0.9817`, giving
`scale≈30` - matching the earlier session's own best hand-picked trial almost exactly, a real
coincidence confirmed by direct calculation, not fudged.

**The Kt convergence check's "NOT converged" flag: root-caused with real decisive evidence, not
fully cleared.** First attempt: pick the check's second radial probe from the envelope's own
saturation curve instead of a flat `1.5x` multiplier. Real end-to-end testing showed this was a
near NO-OP at the derived scale (the envelope saturates fast enough at `scale≈30` that both
probes already sit past its steep zone). A dedicated diagnostic test
(`kirsch_hole_correction::closed_form_only_kt_varies_meaningfully_between_the_two_radial_probe_
points_at_real_scale`) then found the real dominant driver: the PURE closed-form baseline (zero
network contribution) alone varies **7.35%** between the two radial probes for this real
geometry - most of a real trained run's own ~10-12% raw radial Δ. This is genuine, expected
physical field curvature this close to a hole boundary, not a training-convergence signal.

**Real fix**: `closed_form_only_kt_at_margin` (pure host math, `ansatz.additive()`-based, zero
model/network involved) and a new `KtConvergenceReport.radial_residual_kt_delta: Option<f64>`
field - the network's OWN residual-correction drift between the two radii, ADDITIVE to (never
replacing) the existing raw `radial_relative_change`/`converged` gate. `user_runner.rs`'s
printed diagnostic now shows both with an explanatory note.

**Honest real result** (`triple_hole_plate.toml`-shaped fixture, derived `target≈0.9817`):
hole0/hole2 Kt=2.884/3.032 (close to FEM ≈2.98-3.06/2.97-3.02). Raw radial Δ still flags "NOT
converged" for both Free holes (0.102/0.119) - but now with decomposed evidence:
`network-residual Δ`=0.107/0.173 (Kt-units), proving a real, non-negligible amount of genuine
network-side variation remains even after subtracting the known closed-form curvature. **Item 1
is not "fixed" in the sense of "the flag now clears"** - it's fixed in the sense of replacing a
single opaque, curvature-contaminated number with a decomposed, honest diagnostic a structural
engineer can actually act on, backed by real evidence of what's really driving it. N=1
(`issue_77_l5_hard_constraint.toml`) stayed byte-verified unchanged: `Kt=2.4447` (matches
PH4-42's documented value); `network-residual Δ=0.000` there (the single-hole ansatz is exact,
so a near-zero residual is the correct expected result, not a bug).

**A real bug found and fixed during verification, not production code**: the first cross-check
test comparing `closed_form_only_kt_at_margin`'s generic (`ansatz.additive()`-based)
computation against an independent, hand-derived reimplementation in `kirsch_hole_correction.rs`
failed by ~40% - traced via a dedicated isolation test
(`debug_ansatz_additive_matches_direct_kirsch_hole_displacement_sum_at_one_point`) to the TEST
itself omitting the affine uniaxial background the reference computation included, not a
production bug. Fixed the test, not the implementation - confirmed by the isolation test's own
tight agreement (~1e-6 relative, `f32`-precision-limited) once the actual mismatch was found.

19 new/updated unit tests. Full regression suite: 563 passed (+10 net new), same 1
pre-existing unrelated failure (`compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key`),
zero regressions.

