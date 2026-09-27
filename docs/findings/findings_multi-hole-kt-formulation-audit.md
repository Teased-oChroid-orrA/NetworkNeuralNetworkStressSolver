# Multi-Free-hole Kt formulation audit, and the persistent-AMR/hole-bias interaction bug

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. Independent zero-build closed-form re-derivation proving the saturation_scale fix broke the formulation ceiling, and a real bug found while cross-checking against FEM: `hole_bias_fraction`/`hole_bias_include_fixed` are silently inert on the GUI-streaming path whenever persistent-adaptive AMR is active (the default).

## Multi-Free-hole Kt formulation audit: the "5-axis-falsified, formulation-level ceiling"
## framing is STALE - the saturation_scale fix (already shipped) already broke that ceiling

A later session was handed a standing audit task built on the Stage-2-second-follow-up framing
above ("collocation density/capacity/LR/embedding/Fixed-hole-bias all falsified, Kt≈2.507-2.524
regardless, formulation-level ceiling, not a training problem"). Re-reading this file's OWN
chronologically-later sections (the "third follow-up" and "fourth follow-up," which the audit
handoff had not fully reconciled against) shows that ceiling was already correctly diagnosed and
already broken: `traction_free_envelope_scaled`'s vanishing-gradient bottleneck at the Kt
measurement margin (network contribution ≈0.44% suppression at `scale=1.0`) is exactly why all
five axes were inert, and the geometrically-derived `saturation_scale`/`TARGET_PHI_AT_MARGIN`
fix already shipped in this tree resolves it. This section closes the audit with **independent,
zero-build evidence** (a standalone Python re-derivation of `kirsch_hole_displacement`/
`multi_hole_additive`/`ring_anchor_margin_m`/`target_phi_at_margin`/`multi_hole_saturation_scale`,
transcribed from the current on-disk Rust source, not recalled) that the fix is real and the
gap is no longer a formulation ceiling.

**Method**: for `triple_hole_plate.toml`'s exact shipped parameters (`half_w=0.15, half_h=0.06,
E=71.7e9, nu=0.33, px=6.9e7, py=0, fd_h=1e-3`, two Free holes at `(∓0.06, 0.02)` r=0.009, one
Fixed hole at `(0, -0.02)` r=0.007), computed three closed-form-only (zero network) Kt values at
the real margin (`ring_anchor_margin_m=6e-4 m`, confirmed `= RING_ANCHOR_SAFETY_FACTOR(4.0) *
fd_h * max(half_w,half_h)` exactly as the Rust source defines it) via a direct 4-point central-FD
von-Mises sweep (`h=1e-7`, 720 angular samples) - same formula `closed_form_only_kt_at_margin`
implements, reimplemented independently in Python rather than calling the Rust test:

| Quantity | Result |
|---|---|
| `target_phi_at_margin(6e-4, 1.5e-4)` | 0.981684 (exact match to this file's own documented ≈0.9817) |
| derived `saturation_scale` for r=0.009 | 30.0 (exact match to this file's own documented ≈30) |
| **closed-form-only Kt, both holes interacting** (affine + BOTH Free holes' additive correction) | **2.5075** (both holes, by mirror symmetry) |
| **closed-form-only Kt, single-hole-only** (affine + only that hole's own correction, no interaction term) | **2.5221** (both holes) |
| → N=2 additive-superposition interaction error at this exact geometry/margin | **0.58%** relative (`(2.5221-2.5075)/2.5221`) - a real, now precisely quantified instance of the qualitative "0.1-4.4%" range this file cites elsewhere, not a new number in conflict with it |

**The decisive comparison**: the closed-form-only baseline (2.5075, IDENTICAL to the
pre-saturation_scale-fix trained Kt this file's Stage-2-second-follow-up section reports for
this exact geometry) is independent of `saturation_scale` by construction (the envelope only
gates the NETWORK's contribution; the additive closed form never uses it) - so 2.5075 is what
Kt looks like when the network contributes nothing, at ANY scale. The fourth-follow-up section's
own real post-fix trained result for this geometry (hole0/hole2 Kt=2.884/3.032) is **13-21%
higher** than both closed-form baselines above, in a geometry-symmetry-breaking way. That is
only possible if the network's own learned correction near each Free hole is now substantial -
exactly the opposite of the pre-fix state, and exactly what the saturation_scale fix was
designed to unlock. **This closes audit angles 2 and 3 from the original task with hard,
independently-reproduced numbers**: the additive-superposition interaction error (0.58%) is
real but small and was never the dominant term; the network is no longer inert near the holes.

**What this means for the remaining ~2-8% gap to FEM (≈2.98-3.06)**: it is evidence AGAINST a
residual formulation ceiling and FOR ordinary incomplete training convergence - consistent with
this file's own already-disclosed, not-newly-discovered caveat that hole0's Kt convergence check
still flags `NOT converged` post-fix. The mirror-symmetry violation itself (hole0=2.884 vs.
hole2=3.032, ~5% apart, for a geometry whose TRUE solution must be exactly symmetric about
`x=0` since the Fixed hole sits on that axis) is a real, previously-uncommented-on signal of
uneven per-hole training convergence, not measurement noise - worth watching in any future
retrain of this geometry, but not chased further here (would require a real GPU training run,
out of scope for a zero-build closed-form audit).

**`ring_anchor_margin_m`'s plate-scaled-not-hole-scaled margin (the other open audit angle)**:
confirmed genuinely moot for every currently-shipped multi-hole geometry, not just assumed -
`triple_hole_plate.toml`'s two Free holes share the same radius (0.009 m each), so the same
plate-derived margin produces the same `margin/radius` ratio and the same derived
`saturation_scale` for both. Still a real, disclosed, untested risk for a FUTURE geometry with
unequal-radius Free holes (a smaller hole would get a proportionally larger `margin/radius`,
hence a smaller derived `saturation_scale`, hence a slower-recovering envelope right where that
hole's own Kt is measured) - not fixed here since no shipped geometry currently exercises it,
matching this file's own "BLOCKED documented with evidence is not the same as complete" standard
applied in the other direction (correctly-scoped "not yet exercised," not silently ignored).

Reproduction: `/tmp/.../scratchpad/closed_form_n2_kt.py` (this session's scratchpad) - pure
Python, no `cargo` build, transcribes `kirsch_hole_displacement`/`affine_strain`/
`ring_anchor_margin_m`/`target_phi_at_margin`/`multi_hole_saturation_scale` verbatim from the
current Rust source (line references: `user_problem.rs:163-189,228-232,344-346`,
`kirsch_hole_correction.rs:73-109`). No Rust code changed by this audit - it is a pure
verification pass confirming already-shipped code behaves as its own doc comments claim.

## Real bug found while chasing item 1 of the audit handoff: `hole_bias_fraction`/
## `hole_bias_include_fixed` are silently INERT on the GUI-streaming path whenever
## `amr_enabled=true` (the default) - and this likely invalidates the earlier-documented
## "9.18%→2.93% mirror-asymmetry" Fixed-hole-bias finding for that exact trial

The handoff's item 1 asked for a real FEM cross-check of the Fixed-hole-bias section above
("near-Fixed-hole collocation-density bias... mirror-asymmetry 9.18%→2.93% at 1000 steps").
Re-running the EXACT unmodified control/treatment trial
(`examples/problems/notched_plate_fixed_hole_bias_{control,treatment}.toml`, byte-identical to
their original commit `4016878` - confirmed via `git diff`) through the SAME tool the original
finding cites (`cargo run -p pinn-solver --release --features ndarray-backend --example
audit_trace`, deterministic NdArray backend for a controlled A/B) produced a **decisive,
unexpected result: the two runs' `updates.jsonl` outputs are byte-for-byte identical (same
MD5) across all 1000 steps** - identical `total_loss`/`energy_loss` every step, identical final
Kt for both holes (`hole0=2.8980`, `hole1=1.0866`), identical per-step angular profile. The
ONLY difference between the two spec files is `hole_bias_include_fixed` (`false` vs `true`).

**Root cause, traced and confirmed, not guessed**: a standalone diagnostic (built and run this
session, then deleted - not a permanent addition) confirmed `UserSamplingStrategy::
sample_interior` DOES correctly produce different point sets for the two flag values when
called directly - the flag's own wiring through `UserDefinedProblem::new_with_hard_constraint_
ansatz` is correct. But `runner.rs::run_user_problem_training_from`'s actual per-step loop
(the function `run_training_user_problem`/`audit_trace.rs` both go through) calls
`apply_persistent_adaptive_interior_sample` (issue #75, commit `2f32b22`, predates this trial)
**unconditionally on EVERY step, whenever `spec.training.amr_enabled && !spec.geometry.holes.
is_empty()`** (`amr_enabled` defaults to `true`) - and its own doc comment says exactly this:
"runs EVERY step..., overwriting whatever `resample_plate_step_data` drew for `data.int_norm`
moments ago... active from step 0." When it fires (confirmed via the run's own `[persistent-AMR
step N] source=adaptive` log line, present from `step 0` in both runs), it **replaces both the
sampled points AND the interior weights** (`problem.set_interior_weights(persistent_weights)`
unconditionally overwrites whatever `hole_bias_quadrature_weights_including_fixed` computed a
few lines earlier in the same step). The entire `hole_bias_fraction>0.0` block - including the
`hole_bias_include_fixed`-dependent weight computation - is dead code for the rest of that
step's actual training, for ANY hole-bearing geometry with AMR on (the default).

**Scope, checked, not assumed**: this override exists ONLY in `runner.rs` (the GUI-streaming/
TUI/`audit_trace` path). `user_runner::run_headless_user_problem` (the `--headless
--problem-spec` CLI path this file's Issue #77/#78 sections' real trained-Kt numbers were
obtained through) never calls `apply_persistent_adaptive_interior_sample` - confirmed by a
direct `grep` across every `.rs` file in the crate, the two other call sites are inside
`#[cfg(test)]` fixtures deep in `user_problem.rs`'s own test module, not `user_runner.rs`. **The
extensively-documented Issue #77/#78 multi-hole Kt trained results in this file are headless-
path results and are NOT affected by this bug.** Only GUI-streaming/TUI-driven training of a
hole-bearing geometry with AMR on is affected - which is exactly what `audit_trace.rs` (and, by
the same mechanism, an interactive TUI/GUI session) uses.

**This directly contradicts the earlier "9.18%→2.93%" finding for this specific trial**, which
was measured via this same `audit_trace` tool on this same GUI-streaming path per its own
citation. Given the override is unconditional and deterministic (confirmed via NdArray, not
Wgpu), the most likely explanation for how a real, nonzero difference was ever measured between
these two byte-identical-outcome configs is that the ORIGINAL measurement ran on the default
Wgpu backend (not `--features ndarray-backend`), whose weight initialization is GPU-driven and
already independently documented elsewhere in this file as non-deterministic run-to-run
("Wgpu weight-init non-determinism," cited as a known CI/local flake source) - two separately-
launched Wgpu processes with different random initial weights would produce a genuinely
different trajectory for reasons having nothing to do with `hole_bias_include_fixed`, and
nothing in the original write-up records a re-run or a same-seed control confirming the effect
survives backend determinism. **Not confirmed which explanation is correct** (no artifact from
the original run survives to inspect) - but the override mechanism itself, and the byte-
identical reproduction under a controlled deterministic backend, are both directly verified
facts from this session, not a hypothesis.

**Practical consequence for item 2 of the audit handoff** ("does increasing `hole_bias_fraction`
further, or combining with per-hole embedding, compound the improvement") - moot for this exact
trial config as currently written: any `hole_bias_fraction`/`hole_bias_include_fixed` change
tested via `audit_trace`/GUI-streaming on this geometry will show zero effect for the same
reason, not because the hypothesis is falsified. A real test of item 2 (or a real re-run of
item 1) needs EITHER a config with `amr_enabled = false`, or a fix that makes `apply_persistent_
adaptive_interior_sample`'s own geometry-aware annulus-per-hole seeding respect `hole_bias_
include_fixed` (it already seeds an annulus around every hole present, Free and Fixed alike, per
its own doc comment - so the more likely correct fix is arguably "AMR already does this densely
for every hole regardless of the flag, and the flag/weighting machinery should stand down when
persistent-adaptive AMR is active," not "make AMR conditional on the flag too" - a real design
decision for a human to make, not resolved here).

**Real FEM ground truth obtained for this exact geometry** (`half_w=0.10, half_h=0.05`, Free
hole r=0.01 at `(-0.03,0)`, Fixed hole r=0.008 at `(0.03,0)`, `E=71.7e9, nu=0.33, px=6.9e7`), via
`tools/multi_hole_reference.py` at a converged mesh (4 refinement levels, `max_relative_change`
0.010 at the finest, still trending up ~1%/level so the true value is slightly higher than
shown): **hole0 (Free) Kt_vm≈3.21 (trending toward ~3.22-3.25), hole1 (Fixed) Kt_vm≈1.35**. The
real (byte-identical control/treatment) trained result - hole0=2.898 (~10% low), hole1=1.087
(~20% low) - is not yet close on either hole, consistent with 1000 steps on a config whose
near-Fixed-hole density intervention turned out to be a no-op; a real comparison after either
fix above would need a fresh run.

Scipy (required by `tools/multi_hole_reference.py`) was not present in this environment and was
installed into a throwaway venv (`.../scratchpad/venv`) rather than the system Python, since
`pip install --user` is blocked by this system's externally-managed-environment policy - not a
permanent environment change.

## Handoff item 1 closed: real, non-confounded AMR-on-vs-off comparison for the mixed
## Free/Fixed geometry - AMR-on is a real, verified improvement, but does not close the gap alone

The prior handoff's item 1 asked for a genuine AMR-on reading of
`notched_plate_fixed_hole_bias_{control,treatment}.toml` now that the override bug above is
understood, to compare against the `_no_amr` ablation pair already run this session. **No new
training run was needed to get this**: the runner.rs fix above is diagnostic-only (adds a
one-time `[warning]` println), it does not change `apply_persistent_adaptive_interior_sample`'s
override behavior at all - so the control/treatment run already captured earlier this session
(pre-fix, byte-identical MD5 `62d7a812...` for both, `Debug_runs/fixed_hole_bias_audit/{control,
treatment}/updates.jsonl`) is already the correct AMR-on data point. Confirmed by direct
inspection (not re-run) that `spec.architecture.hole_bias_fraction=0.4 > 0.0 &&
spec.training.amr_enabled=true && !holes.is_empty()` all hold for this config, so the new warning
condition is satisfied by construction - the print itself was not observed live this session
(would require a fresh run, which would reproduce the same byte-identical trajectory), but the
boolean logic determining it is verified by direct source inspection, not assumed.

**Real comparison, same geometry, same 1000 steps, only `amr_enabled` differs**:

| Config | hole0 (Free) Kt | hole0 peak θ | hole1 (Fixed) Kt | hole1 peak θ |
|---|---|---|---|---|
| `_no_amr` (control, `hole_bias_include_fixed=false`) | 2.8785 | **85°** | 0.9336 | 180° |
| `_no_amr` (treatment, `hole_bias_include_fixed=true`) | 2.8968 | **85°** | 0.9632 | 180° |
| original AMR-on (control/treatment, byte-identical either way) | **2.8980** | **270°** | **1.0866** | 180° |
| Fresh FEM ground truth (this session, converged mesh) | ≈3.21 | 270° (Kirsch-canonical, ⊥ to load axis) | ≈1.35 | — |

**AMR-on is a real, verified improvement on two independent axes, not noise**: (1) the Fixed-hole
Kt gap shrinks from ~31% low (`_no_amr`) to ~19.5% low (AMR-on) - a genuine ~13pp recovery; (2)
the Free hole's peak-stress angular location moves from 85° (AMR-off) to 270° - the
Kirsch-canonical location (perpendicular to the remote-tension axis), matching the FEM baseline's
own peak location exactly. The 85° peak under `_no_amr` is a previously-undocumented signal that
AMR-off training for this geometry converges to a qualitatively wrong angular stress
distribution near the Free hole, not merely an undertrained-but-correctly-shaped one - worth
flagging for any future geometry run without AMR.

**AMR-on does NOT close the gap by itself**: Free-hole Kt is still ~9.7% low (2.898 vs 3.21) even
with the angular location now correct, and Fixed-hole Kt is still ~19.5% low. So "the app must
converge" is not yet satisfied for this geometry even with the best currently-available
combination (AMR-on, `hole_bias_include_fixed` either way since it's fully inert).

**Practical recommendation, evidence-based**: for any hole-bearing geometry trained via the
GUI-streaming/TUI/`audit_trace` path (where AMR-on is the default and is strictly better on both
measured axes here), `hole_bias_include_fixed`/`hole_bias_fraction` provide no benefit and should
not be relied on - leave `amr_enabled=true` (the default) rather than reaching for the hole-bias
knob for this geometry class. Retiring the flag outright is NOT justified yet: it is only proven
inert when AMR is active, and the flag remains the only near-hole density knob on the
`--headless`/`--problem-spec` CLI path (which never runs AMR at all - confirmed by grep, zero
`amr`/`Amr`/`AMR` occurrences in `user_runner.rs`) or on any future `amr_enabled=false` config.
The remaining ~10-20% gap is real and unresolved by this experiment - see the next section for
whether it's a training-imbalance or structural issue on a different (all-Free, mirror-symmetric)
geometry.

## Follow-up, same session: the AMR divergence above was fixed at the root, not just measured -
## `--headless`/`--problem-spec` now runs the SAME sampling+AMR code as GUI/TUI/`audit_trace`

The user directed that there be exactly one source of truth for per-step sampling/AMR, with
`--headless` as that source and the other three entry points referencing it. Implemented as
`crate::user_problem::resample_step_data_with_amr` (`user_problem.rs`, next to
`resample_plate_step_data`/`apply_persistent_adaptive_interior_sample`, which it now composes):
the ONE implementation of per-step resample, the `hole_bias_fraction`-vs-AMR override check
(the warning above now fires identically on every entry point instead of citing headless as
exempt), the residual-driven AMR sweep, and persistent-adaptive AMR (issue #75). Both
`user_runner::run_headless_user_problem` and `runner::run_user_problem_training_from` now call
this one function every step instead of maintaining independent copies - `runner.rs`'s copy
(~250 lines) was deleted outright. The three annular-decomposition dispatch branches
(`SequentialTwoStage`/single/multi Free-hole) were already correctly shared via callback
injection before this change and were not touched.

**Real, decisive end-to-end verification** (not just compilation): re-ran
`notched_plate_fixed_hole_bias_control.toml` (the exact config the AMR-on-vs-off comparison
above used) via `--headless --problem-spec` for the first time with AMR active. Result:
`hole0(Free) Kt=2.8980, hole1(Fixed) Kt=1.0866` - **bit-for-bit identical** to the GUI/
`audit_trace` path's own previously-recorded AMR-on result for this same config. Headless also
now prints the `[warning] hole_bias_fraction=...`/`[persistent-AMR step N]` lines this file's
control/treatment section above describes, for the first time. Full regression suite
(`cargo test -p pinn-solver -p pinn-core --release`): 742 passed, 0 failed, 59 ignored - no
regressions from the consolidation.

## Follow-up: the `hole0=2.884/hole2=3.032` mirror-asymmetry did NOT reproduce in a fresh,
## correctly-configured run - but a much bigger, real, previously-undocumented mechanism did:
## `equilibrium` holds ~100% of SAW-BRDR gradient share for this entire 3000-step run

The on-disk `triple_hole_plate.toml` has no `[architecture]` section (so it trains the SOFT
`hole_free` penalty, not the hard-constraint ansatz) - the documented `2.884/3.032` result came
from a Rust test fixture, not this file. Created
`examples/problems/triple_hole_plate_hard_constraint.toml` (same geometry/material/load/training,
`hard_constraint_ansatz=true`, `training_procedure="SingleDomain"` to force the plain
`MultiHoleHardConstraint` path rather than `MultiAnnularDecompositionProblem`'s kinematic
decomposition, which would otherwise also qualify for this N=2-Free-hole geometry) and ran it via
`audit_trace` (3000 steps, NdArray, real training, not a unit test).

**Result: hole0/hole2 Kt=2.5104/2.5096 - genuinely symmetric (0.03% apart), not the documented
asymmetric 2.884/3.032, and not the higher magnitude either.** This is evidence AGAINST a
structural/formulation symmetry bug in the additive multi-hole correction (a correctly-configured
fresh run shows near-perfect mirror symmetry) - but it also didn't reproduce the higher,
"saturation_scale fix unlocked it" Kt value either. Not enough evidence survives to say whether
the original fixture's run had a different seed, different intermediate state, or a difference
in some setting not visible from the docs alone - genuinely unresolved, not asserted either way.

**The much bigger, decisive, real finding**: `gradient_shares` in the captured `updates.jsonl`
(`Debug_runs/triple_hole_hard_constraint_audit/`) shows the `equilibrium` loss term holds
**99.95%-100.00% of total SAW-BRDR gradient share at EVERY sampled checkpoint from step 0 through
step 2999** (sampled every 100 steps, no exception) - `interior_energy` (the term through which
the network's own near-hole stress correction is actually learned, per this file's own earlier
closed-form audit section above) never exceeds **0.00006** gradient share the entire run. Every
other term (`hole_fixed`, `outer_traction`, `external_work`) is similarly starved to
near-zero. This is not a transient spike or a late-training collapse - it is present from step 0
and never recovers, for the FULL run.

**Why this plausibly explains the low Kt (2.51, matching the OLD pre-saturation_scale-fix
"ceiling" value cited earlier in this file, not the fixed value)**: if `interior_energy` gets
functionally zero gradient share for 3000 steps, the network has almost no incentive to learn the
near-hole correction the saturation_scale fix was specifically designed to unlock - regardless of
whether that fix's own math is correct (independently re-verified earlier in this file). SAW-BRDR
is supposed to adaptively rebalance gradient contribution across terms; for this specific
ansatz + N=2-Free-hole geometry combination, it is not doing so at all, for the entire run.

**Chased to a real fix, with real A/B evidence on both affected geometries.** Added
`LAM_EQUILIBRIUM_PLATE_HOLE_BEARING` (`user_problem.rs`), selected in `UserDefinedProblem::
base_weight` by `self.spec.geometry.holes.is_empty()` - the no-hole path is byte-identical
(still `LAM_EQUILIBRIUM_PLATE = 50.0`, never touched). **Zero risk to L5 turned out to be
provable, not just plausible**: `issue_77_l5_hard_constraint.toml` uses `formulation =
"Variational"`, which never registers `"equilibrium"` as a loss term at all (enforced by its own
test assertions) - `base_weight("equilibrium")` is simply never called for L5, confirmed by
running `issue_77_l5_single_domain_hard_constraint_converges_to_fem_reference` (`--ignored`)
against the changed code: **passed**, Kt unchanged.

**Tuning process, real numbers, not guessed**: base-weight cuts of 1000x (`0.05`) and even
5,000,000x (`0.00001`) left `equilibrium`'s SAW-BRDR gradient share pinned at 0.9999-1.0000 at
EVERY checkpoint of a 300-step fast-iteration run, including step 0 (before any SAW-BRDR
adaptation could occur) - proving the raw magnitude gap between `equilibrium` and every other
term is enormous (many orders of magnitude), not a few-x imbalance a modest cut could fix. The
`0.00001` cut DID move Kt in the right direction dramatically at 300 steps (hole1 Fixed: 1.19→
1.36, near FEM's ≈1.35) - but a full 3000-step run at that value **diverged** (Kt climbing
unboundedly past FEM, 6.9/7.3 by step 2999, `refinement_converged=false`) - removing nearly all
of `equilibrium`'s stabilizing PDE-consistency pull let training run away. A 300-step snapshot
is NOT sufficient evidence of stability for this term - only a full-length run is.

**Final, verified value: `LAM_EQUILIBRIUM_PLATE_HOLE_BEARING = 0.001`** (50,000x smaller than the
no-hole value) - a real full 3000-step `audit_trace` run of `triple_hole_plate_hard_constraint.
toml` converges STABLY (no divergence, monotonic-ish rise then plateau from step ~2000) to
**hole0=2.9006, hole1(Fixed)=1.4571, hole2=2.8644, all three `refinement_converged=true`** - within
~3-5% of FEM (≈2.98-3.06/2.97-3.02 for the two Free holes) and MORE mirror-symmetric than the
original documented result (1.3% hole0-vs-hole2 gap, vs. the original 5%). Full regression suite
(`cargo test -p pinn-solver -p pinn-core --release`) re-run after this change - see this file's
own edit history / commit for the pass count at the time this was written.

**New permanent example**: `examples/problems/triple_hole_plate_hard_constraint.toml` (the
geometry/material/load this file already describes, `hard_constraint_ansatz=true`,
`training_procedure="SingleDomain"` to avoid `MultiAnnularDecompositionProblem`'s kinematic-
decomposition dispatch) - the FIRST on-disk, `--headless`/`--tui`/`audit_trace`-reachable config
for this architecture combination; previously only reachable via a Rust test fixture.

**Consequence for every prior headless-obtained Kt number in this project's history** (single-
hole L5, N-hole ansatz-only via `--headless`, any hole-bearing config run without `--tui`/GUI/
`audit_trace`): none of them benefited from AMR - they were all obtained before this fix
existed. Re-running any of them will very likely change the reported Kt (generally toward better
accuracy, per the AMR-on-vs-off comparison above) - a real, intentional behavior change, not a
regression, and not silently made: the numbers in `docs/findings/findings_multi-hole-kt-ansatz-
saturation-scale.md` and `findings_multi-hole-kt-trainable-scale-decomposition.md` (2.507-3.032
range, N-hole ansatz-only path) predate this fix and were obtained via a fixture/test harness
call path, not `--headless` directly - re-verify before citing them as current headless
behavior.
