# Investigation + plan: is a multi-hole Kt accuracy fix even tractable?

Follow-up to `docs/multi-hole-and-boundary-shape-epic.md` (issue #78, closed) - that epic
explicitly left multi-hole Kt *accuracy* unfixed. Before attempting any PINN-side fix, this
investigates whether a fix is even well-posed: is there ground truth to validate against, and
does a known approximation (superposition of single-hole corrections) have a chance of working
for this project's actual multi-hole geometries. This file is both the plan and the running
progress record - update it, not just a separate status note, as each stage completes.

## Progress

- [x] **Phase A — build multi-hole FEM ground truth.** Done (see "What was built" / "Real
  result" / "Disclosed, unresolved limitation" below). Real ground truth exists for both
  `notched_plate.toml`'s (2-hole) and `triple_hole_plate.toml`'s (3-hole) geometries; the tool
  has a known, documented mesh-density limit.
- [x] **Phase B — check superposition against the real ground truth.** Done, strongly
  positive (see "Phase B result" below) - every hole, both geometries, under 5% error at the
  cheapest possible (zeroth-order, no self-consistent correction) approximation.
- [x] **Decision point** (see "Decision point" below) - superposition-based PINN ansatz is now
  the evidenced favorite over a full N-hole kinematic decomposition. A genuine PINN-side
  implementation plan is the natural next stage, deliberately NOT started in this pass - it
  deserves its own dedicated planning session rather than being appended here.

## What this found

**No multi-hole ground truth existed anywhere in this project.** The single-hole FEM reference
(`FEM_KT=2.460638516`) came from `tools/finite_plate_reference.py`, an independent, validated
Python CST solver - but its mesh generator exploits double (quarter-plate) symmetry about a
single centered hole and cannot represent an asymmetric multi-hole domain at all.

## What was built: `tools/multi_hole_reference.py`

A new, separate tool (the existing single-hole tool is untouched - its own hardcoded reference
numbers must stay byte-reproducible). Reuses `finite_plate_reference.py`'s proven CST/PCG
numerical core (`cst_matrix`, `element_pcg`) via import; only the meshing, load/BC application
(no symmetry to exploit - traction applied on both left AND right edges, explicit rigid-body
pin+roller since there's no symmetry-derived translation/rotation fix), and point-location
(via `scipy.spatial.Delaunay.find_simplex`, not the existing tool's structured-mesh candidate
search) are new.

**New dependency: scipy** (Delaunay triangulation of a point cloud combining graded rings
around each hole + outer boundary + background fill). A deliberate, disclosed departure from
the existing tool's "NumPy only" design - a general N-arbitrary-hole domain genuinely needs a
real triangulation, and scipy's (wrapping Qhull) is far more trustworthy for a ground-truth
tool than a hand-rolled one would be.

8 new structural/regression tests (`tools/test_multi_hole_reference.py`), all passing: mesh
validation, exact circular hole boundaries, CCW/outside-every-hole triangle invariants, energy/
reaction balance, and a Kt sanity range (not a converged-value assertion - see below).

## Real result: `notched_plate.toml`'s geometry (half_w=0.10, half_h=0.05, holes at
(-0.03,0,r=0.01) and (0.03,0,r=0.008), E=71.7e9, ν=0.33, traction=69e6)

Well-converged range (28x4 through 100x14 mesh levels, monotonically approaching a limit,
relative change dropping from ~9% to ~2%):

| mesh | hole0 (r=0.01) kt_vm | hole1 (r=0.008) kt_vm |
|---|---|---|
| 28x4 | 2.774 | 2.596 |
| 48x6 | 2.888 | 2.759 |
| 80x10 | 3.001 | 2.900 |
| 100x14 | 3.063 | 2.937 |

**Both holes' Kt is landing close to the isolated-hole Kirsch value of 3.0**, not obviously
depressed or amplified by the two-hole interaction at this spacing (~3.3x radius-sum
separation).

## Real result: `triple_hole_plate.toml`'s geometry (half_w=0.15, half_h=0.06, holes at
(-0.06,0.02,r=0.009), (0.0,-0.02,r=0.007), (0.06,0.02,r=0.009), all treated as traction-free
for this check - the tool has no Fixed-BC support, and the real spec's middle hole is Fixed;
an internally-consistent Free/Free/Free comparison, not a like-for-like match to the mixed-BC
production spec)

Well-converged range (44x6 through 76x10, before the same anomaly - see below - hit at the
finest level tested):

| mesh | hole0 (r=0.009) kt_vm | hole1 (r=0.007) kt_vm | hole2 (r=0.009) kt_vm |
|---|---|---|---|
| 44x6 | 2.971 | 2.991 | 2.983 |
| 60x8 | 3.089 | 2.986 | 3.065 |
| 76x10 | 3.133 | 2.926 | 3.058 |

Same pattern: all three holes cluster around Kt≈2.9-3.15, close to the isolated value.

## Disclosed, unresolved limitation

Found on `notched_plate.toml` and independently REPRODUCED on `triple_hole_plate.toml`: past a
mesh-density threshold, a hole's Kt moves the WRONG way between successive refinements.
`notched_plate.toml` at 128x18: `hole1` (2.937 -> 2.837, ~4% regression). `triple_hole_plate.
toml` at 92x12: `hole1` — the MIDDLE hole, flanked by both others — dropped from 2.926 to 2.706,
an 11.8% regression, the biggest jump seen so far. This consistently hitting the most
"squeezed" hole (flanked on two sides) both times is real evidence for the working hypothesis:
a mesh-quality issue (thin/sliver elements) in the narrow background-grid strip between two
close hole rings' outer extents, worse the more such strips a given hole has adjacent to it.
NOT a formulation bug - energy/reaction-balance identities held to ~1e-13 at every level tested
on both geometries, including every anomalous one (the linear algebra is correct; the mesh
geometry at that specific density needs work). **The tool is trustworthy in the ranges tested
above (up to ~100x14 for the 2-hole case, ~76x10 for the 3-hole case); do not push finer for
holes at this kind of spacing without first diagnosing this.**

## Phase B result: superposition vs. real FEM ground truth

Built `tools/superposition_check.py` (pure NumPy, no PINN code, no scipy dependency): each
hole's own EXACT isolated Kirsch solution at its own boundary, plus every other hole's own
Kirsch PERTURBATION (far-field term subtracted so it's never double-counted) evaluated at that
same point - the classical "zeroth iteration" of the method of successive images, no
self-consistent correction pass.

| geometry | hole | superposition kt_vm | FEM kt_vm | relative error |
|---|---|---|---|---|
| notched_plate (2 holes) | 0 | 2.960 | 3.063 | 3.4% |
| notched_plate (2 holes) | 1 | 2.934 | 2.937 | 0.1% |
| triple_hole_plate (3 holes) | 0 | 2.994 | 3.133 | 4.4% |
| triple_hole_plate (3 holes) | 1 | 3.028 | 2.926 | 3.5% |
| triple_hole_plate (3 holes) | 2 | 2.994 | 3.058 | 2.1% |

**Every hole, both geometries: under 5% error at the cheapest possible approximation.** Real,
decisive, positive evidence that superposition is viable for this project's actual multi-hole
geometries. 4 new tests in `tools/test_superposition_check.py` (isolated-hole Kt=3.0 recovery,
traction-free-boundary identity, a real regression guard on the measured notched_plate numbers,
a qualitative "farther apart -> closer to isolated Kt" sanity check), all passing.

## Decision point

Phase A proved ground truth is obtainable; Phase B proved a cheap approximation already lands
within 5% of it, on both real shipped multi-hole geometries, with zero self-consistent
correction. The real engineering question is now "which approach to build," not "is a fix even
tractable":

- **Superposition-based PINN ansatz** (extending `kirsch_hole_correction`'s existing
  single-hole `HoleTractionFreeAnsatz` to sum N holes' own closed-form corrections - the same
  additive-correction pattern already proven for one hole) is the evidenced favorite. Real
  remaining engineering work: making the ansatz differentiable/batchable in `burn` for N holes
  (straightforward - a closed-form sum, no new physics), and deciding whether the
  self-consistent correction pass is even needed given how close zeroth-order already lands.
- **Full multi-hole kinematic decomposition** (N+1 domains) is now the evidenced LOWER
  priority - Phase B's own numbers give no indication it's needed for this project's realistic
  hole spacings.

**Not started in this pass** - a genuine PINN-side implementation plan deserves its own
dedicated planning session.

## PINN-side implementation (follow-up session): N-hole hard-constraint ansatz

The dedicated planning session called for above happened, produced an approved plan
("N-hole hard-constraint ansatz: extending the single-hole Kirsch correction to N Free
holes"), and was implemented, verified, and is now real evidence - not a foregone conclusion,
per that plan's own Verification section.

**What was built** (`crates/pinn-solver/src/kirsch_hole_correction.rs`,
`crates/pinn-solver/src/user_problem.rs`): `AnnulusAnsatz::MultiHoleHardConstraint(Vec<
HoleTractionFreeAnsatz>)` - one closed-form Kirsch correction per `HoleBc::Free` hole, product
of envelopes for the multiplicative suppression (exact zero at every hole's own boundary
regardless of N, proven by `multi_hole_envelope_is_exactly_zero_at_every_holes_own_boundary_
for_two_holes`), sum of corrections for the additive baseline (only approximately
traction-free for N>1, matching Phase B's own 0.1-4.4% measured residual - proven numerically
by `multi_hole_additive_residual_traction_is_small_and_matches_the_measured_interaction_
order`). N=1 reduces byte-identically to the pre-existing single-hole `HardConstraint` path
(`multi_hole_reduces_to_single_hole_hard_constraint_when_n_equals_one`). The single-hole
`decomposition_applicable` centering/single-hole gate was confirmed to be exactly what its own
code comment always said - a deliberate first-verified-case scope narrowing, not a
mathematical constraint - so the new gate (`free_holes`) drops both restrictions: any count,
any position, every `HoleBc::Free` hole is eligible.

**A real root-cause bug found and fixed along the way, not anticipated by the plan.** The
first real training run under the new N-hole ansatz (off-center, `triple_hole_plate.toml`'s
own 2-Free/1-Fixed geometry) collapsed to Kt≈0.004-0.014 - a trivial-solution collapse, not
merely "not yet converged." Root cause: `UserDefinedProblem::loss_terms()`'s
`affine_strain_pair` mechanism (the constant far-field background strain added to the
network's own learning target under kinematic decomposition) was gated ONLY on
`decomposition_applicable` (the single-centered-hole case), never on the new N-hole
`MultiHoleHardConstraint` case. Off-center/multi-hole hard-constraint specs therefore trained
with the network having to learn the ENTIRE affine far-field background from scratch on top of
refining the hole correction - the exact gradient-competition failure mode issue #77's original
decomposition fix existed to eliminate, silently reintroduced for the one case that mattered
here. Fixed by generalizing the gate to `decomposed || hard_constraint_active()` (a strict OR,
byte-identical for every pre-existing case, since `affine_strain_pair`'s constant value is
independent of hole count/position by construction). This alone moved the trained Kt from
≈0.5 (after just fixing the formulation to Variational+measure_aware) to ≈2.5.

**A second, separate, real diagnostic bug found while investigating a leftover P2-09
trivial-solution warning that persisted even after the Kt fix above.** `probe_load_transfer`/
`probe_reaction_force`/`probe_boundary_residuals` (`user_problem.rs`) had never been updated to
accept an `ansatz`/`affine_strain_pair` parameter at all - they read the model via a bare
`fwd_embedded` forward pass, completely bypassing any active `AnnulusAnsatz` and any affine
background, for ANY spec (this bug predates issue #78 and would have affected the original L5
hard-constraint case too, just never noticed because that case's real verification went through
a different, correctly-wired diagnostic function, `run_user_problem_training_with_
diagnostics`/`user_problem_l5_diagnostic`, not `run_headless_user_problem`'s own printed
diagnostics). Fixed by threading `ansatz: &dyn DirichletAnsatz, affine_strain_pair:
Option<(f64,f64)>` through all three functions (mirroring `probe_hole_boundary_profile_
derived`'s own established PH4-41 pattern exactly - `stencil_forward_with_ansatz` instead of
`fwd_embedded`, affine added to FD-derived strain before `compute_stress`) and updating every
call site (headless CLI diagnostic loop, GUI-streaming vis-cadence block, GUI checkpoint-save
serving loop, GUI checkpoint-load serving path (kept `IdentityAnsatz`-only there deliberately -
checkpoint metadata doesn't carry `ProblemSpec.architecture` yet, a real disclosed gap, not
silently pretended away), `run_no_hole_benchmark` (kept `IdentityAnsatz`-only, correctly - a
no-hole geometry has no ansatz concept), and 8 pure-logic tests). This closed the remaining
false P2-09 warning: load transfer ratio went from 0.02-0.07 (spurious "collapsed solution")
to 1.00-1.01 (genuinely healthy) with no change to the actual trained model - purely a
diagnostic-reconstruction fix.

**Real trained result, both geometries, `formulation="Variational"` +
`measure_aware_training=true` (the same formulation PH4-42's own L5 verification used - the
codebase's own `default_formulation()` is the pre-#77 Hybrid path and does NOT get this fix's
benefit; that mismatch was itself an early false trail in this session's own investigation),
3000 steps, `hole_bias_fraction=0.4`, both fixes applied:**

| geometry | hole (BC) | trained Kt | FEM ground truth (this doc, Free/Free/Free) | superposition (Phase B) |
|---|---|---|---|---|
| `notched_plate.toml` (N=1 Free, off-center) | hole0 (Free) | 2.694 | 3.063 | 2.960 |
| `triple_hole_plate.toml` (N=2 Free, off-center) | hole0 (Free) | 2.507 | 3.133 | 2.994 |
| `triple_hole_plate.toml` | hole2 (Free) | 2.507 | 3.058 | 2.994 |
| `triple_hole_plate.toml` | hole1 (Fixed) | 1.048 | 2.926 (Free/Free/Free, not comparable) | 3.028 (same caveat) |

Load transfer ratio 1.00-1.01, reaction-force equilibrium error ~1% on every Free-hole run -
genuinely healthy, non-collapsed solutions, not just "Kt looks less wrong."

**Honest assessment - real, substantial improvement, not full convergence.** Trained Kt for
every `HoleBc::Free` hole landed within 12-20% of FEM ground truth, up from off by ~6-600x
before this session's two bug fixes. This is NOT the 0.67-1.23% PH4-42 achieved for the
single-centered-hole case - two known, disclosed reasons, both consistent with the plan's own
predicted nuance: (1) off-center/multi-hole hard-constraint specs get the ansatz's own
correction but never the single-hole path's OTHER accuracy machinery (hole-biased
`coordinate_embedding`/log-polar reparameterization, `SequentialTwoStage` training procedure -
all still gated single-hole-only, out of this pass's scope per the approved plan's own "Single-
domain only" design section); (2) the additive baseline itself is only approximately
traction-free for N>1 (Phase B's own 0.1-4.4% closed-form residual, now also present in the
trained network's own starting point, not just measured in isolation).

**The flat-loss-plateau observation above has since been definitively investigated (this
session's own follow-up) - real evidence in hand, not just a flagged hypothesis.**

## Follow-up: the flat-loss plateau, and a real FEM ground-truth bug found solving the BC-mismatch caveat

Two more real issues, both raised by the user reviewing the numbers above, both investigated
to a real conclusion (one fixed, one definitively diagnosed but not yet closed).

### BC-mismatch bug found and fixed: `multi_hole_reference.py` never supported Fixed holes

Every FEM number in this document up to this point was computed treating EVERY hole as
traction-free (`multi_hole_reference.py` had no Fixed-BC concept at all), while the real
PINN specs (`notched_plate.toml`/`triple_hole_plate.toml`) have one genuinely `Fixed`
(zero-displacement) hole each - a real, disclosed BC mismatch. Fixed by adding a `bcs`
parameter to `solve()` (`"free"`/`"fixed"` per hole, default all-`"free"` = byte-identical to
every pre-existing call) - a `"fixed"` hole gets a real Dirichlet (zero-displacement)
constraint applied to every mesh node on its own exact boundary ring (`free[dof]=False` for
both `u`/`v`, the same `element_pcg` free-DOF mechanism the tool already used for its rigid-
body pins), proven zero to `1e-15` at those nodes by a real regression test.

**A second, more serious, real bug found WHILE building and verifying the first one - the
tool's own rigid-body-motion pin scheme was never actually mirror-symmetric, and this had been
silently invisible until a Fixed hole exposed it.** `solve()`'s original pin scheme (full
`ux=uy=0` at the left edge midpoint, `uy=0`-only roller at the right) is asymmetric by
construction. This never mattered for the all-`Free` geometries this tool shipped with before:
a self-equilibrated external load (uniaxial tension, zero net force/moment) plus zero-reaction
`Free`-hole boundaries left the pins carrying zero reaction force regardless of where they sat,
so an arbitrary asymmetric CHOICE of zero-force gauge never perturbed the solution. The instant
Fixed-BC support was added, `triple_hole_plate.toml`'s own real geometry (two mirror-symmetric
off-center Free holes flanking one on-axis Fixed hole) exposed it: the two Free holes' Kt,
which MUST be identical by physical mirror symmetry, came out ~30-50% different (e.g.
kt0=1.858, kt2=2.974 at one mesh level) - a real, reproducible, deterministic discrepancy
(confirmed via a geometry-mirroring cross-check that ruled out mesh/solve nondeterminism:
solving the exact mirrored geometry reproduced the SAME per-position values to full precision,
proving the asymmetry was coming from the solve itself, not noise). Root cause: a `Fixed` hole
IS a genuine internal support with a real, generally nonzero net reaction; the old asymmetric
external pin then had to carry part of that reaction asymmetrically, measurably breaking
left/right symmetry - invisible before only because Free-hole "reactions" are always exactly
zero.

**Fixed** by replacing the pin scheme with a mirror-symmetric one: `uy=0` at BOTH the left and
right edge midpoints (a symmetric pair - `uy` is even under `x`-mirror), `ux=0` at a single
node as close as the mesh allows to the bottom edge's own midpoint (`x≈0`, the mirror axis
itself - `ux` is odd under `x`-mirror, so anchoring it AT the axis is the only single-point
choice that doesn't privilege a side). Verified two ways: (1) the exact symmetric geometry now
gives kt0/kt2 within 0.6% (was 30-50%); (2) a direct stress-field comparison on the SAME mesh
between the old and new pin choice, for an all-`Free` geometry, shows the two schemes give
IDENTICAL stress fields to `~1.6e-10` relative precision - proving the fix is a pure bug fix
with zero effect on every all-`Free` number already in this document (Phase A/B's own tables
above are unaffected and remain valid). 6 new regression tests in
`tools/test_multi_hole_reference.py` (BC validation, exact zero-displacement proof, the
mirror-symmetry regression proof itself, and a general gauge-invariance identity proof).

**Real, corrected, BC-matched FEM ground truth** (well-converged range, before the
already-documented fine-mesh anomaly re-appears - it does, at the same kind of density, for
the Fixed-hole case too, confirming it is a general mesh-quality property of this tool's
background/ring transition, not something specific to the BC bug just fixed):

| geometry | hole (real BC) | corrected FEM kt_vm |
|---|---|---|
| `notched_plate.toml` | hole0 (Free) | ≈2.96-3.02 |
| `notched_plate.toml` | hole1 (Fixed) | ≈1.68-1.72 |
| `triple_hole_plate.toml` | hole0 (Free) | ≈2.98-3.06 |
| `triple_hole_plate.toml` | hole1 (Fixed) | ≈1.49-1.56 |
| `triple_hole_plate.toml` | hole2 (Free) | ≈2.97-3.02 |

Compared against the real trained PINN Kt from this document's own earlier section
(notched_plate hole0=2.694, hole1=1.048; triple_hole_plate hole0/hole2=2.507/2.507, hole1=
1.048): the Free-hole comparison changes little (still ~10-19% low, now against the CORRECT
reference for the right reason instead of coincidentally similar numbers against the wrong
one), but the Fixed-hole comparison - previously disclosed as "not comparable" - is now
genuinely meaningful, and shows the PINN's own Fixed-hole Kt reads substantially LOW (~30-38%)
against real ground truth. This is real, new, actionable information: the soft `hole_fixed`
penalty (a `mean(u²+v²)` anchor) satisfies its own zero-displacement target almost exactly
(raw residual ~1e-11 to 1e-14 throughout every training run) but that alone does not guarantee
the DERIVED stress reading near it is accurate - a real, disclosed gap in how well this
codebase's existing Fixed-hole treatment reconstructs the local stress field, not investigated
further in this pass.

### The flat-loss plateau: real evidence, not a masked-progress artifact - definitively diagnosed, not yet closed

A new per-step Kt diagnostic (`user_runner.rs::run_headless_user_problem`, printed at the same
10-checkpoint cadence as the loss line - the first time this codebase has ever measured Kt
DURING training rather than only at the end) answers the open question directly: **Kt reaches
its final value almost immediately (within the first ~300 steps) and then genuinely does not
move for the remaining ~2700 steps** - real, measured (`hole0=2.5382→2.5070→2.5071→2.5071...`
across 4 checkpoints spanning steps 0-900). This is NOT a case of total_loss hiding continued
Kt improvement behind a noisy scalar metric - the network has genuinely converged to a stable
point, and training stalling there is real convergence, not a training bug or stuck optimizer.

**Three independent, cheap experiments, each changing a major hyperparameter axis, each
producing the IDENTICAL converged Kt (to 3-4 significant figures) and the identical total_loss
value (~0.967) by step 300:**
1. `hole_bias_fraction` 0.4 → 1.0 (doubling near-hole collocation density per Free hole, from
   0.2 to 0.5 - matching PH4-42's own verified single-hole density exactly): Kt=2.5070→2.5076.
   No change.
2. `hidden_dim` 64 → 128 (doubling network capacity): Kt=2.5070→2.5071. No change.
3. `training.lr` 1e-3 → 5e-3 (5x peak learning rate): Kt=2.5078→2.5070 at the same checkpoint.
   No change.

This rules out collocation density, network capacity, and optimizer step size as the cause -
three of the most common "just needs more of X" explanations for undertraining, all cleanly
falsified with real numbers. A supporting data point: `superposition_check.py`'s own closed
form, evaluated with ONLY the two Free holes' own corrections summed (matching
`MultiHoleHardConstraint`'s exact scope - the Fixed hole contributes no closed-form correction
to the ansatz at all), gives kt≈2.985 - closer to the corrected FEM ground truth than the
TRAINED network's own 2.507, meaning the network's own learned residual is moving the field
AWAY from the closed form's already-reasonable estimate, not refining it further.

**Working conclusion (not yet verified by implementing it): the remaining gap is a structural/
formulation limitation, not a tuning problem, and the earlier framing ("the ansatz-only
approach vs. the fully-generalized kinematic decomposition") from this session's own live
discussion is the most likely candidate.** The single-hole L5 case's real 0.67-1.23% accuracy
came from TWO combined mechanisms: the hard-constraint ansatz (generalized to N holes this
session) AND kinematic decomposition (`u = u_affine + u_hole`, which lets the network learn
ONLY a small residual beyond both the affine background and the hole's own closed-form
correction - still scoped to `decomposition_applicable`'s single-centered-hole case only, NOT
generalized this session). The N-hole ansatz alone gives the network a LARGER, less-constrained
learning target than the single-hole case ever had (the full field minus only the multiplica-
tive-envelope-suppressed residual, with no additional loss-term-level simplification beyond the
affine constant this session's other fix added) - consistent with training converging quickly
and robustly to a DIFFERENT, worse local optimum regardless of capacity/data/LR, because the
loss landscape itself (not the optimizer's ability to search it) has a strong basin there.
**Full closure would mean generalizing kinematic decomposition itself to N holes** (retargeting
the loss-term structure the same way `MultiHoleHardConstraint` retargeted the ansatz) - real,
substantial, scoped new engineering work, not attempted in this pass. Per this project's own
"BLOCKED documented with evidence is not the same as complete" standard (see the Issue #63
Phase 4 close-out section of `CLAUDE.md`): this is a definitively diagnosed, disclosed,
open item, not a silently-accepted ceiling.
