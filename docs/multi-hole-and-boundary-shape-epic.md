# Epic #78: multi-hole hardening + hole-placement-by-reference

Tracked at: https://github.com/Teased-oChroid-orrA/NetworkNeuralNetworkStressSolver/issues/78

## Context

User wants pinn-app to handle any problem thrown at it: multi-hole, non-rectangular/non-square
outer shapes, "anything else." Two Explore investigations (this session) established the real
state of each:

**Multi-hole (N≥2 holes, default architecture)** — machinery is genuinely proven for N holes:
containment, signed distance, boundary normals/measures, valid-stencil, interior/boundary
sampling (including narrow-ligament rejection via `contains_for_collocation`'s per-hole union),
AMR lock zones, per-hole `HoleBcTerm` emission, and per-hole Kt reporting in both headless and
GUI default paths all loop over `holes.iter()` correctly and are unit-tested on real 2-hole
fixtures. No hidden `holes[0]`-only bug exists on the default path. What's real and unfixed:

1. **Diagnostic/weighting collision when two holes share a BC.** `HoleBcTerm::name()` returns
   only `"hole_free"`/`"hole_fixed"` regardless of hole index (`user_problem.rs:1409-1411`), so
   two Free holes produce two terms with the SAME name. `training_core.rs`'s `lam_by_name`/
   `raw_scalar_by_name`/`term_grad_norms` are `HashMap<&str, _>` keyed by that name
   (`:1877-1912`, `:2068`, `:2159`) — the second hole's entry silently overwrites the first, and
   **both holes end up weighted by whichever hole's SAW lambda was computed last**, discarding
   per-hole adaptation SAW itself correctly computed. Both holes still get real gradients and
   contribute to `total` — this degrades weighting fidelity and diagnostics, it does not drop a
   hole's physics.
2. **No hole-overlap/spacing validation anywhere.** `ProblemSpec`/`UserGeometry` has no
   `validate()`. Overlapping or near-touching holes are accepted silently; `amr.rs:75-76`
   already documents "Overlap is not currently prohibited" for its own area-integration
   fallback, but the sampler has no equivalent fallback.
3. **Hole-ring/Kt-probe points aren't checked against OTHER holes.** `named_point_sets`
   generates each hole's ring purely from its own circle; for closely-spaced holes a ring point
   or its FD stencil arm could land inside a neighboring hole. No validation catches this today.
4. **Kt numerical accuracy has no path to improvement for N≥2, structurally.** Every accuracy
   mechanism the single-hole investigation already established (hard-constraint ansatz,
   hole-biased sampling + quadrature compensation, chart coordinate embedding) is explicitly
   gated to exactly one centered Free hole (`decomposition_applicable`, `hole_bias_fraction`'s
   `holes.len()==1` guard, `coordinate_embedding`'s `[hole] = ...` pattern). A multi-hole run
   gets the plain raw-coordinate network with none of that — strictly less accuracy machinery
   than the single-hole default case, which itself only reaches Kt≈1.0-1.5 against a true
   ≈2.4-3.0. This project's own docs already classify multi-hole as "machinery operational, Kt
   accuracy explicitly not claimed" (`docs/FORMULATION_SUPPORT_MATRIX.md`,
   `docs/GENERAL_SOLVER_OPERATIONAL_STATUS.md`) — not a bug to fix here, a ceiling to disclose.

**Non-rectangular outer boundary, AND non-circular cutouts (slots, keyways, rectangular
pockets, arbitrary polygons)** — real, multi-week architectural undertaking, not a config flag,
and the SAME underlying gap in two places. `half_w`/`half_h` are plain scalar fields read ~600
times across 27 files, with the rectangle's closed-form consequences (area `4·hw·hh`,
perimeter, per-edge `ds`, four constant boundary normals, `BoundaryRef`'s closed 4-edge enum,
`sample_boundary`'s hardcoded "4 edges, N/4 points each") duplicated independently rather than
derived from one place. `HoleSpec` has the identical problem one level in: `{center, radius,
bc}` is circle-only — `contains`'s per-hole test, `signed_distance`'s hole term, `boundary_
normal_for`'s radial normal, and `named_point_sets`'s hole-ring sampler (a `theta` sweep on a
circle) all hardcode "hole = circle" exactly the way the outer edge hardcodes "boundary =
axis-aligned rectangle." A slot or rectangular cutout is not representable today at all, for
the same structural reason a non-rectangular plate isn't.

This means the boundary-shape work should be scoped as ONE unified future epic — "arbitrary
boundary shape, inner and outer" — not two, since both need the same real fix: a general
boundary abstraction that can emit `contains`/signed-distance/boundary-points-with-normal-and-
arc-length for ANY shape (circle, polygon, rounded-rect/stadium slot, or a general SDF),
replacing `BoundaryRef`'s closed enum and the ~5 places that currently re-derive `ds` from
`half_w`/`half_h`/`radius` by hand. The single highest-risk coupling found: `sample_boundary`'s
point-emission order and `ExternalWorkTerm::ds_per_point`'s quadrature-weight reconstruction are
two independent hardcodings of the same rectangle, kept in sync only by comment, not by type —
any new shape abstraction must retire that pattern, not extend it per-shape. Bounding-box
network-input normalization would still be mathematically valid for any outer shape (the good
news), but `to_placeholder()`'s "bounding box is the real domain" fiction is load-bearing
everywhere that reads `x_range()`/`y_range()`/area/perimeter today. This needs its own dedicated
investigation-to-design pass before any code — NOT this epic's scope.

## Stage 0 (this epic): hole placement by edge-referenced distance

`HoleSpec.center: [f64; 2]` already accepts arbitrary (non-centered) coordinates today — nothing
blocks off-center placement at the data/physics level. What's missing is an engineering-drawing-
style input convention: placing a hole by distance from a reference (a plate edge) instead of
only by raw `[x, y]` relative to the plate centroid. Pure TOML-input-layer sugar — zero solver,
sampling, or physics change, and applies identically whether there's 1 hole or N.

Scope: distance from the plate's own edges only (`from_left`/`from_right`/`from_top`/
`from_bottom`, physical units, matching every other length field in this file) — NOT relative-
to-another-hole referencing (real added complexity: ordering/cycle concerns, deferred, flag if
actually wanted later). A hole gives either the existing `center = [x, y]` OR exactly one X-axis
reference (`from_left` XOR `from_right`) plus exactly one Y-axis reference (`from_top` XOR
`from_bottom`) — never a mix, never zero, never two on the same axis. Loud, clear error on an
inconsistent/incomplete spec (matches this codebase's established "fail fast, not silently"
convention), not a silent fallback to `[0,0]`.

Design: `HoleSpec` itself is unchanged (zero breakage to the many existing Rust call sites that
construct `HoleSpec { center, radius, bc }` directly). Implemented as a custom `Deserialize`
impl for `UserGeometry` (`pinn-core/src/user_geometry.rs`, replacing the derived one) that
deserializes an intermediate `HoleSpecRaw { center: Option<[f64;2]>, from_left/right/top/
bottom: Option<f64>, radius, bc }` per hole, then — now that `half_w`/`half_h` are known from
the sibling fields in the SAME `UserGeometry` deserialization — `resolve_hole` resolves each raw
hole into a real `HoleSpec` via `center = [-half_w + from_left, -half_h + from_bottom]` (or the
mirrored `half_w - from_right`/`half_h - from_top` forms), erroring on an invalid combination
before `UserGeometry` is ever constructed. Every existing TOML using plain `center = [x, y]`
stays byte-identical.

## Stage 1 (this epic): harden and verify multi-hole (circular holes only)

Scope note: this stage stays within the existing circular-`HoleSpec` representation — N
circular holes, any mix of Free/Fixed. Slots and other non-circular cutouts are the future
boundary-shape epic's problem, not attempted here.

1. Fix the duplicate-hole-name diagnostic/weighting collision — per-hole-index term names
   (e.g. `"hole_free_0"`, `"hole_fixed_1"`) instead of the constant `"hole_free"`/`"hole_fixed"`,
   mirroring the existing `hole_names`/`hole_fd_names` per-index-leaked-`&'static str`
   convention `UserSamplingStrategy::new` already uses.
2. Add hole geometry validation — reject a hole extending outside the plate, or two holes
   overlapping/closer than a safety margin, with a clear parse/construction-time error.
3. Verify machinery on cheap, fast configs (small network, few hundred steps,
   `amr_enabled=false`, both backends, via `--headless`) — no panic/NaN, finite per-hole Kt,
   sane equilibrium/energy-balance error. Machinery smoke check, not a Kt-accuracy claim.
4. Document the real ceiling in `CLAUDE.md` — multi-hole Kt accuracy has no correction path yet.

## Stage 2 (follow-up session, tracked in `docs/multi-hole-fem-ground-truth-investigation.md`)

Stage 1.4's "multi-hole Kt accuracy has no correction path yet" is no longer accurate as of a
follow-up session that built `AnnulusAnsatz::MultiHoleHardConstraint` (N-hole generalization of
the single-hole hard-constraint ansatz) and fixed two real bugs found verifying it end-to-end
(a missing affine-background gate for off-center/multi-hole specs, and a stale diagnostic
reconstruction in `probe_load_transfer`/`probe_reaction_force`/`probe_boundary_residuals` that
predates this epic entirely). Real trained Kt for every `HoleBc::Free` hole now lands within
12-20% of FEM ground truth (up from off by 6-600x) on both real shipped multi-hole geometries.
Full detail, including the honestly-disclosed remaining gap, lives in
`docs/multi-hole-fem-ground-truth-investigation.md`'s own "PINN-side implementation" section
and `CLAUDE.md`'s "Issue #78 Stage 2" section - not duplicated here to avoid a third copy of
the same numbers drifting out of sync.

**Follow-up (same session): two more real bugs found and fixed, one open item definitively
diagnosed.** (1) The FEM ground-truth tool itself never supported `Fixed` holes - fixed, and
building that fix surfaced (2) a real, previously-invisible mirror-symmetry bug in the same
tool's rigid-body-pin scheme (fixed, 6 new regression tests, proven zero-effect on every prior
all-Free number). (3) The flat-loss-plateau this document's own earlier section flagged as
unresolved is now definitively diagnosed (real convergence to a robust local optimum, not
masked progress - proven via a new live per-step Kt diagnostic). See the investigation doc's
own "Follow-up" section for the full record.

**Second follow-up (same session): `CoordinateEmbedding::MultiHoleChart` closes a real,
previously-open gap - the N-hole generalization of `SingleHoleChart`'s own hole-relative
geometric features (every multi-hole geometry used to fall back to raw coordinates, zero
geometric hole-awareness). A real bug in `probe_boundary_residuals` this exposed is also fixed.
This did NOT move the trained Kt, joining three earlier hyperparameter experiments as a
FIFTH independently falsified hypothesis (data density, capacity, learning rate, coordinate
embedding, Fixed-hole sampling bias - all reproduce the identical converged Kt). Working
conclusion, updated from "needs kinematic decomposition generalized to N holes" (which turned
out to already be done): the remaining gap looks like a genuine property of the Π functional's
own formulation for a multi-hole domain, not a training/representation deficiency - a
formulation-level audit, not further hyperparameter search, is the real next step. See the
investigation doc's own "Second follow-up" section for the full record.

## Future epic (not this one): arbitrary boundary shape, inner and outer

See Context above. Needs its own investigate-first pass and explicit design decision before any
implementation — deliberately not spec'd to file-level detail here.

## Progress

- [x] Stage 0: hole placement by edge-referenced distance — implemented (`UserGeometry`'s
      custom `Deserialize`, `resolve_hole`), 6 new tests in `pinn-core` (all pass, 146/146 full
      suite), shipped `examples/problems/hole_by_edge_reference.toml`, end-to-end headless smoke
      run confirmed (Wgpu backend, real training dispatch, no panic, finite Kt at step 5/5).
- [x] Stage 1.1: duplicate-hole-name diagnostic/weighting collision — fixed via
      `hole_bc_term_name` (per-BC-occurrence naming, first hole of a BC keeps the exact
      pre-#78 unsuffixed name, only the 2nd+ same-BC hole gets a numeric suffix) + `base_weight`
      prefix-match update. 3 new regression tests (distinct names for 2 Free holes proven,
      `base_weight` resolves suffixed names, pure naming-rule unit test). All pass.
- [x] Stage 1.2: hole geometry validation — `UserGeometry::validate()` (pure geometric check:
      every hole fully in-bounds, positive radius, no two holes overlapping), wired into both
      `pinn-app`'s `--problem-spec` headless path and `pinn-gui`'s Load button. 7 new tests +
      every shipped example asserted to still pass validation. End-to-end CLI rejection
      confirmed with a real overlapping-hole spec (`Error: invalid geometry in '...': geometry.
      holes[0] and geometry.holes[1] overlap: ...`).
- [x] Stage 1.3: multi-hole machinery verification — real headless runs, trimmed
      `notched_plate.toml` (2 holes) / `triple_hole_plate.toml` (3 holes) copies (hidden_dim=32,
      300 steps, `amr_enabled=false`), both Wgpu and NdArray backends (4 combinations total).
      All 4 completed cleanly: no panic, no NaN, finite per-hole Kt, equilibrium error 0.9-1.6%.
      Machinery smoke check only, not a Kt-accuracy claim (see Stage 1's own scope note).
- [x] Stage 1.4: documented the real ceiling in `CLAUDE.md` (new "Issue #78 Stage 1" section) -
      multi-hole Kt accuracy has no correction path yet, and why (every single-hole accuracy
      mechanism is explicitly gated to exactly one centered Free hole).

**Epic scope complete.** Final full-workspace regression (`cargo test -p pinn-solver -p
pinn-core -p pinn-app -p pinn-gui --release -- --test-threads=1`): 540 passed, 1 failed - the
same pre-existing, unrelated `compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key`
failure confirmed failing on unmodified `d9f38fe` (before this session's own work began) via
`git stash`. Zero new regressions. All 3 workspace warnings found along the way (`pinn-gui`'s
11 `f32: From<f64>` numeric-literal-fallback warnings, 2 real `dead_code` false-positives in
`pinn-solver`) fixed too, verified against CI's exact build command
(`cargo build --features eframe/x11,eframe/wayland --verbose`) - zero warnings remain except
`block v0.1.6`, a macOS-only transitive dependency that never appears on the Linux CI runners.
- [ ] Stage 1.1: duplicate-hole-name diagnostic/weighting fix
- [ ] Stage 1.2: hole geometry validation
- [ ] Stage 1.3: multi-hole machinery verification (cheap/fast configs, both backends)
- [ ] Stage 1.4: CLAUDE.md documentation of the multi-hole Kt-accuracy ceiling

Update both this file and the GitHub issue's own checklist as each item completes — keep them
in sync (this file carries the full technical detail; the issue is the tracking surface).
