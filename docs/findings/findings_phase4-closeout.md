# Issue #63 Phase 4 close-out: what's operational, what isn't, and why L5 is blocked

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. The no-hole variational benchmark passing cleanly vs. hole/Kt numerical accuracy (L5) still blocked, the real upstream burn-autodiff race fix, and why AMR alone didn't close L5.

## Issue #63 Phase 4 close-out: what's operational, what isn't, and why L5 is blocked

Phase 4 (issues #64-#73, epic #63) reached VERIFIED-or-BLOCKED-with-evidence on every item once
but epic #63 was reopened — per this project's own standard, `BLOCKED` documented with evidence
is not the same as complete, and #74/#70/#71's remaining gaps are real work, not paperwork.
Full detail lives in
`docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`, `docs/FORMULATION_SUPPORT_MATRIX.md`, and
`docs/GENERAL_SOLVER_OPERATIONAL_STATUS.md` — this section is the short pointer for a future
session, not a duplicate of those documents.

**Operational**: Variational no-hole L4, both square and non-square geometries (real headless
runs pass all five P2-14 hard thresholds with real margin). **Not yet operational**: hole/Kt
numerical accuracy (L5) — real, evidenced, and explicitly not hidden behind a passing benchmark.

**Issue #74 (AMR+Variational autodiff crash) is FIXED and verified — via a DIFFERENT root cause
than first diagnosed.** See PH4-23 in the manifest for the full, corrected record; PH4-09's own
entry documents the original (real but incomplete) diagnosis for history. One-line summary: the
`probe_interior_energy_residuals`-through-`BInner` fix (routing the AMR residual probe through
the non-autodiff backend) is a genuine, valid improvement but was **not** the actual cause of
this crash — the crash fires on a fresh model's very first `.backward()` call, before AMR's own
warmup period even ends. The REAL cause is a confirmed, currently-unreleased upstream burn-
autodiff bug (`tracel-ai/burn` issue #5573 - a concurrent `backward()` can free another thread's
still-being-registered graph node via the library's own process-global post-backward cleanup
sweep; fixed by burn PR #5647, not yet in any published release), triggered in OUR OWN test
harnesses by two compounding causes: spawning a separate OS thread per comparison arm, and
building one `initial_model` then handing `.clone()` to one arm while moving the original into
the other - **a burn `Module`/`Tensor` clone is cheap and shares the same underlying autodiff
`NodeId`, it does not mint a fresh leaf**, so the second arm ends up reusing an identity the
first arm already drove through thousands of real backward passes. Fixed in
`ph4_09_controlled_comparison_...`/`ph3_12_controlled_comparison_...`/`gui_streaming_step_zero_
...` by (a) calling the training function synchronously in the test's own thread instead of
spawning one per arm, and (b) building independently-`.init()`'d models from the same seed
instead of clone/move. This is also the likely real explanation for `gui_streaming_step_zero_
...`'s long-documented "contention-flaky" behavior this whole epic attributed to vague floating-
point nondeterminism. AMR still stays OFF for the *canonical no-hole* benchmark specifically —
that was always a separate, independently-measured quality-regression finding, unrelated to
either crash mechanism.

**A `cargo test` run WITHOUT `--test-threads=1` (cargo's own default) can still occasionally
show `gui_streaming_step_zero_...` fail** - this is the SAME upstream bug at the cross-test-
function level (some unrelated, concurrently-running test's own training thread racing with
this one), not a new or unfixed issue. **CI is unaffected**: `.github/workflows/rust.yml`
already runs every job with `--test-threads=1` for an unrelated, pre-existing GPU/lavapipe-
contention reason. Use `--test-threads=1` for a reliable full-suite run locally too.

**Durable lesson**: a burn `Module`/`Tensor` `.clone()` is a cheap, identity-sharing clone, not
a deep copy that mints a fresh autodiff leaf. Any future test wanting two genuinely independent
training runs from "the same starting weights" must build them via two separate `.init()` calls
under the same seed, never via `.clone()`/move of one shared instance.

**Why L5 doesn't pass (issue #70), and why AMR alone doesn't fix it either**: a small hole
(radius/half-width ratio ~0.05) gets almost no collocation density near its own boundary under
uniform Monte-Carlo sampling — confirmed by a zero-cost sampling-only check (no training): only
~0.55% of interior points land within 2 hole-radii of the boundary. A real 3000-step, 4096-point
run gets `Kt=1.008` against the theoretical `3.0` (66% relative error). With #74 fixed, the same
config was re-run WITH AMR enabled (`issue_70_real_l5_with_amr_enabled_after_issue_74_fix`,
`user_problem.rs`, `#[ignore]`d) — AMR genuinely re-densified the point set across 3 sweeps
(`4096→706→1381→2281`, confirmed working by a companion zero-cost fixture proving the refinement
mechanism itself is sound) but Kt barely moved: `1.0088`, essentially unchanged. **Don't assume
"fix #74, enable AMR" solves L5 — it doesn't, at least not with AMR's current residual-driven
refinement strategy.** Working hypothesis (NOT verified): AMR's signal reflects where the
CURRENT network's residual is large, which may not correlate with the true stress concentration
early in training, so refinement may be concentrating in the wrong place, or too late in the
budget, to help. A future attempt should test this hypothesis directly (e.g. earlier/more
frequent sweeps, or a structural initial-density bias seeded from hole geometry rather than
waiting on the network's own residual) rather than assuming more of the same AMR config will
eventually converge.

**A pre-existing CI-only flake, found and fixed during this close-out**: CI (not local `wgpu`)
uses a software/different Wgpu backend than a local machine, and
`network::tests::coordinate_skip_represents_affine_displacement_and_leaves_stress_mlp_only` used
to assert exact `f32` equality on a hand-computed matmul+bias result — failing on CI with a
*different* mismatched value each run (`1.1000001` vs `1.1`, `-0.49999997` vs `-0.5`), proving
float rounding noise, not a logic bug. Now uses an epsilon-tolerant comparison. If a similarly
CI-only-flaky test turns up again, check for exact `assert_eq!` on raw `f32`/`Vec<f32>` values
first before assuming a real regression.

