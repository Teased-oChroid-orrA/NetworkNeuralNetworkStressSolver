# Issue #78 Stage 1: multi-hole hardening — machinery vs. accuracy

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. N-hole sampling/AMR/Kt-reporting machinery proven correct for N>=2, and why single-hole Kt accuracy machinery does not generalize to N holes without further work.

## Issue #78 Stage 1: multi-hole hardening — machinery vs. accuracy, and why they're separate

Full plan/progress: `docs/multi-hole-and-boundary-shape-epic.md`. Short version for anyone
touching a multi-hole (N≥2 circular `HoleSpec`) spec on the default (non-`hard_constraint_
ansatz`) `UserDefinedProblem` path:

**Machinery is genuinely proven for N holes** — containment, signed distance, boundary
normals/measures, valid-stencil, interior/boundary sampling (including narrow-ligament
rejection via `UserSamplingStrategy::contains_for_collocation`'s per-hole union), AMR lock
zones, one `HoleBcTerm` per hole, and per-hole Kt reporting in both headless and GUI default
paths all loop over `holes.iter()` correctly. Real headless runs (both Wgpu and NdArray
backends) on 2-hole and 3-hole configs complete cleanly — no panic, no NaN, finite per-hole Kt,
sane equilibrium error.

**Kt numerical accuracy has NO correction path for N≥2, and that's a structural fact, not an
oversight to file a follow-up for.** Every accuracy mechanism this codebase built for the
single-hole case is explicitly gated to exactly one centered `Free` hole:
`decomposition_applicable` (`user_problem.rs`), `hole_bias_fraction`'s `holes.len()==1` guard,
and `coordinate_embedding`'s `[hole] = self.holes.as_slice() else { return Raw }` pattern all
fall back to the plain raw-coordinate model for N≠1. A multi-hole run therefore has strictly
LESS accuracy machinery than the single-hole default case, which itself only reaches Kt≈1.0-1.5
against a true ≈2.4-3.0 (see the PH4-4x sections above). **Do not assume the single-hole
hard-constraint-ansatz fix "just needs generalizing" to N holes** — it's a real, separate,
harder problem (the ansatz's closed-form correction and the sampling bias are both derived
relative to ONE hole's own center/radius; a multi-hole generalization needs its own kinematic
decomposition scheme, not a loop over the existing one) and is explicitly out of this epic's
scope.

**A real, previously-unfixed bug closed alongside this**: `HoleBcTerm::name()` used to return
the CONSTANT `"hole_free"`/`"hole_fixed"` regardless of which hole a term belonged to. Two holes
sharing a BC (`triple_hole_plate.toml`'s own two `Free` holes, a real shipped example) collided
in `training_core.rs`'s `lam_by_name`/`raw_scalar_by_name`/`term_grad_norms` `HashMap<&str, _>`s
— the second hole's entry silently overwrote the first, so both ended up weighted by whichever
hole's SAW-adapted lambda was computed last. Both holes still received real gradients and
contributed to the training objective throughout — this was a diagnostics/weighting-fidelity
bug, not a dropped-physics one. Fixed via `hole_bc_term_name` (`user_problem.rs`): the first
hole of a given BC keeps the exact pre-fix unsuffixed name (every single-hole or mixed-BC, e.g.
one `Free` + one `Fixed`, spec's term names/diagnostics stay byte-identical); only the 2nd+ hole
sharing that BC gets a numeric suffix (`"hole_free_1"`, `"hole_free_2"`, ...). `UserDefinedProblem::
base_weight` matches by prefix (`starts_with("hole_free_")`) rather than needing a second exact
arm per suffix.

**Also new**: `UserGeometry::validate()` — a real, previously-nonexistent check (overlap and
out-of-plate holes were silently accepted before this) wired into both `pinn-app`'s
`--problem-spec` headless path and `pinn-gui`'s spec-Load button. Deliberately a pure geometric
check (in-bounds, non-overlapping), NOT an FD-stencil-safety-margin check — `pinn-core` has no
dependency on `pinn-solver`, where the real per-run margin formula (a function of
`training.fd_h`) lives, so "too close for a numerically safe FD stencil at this run's specific
`fd_h`" stays a real, narrower, still-open gap.

**Also new**: hole placement by edge-referenced distance (`from_left`/`from_right`/`from_top`/
`from_bottom` in `[[geometry.holes]]`, alongside the existing `center = [x, y]`) - pure
TOML-input-layer sugar via a custom `Deserialize` impl on `UserGeometry`, zero solver/physics
change. See `examples/problems/hole_by_edge_reference.toml`.

