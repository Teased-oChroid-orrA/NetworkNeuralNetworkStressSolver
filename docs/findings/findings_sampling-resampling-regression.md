# Issue #64/#66/#73: sampling-resampling regression, its GUI-path twin, and the fix

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. The frozen-quadrature-node overfitting bug, the GUI-streaming path's own separate stale-cache copy of it, and the shared-function unification that closed both.

## Issue #64/#66/#73: sampling-resampling regression, its GUI-path twin, and the fix

`UserSamplingStrategy::sample_interior`/`sample_boundary` (`user_problem.rs`) draw a genuinely
different jittered-stratified point set on every call, seeded from a per-instance `AtomicU64`
call counter (`interior_calls`/`boundary_calls`) — **not** the fixed-seed pure functions they
were before issue #64. Before that fix, both no-hole methods returned the byte-identical point
cloud on every call (deterministic grid, and a rejection fallback that reseeded from the same
constant and was never reached with zero holes to reject), which silently defeated the training
loop's own intended "resample every step" and let the network overfit that one frozen finite
quadrature-node set — `InteriorEnergyTerm`/`PhysicalPotentialEnergyTerm`'s `U` and
`ExternalWorkTerm`'s `W_ext` are both plain `mean(f(x_i))` Monte-Carlo estimators, unbiased only
if the `x_i` genuinely vary across the optimization trajectory. Fixed by making both methods
draw a real per-call jittered sample; still fully reproducible run-to-run (same base seed
constant ⇒ same full sequence of per-call point sets).

**Any caller of `sample_interior`/`sample_boundary` that caches the result across multiple
steps now gets stale, non-varying points again — the exact bug class this fix exists to
prevent.** This bit `runner::run_user_problem_training_from` (the GUI-streaming path) directly:
it cached `sample_boundary`'s output once before its training loop (a real, correct-at-the-time
perf optimization from before issue #64, whose own doc comment said so and was true then) and
was never updated, silently re-freezing the boundary point cloud for the entire run. Caught
only because sub-issue #66 strengthened a benchmark assertion on that exact path. If you add a
new plate-problem training loop or diagnostic that resamples, resample every call it's meant to
vary on — never cache `sample_interior`/`sample_boundary`'s output across more than one call.

**AMR defaults to `amr_enabled=true` but the canonical no-hole benchmark needs it off.** Issue
#63's own PH4-09 policy states "AMR = OFF, uniform sampling = ON until the baseline is
mathematically correct" — AMR has no dedicated evidence yet (tracked separately). The shipped
`examples/problems/variational_no_hole_plate.toml` now sets `amr_enabled = false` explicitly to
actually honor that policy; omitting it silently pulls in the default and reintroduces
borderline-variance behavior on the GUI-streaming path specifically (headless never implements
AMR at all, so it was unaffected either way).

**Single source of truth (issue #73)**: `user_problem::resample_plate_step_data`/
`plate_multi_step_ctx` are now the ONLY implementation of the plate problem's per-step
resampling and `MultiStepCtx` construction — both `user_runner::run_headless_user_problem`
(headless CLI) and `runner::run_user_problem_training_from` (GUI-streaming) call them instead
of independently duplicating this logic, which is exactly how the boundary-caching bug above
happened (one copy got updated for issue #64's fix, the other didn't). Regression-proven by
`runner::tests::gui_streaming_step_zero_matches_independent_shared_function_computation`: the
GUI-streaming path's real, captured step-0 `total_loss` is asserted to exactly equal an
independent reference computed by calling the same two shared functions directly — genuine
unification, not just two call sites that happen to both compile.

**Deliberately NOT unified with Kirsch/pin-lug.** `headless::run_headless`/`runner::
run_training` (Kirsch) and `headless::run_headless_pinlug`/`runner::run_training_pinlug`
(pin-lug) remain two separate, independently-maintained implementations each — this is the
same precedent as `step_physics`/`step_physics_multi` (frozen, byte-exact-tested paths must
never be refactored into a thin wrapper around something else) and was never in scope for #73:
they've never exhibited this bug class, and forcing them through a shared abstraction would
carry real regression risk against their own frozen paths for no evidenced benefit. Only the
user-defined plate problem's headless-vs-GUI-streaming pair was unified.

**`parametric_problem.rs`'s own `FixedPoints`/`build_fixed_points`** (a *different* per-step
caching pattern, for the separate "instant inference over varying E/nu/P" feature) was
investigated during the same session and found to have the identical "sampled once, reused
every step" shape — but parametric problems never claimed per-step resampling and are not
regressed by issue #64 (their pre- and post-#64 behavior is the same: one static point draw for
the whole run). Left alone, out of scope for #64/#66/#73.

