# User-defined problem ingestion (`--problem-spec`)

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. How an arbitrary N-hole TOML plate spec is trained headless via `UserDefinedProblem`/`step_physics_multi`, and its deliberate v1 scope cuts.

## User-defined problem ingestion (`--problem-spec`)

`pinn_core::user_geometry`/`problem_spec` + `pinn_solver::user_problem`/`user_runner` let a
user design a NEW 2D elasticity joint (rectangular plate, N circular holes, each either
`Free` or `Fixed`) from a TOML file, instead of writing new Rust — the missing piece
`--headless` dispatch didn't have (it only ever routed `ProblemKind::{Kirsch,PinLug}`
through concrete `SolverConfig`s). Run: `cargo run -p pinn-app --release -- --headless
--problem-spec <path.toml>` (see `examples/problems/notched_plate.toml` for a documented
template). **Absent this flag, `main()`'s existing Kirsch/pin-lug dispatch runs completely
unchanged** — the flag is checked and `main()` returns before any of that code executes.

Drives the SAME `BoundaryValueProblem`/`LossTerm`/`DomainSamplingStrategy` trait family
Kirsch/pin-lug use, through the already-generic `step_physics_multi` driver
(`training_core.rs`) — this is genuinely its first live production user (Kirsch's own trait
impl is test-fixture-only, per the hardware-adaptive-execution epic's earlier investigation).
No new physics math: every loss term is a near-verbatim copy of an existing Kirsch/pin-lug
term's shape (`InteriorEnergyTerm`→`dem_energy_loss`, `OuterTractionTerm`→`neumann_loss`,
`HoleBcTerm::Free`→`hole_traction_loss_direct`, `HoleBcTerm::Fixed`→`mean(u²+v²)` anchor,
mirroring `LugShankAnchorTerm`). Essential BCs are soft-penalty (`IdentityAnsatz`, reused
directly from `pinlug_problem.rs`), not a hard ansatz — there's no closed-form scale factor
that zeroes displacement on an arbitrary circle at an arbitrary position, the same reason
pin-lug itself doesn't hard-enforce its shank anchor.

**Deliberately separate geometry type, zero changes to any existing file.**
`GeometryConfig` hardcodes exactly one hole and is load-bearing for Kirsch/pin-lug;
`UserGeometry`/`HoleSpec` (N holes) is a fully independent type. `UserSamplingStrategy`
ignores the `&GeometryConfig` parameter every `DomainSamplingStrategy` method takes, using
its own captured `UserGeometry` instead — an established, precedented pattern (see
`FakeInterfaceSampling` in `pinn-core/src/problem.rs`'s own tests). `DomainSpec.geometry`
gets `UserGeometry::to_placeholder()`, a `GeometryConfig` sized to the real bounding box
(`hole: HoleType::None`, `symmetry: Full`) — safe because nothing generic reads a domain's
`GeometryConfig` except bounding-box math (`x_range`/`y_range`), confirmed by tracing
`compute_domain_forwards`; the only `SolverConfig` field that function reads at all is
`config.load.px` (mDEM stress-column physical-unit scaling) — `user_runner` builds its
`SolverConfig` from `SolverConfig::default_kirsch()` with only `.load` overridden.

**v1 scope, deliberately**: single domain (not full N-domain/contact generality — no product
need yet, and pin-lug's 2-domain coupling isn't generalized), headless-only (no GUI wiring),
no curriculum/AMR/decision-maker/SAW-BRDR-tiering (plain constant/scheduled-LR AdamW loop via
`step_physics_multi` directly, matching `toy_beam`'s own "prove the formulation converges
before adding curriculum machinery" discipline). No closed-form convergence oracle exists for
an arbitrary user geometry (unlike Kirsch's K_t) — verification is qualitative: loss should
decrease, and `max|displacement|` should be non-negligible (a real trivial-collapse check,
printed by `user_runner`).

**Verified end-to-end** (`examples/problems/notched_plate.toml` — 0.2×0.1 m plate, one Free
hole, one Fixed hole, ~10 ksi far-field traction, `hidden_dim=64`, 2000 steps): loss decreased
monotonically 5.5 → 0.47, `max|displacement|~9.86e-5 m` — matches the back-of-envelope
estimate `(px/E)·half_w ≈ 9.6e-5 m` almost exactly, a strong physical sanity check that this
is real elasticity, not noise. Confirmed the no-flag default (`--headless`, no
`--problem-spec`) still produces byte-for-byte the same Kirsch banner/output as before this
feature. Full workspace suite: 229 passed, 0 failed, 2 ignored (the pre-existing `toy_beam`
ignores) — purely additive, zero regressions.

