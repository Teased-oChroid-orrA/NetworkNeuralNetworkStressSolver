# pinn-core

## Purpose

Owns: geometry (`UserGeometry`/`HoleSpec`/`GeometryConfig`), material properties, `ProblemSpec`/
`ArchitectureSpec` (the TOML-parsed problem definition), sampling primitives (`AdaptiveGrid`/AMR,
`DomainSamplingStrategy` trait), units, and the `BoundaryValueProblem`/`LossTerm`/
`DirichletAnsatz` trait definitions themselves. **No ML dependency** (no `burn`) — this is the
one crate every other crate can assume is pure/cheap to compile and has no tensor-backend
concerns.

Does not own: any trait *implementation* (those are `pinn-solver`'s job — this crate defines the
interfaces, `pinn-solver` implements them), training loop, optimizer, or physics.

Full chronological history for everything below: [`docs/findings/INDEX.md`](../../docs/findings/INDEX.md).

## Code Map

### Find It Fast

| Looking for... | Go to |
|---|---|
| N-hole plate geometry, hole placement | `user_geometry.rs` (`UserGeometry`, `HoleSpec`, `annular_partition(s)`, edge-referenced placement) |
| Single-hole Kirsch/pin-lug geometry | `geometry.rs` (`GeometryConfig`) — a DIFFERENT, older type from `UserGeometry`, deliberately not unified (see Contracts) |
| `ProblemSpec`/`ArchitectureSpec` (TOML schema for `--problem-spec`) | `problem_spec.rs` |
| Material properties, Young's modulus/Poisson/UTS | `material.rs` (`MaterialProps`) |
| Unit conversion (SI ↔ US customary) | `units.rs` — the ONLY place hand-rolled psi/ksi/in literals should ever appear |
| AMR (adaptive mesh refinement) grid/quadtree | `amr.rs` — `AdaptiveGrid`, `DensitySample`, `compensation_weights`, `lock_zones` |
| `BoundaryValueProblem`/`LossTerm`/`DomainSamplingStrategy` traits | `problem.rs` |
| Far-field load convention (`px`/`py` are stress, not force) | `loading.rs` (`LoadConfig`) |
| Per-domain collocation sampling primitives | `sampling.rs` |
| Solver config for the 1D toy-beam sanity check | `beam_spec.rs` |
| Solver config for the varying-(E,ν,P) parametric problem | `parametric_spec.rs` |
| Runtime channel message types (`TrainingMsg`/`ControlMsg`, GUI protocol) | `messages.rs` |
| Training-state enum shared by GUI/TUI | `state.rs` |
| Parameter-space distance metric (for warm-start/checkpoint compatibility) | `param_distance.rs` |
| Closed-form Kirsch stress solution (analytical reference, not the ansatz) | `kirsch.rs` |
| Post-training inference envelope / bounds checking | `inference_envelope.rs` |

### Key Relationships

- Zero inbound dependencies — this crate has no `pinn-*` dependency of its own. Any import
  pointing FROM `pinn-core` back INTO `pinn-solver`/`pinn-gui`/`pinn-app` is a layering
  violation.
- `problem.rs`'s trait definitions (`BoundaryValueProblem`, `LossTerm`, `DomainSamplingStrategy`,
  `DirichletAnsatz`) are pure interfaces — every real implementation lives in `pinn-solver`.
  Reading this file alone tells you the CONTRACT, not the behavior.

## Entry Points

| Task | Start Here |
|---|---|
| Add a new unit conversion | `units.rs` (add the named constant, never a hand-rolled literal elsewhere) |
| Add a field to the TOML problem spec | `problem_spec.rs` (`ProblemSpec`/`ArchitectureSpec`, `#[serde(default = ...)]` so every existing spec keeps parsing unchanged) |
| Add a new hole-placement mode | `user_geometry.rs` (`UserGeometry`'s custom `Deserialize` impl — see `HoleSpecRaw`/`resolve_hole`) |
| Add/change an AMR refinement rule | `amr.rs` (`AdaptiveGrid`, `lock_zones`, `should_adapt`) |

## Contracts

- Internal storage is ALWAYS SI (Pa, m). Every display-layer conversion to US customary
  (psi/ksi/Msi/inches) must go through `units.rs`'s named constants
  (`IN_TO_M`/`PSI_TO_PA`/`KSI_TO_PA`/`MSI_TO_PA`) — never a hand-rolled literal.
- `GeometryConfig` (Kirsch/pin-lug, exactly one hole, hardcoded) and `UserGeometry` (N holes,
  used only by the `--problem-spec` path) are deliberately separate types, not unified into one
  generic "N≥0 hole geometry." `GeometryConfig` is load-bearing for the frozen Kirsch/pin-lug
  paths; a generic replacement would carry real regression risk against those byte-tested paths
  for no evidenced benefit. `UserGeometry::to_placeholder()` produces a `GeometryConfig` sized to
  the real bounding box specifically so nothing generic that reads a domain's `GeometryConfig`
  (only bounding-box math) needs to know which type originally produced it.
- `MaterialProps::ultimate_strength_pa` is used ONLY as an opt-in normalization reference
  (`SolverConfig::use_ultimate_strength_scaling`) — never read by `lame()`/`plane_stress_c()`/the
  constitutive law itself. Don't wire it into stress/strain computation without re-reading that
  field's own doc comment on the temper-dependence caveat (4340's real UTS spans
  125,000–287,000 psi by temper; the stored value is one representative condition, not a
  material-invariant constant like `e`/`nu`).
- `LoadConfig.px`/`py` are far-field STRESS (traction) boundary conditions, in Pa — not forces in
  N/lbf, despite the temptation to read "load" as force. This is why they live in Pa/ksi like
  any other stress quantity in this crate.

## Pitfalls

- `AdaptiveGrid`'s quad-coarsening used to require unanimous 4-of-4 agreement before coarsening a
  refined region back down — relaxed to a 3-of-4 quorum (issue: AMR coarsening bias) because
  requiring unanimity made the grid pathologically resistant to coarsening once ANY neighbor's
  residual stayed noisy, silently keeping density high (and training slower) long after it was
  needed.
- `UserGeometry::validate()` is a pure GEOMETRIC check (in-bounds, non-overlapping holes) — it is
  NOT an FD-stencil-safety-margin check. That real per-run margin (a function of
  `training.fd_h`) lives in `pinn-solver` (no dependency from here to there), so "too close for a
  numerically safe FD stencil at this run's specific `fd_h`" is a real, narrower gap this
  validator does not and cannot catch.
- `MaterialProps::dimensionless_modulus(f_c)` (`E/f_c`) is a pure diagnostic helper — it is never
  wired into any constitutive law. Don't assume calling it changes any training behavior.

## Boundaries

### Never

- Never add a `pinn-solver`/`pinn-gui`/`pinn-app` dependency to this crate's `Cargo.toml` — the
  "no ML deps, cheap to compile, importable by everything" property is load-bearing for the
  whole workspace's build-time discipline (see the workspace root `CLAUDE.md`'s own note on
  `cargo check --workspace --tests` as the pre-flight for exactly this class of layering slip).
- Never hand-roll a stress/length unit conversion literal outside `units.rs` — grep for the
  target unit's constant there first.
