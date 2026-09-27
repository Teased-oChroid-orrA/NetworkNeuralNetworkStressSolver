# GUI panel wiring and the ratatui TUI dashboard

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. Kirsch/pin-lug/user-defined-plate GUI streaming, the training-panel redesign, N-hole heatmap overlays, and the `--tui`/`-T` terminal dashboard that reuses the same channel protocol.

## GUI

`pinn_core::messages::ProblemKind` (`Kirsch`/`PinLug`) is the single shared selector — `pinn-app`'s
CLI parsing and `pinn-gui`'s problem-kind radio both read/write the same enum, not independently
drifting copies. `pinn_solver::run_training` (Kirsch) and `run_training_pinlug` (pin-lug) are two
separate GUI-driving functions, not one branching function, mirroring the `step_physics`/
`step_physics_multi` precedent: forcing two structurally different training loops through a
shared abstraction increases regression risk on the proven Kirsch path for no benefit. Pin-lug's
two domains are visualized via `PinLugVisFields { pin: VisFields, lug: VisFields }`, sent as a
dedicated `TrainingMsg::PinLugUpdate` variant (not an extension of the existing `Update`/
`VisFields`, which stay exactly as they were — zero regression risk on the Kirsch GUI path).
Pin-lug's contact-pressure CSV export is triggered via `ControlMsg::ExportContactPressure`
(solver-side write, confirmed back to the GUI via `TrainingMsg::ExportComplete`) rather than
sending the trained model over the channel. Pin-lug's `WarmStart` handling is a deliberate scope
cut in this slice: only scalar config fields are honored (no full two-domain resample) — the
GUI disables the warm-start button entirely when `ProblemKind::PinLug` is selected rather than
silently doing a partial warm-start.

## GUI wiring for user-defined problems + redesigned training panel

`pinn-gui` now supports a third, GUI-driven mode for the same `ProblemSpec`/
`UserDefinedProblem` path above: a "User-Defined" radio option in `params.rs` with a spec
path field + "Load" button, streaming live progress via a new `runner::
run_training_user_problem` into the heatmap/training panels.

**Deliberately did NOT add a variant to `pinn_core::messages::ProblemKind`.** Doing so would
ripple required-arm additions through every exhaustive match on it in the frozen Kirsch/
pin-lug paths for no benefit. Instead `pinn-gui`'s own `StressSolverApp` gained separate
fields (`user_defined_active: bool`, `user_spec_path/spec/error`) that override
`problem_kind`-driven dispatch only where needed — every existing `match problem_kind {
Kirsch, PinLug }` site is byte-for-byte unchanged.

**`run_training_user_problem` reuses `TrainingMsg::Update(Box<TrainingUpdate>)` — not a new
message variant.** `TrainingUpdate`'s fields are already generically named (`energy_loss`/
`neumann_loss`, not Kirsch-specific names), and this problem is single-domain like Kirsch,
not two-domain like pin-lug. `energy_loss = out.e_scalar` (this problem's energy term is
literally named `"interior_energy"`, matching `StepOutput::e_scalar`'s lookup for every
problem); `neumann_loss = out.total_scalar - out.e_scalar`, mirroring `run_training_pinlug`'s
own exact convention for aggregating an arbitrary number of differently-named BC terms into
one number without needing this problem's own term names to match any hardcoded accessor.

**`evaluate_user_vis_grid`** (`user_problem.rs`) mirrors `runner.rs`'s private
`evaluate_vis_grid_mdem` (same mDEM direct-column read — no FD stencil needed for
visualization — same von Mises formula), adapted for `UserGeometry`'s N-hole containment
check instead of `GeometryConfig`'s single-hole one.

**Heatmap N-hole overlay**: `heatmap::draw_overlays` gained a `user_holes: Option<&[HoleSpec]>`
parameter — when `Some`, draws each hole as its own full circle (no `QuarterSymm` assumption)
color-coded by `HoleBc::Free` (green) / `Fixed` (red), instead of the single-hole Kirsch/
pin-lug arc. The pixel/texture/colorbar core (`field_to_pixels`, `select_field`) needed zero
changes — already field-agnostic.

**Training panel redesign** (`training.rs`): added a row of 4 stat cards (total loss, energy
term, boundary term, learning rate — `egui::Frame` with a colored left accent bar) above the
existing `egui_plot` curves, restyled to a shared accent palette (teal/blue/amber/violet).
This panel is shared — Kirsch and pin-lug get the same visual upgrade with zero per-mode
branching, since it only ever reads `TrainingState`'s already-generic fields. Palette/layout
were prototyped first as an HTML design-reference artifact (dark "instrument panel" aesthetic)
before writing any egui code — see the artifact link in that session's conversation; egui's
real capability (immediate-mode 2D, no CSS effects) means the in-app result approximates that
reference's palette/layout, not a pixel-perfect port.

Verified: `cargo build --workspace` clean, full suite 229 passed/0 failed/2 ignored (same as
above — purely additive), release binary launches without crashing. Full interactive
click-through (load a spec, Start, confirm the heatmap/stat cards update) was NOT
independently verified end-to-end in this pass — the training math itself was already proven
correct via the headless path's own verification above, and this GUI runner is a thin
streaming wrapper around the identical `UserDefinedProblem`/`step_physics_multi` call, but an
actual mouse-driven session is worth a real check before relying on this heavily.

## `--tui`/`-T`: a ratatui terminal dashboard for `pinn-app`

A third live-training run mode alongside `--headless` (plain scrolling logs, no AMR wired for
the plate path) and the default GUI (needs a display/window compositor): a terminal-only
dashboard reached via `--tui`/`-T`, reusing the SAME training entry points the GUI already
drives (`run_training`, `run_training_pinlug`, `pinn_solver::runner::run_training_user_problem`)
over the identical `(Sender<TrainingMsg>, Receiver<ControlMsg>)` channel protocol `pinn-gui`'s
own `start_solver`/`drain_channel` uses - zero solver-side change, Kirsch/pin-lug/every
`[architecture]`-selectable User-Defined plate config all supported for free. Design doc:
`docs/tui-mode-plan.md` (its own "Status" section has the full implementation record, including
several real corrections a pre-implementation investigation found against its original design
claims - `Done` is a blocking send, the plate path doesn't return on its own after `Done`,
`TrainingMsg` has 10 variants, `PinLugTrainingUpdate.amr_sweep` is a `Vec` not an `Option`).

`crates/pinn-app/src/tui/{mod,state,ui}.rs` - three files, one job each: `mod.rs` is the ONLY
file that touches `crossterm` I/O (terminal lifecycle, panic hook, event loop); `state.rs` is
pure data + `TuiState::apply`/`apply_pinlug`, zero `ratatui`/`crossterm` dependency, unit-
testable with plain `TrainingUpdate`/`PinLugTrainingUpdate` literals; `ui.rs` is pure rendering,
zero I/O. `main.rs`'s `--tui` check is dispatched FIRST (before the `--problem-spec` early
return and the `--headless` check) - `--tui --problem-spec <path>` spawns `run_training_user_
problem` (gets AMR, which `--problem-spec` alone does NOT - the channel-less `run_headless_
user_problem` never wires it); `--tui` alone builds `config` exactly like the GUI branch and
spawns `run_training`/`run_training_pinlug` per `--problem kirsch|pinlug`. Every pre-existing
path (`--headless`, `--problem-spec` without `--tui`, plain GUI) is byte-for-byte unchanged -
`--tui`'s check returns before any of that code runs.

**A real, disclosed v1 gap found via this session's own binary smoke test, not a crash**: a
handful of unconditional `println!` decision-maker tier-transition diagnostics in the SHARED
`runner.rs`/`headless.rs` training code (correct and intended for `--headless`) write directly
to the process's real stdout, bypassing the `TrainingMsg` channel - since the TUI's alternate
screen occupies that same stdout, a tier transition firing mid-run visibly (transiently)
corrupts the dashboard for one line before the next redraw overwrites it. A proper fix (OS-level
stdout redirect, or plumbing a "quiet" sink through the shared training functions) was
deliberately not attempted - it touches code this project has repeatedly flagged as risky to
modify casually, for a cosmetic (not correctness/data-loss) issue. See `tui/mod.rs`'s own doc
comment.

Verified: `cargo build -p pinn-app` and `--features ndarray-backend` both clean; 12 new pure-
logic unit tests, all passing; a real binary smoke run confirmed the process does not panic
(this sandbox's shell apparently satisfies `enable_raw_mode()` even under `< /dev/null`
redirection, so it entered a genuinely live training session rather than failing cleanly as
first expected - which is how the stdout-corruption gap above was actually found). A real
mouse/keyboard-driven interactive session was NOT performed - not possible from this sandbox,
the same disclosed gap every other GUI-adjacent feature in this file already carries.

