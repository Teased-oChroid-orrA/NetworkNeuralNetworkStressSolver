# Ratatui TUI mode for pinn-app

## Context

`pinn-app` currently has exactly two run modes: `--headless` (plain scrolling `println!` logs,
no AMR wired for the user-defined-plate path) and the default GUI (`eframe`/`egui`, needs a
display/window compositor). There's no middle ground — a way to watch a run live (loss curve,
Kt, per-hole stress, AMR/architecture events) over SSH or in a terminal-only environment without
launching a window. The user wants a `ratatui`-based terminal dashboard, reached via a new CLI
flag, that reuses the SAME training entry points the GUI already drives (`run_training`,
`run_training_pinlug`, `runner::run_training_user_problem`) rather than a new physics/output
path — so it gets Kirsch, pin-lug, and the User-Defined plate (including every PH4-45 `[architecture]`
selection) for free, with zero solver-side changes.

## Design

**Reuse, don't reimplement.** All three `run_training*` functions already take
`(input, Sender<TrainingMsg>, Receiver<ControlMsg>)` and stream `TrainingMsg` — the exact
protocol `pinn-gui`'s `app.rs::start_solver`/`drain_channel` already consumes over a spawned
thread (`bounded(1)` channel, "keep only the latest message" drain). The TUI reuses that same
threading pattern verbatim; nothing in `pinn-solver`/`pinn-core` changes.

**State-application logic is NOT shared with `pinn-gui`.** `pinn-gui::app.rs`'s `apply_msg` is
private and egui-coupled (touches `TextureHandle`, `ColorbarRange`); `app-egui` (the sibling
PowerShell-toolbox project) already independently reimplements the same `TrainingMsg` ->
state-struct mapping for its own richer UI. A third, TUI-scoped implementation is consistent
with that existing precedent, not a new pattern — and keeps `pinn-app` from depending on `egui`
types it doesn't need for a terminal renderer.

**Modularity** — three focused files, each with one job and minimal coupling to the others:

```
crates/pinn-app/src/tui/
  mod.rs    - public entry points + terminal lifecycle (raw mode, alt screen, panic hook,
              thread spawn, main event loop). The only file that touches crossterm I/O.
  state.rs  - TuiState struct + TuiState::apply(TrainingMsg) / apply_pinlug(PinLugTrainingUpdate).
              Pure data, zero ratatui/crossterm dependency — unit-testable with plain TrainingUpdate
              literals, no terminal needed.
  ui.rs     - render(frame, &TuiState) - pure ratatui drawing, zero I/O. Testable in isolation
              via ratatui::backend::TestBackend if useful, though not required for v1.
```

`state.rs` mirrors the fields `pinn_core::TrainingState` already has (this session's own PH4-45
work added `hole_analyses` there) — `TuiState` is a plain struct in `pinn-app`, not a reuse of
`pinn_core::TrainingState` itself, since the TUI has no use for `TrainingState`'s `Array2<f32>`
vis-grid fields (no heatmap in v1 — see below) and duplicating five scalar/Vec fields is cheaper
than carrying that dependency.

**Scope: Kirsch, pin-lug, and User-Defined plate — no heatmap in v1.** All three CLI-reachable
problem kinds get the TUI, since the shared skeleton (title/summary bar, loss chart, stats
sidebar, status/help footer) covers all of them with only the Kt-vs-`convergence_metric` and
`TrainingMsg::Update`-vs-`PinLugUpdate` branches differing. Rendering the actual stress-field
heatmap as terminal half-block/braille cells is deliberately deferred — real added complexity
(color-cell mapping, colormap duplication from `pinn-gui::heatmap`, terminal color-depth
assumptions) for a v1 whose main value is the numbers a user actually watches a run for (loss,
Kt, AMR/architecture events, equilibrium/energy-balance error once available). Flagged as a
natural v2, not silently dropped.

**Dashboard layout** (ratatui `Layout`, vertical split):
- Top: one-line problem summary (geometry/material/load for plate; config summary for Kirsch/pin-lug).
- Middle, horizontal split:
  - Left ~65%: `Chart` widget, total/energy/neumann loss history (`Dataset`s, `Marker::Braille`),
    y-axis pre-transformed to `ln(loss)` (ratatui has no native log axis) with an explicit axis label.
  - Right ~35%: stats block — step/max_steps, lr, n_colloc, grad_norm, Kt (`kt_estimate` and/or
    per-hole `hole_analyses` for plate; `convergence_metric` for pin-lug), latest AMR sweep
    (points before/after, step), latest architecture event if any.
- Bottom: status line (Idle/Running/Done/Error + elapsed) and a keybinding hint (`q`/`Ctrl+C`:
  stop and quit; training also exits cleanly on its own `Done`).

**Event loop** (`mod.rs`): redraw on a ~120ms tick via `crossterm::event::poll`, non-blocking
`try_recv`-drain the training channel each tick (identical pattern to `drain_channel`), apply the
latest message into `TuiState`, `terminal.draw(|f| ui::render(f, &state))`. On `q`/Esc/Ctrl+C:
send `ControlMsg::Stop`, break. **Panic hook is mandatory** — ratatui's own guidance: install a
hook that restores the terminal (disable raw mode, leave alternate screen) before delegating to
the default panic hook, or a mid-run panic leaves the user's shell broken.

**CLI wiring** (`main.rs`): add `--tui`/`-T` flag parsing (same style as the existing
`parse_problem_spec_arg`/headless-flag checks). Dispatch reordered so `--tui` is checked
*before* the current unconditional `--problem-spec` -> `run_headless_user_problem` early return
(today `--problem-spec` always goes headless regardless of any other flag):
- `--tui --problem-spec <path>`: parse the same way the existing headless branch does, spawn
  `runner::run_training_user_problem` on a thread, run the TUI loop.
- `--tui` alone: build `config` exactly like the existing GUI branch does (`default_kirsch`/
  `default_pinlug` + `apply_env`), spawn `run_training`/`run_training_pinlug` per `problem_kind`,
  run the TUI loop.
- Every existing path (`--headless`, `--problem-spec` without `--tui`, plain GUI) is untouched —
  `--tui` is checked first but every branch it doesn't take falls through to today's exact code.

**Dependencies** (`Cargo.toml`, following this workspace's own `[workspace.dependencies]` +
justification-comment convention): add `ratatui` and `crossterm` there, referenced via
`{ workspace = true }` in `pinn-app/Cargo.toml` only (no other crate needs them). No network
access from this sandbox to confirm exact latest patch versions — start with `ratatui = "0.29"`,
`crossterm = "0.28"` and let `cargo build` resolve/report a mismatch if the pinned minors don't
line up; adjust empirically rather than guessing further.

## Verification

- `cargo build -p pinn-app` and `cargo build -p pinn-app --features ndarray-backend`, both clean.
- `cargo test -p pinn-app` — existing 5 tests unaffected, plus new pure-logic tests in `state.rs`
  for `TuiState::apply`/`apply_pinlug` (construct a `TrainingUpdate`/`PinLugTrainingUpdate`
  literal — same pattern this session already used for the app-egui `kt_estimate` regression
  tests — assert the right fields land, `hole_analyses` vs `kt_estimate` mutual exclusivity for
  the plate path, loss history appends not overwrites).
- Manual smoke run (mine, in this sandbox): confirm the binary starts, doesn't panic, and prints
  a clear error rather than corrupting the terminal if `enable_raw_mode()` fails in a non-tty
  context (this sandbox's shell may not be a real interactive tty, so a full rendered-frame
  check likely isn't possible here — disclosed explicitly, matching this project's own existing
  "not independently verified end-to-end" precedent for GUI work).
- **Real interactive check is on you**: `cargo run -p pinn-app --release -- --tui --problem-spec examples/problems/single_hole_plate.toml`
  and `--tui --problem kirsch`, confirm the chart/stats update live and `q` restores the terminal
  cleanly.
- Update `CLAUDE.md`'s GUI section (or a new short section) noting the TUI mode exists, what it
  covers, and the disclosed heatmap-deferred/interactive-check-not-verified gaps — matching this
  project's own documentation discipline for every other GUI-adjacent feature in this file.

## Status: implemented, per this plan with real corrections found along the way

`crates/pinn-app/src/tui/{mod,state,ui}.rs` + `main.rs`'s `--tui`/`-T` dispatch are all
implemented, matching this plan's module layout exactly. A real, session-only investigation
pass (before writing code) corrected several claims this plan made based on assumption rather
than verification - see the corresponding "Investigated facts" section of the issue #78
five-item plan for the full record. In summary, all folded into the implementation:

- `run_training_user_problem`'s default (Joint) path does NOT return after `Done` - it stays
  alive serving `SaveCheckpoint` until it sees `ControlMsg::Stop`. `mod.rs`'s event loop sends
  `Stop` explicitly on `Done`/`Error`, unconditionally (harmless for the other paths that
  already return on their own - the receiver is already dropped, `send` just no-ops).
- `Done` is a BLOCKING `tx.send` on the `bounded(1)` data channel (unlike `Update`'s
  `try_send`) - the event loop keeps draining every tick even after the UI "looks done," never
  stopping polling early.
- `TrainingMsg` has 10 variants (not `#[non_exhaustive]`) - `mod.rs`'s `match` covers all of
  them explicitly, no-op arms for the 6 unreachable-from-this-TUI ones.
- Neither `TrainingUpdate` nor `PinLugTrainingUpdate` carries `max_steps` - `TuiState::new`
  takes it from the caller's own `config`/`spec`, captured before the thread spawns.
- `PinLugTrainingUpdate.amr_sweep` is `Vec<AmrSweepReport>` (0-2 entries), NOT `Option` like
  `TrainingUpdate`'s - `apply`/`apply_pinlug` handle the two shapes distinctly (`apply_pinlug`
  takes the LAST entry, i.e. the most recently reported domain).
- `--tui --headless`: `--tui` wins (checked and dispatched first in `main.rs`, before either
  the `--problem-spec` early return or the `--headless` check) - the strictly more capable
  request.

**A real, disclosed v1 gap found during this session's own smoke test** (not anticipated by
this plan): a handful of unconditional `println!` decision-maker tier-transition diagnostics in
the SHARED `runner.rs`/`headless.rs` training code write directly to the process's real stdout,
bypassing the `TrainingMsg` channel - since the TUI's alternate screen occupies that same
stdout, a tier transition firing mid-run visibly (if transiently) corrupts the dashboard for one
line. A proper fix (OS-level stdout redirect, or plumbing a "quiet" sink through the shared
training functions) was deliberately NOT attempted this pass - it touches code this project has
repeatedly flagged as risky to modify casually. Cosmetic only, not a crash or data-loss risk -
see `crates/pinn-app/src/tui/mod.rs`'s own doc comment for the full record.

**Verification actually performed**: `cargo build -p pinn-app` and `--features ndarray-backend`
both clean; 12 new pure-logic unit tests for `TuiState::apply`/`apply_pinlug` (all passing -
history appends not overwrites, `kt_estimate` vs. `hole_analyses` precedence, the pin-lug
`Vec`-vs-`Option` `amr_sweep` shape, `architecture_event` surfaced); a real binary smoke run in
this sandbox confirmed the process does NOT panic and (surprisingly - this sandbox's shell
apparently does present something `crossterm::terminal::enable_raw_mode` accepts, even under
`< /dev/null` redirection) enters a genuinely live TUI training session rather than failing
cleanly as originally expected - which is how the stdout-corruption gap above was actually
found. **A real, mouse/keyboard-driven interactive session (confirming the chart/stats visibly
update and `q` restores the terminal cleanly) was NOT performed** - not possible from this
sandbox, matching this project's own established precedent for every other GUI-adjacent
feature's own disclosed gap.
