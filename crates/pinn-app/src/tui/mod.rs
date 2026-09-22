//! `--tui` terminal dashboard: reuses the SAME training entry points (`run_training`,
//! `run_training_pinlug`, `pinn_solver::runner::run_training_user_problem`) the GUI already
//! drives over a `(Sender<TrainingMsg>, Receiver<ControlMsg>)` channel pair - no solver-side
//! change, zero new physics/output path. See `docs/tui-mode-plan.md` for the full design.
//!
//! This is the ONLY file in this module that touches `crossterm` I/O (terminal lifecycle, raw
//! mode, alternate screen, panic hook, the event loop). `state.rs` is pure data; `ui.rs` is
//! pure rendering - neither touches a terminal or a channel.
//!
//! **Known, disclosed v1 gap, found during this session's own smoke test**: a small number of
//! unconditional `println!` calls in the shared `runner.rs`/`headless.rs` decision-maker
//! tier-transition diagnostics (`"[DM@{step}] ... -> ..."`, correct and intended for
//! `--headless`) write directly to the process's real stdout, bypassing the `TrainingMsg`
//! channel entirely - since this TUI's alternate screen occupies that SAME stdout, a tier
//! transition firing mid-run will visibly corrupt the rendered dashboard for one line before
//! the next redraw overwrites it. A real OS-level stdout redirect (or plumbing a "quiet" sink
//! through the shared training functions) would fix this properly but touches code this
//! project has repeatedly, explicitly flagged as risky to modify casually (the frozen/shared
//! training-loop paths) - deliberately NOT attempted in this pass. Cosmetic only (one
//! transient garbled line, not a crash, not lost training state, not incorrect physics) - a
//! real, scoped follow-up, not silently glossed over.

mod state;
mod ui;

use std::time::Duration;

use crossbeam_channel::{bounded, unbounded, TryRecvError};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use pinn_core::messages::{ControlMsg, ProblemKind, SolverConfig, TrainingMsg};
use pinn_core::problem_spec::ProblemSpec;

use state::TuiState;

/// Redraw/drain tick - non-blocking `try_recv` each tick (identical pattern to `pinn-gui`'s own
/// `drain_channel`), so this never blocks waiting for a message that may not come this tick.
const TICK: Duration = Duration::from_millis(120);

/// Sets up raw mode + alternate screen, installs the mandatory panic hook (ratatui's own
/// guidance: a mid-run panic must not leave the user's shell in raw mode / on the alternate
/// screen), runs `body`, then restores the terminal unconditionally (even if `body` returned an
/// error) before propagating that error.
fn with_terminal<F>(body: F) -> anyhow::Result<()>
where
    F: FnOnce(&mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>) -> anyhow::Result<()>,
{
    enable_raw_mode().map_err(|e| anyhow::anyhow!("--tui requires an interactive terminal (enable_raw_mode failed: {e})"))?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| anyhow::anyhow!("failed to enter alternate screen: {e}"))?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend).map_err(|e| anyhow::anyhow!("failed to construct terminal: {e}"))?;

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
        default_hook(info);
    }));

    let result = body(&mut terminal);

    let _ = disable_raw_mode();
    let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    // Restore the default panic hook so a LATER, unrelated panic (after this function returns)
    // isn't attributed to a terminal state this function no longer owns.
    let _ = std::panic::take_hook();

    result
}

/// Non-blocking: was `q`/`Esc`/Ctrl+C pressed since the last poll?
fn quit_requested() -> anyhow::Result<bool> {
    if !event::poll(Duration::from_millis(0)).map_err(|e| anyhow::anyhow!("input poll failed: {e}"))? {
        return Ok(false);
    }
    if let Event::Key(key) = event::read().map_err(|e| anyhow::anyhow!("input read failed: {e}"))? {
        let ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        return Ok(matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) || ctrl_c);
    }
    Ok(false)
}

/// The shared event loop both entry points below run: drain the training channel (keeping the
/// LATEST message only, per `TrainingMsg`'s own `bounded(1)`/`try_send` "drop when full"
/// contract - draining every tick, not stopping early, is required since `Done` is a BLOCKING
/// `tx.send` on that same channel elsewhere; a TUI that stopped polling could hang the solver
/// thread), apply it into `TuiState`, redraw, and watch for quit/completion.
fn event_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    rx: crossbeam_channel::Receiver<TrainingMsg>,
    tx_ctrl: crossbeam_channel::Sender<ControlMsg>,
    mut state: TuiState,
) -> anyhow::Result<()> {
    loop {
        let mut last = None;
        loop {
            match rx.try_recv() {
                Ok(msg) => last = Some(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        let mut done = false;
        if let Some(msg) = last {
            match msg {
                TrainingMsg::Update(u) => state.apply(&u),
                TrainingMsg::PinLugUpdate(u) => state.apply_pinlug(&u),
                TrainingMsg::Done => { state.set_done(); done = true; }
                TrainingMsg::Error(e) => { state.set_error(e); done = true; }
                // Not reachable from this TUI's three dispatch paths (Kirsch/pin-lug/plate),
                // but `TrainingMsg` isn't `#[non_exhaustive]` - every variant must be covered.
                TrainingMsg::BeamUpdate(_)
                | TrainingMsg::ParametricUpdate(_)
                | TrainingMsg::ParametricReady
                | TrainingMsg::ParametricInferResult(_)
                | TrainingMsg::ExportComplete(_)
                | TrainingMsg::CheckpointSaved(_) => {}
            }
        }

        terminal.draw(|frame| ui::render(frame, &state))?;

        if done {
            // Issue: `run_training_user_problem`'s default (Joint) path does NOT return after
            // sending `Done` - it stays alive serving `SaveCheckpoint` until it sees
            // `ControlMsg::Stop`. Every other path (Kirsch, pin-lug, SequentialTwoStage,
            // annular) already returns on its own - sending `Stop` there is harmless (the
            // receiver is already dropped; `send` just returns an ignorable `Err`).
            let _ = tx_ctrl.send(ControlMsg::Stop);
            // One more tick so the user actually sees the Done/Error screen before the loop
            // exits, rather than the terminal flashing back immediately.
            std::thread::sleep(TICK);
            let _ = quit_requested()?;
            return Ok(());
        }

        if quit_requested()? {
            let _ = tx_ctrl.send(ControlMsg::Stop);
            return Ok(());
        }

        std::thread::sleep(TICK);
    }
}

/// `--tui --problem-spec <path>`: the User-Defined plate path, spawned on its own thread
/// exactly like `pinn-gui`'s own `start_solver` does. Gets AMR + every `[architecture]`
/// selection "for free" - a strictly more capable path than `--problem-spec` alone (which
/// routes through the channel-less, AMR-free `run_headless_user_problem`).
pub fn run_tui_plate(spec: ProblemSpec) -> anyhow::Result<()> {
    let max_steps = spec.training.max_steps;
    let (tx_train, rx_train) = bounded(1);
    let (tx_ctrl, rx_ctrl) = unbounded();
    let handle = std::thread::spawn(move || {
        pinn_solver::runner::run_training_user_problem(spec, tx_train, rx_ctrl);
    });
    let result = with_terminal(|terminal| event_loop(terminal, rx_train, tx_ctrl, TuiState::new(max_steps)));
    let _ = handle.join();
    result
}

/// `--tui` alone (Kirsch or pin-lug, selected the same way `--headless`/the GUI already do via
/// `--problem <kirsch|pinlug>`): builds `config` exactly like the existing GUI branch does
/// (`default_kirsch`/`default_pinlug` + `apply_env`, already applied by the caller before this
/// is invoked) and spawns the matching `run_training*`.
pub fn run_tui_live(config: SolverConfig, problem_kind: ProblemKind) -> anyhow::Result<()> {
    let max_steps = config.max_steps;
    let (tx_train, rx_train) = bounded(1);
    let (tx_ctrl, rx_ctrl) = unbounded();
    let handle = std::thread::spawn(move || match problem_kind {
        ProblemKind::Kirsch => pinn_solver::run_training(config, tx_train, rx_ctrl),
        ProblemKind::PinLug => pinn_solver::run_training_pinlug(config, tx_train, rx_ctrl),
    });
    let result = with_terminal(|terminal| event_loop(terminal, rx_train, tx_ctrl, TuiState::new(max_steps)));
    let _ = handle.join();
    result
}
