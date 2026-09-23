//! `--tui` terminal dashboard: reuses the SAME training entry points (`run_training`,
//! `run_training_pinlug`, `pinn_solver::runner::run_training_user_problem`) the GUI already
//! drives over a `(Sender<TrainingMsg>, Receiver<ControlMsg>)` channel pair - no solver-side
//! change, zero new physics/output path. See `docs/tui-mode-plan.md` for the full design.
//!
//! This is the ONLY file in this module that touches `crossterm` I/O (terminal lifecycle, raw
//! mode, alternate screen, panic hook, the event loop). `state.rs` is pure data; `ui.rs` is
//! pure rendering - neither touches a terminal or a channel.
//!
//! **Stray-`println!` corruption, found during this session's own smoke test, now fixed via an
//! OS-level stdout redirect (Unix only)**: a small number of unconditional `println!` calls in
//! the shared `runner.rs`/`headless.rs` decision-maker tier-transition diagnostics
//! (`"[DM@{step}] ... -> ..."`, correct and intended for `--headless`) write directly to the
//! process's real stdout (fd 1), bypassing the `TrainingMsg` channel entirely. Rather than touch
//! that shared, repeatedly-flagged-as-risky-to-modify training-loop code, `StdoutGuard::install`
//! `dup()`s the real terminal fd aside for ratatui's OWN rendering to write through, then
//! `dup2()`s fd 1 onto `/dev/null` for the duration of the alternate screen - any `println!`/
//! `print!` anywhere in the process (this training thread included) is silently swallowed,
//! while the TUI's own frames (written through the duped fd, never through `std::io::stdout()`)
//! render normally. `Drop` restores fd 1 to the real terminal. On non-Unix targets this guard is
//! a no-op (`stdout()` is used directly, same as before) - the corruption is Unix-fixed, Windows
//! keeps the pre-existing, disclosed cosmetic gap.

mod setup;
mod state;
mod ui;

pub use setup::{run_setup_menu, SetupChoice};

use std::io::Write;
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

#[cfg(unix)]
mod stdout_guard {
    use std::ffi::CString;
    use std::fs::File;
    use std::os::unix::io::{FromRawFd, RawFd};

    fn check(fd: RawFd) -> std::io::Result<RawFd> {
        if fd < 0 { Err(std::io::Error::last_os_error()) } else { Ok(fd) }
    }

    /// Redirects the process's real fd 1 to `/dev/null` and hands back a `File` (a separate
    /// `dup()` of the ORIGINAL terminal fd) for the TUI's own rendering to write through
    /// instead. `Drop` restores fd 1 to the real terminal.
    pub struct StdoutGuard {
        real_fd: RawFd,
    }

    impl StdoutGuard {
        pub fn install() -> std::io::Result<(Self, File)> {
            unsafe {
                let real_fd = check(libc::dup(1))?;
                let devnull = CString::new("/dev/null").unwrap();
                let devnull_fd = check(libc::open(devnull.as_ptr(), libc::O_WRONLY))?;
                let dup_result = libc::dup2(devnull_fd, 1);
                libc::close(devnull_fd);
                check(dup_result)?;
                // A second dup of real_fd so the File this returns owns an independent
                // descriptor - StdoutGuard's own Drop closes `real_fd` separately on restore.
                let render_fd = check(libc::dup(real_fd))?;
                Ok((StdoutGuard { real_fd }, File::from_raw_fd(render_fd)))
            }
        }
    }

    impl Drop for StdoutGuard {
        fn drop(&mut self) {
            unsafe {
                libc::dup2(self.real_fd, 1);
                libc::close(self.real_fd);
            }
        }
    }
}

#[cfg(unix)]
type TerminalOut = std::fs::File;
#[cfg(not(unix))]
type TerminalOut = std::io::Stdout;

#[cfg(unix)]
fn open_terminal_out() -> anyhow::Result<(Option<stdout_guard::StdoutGuard>, TerminalOut)> {
    let (guard, file) = stdout_guard::StdoutGuard::install()
        .map_err(|e| anyhow::anyhow!("failed to redirect stdout for --tui ({e}) - falling back is not attempted, this is a hard error so the cause is never silently swallowed"))?;
    Ok((Some(guard), file))
}
#[cfg(not(unix))]
fn open_terminal_out() -> anyhow::Result<(Option<()>, TerminalOut)> {
    Ok((None, std::io::stdout()))
}

/// Sets up raw mode + alternate screen (writing through a fd that is NEVER the process's real
/// stdout - see `StdoutGuard` above), installs the mandatory panic hook (ratatui's own guidance:
/// a mid-run panic must not leave the user's shell in raw mode / on the alternate screen), runs
/// `body`, then restores the terminal unconditionally (even if `body` returned an error) before
/// propagating that error.
fn with_terminal<F>(body: F) -> anyhow::Result<()>
where
    F: FnOnce(&mut ratatui::Terminal<ratatui::backend::CrosstermBackend<TerminalOut>>) -> anyhow::Result<()>,
{
    enable_raw_mode().map_err(|e| anyhow::anyhow!("--tui requires an interactive terminal (enable_raw_mode failed: {e})"))?;
    let (_guard, mut out) = open_terminal_out()?;
    execute!(out, EnterAlternateScreen).map_err(|e| anyhow::anyhow!("failed to enter alternate screen: {e}"))?;
    let backend = ratatui::backend::CrosstermBackend::new(out);
    let mut terminal = ratatui::Terminal::new(backend).map_err(|e| anyhow::anyhow!("failed to construct terminal: {e}"))?;

    // The panic hook only needs to leave the alternate screen / disable raw mode - it must not
    // try to write through `_guard`'s (possibly already-dropped-by-then) fd, so it opens its own
    // short-lived handle on whatever the real terminal currently is. On Unix that's `/dev/tty`
    // (fd 1 may still be redirected to `/dev/null` at panic time); elsewhere, `stdout()`.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        #[cfg(unix)]
        {
            if let Ok(mut tty) = std::fs::OpenOptions::new().write(true).open("/dev/tty") {
                let _ = execute!(tty, LeaveAlternateScreen);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
        }
        default_hook(info);
    }));

    let result = body(&mut terminal);

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.backend_mut().flush();
    // Restore the default panic hook so a LATER, unrelated panic (after this function returns)
    // isn't attributed to a terminal state this function no longer owns. `_guard` (if Unix)
    // drops here too, restoring the process's real fd 1 before this function returns.
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
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<TerminalOut>>,
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
