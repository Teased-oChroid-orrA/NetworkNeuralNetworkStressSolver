//! Terminal ownership, bounded channels, and input handling. No physics calculations.
mod setup;
mod state;
mod ui;
pub use setup::{run_setup_menu, SetupChoice};
use std::{io::Write, time::{Duration, Instant}};
use crossbeam_channel::{bounded, TryRecvError};
use crossterm::{event::{self, Event, KeyCode, KeyModifiers, KeyEventKind}, execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen}};
use pinn_core::{messages::{ControlMsg, ProblemKind, SolverConfig, TrainingMsg}, problem_spec::ProblemSpec};
use state::{RunStatus, TuiState};
const TICK: Duration = Duration::from_millis(120);

#[cfg(unix)]
mod stdout_guard {
    use std::{fs::File, os::fd::{AsRawFd, FromRawFd}};
    fn duplicate(fd: i32) -> std::io::Result<File> {
        let copy = unsafe { libc::dup(fd) };
        if copy < 0 { Err(std::io::Error::last_os_error()) }
        else { Ok(unsafe { File::from_raw_fd(copy) }) }
    }
    pub struct StdoutGuard { stdout: File, stderr: File }
    impl StdoutGuard {
        pub fn install(log_path: Option<&std::path::Path>) -> std::io::Result<(Self, File)> {
            // Own each descriptor immediately, including all partial-failure paths.
            let guard = Self { stdout: duplicate(1)?, stderr: duplicate(2)? };
            let render = guard.stdout.try_clone()?;
            let sink = if let Some(path) = log_path {
                std::fs::OpenOptions::new().create(true).append(true).open(path)?
            } else { std::fs::OpenOptions::new().write(true).open("/dev/null")? };
            for target in [1, 2] {
                if unsafe { libc::dup2(sink.as_raw_fd(), target) } < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok((guard, render))
        }
    }
    impl Drop for StdoutGuard {
        fn drop(&mut self) {
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            unsafe { libc::dup2(self.stdout.as_raw_fd(), 1); libc::dup2(self.stderr.as_raw_fd(), 2); }
        }
    }
    use std::io::Write;
}
#[cfg(unix)]
type TerminalOut = std::fs::File;
#[cfg(not(unix))]
type TerminalOut = std::io::Stdout;

fn restore_terminal() {
    let _ = disable_raw_mode();
    #[cfg(unix)]
    if let Ok(mut tty) = std::fs::OpenOptions::new().write(true).open("/dev/tty") {
        let _ = execute!(tty, LeaveAlternateScreen, crossterm::cursor::Show);
    }
    #[cfg(not(unix))]
    { let _ = execute!(std::io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show); }
}
struct TerminalGuard;
impl Drop for TerminalGuard { fn drop(&mut self) { restore_terminal(); } }

fn with_terminal<F>(body: F) -> anyhow::Result<()>
where F: FnOnce(&mut ratatui::Terminal<ratatui::backend::CrosstermBackend<TerminalOut>>) -> anyhow::Result<()> {
    with_terminal_log(None, body)
}

fn with_terminal_log<F>(log_path: Option<&std::path::Path>, body: F) -> anyhow::Result<()>
where F: FnOnce(&mut ratatui::Terminal<ratatui::backend::CrosstermBackend<TerminalOut>>) -> anyhow::Result<()> {
    enable_raw_mode().map_err(|e| anyhow::anyhow!("--tui requires an interactive terminal: {e}"))?;
    let _terminal_guard = TerminalGuard; // Covers every subsequent setup error and unwind.
    #[cfg(unix)]
    let (_output_guard, mut out) = stdout_guard::StdoutGuard::install(log_path)?;
    #[cfg(not(unix))]
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut terminal = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(out))?;
    let previous = std::sync::Arc::new(std::panic::take_hook());
    let saved = previous.clone();
    let ui_thread = std::thread::current().id();
    std::panic::set_hook(Box::new(move |info| {
        if std::thread::current().id() == ui_thread { restore_terminal(); }
        saved(info);
    }));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&mut terminal)));
    drop(std::panic::take_hook());
    match std::sync::Arc::try_unwrap(previous) {
        Ok(hook) => std::panic::set_hook(hook),
        Err(hook) => std::panic::set_hook(Box::new(move |info| hook(info))),
    }
    match result { Ok(result) => result, Err(payload) => std::panic::resume_unwind(payload) }
}

/// Own cancellation even when drawing/input fails or the UI unwinds.
struct WorkerGuard { control: crossbeam_channel::Sender<ControlMsg> }
impl Drop for WorkerGuard { fn drop(&mut self) { let _ = self.control.try_send(ControlMsg::Stop); } }

fn handle_key(key: crossterm::event::KeyEvent, state: &mut TuiState,
    tx: &crossbeam_channel::Sender<ControlMsg>) -> bool {
    if key.kind != KeyEventKind::Press { return false; }
    if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) { return true; }
    match key.code {
        KeyCode::Tab => { state.tab = (state.tab + 1) % ui::TABS.len(); state.scroll = 0; }
        KeyCode::BackTab => { state.tab = (state.tab + ui::TABS.len() - 1) % ui::TABS.len(); state.scroll = 0; }
        KeyCode::Char(c @ '1'..='6') => { state.tab = c as usize - '1' as usize; state.scroll = 0; }
        KeyCode::Down => state.scroll = state.scroll.saturating_add(1),
        KeyCode::Up => state.scroll = state.scroll.saturating_sub(1),
        KeyCode::Char('f') => state.warnings_only = !state.warnings_only,
        KeyCode::Char(c @ ('p' | 'r')) if state.status != Some(RunStatus::Done) => {
            let (command, label) = if c == 'p' { (ControlMsg::Pause, "Pause") } else { (ControlMsg::Resume, "Resume") };
            state.control_status = match tx.try_send(command) {
                Ok(()) => format!("{label} requested; solver does not acknowledge this control."),
                Err(e) => format!("ERROR {label} request failed: {e}"),
            };
            state.log(state.control_status.clone());
        }
        KeyCode::Char('c') => { state.control_status = "Checkpoint control unavailable here; use the GUI's supported save workflow.".into(); }
        _ => {}
    }
    false
}

fn event_loop(terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<TerminalOut>>,
    rx: crossbeam_channel::Receiver<TrainingMsg>, tx_ctrl: crossbeam_channel::Sender<ControlMsg>,
    mut state: TuiState, mut diagnostics: std::fs::File) -> anyhow::Result<()> {
    let _worker_guard = WorkerGuard { control: tx_ctrl.clone() };
    let started = Instant::now();
    loop {
        // Apply all queued events in order. A terminal event must not discard its preceding final update.
        for _ in 0..32 {
            match rx.try_recv() {
                Ok(TrainingMsg::Update(u)) => {
                    state.apply(&u);
                    if let Some(objective) = &u.objective {
                        for term in &objective.terms {
                            writeln!(diagnostics, "{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:?},{:.9e}", u.step,
                                term.name, term.normalized_raw, term.effective_weight, term.weighted,
                                objective.total, objective.grad_norm_before_optimizer, objective.reconstruction_error)?;
                        }
                    }
                }
                Ok(TrainingMsg::PinLugUpdate(u)) => state.apply_pinlug(&u),
                Ok(TrainingMsg::Done) => { state.set_done(); state.log("Run completed; physical acceptance must be checked separately.".into()); diagnostics.flush()?; }
                Ok(TrainingMsg::Error(e)) => { state.log(format!("ERROR {e}")); state.set_error(e); diagnostics.flush()?; }
                Ok(TrainingMsg::CheckpointSaved(result)) => state.log(format!("Checkpoint: {result:?}")),
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if !matches!(state.status, Some(RunStatus::Done | RunStatus::Error)) {
                        state.set_error("Training worker disconnected before reporting completion".into());
                    }
                    break;
                }
            }
        }
        terminal.draw(|frame| ui::render(frame, &state))?;
        if event::poll(TICK)? {
            if let Event::Key(key) = event::read()? {
                if handle_key(key, &mut state, &tx_ctrl) { break; }
            }
        }
        if state.status == Some(RunStatus::Error) {
            anyhow::bail!("{} (elapsed {:.1}s)", state.error_msg.as_deref().unwrap_or("training failed"), started.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

fn run_worker(summary: String, max_steps: usize,
    worker: impl FnOnce(crossbeam_channel::Sender<TrainingMsg>, crossbeam_channel::Receiver<ControlMsg>) + Send + 'static) -> anyhow::Result<()> {
    // Create a unique run directory before training; never overwrite user checkpoints/reports.
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let run_dir = std::env::temp_dir().join(format!("pinn-run-{stamp}-{}", std::process::id()));
    std::fs::create_dir(&run_dir)?;
    std::fs::write(run_dir.join("configuration.txt"), &summary)?;
    let mut diagnostics = std::fs::OpenOptions::new().write(true).create_new(true).open(run_dir.join("objective.csv"))?;
    writeln!(diagnostics, "step,term,normalized_raw,effective_weight,weighted,total,gradient_norm,reconstruction_error")?;
    let mut state = TuiState::new(max_steps);
    state.run_summary = format!("{summary}\nDiagnostics: {}\nHistory: last 1024 updates. UI refresh <= 8.4 Hz.\nCheckpoint/provenance details unavailable unless supplied by solver.", run_dir.display());
    let result = with_terminal_log(Some(&run_dir.join("solver.log")), |terminal| {
        let (tx, rx) = bounded(8);
        let (ctrl, controls) = bounded(8);
        let worker_guard = WorkerGuard { control: ctrl.clone() };
        let failure = tx.clone();
        let handle = std::thread::spawn(move || {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker(tx, controls))) {
                let message = payload.downcast_ref::<String>().cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "unknown worker panic".into());
                let _ = failure.send(TrainingMsg::Error(message));
            }
        });
        let result = event_loop(terminal, rx, ctrl, state, diagnostics);
        drop(worker_guard);
        // event_loop has dropped the receiver, so a worker's final blocking send cannot deadlock this join.
        let deadline = Instant::now() + Duration::from_secs(30);
        while !handle.is_finished() && Instant::now() < deadline { std::thread::sleep(Duration::from_millis(20)); }
        anyhow::ensure!(handle.is_finished(), "worker did not stop within 30 seconds after cancellation");
        handle.join().map_err(|_| anyhow::anyhow!("training worker panicked"))?;
        result
    });
    eprintln!("Run diagnostics: {}", run_dir.display());
    result
}

pub fn run_tui_plate(spec: ProblemSpec) -> anyhow::Result<()> {
    let summary = format!("Plate / GUI-shared training path\nBackend: {} / f32\n{}", backend_name(), toml::to_string_pretty(&spec)?);
    let max_steps = spec.training.max_steps;
    run_worker(summary, max_steps, move |tx, rx| pinn_solver::runner::run_training_user_problem(spec, tx, rx))
}
pub fn run_tui_live(config: SolverConfig, kind: ProblemKind) -> anyhow::Result<()> {
    let summary = format!("Problem: {kind:?}\nBackend: {} / f32\nNetwork: {} x {}\nInterior/boundary: {}/{}\nFD h: {}\nMaterial E={} Pa, nu={}\nLoad px={}, py={} Pa\nHalf width/height: {}/{} m\nSOAP-Muon: {}; decision maker: {}\nRequested configuration; engine-derived overrides may apply.",
        backend_name(), config.hidden_dim, config.n_hidden, config.n_interior, config.n_boundary,
        config.fd_h, config.material.e, config.material.nu, config.load.px, config.load.py,
        config.geometry.half_w, config.geometry.half_h, config.use_soap_muon, config.decision_maker.enabled);
    let max_steps = config.max_steps;
    run_worker(summary, max_steps, move |tx, rx| match kind {
        ProblemKind::Kirsch => pinn_solver::run_training(config, tx, rx),
        ProblemKind::PinLug => pinn_solver::run_training_pinlug(config, tx, rx),
    })
}
fn backend_name() -> &'static str { if cfg!(feature = "ndarray-backend") { "NdArray" } else { "WGPU" } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn navigation_and_controls_are_requests_not_false_acknowledgments() {
        let (tx, rx) = bounded(2); let mut state = TuiState::new(10);
        let key = |c| crossterm::event::KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
        assert!(!handle_key(key('2'), &mut state, &tx)); assert_eq!(state.tab, 1);
        handle_key(key('p'), &mut state, &tx);
        assert!(matches!(rx.try_recv(), Ok(ControlMsg::Pause)));
        assert!(state.control_status.contains("requested"));
        assert!(handle_key(key('q'), &mut state, &tx));
    }
    #[test]
    fn worker_guard_requests_stop_on_error_paths() {
        let (tx, rx) = bounded(2); { let _guard = WorkerGuard { control: tx }; }
        assert!(matches!(rx.recv().unwrap(), ControlMsg::Stop));
    }
}
