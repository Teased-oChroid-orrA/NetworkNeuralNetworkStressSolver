//! `--tui` alone (no `--problem`/`--problem-spec`): an interactive setup screen letting the user
//! pick the problem and, for a User-Defined plate, load a `ProblemSpec` TOML file from WITHIN
//! the TUI, instead of requiring those choices as CLI flags before launch. Self-contained -
//! `run_setup_menu` owns its own `with_terminal` session (mirrors `run_tui_plate`/`run_tui_live`
//! each owning their own), so `mod.rs`'s training event loop needs zero changes; the menu simply
//! runs first and hands back what to launch.
//!
//! Explicit `--problem <kirsch|pinlug>` / `--problem-spec <path>` on the command line still skip
//! this screen entirely (existing, unaffected direct-launch behavior) - the menu only appears
//! when the user gave `--tui` with no other problem-selecting flag.

use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use pinn_core::problem_spec::ProblemSpec;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use super::{with_terminal, TICK};

/// What the user picked, once confirmed.
pub enum SetupChoice {
    Kirsch,
    PinLug,
    Plate(ProblemSpec),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Option_ {
    Kirsch,
    PinLug,
    UserDefined,
}
const OPTIONS: [Option_; 3] = [Option_::Kirsch, Option_::PinLug, Option_::UserDefined];

impl Option_ {
    fn label(self) -> &'static str {
        match self {
            Option_::Kirsch => "Kirsch (plate with a circular hole, remote tension)",
            Option_::PinLug => "Pin-in-Lug (two-domain contact-mechanics)",
            Option_::UserDefined => "User-Defined (load a --problem-spec TOML)",
        }
    }
}

struct SetupState {
    selected: usize,
    spec_path: String,
    /// `true` once User-Defined is selected and the text field has focus (Tab toggles focus
    /// between the option list and the path field, matching a familiar two-pane form).
    editing_path: bool,
    error: Option<String>,
}

impl SetupState {
    fn new() -> Self {
        Self { selected: 0, spec_path: String::new(), editing_path: false, error: None }
    }

    fn try_confirm(&mut self) -> Option<SetupChoice> {
        match OPTIONS[self.selected] {
            Option_::Kirsch => Some(SetupChoice::Kirsch),
            Option_::PinLug => Some(SetupChoice::PinLug),
            Option_::UserDefined => {
                let path = self.spec_path.trim();
                if path.is_empty() {
                    self.error = Some("enter a --problem-spec TOML path first".to_string());
                    return None;
                }
                match load_and_validate_spec(path) {
                    Ok(spec) => Some(SetupChoice::Plate(spec)),
                    Err(e) => {
                        self.error = Some(e);
                        None
                    }
                }
            }
        }
    }
}

/// Same load+parse+validate sequence `main.rs`'s own `--problem-spec` (non-TUI) path already
/// runs - kept identical here rather than reimplemented, so a spec that's valid on the CLI is
/// valid from this menu too.
fn load_and_validate_spec(path: &str) -> Result<ProblemSpec, String> {
    let spec_str = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read '{path}': {e}"))?;
    let spec: ProblemSpec = toml::from_str(&spec_str)
        .map_err(|e| format!("failed to parse TOML '{path}': {e}"))?;
    spec.geometry.validate().map_err(|e| format!("invalid geometry in '{path}': {e}"))?;
    Ok(spec)
}

fn render(frame: &mut Frame, state: &SetupState) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Span::styled(
            "PINN Structural Stress Solver — select a problem",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new("↑/↓ select   Tab: focus path field (User-Defined)   Enter: start   q/Esc: quit")
            .style(Style::default().fg(Color::DarkGray)),
        rows[1],
    );

    let items: Vec<ListItem> = OPTIONS
        .iter()
        .enumerate()
        .map(|(i, opt)| {
            let marker = if i == state.selected && !state.editing_path { "> " } else { "  " };
            let style = if i == state.selected {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(format!("{marker}{}", opt.label()), style)))
        })
        .collect();
    frame.render_widget(
        List::new(items).block(Block::default().borders(Borders::ALL).title("Problem")),
        rows[2],
    );

    let path_style = if state.editing_path {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    let path_display = if state.spec_path.is_empty() && !state.editing_path {
        "(select User-Defined, then Tab here to type a path)".to_string()
    } else {
        state.spec_path.clone()
    };
    frame.render_widget(
        Paragraph::new(path_display)
            .style(path_style)
            .block(Block::default().borders(Borders::ALL).title("--problem-spec path")),
        rows[3],
    );

    if let Some(err) = &state.error {
        frame.render_widget(
            Paragraph::new(err.as_str()).style(Style::default().fg(Color::Red)),
            rows[4],
        );
    }
}

/// Runs the interactive setup screen in its own terminal session; returns `Ok(None)` if the
/// user quit (`q`/`Esc`/Ctrl+C) without confirming a choice.
pub fn run_setup_menu() -> anyhow::Result<Option<SetupChoice>> {
    let mut state = SetupState::new();
    let mut choice: Option<SetupChoice> = None;

    with_terminal(|terminal| {
        loop {
            terminal.draw(|frame| render(frame, &state))?;

            if !event::poll(Duration::from_millis(0))
                .map_err(|e| anyhow::anyhow!("input poll failed: {e}"))?
            {
                std::thread::sleep(TICK);
                continue;
            }
            let Event::Key(key) = event::read().map_err(|e| anyhow::anyhow!("input read failed: {e}"))? else {
                continue;
            };
            let ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
            if ctrl_c || (!state.editing_path && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)) {
                return Ok(());
            }

            if state.editing_path {
                match key.code {
                    KeyCode::Esc => state.editing_path = false,
                    KeyCode::Tab | KeyCode::BackTab => state.editing_path = false,
                    KeyCode::Enter => {
                        state.error = None;
                        if let Some(c) = state.try_confirm() {
                            choice = Some(c);
                            return Ok(());
                        }
                    }
                    KeyCode::Backspace => { state.spec_path.pop(); }
                    KeyCode::Char(c) => state.spec_path.push(c),
                    _ => {}
                }
                continue;
            }

            match key.code {
                KeyCode::Up => state.selected = state.selected.checked_sub(1).unwrap_or(OPTIONS.len() - 1),
                KeyCode::Down => state.selected = (state.selected + 1) % OPTIONS.len(),
                KeyCode::Tab => {
                    if OPTIONS[state.selected] == Option_::UserDefined {
                        state.editing_path = true;
                    }
                }
                KeyCode::Enter => {
                    state.error = None;
                    if let Some(c) = state.try_confirm() {
                        choice = Some(c);
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
    })?;

    Ok(choice)
}
