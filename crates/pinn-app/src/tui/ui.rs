//! Pure `ratatui` drawing - `render(frame, &TuiState)`, zero I/O. Reads `TuiState` only; never
//! touches `crossterm` directly (`mod.rs` owns the terminal). Testable in isolation via
//! `ratatui::backend::TestBackend` if ever useful, not exercised here (a real interactive check
//! is the disclosed gap - see `docs/tui-mode-plan.md`'s own verification section).

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph},
    Frame,
};

use super::state::{RunStatus, TuiKtSource, TuiState};

pub fn render(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(5), Constraint::Length(2)])
        .split(area);

    render_summary_bar(frame, rows[0], state);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(rows[1]);
    render_loss_chart(frame, cols[0], state);
    render_stats_sidebar(frame, cols[1], state);

    render_status_bar(frame, rows[2], state);
}

fn render_summary_bar(frame: &mut Frame, area: Rect, state: &TuiState) {
    let line = Line::from(vec![
        Span::styled("PINN Structural Stress Solver", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("   step {}/{}", state.step, state.max_steps)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_loss_chart(frame: &mut Frame, area: Rect, state: &TuiState) {
    // ratatui has no native log axis - pre-transform to ln(loss), matching the design doc's
    // own plan (an explicit axis label discloses the transform rather than mislabeling it).
    let points: Vec<(f64, f64)> = state
        .total_loss_history
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, (v.max(1e-12) as f64).ln()))
        .collect();
    let dataset = Dataset::default()
        .name("ln(total_loss)")
        .marker(Marker::Braille)
        .style(Style::default().fg(Color::Cyan))
        .data(&points);

    let (y_min, y_max) = points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, y)| {
        (lo.min(y), hi.max(y))
    });
    let (y_min, y_max) = if y_min.is_finite() && y_max.is_finite() && y_min < y_max {
        (y_min, y_max)
    } else {
        (-1.0, 1.0)
    };
    let x_max = (points.len().max(1) - 1) as f64;

    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title("Loss (ln scale)"))
        .x_axis(Axis::default().bounds([0.0, x_max.max(1.0)]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
            Line::from(format!("{y_min:.1}")),
            Line::from(format!("{y_max:.1}")),
        ]));
    frame.render_widget(chart, area);
}

fn render_stats_sidebar(frame: &mut Frame, area: Rect, state: &TuiState) {
    let mut lines = vec![
        Line::from(format!("step        {} / {}", state.step, state.max_steps)),
        Line::from(format!("total_loss  {:.4e}", state.total_loss)),
        Line::from(format!("energy_loss {:.4e}", state.energy_loss)),
        Line::from(format!("neumann     {:.4e}", state.neumann_loss)),
        Line::from(format!("lr          {:.3e}", state.lr)),
        Line::from(format!("n_colloc    {}", state.n_colloc)),
    ];
    if let Some(g) = state.grad_norm {
        lines.push(Line::from(format!("grad_norm   {g:.4e}")));
    }

    match state.kt_source {
        TuiKtSource::KtEstimate => {
            if let Some(kt) = state.kt_estimate {
                lines.push(Line::from(Span::styled(format!("Kt          {kt:.4}"), Style::default().fg(Color::Yellow))));
            }
        }
        TuiKtSource::HoleAnalyses => {
            for h in &state.hole_analyses {
                lines.push(Line::from(Span::styled(
                    format!("hole {} Kt   {:.4}", h.hole_index, h.concentration.kt),
                    Style::default().fg(Color::Yellow),
                )));
            }
        }
        TuiKtSource::ConvergenceMetric => {
            if let Some(m) = state.convergence_metric {
                lines.push(Line::from(Span::styled(format!("convergence {m:.4e}"), Style::default().fg(Color::Yellow))));
            }
        }
        TuiKtSource::None => {}
    }

    if let Some(sweep) = &state.latest_amr_sweep {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Latest AMR sweep", Style::default().add_modifier(Modifier::BOLD))));
        lines.push(Line::from(format!(
            "{}: {} -> {} pts (step {})",
            sweep.domain_label, sweep.points_before, sweep.points_after, sweep.step
        )));
    }
    if let Some(event) = &state.latest_architecture_event {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Latest architecture event", Style::default().add_modifier(Modifier::BOLD))));
        lines.push(Line::from(format!("step {}: {}", event.step, event.description)));
    }

    frame.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Stats")),
        area,
    );
}

fn render_status_bar(frame: &mut Frame, area: Rect, state: &TuiState) {
    let (label, color) = match state.status {
        None => ("Idle", Color::Gray),
        Some(RunStatus::Running) => ("Running", Color::Green),
        Some(RunStatus::Done) => ("Done", Color::Blue),
        Some(RunStatus::Error) => ("Error", Color::Red),
    };
    let mut spans = vec![Span::styled(label, Style::default().fg(color).add_modifier(Modifier::BOLD))];
    if let Some(msg) = &state.error_msg {
        spans.push(Span::raw(format!("  {msg}")));
    }
    spans.push(Span::raw("   (q / Ctrl+C: stop and quit)"));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
