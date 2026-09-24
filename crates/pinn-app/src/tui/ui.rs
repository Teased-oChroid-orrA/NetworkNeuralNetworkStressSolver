//! Rendering only: all physical metrics and objective terms come from solver telemetry.
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style}, symbols::Marker, text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph, Tabs, Wrap}, Frame,
};
use super::state::{RunStatus, TuiState};
pub const TABS: [&str; 6] = ["Overview", "Loss / gradients", "Physics / holes", "Sampling", "Run / backend", "Logs"];

pub fn render(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();
    if area.width < 65 || area.height < 18 {
        frame.render_widget(Paragraph::new(format!(
            "PINN {:?} step {}/{}\nloss {:.4e}  grad {:?}\nSmall terminal: resize for panels.\nq cancel/quit | p pause | r resume\n{}",
            state.status, state.step, state.max_steps, state.total_loss, state.grad_norm, state.control_status
        )).wrap(Wrap { trim: false }), area);
        return;
    }
    let rows = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(3)]).split(area);
    frame.render_widget(Tabs::new(TABS).select(state.tab)
        .highlight_style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL).title(format!("PINN engineering console | step {}/{}", state.step, state.max_steps))), rows[0]);
    match state.tab {
        0 => {
            let cols = Layout::default().direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)]).split(rows[1]);
            loss_chart(frame, cols[0], state);
            panel(frame, cols[1], "Status", vec![
                format!("Run: {:?}", state.status),
                format!("Total objective: {:.6e}", state.total_loss),
                format!("Energy channel*: {:.6e}", state.energy_loss),
                format!("Other/BC channel*: {:.6e}", state.neumann_loss),
                "*Legacy channels are path-dependent; see Loss tab.".into(),
                format!("LR: {:.4e}", state.lr), format!("Gradient norm: {:?}", state.grad_norm),
                format!("Collocation points: {}", state.n_colloc),
                "Negative potential energy can be valid.".into(),
                "Run completion does not establish physical accuracy.".into(),
                state.control_status.clone(), state.error_msg.clone().unwrap_or_default(),
            ], state.scroll);
        }
        1 => {
            let mut lines = vec!["Normalized unweighted terms × effective weights = contributions".into()];
            if let Some(objective) = &state.objective {
                lines.push(format!("Optimizer tier: {} (0 Explore, 1 Align, 2 Converge)", objective.optimizer_tier));
                for t in &objective.terms { lines.push(format!("{}: {:.5e} × {:.5e} = {:.5e}", t.name, t.normalized_raw, t.effective_weight, t.weighted)); }
                lines.push(format!("Total: {:.6e}; reconstruction difference: {:.3e}", objective.total, objective.reconstruction_error));
                lines.push(format!("Gradient L2 before optimizer: {:?}", objective.grad_norm_before_optimizer));
            } else { lines.push("Term decomposition unavailable on this training path.".into()); }
            lines.push("Clipping / post-clipping norm: not reported by this path.".into());
            if let Some(shares) = &state.gradient_shares {
                lines.push("Latest measured shares: norm_i / sum(norm_j), not additive total-gradient fractions".into());
                for (name, share) in &shares.shares { lines.push(format!("{name}: {:.2}%", share * 100.0)); }
                lines.push(format!("Dominant: {:?}; inert: {:?}", shares.dominant, shares.inert));
            } else { lines.push("Per-term gradients not measured yet.".into()); }
            panel(frame, rows[1], "Objective / gradients", lines, state.scroll);
        }
        2 => {
            let mut lines = Vec::new();
            if let Some(e) = state.energy_balance {
                lines.push(format!("U = {:.6e} J; full W_ext = {:.6e} J; Pi = {:.6e} J", e.internal_energy, e.load_potential_work(), e.physical_potential()));
                lines.push(format!("Linear-elastic balance error |U-W_ext/2| / |W_ext/2| = {:.4e}", e.energy_balance_error));
            } else { lines.push("Physical energy not measured on this path / not available yet.".into()); }
            if let Some(f) = state.reaction_force { lines.push(format!("Net force [N]: ({:.4e}, {:.4e}); normalized imbalance {:.4e}", f.net_fx, f.net_fy, f.equilibrium_error)); }
            if let Some((step, rms, max)) = state.boundary_residual { lines.push(format!("Boundary residual at step {step}: RMS {rms:.4e}; max {max:.4e} (solver units)")); }
            if let Some(kt) = state.kt_estimate { lines.push(format!("Kt estimate: {kt:.5}; reference acceptance not inferred")); }
            for h in &state.hole_analyses { lines.push(format!("Hole {}: Kt {:.5}; nominal {:.4e} Pa; {} / {}", h.hole_index, h.concentration.kt, h.concentration.nominal_stress, h.concentration.stress_projection, h.concentration.domain_classification)); }
            if let Some(e) = &state.convergence { lines.push(format!("Convergence evidence: {e:?}")); }
            else { lines.push("Convergence: not evaluated yet; no accuracy claim.".into()); }
            for (name, source) in &state.sources { lines.push(format!("{name}: stress source {source}")); }
            panel(frame, rows[1], "Latest solver physics probes", lines, state.scroll);
        }
        3 => {
            let mut lines = vec![format!("Current collocation count: {}", state.n_colloc)];
            if let Some(s) = &state.latest_amr_sweep {
                lines.extend([format!("{} step {}: {} -> {} points", s.domain_label, s.step, s.points_before, s.points_after),
                    format!("Residual RMS before {:.5e}, after {:.5e}", s.residual_rms_before, s.residual_rms_after),
                    format!("Sweep {:.1} ms; hole density {:.4e} -> {:.4e}", s.sweep_duration_ms, s.hole_zone_density_before, s.hole_zone_density_after)]);
            } else { lines.push("No AMR sweep reported.".into()); }
            if let Some(e) = &state.latest_architecture_event { lines.push(format!("Architecture step {}: {}", e.step, e.description)); }
            lines.push("Sampling counts by boundary and RNG state: not reported by this channel.".into());
            panel(frame, rows[1], "Sampling / AMR", lines, state.scroll);
        }
        4 => panel(frame, rows[1], "Run configuration / provenance", state.run_summary.lines().map(str::to_owned).collect(), state.scroll),
        _ => panel(frame, rows[1], if state.warnings_only { "Warnings/errors (f: all)" } else { "Events (f: warnings/errors)" },
            state.logs.iter().filter(|s| !state.warnings_only || s.contains("ERROR") || s.contains("WARN")).cloned().collect(), state.scroll),
    }
    let label = match state.status { Some(RunStatus::Error) => "ERROR", Some(RunStatus::Done) => "Completed (accuracy separate)", _ => "Running" };
    frame.render_widget(Paragraph::new(format!("{label} | Tab/1–6 panels | ↑↓ scroll | f log filter | p pause request | r resume request | q stop/quit\n{}", state.control_status))
        .block(Block::default().borders(Borders::ALL)), rows[2]);
}
fn panel(frame: &mut Frame, area: Rect, title: &str, lines: Vec<String>, scroll: u16) {
    frame.render_widget(Paragraph::new(lines.join("\n")).wrap(Wrap { trim: false }).scroll((scroll, 0))
        .block(Block::default().borders(Borders::ALL).title(title)), area);
}
fn signed_log(value: f32) -> f64 { let v = value as f64; v.signum() * v.abs().ln_1p() }
fn loss_chart(frame: &mut Frame, area: Rect, state: &TuiState) {
    let points: Vec<_> = state.history_steps.iter().zip(&state.total_loss_history)
        .filter(|(_, v)| v.is_finite()).map(|(&step, &v)| (step as f64, signed_log(v))).collect();
    let (mut lo, mut hi) = points.iter().fold((0.0_f64, 0.0_f64), |(lo, hi), &(_, v)| (lo.min(v), hi.max(v)));
    if lo == hi { lo -= 1.0; hi += 1.0; }
    let x0 = points.first().map_or(0.0, |p| p.0);
    let x1 = points.last().map_or(x0 + 1.0, |p| p.0).max(x0 + 1.0);
    frame.render_widget(Chart::new(vec![Dataset::default().name("sign(loss) × ln(1+|loss|)")
        .marker(Marker::Braille).style(Style::default().fg(Color::Cyan)).data(&points)])
        .block(Block::default().borders(Borders::ALL).title("Signed objective history (actual steps)"))
        .x_axis(Axis::default().bounds([x0, x1]).labels([Line::from(format!("{x0:.0}")), Line::from(format!("{x1:.0}"))]))
        .y_axis(Axis::default().bounds([lo, hi]).labels([Line::from(format!("{lo:.2}")), Line::from(format!("{hi:.2}"))])), area);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_transform_retains_negative_values() {
        assert!(signed_log(-10.0) < 0.0);
        assert_eq!(signed_log(-10.0), -signed_log(10.0));
        assert_eq!(signed_log(0.0), 0.0);
    }
    #[test]
    fn every_panel_renders_at_small_and_large_sizes() {
        for (w,h) in [(1,1), (40,8), (80,24), (140,45)] {
            let backend = ratatui::backend::TestBackend::new(w,h);
            let mut terminal = ratatui::Terminal::new(backend).unwrap();
            let mut state = TuiState::new(100);
            state.total_loss_history = vec![-4.0, f32::NAN, 2.0];
            state.history_steps = vec![0,10,20];
            for tab in 0..TABS.len() { state.tab = tab; terminal.draw(|f| render(f, &state)).unwrap(); }
        }
    }
}
