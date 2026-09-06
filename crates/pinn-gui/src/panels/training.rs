use egui::{Color32, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use pinn_core::TrainingState;

// Shared accent palette — kept in one place so the stat cards and the curves below them
// agree on which color means what (total/energy/boundary/lr), matching the design
// reference this panel was redrawn from.
const ACCENT_TOTAL:  Color32 = Color32::from_rgb(94, 230, 200);
const ACCENT_ENERGY: Color32 = Color32::from_rgb(88, 166, 255);
const ACCENT_BC:      Color32 = Color32::from_rgb(240, 169, 78);
const ACCENT_LR:      Color32 = Color32::from_rgb(199, 146, 234);
const TEXT_DIM:       Color32 = Color32::from_rgb(125, 139, 154);
const CARD_BG:        Color32 = Color32::from_rgb(16, 22, 31);

/// One small stat card: a label, a big value, and a colored left accent bar — the
/// "instrument panel" look from the design reference. `value` is pre-formatted by the
/// caller (each stat has its own natural precision/units).
fn stat_card(ui: &mut Ui, label: &str, value: &str, accent: Color32) {
    let resp = egui::Frame::none()
        .fill(CARD_BG)
        .rounding(6.0)
        .inner_margin(egui::Margin::symmetric(10.0, 8.0))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(label).size(10.5).color(TEXT_DIM));
                ui.label(egui::RichText::new(value).size(17.0).strong().monospace());
            });
        });
    let rect = resp.response.rect;
    let bar = egui::Rect::from_min_max(rect.min, egui::pos2(rect.min.x + 3.0, rect.max.y));
    ui.painter().rect_filled(bar, 3.0, accent);
}

pub fn show(ui: &mut Ui, state: &TrainingState) {
    ui.heading("Training Telemetry");
    ui.separator();

    let n = state.total_loss.len();

    if n == 0 {
        ui.label("Training not started yet.");
        ui.label("Click ▶ Solve to begin.");
        return;
    }

    // ── Stat cards ─────────────────────────────────────────
    let total = state.total_loss.last().copied().unwrap_or(0.0);
    let energy = state.energy_loss.last().copied().unwrap_or(0.0);
    let boundary = state.neumann_loss.last().copied().unwrap_or(0.0);
    let lr = state.lr_history.last().copied().unwrap_or(0.0);

    ui.columns(4, |cols| {
        stat_card(&mut cols[0], "TOTAL LOSS", &format!("{total:.3e}"), ACCENT_TOTAL);
        stat_card(&mut cols[1], "ENERGY TERM", &format!("{energy:.3e}"), ACCENT_ENERGY);
        stat_card(&mut cols[2], "BOUNDARY TERM", &format!("{boundary:.3e}"), ACCENT_BC);
        stat_card(&mut cols[3], "LEARNING RATE", &format!("{lr:.3e}"), ACCENT_LR);
    });

    ui.add_space(6.0);
    ui.separator();

    // ── Loss curves ────────────────────────────────────────
    let height = (ui.available_height() / 3.0 - 20.0).max(80.0);

    ui.label(egui::RichText::new("Loss Trajectory (log scale)").color(TEXT_DIM));
    Plot::new("total_loss")
        .height(height)
        .y_axis_label("log10(loss)")
        .x_axis_label("step")
        .show(ui, |pui| {
            let pts: PlotPoints = state.total_loss.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                .collect();
            pui.line(Line::new(pts).name("Total").width(2.0).color(ACCENT_TOTAL));

            let pts_e: PlotPoints = state.energy_loss.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                .collect();
            pui.line(Line::new(pts_e).name("Energy").width(1.3).color(ACCENT_ENERGY));

            let pts_n: PlotPoints = state.neumann_loss.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                .collect();
            pui.line(Line::new(pts_n).name("Boundary").width(1.3).color(ACCENT_BC));
        });

    ui.separator();

    // ── Learning rate ──────────────────────────────────
    ui.label(egui::RichText::new("Learning Rate").color(TEXT_DIM));
    Plot::new("lr_plot")
        .height(height)
        .y_axis_label("lr")
        .x_axis_label("step")
        .show(ui, |pui| {
            let pts: PlotPoints = state.lr_history.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                .collect();
            pui.line(Line::new(pts).name("LR").width(2.0).color(ACCENT_LR));
        });

    ui.separator();

    // ── SAW-BRDR weights ───────────────────────────────
    ui.label(egui::RichText::new("Adaptive Loss Weights (λ_energy, λ_neumann)").color(TEXT_DIM));
    Plot::new("weights_plot")
        .height(height)
        .y_axis_label("weight")
        .x_axis_label("step")
        .show(ui, |pui| {
            let pts_e: PlotPoints = state.lam_energy.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v as f64])
                .collect();
            pui.line(Line::new(pts_e).name("λ_energy").width(1.5).color(ACCENT_ENERGY));

            let pts_n: PlotPoints = state.lam_neumann.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64 * 10.0, v as f64])
                .collect();
            pui.line(Line::new(pts_n).name("λ_neumann").width(1.5).color(ACCENT_BC));
        });
}
