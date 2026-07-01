use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints};
use pinn_core::TrainingState;

pub fn show(ui: &mut Ui, state: &TrainingState) {
    ui.heading("Training Curves");
    ui.separator();

    // ── Loss curves ────────────────────────────────────────
    let n = state.total_loss.len();
    let height = (ui.available_height() / 3.0 - 20.0).max(80.0);

    if n > 1 {
        ui.label("Total Loss (log scale)");
        Plot::new("total_loss")
            .height(height)
            .y_axis_label("loss")
            .x_axis_label("step")
            .show(ui, |pui| {
                let pts: PlotPoints = state.total_loss.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                    .collect();
                pui.line(Line::new(pts).name("Total").width(1.5));

                let pts_e: PlotPoints = state.energy_loss.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                    .collect();
                pui.line(Line::new(pts_e).name("Energy").width(1.0)
                    .color(egui::Color32::from_rgb(255, 100, 100)));

                let pts_n: PlotPoints = state.neumann_loss.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                    .collect();
                pui.line(Line::new(pts_n).name("Neumann").width(1.0)
                    .color(egui::Color32::from_rgb(100, 200, 255)));
            });

        ui.separator();

        // ── Learning rate ──────────────────────────────────
        ui.label("Learning Rate");
        Plot::new("lr_plot")
            .height(height)
            .y_axis_label("lr")
            .x_axis_label("step")
            .show(ui, |pui| {
                let pts: PlotPoints = state.lr_history.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v.max(1e-12).log10() as f64])
                    .collect();
                pui.line(Line::new(pts).name("LR").width(1.5)
                    .color(egui::Color32::from_rgb(255, 200, 80)));
            });

        ui.separator();

        // ── SAW-BRDR weights ───────────────────────────────
        ui.label("Adaptive Loss Weights (λ_energy, λ_neumann)");
        Plot::new("weights_plot")
            .height(height)
            .y_axis_label("weight")
            .x_axis_label("step")
            .show(ui, |pui| {
                let pts_e: PlotPoints = state.lam_energy.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v as f64])
                    .collect();
                pui.line(Line::new(pts_e).name("λ_energy").width(1.5)
                    .color(egui::Color32::from_rgb(200, 100, 255)));

                let pts_n: PlotPoints = state.lam_neumann.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64 * 10.0, v as f64])
                    .collect();
                pui.line(Line::new(pts_n).name("λ_neumann").width(1.5)
                    .color(egui::Color32::from_rgb(100, 255, 180)));
            });
    } else {
        ui.label("Training not started yet.");
        ui.label("Click ▶ Solve to begin.");
    }
}
