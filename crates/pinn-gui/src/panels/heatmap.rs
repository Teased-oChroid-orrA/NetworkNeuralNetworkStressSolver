use egui::{ColorImage, TextureHandle, TextureOptions, Ui};
use pinn_core::{messages::SolverConfig, units::{IN_TO_M, KSI_TO_PA}, FieldType, TrainingState};

use crate::colormap::field_to_pixels;

pub fn show(
    ui: &mut Ui,
    state: &TrainingState,
    config: &SolverConfig,
    selected_field: FieldType,
    texture: &mut Option<TextureHandle>,
    colorbar_range: &mut (f32, f32),
) {
    ui.heading("Stress Field");
    ui.separator();

    let field = state.field(selected_field);
    let [nx, ny] = state.vis_grid;

    // Flip rows so physical y=0 (hole edge) appears at screen bottom, not top.
    // Array storage: row 0 = y_min (bottom of plate) which egui renders at top.
    let (pixels, vmin, vmax) = field_to_pixels(field, true);
    *colorbar_range = (vmin, vmax);

    let img = ColorImage {
        size: [nx, ny],
        pixels,
    };

    let handle = ui.ctx().load_texture(
        format!("stress_{:?}", selected_field),
        img,
        TextureOptions::LINEAR,
    );
    *texture = Some(handle);

    // Render heatmap filling available width; capture rect for overlays.
    let img_rect = if let Some(ref tex) = texture {
        let avail = ui.available_size();
        let aspect = nx as f32 / ny as f32;
        let w = avail.x.min(avail.y * aspect);
        let h = w / aspect;
        let resp = ui.image((tex.id(), egui::vec2(w, h)));
        Some(resp.rect)
    } else {
        None
    };

    // Draw hole boundary and domain grid overlay.
    if let Some(rect) = img_rect {
        draw_overlays(ui, rect, config, [nx, ny]);
    }

    ui.separator();

    // ── Colorbar ───────────────────────────────────────────
    // Stress fields are stored in Pa, displacement fields in meters — display each in
    // its own US Customary unit (ksi / inches) rather than raw SI.
    let (vmin, vmax) = *colorbar_range;
    let (vmin_disp, vmax_disp, unit) = if selected_field.is_displacement() {
        (vmin / IN_TO_M as f32, vmax / IN_TO_M as f32, "in")
    } else {
        (vmin / KSI_TO_PA as f32, vmax / KSI_TO_PA as f32, "ksi")
    };
    ui.label(format!(
        "{} — [{:.3} … {:.3}] {unit}",
        selected_field.label(),
        vmin_disp,
        vmax_disp,
    ));

    // Simple colour strip (10 samples)
    ui.horizontal(|ui| {
        for i in 0..10 {
            let t = i as f32 / 9.0;
            let [r, g, b, _] = crate::colormap::viridis(t).to_array();
            let rect = egui::Rect::from_min_size(
                ui.cursor().min,
                egui::vec2(14.0, 14.0),
            );
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(r, g, b));
            ui.add_space(14.0);
        }
    });
    ui.horizontal(|ui| {
        ui.label(format!("{:.3}", vmin_disp));
        ui.add_space(90.0);
        ui.label(format!("{:.3}", vmax_disp));
    });
}

/// Draw hole boundary circle and a faint measurement grid over the heatmap.
fn draw_overlays(
    ui: &Ui,
    rect: egui::Rect,
    config: &SolverConfig,
    [nx, ny]: [usize; 2],
) {
    use pinn_core::geometry::HoleType;

    let painter = ui.painter_at(rect);
    let (x0, x1) = config.geometry.x_range();
    let (y0, y1) = config.geometry.y_range();
    let dom_w = (x1 - x0) as f32;
    let dom_h = (y1 - y0) as f32;

    // Convert physical (x,y) → screen pos inside rect.
    // After y-flip: y_phys=y0 (bottom of plate) → screen bottom (rect.max.y).
    let to_screen = |x_phys: f32, y_phys: f32| -> egui::Pos2 {
        let sx = rect.min.x + (x_phys - x0 as f32) / dom_w * rect.width();
        let sy = rect.max.y - (y_phys - y0 as f32) / dom_h * rect.height();
        egui::pos2(sx, sy)
    };

    // ── Grid lines every ~1 inch (or auto-spaced to ~5 lines) ─────────────
    let domain_inch_w = dom_w / IN_TO_M as f32;
    let domain_inch_h = dom_h / IN_TO_M as f32;
    let grid_step_inch = {
        let target_lines = 5;
        let raw = domain_inch_w.max(domain_inch_h) / target_lines as f32;
        [0.25_f32, 0.5, 1.0, 2.0, 5.0].into_iter()
            .find(|&s| s >= raw).unwrap_or(raw)
    };
    let grid_step_m = grid_step_inch * IN_TO_M as f32;
    let grid_col = egui::Color32::from_rgba_premultiplied(200, 200, 200, 40);

    // Vertical grid lines
    let n_vlines = ((x1 as f32 - x0 as f32) / grid_step_m).floor() as i32;
    for i in 0..=n_vlines {
        let gx = x0 as f32 + i as f32 * grid_step_m;
        let top    = to_screen(gx, y1 as f32);
        let bottom = to_screen(gx, y0 as f32);
        painter.line_segment([top, bottom], egui::Stroke::new(0.5, grid_col));
    }
    // Horizontal grid lines
    let n_hlines = ((y1 as f32 - y0 as f32) / grid_step_m).floor() as i32;
    for i in 0..=n_hlines {
        let gy = y0 as f32 + i as f32 * grid_step_m;
        let left  = to_screen(x0 as f32, gy);
        let right = to_screen(x1 as f32, gy);
        painter.line_segment([left, right], egui::Stroke::new(0.5, grid_col));
    }

    // ── Domain boundary (white border) ───────────────────────────────────
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.5, egui::Color32::WHITE));

    // ── Hole boundary circle arc ──────────────────────────────────────────
    if let HoleType::Circular { radius } = config.geometry.hole {
        let r = radius as f32;
        let center = to_screen(x0 as f32, y0 as f32); // hole at domain origin (0,0)
        let r_px_x = r / dom_w * rect.width();
        let r_px_y = r / dom_h * rect.height();
        // Use average radius in screen pixels (square pixels assumed)
        let r_px = (r_px_x + r_px_y) * 0.5;

        // Draw arc from 0° to 90° (QuarterSymm corner arc) or full circle
        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 60));
        let n_seg = 32;
        let angle_max = match config.geometry.symmetry {
            pinn_core::geometry::SymmetryMode::QuarterSymm => std::f32::consts::FRAC_PI_2,
            _ => 2.0 * std::f32::consts::PI,
        };
        for i in 0..n_seg {
            let t0 = i as f32 / n_seg as f32 * angle_max;
            let t1 = (i + 1) as f32 / n_seg as f32 * angle_max;
            // Angle 0 = +x axis, increases counterclockwise in physics.
            // In screen coords y is flipped so angles map correctly after to_screen().
            let p0 = egui::pos2(center.x + r_px * t0.cos(), center.y - r_px * t0.sin());
            let p1 = egui::pos2(center.x + r_px * t1.cos(), center.y - r_px * t1.sin());
            painter.line_segment([p0, p1], stroke);
        }

        // Label: "r = X.XX in" next to arc
        let label_pos = to_screen(r * 1.15, r * 0.15);
        painter.text(
            label_pos,
            egui::Align2::LEFT_CENTER,
            format!("r={:.3}in", r / IN_TO_M as f32),
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(255, 220, 60),
        );
    }

    // ── Axis labels at corners ─────────────────────────────────────────────
    let label_col = egui::Color32::from_rgba_premultiplied(255, 255, 255, 180);
    let font = egui::FontId::proportional(10.0);
    let origin = to_screen(x0 as f32, y0 as f32);
    painter.text(origin + egui::vec2(3.0, -3.0), egui::Align2::LEFT_BOTTOM,
        format!("({:.1},{:.1})in", x0 / IN_TO_M, y0 / IN_TO_M), font.clone(), label_col);

    let _ = nx; let _ = ny; // used indirectly via grid spacing
}
