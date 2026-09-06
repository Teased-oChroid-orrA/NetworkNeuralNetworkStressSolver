use egui::Ui;
use pinn_core::{
    messages::{ProblemKind, SolverConfig},
    problem_spec::ProblemSpec,
    units::{IN_TO_M, KSI_TO_PA, MSI_TO_PA},
    FieldType, SolverStatus, TrainingState,
};

/// Bundles the user-defined-problem UI's mutable state — kept as one parameter rather than
/// 4 more loose ones, since (unlike `problem_kind`) none of this lives in `pinn_core` (see
/// CLAUDE.md's "User-defined problem ingestion" section for why `ProblemKind` itself stays
/// untouched).
pub struct UserProblemUi<'a> {
    pub active: &'a mut bool,
    pub spec_path: &'a mut String,
    pub spec: &'a mut Option<ProblemSpec>,
    pub error: &'a mut Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut Ui,
    config: &mut SolverConfig,
    state: &TrainingState,
    selected_field: &mut FieldType,
    problem_kind: &mut ProblemKind,
    export_status: Option<&str>,
    on_solve: &mut bool,
    on_warm_start: &mut bool,
    on_stop: &mut bool,
    on_export: &mut bool,
    user: &mut UserProblemUi,
) {
    ui.heading("PINN Stress Solver");
    ui.separator();

    // ── Problem kind ─────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label("Problem:");
        if ui.radio_value(problem_kind, ProblemKind::Kirsch, "Kirsch").changed() {
            *config = SolverConfig::default_kirsch();
            *user.active = false;
        }
        if ui.radio_value(problem_kind, ProblemKind::PinLug, "Pin-in-Lug").changed() {
            *config = SolverConfig::default_pinlug();
            *user.active = false;
        }
        if ui.radio(*user.active, "User-Defined").clicked() {
            *user.active = true;
        }
    });

    ui.separator();

    if *user.active {
        egui::CollapsingHeader::new("User-Defined Problem").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Spec path:");
                ui.text_edit_singleline(user.spec_path);
            });
            if ui.button("Load").clicked() {
                match std::fs::read_to_string(user.spec_path.as_str()) {
                    Ok(contents) => match toml::from_str::<ProblemSpec>(&contents) {
                        Ok(spec) => {
                            *user.spec = Some(spec);
                            *user.error = None;
                        }
                        Err(e) => *user.error = Some(format!("parse error: {e}")),
                    },
                    Err(e) => *user.error = Some(format!("read error: {e}")),
                }
            }
            if let Some(err) = user.error.as_deref() {
                ui.colored_label(egui::Color32::from_rgb(248, 113, 113), err);
            }
            if let Some(spec) = user.spec.as_ref() {
                ui.separator();
                ui.label(format!(
                    "Plate: {:.4}×{:.4} m", 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h
                ));
                ui.label(format!("Holes: {}", spec.geometry.holes.len()));
                for (i, h) in spec.geometry.holes.iter().enumerate() {
                    ui.label(format!(
                        "  #{i}: r={:.4} m @ ({:.3}, {:.3})  [{:?}]",
                        h.radius, h.center[0], h.center[1], h.bc
                    ));
                }
                ui.label(format!("Material E = {:.3e} Pa,  ν = {:.3}", spec.material.e, spec.material.nu));
                ui.label(format!("Load Px = {:.3e} Pa,  Py = {:.3e} Pa", spec.load.px, spec.load.py));
                ui.label(format!("Steps: {}   Interior: {}", spec.training.max_steps, spec.training.n_interior));
            }
        });

        ui.separator();
    } else {
    match problem_kind {
        ProblemKind::Kirsch => {
            // ── Material ───────────────────────────────────────
            egui::CollapsingHeader::new("Material").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("E [Msi]:");
                    let mut e_msi = config.material.e / MSI_TO_PA;
                    if ui.add(egui::DragValue::new(&mut e_msi).speed(0.1).range(0.1..=75.0)).changed() {
                        config.material.e = e_msi * MSI_TO_PA;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("ν:");
                    ui.add(egui::DragValue::new(&mut config.material.nu).speed(0.001).range(0.0..=0.499));
                });
            });

            ui.separator();

            // ── Geometry ───────────────────────────────────────
            egui::CollapsingHeader::new("Geometry").default_open(true).show(ui, |ui| {
                use pinn_core::geometry::HoleType;

                ui.horizontal(|ui| {
                    ui.label("Half-W [in]:");
                    let mut hw_in = config.geometry.half_w / IN_TO_M;
                    if ui.add(egui::DragValue::new(&mut hw_in).speed(0.1).range(0.5..=50.0)).changed() {
                        config.geometry.half_w = hw_in * IN_TO_M;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Half-H [in]:");
                    let mut hh_in = config.geometry.half_h / IN_TO_M;
                    if ui.add(egui::DragValue::new(&mut hh_in).speed(0.1).range(0.5..=50.0)).changed() {
                        config.geometry.half_h = hh_in * IN_TO_M;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Thickness [in]:");
                    let mut t_in = config.geometry.thickness / IN_TO_M;
                    if ui.add(egui::DragValue::new(&mut t_in).speed(0.01).range(0.01..=5.0)).changed() {
                        config.geometry.thickness = t_in * IN_TO_M;
                    }
                });

                if let HoleType::Circular { ref mut radius } = config.geometry.hole {
                    ui.horizontal(|ui| {
                        ui.label("Hole r [in]:");
                        let mut r_in = *radius / IN_TO_M;
                        if ui.add(egui::DragValue::new(&mut r_in).speed(0.005).range(0.01..=2.0)).changed() {
                            *radius = r_in * IN_TO_M;
                        }
                    });
                }
            });

            ui.separator();

            // ── Loads ────────────────────────────────────────
            egui::CollapsingHeader::new("Loads").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Px [ksi]:");
                    let mut px_ksi = config.load.px / KSI_TO_PA;
                    if ui.add(egui::Slider::new(&mut px_ksi, 0.1..=75.0)).changed() {
                        config.load.px = px_ksi * KSI_TO_PA;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Py [ksi]:");
                    let mut py_ksi = config.load.py / KSI_TO_PA;
                    if ui.add(egui::Slider::new(&mut py_ksi, 0.0..=75.0)).changed() {
                        config.load.py = py_ksi * KSI_TO_PA;
                    }
                });
            });
        }
        ProblemKind::PinLug => {
            // Fixed pin-lug configuration — read-only summary pulled straight from
            // `config` (which was set to `SolverConfig::default_pinlug()` when the
            // problem-kind radio flipped). No DragValue/Slider — this is a documented,
            // fixed setup for this slice (no full two-domain resample support yet).
            egui::CollapsingHeader::new("Material (fixed: 4340 steel)").default_open(true).show(ui, |ui| {
                ui.label(format!("E = {:.1} Msi", config.material.e / MSI_TO_PA));
                ui.label(format!("ν = {:.3}", config.material.nu));
                ui.label(format!(
                    "Ultimate strength = {:.0} ksi",
                    config.material.ultimate_strength_pa / KSI_TO_PA
                ));
            });

            ui.separator();

            egui::CollapsingHeader::new("Geometry (fixed)").default_open(true).show(ui, |ui| {
                ui.label(format!("Lug half-W = {:.3} in", config.geometry.half_w / IN_TO_M));
                ui.label(format!("Lug half-H = {:.3} in", config.geometry.half_h / IN_TO_M));
                ui.label(format!("Thickness = {:.3} in", config.geometry.thickness / IN_TO_M));
                if let pinn_core::geometry::HoleType::Circular { radius } = config.geometry.hole {
                    ui.label(format!("Pin/hole radius = {:.3} in", radius / IN_TO_M));
                }
            });

            ui.separator();

            egui::CollapsingHeader::new("Loads (fixed: 20,000 lbf)").default_open(true).show(ui, |ui| {
                ui.label("Total axial force = 20,000 lbf");
                ui.label(format!(
                    "Equivalent traction = {:.2} ksi (= F / (2·r·t))",
                    config.load.px / KSI_TO_PA
                ));
            });
        }
    }
    } // end `if *user.active { ... } else { <existing match> }`

    ui.separator();

    // ── Training settings ──────────────────────────────────
    egui::CollapsingHeader::new("Solver Settings").default_open(false).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Max steps:");
            ui.add(egui::DragValue::new(&mut config.max_steps).speed(100).range(100..=100_000));
        });
        ui.horizontal(|ui| {
            ui.label("Interior pts:");
            ui.add(egui::DragValue::new(&mut config.n_interior).speed(10).range(64..=8192));
        });
        ui.horizontal(|ui| {
            ui.label("Hidden dim:");
            ui.add(egui::DragValue::new(&mut config.hidden_dim).speed(1).range(8..=256));
        });
    });

    ui.separator();

    // ── Status ─────────────────────────────────────────────
    let status_text = match state.status {
        SolverStatus::Idle      => "Idle",
        SolverStatus::Running   => "Running...",
        SolverStatus::Paused    => "Paused",
        SolverStatus::Converged => "Converged",
        SolverStatus::Error     => "Error",
    };
    ui.label(format!("Status: {status_text}"));

    if state.step > 0 {
        ui.label(format!("Step: {}", state.step));
        if let Some(last) = state.total_loss.last() {
            ui.label(format!("Loss: {:.3e}", last));
        }
        if let Some(last) = state.lr_history.last() {
            ui.label(format!("LR: {:.2e}", last));
        }
        ui.label(format!("Colloc pts: {}", state.n_colloc));
        // No closed-form convergence metric exists for an arbitrary user-defined geometry
        // (unlike Kirsch's K_t or pin-lug's interface-gap RMS) — nothing to show here.
        if !*user.active {
            match problem_kind {
                ProblemKind::Kirsch => {
                    if let Some(kt) = state.kt_estimate {
                        ui.label(format!("K_t = {:.3}  (theory: 3.000)", kt));
                    }
                }
                ProblemKind::PinLug => {
                    if let Some(metric) = state.convergence_metric {
                        ui.label(format!("Interface gap RMS = {:.3e} m (target: 0)", metric));
                    }
                }
            }
        }
    }

    if let Some(msg) = export_status {
        ui.label(msg);
    }

    ui.separator();

    // ── Field selector ─────────────────────────────────────
    ui.label("Display field:");
    for &ft in &[
        FieldType::VonMises, FieldType::SigmaXX, FieldType::SigmaYY,
        FieldType::SigmaXY, FieldType::DispU, FieldType::DispV,
    ] {
        ui.radio_value(selected_field, ft, ft.label());
    }

    ui.separator();

    // ── Buttons ────────────────────────────────────────────
    ui.horizontal(|ui| {
        let running = state.status == SolverStatus::Running;
        let can_solve = !running && (!*user.active || user.spec.is_some());
        if ui.add_enabled(can_solve, egui::Button::new("▶ Solve")).clicked() {
            *on_solve = true;
        }
        if ui.add_enabled(running, egui::Button::new("⏹ Stop")).clicked() {
            *on_stop = true;
        }
    });

    // Pin-lug's `run_training_pinlug` only honors scalar config fields on `WarmStart`
    // (no full two-domain resample in this slice) — disable the warm-start trigger
    // entirely for PinLug rather than silently pretending full warm-start works.
    // `run_training_user_problem` accepts but ignores `WarmStart` entirely (a loaded
    // `ProblemSpec` isn't a `SolverConfig` there's anything scalar to warm-start into) —
    // same disabled treatment.
    let can_warm = !*user.active
        && *problem_kind == ProblemKind::Kirsch
        && state.step > 0
        && state.status != SolverStatus::Running;
    if ui.add_enabled(can_warm, egui::Button::new("⚡ Warm-Start")).clicked() {
        *on_warm_start = true;
    }

    if !*user.active && *problem_kind == ProblemKind::PinLug {
        let can_export = state.step > 0;
        if ui.add_enabled(can_export, egui::Button::new("Export Contact Pressure CSV")).clicked() {
            *on_export = true;
        }
    }
}
