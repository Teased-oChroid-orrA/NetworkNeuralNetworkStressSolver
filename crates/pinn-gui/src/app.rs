use std::{
    sync::{Arc, Mutex},
    thread,
};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use egui::TextureHandle;
use pinn_core::{
    messages::{ControlMsg, ProblemKind, SolverConfig, TrainingMsg},
    problem_spec::ProblemSpec,
    FieldType, SolverStatus, TrainingState,
};

use crate::panels;

pub struct StressSolverApp {
    config: SolverConfig,
    state:  Arc<Mutex<TrainingState>>,
    problem_kind: ProblemKind,

    // User-defined-problem state — deliberately NOT folded into `problem_kind`
    // (`pinn_core::messages::ProblemKind` stays untouched; see CLAUDE.md's "User-defined
    // problem ingestion" section for why). `user_defined_active` is the 3rd radio option's
    // selection flag; when true, it overrides `problem_kind`-driven dispatch everywhere
    // below (`start_solver`, the heatmap's hole overlay) without changing any of the
    // existing Kirsch/pin-lug code paths.
    user_defined_active: bool,
    user_spec_path: String,
    user_spec: Option<ProblemSpec>,
    user_spec_error: Option<String>,

    // Channels
    tx_control: Option<Sender<ControlMsg>>,
    rx_training: Option<Receiver<TrainingMsg>>,

    // UI state
    selected_field: FieldType,
    texture: Option<TextureHandle>,
    colorbar_range: (f32, f32),
    prev_geo_hash: u64,
    /// Informational status text from the last completed export (path or error) — purely
    /// for display, never affects `state.status`.
    export_status: Option<String>,
}

impl StressSolverApp {
    /// `problem_kind` seeds the initial problem-kind radio selection (e.g. from
    /// `--problem pinlug` on the CLI) — the user can still switch it at runtime via the
    /// GUI's own selector regardless of what's passed here.
    pub fn new(_cc: &eframe::CreationContext<'_>, config: SolverConfig, problem_kind: ProblemKind) -> Self {
        Self::from_config_with_problem_kind(config, problem_kind)
    }

    /// Defaults to `ProblemKind::Kirsch` — see [`Self::from_config_with_problem_kind`] to
    /// seed a different starting selection.
    pub fn from_config(config: SolverConfig) -> Self {
        Self::from_config_with_problem_kind(config, ProblemKind::Kirsch)
    }

    pub fn from_config_with_problem_kind(config: SolverConfig, problem_kind: ProblemKind) -> Self {
        let [nx, ny] = config.vis_grid;
        let state = Arc::new(Mutex::new(TrainingState::new([nx, ny])));
        let geo_hash = config.geometry.geometry_hash();
        Self {
            config,
            state,
            problem_kind,
            user_defined_active: false,
            user_spec_path: String::new(),
            user_spec: None,
            user_spec_error: None,
            tx_control:    None,
            rx_training:   None,
            selected_field: FieldType::VonMises,
            texture:        None,
            colorbar_range: (0.0, 1.0),
            prev_geo_hash:  geo_hash,
            export_status:  None,
        }
    }

    fn start_solver(&mut self) {
        // Stop any existing solver
        self.stop_solver();

        let config = self.config.clone();
        let state  = Arc::clone(&self.state);
        {
            let mut s = state.lock().expect("state mutex poisoned");
            *s = TrainingState::new(config.vis_grid);
            s.status = SolverStatus::Running;
        }

        let (tx_train, rx_train) = bounded(1); // bounded "latest value" channel
        let (tx_ctrl, rx_ctrl)   = crossbeam_channel::unbounded();

        self.rx_training = Some(rx_train);
        self.tx_control  = Some(tx_ctrl);
        self.prev_geo_hash = config.geometry.geometry_hash();

        let problem_kind = self.problem_kind;
        if self.user_defined_active {
            let spec = match self.user_spec.clone() {
                Some(s) => s,
                None => {
                    let mut s = state.lock().expect("state mutex poisoned");
                    s.status = SolverStatus::Error;
                    s.error_msg = Some("No user-defined problem loaded — click Load first".to_string());
                    return;
                }
            };
            thread::Builder::new()
                .name("pinn-solver".into())
                .spawn(move || {
                    pinn_solver::runner::run_training_user_problem(spec, tx_train, rx_ctrl);
                })
                .expect("failed to spawn solver thread");
            return;
        }

        thread::Builder::new()
            .name("pinn-solver".into())
            .spawn(move || match problem_kind {
                ProblemKind::Kirsch => {
                    pinn_solver::run_training(config, tx_train, rx_ctrl);
                }
                ProblemKind::PinLug => {
                    pinn_solver::runner::run_training_pinlug(config, tx_train, rx_ctrl);
                }
            })
            .expect("failed to spawn solver thread");
    }

    fn stop_solver(&mut self) {
        if let Some(ref tx) = self.tx_control {
            let _ = tx.send(ControlMsg::Stop);
        }
        self.tx_control  = None;
        self.rx_training = None;
    }

    fn send_warm_start(&mut self) {
        let new_geo_hash = self.config.geometry.geometry_hash();
        let geometry_changed = new_geo_hash != self.prev_geo_hash;

        if let Some(ref tx) = self.tx_control {
            let _ = tx.send(ControlMsg::WarmStart {
                config: self.config.clone(),
                geometry_changed,
            });
        } else {
            // No solver running — start fresh
            self.start_solver();
            return;
        }
        self.prev_geo_hash = new_geo_hash;

        // Reset state for warm-start
        let mut s = self.state.lock().expect("state mutex poisoned");
        s.status = SolverStatus::Running;
        s.total_loss.clear();
        s.energy_loss.clear();
        s.neumann_loss.clear();
        s.lr_history.clear();
        s.lam_energy.clear();
        s.lam_neumann.clear();
        s.n_colloc = 0;
        s.kt_estimate = None;
    }

    fn drain_channel(&mut self) {
        let rx = match &self.rx_training {
            Some(r) => r,
            None    => return,
        };

        // Drain, keeping only the latest message
        let mut last = None;
        loop {
            match rx.try_recv() {
                Ok(msg) => last = Some(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Solver thread ended without sending a final Done/Error message
                    // (e.g. it panicked) — surface that as an error rather than silently
                    // leaving the UI showing a stale "Running" status.
                    let mut s = self.state.lock().expect("state mutex poisoned");
                    if s.status == SolverStatus::Running {
                        s.status = SolverStatus::Error;
                        s.error_msg = Some("Solver thread ended unexpectedly".to_string());
                    }
                    drop(s);
                    self.rx_training = None;
                    self.tx_control  = None;
                    break;
                }
            }
        }

        if let Some(msg) = last {
            self.apply_msg(msg);
        }
    }

    fn apply_msg(&mut self, msg: TrainingMsg) {
        match msg {
            TrainingMsg::Update(upd) => {
                let mut s = self.state.lock().expect("state mutex poisoned");
                s.step = upd.step;
                s.total_loss.push(upd.total_loss);
                s.energy_loss.push(upd.energy_loss);
                s.neumann_loss.push(upd.neumann_loss);
                s.lr_history.push(upd.lr);
                s.lam_energy.push(upd.lam_energy);
                s.lam_neumann.push(upd.lam_neumann);
                s.n_colloc = upd.n_colloc;
                s.kt_estimate = upd.kt_estimate;
                s.status = SolverStatus::Running;

                if let Some(vis) = upd.vis {
                    s.von_mises = vis.von_mises;
                    s.sigma_xx  = vis.sigma_xx;
                    s.sigma_yy  = vis.sigma_yy;
                    s.sigma_xy  = vis.sigma_xy;
                    s.disp_u    = vis.disp_u;
                    s.disp_v    = vis.disp_v;
                }
            }
            TrainingMsg::PinLugUpdate(upd) => {
                let mut s = self.state.lock().expect("state mutex poisoned");
                s.step = upd.step;
                s.total_loss.push(upd.total_loss);
                s.energy_loss.push(upd.energy_loss);
                s.neumann_loss.push(upd.neumann_loss);
                s.lr_history.push(upd.lr);
                s.lam_energy.push(upd.lam_energy);
                s.lam_neumann.push(upd.lam_neumann);
                s.n_colloc = upd.n_colloc;
                s.convergence_metric = upd.convergence_metric;
                s.status = SolverStatus::Running;

                if let Some(vis) = upd.vis {
                    s.pinlug_pin = Some(vis.pin);
                    s.pinlug_lug = Some(vis.lug);
                }
            }
            TrainingMsg::Done => {
                let mut s = self.state.lock().expect("state mutex poisoned");
                s.status = SolverStatus::Converged;
                self.tx_control  = None;
                self.rx_training = None;
            }
            TrainingMsg::Error(e) => {
                let mut s = self.state.lock().expect("state mutex poisoned");
                s.status    = SolverStatus::Error;
                s.error_msg = Some(e);
                self.tx_control  = None;
                self.rx_training = None;
            }
            TrainingMsg::ExportComplete(path) => {
                // Purely informational — must NOT change state.status (Running stays
                // Running, Converged stays Converged).
                self.export_status = Some(format!("Exported: {path}"));
            }
        }
    }
}

impl eframe::App for StressSolverApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_channel();

        // Request repaint while solver is running
        let running = {
            let s = self.state.lock().expect("state mutex poisoned");
            s.status == SolverStatus::Running
        };
        if running {
            ctx.request_repaint();
        }

        let state_snap = {
            self.state.lock().expect("state mutex poisoned").clone()
        };

        let mut on_solve      = false;
        let mut on_warm_start = false;
        let mut on_stop       = false;
        let mut on_export     = false;
        let prev_problem_kind = self.problem_kind;

        // ── Left panel: parameters ──────────────────────────────
        let mut user_ui = panels::params::UserProblemUi {
            active: &mut self.user_defined_active,
            spec_path: &mut self.user_spec_path,
            spec: &mut self.user_spec,
            error: &mut self.user_spec_error,
        };
        egui::SidePanel::left("params_panel")
            .min_width(220.0)
            .max_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    panels::params::show(
                        ui,
                        &mut self.config,
                        &state_snap,
                        &mut self.selected_field,
                        &mut self.problem_kind,
                        self.export_status.as_deref(),
                        &mut on_solve,
                        &mut on_warm_start,
                        &mut on_stop,
                        &mut on_export,
                        &mut user_ui,
                    );
                });
            });

        // Problem kind flipped this frame — swap in the matching default config. Only
        // relevant when the user-defined radio isn't active (its own "Load" button is the
        // equivalent trigger for that mode, handled inside `params::show` itself).
        if !self.user_defined_active && self.problem_kind != prev_problem_kind {
            self.config = match self.problem_kind {
                ProblemKind::Kirsch => SolverConfig::default_kirsch(),
                ProblemKind::PinLug => SolverConfig::default_pinlug(),
            };
        }

        // ── Right panel: stress heatmap ─────────────────────────
        // In user-defined mode, `state_snap`'s vis fields (`von_mises`/etc.) are populated
        // by the SAME `TrainingMsg::Update` arm Kirsch uses (see CLAUDE.md's ingestion
        // section — this problem is single-domain, so it reuses `TrainingUpdate`/
        // `select_field`'s existing `ProblemKind::Kirsch` branch verbatim); only the hole
        // overlay differs, via `user_holes` below. `heatmap_config` swaps in a placeholder
        // `SolverConfig` whose geometry matches the loaded spec's real bounding box (so the
        // grid/border draw correctly), leaving `self.config` itself untouched.
        let heatmap_config = if self.user_defined_active {
            self.user_spec.as_ref().map(|spec| {
                let mut c = SolverConfig::default_kirsch();
                c.geometry = spec.geometry.to_placeholder();
                c
            })
        } else {
            None
        };
        let heatmap_config = heatmap_config.as_ref().unwrap_or(&self.config);
        let heatmap_problem_kind = if self.user_defined_active { ProblemKind::Kirsch } else { self.problem_kind };
        let user_holes = self.user_spec.as_ref().map(|s| s.geometry.holes.as_slice());

        egui::SidePanel::right("heatmap_panel")
            .min_width(300.0)
            .show(ctx, |ui| {
                panels::heatmap::show(
                    ui,
                    &state_snap,
                    heatmap_config,
                    heatmap_problem_kind,
                    self.selected_field,
                    &mut self.texture,
                    &mut self.colorbar_range,
                    if self.user_defined_active { user_holes } else { None },
                );
            });

        // ── Central panel: training curves ──────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            panels::training::show(ui, &state_snap);
        });

        // Handle button actions
        if on_solve      { self.start_solver(); }
        if on_warm_start { self.send_warm_start(); }
        if on_stop       {
            self.stop_solver();
            let mut s = self.state.lock().expect("state mutex poisoned");
            s.status = SolverStatus::Idle;
        }
        if on_export {
            if let Some(ref tx) = self.tx_control {
                let _ = tx.send(ControlMsg::ExportContactPressure);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::messages::{PinLugTrainingUpdate, PinLugVisFields};

    fn fresh_app() -> StressSolverApp {
        StressSolverApp::from_config(SolverConfig::default_kirsch())
    }

    #[test]
    fn from_config_with_problem_kind_seeds_the_starting_selection() {
        let app = StressSolverApp::from_config_with_problem_kind(
            SolverConfig::default_pinlug(), ProblemKind::PinLug,
        );
        assert_eq!(app.problem_kind, ProblemKind::PinLug);
    }

    #[test]
    fn from_config_defaults_to_kirsch() {
        let app = fresh_app();
        assert_eq!(app.problem_kind, ProblemKind::Kirsch);
    }

    fn tiny_vis_fields() -> pinn_core::messages::VisFields {
        let grid = ndarray::Array2::zeros((2, 2));
        pinn_core::messages::VisFields {
            von_mises: grid.clone(),
            sigma_xx:  grid.clone(),
            sigma_yy:  grid.clone(),
            sigma_xy:  grid.clone(),
            disp_u:    grid.clone(),
            disp_v:    grid,
        }
    }

    #[test]
    fn apply_msg_pinlug_update_populates_pinlug_vis_fields_not_kirsch_fields() {
        let mut app = fresh_app();

        let upd = PinLugTrainingUpdate {
            step: 42,
            total_loss:   1.0,
            energy_loss:  0.5,
            neumann_loss: 0.5,
            lr:           1e-3,
            lam_energy:   1.0,
            lam_neumann:  1.0,
            n_colloc:     128,
            convergence_metric: Some(0.0012),
            vis: Some(PinLugVisFields { pin: tiny_vis_fields(), lug: tiny_vis_fields() }),
        };

        app.apply_msg(TrainingMsg::PinLugUpdate(Box::new(upd)));

        let s = app.state.lock().expect("state mutex poisoned");
        assert!(s.pinlug_pin.is_some(), "pinlug_pin must be populated");
        assert!(s.pinlug_lug.is_some(), "pinlug_lug must be populated");
        assert_eq!(s.convergence_metric, Some(0.0012));
        assert_eq!(s.step, 42);
        assert!(s.kt_estimate.is_none(), "PinLugUpdate must never populate the Kirsch-only field");
    }

    #[test]
    fn apply_msg_export_complete_does_not_change_solver_status() {
        let mut app = fresh_app();
        {
            let mut s = app.state.lock().expect("state mutex poisoned");
            s.status = SolverStatus::Running;
        }

        app.apply_msg(TrainingMsg::ExportComplete("out/contact_pressure.csv".to_string()));

        let s = app.state.lock().expect("state mutex poisoned");
        assert_eq!(s.status, SolverStatus::Running, "export completion must not change status");
    }

    #[test]
    fn apply_msg_existing_kirsch_variants_still_handled() {
        let mut app = fresh_app();
        app.apply_msg(TrainingMsg::Done);

        let s = app.state.lock().expect("state mutex poisoned");
        assert_eq!(s.status, SolverStatus::Converged);
    }
}
