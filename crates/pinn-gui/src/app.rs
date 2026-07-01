use std::{
    sync::{Arc, Mutex},
    thread,
};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use egui::TextureHandle;
use pinn_core::{
    messages::{ControlMsg, SolverConfig, TrainingMsg},
    FieldType, SolverStatus, TrainingState,
};

use crate::panels;

pub struct StressSolverApp {
    config: SolverConfig,
    state:  Arc<Mutex<TrainingState>>,

    // Channels
    tx_control: Option<Sender<ControlMsg>>,
    rx_training: Option<Receiver<TrainingMsg>>,

    // UI state
    selected_field: FieldType,
    texture: Option<TextureHandle>,
    colorbar_range: (f32, f32),
    prev_geo_hash: u64,
}

impl StressSolverApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, config: SolverConfig) -> Self {
        let [nx, ny] = config.vis_grid;
        let state = Arc::new(Mutex::new(TrainingState::new([nx, ny])));
        let geo_hash = config.geometry.geometry_hash();
        Self {
            config,
            state,
            tx_control:    None,
            rx_training:   None,
            selected_field: FieldType::VonMises,
            texture:        None,
            colorbar_range: (0.0, 1.0),
            prev_geo_hash:  geo_hash,
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

        thread::Builder::new()
            .name("pinn-solver".into())
            .spawn(move || {
                pinn_solver::run_training(config, tx_train, rx_ctrl);
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

        // ── Left panel: parameters ──────────────────────────────
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
                        &mut on_solve,
                        &mut on_warm_start,
                        &mut on_stop,
                    );
                });
            });

        // ── Right panel: stress heatmap ─────────────────────────
        egui::SidePanel::right("heatmap_panel")
            .min_width(300.0)
            .show(ctx, |ui| {
                panels::heatmap::show(
                    ui,
                    &state_snap,
                    &self.config,
                    self.selected_field,
                    &mut self.texture,
                    &mut self.colorbar_range,
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
    }
}
