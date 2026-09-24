//! Pure data + update logic for the `--tui` dashboard. Deliberately zero `ratatui`/`crossterm`
//! dependency (see `docs/tui-mode-plan.md`'s own module-boundary design) - unit-testable with
//! plain `TrainingUpdate`/`PinLugTrainingUpdate` literals, no terminal needed. `ui.rs` reads
//! this struct to render; `mod.rs` is the only file that constructs/updates it from a live
//! `TrainingMsg` stream.

use pinn_core::messages::{AmrSweepReport, ArchitectureEvent, HoleAnalysis, PinLugTrainingUpdate, TrainingUpdate};

/// Which of the two live-training shapes this run is - Kirsch/plate (`TrainingUpdate`, `Kt`)
/// vs. pin-lug (`PinLugTrainingUpdate`, a generic `convergence_metric`). Mirrors
/// `pinn_core::TrainingState`'s own "one struct, either shape populated" precedent rather than
/// inventing a new convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuiKtSource {
    /// Nothing populated yet (before the first `Update`/`PinLugUpdate` arrives).
    #[default]
    None,
    /// Kt read from `kt_estimate` directly (Kirsch, or a User-Defined plate spec with the
    /// annular-decomposition architecture active).
    KtEstimate,
    /// Kt read per-hole from `hole_analyses` (the default single-domain User-Defined plate
    /// path, which has no single scalar `kt_estimate`).
    HoleAnalyses,
    /// `convergence_metric` (pin-lug - not literally Kt, a generic scalar the problem defines;
    /// see `ConvergenceTracker`'s own doc comment).
    ConvergenceMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Done,
    Error,
}

/// Dashboard state - a plain struct, NOT a reuse of `pinn_core::TrainingState` (that type
/// carries `Array2<f32>` heatmap grids this v1 TUI has no use for - see `docs/tui-mode-plan.md`'s
/// own "no heatmap in v1" scope note; duplicating five scalar/Vec fields here is cheaper than
/// carrying that dependency for fields never read).
#[derive(Debug, Clone, Default)]
pub struct TuiState {
    pub tab: usize,
    pub scroll: u16,
    pub warnings_only: bool,
    pub run_summary: String,
    pub control_status: String,
    pub logs: Vec<String>,
    pub objective: Option<pinn_core::messages::ObjectiveTelemetry>,
    pub energy_balance: Option<pinn_core::messages::EnergyBalance>,
    pub reaction_force: Option<pinn_core::messages::ReactionForce>,
    pub boundary_residual: Option<(usize, f64, f64)>,
    pub gradient_shares: Option<pinn_core::messages::GradientShareSummary>,
    pub convergence: Option<pinn_core::messages::ConvergenceEvidenceSummary>,
    pub sources: Vec<(&'static str, &'static str)>,
    pub history_steps: Vec<usize>,
    pub status: Option<RunStatus>,
    pub error_msg: Option<String>,
    pub step: usize,
    /// Captured from the caller's own `config`/`spec` BEFORE spawning the training thread -
    /// neither `TrainingUpdate` nor `PinLugTrainingUpdate` carries `max_steps` on the wire.
    pub max_steps: usize,
    pub total_loss: f32,
    pub energy_loss: f32,
    pub neumann_loss: f32,
    pub lr: f32,
    pub lam_energy: f32,
    pub lam_neumann: f32,
    pub n_colloc: usize,
    pub grad_norm: Option<f32>,
    pub kt_source: TuiKtSource,
    pub kt_estimate: Option<f32>,
    pub hole_analyses: Vec<HoleAnalysis>,
    pub convergence_metric: Option<f32>,
    pub latest_amr_sweep: Option<AmrSweepReport>,
    pub latest_architecture_event: Option<ArchitectureEvent>,
    /// Total-loss history for the chart, oldest first - appended to, never overwritten, so a
    /// dropped intermediate `Update` (the data channel is `bounded(1)`, drop-oldest) just means
    /// a coarser chart, not corrupted history.
    pub total_loss_history: Vec<f32>,
}

impl TuiState {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps, ..Default::default() }
    }

    /// Apply a Kirsch/plate `TrainingUpdate`. `kt_estimate` (annular-decomposition/Kirsch path)
    /// takes priority over `hole_analyses` (default plate path) when both happen to be present,
    /// matching `pinn-gui`'s own params panel precedent of showing both sources rather than
    /// picking one exclusively - but for `TuiKtSource` (this dashboard's own single headline
    /// number) `kt_estimate` is the more authoritative of the two when both exist.
    pub fn apply(&mut self, u: &TrainingUpdate) {
        if self.status != Some(RunStatus::Error) {
            self.status = Some(RunStatus::Running);
        }
        self.step = u.step;
        self.total_loss = u.total_loss;
        self.energy_loss = u.energy_loss;
        self.neumann_loss = u.neumann_loss;
        self.lr = u.lr;
        self.lam_energy = u.lam_energy;
        self.lam_neumann = u.lam_neumann;
        self.n_colloc = u.n_colloc;
        self.grad_norm = u.grad_norm;
        self.push_history(u.step, u.total_loss);
        if !u.total_loss.is_finite() || u.grad_norm.is_some_and(|g| !g.is_finite()) {
            self.log("ERROR non-finite loss or gradient reported by solver".into());
            self.set_error("non-finite loss or gradient reported by solver".into());
        }

        self.objective = u.objective.clone();
        if let Some(energy) = u.energy_balance { self.energy_balance = Some(energy); }
        if let Some(force) = u.reaction_force { self.reaction_force = Some(force); }
        if u.vis.is_some() { self.boundary_residual = Some((u.step, u.bc_residual_rms, u.bc_residual_max)); }
        if let Some(shares) = &u.gradient_share_report { self.gradient_shares = Some(shares.clone()); }
        if let Some(evidence) = &u.convergence_evidence { self.convergence = Some(evidence.clone()); }
        if !u.stress_source_report.is_empty() { self.sources = u.stress_source_report.clone(); }
        if !u.hole_analyses.is_empty() { self.hole_analyses = u.hole_analyses.clone(); }

        if let Some(kt) = u.kt_estimate {
            self.kt_source = TuiKtSource::KtEstimate;
            self.kt_estimate = Some(kt);
        } else if !u.hole_analyses.is_empty() {
            self.kt_source = TuiKtSource::HoleAnalyses;
            self.hole_analyses = u.hole_analyses.clone();
        }
        if let Some(sweep) = &u.amr_sweep {
            self.latest_amr_sweep = Some(sweep.clone());
        }
        if let Some(event) = &u.architecture_event {
            self.latest_architecture_event = Some(event.clone());
        }
    }

    /// Apply a pin-lug `PinLugTrainingUpdate` - a genuinely different shape from `apply` above,
    /// not a thin wrapper: `convergence_metric` instead of Kt, `amr_sweep` is a `Vec` (0-2
    /// entries, one per domain) instead of `TrainingUpdate`'s `Option`, no `hole_analyses`/
    /// `architecture_event` field exists on this type at all.
    pub fn apply_pinlug(&mut self, u: &PinLugTrainingUpdate) {
        if self.status != Some(RunStatus::Error) {
            self.status = Some(RunStatus::Running);
        }
        self.step = u.step;
        self.total_loss = u.total_loss;
        self.energy_loss = u.energy_loss;
        self.neumann_loss = u.neumann_loss;
        self.lr = u.lr;
        self.lam_energy = u.lam_energy;
        self.lam_neumann = u.lam_neumann;
        self.n_colloc = u.n_colloc;
        self.grad_norm = u.grad_norm;
        self.push_history(u.step, u.total_loss);
        if !u.total_loss.is_finite() || u.grad_norm.is_some_and(|g| !g.is_finite()) {
            self.log("ERROR non-finite loss or gradient reported by solver".into());
            self.set_error("non-finite loss or gradient reported by solver".into());
        }

        self.kt_source = TuiKtSource::ConvergenceMetric;
        self.convergence_metric = u.convergence_metric;
        if let Some(sweep) = u.amr_sweep.last() {
            self.latest_amr_sweep = Some(sweep.clone());
        }
    }

    fn push_history(&mut self, step: usize, loss: f32) {
        const HISTORY_LIMIT: usize = 1024;
        if self.history_steps.last().is_some_and(|&last| step < last) {
            self.history_steps.clear();
            self.total_loss_history.clear();
        }
        if self.history_steps.len() == HISTORY_LIMIT {
            self.history_steps.remove(0);
            self.total_loss_history.remove(0);
        }
        self.history_steps.push(step);
        self.total_loss_history.push(loss);
    }

    pub fn log(&mut self, message: String) {
        if self.logs.len() == 256 { self.logs.remove(0); }
        self.logs.push(message);
    }

    pub fn set_done(&mut self) {
        if self.status != Some(RunStatus::Error) {
            self.status = Some(RunStatus::Done);
        }
    }

    pub fn set_error(&mut self, msg: String) {
        self.status = Some(RunStatus::Error);
        self.error_msg = Some(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::messages::{HoleBoundaryPoint, StressConcentration};

    fn base_update() -> TrainingUpdate {
        TrainingUpdate {
            objective: None,
            step: 10,
            total_loss: 1.5,
            energy_loss: 1.0,
            neumann_loss: 0.5,
            lr: 1e-3,
            lam_energy: 1.0,
            lam_neumann: 1.0,
            n_colloc: 4096,
            kt_estimate: None,
            vis: None,
            amr_sweep: None,
            hole_analyses: Vec::new(),
            grad_norm: Some(0.25),
            bc_residual_rms: 0.0,
            bc_residual_max: 0.0,
            reaction_force: None,
            energy_balance: None,
            network_snapshot: None,
            architecture_event: None,
            gradient_share_report: None,
            gradient_conflict_report: None,
            stress_source_report: Vec::new(),
            boundary_operator_report: Vec::new(),
            derivative_order_report: Vec::new(),
            formulation_kind_report: Vec::new(),
            constraint_report: Vec::new(),
            no_hole_benchmark: None,
            ad_fd_strain_diagnostic: None,
            convergence_evidence: None,
        }
    }

    fn hole_analysis(kt: f64) -> HoleAnalysis {
        HoleAnalysis {
            hole_index: 0,
            profile: Vec::<HoleBoundaryPoint>::new(),
            concentration: StressConcentration {
                nominal_stress: 1e7,
                max_von_mises: kt * 1e7,
                max_theta_deg: 90.0,
                kt,
                stress_projection: "VonMises",
                angular_refinement_relative_change: None,
                radial_offset_refinement_relative_change: None,
                refinement_converged: None,
                domain_classification: "FiniteDomainReference",
            },
            stress_diagnostic: None,
        }
    }

    #[test]
    fn apply_populates_scalar_fields_and_appends_loss_history_not_overwrites() {
        let mut state = TuiState::new(1000);
        state.apply(&base_update());
        assert_eq!(state.step, 10);
        assert_eq!(state.total_loss, 1.5);
        assert_eq!(state.grad_norm, Some(0.25));
        assert_eq!(state.total_loss_history, vec![1.5]);
        assert_eq!(state.status, Some(RunStatus::Running));

        let mut second = base_update();
        second.step = 20;
        second.total_loss = 1.2;
        state.apply(&second);
        assert_eq!(state.total_loss_history, vec![1.5, 1.2], "history must append, not overwrite");
        assert_eq!(state.step, 20);
    }

    #[test]
    fn apply_prefers_kt_estimate_over_hole_analyses_when_both_present() {
        let mut u = base_update();
        u.kt_estimate = Some(2.5);
        u.hole_analyses = vec![hole_analysis(2.7)];
        let mut state = TuiState::new(1000);
        state.apply(&u);
        assert_eq!(state.kt_source, TuiKtSource::KtEstimate);
        assert_eq!(state.kt_estimate, Some(2.5));
    }

    #[test]
    fn apply_falls_back_to_hole_analyses_when_no_kt_estimate() {
        let mut u = base_update();
        u.kt_estimate = None;
        u.hole_analyses = vec![hole_analysis(2.7), hole_analysis(2.9)];
        let mut state = TuiState::new(1000);
        state.apply(&u);
        assert_eq!(state.kt_source, TuiKtSource::HoleAnalyses);
        assert_eq!(state.hole_analyses.len(), 2);
    }

    #[test]
    fn apply_surfaces_amr_sweep_and_architecture_event() {
        let mut u = base_update();
        u.amr_sweep = Some(AmrSweepReport {
            domain_label: "interior", step: 10, points_before: 4096, points_after: 5000,
            residual_rms_before: 1.0, residual_max_before: 2.0,
            residual_rms_after: 0.5, residual_max_after: 1.0,
            sweep_duration_ms: 12.0,
            hole_zone_density_before: 0.0, hole_zone_density_after: 0.0,
            domain_mean_density_before: 0.0, domain_mean_density_after: 0.0,
        });
        u.architecture_event = Some(ArchitectureEvent {
            step: 10, description: "Grew width 64 -> 96".to_string(),
            hidden_dim_before: 64, hidden_dim_after: 96, n_hidden_before: 3, n_hidden_after: 3,
        });
        let mut state = TuiState::new(1000);
        state.apply(&u);
        assert!(state.latest_amr_sweep.is_some());
        assert_eq!(state.latest_architecture_event.as_ref().unwrap().description, "Grew width 64 -> 96");
    }

    fn base_pinlug_update() -> PinLugTrainingUpdate {
        PinLugTrainingUpdate {
            step: 5,
            total_loss: 2.0,
            energy_loss: 1.5,
            neumann_loss: 0.5,
            lr: 1e-3,
            lam_energy: 1.0,
            lam_neumann: 1.0,
            n_colloc: 2048,
            convergence_metric: Some(0.01),
            vis: None,
            amr_sweep: Vec::new(),
            grad_norm: Some(0.1),
        }
    }

    #[test]
    fn apply_pinlug_populates_convergence_metric_not_kt() {
        let mut state = TuiState::new(500);
        state.apply_pinlug(&base_pinlug_update());
        assert_eq!(state.kt_source, TuiKtSource::ConvergenceMetric);
        assert_eq!(state.convergence_metric, Some(0.01));
        assert_eq!(state.total_loss_history, vec![2.0]);
    }

    #[test]
    fn apply_pinlug_amr_sweep_is_a_vec_not_an_option() {
        let mut u = base_pinlug_update();
        u.amr_sweep = vec![
            AmrSweepReport {
                domain_label: "pin", step: 5, points_before: 100, points_after: 150,
                residual_rms_before: 1.0, residual_max_before: 2.0,
                residual_rms_after: 0.5, residual_max_after: 1.0,
                sweep_duration_ms: 5.0,
                hole_zone_density_before: 0.0, hole_zone_density_after: 0.0,
                domain_mean_density_before: 0.0, domain_mean_density_after: 0.0,
            },
            AmrSweepReport {
                domain_label: "lug", step: 5, points_before: 200, points_after: 250,
                residual_rms_before: 1.0, residual_max_before: 2.0,
                residual_rms_after: 0.5, residual_max_after: 1.0,
                sweep_duration_ms: 6.0,
                hole_zone_density_before: 0.0, hole_zone_density_after: 0.0,
                domain_mean_density_before: 0.0, domain_mean_density_after: 0.0,
            },
        ];
        let mut state = TuiState::new(500);
        state.apply_pinlug(&u);
        // "last" wins - the most recently reported domain sweep, not the first.
        assert_eq!(state.latest_amr_sweep.unwrap().domain_label, "lug");
    }

    #[test]
    fn set_done_and_set_error_update_status() {
        let mut state = TuiState::new(100);
        state.set_done();
        assert_eq!(state.status, Some(RunStatus::Done));
        let mut state2 = TuiState::new(100);
        state2.set_error("boom".to_string());
        assert_eq!(state2.status, Some(RunStatus::Error));
        assert_eq!(state2.error_msg.as_deref(), Some("boom"));
        state2.set_done();
        assert_eq!(state2.status, Some(RunStatus::Error), "Done must not hide a prior error");
    }

    #[test]
    fn non_finite_updates_fail_closed_for_plate_and_pinlug() {
        let mut plate = base_update();
        plate.total_loss = f32::NAN;
        let mut state = TuiState::new(100);
        state.apply(&plate);
        state.set_done();
        assert_eq!(state.status, Some(RunStatus::Error));

        let mut pinlug = base_pinlug_update();
        pinlug.grad_norm = Some(f32::INFINITY);
        let mut state = TuiState::new(100);
        state.apply_pinlug(&pinlug);
        state.set_done();
        assert_eq!(state.status, Some(RunStatus::Error));
    }
    #[test]
    fn signed_history_is_bounded_and_keeps_actual_step_numbers() {
        let mut state = TuiState::new(2000);
        for step in 0..1500 { state.push_history(step * 10, -(step as f32)); }
        assert_eq!(state.total_loss_history.len(), 1024);
        assert_eq!(state.history_steps[0], 4760);
        assert_eq!(*state.total_loss_history.last().unwrap(), -1499.0);
        state.push_history(0, -2.0);
        assert_eq!(state.history_steps, vec![0]);
        assert_eq!(state.total_loss_history, vec![-2.0]);
        for _ in 0..300 { state.log("WARN example".into()); }
        assert_eq!(state.logs.len(), 256);
    }

}
