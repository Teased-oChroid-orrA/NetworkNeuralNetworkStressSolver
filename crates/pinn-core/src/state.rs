use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverStatus {
    Idle,
    Running,
    Paused,
    Converged,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    VonMises,
    SigmaXX,
    SigmaYY,
    SigmaXY,
    DispU,
    DispV,
}

impl FieldType {
    pub fn label(&self) -> &'static str {
        match self {
            FieldType::VonMises => "Von Mises σ",
            FieldType::SigmaXX  => "σ_xx",
            FieldType::SigmaYY  => "σ_yy",
            FieldType::SigmaXY  => "σ_xy",
            FieldType::DispU    => "u_x",
            FieldType::DispV    => "u_y",
        }
    }

    /// True for the two displacement fields (stored in meters); false for the four
    /// stress fields (stored in Pa). Used to pick the right display unit.
    pub fn is_displacement(&self) -> bool {
        matches!(self, FieldType::DispU | FieldType::DispV)
    }
}

/// Complete state visible to the GUI, updated by the solver thread
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub step: usize,

    // Per-step training history
    pub total_loss:   Vec<f32>,
    pub energy_loss:  Vec<f32>,
    pub neumann_loss: Vec<f32>,
    pub lr_history:   Vec<f32>,
    pub lam_energy:   Vec<f32>,
    pub lam_neumann:  Vec<f32>,

    // Visualization fields — shape [Ny_vis × Nx_vis], NaN where outside domain
    pub von_mises: Array2<f32>,
    pub sigma_xx:  Array2<f32>,
    pub sigma_yy:  Array2<f32>,
    pub sigma_xy:  Array2<f32>,
    pub disp_u:    Array2<f32>,
    pub disp_v:    Array2<f32>,

    /// Number of active collocation points
    pub n_colloc: usize,

    /// K_t estimate from current PINN solution
    pub kt_estimate: Option<f32>,

    pub status: SolverStatus,
    pub error_msg: Option<String>,

    /// [Nx_vis, Ny_vis]
    pub vis_grid: [usize; 2],
}

impl TrainingState {
    pub fn new(vis_grid: [usize; 2]) -> Self {
        let [nx, ny] = vis_grid;
        let empty = Array2::zeros((ny, nx));
        Self {
            step: 0,
            total_loss:   Vec::new(),
            energy_loss:  Vec::new(),
            neumann_loss: Vec::new(),
            lr_history:   Vec::new(),
            lam_energy:   Vec::new(),
            lam_neumann:  Vec::new(),
            von_mises: empty.clone(),
            sigma_xx:  empty.clone(),
            sigma_yy:  empty.clone(),
            sigma_xy:  empty.clone(),
            disp_u:    empty.clone(),
            disp_v:    empty,
            n_colloc: 0,
            kt_estimate: None,
            status: SolverStatus::Idle,
            error_msg: None,
            vis_grid,
        }
    }

    pub fn field(&self, kind: FieldType) -> &Array2<f32> {
        match kind {
            FieldType::VonMises => &self.von_mises,
            FieldType::SigmaXX  => &self.sigma_xx,
            FieldType::SigmaYY  => &self.sigma_yy,
            FieldType::SigmaXY  => &self.sigma_xy,
            FieldType::DispU    => &self.disp_u,
            FieldType::DispV    => &self.disp_v,
        }
    }
}
