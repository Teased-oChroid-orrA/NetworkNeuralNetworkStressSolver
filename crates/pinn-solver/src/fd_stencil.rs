/// Finite-difference stencil for computing spatial derivatives of the network output.
///
/// Network inputs are in normalised coordinates x_norm ∈ [-1,1].
/// Network outputs (u,v,w) are in physical units [m].
/// Physical strains require scaling: ε_xx_phys = (∂u/∂x_norm) * sx
/// where sx = 2/(x1−x0) maps normalised derivatives to physical ones.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use crate::network::ElasticityNet;

#[derive(Debug, Clone, Copy)]
pub struct FdConfig {
    /// FD step in normalised x-coordinate
    pub hx: f32,
    /// FD step in normalised y-coordinate
    pub hy: f32,
    /// Physical strain scale: ε_phys = ε_norm * sx  (sx = 2/(x1-x0))
    pub sx: f64,
    /// Physical strain scale: ε_phys = ε_norm * sy  (sy = 2/(y1-y0))
    pub sy: f64,
}

impl FdConfig {
    /// `h`: step in normalised coords.  `domain_width`, `domain_height`: (x1-x0), (y1-y0) in m.
    pub fn new(h: f32, domain_width: f64, domain_height: f64) -> Self {
        Self {
            hx: h,
            hy: h,
            sx: 2.0 / domain_width,
            sy: 2.0 / domain_height,
        }
    }
}

/// Convert a slice of (x_norm, y_norm) pairs → [N, 3] Tensor (appends z=0 column).
pub fn norm_pts_to_tensor<B: Backend>(pts: &[[f32; 2]], device: &B::Device) -> Tensor<B, 2> {
    let n = pts.len();
    let flat: Vec<f32> = pts.iter().flat_map(|p| [p[0], p[1], 0.0f32]).collect();
    Tensor::<B, 2>::from_data(TensorData::new(flat, vec![n, 3]), device)
}

/// Build a [5*N, 3] stencil batch from a [N, 3] normalised-coordinate tensor.
///
/// Row layout of the output:
///   [0..N)     centre
///   [N..2N)    x+hx
///   [2N..3N)   x−hx
///   [3N..4N)   y+hy
///   [4N..5N)   y−hy
pub fn assemble_stencil<B: Backend>(
    pts: &Tensor<B, 2>,
    fd: &FdConfig,
    device: &B::Device,
) -> Tensor<B, 2> {
    // A [1, 3] offset broadcasts against [n, 3] — avoids building and uploading an
    // O(n)-sized offset tensor per call (the offset is the same 3 values for every row).
    let make_offset = |dx: f32, dy: f32| -> Tensor<B, 2> {
        Tensor::<B, 2>::from_data(TensorData::new(vec![dx, dy, 0.0f32], vec![1, 3]), device)
    };

    Tensor::cat(
        vec![
            pts.clone(),
            pts.clone() + make_offset(fd.hx, 0.0),
            pts.clone() + make_offset(-fd.hx, 0.0),
            pts.clone() + make_offset(0.0, fd.hy),
            pts.clone() + make_offset(0.0, -fd.hy),
        ],
        0,
    )
}

/// Compute plane-stress strains (physical units, 1/1 = dimensionless strain) from stencil output.
///
/// `stencil_out` : [5N, 3] — network output at all stencil points (u, v, w)
/// Returns `(ε_xx, ε_yy, ε_xy)` as shape-[N] tensors in physical strain units.
pub fn compute_strains<B: Backend>(
    stencil_out: Tensor<B, 2>,
    n: usize,
    fd: &FdConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    // Combined scale: derivative in norm-coords * coordinate-mapping scale = physical strain
    let c_x = fd.sx / (2.0 * fd.hx as f64); // sx / (2hx)
    let c_y = fd.sy / (2.0 * fd.hy as f64); // sy / (2hy)

    let col = |r0: usize, r1: usize, c: usize| -> Tensor<B, 1> {
        stencil_out.clone().slice([r0..r1, c..c + 1]).reshape([n])
    };

    let u_xp = col(n,     2 * n, 0);
    let u_xm = col(2 * n, 3 * n, 0);
    let u_yp = col(3 * n, 4 * n, 0);
    let u_ym = col(4 * n, 5 * n, 0);

    let v_xp = col(n,     2 * n, 1);
    let v_xm = col(2 * n, 3 * n, 1);
    let v_yp = col(3 * n, 4 * n, 1);
    let v_ym = col(4 * n, 5 * n, 1);

    // ε_xx = ∂u/∂x_phys = (u_xp - u_xm) * c_x
    let eps_xx = (u_xp - u_xm).mul_scalar(c_x);
    // ε_yy = ∂v/∂y_phys = (v_yp - v_ym) * c_y
    let eps_yy = (v_yp - v_ym).mul_scalar(c_y);
    // ε_xy = ½(∂u/∂y + ∂v/∂x)
    let eps_xy = ((u_yp - u_ym).mul_scalar(c_y) + (v_xp - v_xm).mul_scalar(c_x)).mul_scalar(0.5);

    (eps_xx, eps_yy, eps_xy)
}

/// All-in-one: evaluate network strains at a set of normalised interior points.
pub fn network_strains<B: Backend>(
    model: &ElasticityNet<B>,
    pts_norm: &Tensor<B, 2>,
    fd: &FdConfig,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let n = pts_norm.dims()[0];
    let stencil = assemble_stencil(pts_norm, fd, device);
    let out = model.forward(stencil);
    compute_strains(out, n, fd)
}
