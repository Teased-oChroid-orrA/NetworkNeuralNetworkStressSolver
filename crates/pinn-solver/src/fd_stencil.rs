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

/// Widening factor applied to `hx`/`hy` when building the FD step used ONLY by the
/// second-order (Hessian) stencil - NOT the first-derivative `assemble_stencil`/
/// `compute_strains` path, which is untouched and keeps using the caller's own `fd_h` exactly
/// as before.
///
/// Second-order FD divides by `h²` (`compute_hessian`'s `cxx = sx²/hx²`), so at production's
/// typical `fd_h≈1e-3` the raw second-difference signal is subtracting near-equal f32 values
/// and then getting divided by an already-tiny `h²`, which amplifies f32 rounding noise by a
/// huge factor - confirmed empirically twice: (1) the manufactured-quadratic-field analytical
/// test (`energy::tests::hessian_recovers_exact_second_derivatives_...`) needed a 10x larger h
/// than production's `fd_h` to resolve known-nonzero curvature to <0.1% error; (2) the real
/// no-hole-plate term-gradient diagnostic, run with the Hessian-based `equilibrium` term at
/// production's unwidened `fd_h`, still showed `equilibrium`'s gradient 5-6 orders of
/// magnitude smaller than every other term's - i.e. STILL functionally inert, this time
/// because its signal was swamped by FD noise rather than (as with the direct-σ version)
/// having no real signal to begin with. `10.0` is the same factor the analytical test found
/// sufficient.
pub const HESSIAN_FD_SAFETY_MULT: f32 = 10.0;

/// Build the (wider) `FdConfig` the second-order stencil should use, from the same base
/// config every first-derivative caller already has - see `HESSIAN_FD_SAFETY_MULT`'s doc
/// comment. `sx`/`sy` (the coordinate-mapping scale, NOT a step size) are unchanged; only
/// `hx`/`hy` are widened.
pub fn hessian_fd_config(base: &FdConfig) -> FdConfig {
    FdConfig {
        hx: base.hx * HESSIAN_FD_SAFETY_MULT,
        hy: base.hy * HESSIAN_FD_SAFETY_MULT,
        sx: base.sx,
        sy: base.sy,
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

/// Full Hessian (second spatial derivatives) of a scalar field `f`, physical units - the 6
/// components `∂²u/∂x²,∂²u/∂y²,∂²u/∂x∂y,∂²v/∂x²,∂²v/∂y²,∂²v/∂x∂y` needed to compute
/// `∇·(C:ε(u))` (equilibrium on DERIVED, not direct, stress - see `energy::
/// equilibrium_from_displacement_hessian_loss`'s doc comment for why this exists). `(u_xx,
/// u_yy, u_xy, v_xx, v_yy, v_xy)`, in that order.
pub type Hessian<B> = (
    Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>,
    Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>,
);

/// Build a `[9*N, 3]` stencil batch for second-derivative (Hessian) computation - the standard
/// 9-point stencil (center + 4 axial neighbors + 4 diagonal neighbors), extending
/// `assemble_stencil`'s 5-point layout with the 4 diagonal points a mixed partial derivative
/// needs. Row layout:
///   [0..N)     centre
///   [N..2N)    x+hx
///   [2N..3N)   x−hx
///   [3N..4N)   y+hy
///   [4N..5N)   y−hy
///   [5N..6N)   x+hx, y+hy
///   [6N..7N)   x+hx, y−hy
///   [7N..8N)   x−hx, y+hy
///   [8N..9N)   x−hx, y−hy
pub fn assemble_second_order_stencil<B: Backend>(
    pts: &Tensor<B, 2>,
    fd: &FdConfig,
    device: &B::Device,
) -> Tensor<B, 2> {
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
            pts.clone() + make_offset(fd.hx, fd.hy),
            pts.clone() + make_offset(fd.hx, -fd.hy),
            pts.clone() + make_offset(-fd.hx, fd.hy),
            pts.clone() + make_offset(-fd.hx, -fd.hy),
        ],
        0,
    )
}

/// Compute the Hessian of both `u` (column 0) and `v` (column 1) from a `[9N, C]`
/// second-order stencil output (`C >= 2`; extra columns, e.g. mDEM's direct σ, are ignored).
///
/// Second-derivative physical scaling needs the coordinate-mapping factor SQUARED, NOT the
/// same `sx`/`sy` first derivatives use (`compute_strains`'s `c_x`/`c_y`) - a genuinely
/// different, easy-to-get-wrong formula, not a copy-paste of the first-derivative one:
///   ∂²f/∂x²_phys  = sx² · (f_xp − 2f_c + f_xm) / hx²
///   ∂²f/∂y²_phys  = sy² · (f_yp − 2f_c + f_ym) / hy²
///   ∂²f/∂x∂y_phys = sx·sy · (f_pp − f_pm − f_mp + f_mm) / (4·hx·hy)
/// Verified against an exact manufactured polynomial field before ever being wired into a
/// real loss term - see `energy::tests::hessian_recovers_exact_second_derivatives_of_a_
/// manufactured_quadratic_field`.
pub fn compute_hessian<B: Backend>(
    stencil_out: Tensor<B, 2>,
    n: usize,
    fd: &FdConfig,
) -> Hessian<B> {
    let cxx = fd.sx * fd.sx / (fd.hx as f64 * fd.hx as f64);
    let cyy = fd.sy * fd.sy / (fd.hy as f64 * fd.hy as f64);
    let cxy = fd.sx * fd.sy / (4.0 * fd.hx as f64 * fd.hy as f64);

    let col = |r0: usize, r1: usize, c: usize| -> Tensor<B, 1> {
        stencil_out.clone().slice([r0..r1, c..c + 1]).reshape([n])
    };

    let second_deriv = |field_col: usize| -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let f_c  = col(0,     n,     field_col);
        let f_xp = col(n,     2 * n, field_col);
        let f_xm = col(2 * n, 3 * n, field_col);
        let f_yp = col(3 * n, 4 * n, field_col);
        let f_ym = col(4 * n, 5 * n, field_col);
        let f_pp = col(5 * n, 6 * n, field_col);
        let f_pm = col(6 * n, 7 * n, field_col);
        let f_mp = col(7 * n, 8 * n, field_col);
        let f_mm = col(8 * n, 9 * n, field_col);

        let f_xx = (f_xp.clone() - f_c.clone().mul_scalar(2.0) + f_xm).mul_scalar(cxx);
        let f_yy = (f_yp - f_c.mul_scalar(2.0) + f_ym).mul_scalar(cyy);
        let f_xy = (f_pp - f_pm - f_mp + f_mm).mul_scalar(cxy);
        (f_xx, f_yy, f_xy)
    };

    let (u_xx, u_yy, u_xy) = second_deriv(0);
    let (v_xx, v_yy, v_xy) = second_deriv(1);
    (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy)
}
