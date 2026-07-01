/// Dirichlet (displacement) BC enforcement via distance-function ansatz.
///
/// For QuarterSymm geometry:
///   u = 0 on x_norm = -1 (physical x = 0, symmetry plane)
///   v = 0 on y_norm = -1 (physical y = 0, symmetry plane)
///
/// Hard enforcement on ANY batch size M:
///   u_modified = tanh(k*(x_norm + 1)) * u_raw
///   v_modified = tanh(k*(y_norm + 1)) * v_raw
///
/// k = max_dim / r_hole (from engine) ensures tanh ≈ 0.96 at the hole surface,
/// giving O(1) gradient signal at near-hole collocation points.

use burn::tensor::{backend::Backend, Tensor};
use pinn_core::SymmetryMode;

/// Apply the Dirichlet distance-function ansatz to a displacement (or mDEM) tensor,
/// given the corresponding normalised coordinate tensor (x, y, z in columns).
///
/// Handles two output widths automatically:
/// - 3 columns (plain DEM: u, v, w): applies tanh ansatz to u,v; zeros w (plane stress).
/// - 5 columns (mDEM: u, v, σ_xx, σ_yy, σ_xy): applies tanh ansatz to u,v; passes σ unchanged.
///
/// `k`: ansatz saturation factor from engine (auto-computed as max_dim / r_hole).
///
/// Works for ANY batch size M — use on the full [5N, D] stencil batch so FD
/// differences already respect symmetry constraints.
pub fn apply_dirichlet_ansatz<B: Backend>(
    raw_out:  Tensor<B, 2>,
    coords:   &Tensor<B, 2>,
    symmetry: SymmetryMode,
    k:        f32,
) -> Tensor<B, 2> {
    match symmetry {
        SymmetryMode::QuarterSymm => {
            let m = raw_out.dims()[0];
            let output_dim = raw_out.dims()[1];

            let x_col = coords.clone().slice([0..m, 0..1]); // [M, 1]
            let y_col = coords.clone().slice([0..m, 1..2]); // [M, 1]

            // D_x = tanh(k*(x_norm+1)) — exactly 0 at x_norm=-1
            let dx: Tensor<B, 2> = x_col.add_scalar(1.0_f32).mul_scalar(k).tanh();
            // D_y = tanh(k*(y_norm+1)) — exactly 0 at y_norm=-1
            let dy: Tensor<B, 2> = y_col.add_scalar(1.0_f32).mul_scalar(k).tanh();

            let u: Tensor<B, 2> = raw_out.clone().slice([0..m, 0..1]);
            let v: Tensor<B, 2> = raw_out.clone().slice([0..m, 1..2]);

            if output_dim == 5 {
                // mDEM: pass stress columns (2..5) unchanged — their values are direct outputs.
                let s_xx = raw_out.clone().slice([0..m, 2..3]);
                let s_yy = raw_out.clone().slice([0..m, 3..4]);
                let s_xy = raw_out.slice([0..m, 4..5]);
                Tensor::cat(vec![u * dx, v * dy, s_xx, s_yy, s_xy], 1)
            } else {
                // Plain DEM: zero w (plane stress; keep in autodiff graph).
                let w: Tensor<B, 2> = raw_out.slice([0..m, 2..3]);
                let w_zero: Tensor<B, 2> = w.mul_scalar(0.0_f32);
                Tensor::cat(vec![u * dx, v * dy, w_zero], 1)
            }
        }
        SymmetryMode::Full => raw_out,
    }
}
