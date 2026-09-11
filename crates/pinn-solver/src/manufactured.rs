//! Manufactured-solution test fixtures - General-PINN architecture recommendations §23
//! ("manufactured-solution framework"), generalizing this codebase's own pre-existing ad-hoc
//! technique (`energy::tests::hessian_recovers_exact_second_derivatives_of_a_manufactured_
//! quadratic_field`, which hand-built a quadratic displacement field, its exact derivatives,
//! and a stencil-offset tensor inline) into a reusable module.
//!
//! This codebase has no symbolic differentiation engine (no CAS), so a
//! [`ManufacturedField`] is not "solve for the field given a target PDE residual" - it is the
//! narrower, honest thing this codebase actually needs and already used once ad hoc: a
//! displacement field `u(x,y), v(x,y)` paired with its EXACT partial derivatives, supplied by
//! hand by the caller (each concrete constructor documents its own by-hand derivation), used to
//! verify a numerical differentiation path (`fd_stencil::compute_strains`/`compute_hessian`)
//! against a KNOWN-exact answer instead of a hardcoded literal. `verify_strain`/`verify_hessian`
//! below build the same "raw stencil output" tensor the ad-hoc test built by hand, so a new
//! derivative-verification test is a few lines, not a reimplementation of stencil-row
//! bookkeeping every time.

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// A displacement field with hand-supplied exact first and second partial derivatives. See the
/// module doc comment for why "exact" here means "the caller derived it by hand and asserts
/// it's correct," not "symbolically differentiated by this code."
pub struct ManufacturedField {
    pub u: Box<dyn Fn(f64, f64) -> f64>,
    pub v: Box<dyn Fn(f64, f64) -> f64>,
    pub u_x: Box<dyn Fn(f64, f64) -> f64>,
    pub u_y: Box<dyn Fn(f64, f64) -> f64>,
    pub v_x: Box<dyn Fn(f64, f64) -> f64>,
    pub v_y: Box<dyn Fn(f64, f64) -> f64>,
    pub u_xx: Box<dyn Fn(f64, f64) -> f64>,
    pub u_yy: Box<dyn Fn(f64, f64) -> f64>,
    pub u_xy: Box<dyn Fn(f64, f64) -> f64>,
    pub v_xx: Box<dyn Fn(f64, f64) -> f64>,
    pub v_yy: Box<dyn Fn(f64, f64) -> f64>,
    pub v_xy: Box<dyn Fn(f64, f64) -> f64>,
}

impl ManufacturedField {
    /// `u(x,y) = a·x² + c·xy`, `v(x,y) = b·y² + d·xy` - nonzero, distinct curvature in every
    /// one of the 6 Hessian components (the exact field `hessian_recovers_exact_second_
    /// derivatives_of_a_manufactured_quadratic_field` already used ad hoc). Central-difference
    /// FD is EXACT (no truncation error) for both the first AND second derivative of any
    /// polynomial of degree <= 2 in each stencil direction - the same field verifies both
    /// `compute_strains` (first-order) and `compute_hessian` (second-order) to near machine
    /// precision, not just "close".
    pub fn quadratic(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            u: Box::new(move |x, y| a * x * x + c * x * y),
            v: Box::new(move |x, y| b * y * y + d * x * y),
            u_x: Box::new(move |x, y| 2.0 * a * x + c * y),
            u_y: Box::new(move |x, _y| c * x),
            v_x: Box::new(move |_x, y| d * y),
            v_y: Box::new(move |x, y| 2.0 * b * y + d * x),
            u_xx: Box::new(move |_x, _y| 2.0 * a),
            u_yy: Box::new(move |_x, _y| 0.0),
            u_xy: Box::new(move |_x, _y| c),
            v_xx: Box::new(move |_x, _y| 0.0),
            v_yy: Box::new(move |_x, _y| 2.0 * b),
            v_xy: Box::new(move |_x, _y| d),
        }
    }

    /// Exact `(ε_xx, ε_yy, ε_xy)` at `(x,y)` from this field's own supplied derivatives -
    /// `ε_xx=u_x`, `ε_yy=v_y`, `ε_xy=½(u_y+v_x)` (tensor convention, matching `fd_stencil::
    /// compute_strains`'s own formula).
    pub fn exact_strain(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let exx = (self.u_x)(x, y);
        let eyy = (self.v_y)(x, y);
        let exy = 0.5 * ((self.u_y)(x, y) + (self.v_x)(x, y));
        (exx, eyy, exy)
    }

    /// Exact `(u_xx, u_yy, u_xy, v_xx, v_yy, v_xy)` at `(x,y)`.
    pub fn exact_hessian(&self, x: f64, y: f64) -> (f64, f64, f64, f64, f64, f64) {
        (
            (self.u_xx)(x, y), (self.u_yy)(x, y), (self.u_xy)(x, y),
            (self.v_xx)(x, y), (self.v_yy)(x, y), (self.v_xy)(x, y),
        )
    }
}

fn read_scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_data().to_vec::<f32>().unwrap()[0] as f64
}

/// Numerically differentiate `field` at physical point `(x0,y0)` via `fd_stencil::
/// compute_strains`, bypassing the network entirely (the stencil "output" is the field's own
/// exact values at each offset point, exactly what a perfectly-trained network would produce) -
/// isolates the FD MATH from network approximation error, the same isolation `hessian_
/// recovers_exact_second_derivatives_...` already established for the Hessian path. Row order
/// matches `assemble_stencil` exactly: centre, x+hx, x-hx, y+hy, y-hy.
pub fn verify_strain<B: Backend>(
    field: &ManufacturedField,
    half_w: f64,
    half_h: f64,
    fd_h: f32,
    x0: f64,
    y0: f64,
    device: &B::Device,
) -> (f64, f64, f64) {
    use crate::fd_stencil::{compute_strains, FdConfig};

    let hx_phys = fd_h as f64 * half_w;
    let hy_phys = fd_h as f64 * half_h;
    let offsets = [(0.0, 0.0), (hx_phys, 0.0), (-hx_phys, 0.0), (0.0, hy_phys), (0.0, -hy_phys)];
    let mut data = Vec::with_capacity(offsets.len() * 2);
    for &(dx, dy) in &offsets {
        let (x, y) = (x0 + dx, y0 + dy);
        data.extend_from_slice(&[(field.u)(x, y) as f32, (field.v)(x, y) as f32]);
    }
    let raw = Tensor::<B, 2>::from_data(TensorData::new(data, vec![offsets.len(), 2]), device);

    let fd = FdConfig::new(fd_h, 2.0 * half_w, 2.0 * half_h);
    let (exx, eyy, exy) = compute_strains::<B>(raw, 1, &fd);
    (read_scalar(exx), read_scalar(eyy), read_scalar(exy))
}

/// Same technique as [`verify_strain`], for `fd_stencil::compute_hessian`'s 9-point stencil -
/// applies `hessian_fd_config`'s widened step (production's own safety multiplier - see that
/// function's doc comment) so this is a true regression guard for what `compute_domain_
/// forwards` actually does at the Hessian path, not a separately-chosen step that could drift
/// out of sync with it. Row order matches `assemble_second_order_stencil` exactly.
pub fn verify_hessian<B: Backend>(
    field: &ManufacturedField,
    half_w: f64,
    half_h: f64,
    base_fd_h: f32,
    x0: f64,
    y0: f64,
    device: &B::Device,
) -> (f64, f64, f64, f64, f64, f64) {
    use crate::fd_stencil::{compute_hessian, hessian_fd_config, FdConfig};

    let fd_h = hessian_fd_config(&FdConfig::new(base_fd_h, 2.0 * half_w, 2.0 * half_h)).hx;
    let hx_phys = fd_h as f64 * half_w;
    let hy_phys = fd_h as f64 * half_h;
    let offsets = [
        (0.0, 0.0), (hx_phys, 0.0), (-hx_phys, 0.0), (0.0, hy_phys), (0.0, -hy_phys),
        (hx_phys, hy_phys), (hx_phys, -hy_phys), (-hx_phys, hy_phys), (-hx_phys, -hy_phys),
    ];
    let mut data = Vec::with_capacity(offsets.len() * 2);
    for &(dx, dy) in &offsets {
        let (x, y) = (x0 + dx, y0 + dy);
        data.extend_from_slice(&[(field.u)(x, y) as f32, (field.v)(x, y) as f32]);
    }
    let raw = Tensor::<B, 2>::from_data(TensorData::new(data, vec![offsets.len(), 2]), device);

    let fd = FdConfig::new(fd_h, 2.0 * half_w, 2.0 * half_h);
    let (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy) = compute_hessian::<B>(raw, 1, &fd);
    (
        read_scalar(u_xx), read_scalar(u_yy), read_scalar(u_xy),
        read_scalar(v_xx), read_scalar(v_yy), read_scalar(v_xy),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_core::BInner as TB;

    #[test]
    fn quadratic_exact_strain_matches_hand_derived_formula() {
        let field = ManufacturedField::quadratic(5.0, -3.0, 2.0, -1.5);
        let (x, y) = (0.03, 0.02);
        let (exx, eyy, exy) = field.exact_strain(x, y);
        // u_x = 2a x + c y, v_y = 2b y + d x, exy = 0.5*(u_y + v_x) = 0.5*(c x + d y)
        assert!((exx - (2.0 * 5.0 * x + 2.0 * y)).abs() < 1e-12);
        assert!((eyy - (2.0 * -3.0 * y + -1.5 * x)).abs() < 1e-12);
        assert!((exy - 0.5 * (2.0 * x + -1.5 * y)).abs() < 1e-12);
    }

    #[test]
    fn quadratic_exact_hessian_matches_hand_derived_constants() {
        let field = ManufacturedField::quadratic(5.0, -3.0, 2.0, -1.5);
        let (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy) = field.exact_hessian(0.03, 0.02);
        assert_eq!((u_xx, u_yy, u_xy, v_xx, v_yy, v_xy), (10.0, 0.0, 2.0, 0.0, -6.0, -1.5));
    }

    #[test]
    fn verify_strain_recovers_exact_first_derivatives_of_a_manufactured_quadratic_field() {
        let device = Default::default();
        let (half_w, half_h) = (0.1_f64, 0.1_f64);
        let field = ManufacturedField::quadratic(5.0, -3.0, 2.0, -1.5);
        let (x0, y0) = (0.03_f64, 0.02_f64);

        let (exx, eyy, exy) = verify_strain::<TB>(&field, half_w, half_h, 1e-3, x0, y0, &device);
        let (exx_exact, eyy_exact, exy_exact) = field.exact_strain(x0, y0);

        let tol = 1e-3; // relative
        assert!((exx - exx_exact).abs() / exx_exact.abs() < tol, "exx {exx} vs {exx_exact}");
        assert!((eyy - eyy_exact).abs() / eyy_exact.abs() < tol, "eyy {eyy} vs {eyy_exact}");
        assert!((exy - exy_exact).abs() / exy_exact.abs() < tol, "exy {exy} vs {exy_exact}");
    }
}
