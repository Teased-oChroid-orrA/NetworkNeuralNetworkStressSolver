//! Differentiation-backend abstraction - General-PINN Phase 2 remediation (issue #61, epic
//! P2-02, "DifferentialOperator abstraction").
//!
//! This codebase's physics layer (`fd_stencil.rs`) computes every spatial derivative (strain,
//! Hessian) via ONE method: central-difference finite differences. That's a real gap this epic
//! closes with a genuine second, independently-verified backend for FIRST derivatives
//! (automatic differentiation through the network's own computation graph, w.r.t. its INPUT
//! coordinates - a different differentiation target than the training loop's own per-step
//! `.backward()`, which differentiates w.r.t. the network's WEIGHTS) plus an explicit, recorded
//! backend-selection policy (`DerivativeBackendPolicy`) so a caller never silently gets one
//! method when it asked for another.
//!
//! **Second derivatives (Hessian) are FD-only, and this is a confirmed backend limitation, not
//! an unimplemented stub**: `burn-autodiff` 0.21 (the version this workspace depends on) has no
//! higher-order/nested-autodiff support - confirmed by inspecting its vendored source directly
//! (`~/.cargo/registry/.../burn-autodiff-0.21.0/src/`), which contains no "double backward" /
//! "create_graph" / second-order mechanism anywhere. A first `.backward()` call consumes the
//! tracked graph; the resulting gradient tensor is returned on the INNER (non-autodiff) backend
//! and cannot be differentiated again without re-establishing a fresh graph rooted at a NEW
//! leaf - which no longer has any connection to the original input, so a true `d²u/dx²` via
//! nested AD is not obtainable this way. Per issue #61 §1.3 ("no dead abstractions... completion
//! requires proof the live solver actually uses the implementation") and §6 ("SHALL NOT mark
//! TODO/stubs as complete"), this module does NOT ship a fake/zero AD-Hessian function - it
//! documents the limitation honestly instead. `DerivativeBackend::Ad` therefore applies to
//! [`ad_strain`] (first derivatives) only; Hessian computation remains
//! `fd_stencil::compute_hessian` regardless of the selected backend.
//!
//! Per issue #61 §1.4 ("no destructive refactors... identify call sites before replacement")
//! and P2-15's staged migration order ("add abstractions" BEFORE "adapt existing paths"), this
//! module does NOT invasively rewrite `training_core::compute_domain_forwards` (the shared
//! forward pass every problem type - Kirsch, pin-lug, and the generic plate path - depends on)
//! in this same slice. `fd_stencil::compute_strains` remains the FD implementation; [`ad_strain`]
//! is a genuine, cross-validated alternative a caller CAN route through today (proven via its
//! own tests against the exact manufactured field `crate::manufactured` already established in
//! a prior pass), not yet the path `compute_domain_forwards` itself calls.
//!
//! **Issue #62 PH3-06's own real, load-bearing finding: [`ad_strain`] can NEVER become the live
//! TRAINING-loss derivative backend with this burn-autodiff version, for a concrete, verifiable
//! (type-signature-level, not just theoretical) reason.** `ad_strain` computes its result via
//! `pi.backward()` + `pts.grad(&grads)` - burn's gradient-RETRIEVAL API, which returns the
//! gradient VALUE on `Tensor<B::InnerBackend, _>` (this module's own top comment already
//! documents why: burn-autodiff 0.21 has no "double backward"/`create_graph` mechanism, so a
//! retrieved gradient is necessarily detached from any further autodiff graph). `LossTerm::
//! compute()` (`problem.rs`) MUST return `Tensor<B, 1>` - connected to the WEIGHT-autodiff graph
//! `step_physics_multi`'s own outer `.backward()` differentiates through to update the model.
//! `Tensor<B::InnerBackend, 1>` and `Tensor<B, 1>` are DIFFERENT ASSOCIATED TYPES for a real
//! `AutodiffBackend` (`B::InnerBackend != B`) - passing one where the other is required is a
//! compile error, not a subtle runtime bug. There is therefore no way to route `InteriorEnergy
//! Term`/`ExternalWorkTerm`/`EquilibriumTerm`'s live strain computation through `ad_strain`
//! without first solving nested autodiff itself (a burn-upstream limitation, not something this
//! codebase can work around). [`ad_fd_strain_agreement`] is the honest, ACHIEVABLE version of
//! "migrate DifferentialOperator into live consumers" this constraint still permits: a live
//! DIAGNOSTIC cross-check of AD against FD at the model's CURRENT training state (not just a
//! synthetic manufactured field), proving real live use of the AD backend without claiming it
//! replaces FD as the optimization ingredient.

use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::fd_stencil::{compute_strains, FdConfig};

/// Which numerical method actually supplies a derivative - explicit and recorded, never
/// silently switched (issue #61 P2-02's "No silent backend switching").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeBackend {
    /// Central-difference finite differences (`fd_stencil::assemble_stencil`/`compute_strains`/
    /// `compute_hessian`) - this codebase's original, pre-remediation mechanism. The only
    /// backend available for second derivatives (Hessian) - see this module's own doc comment.
    Fd,
    /// Automatic differentiation through the network's own computation graph, w.r.t. its INPUT
    /// coordinates. First derivatives (strain) only.
    Ad,
    /// Exact, hand-derived closures - only meaningful for a `manufactured::ManufacturedField`
    /// (a real trained network has no closed-form derivative to fall back on).
    Analytic,
}

/// Records which backend is PRIMARY, which (if any) VERIFIES it, and what happens if the
/// primary is unavailable - issue #61 P2-02's explicit "hybrid policy" requirement:
/// ```text
/// primary = AD
/// verification = FD
/// known_expression = Analytic
/// fallback = declared policy
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DerivativeBackendPolicy {
    pub primary: DerivativeBackend,
    pub verification: Option<DerivativeBackend>,
    pub fallback: Option<DerivativeBackend>,
}

impl DerivativeBackendPolicy {
    /// This codebase's own pre-remediation behavior everywhere: FD only, no cross-check, no
    /// fallback (there was only ever one backend before this epic existed).
    pub const FD_ONLY: Self = Self { primary: DerivativeBackend::Fd, verification: None, fallback: None };
}

// ─── Scalar-field derivatives (issue #61's own literal acceptance-test shape) ────────────────

/// A scalar field's first and second derivatives at a point - General-PINN P2-02's own literal
/// acceptance target (`f(x,y)=x^2+3xy+2y^2`). Deliberately separate from the tensor-batch
/// strain machinery below (which differentiates a NETWORK's vector output `u,v` over many
/// points at once): a plain scalar-field check, generic over any `Fn(f64,f64)->f64`, with no
/// burn/Tensor/network dependency at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarDerivatives {
    pub fx: f64,
    pub fy: f64,
    pub fxx: f64,
    pub fyy: f64,
    pub fxy: f64,
}

/// Central-difference scalar derivatives - the same central-difference MATH `fd_stencil`'s
/// tensor version uses (`compute_strains`'s `(f(x+h)-f(x-h))/(2h)`, `compute_hessian`'s
/// `(f(x+h)-2f(x)+f(x-h))/h²` and mixed-partial four-corner formula), expressed directly over a
/// plain closure instead of a burn `Tensor` batch.
pub fn fd_scalar_derivatives(f: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> ScalarDerivatives {
    let fx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
    let fy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
    let fxx = (f(x + h, y) - 2.0 * f(x, y) + f(x - h, y)) / (h * h);
    let fyy = (f(x, y + h) - 2.0 * f(x, y) + f(x, y - h)) / (h * h);
    let fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4.0 * h * h);
    ScalarDerivatives { fx, fy, fxx, fyy, fxy }
}

// ─── Tensor-batch AD strain (the genuinely new capability) ───────────────────────────────────

/// Autodiff-based strain, cross-validating against `fd_stencil::compute_strains`. `forward`
/// computes network-style output `[N,2+]` (columns 0,1 = u,v) from a `[N,3]` normalized-
/// coordinate batch (x_norm, y_norm, z=0) - the SAME input/output column convention
/// `ElasticityNet::forward` and `fd_stencil::assemble_stencil`'s stencil rows already use, so
/// this is a real drop-in alternative, not a parallel convention.
///
/// Uses the batched "sum-trick": since each output row depends ONLY on its own input row (a
/// pointwise network evaluation with no cross-row coupling), `d(sum(u))/d(pts)[i,0]` recovers
/// `du_i/dx_i` for every row `i` simultaneously from ONE backward pass, rather than needing N
/// independent passes (the standard technique for computing a batched Jacobian diagonal via
/// autodiff without materializing the full N×N Jacobian). `u` and `v` need their OWN backward
/// pass each (burn does not support extracting two independent gradients-of-different-outputs
/// from one `.backward()` call), so `forward` is evaluated twice.
pub fn ad_strain<B: AutodiffBackend>(
    forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    pts_norm: &Tensor<B, 2>,
    fd: &FdConfig,
) -> (Tensor<B::InnerBackend, 1>, Tensor<B::InnerBackend, 1>, Tensor<B::InnerBackend, 1>) {
    let n = pts_norm.dims()[0];

    let pts_u = pts_norm.clone().require_grad();
    let u = forward(pts_u.clone()).slice([0..n, 0..1]).reshape([n]);
    let grads_u = u.sum().backward();
    let du = pts_u.grad(&grads_u).expect("ad_strain: input coordinates must be part of the autodiff graph (u)");

    let pts_v = pts_norm.clone().require_grad();
    let v = forward(pts_v.clone()).slice([0..n, 1..2]).reshape([n]);
    let grads_v = v.sum().backward();
    let dv = pts_v.grad(&grads_v).expect("ad_strain: input coordinates must be part of the autodiff graph (v)");

    let du_dx = du.clone().slice([0..n, 0..1]).reshape([n]);
    let du_dy = du.slice([0..n, 1..2]).reshape([n]);
    let dv_dx = dv.clone().slice([0..n, 0..1]).reshape([n]);
    let dv_dy = dv.slice([0..n, 1..2]).reshape([n]);

    // Physical scaling: same `sx`/`sy` coordinate-mapping factors `compute_strains` applies -
    // derivative-in-normalized-coords * mapping-scale = physical derivative. No `1/(2h)` factor
    // here (unlike `compute_strains`) since AD gives the EXACT derivative directly, not a
    // finite difference of it.
    let eps_xx = du_dx.mul_scalar(fd.sx);
    let eps_yy = dv_dy.mul_scalar(fd.sy);
    let eps_xy = (du_dy.mul_scalar(fd.sy) + dv_dx.mul_scalar(fd.sx)).mul_scalar(0.5);
    (eps_xx, eps_yy, eps_xy)
}

/// FD strain via the SAME `forward`/`pts_norm`/`fd` inputs `ad_strain` takes - a thin adapter
/// over the pre-existing `fd_stencil::assemble_stencil`/`compute_strains` pair, so a caller can
/// genuinely swap `ad_strain`/`fd_strain_via` for each other behind one call site.
pub fn fd_strain_via<B: burn::tensor::backend::Backend>(
    forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    pts_norm: &Tensor<B, 2>,
    fd: &FdConfig,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let n = pts_norm.dims()[0];
    let stencil = crate::fd_stencil::assemble_stencil::<B>(pts_norm, fd, device);
    let out = forward(stencil);
    compute_strains::<B>(out, n, fd)
}

/// Issue #62 PH3-06's own real, live-use result: how closely AD and FD agree on FIRST
/// derivatives (strain) at a REAL model's CURRENT training state - see this module's own
/// "PH3-06" doc comment section above for exactly why this is a DIAGNOSTIC cross-check, never a
/// live training-loss ingredient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdFdStrainAgreement {
    pub eps_xx_rms_relative_diff: f64,
    pub eps_yy_rms_relative_diff: f64,
    pub eps_xy_rms_relative_diff: f64,
}

fn rms_relative_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().max(1) as f64;
    let sq_diff: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
    let sq_b: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
    let rms_diff = (sq_diff / n).sqrt();
    let rms_b = (sq_b / n).sqrt();
    if rms_b > 1e-12 { rms_diff / rms_b } else { rms_diff }
}

/// Issue #62 PH3-06: runs BOTH [`ad_strain`] and [`fd_strain_via`] against the SAME `forward`/
/// `pts_norm`/`fd` inputs and reports how closely they agree - the real, live-model-state
/// counterpart of this module's own `ad_strain_matches_fd_strain_on_the_same_manufactured_
/// field` unit test, callable against an ACTUAL currently-training model rather than only a
/// synthetic manufactured field. `forward` MUST be built from an identity-ansatz domain's raw
/// network output only (no Dirichlet-ansatz scaling baked in) - see this module's own doc
/// comment on why AD cannot be used for a non-identity ansatz (Kirsch's `QuarterSymmAnsatz`)
/// without also making `ansatz.eval` itself a differentiable tensor operation, which does not
/// exist and is out of this item's scope.
pub fn ad_fd_strain_agreement<B: AutodiffBackend>(
    forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    pts_norm: &Tensor<B, 2>,
    fd: &FdConfig,
) -> AdFdStrainAgreement {
    let device = pts_norm.device();
    let (ad_xx, ad_yy, ad_xy) = ad_strain::<B>(&forward, pts_norm, fd);
    let (fd_xx, fd_yy, fd_xy) = fd_strain_via::<B>(&forward, pts_norm, fd, &device);
    let v_inner = |t: Tensor<B::InnerBackend, 1>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
    let v_outer = |t: Tensor<B, 1>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
    AdFdStrainAgreement {
        eps_xx_rms_relative_diff: rms_relative_diff(&v_inner(ad_xx), &v_outer(fd_xx)),
        eps_yy_rms_relative_diff: rms_relative_diff(&v_inner(ad_yy), &v_outer(fd_yy)),
        eps_xy_rms_relative_diff: rms_relative_diff(&v_inner(ad_xy), &v_outer(fd_xy)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Issue #61 P2-02's own literal acceptance test ────────────────────────────────────
    // f(x,y) = x^2 + 3xy + 2y^2 -> fx=2x+3y, fy=3x+4y, fxx=2, fyy=4, fxy=3 (exact, everywhere).

    #[test]
    fn fd_scalar_derivatives_matches_analytic_values_for_the_acceptance_polynomial() {
        let f = |x: f64, y: f64| x * x + 3.0 * x * y + 2.0 * y * y;
        let (x0, y0) = (1.3, -0.7);
        let d = fd_scalar_derivatives(&f, x0, y0, 1e-3);
        let tol = 1e-6; // central FD is exact (no truncation error) for a degree-2 polynomial
        assert!((d.fx - (2.0 * x0 + 3.0 * y0)).abs() < tol, "{d:?}");
        assert!((d.fy - (3.0 * x0 + 4.0 * y0)).abs() < tol, "{d:?}");
        assert!((d.fxx - 2.0).abs() < tol, "{d:?}");
        assert!((d.fyy - 4.0).abs() < tol, "{d:?}");
        assert!((d.fxy - 3.0).abs() < tol, "{d:?}");
    }

    #[test]
    fn fd_scalar_derivatives_matches_manufactured_field_exact_strain_formula() {
        // Cross-check against `crate::manufactured::ManufacturedField::quadratic`'s own exact
        // strain formula for the SAME field shape, proving the two independently-written
        // "exact answer" sources agree.
        use crate::manufactured::ManufacturedField;
        let field = ManufacturedField::quadratic(5.0, -3.0, 2.0, -1.5);
        let u = |x: f64, y: f64| 5.0 * x * x + 2.0 * x * y;
        let (x0, y0) = (0.03, 0.02);
        let d = fd_scalar_derivatives(&u, x0, y0, 1e-3);
        let (exx_exact, _, _) = field.exact_strain(x0, y0);
        assert!((d.fx - exx_exact).abs() < 1e-9, "u_x={} vs exact eps_xx={}", d.fx, exx_exact);
    }

    // ─── AD strain vs. FD strain vs. exact (manufactured field) ───────────────────────────

    type TB = burn::backend::Autodiff<crate::training_core::BInner>;

    /// Builds the `[N,2]` (u,v) output tensor for `ManufacturedField::quadratic(a,b,c,d)`
    /// directly from burn tensor ops on the `[N,3]` (x,y,z) input - a differentiable graph
    /// representing the exact same field `crate::manufactured`'s f64 closures compute, so AD
    /// and the manufactured field's own exact formulas can be cross-validated against a THIRD,
    /// independent implementation of the same math.
    /// `pts` carries NORMALIZED coordinates (x_norm, y_norm) - the same convention every real
    /// network input uses (`fd_stencil::assemble_stencil`'s own stencil rows). This forward
    /// function denormalizes to PHYSICAL x,y first (`x_phys = x_norm * half_w`), then evaluates
    /// the manufactured quadratic field in physical units, matching how a real trained network
    /// conceptually maps normalized input to physical-unit output.
    fn quadratic_field_forward<B: burn::tensor::backend::Backend>(
        a: f64, b: f64, c: f64, d: f64, half_w: f64, half_h: f64,
    ) -> impl Fn(Tensor<B, 2>) -> Tensor<B, 2> {
        move |pts: Tensor<B, 2>| {
            let n = pts.dims()[0];
            let x = pts.clone().slice([0..n, 0..1]).reshape([n]).mul_scalar(half_w);
            let y = pts.slice([0..n, 1..2]).reshape([n]).mul_scalar(half_h);
            let u = x.clone().mul_scalar(a).mul(x.clone()) + x.clone().mul(y.clone()).mul_scalar(c);
            let v = y.clone().mul_scalar(b).mul(y.clone()) + x.mul(y).mul_scalar(d);
            Tensor::cat(vec![u.reshape([n, 1]), v.reshape([n, 1])], 1)
        }
    }

    #[test]
    fn ad_strain_matches_exact_manufactured_strain() {
        use crate::manufactured::ManufacturedField;
        let device = Default::default();
        let (a, b, c, d) = (5.0_f64, -3.0_f64, 2.0_f64, -1.5_f64);
        let field = ManufacturedField::quadratic(a, b, c, d);
        let (x0, y0) = (0.03_f64, 0.02_f64);
        let (half_w, half_h) = (0.1_f64, 0.1_f64);
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);

        let (x0_norm, y0_norm) = (x0 / half_w, y0 / half_h);
        let pts = Tensor::<TB, 2>::from_data(
            burn::tensor::TensorData::new(vec![x0_norm as f32, y0_norm as f32, 0.0f32], vec![1, 3]), &device,
        );
        let forward = quadratic_field_forward::<TB>(a, b, c, d, half_w, half_h);
        let (exx, eyy, exy) = ad_strain::<TB>(forward, &pts, &fd);
        let get = |t: Tensor<crate::training_core::BInner, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let (exx_v, eyy_v, exy_v) = (get(exx), get(eyy), get(exy));
        let (exx_exact, eyy_exact, exy_exact) = field.exact_strain(x0, y0);

        let tol = 1e-4; // relative, generous for f32 tensor round-trips
        assert!((exx_v - exx_exact).abs() / exx_exact.abs() < tol, "exx {exx_v} vs exact {exx_exact}");
        assert!((eyy_v - eyy_exact).abs() / eyy_exact.abs() < tol, "eyy {eyy_v} vs exact {eyy_exact}");
        assert!((exy_v - exy_exact).abs() / exy_exact.abs() < tol, "exy {exy_v} vs exact {exy_exact}");
    }

    #[test]
    fn ad_strain_matches_fd_strain_on_the_same_manufactured_field() {
        let device = Default::default();
        let (a, b, c, d) = (5.0_f64, -3.0_f64, 2.0_f64, -1.5_f64);
        let (x0, y0) = (0.03_f64, 0.02_f64);
        let (half_w, half_h) = (0.1_f64, 0.1_f64);
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);

        let (x0_norm, y0_norm) = (x0 / half_w, y0 / half_h);
        let pts_ad = Tensor::<TB, 2>::from_data(
            burn::tensor::TensorData::new(vec![x0_norm as f32, y0_norm as f32, 0.0f32], vec![1, 3]), &device,
        );
        let (exx_ad, eyy_ad, exy_ad) = ad_strain::<TB>(quadratic_field_forward::<TB>(a, b, c, d, half_w, half_h), &pts_ad, &fd);

        let pts_fd = Tensor::<crate::training_core::BInner, 2>::from_data(
            burn::tensor::TensorData::new(vec![x0_norm as f32, y0_norm as f32, 0.0f32], vec![1, 3]), &device,
        );
        let (exx_fd, eyy_fd, exy_fd) = fd_strain_via::<crate::training_core::BInner>(
            quadratic_field_forward::<crate::training_core::BInner>(a, b, c, d, half_w, half_h), &pts_fd, &fd, &device,
        );

        let get = |t: Tensor<crate::training_core::BInner, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let (exx_ad_v, eyy_ad_v, exy_ad_v) = (get(exx_ad), get(eyy_ad), get(exy_ad));
        let (exx_fd_v, eyy_fd_v, exy_fd_v) = (get(exx_fd), get(eyy_fd), get(exy_fd));
        let tol = 1e-3; // relative - FD carries real (if small) truncation/cancellation error AD doesn't
        assert!((exx_ad_v - exx_fd_v).abs() / exx_fd_v.abs() < tol, "{exx_ad_v} vs {exx_fd_v}");
        assert!((eyy_ad_v - eyy_fd_v).abs() / eyy_fd_v.abs() < tol, "{eyy_ad_v} vs {eyy_fd_v}");
        assert!((exy_ad_v - exy_fd_v).abs() / exy_fd_v.abs() < tol, "{exy_ad_v} vs {exy_fd_v}");
    }

    // ─── Issue #62 PH3-06: live AD-vs-FD cross-validation diagnostic ───────────────────────

    #[test]
    fn ad_fd_strain_agreement_reports_a_small_relative_difference_on_a_smooth_manufactured_field() {
        let device = Default::default();
        let (a, b, c, d) = (5.0_f64, -3.0_f64, 2.0_f64, -1.5_f64);
        let (half_w, half_h) = (0.1_f64, 0.1_f64);
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);
        // A batch of several points (not just one), matching how this would actually be called
        // against a real interior collocation point set.
        let pts_data: Vec<f32> = vec![
            0.1, 0.2, 0.0,  -0.3, 0.4, 0.0,  0.5, -0.1, 0.0,  -0.2, -0.4, 0.0,
        ];
        let pts = Tensor::<TB, 2>::from_data(burn::tensor::TensorData::new(pts_data, vec![4, 3]), &device);
        let forward = quadratic_field_forward::<TB>(a, b, c, d, half_w, half_h);
        let agreement = ad_fd_strain_agreement::<TB>(forward, &pts, &fd);
        let tol = 1e-3; // same real FD truncation/cancellation tolerance as the single-point test above
        assert!(agreement.eps_xx_rms_relative_diff < tol, "{agreement:?}");
        assert!(agreement.eps_yy_rms_relative_diff < tol, "{agreement:?}");
        assert!(agreement.eps_xy_rms_relative_diff < tol, "{agreement:?}");
    }

    #[test]
    fn rms_relative_diff_is_zero_for_identical_slices_and_positive_for_differing_ones() {
        assert_eq!(rms_relative_diff(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
        assert!(rms_relative_diff(&[1.1, 2.0, 3.0], &[1.0, 2.0, 3.0]) > 0.0);
    }
}
