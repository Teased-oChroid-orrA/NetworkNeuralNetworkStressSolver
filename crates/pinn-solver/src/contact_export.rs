//! Post-processing/export for the pin-in-lug contact problem: sample the trained LUG
//! network at the contact interface (r = lug hole radius) over the physically loaded half
//! of the hole boundary (theta in [-pi/2, pi/2] — see `pinlug_problem.rs`'s module doc
//! comment: "the pin only bears on roughly half the hole boundary"), compute the normal
//! contact (radial) stress via `crate::signorini::decompose_radial`, and write the
//! resulting (theta, sigma_rr) profile to a CSV file.
//!
//! Reuses the exact same raw-output -> physical-stress scaling `step_physics_multi` applies
//! for mDEM domains (`training_core.rs`: stress columns 2..5 multiplied by `config.load.px`
//! [Pa], displacement columns by `u_ref` [m]) so the exported values are consistent with
//! what training actually optimized against, not a re-derived convention.

use std::f64::consts::PI;
use std::io::Write;
use std::path::Path;

use burn::tensor::backend::Backend;

use pinn_core::geometry::{GeometryConfig, HoleType};
use pinn_core::units::KSI_TO_PA;

use crate::fd_stencil::norm_pts_to_tensor;
use crate::network::{fwd, ElasticityNet};
use crate::signorini::decompose_radial;

/// Number of interface samples over theta in [-pi/2, pi/2] for the exported contact-pressure
/// profile. Within the "64-128" range suggested for a usable profile.
pub const CONTACT_EXPORT_N_SAMPLES: usize = 96;

/// Default output CSV filename (current working directory) — no existing `--output`/
/// output-dir convention was found in `headless.rs`/`pinn.env`/`pinn-app/src/main.rs`, so
/// this is the reasonable default named in the task brief.
pub const DEFAULT_CONTACT_EXPORT_PATH: &str = "pinlug_contact_pressure.csv";

/// One exported (theta, sigma_rr) sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPressureSample {
    pub theta_deg: f64,
    pub theta_rad: f64,
    pub sigma_rr_ksi: f64,
}

/// Sample the trained LUG network's stress field at its own hole boundary (r = lug hole
/// radius) over theta in [-pi/2, pi/2] (the pin-loaded half, per `pinlug_problem.rs`'s
/// module doc comment), and compute the normal contact stress sigma_rr(theta) via
/// `decompose_radial`.
///
/// `lug_geom` must have a circular hole (panics otherwise — a malformed lug geometry is a
/// programming error, not a runtime condition to recover from). `px_pa` is the same
/// mDEM stress-column scale `step_physics_multi` uses (`config.load.px`, equal to
/// `PinLugProblem::equivalent_traction_pa()` for the pin-in-lug default config).
pub fn sample_contact_pressure<Bk: Backend>(
    lug_model: &ElasticityNet<Bk>,
    lug_geom: &GeometryConfig,
    px_pa: f64,
    n_samples: usize,
    device: &Bk::Device,
) -> Vec<ContactPressureSample> {
    let HoleType::Circular { radius: r_lug } = lug_geom.hole else {
        panic!("sample_contact_pressure: lug geometry must have a circular hole");
    };

    let n = n_samples.max(1);
    let thetas: Vec<f64> = (0..n)
        .map(|i| {
            if n == 1 {
                0.0
            } else {
                -PI / 2.0 + PI * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();

    let norm_pts: Vec<[f32; 2]> = thetas.iter()
        .map(|&theta| normalize_point_generic(r_lug * theta.cos(), r_lug * theta.sin(), lug_geom))
        .collect();

    let pts_t = norm_pts_to_tensor::<Bk>(&norm_pts, device);
    // Plain-DEM/mDEM raw forward pass — pin-in-lug uses n_fourier=0 (see
    // `PinLugProblem::convergence_metric`'s doc comment for the same convention).
    const N_FOURIER: usize = 0;
    let raw = fwd::<Bk>(lug_model, pts_t, N_FOURIER, device);

    let sxx_raw: Vec<f32> = raw.clone().slice([0..n, 2..3]).reshape([n]).into_data().to_vec().unwrap_or_default();
    let syy_raw: Vec<f32> = raw.clone().slice([0..n, 3..4]).reshape([n]).into_data().to_vec().unwrap_or_default();
    let sxy_raw: Vec<f32> = raw.slice([0..n, 4..5]).reshape([n]).into_data().to_vec().unwrap_or_default();

    thetas.iter().enumerate().map(|(i, &theta)| {
        let sxx = sxx_raw[i] as f64 * px_pa;
        let syy = syy_raw[i] as f64 * px_pa;
        let sxy = sxy_raw[i] as f64 * px_pa;
        let (s_rr, _s_tt, _s_rt) = decompose_radial(sxx, syy, sxy, theta);
        ContactPressureSample {
            theta_deg: theta.to_degrees(),
            theta_rad: theta,
            sigma_rr_ksi: s_rr / KSI_TO_PA,
        }
    }).collect()
}

/// Write the (theta, sigma_rr) contact-pressure profile to a CSV file at `path`. Plain
/// hand-rolled comma-separated writing (no CSV crate dependency exists elsewhere in this
/// workspace to reuse — see `Cargo.toml` audit — and this format is simple enough not to
/// warrant adding one).
pub fn write_contact_pressure_csv(
    samples: &[ContactPressureSample],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let mut buf = String::with_capacity(32 + samples.len() * 32);
    buf.push_str("theta_deg,theta_rad,sigma_rr_ksi\n");
    for s in samples {
        buf.push_str(&format!("{:.6},{:.9},{:.6}\n", s.theta_deg, s.theta_rad, s.sigma_rr_ksi));
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(buf.as_bytes())
}

/// Convenience: sample + write in one call, using `CONTACT_EXPORT_N_SAMPLES` and
/// `DEFAULT_CONTACT_EXPORT_PATH`. Called automatically at the end of `run_headless_pinlug`.
pub fn export_contact_pressure<Bk: Backend>(
    lug_model: &ElasticityNet<Bk>,
    lug_geom: &GeometryConfig,
    px_pa: f64,
    device: &Bk::Device,
) -> std::io::Result<Vec<ContactPressureSample>> {
    let samples = sample_contact_pressure::<Bk>(lug_model, lug_geom, px_pa, CONTACT_EXPORT_N_SAMPLES, device);
    write_contact_pressure_csv(&samples, DEFAULT_CONTACT_EXPORT_PATH)?;
    Ok(samples)
}

/// Normalize a physical coordinate to [-1,1]^2 for an arbitrary geometry's own bounds.
/// Duplicated (deliberately, to avoid making `headless.rs`'s private helper `pub`) from
/// `headless::normalize_point_generic` — identical formula, single well-understood
/// responsibility, not worth a cross-module visibility change for one helper this small.
fn normalize_point_generic(x: f64, y: f64, geom: &GeometryConfig) -> [f32; 2] {
    let (x0, x1) = geom.x_range();
    let (y0, y1) = geom.y_range();
    let dw = x1 - x0;
    let dh = y1 - y0;
    [(2.0 * (x - x0) / dw - 1.0) as f32, (2.0 * (y - y0) / dh - 1.0) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::Tensor;
    use crate::network::ElasticityNetConfig;
    use crate::problem::{B, BDevice};

    /// Forces every Linear layer's weight to zero, and rewrites any 1-D bias whose length
    /// matches `output_dim` (i.e. only the final `out` layer's bias — hidden-layer biases
    /// have length `hidden_dim`, deliberately chosen to differ from `output_dim` by the test
    /// harness below) to `output`, leaving all other biases (hidden layers) zeroed. With
    /// every weight zero, `layers[0].forward(x).tanh() == tanh(bias0) `, and each subsequent
    /// `tanh(hidden_bias)` similarly collapses to a constant independent of `x` — but since
    /// hidden biases are zeroed too, every hidden activation is `tanh(0) = 0`, so
    /// `out.forward(0) == out_bias` exactly, i.e. the network output is deterministically
    /// `output` for ANY input. Same technique `pinlug_problem.rs`'s `convergence_metric`
    /// tests use (`ZeroMapper`), extended to pin a known NONZERO constant output.
    struct ConstantOutputMapper {
        output_dim: usize,
        output: Vec<f32>,
    }
    impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for ConstantOutputMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let dims = param.val().dims();
            if D == 1 && dims[0] == self.output_dim {
                let device = param.device();
                let data = burn::tensor::TensorData::new(self.output.clone(), dims.to_vec());
                let new_t = Tensor::<B, D>::from_data(data, &device);
                param.map(|_| new_t)
            } else {
                param.map(|t| t.zeros_like())
            }
        }
    }

    /// Build a tiny ElasticityNet whose output is forced to a known constant vector for any
    /// input, by zeroing all weights/hidden-biases and rewriting the final layer's bias.
    /// `hidden_dim` (8) is chosen distinct from `output.len()` (5) so the mapper can tell
    /// hidden biases and the output bias apart purely by shape.
    fn constant_output_net(output: &[f32]) -> ElasticityNet<B> {
        let device: BDevice = Default::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(output.len())
            .with_use_piratenet(false);
        let model: ElasticityNet<B> = net_cfg.init(&device);
        let mut mapper = ConstantOutputMapper { output_dim: output.len(), output: output.to_vec() };
        model.map(&mut mapper)
    }

    #[test]
    fn export_sigma_rr_matches_hand_computed_values_at_theta_0_and_pi_2() {
        // Constant network output: u=0, v=0, sxx=1.0 (raw), syy=0.5 (raw), sxy=0.0 (raw).
        // With px_pa=1.0 (identity scale), physical sxx=1.0, syy=0.5, sxy=0.0 everywhere.
        let model = constant_output_net(&[0.0, 0.0, 1.0, 0.5, 0.0]);
        let device: BDevice = Default::default();
        let lug_geom = GeometryConfig::pinlug_lug_inches();
        let px_pa = 1.0;

        // Sample exactly at theta=0 and theta=pi/2 (and a couple others) via a custom
        // small n aligned so those two land exactly (n=... use direct helper instead).
        let n = 5; // thetas: -pi/2, -pi/4, 0, pi/4, pi/2 (evenly spaced over [-pi/2, pi/2])
        let samples = sample_contact_pressure::<B>(&model, &lug_geom, px_pa, n, &device);
        assert_eq!(samples.len(), n);

        let at_theta0 = samples.iter().find(|s| s.theta_rad.abs() < 1e-9)
            .expect("theta=0 must be one of the 5 evenly-spaced samples");
        assert!(
            (at_theta0.sigma_rr_ksi - (1.0 / KSI_TO_PA)).abs() < 1e-9,
            "theta=0: sigma_rr must equal sxx exactly; expected {}, got {}",
            1.0 / KSI_TO_PA, at_theta0.sigma_rr_ksi,
        );

        let at_theta_pi2 = samples.iter().find(|s| (s.theta_rad - PI / 2.0).abs() < 1e-9)
            .expect("theta=pi/2 must be one of the 5 evenly-spaced samples");
        assert!(
            (at_theta_pi2.sigma_rr_ksi - (0.5 / KSI_TO_PA)).abs() < 1e-9,
            "theta=pi/2: sigma_rr must equal syy exactly; expected {}, got {}",
            0.5 / KSI_TO_PA, at_theta_pi2.sigma_rr_ksi,
        );

        let at_theta_neg_pi2 = samples.iter().find(|s| (s.theta_rad + PI / 2.0).abs() < 1e-9)
            .expect("theta=-pi/2 must be one of the 5 evenly-spaced samples");
        assert!(
            (at_theta_neg_pi2.sigma_rr_ksi - (0.5 / KSI_TO_PA)).abs() < 1e-9,
            "theta=-pi/2: sigma_rr must equal syy exactly (sin^2 term); expected {}, got {}",
            0.5 / KSI_TO_PA, at_theta_neg_pi2.sigma_rr_ksi,
        );
    }

    #[test]
    fn export_writes_csv_with_expected_header_and_row_count() {
        let model = constant_output_net(&[0.0, 0.0, 2.0, 1.0, 0.3]);
        let device: BDevice = Default::default();
        let lug_geom = GeometryConfig::pinlug_lug_inches();
        let n = 8;
        let samples = sample_contact_pressure::<B>(&model, &lug_geom, 1.0, n, &device);
        assert_eq!(samples.len(), n);

        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join(format!("pinlug_contact_pressure_test_{}.csv", std::process::id()));

        write_contact_pressure_csv(&samples, &path).expect("csv write must succeed");

        let contents = std::fs::read_to_string(&path).expect("csv file must exist and be readable");
        let mut lines = contents.lines();
        let header = lines.next().expect("csv must have a header line");
        assert_eq!(header, "theta_deg,theta_rad,sigma_rr_ksi");

        let data_lines: Vec<&str> = lines.collect();
        assert_eq!(data_lines.len(), n, "expected {n} data rows, got {}", data_lines.len());

        for line in &data_lines {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 3, "each row must have exactly 3 comma-separated fields, got: {line}");
            for f in &fields {
                f.parse::<f64>().unwrap_or_else(|_| panic!("field '{f}' in row '{line}' must parse as f64"));
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_theta_range_is_bounded_to_loaded_half() {
        let model = constant_output_net(&[0.0, 0.0, 1.0, 1.0, 0.0]);
        let device: BDevice = Default::default();
        let lug_geom = GeometryConfig::pinlug_lug_inches();
        let samples = sample_contact_pressure::<B>(&model, &lug_geom, 1.0, 64, &device);
        for s in &samples {
            assert!(
                s.theta_rad >= -PI / 2.0 - 1e-9 && s.theta_rad <= PI / 2.0 + 1e-9,
                "theta_rad={} out of the required [-pi/2, pi/2] loaded-half range",
                s.theta_rad,
            );
        }
    }
}
