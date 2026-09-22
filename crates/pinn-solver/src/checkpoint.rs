//! Model checkpoint save/load — the first model-persistence mechanism in this codebase
//! (confirmed by prior investigation: none existed anywhere, see `pinn_core::
//! inference_envelope`'s doc comment). Built for real inference reuse, not resume-training:
//! only the model's WEIGHTS are saved, via burn's own `NamedMpkGzFileRecorder<
//! HalfPrecisionSettings>` (gzip'd, half-precision binary — an existing, idiomatic burn
//! facility that directly minimizes file size, not a hand-rolled format). No optimizer
//! momentum state is persisted.
//!
//! A companion `.meta.json` sidecar carries everything needed to reconstruct the right network
//! architecture and know what the loaded model is valid for (which spec trained it, how many
//! steps, final loss) — this doubles as real, run-derived Model Contract persistence
//! (`enhancement.md` Phases 20/26/39), not a hand-authored summary.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::{FileRecorder, HalfPrecisionSettings, NamedMpkGzFileRecorder};
use serde::{Deserialize, Serialize};

use pinn_core::{parametric_spec::ParametricProblemSpec, problem_spec::ProblemSpec};

use crate::network::{ElasticityNet, ElasticityNetConfig, LegacyElasticityNet};
use crate::training_core::{BDevice, BInner, B};

/// Which problem type a checkpoint was trained for — determines the network's `input_dim` (3
/// for a plain `ProblemSpec`, 6 for a parametric spec's `x,y,z,e_n,nu_n,p_n`) and which
/// validity checks apply once loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointSpec {
    Plate(ProblemSpec),
    Parametric(ParametricProblemSpec),
}

/// Sidecar metadata saved next to the `.mpk.gz` weights file — real Model Contract data
/// (`enhancement.md` Phases 20/26/39), not hand-authored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub spec: CheckpointSpec,
    /// The step actually reached when this checkpoint was saved (may be less than
    /// `training.max_steps` — training can be stopped early).
    pub steps_completed: usize,
    pub final_loss: f32,
    /// Unix timestamp (seconds), passed in by the caller — this module has no clock read of
    /// its own (kept a pure function of its arguments, same as every other solver-side probe
    /// in this codebase).
    pub saved_at_unix: u64,
    /// Input width actually saved with the weight record. Missing means a checkpoint written
    /// before geometric enrichment metadata existed; those plate checkpoints used raw width 3.
    #[serde(default)]
    pub input_dim: Option<usize>,
    /// Issue #61 EPIC P2-13: reproducibility/provenance metadata - `#[serde(default)]` so
    /// existing `.meta.json` files saved before this field existed still deserialize (with
    /// `RunProvenance::default()`, whose `Option` fields are `None` and whose non-`Option`
    /// fields are their type's zero value - an honest "this old checkpoint predates provenance
    /// tracking", not a fabricated value).
    #[serde(default)]
    pub provenance: crate::provenance::RunProvenance,
    /// Issue #62 PH3-13: the full authoritative report (`RunProvenance` plus L0-L5 verdicts,
    /// integration/sampling mode, AMR state, energy balance, reaction force, convergence
    /// evidence) - `#[serde(default)]`, `None` for a checkpoint saved before this field
    /// existed (an honest "predates the authoritative report", not a fabricated value, same
    /// treatment as `provenance` above before it). `provenance` above is kept, not removed
    /// (this field's own `AuthoritativeReport.provenance` is a real duplicate of it) - avoids
    /// an actually-breaking schema change for the one field every existing `.meta.json` file
    /// already has.
    #[serde(default)]
    pub report: Option<crate::provenance::AuthoritativeReport>,
}

fn recorder() -> NamedMpkGzFileRecorder<HalfPrecisionSettings> {
    NamedMpkGzFileRecorder::<HalfPrecisionSettings>::default()
}

fn meta_path(weights_path: &Path) -> PathBuf {
    let stem = weights_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut p = weights_path.to_path_buf();
    p.set_file_name(format!("{stem}.meta.json"));
    p
}

/// Saves `model`'s weights to `weights_path` (burn appends its own `.mpk.gz` extension) plus a
/// `.meta.json` sidecar next to it. Returns the exact weights-file path actually written.
pub fn save_checkpoint(
    model: ElasticityNet<BInner>,
    meta: &CheckpointMeta,
    weights_path: &Path,
) -> Result<PathBuf, String> {
    let input_dim = model.input_dim();
    model
        .save_file(weights_path, &recorder())
        .map_err(|e| format!("failed to write model weights: {e}"))?;
    let mut meta = meta.clone();
    meta.input_dim = Some(input_dim);
    let json = serde_json::to_string_pretty(&meta)
        .map_err(|e| format!("failed to serialize checkpoint metadata: {e}"))?;
    std::fs::write(meta_path(weights_path), json)
        .map_err(|e| format!("failed to write checkpoint metadata: {e}"))?;
    // `Module::save_file` appends the recorder's extension via `PathBuf::set_extension` (burn-
    // core's own `record/file.rs` macro) - replicate the SAME call here (not a guess) so the
    // path this returns is exactly the file that now exists on disk.
    let mut written = weights_path.to_path_buf();
    written.set_extension(
        <NamedMpkGzFileRecorder<HalfPrecisionSettings> as FileRecorder<BInner>>::file_extension(),
    );
    Ok(written)
}

fn parse_meta(weights_path: &Path) -> Result<CheckpointMeta, String> {
    let meta_json = std::fs::read_to_string(meta_path(weights_path))
        .map_err(|e| format!("failed to read checkpoint metadata (expected a .meta.json file next to the weights): {e}"))?;
    serde_json::from_str(&meta_json)
        .map_err(|e| format!("failed to parse checkpoint metadata: {e}"))
}

// Smart adaptive architecture: `hidden_dim`/`n_hidden` here are the LIVE values the caller
// wrote into `meta.spec` at save time (see `run_training_user_problem`/`run_training_
// parametric`'s own `current_hidden_dim`/`current_n_hidden` tracking), not necessarily the
// original spec's static ones - but the gated-residual STRUCTURE (`use_piratenet`) is fixed
// for the whole run by `adaptive` and must be reproduced exactly, or `load_file` below will
// fail to match the saved record's `gates` shape.
fn net_cfg_for_meta(meta: &CheckpointMeta) -> ElasticityNetConfig {
    // Plate's `input_dim` is no longer a flat `3` - see `UserGeometry::n_fourier`'s doc
    // comment - it depends on whether the saved spec's geometry has a hole, exactly mirroring
    // how the model was actually constructed for training (`runner::run_training_user_
    // problem`). Parametric's `6` (x,y,z + e,nu,px) is unaffected - the Fourier-embedding fix
    // is plate-only, matching every other change in this pass.
    let (input_dim, hidden_dim, n_hidden, adaptive) = match &meta.spec {
        CheckpointSpec::Plate(spec) => (
            meta.input_dim.unwrap_or(3),
            spec.network.hidden_dim,
            spec.network.n_hidden,
            spec.network.adaptive,
        ),
        CheckpointSpec::Parametric(spec) => (
            6,
            spec.network.hidden_dim,
            spec.network.n_hidden,
            spec.network.adaptive,
        ),
    };
    ElasticityNetConfig::new()
        .with_input_dim(input_dim)
        .with_hidden_dim(hidden_dim)
        .with_n_hidden(n_hidden)
        .with_output_dim(5)
        .with_use_piratenet(adaptive)
}

/// Loads a checkpoint's metadata and reconstructs a fresh model of the matching architecture
/// with the saved weights loaded in. `device` should be the inference-only `BInner` backend —
/// matches every other "instant inference" code path in this codebase (no autodiff needed to
/// just evaluate a loaded model). For resuming TRAINING from a checkpoint, see
/// `load_checkpoint_for_training` instead.
pub fn load_checkpoint(
    weights_path: &Path,
    device: &BDevice,
) -> Result<(ElasticityNet<BInner>, CheckpointMeta), String> {
    let meta = parse_meta(weights_path)?;
    let net_cfg = net_cfg_for_meta(&meta);
    let fresh: ElasticityNet<BInner> = net_cfg.init(device);
    let model = match fresh.load_file(weights_path, &recorder(), device) {
        Ok(model) => model,
        Err(current_error) => {
            let legacy: LegacyElasticityNet<BInner> = net_cfg.init_legacy(device)
                .load_file(weights_path, &recorder(), device)
                .map_err(|legacy_error| format!("failed to load model weights ({current_error}); legacy checkpoint fallback also failed: {legacy_error}"))?;
            ElasticityNet::from_legacy(legacy, device)
        }
    };
    Ok((model, meta))
}

/// Graceful-stop-and-resume: identical to `load_checkpoint` (same metadata parsing, same
/// architecture reconstruction), but builds a TRAINABLE (autodiff) model instead of an
/// inference-only one, for `runner::run_training_user_problem_resume` to continue training
/// from. Only the backend type parameter differs - `ElasticityNet<B>`'s `Module::load_file`
/// works the same way regardless of backend.
pub fn load_checkpoint_for_training(
    weights_path: &Path,
    device: &BDevice,
) -> Result<(ElasticityNet<B>, CheckpointMeta), String> {
    let meta = parse_meta(weights_path)?;
    let net_cfg = net_cfg_for_meta(&meta);
    let fresh: ElasticityNet<B> = net_cfg.init(device);
    let model = match fresh.load_file(weights_path, &recorder(), device) {
        Ok(model) => model,
        Err(current_error) => {
            let legacy: LegacyElasticityNet<B> = net_cfg.init_legacy(device)
                .load_file(weights_path, &recorder(), device)
                .map_err(|legacy_error| format!("failed to load model weights ({current_error}); legacy checkpoint fallback also failed: {legacy_error}"))?;
            ElasticityNet::from_legacy(legacy, device)
        }
    };
    Ok((model, meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::problem_spec::{NetworkSpec, TrainingSpec};
    use pinn_core::user_geometry::{HoleBc, HoleSpec, UserGeometry};
    use pinn_core::{loading::LoadConfig, material::MaterialProps};

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pinn_solver_checkpoint_test_{name}_{}",
            std::process::id()
        ))
    }

    fn tiny_plate_spec() -> ProblemSpec {
        ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1,
                half_h: 0.1,
                thickness: 0.005,
                holes: vec![HoleSpec {
                    center: [0.0, 0.0],
                    radius: 0.02,
                    bc: HoleBc::Free,
                }],
            },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec {
                hidden_dim: 8,
                n_hidden: 2,
                ..Default::default()
            },
            training: TrainingSpec::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
            architecture: Default::default(),
        }
    }

    #[test]
    fn save_then_load_round_trips_weights_and_metadata_exactly() {
        let device = BDevice::default();
        let spec = tiny_plate_spec();
        let embedding = spec.geometry.coordinate_embedding();
        // Saved input width, not the current geometry default, is the checkpoint contract.
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(spec.geometry.net_input_dim())
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5);
        let mut model: ElasticityNet<BInner> = net_cfg.init(&device);
        model.force_coordinate_skip_for_test(
            [[0.5, -0.25], [1.0, 0.75], [0.0, 0.0]],
            [0.1, -0.2],
            &device,
        );

        // Capture a real forward-pass output BEFORE saving, to prove the round-tripped model
        // isn't just structurally correct but numerically IDENTICAL - a fresh re-init with the
        // same config would have different random weights and fail this check.
        let probe_pts: Vec<[f32; 2]> = vec![[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]];
        let probe_tensor = crate::fd_stencil::norm_pts_to_tensor::<BInner>(&probe_pts, &device);
        let before =
            crate::network::fwd_embedded::<BInner>(&model, probe_tensor.clone(), embedding.clone(), &device)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        let meta = CheckpointMeta {
            spec: CheckpointSpec::Plate(spec),
            steps_completed: 42,
            final_loss: 0.0123,
            saved_at_unix: 1_700_000_000,
            input_dim: None,
            provenance: Default::default(),
            report: None,
        };
        let weights_path = tmp_path("roundtrip");
        let written = save_checkpoint(model, &meta, &weights_path).expect("save must succeed");
        assert!(
            written.exists(),
            "the weights file reported as written must actually exist on disk: {written:?}"
        );

        let (loaded, loaded_meta) =
            load_checkpoint(&weights_path, &device).expect("load must succeed");
        assert_eq!(loaded_meta.steps_completed, 42);
        assert!((loaded_meta.final_loss - 0.0123).abs() < 1e-6);
        assert_eq!(loaded_meta.saved_at_unix, 1_700_000_000);
        assert_eq!(loaded_meta.input_dim, Some(10));
        match &loaded_meta.spec {
            CheckpointSpec::Plate(s) => assert_eq!(s.network.hidden_dim, 8),
            CheckpointSpec::Parametric(_) => panic!("expected Plate spec"),
        }

        let after = crate::network::fwd_embedded::<BInner>(&loaded, probe_tensor, embedding, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(before.len(), after.len());
        // Half-precision round-trip is lossy - a loose but still meaningful tolerance (not
        // "close to zero", genuinely close to the ORIGINAL values) proves weights actually
        // loaded, not that a freshly-reinitialized (structurally-matching but numerically
        // different) model happened to pass.
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 1e-2,
                "loaded model output diverged from the saved model's own output: {b} vs {a}"
            );
        }

        let _ = std::fs::remove_file(&written);
        let _ = std::fs::remove_file(meta_path(&weights_path));
    }

    #[test]
    fn load_reconstructs_a_parametric_architecture_with_input_dim_six() {
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(6)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5);
        let model: ElasticityNet<BInner> = net_cfg.init(&device);

        let spec = ParametricProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1,
                half_h: 0.1,
                thickness: 0.005,
                holes: vec![],
            },
            e_range: pinn_core::parametric_spec::ParamRange::new(50e9, 100e9),
            nu_range: pinn_core::parametric_spec::ParamRange::new(0.25, 0.35),
            load_range: pinn_core::parametric_spec::ParamRange::new(40e6, 80e6),
            density: 2810.0,
            ultimate_strength_pa: 503e6,
            network: NetworkSpec {
                hidden_dim: 8,
                n_hidden: 2,
                ..Default::default()
            },
            training: TrainingSpec::default(),
        };
        let meta = CheckpointMeta {
            spec: CheckpointSpec::Parametric(spec),
            steps_completed: 6,
            final_loss: 1.0,
            saved_at_unix: 0,
            input_dim: None,
            provenance: Default::default(),
            report: None,
        };
        let weights_path = tmp_path("parametric");
        let written = save_checkpoint(model, &meta, &weights_path).expect("save must succeed");

        let (loaded, loaded_meta) =
            load_checkpoint(&weights_path, &device).expect("load must succeed");
        assert!(matches!(loaded_meta.spec, CheckpointSpec::Parametric(_)));
        // A 6-input forward pass must not panic/error - proves `load_checkpoint` actually
        // reconstructed input_dim=6, not the Plate default of 3.
        // `norm_pts_to_tensor` already emits [x,y,z=0] (3-wide) - concatenated with the
        // (e_n,nu_n,p_n) triple (3-wide) gives the 6-wide input a parametric network expects.
        let pts: Vec<[f32; 2]> = vec![[0.1, 0.2]];
        let xyz = crate::fd_stencil::norm_pts_to_tensor::<BInner>(&pts, &device);
        let params = burn::tensor::Tensor::<BInner, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0f32, 0.0, 0.0], vec![1, 3]),
            &device,
        );
        let input6 = burn::tensor::Tensor::cat(vec![xyz, params], 1);
        assert_eq!(
            input6.dims(),
            [1, 6],
            "sanity check: constructed input must be 1x6 for input_dim=6"
        );
        let out = crate::network::fwd::<BInner>(&loaded, input6, 0, &device);
        assert_eq!(
            out.dims(),
            [1, 5],
            "output should be 1 row x 5 columns (u,v,sxx,syy,sxy) for 1 input point"
        );

        let _ = std::fs::remove_file(&written);
        let _ = std::fs::remove_file(meta_path(&weights_path));
    }

    #[test]
    fn legacy_record_loads_with_zero_coordinate_skip() {
        let device = BDevice::default();
        let spec = tiny_plate_spec();
        let n_fourier = spec.geometry.n_fourier();
        // A genuinely "legacy" (`meta.input_dim: None`) checkpoint predates chart embedding by
        // definition - it was saved back when `net_input_dim()` was unconditionally 3 for every
        // geometry, regardless of hole count. Using `spec.geometry.net_input_dim()` here (now
        // 10 for `tiny_plate_spec()`'s single centered Free hole, post-#77) would build and save
        // a model shape the `input_dim: None` fallback (`unwrap_or(3)`) can never actually
        // reconstruct - exactly the matmul-dimension-mismatch this hardcoded `3` avoids, and the
        // correct fixture for what this test is actually verifying (backward compatibility with
        // a pre-chart-embedding record, not chart embedding itself).
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5);
        let legacy: LegacyElasticityNet<BInner> = net_cfg.init_legacy(&device);
        let meta = CheckpointMeta {
            spec: CheckpointSpec::Plate(spec),
            steps_completed: 1,
            final_loss: 0.0,
            saved_at_unix: 0,
            input_dim: None,
            provenance: Default::default(),
            report: None,
        };
        let weights_path = tmp_path("legacy_fallback");
        legacy
            .save_file(&weights_path, &recorder())
            .expect("legacy record save must succeed");
        std::fs::write(
            meta_path(&weights_path),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
        let (loaded, _) = load_checkpoint(&weights_path, &device).expect("legacy record must load");

        let points = crate::fd_stencil::norm_pts_to_tensor::<BInner>(&[[0.2, -0.4]], &device);
        let old_output = if n_fourier > 0 {
            loaded.forward(crate::network::fourier_embed(
                points.clone(),
                n_fourier,
                &device,
            ))
        } else {
            loaded.forward(points.clone())
        };
        let new_output = crate::network::fwd(&loaded, points, n_fourier, &device);
        let difference: f32 = (old_output - new_output).abs().sum().into_scalar();
        assert_eq!(
            difference, 0.0,
            "legacy fallback must zero-initialize coordinate residual"
        );

        let mut written = weights_path.clone();
        written.set_extension(
            <NamedMpkGzFileRecorder<HalfPrecisionSettings> as FileRecorder<BInner>>::file_extension(
            ),
        );
        let _ = std::fs::remove_file(written);
        let _ = std::fs::remove_file(meta_path(&weights_path));
    }
}
