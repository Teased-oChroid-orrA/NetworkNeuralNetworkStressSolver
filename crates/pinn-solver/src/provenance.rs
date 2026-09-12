//! Issue #61 EPIC P2-13: reproducibility and provenance metadata per saved run.
//!
//! Every field is either a REAL, verified value or an explicit `None`/`false` with a doc
//! comment explaining WHY - never invented or guessed. Two fields in particular are real,
//! honest NEGATIVE findings rather than gaps papered over: `model_init_seeded` is always
//! `false` (confirmed by reading `network::ElasticityNetConfig::init` - it calls burn's
//! `LinearConfig::init(device)` with no explicit seed anywhere in the call chain, so model
//! weight initialization is genuinely NOT reproducible in this codebase today), and
//! `derivative_backend` always reports `"FD"` - and, per issue #62 PH3-06's own real finding,
//! always WILL: FD is the only backend that can ever supply a live TRAINING-loss derivative in
//! this codebase, a structural fact, not a temporary gap. `differential_operator::ad_strain`
//! retrieves its gradient via burn's `.grad()` API, which returns a value on `B::InnerBackend` -
//! detached from any autodiff graph, because burn-autodiff 0.21 has no nested/higher-order
//! autodiff (`differential_operator.rs`'s own module doc comment has the full, verified
//! derivation). A `LossTerm::compute()` result MUST stay on `Tensor<B, 1>` (connected to the
//! model-WEIGHT autodiff graph the optimizer's own `.backward()` differentiates through), so
//! `ad_strain`'s output can never be plugged in there - not "not yet", but "cannot be, given
//! this dependency". AD IS live-wired now, as a DIAGNOSTIC only (`differential_operator::
//! ad_fd_strain_agreement`, opt-in via `TrainingSpec.derivative_operator_diagnostic`, surfaced
//! on `TrainingUpdate.ad_fd_strain_diagnostic`) - real, verified live use of the AD backend
//! against a currently-training model, just never as the thing this field reports.

use serde::{Deserialize, Serialize};

/// See this module's own doc comment.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RunProvenance {
    /// The git commit SHA of the working tree AT THE TIME THIS RUN WAS SAVED (captured by
    /// shelling out to `git rev-parse HEAD` at save time, not baked in at build time - reflects
    /// the actual repo state when a user saves a checkpoint, which can differ from the build
    /// state during iterative development). `None` if git is unavailable or this isn't a git
    /// checkout (e.g. a packaged binary run outside any repository) - a real "unavailable", not
    /// a placeholder.
    pub git_sha: Option<String>,
    /// True iff `git status --porcelain` reported uncommitted changes at save time. `None`
    /// under the same unavailability conditions as `git_sha`.
    pub git_dirty: Option<bool>,
    /// A fast, non-cryptographic fingerprint (`std::collections::hash_map::DefaultHasher` over
    /// the spec's own canonical JSON serialization) for "is this the same config" comparisons
    /// without a deep equality check. Always available - computed from the spec already saved
    /// in `CheckpointMeta.spec`, not looked up externally. Explicitly NOT a cryptographic hash
    /// (SipHash-1-3, collision-resistant enough for this fingerprint use case, but not a
    /// security primitive) - adding a `sha2` dependency for this one use was judged unwarranted
    /// complexity.
    pub problem_hash: String,
    /// This codebase's real, fixed interior-collocation seed (`user_problem::SEED_INTERIOR`,
    /// `90_210`) - `UserSamplingStrategy::sample_interior` always re-seeds with this exact
    /// constant, so interior point SAMPLING is deterministic and reproducible.
    pub interior_sampling_seed: u64,
    /// Always `false` - see this module's own doc comment for the confirmed reason (no
    /// explicit seed anywhere in `ElasticityNetConfig::init`'s call chain). A real, verified
    /// negative finding: MODEL WEIGHT INITIALIZATION is not currently reproducible, even
    /// though interior sampling is.
    pub model_init_seeded: bool,
    /// Which burn backend this binary was compiled with (`Wgpu` or `NdArray` - see `training_
    /// core::BInner`'s own `#[cfg(feature = "ndarray-backend")]` gate).
    pub backend: String,
    /// This codebase's fixed tensor element type (burn's `f32` default - never configured
    /// otherwise anywhere in this codebase).
    pub dtype: String,
    /// The only derivative backend actually wired into live training today - see this module's
    /// own doc comment.
    pub derivative_backend: String,
    /// The problem's own formulation selection (`Variational`/`Strong`/`Hybrid(...)`, P2-01),
    /// if the spec type has one - `None` for `ParametricProblemSpec`, which genuinely has no
    /// formulation field (a real "not applicable", not a lookup failure).
    pub formulation: Option<String>,
}

/// Computes [`RunProvenance`] for `spec` (generic over `ProblemSpec`/`ParametricProblemSpec` -
/// both are `Serialize`, which is all `problem_hash` needs). `formulation` is `Some(...)` for
/// plate specs (`format!("{:?}", spec.formulation)`) and `None` for parametric specs (no such
/// field exists there). `git_sha`/`git_dirty` shell out to the real `git` binary in the current
/// working directory - see those fields' own doc comments for the honest unavailability
/// behavior.
pub fn compute_run_provenance<S: Serialize>(spec: &S, formulation: Option<String>) -> RunProvenance {
    RunProvenance {
        git_sha: git_sha(),
        git_dirty: git_dirty(),
        problem_hash: problem_hash(spec),
        interior_sampling_seed: crate::user_problem::SEED_INTERIOR,
        model_init_seeded: false,
        backend: backend_name().to_string(),
        dtype: "f32".to_string(),
        derivative_backend: "FD".to_string(),
        formulation,
    }
}

fn backend_name() -> &'static str {
    if cfg!(feature = "ndarray-backend") { "NdArray" } else { "Wgpu" }
}

fn git_sha() -> Option<String> {
    let output = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn git_dirty() -> Option<bool> {
    let output = std::process::Command::new("git").args(["status", "--porcelain"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok().map(|s| !s.trim().is_empty())
}

fn problem_hash<S: Serialize>(spec: &S) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let json = serde_json::to_string(spec).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::loading::LoadConfig;
    use pinn_core::material::MaterialProps;
    use pinn_core::problem_spec::ProblemSpec;
    use pinn_core::user_geometry::UserGeometry;

    fn sample_spec() -> ProblemSpec {
        ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        }
    }

    #[test]
    fn compute_run_provenance_reports_known_real_values() {
        let spec = sample_spec();
        let prov = compute_run_provenance(&spec, Some(format!("{:?}", spec.formulation)));
        assert_eq!(prov.interior_sampling_seed, crate::user_problem::SEED_INTERIOR);
        assert!(!prov.model_init_seeded, "model init is genuinely not seeded in this codebase - a real, verified negative finding");
        assert_eq!(prov.derivative_backend, "FD");
        assert_eq!(prov.dtype, "f32");
        assert!(!prov.problem_hash.is_empty());
        assert_eq!(prov.formulation, Some(format!("{:?}", spec.formulation)));
    }

    #[test]
    fn compute_run_provenance_reports_none_formulation_when_not_applicable() {
        let spec = sample_spec();
        let prov = compute_run_provenance(&spec, None);
        assert_eq!(prov.formulation, None, "e.g. ParametricProblemSpec has no formulation field");
    }

    #[test]
    fn problem_hash_is_deterministic_and_distinguishes_different_specs() {
        let spec_a = sample_spec();
        let mut spec_b = sample_spec();
        spec_b.load = LoadConfig::uniaxial_x(2e7); // different load

        let prov_a1 = compute_run_provenance(&spec_a, None);
        let prov_a2 = compute_run_provenance(&spec_a, None);
        assert_eq!(prov_a1.problem_hash, prov_a2.problem_hash, "same spec must hash identically");

        let prov_b = compute_run_provenance(&spec_b, None);
        assert_ne!(prov_a1.problem_hash, prov_b.problem_hash, "different specs must hash differently");
    }

    #[test]
    fn git_sha_and_dirty_are_consistent_with_availability() {
        // This test runs inside a real git checkout (the project itself), so both should
        // resolve to Some(...) here - but the function itself must never panic regardless of
        // git availability, which this call proves by simply not panicking.
        let sha = git_sha();
        let dirty = git_dirty();
        // Either both resolve (real git checkout) or both are None (git unavailable) - never
        // one Some and the other silently wrong, since both query the same repository.
        assert_eq!(sha.is_some(), dirty.is_some(), "git_sha={sha:?} git_dirty={dirty:?}");
    }
}
