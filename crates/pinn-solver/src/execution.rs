//! Hardware-adaptive execution, Phase 1: an explicit seam for *how* host-side
//! work (resampling collocation points) gets executed, without yet adding any
//! real parallel/GPU-dispatch path behind it. Phase 4 adds
//! [`apply_performance_profile`] (makes `Eco` actually reduce sampling
//! density) and [`cpu_thread_count`] (caps whatever CPU-side thread pool the
//! compiled backend uses) - see each function's own doc comment.
//!
//! Scope discipline (see the epic addendum this implements, and the plan it
//! was reconciled against): this module deliberately does NOT wrap
//! `training_core::step_physics`/`step_physics_multi` — those are single
//! batched `burn` tensor graphs, nothing embarrassingly parallel inside them,
//! and CLAUDE.md's own warning against refactoring `step_physics` into a thin
//! wrapper applies with full force to any temptation to route it through an
//! `Executor` here too. It also does NOT introduce a `TensorBackend`
//! abstraction — `burn::tensor::backend::Backend` (already threaded through
//! everything via `training_core::B`/`BInner`) already is that layer; a
//! second hand-rolled one would duplicate it, not add to it.
//!
//! What this module DOES wrap: the free-function resampling calls
//! (`pinn_core::sampling::sample_interior`/`sample_boundary`) that Kirsch's
//! pre-trait, frozen `runner.rs`/`headless.rs` call sites use directly. This
//! is deliberately narrower than `pinn_core::problem::DomainSamplingStrategy`
//! (the existing per-*domain* sampling abstraction pin-lug's
//! `PinLugSamplingStrategy`/Kirsch's own `KirschSamplingStrategy` implement) —
//! that trait already gives per-domain resampling behavior; this one is
//! about per-*hardware* resampling strategy, an orthogonal axis, and Phase 1
//! only wires it at the plain free-function call sites, not through
//! `DomainSamplingStrategy`. Pin-lug's every-step resample
//! (`pinlug_problem::PinLugSamplingStrategy`) is untouched in Phase 1 - it's
//! the one real RNG-order-sensitive hot path a future CPU executor would
//! matter for, and parallelizing it correctly requires deterministic
//! RNG-stream partitioning that is its own reviewed change, not a Phase 1
//! side effect.

use pinn_core::geometry::GeometryConfig;
use pinn_core::loading::{BoundaryPoint, LoadConfig};
use pinn_core::messages::{ExecutionConfig, PerformanceProfile, SolverConfig};
use pinn_core::sampling::{sample_boundary, sample_interior};

/// Executes host-side resampling. `SerialExecutor` is the only implementation
/// in Phase 1 - it's a pure identity wrapper around today's direct
/// `sample_interior`/`sample_boundary` calls, so wiring it in at existing
/// call sites is provably a no-op (see
/// `resample_via_serial_executor_matches_direct_sample_interior_call` in this
/// module's tests). A `RayonExecutor`/`GpuExecutor` would implement this same
/// trait in a later phase, once profiling data (not a guess) justifies one.
pub trait Executor {
    fn sample_interior(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]>;
    fn sample_boundary(&self, geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<BoundaryPoint>;
}

pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn sample_interior(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        sample_interior(geom, n)
    }
    fn sample_boundary(&self, geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        sample_boundary(geom, load, n)
    }
}

/// Resolves an [`ExecutionConfig`] to a concrete [`Executor`]. Phase 1: this
/// is a deliberate stub that always returns [`SerialExecutor`] regardless of
/// `ExecutionMode` - `Auto` and `Serial` both mean "what every code path
/// already does today." Its only job right now is to exist as the single
/// call site a later phase teaches to read profiling data (wall time,
/// resample cost at pin-lug's every-step cadence) before ever choosing
/// anything else. Do not add workload-size heuristics here without real
/// measurements behind them - that's exactly the "guess, then code" pattern
/// the epic's own profiling-driven-optimization principle warns against.
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    pub fn plan(_config: &ExecutionConfig) -> Box<dyn Executor> {
        Box::new(SerialExecutor)
    }
}

/// Divides `n_interior`/`n_boundary` by this much under [`PerformanceProfile::Eco`], floored
/// at [`ECO_MIN_INTERIOR`]/[`ECO_MIN_BOUNDARY`]. Not empirically tuned — a conservative,
/// disclosed first pass (this codebase already has precedent for shipping an explicitly
/// untuned-first-pass constant rather than blocking on tuning data that doesn't exist yet —
/// see pin-lug's reused `PLATEAU_WINDOW` in `controllers.rs`). Revisit if real Eco-profile
/// runs show this is too aggressive or not aggressive enough for actual low-power hardware.
const ECO_SAMPLING_DIVISOR: usize = 4;
const ECO_MIN_INTERIOR: usize = 256;
const ECO_MIN_BOUNDARY: usize = 64;

/// Phase 4: makes [`PerformanceProfile::Eco`] actually reduce resource usage - fewer
/// collocation points means less per-step host-side sampling AND less per-step tensor work
/// (smaller batch dimension into `step_physics`/`step_physics_multi`), matching the epic's
/// own "Eco: small batches, low CPU/RAM utilization" description. Every other profile
/// (`Balanced`/`Performance`/`Maximum`) leaves `n_interior`/`n_boundary` exactly as configured
/// — this function never *increases* them, since more collocation points changes residual
/// sampling density, and inventing a "Performance means more points" behavior wasn't asked
/// for and isn't obviously a resource-usage lever in the way fewer points clearly is.
///
/// **Never alters the mathematical formulation being solved** (same PDE, same BCs, same
/// material/geometry/load) — only how finely the residual is sampled, which is squarely an
/// execution-resource decision, matching `PerformanceProfile`'s own doc comment guarantee.
/// Call once, after `pinn.env`/CLI overrides are applied and before training starts (calling
/// it twice would compound the division) — `pinn-app/src/main.rs`'s `main()` is the one call
/// site today.
pub fn apply_performance_profile(config: &mut SolverConfig) {
    if config.execution.profile == PerformanceProfile::Eco {
        config.n_interior = (config.n_interior / ECO_SAMPLING_DIVISOR).max(ECO_MIN_INTERIOR);
        config.n_boundary = (config.n_boundary / ECO_SAMPLING_DIVISOR).max(ECO_MIN_BOUNDARY);
    }
}

/// CPU thread count to request for whatever the compiled backend's own CPU-side work needs -
/// most relevant when the `ndarray-backend` Cargo feature is active (`burn-ndarray` uses
/// `rayon` internally for its own tensor ops, confirmed by reading `burn-ndarray`'s own
/// `parallel.rs`), a harmless no-op-ish setting for the default Wgpu backend (whose tensor
/// compute is GPU-bound regardless — though other CPU-side work in the process, e.g. `image`/
/// `wgpu`'s own internal use of rayon, still respects the same global pool). `None` means
/// "don't touch rayon's global thread pool at all - let it auto-detect (its own default)."
/// The caller (`pinn-app/src/main.rs`) is responsible for actually calling
/// `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` — this function only
/// decides the number, matching `pinn_solver`'s existing "no rayon dependency of its own"
/// precedent (see this module's own doc comment) - configuring the pool used by dependencies
/// is an app-binary-entry-point concern, not a solver-logic one.
pub fn cpu_thread_count(profile: PerformanceProfile) -> Option<usize> {
    match profile {
        PerformanceProfile::Eco => Some(1),
        PerformanceProfile::Balanced => None,
        PerformanceProfile::Performance | PerformanceProfile::Maximum => {
            std::thread::available_parallelism().ok().map(|n| n.get())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::geometry::GeometryConfig;
    use pinn_core::messages::{ExecutionMode, PerformanceProfile};

    #[test]
    fn resample_via_serial_executor_matches_direct_sample_interior_call() {
        let geom = GeometryConfig::kirsch_plate_inches();
        let load = pinn_core::loading::LoadConfig::default_10ksi();
        let executor = SerialExecutor;
        let via_executor_int = executor.sample_interior(&geom, 64);
        let direct_int = sample_interior(&geom, 64);
        assert_eq!(via_executor_int, direct_int);

        let via_executor_bnd = executor.sample_boundary(&geom, &load, 32);
        let direct_bnd = sample_boundary(&geom, &load, 32);
        assert_eq!(via_executor_bnd, direct_bnd);
    }

    #[test]
    fn planner_always_resolves_to_serial_in_phase_1() {
        for mode in [ExecutionMode::Auto, ExecutionMode::Serial] {
            for profile in [PerformanceProfile::Eco, PerformanceProfile::Balanced, PerformanceProfile::Performance, PerformanceProfile::Maximum] {
                let cfg = ExecutionConfig { mode, profile };
                let executor = ExecutionPlanner::plan(&cfg);
                let geom = GeometryConfig::kirsch_plate_inches();
                // Just confirm it constructs and runs without panicking for every combination -
                // Phase 1 has exactly one behavior regardless of config.
                let pts = executor.sample_interior(&geom, 16);
                assert_eq!(pts.len(), 16);
            }
        }
    }

    #[test]
    fn apply_performance_profile_eco_reduces_sampling_and_floors_at_minimums() {
        use pinn_core::messages::SolverConfig;

        let mut cfg = SolverConfig::default_kirsch();
        cfg.execution.profile = PerformanceProfile::Eco;
        let (n_int_before, n_bnd_before) = (cfg.n_interior, cfg.n_boundary);
        apply_performance_profile(&mut cfg);
        assert_eq!(cfg.n_interior, (n_int_before / ECO_SAMPLING_DIVISOR).max(ECO_MIN_INTERIOR));
        assert_eq!(cfg.n_boundary, (n_bnd_before / ECO_SAMPLING_DIVISOR).max(ECO_MIN_BOUNDARY));
        assert!(cfg.n_interior < n_int_before, "Eco must actually reduce n_interior from Kirsch's default");
        assert!(cfg.n_boundary < n_bnd_before, "Eco must actually reduce n_boundary from Kirsch's default");

        // Floor check: a tiny configured value must not divide below the documented minimum.
        let mut cfg_tiny = SolverConfig::default_kirsch();
        cfg_tiny.execution.profile = PerformanceProfile::Eco;
        cfg_tiny.n_interior = 100;
        cfg_tiny.n_boundary = 20;
        apply_performance_profile(&mut cfg_tiny);
        assert_eq!(cfg_tiny.n_interior, ECO_MIN_INTERIOR);
        assert_eq!(cfg_tiny.n_boundary, ECO_MIN_BOUNDARY);
    }

    #[test]
    fn apply_performance_profile_non_eco_leaves_sampling_unchanged() {
        use pinn_core::messages::SolverConfig;

        for profile in [PerformanceProfile::Balanced, PerformanceProfile::Performance, PerformanceProfile::Maximum] {
            let mut cfg = SolverConfig::default_kirsch();
            cfg.execution.profile = profile;
            let (n_int_before, n_bnd_before) = (cfg.n_interior, cfg.n_boundary);
            apply_performance_profile(&mut cfg);
            assert_eq!(cfg.n_interior, n_int_before);
            assert_eq!(cfg.n_boundary, n_bnd_before);
        }
    }

    #[test]
    fn cpu_thread_count_eco_is_minimal_balanced_is_auto_others_are_all_cores() {
        assert_eq!(cpu_thread_count(PerformanceProfile::Eco), Some(1));
        assert_eq!(cpu_thread_count(PerformanceProfile::Balanced), None);
        let all_cores = std::thread::available_parallelism().ok().map(|n| n.get());
        assert_eq!(cpu_thread_count(PerformanceProfile::Performance), all_cores);
        assert_eq!(cpu_thread_count(PerformanceProfile::Maximum), all_cores);
    }
}
