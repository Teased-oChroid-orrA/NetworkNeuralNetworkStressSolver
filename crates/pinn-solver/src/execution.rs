//! Hardware-adaptive execution, Phase 1: an explicit seam for *how* host-side
//! work (resampling collocation points) gets executed, without yet adding any
//! real parallel/GPU-dispatch path behind it.
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
use pinn_core::messages::ExecutionConfig;
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
}
