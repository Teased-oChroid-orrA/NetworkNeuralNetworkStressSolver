//! Hardware-adaptive-execution epic, Phase 3: measures pin-lug's real
//! per-step host-side resample cost (pin interior + lug interior + lug
//! boundary, via `PinLugSamplingStrategy` - the trait-based path pin-lug's
//! production training loop actually calls every step, unconditionally,
//! unlike Kirsch's AMR-gated resample) at its real shipped default sizes
//! (`SolverConfig::default_pinlug()`: n_interior=2048, n_boundary=512).
//!
//! **This bench exists to answer one question**: is this the bottleneck
//! Phase 3's plan flagged it as a live candidate for parallelizing? A real
//! criterion run (30 samples, release profile) measured ~21.7us per full
//! step's resample - versus tens of milliseconds for `step_physics_multi`'s
//! actual tensor step (see Phase 2's real headless measurement in
//! `CLAUDE.md`). That's roughly 0.05-0.1% of step cost. **Conclusion: not a
//! bottleneck. Rayon was NOT added in Phase 3** - see `CLAUDE.md`'s Phase 3
//! section for the full reasoning. This bench is the permanent,
//! re-runnable record of that measurement, not a one-off probe - if a
//! future change to sampling (larger n_interior, a more expensive
//! rejection test, a different problem) changes this cost profile
//! materially, re-running this bench is how that would be caught, and
//! it's the evidence a future decision to revisit Rayon should be based on
//! rather than re-guessing.

use criterion::{criterion_group, criterion_main, black_box, Criterion};

use pinn_core::geometry::GeometryConfig;
use pinn_core::loading::LoadConfig;
use pinn_core::material::MaterialProps;

use pinn_solver::pinlug_problem::{PinLugProblem, PinLugScalingMode};
use pinn_solver::problem::BoundaryValueProblem;

fn bench_pinlug_resample(c: &mut Criterion) {
    let problem = PinLugProblem::new(
        MaterialProps::steel_4340(), 5, usize::MAX, 64, PinLugScalingMode::AppliedLoad,
    );
    let pin_geom = GeometryConfig::pinlug_pin_inches();
    let lug_geom = GeometryConfig::pinlug_lug_inches();
    let load = LoadConfig::default_10ksi();
    // SolverConfig::default_pinlug()'s real shipped sizes.
    let n_interior = 2048;
    let n_boundary = 512;

    let pin_strategy = problem.sampling_strategy(0);
    let lug_strategy = problem.sampling_strategy(1);

    c.bench_function("pinlug_resample/full_step_at_default_size", |b| {
        b.iter(|| {
            let pin_int = pin_strategy.sample_interior(&pin_geom, n_interior);
            let lug_int = lug_strategy.sample_interior(&lug_geom, n_interior);
            let lug_bnd = lug_strategy.sample_boundary(&lug_geom, &load, n_boundary);
            black_box((pin_int.len(), lug_int.len(), lug_bnd.len()))
        });
    });
}

criterion_group!(benches, bench_pinlug_resample);
criterion_main!(benches);
