//! Performance-regression benchmarks for `training_core::step_physics` (Kirsch's
//! frozen single-domain path) at five workload tiers, anchored to this
//! problem's own real config axes rather than arbitrary numbers - see
//! `pinn.env`/`SolverConfig::default_kirsch` for where `medium` comes from.
//!
//! Scope discipline (hardware-adaptive-execution epic, Phase 1): this file
//! only READS `step_physics` and other already-`pub` production setup
//! functions (`EngineParams::analyze`, `FdConfig::new`, `KirschProblem::new`,
//! `build_gathered_boundary_tensors`, etc. - the exact same functions
//! `headless::run_headless` calls at startup) to construct a representative
//! `StepCtx`. It does not modify `training_core.rs` or any other production
//! file, and it benchmarks Phase 1's SAW-BRDR vector (`engine.init_weights()`,
//! `phase2_active: false`) rather than Phase 2's curriculum stage - a
//! deliberate simplification since this bench's purpose is characterizing
//! `step_physics`'s cost curve as a function of problem size, not curriculum
//! behavior.
//!
//! Serial-only in this phase (nothing else exists yet to compare against) -
//! this harness is what a later phase's Rayon/GPU-backend comparison plugs
//! into once one exists.

use criterion::{criterion_group, criterion_main, black_box, Criterion};

use pinn_core::geometry::GeometryConfig;
use pinn_core::loading::LoadConfig;
use pinn_core::material::MaterialProps;
use pinn_core::messages::SolverConfig;
use pinn_core::sampling::{sample_boundary, sample_interior, sample_eq_ring};

use pinn_solver::engine::EngineParams;
use pinn_solver::fd_stencil::FdConfig;
use pinn_solver::kirsch_problem::KirschProblem;
use pinn_solver::lr_schedule::LrSchedule;
use pinn_solver::network::{ElasticityNet, ElasticityNetConfig};
use pinn_solver::optim::{make_bias_optim, make_gate_optim, WeightOptim};
use pinn_solver::problem::validate_loss_terms;
use pinn_solver::saw_brdr::SawBrdr;
use pinn_solver::training_core::{
    build_gathered_boundary_tensors, compute_reference_scales, extract_boundary_indices,
    normalize_point, step_physics, StepCtx, BDevice, B,
};

struct Tier {
    name: &'static str,
    n_interior: usize,
    n_boundary: usize,
    hidden_dim: usize,
    n_hidden: usize,
}

/// `medium` is Kirsch's actual shipped default (`pinn.env`'s `N_INTERIOR=4096`/
/// `N_BOUNDARY=1024`/`HIDDEN_DIM=128`/`N_HIDDEN=5`, `SolverConfig::default_kirsch`)
/// - the one tier that must match production exactly. The others are chosen to
/// bracket it: `tiny`/`small` for sub-second CI iteration, `large`/`very_large`
/// as headroom/stress tiers for a future Wgpu-vs-NdArray backend comparison.
const TIERS: &[Tier] = &[
    Tier { name: "tiny",       n_interior: 256,   n_boundary: 64,   hidden_dim: 32,  n_hidden: 2 },
    Tier { name: "small",      n_interior: 1024,  n_boundary: 256,  hidden_dim: 64,  n_hidden: 3 },
    Tier { name: "medium",     n_interior: 4096,  n_boundary: 1024, hidden_dim: 128, n_hidden: 5 },
    Tier { name: "large",      n_interior: 8192,  n_boundary: 2048, hidden_dim: 256, n_hidden: 6 },
    Tier { name: "very_large", n_interior: 16384, n_boundary: 4096, hidden_dim: 512, n_hidden: 8 },
];

fn bench_step_physics(c: &mut Criterion) {
    for tier in TIERS {
        let mut config = SolverConfig {
            material: MaterialProps::al7075_t6(),
            geometry: GeometryConfig::kirsch_plate_inches(),
            load: LoadConfig::default_10ksi(),
            n_interior: tier.n_interior,
            n_boundary: tier.n_boundary,
            hidden_dim: tier.hidden_dim,
            n_hidden: tier.n_hidden,
            ..SolverConfig::default_kirsch()
        };
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);
        let device = BDevice::default();

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        // Bench measures step_physics at the tier's FULL n_interior directly - bypassing
        // EngineParams::analyze's own Phase-1-curriculum floor (`phase1_n_interior =
        // (n_interior/16).max(512)`), since the bench's purpose is the steady-state
        // full-size step cost, not the curriculum's smaller early-phase step.
        let int_pts_phys = sample_interior(&config.geometry, config.n_interior);
        let int_norm: Vec<[f32; 2]> = int_pts_phys.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let bnd_pts = sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let gathered = build_gathered_boundary_tensors(&trac_idx, &hole_idx, &bnd_nx, &bnd_ny, &bnd_tx, &bnd_ty, &device);
        let eq_ring_norm: Vec<[f32; 2]> = sample_eq_ring(&config.geometry, engine.n_eq_ring)
            .iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let mut model: ElasticityNet<B> = net_cfg.init(&device);

        let problem = KirschProblem::new(config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt);
        validate_loss_terms(&problem);

        let mut optim_w = WeightOptim::new(config.use_soap_muon);
        let mut optim_b = make_bias_optim();
        let mut optim_gate = make_gate_optim();
        let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);

        let ctx = StepCtx {
            config: &config, engine: &engine, problem: &problem, fd: &fd,
            k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm: &int_norm, bnd_norm: &bnd_norm,
            bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
            trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx, gathered: &gathered,
            eq_ring_norm: &eq_ring_norm,
            dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
            phase2_active: false, step: 0,
        };

        c.bench_function(&format!("step_physics/{}", tier.name), |b| {
            b.iter(|| {
                let (m, out) = step_physics(
                    model.clone(), &mut optim_w, &mut optim_b, &mut optim_gate,
                    &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
                );
                model = m;
                black_box(out.total_scalar)
            });
        });
    }
}

criterion_group!(benches, bench_step_physics);
criterion_main!(benches);
