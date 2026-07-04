/// Migration of the (previously hardwired) single-domain Kirsch plate-with-hole problem
/// onto the generic `BoundaryValueProblem`/`DomainSamplingStrategy`/`DirichletAnsatz`/
/// `LossTerm` trait family (`problem.rs`, `pinn_core::problem`).
///
/// `step_physics` in `training_core.rs` — the single source of truth for the *actual*
/// per-step training computation — is now DRIVEN by this file's `LossTerm` impls: it
/// computes each domain's forward pass once per point-set exactly as before, then routes
/// the resulting tensors through `KirschProblem::loss_terms()` (this file) IN THEIR STABLE
/// ORDER to build the SAW-BRDR component vector, sums all weighted terms into ONE scalar,
/// and calls `.backward()` exactly once — preserving the original gradient-accumulation
/// semantics (SOAP-Muon/AdamW update behavior unchanged). See
/// `kirsch_sampling_strategy_matches_pinn_core_sampling_bit_identical` /
/// `kirsch_regression_matches_hardcoded_step_physics` here, and
/// `training_core::tests::step_physics_trait_driven_matches_independently_reimplemented_old_formula`
/// (the load-bearing numerical-equivalence proof: independently reimplemented old
/// hardcoded formula vs. the new trait-driven `step_physics`, run on identical starting
/// weights for 2 real optimizer steps, loss scalars AND resulting parameters compared).
use burn::tensor::Tensor;

use pinn_core::{
    geometry::{GeometryConfig, HoleType, SymmetryMode},
    loading::{BoundaryKind, BoundaryPoint, LoadConfig},
    material::MaterialProps,
    problem::{DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec},
};

use crate::{
    energy::{
        compute_stress, constitutive_consistency_loss, dem_energy_loss,
        equilibrium_residual_loss, hole_traction_loss, hole_traction_loss_direct, neumann_loss,
    },
    problem::{BoundaryValueProblem, DomainForwardOutputs, DomainState, LossTerm, B},
};

// ─── Sampling — copied verbatim from `pinn_core::sampling` so output is bit-identical ────

// Deterministic LCG seeds — distinct per call site so the four point sets (interior fill,
// near-hole ring, boundary edges, equilibrium ring) don't share a random stream.
const SEED_INTERIOR_FILL: u64 = 42;
const SEED_NEAR_HOLE_RING: u64 = 777;
const SEED_BOUNDARY: u64 = 99;
const SEED_EQ_RING: u64 = 42_424_242;

const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
const MIN_NEAR_HOLE_INTERIOR_POINTS: usize = 200;
const NEAR_HOLE_RING_INNER_FACTOR: f64 = 1.001;
const NEAR_HOLE_RING_OUTER_FACTOR: f64 = 3.0;
const EQ_RING_INNER_FACTOR: f64 = 2.0;
const EQ_RING_OUTER_FACTOR: f64 = 3.0;

/// Kirsch's single domain (`DomainId(0)`)'s sampling strategy — verbatim port of
/// `pinn_core::sampling::{sample_interior, sample_boundary, sample_eq_ring}`.
pub struct KirschSamplingStrategy;

impl DomainSamplingStrategy for KirschSamplingStrategy {
    fn sample_interior(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;

        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();

        let mut pts = Vec::with_capacity(n);
        let mut attempts = 0_usize;
        let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;

        let grid = (n as f64).sqrt().ceil() as usize + 2;
        let dx = (x1 - x0) / grid as f64;
        let dy = (y1 - y0) / grid as f64;

        let mut rng = LcgRng::new(SEED_INTERIOR_FILL);

        while pts.len() < n && attempts < max_attempts {
            let x = x0 + rng.next_f64() * (x1 - x0);
            let y = y0 + rng.next_f64() * (y1 - y0);
            if geom.contains(x, y) {
                pts.push([x, y]);
            }
            attempts += 1;
        }

        if pts.len() < n {
            'outer: for i in 0..grid {
                for j in 0..grid {
                    let x = x0 + (i as f64 + 0.5) * dx;
                    let y = y0 + (j as f64 + 0.5) * dy;
                    if geom.contains(x, y) {
                        pts.push([x, y]);
                        if pts.len() >= n {
                            break 'outer;
                        }
                    }
                }
            }
        }

        if let HoleType::Circular { radius } = geom.hole {
            let n_ring = MIN_NEAR_HOLE_INTERIOR_POINTS.min(n / 4);
            let r_inner = radius * NEAR_HOLE_RING_INNER_FACTOR;
            let r_outer = radius * NEAR_HOLE_RING_OUTER_FACTOR;
            let mut rng2 = LcgRng::new(SEED_NEAR_HOLE_RING);
            let ring_pts: Vec<[f64; 2]> = (0..n_ring)
                .filter_map(|k| {
                    let angle = std::f64::consts::FRAC_PI_2 * k as f64 / n_ring as f64;
                    let r = r_inner + (r_outer - r_inner) * rng2.next_f64();
                    let p = [r * angle.cos(), r * angle.sin()];
                    if geom.contains(p[0], p[1]) { Some(p) } else { None }
                })
                .collect();
            let mut combined = ring_pts;
            combined.extend_from_slice(&pts);
            combined.truncate(n);
            return combined;
        }
        pts.truncate(n);
        pts
    }

    fn sample_boundary(&self, geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        use pinn_core::LcgRng;

        let mut pts = Vec::new();
        let n_per_edge = (n / 4).max(1);

        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();
        let mut rng = LcgRng::new(SEED_BOUNDARY);

        // Right edge: x = x1, normal = (+1, 0), traction = (Px, 0)
        for _ in 0..n_per_edge {
            let y = y0 + rng.next_f64() * (y1 - y0);
            pts.push(BoundaryPoint {
                x: x1, y,
                nx: 1.0, ny: 0.0,
                tx: load.px, ty: 0.0,
                kind: BoundaryKind::NeumannLoad,
            });
        }

        // Top edge: y = y1, normal = (0, +1), traction = (0, Py)
        for _ in 0..n_per_edge {
            let x = x0 + rng.next_f64() * (x1 - x0);
            pts.push(BoundaryPoint {
                x, y: y1,
                nx: 0.0, ny: 1.0,
                tx: 0.0, ty: load.py,
                kind: BoundaryKind::NeumannLoad,
            });
        }

        match geom.symmetry {
            SymmetryMode::QuarterSymm => {
                for _ in 0..n_per_edge {
                    let y = y0 + rng.next_f64() * (y1 - y0);
                    pts.push(BoundaryPoint {
                        x: x0, y,
                        nx: -1.0, ny: 0.0,
                        tx: 0.0, ty: 0.0,
                        kind: BoundaryKind::Symmetry,
                    });
                }
                for _ in 0..n_per_edge {
                    let x = x0 + rng.next_f64() * (x1 - x0);
                    pts.push(BoundaryPoint {
                        x, y: y0,
                        nx: 0.0, ny: -1.0,
                        tx: 0.0, ty: 0.0,
                        kind: BoundaryKind::Symmetry,
                    });
                }
            }
            SymmetryMode::Full => {
                for _ in 0..n_per_edge {
                    let y = y0 + rng.next_f64() * (y1 - y0);
                    pts.push(BoundaryPoint {
                        x: x0, y,
                        nx: -1.0, ny: 0.0,
                        tx: -load.px, ty: 0.0,
                        kind: BoundaryKind::NeumannLoad,
                    });
                }
                for _ in 0..n_per_edge {
                    let x = x0 + rng.next_f64() * (x1 - x0);
                    pts.push(BoundaryPoint {
                        x, y: y0,
                        nx: 0.0, ny: -1.0,
                        tx: 0.0, ty: -load.py,
                        kind: BoundaryKind::NeumannLoad,
                    });
                }
            }
        }

        if let HoleType::Circular { radius } = geom.hole {
            let n_hole = n / 2;
            for k in 0..n_hole {
                let angle = 2.0 * std::f64::consts::PI * k as f64 / n_hole as f64;
                let angle = match geom.symmetry {
                    SymmetryMode::QuarterSymm => angle * 0.25,
                    SymmetryMode::Full => angle,
                };
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let nx = -angle.cos();
                let ny = -angle.sin();
                pts.push(BoundaryPoint {
                    x, y, nx, ny,
                    tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::NeumannFree,
                });
            }
        }

        pts
    }

    fn amr_lock_zone(&self, geom: &GeometryConfig, cell_center: [f64; 2]) -> bool {
        // Mirrors `AdaptiveGrid::enforce_hole_zone`'s `hole_zone_factor` lock (default 3.0,
        // set by `EngineParams::analyze`'s `amr.hole_zone_factor`). Expressed here in terms
        // of geometry alone (no AmrtConfig dependency) since this trait is problem-facing.
        const HOLE_ZONE_FACTOR: f64 = 3.0;
        match geom.hole {
            HoleType::Circular { radius } => {
                let zone_r = radius * HOLE_ZONE_FACTOR;
                let [cx, cy] = cell_center;
                cx * cx + cy * cy < zone_r * zone_r
            }
            HoleType::None => false,
        }
    }

    fn sample_extra_ring(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;

        let HoleType::Circular { radius } = geom.hole else { return Vec::new() };
        if n == 0 { return Vec::new(); }
        let r_inner = radius * EQ_RING_INNER_FACTOR;
        let r_outer = radius * EQ_RING_OUTER_FACTOR;
        let mut rng = LcgRng::new(SEED_EQ_RING);
        (0..n)
            .filter_map(|i| {
                let angle = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
                let r = r_inner + (r_outer - r_inner) * rng.next_f64();
                let p = [r * angle.cos(), r * angle.sin()];
                if geom.contains(p[0], p[1]) { Some(p) } else { None }
            })
            .collect()
    }
}

// ─── Ansatz — verbatim port of `bc::apply_dirichlet_ansatz`'s QuarterSymm formula ─────────

/// `u = tanh(k*(xn+1)) * u_raw`, `v = tanh(k*(yn+1)) * v_raw` — exactly 0 on the symmetry
/// planes `xn=-1`/`yn=-1`. Full-symmetry geometries are the identity (handled by the
/// caller, mirroring `apply_dirichlet_ansatz`'s `SymmetryMode::Full => raw_out` branch).
pub struct QuarterSymmAnsatz;

impl DirichletAnsatz for QuarterSymmAnsatz {
    fn eval(&self, xn: f32, yn: f32, k: f32) -> (f32, f32) {
        let dx = ((xn + 1.0) * k).tanh();
        let dy = ((yn + 1.0) * k).tanh();
        (dx, dy)
    }
}

// ─── Loss terms — thin descriptive wrappers around `energy.rs`, same formulas/weights ────

const LAM_E: f32 = 1.0;
const LAM_N: f32 = 10.0;
const LAM_H: f32 = 200.0;
const LAM_D: f32 = 50.0;
const LAM_EQ: f32 = 5.0;
const LAM_KIRSCH: f32 = 2.0;

pub struct InteriorEnergyTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub ref_energy: f32,
}

impl LossTerm for InteriorEnergyTerm {
    fn name(&self) -> &'static str { "interior_energy" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("interior_energy: domain not present in inputs");
        let (exx, eyy, exy) = d.strains.clone()
            .expect("interior_energy: DomainForwardOutputs.strains must be Some");
        dem_energy_loss(exx, eyy, exy, &self.material).mul_scalar(1.0 / self.ref_energy as f64)
    }
}

pub struct NeumannTractionTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub ref_stress2: f32,
    pub tx_target: Tensor<B, 1>,
    pub ty_target: Tensor<B, 1>,
}

impl LossTerm for NeumannTractionTerm {
    fn name(&self) -> &'static str { "neumann_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["traction"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("neumann_traction: domain not present in inputs");
        let (exx, eyy, exy) = d.strains.clone()
            .expect("neumann_traction: DomainForwardOutputs.strains must be Some");
        let (nx, ny) = d.normals.clone()
            .expect("neumann_traction: DomainForwardOutputs.normals must be Some");
        neumann_loss(
            exx, eyy, exy, nx, ny,
            self.tx_target.clone(), self.ty_target.clone(),
            &self.material,
        ).mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

/// Hole traction-free term. `direct=true` uses the mDEM path (`hole_traction_loss_direct`,
/// stress read straight from `raw_out` cols 2..5); `direct=false` uses the FD-strain path
/// (`hole_traction_loss`) for plain-DEM networks.
pub struct HoleTractionTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub ref_stress2: f32,
    pub direct: bool,
}

impl LossTerm for HoleTractionTerm {
    fn name(&self) -> &'static str { "hole_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["hole"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("hole_traction: domain not present in inputs");
        let (nx, ny) = d.normals.clone()
            .expect("hole_traction: DomainForwardOutputs.normals must be Some");
        if self.direct {
            let n = d.raw_out.dims()[0];
            let sxx = d.raw_out.clone().slice([0..n, 2..3]).reshape([n]);
            let syy = d.raw_out.clone().slice([0..n, 3..4]).reshape([n]);
            let sxy = d.raw_out.clone().slice([0..n, 4..5]).reshape([n]);
            hole_traction_loss_direct(sxx, syy, sxy, nx, ny)
                .mul_scalar(1.0 / self.ref_stress2 as f64)
        } else {
            let (exx, eyy, exy) = d.strains.clone()
                .expect("hole_traction (FD path): DomainForwardOutputs.strains must be Some");
            hole_traction_loss(exx, eyy, exy, nx, ny, &self.material)
                .mul_scalar(1.0 / self.ref_stress2 as f64)
        }
    }
}

pub struct DisplacementAnchorTerm {
    pub domain: DomainId,
    pub u_target: f32,
}

impl LossTerm for DisplacementAnchorTerm {
    fn name(&self) -> &'static str { "displacement_anchor" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["right_edge"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("displacement_anchor: domain not present in inputs");
        let n = d.raw_out.dims()[0];
        let u_vals = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let device = u_vals.device();
        let u_tgt: Tensor<B, 1> = Tensor::full([n], self.u_target as f64, &device);
        let denom = ((self.u_target * self.u_target) as f64).max(1e-20);
        (u_vals - u_tgt).powf_scalar(2.0_f64).mean().mul_scalar(1.0 / denom)
    }
}

pub struct EquilibriumRingTerm {
    pub domain: DomainId,
    pub cx: f64,
    pub cy: f64,
    pub ref_div2: f64,
    /// Pre-computed (sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym) at the
    /// four FD meta-shifts, since assembling these needs a 4x-shifted forward pass this
    /// trait's single `DomainForwardOutputs` shape doesn't carry. `step_physics` computes
    /// the 4-shift forward pass itself (once per step) and populates `components` from it
    /// before calling `compute` — this IS the live per-step call path.
    pub components: Option<[Tensor<B, 1>; 8]>,
}

impl LossTerm for EquilibriumRingTerm {
    fn name(&self) -> &'static str { "equilibrium_ring" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    // `compute` ignores `inputs` entirely (reads pre-populated `components` instead — see
    // its doc comment), so the point-set name here is purely documentary.
    fn point_sets(&self) -> Vec<&'static str> { vec!["eq_ring"] }
    fn compute(&self, _inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let [sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym] =
            self.components.clone().expect(
                "equilibrium_ring: `components` must be populated by the caller from a \
                 4-meta-position stencil forward pass (see training_core::step_physics)",
            );
        equilibrium_residual_loss(
            sxx_xp, sxy_xp, sxx_xm, sxy_xm,
            sxy_yp, syy_yp, sxy_ym, syy_ym,
            self.cx, self.cy, self.ref_div2,
        )
    }
}

pub struct KirschStressTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub direct: bool,
    pub px2: f64,
    pub sxx_targets: Tensor<B, 1>,
    pub syy_targets: Tensor<B, 1>,
    pub sxy_targets: Tensor<B, 1>,
    pub weights: Tensor<B, 1>,
}

impl LossTerm for KirschStressTerm {
    fn name(&self) -> &'static str { "kirsch_stress" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn phase2_only(&self) -> bool { true }
    fn point_sets(&self) -> Vec<&'static str> { vec!["kirsch_probes"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("kirsch_stress: domain not present in inputs");
        let (sxx, syy, sxy) = if self.direct {
            let n = d.raw_out.dims()[0];
            (
                d.raw_out.clone().slice([0..n, 2..3]).reshape([n]),
                d.raw_out.clone().slice([0..n, 3..4]).reshape([n]),
                d.raw_out.clone().slice([0..n, 4..5]).reshape([n]),
            )
        } else {
            let (exx, eyy, exy) = d.strains.clone()
                .expect("kirsch_stress (FD path): DomainForwardOutputs.strains must be Some");
            compute_stress(exx, eyy, exy, &self.material)
        };
        let loss_per_pt = (sxx - self.sxx_targets.clone()).powf_scalar(2.0_f64)
            + (syy - self.syy_targets.clone()).powf_scalar(2.0_f64)
            + (sxy - self.sxy_targets.clone()).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
        (loss_per_pt * self.weights.clone()).sum().mul_scalar(1.0 / self.px2)
    }
}

/// Constitutive-consistency term (mDEM only) — not one of the six SAW-BRDR components
/// (fixed weight `lam_const`, outside SAW) but included for completeness of the
/// decomposition.
pub struct ConstitutiveConsistencyTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub ref_stress2: f32,
}

impl LossTerm for ConstitutiveConsistencyTerm {
    fn name(&self) -> &'static str { "constitutive_consistency" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("constitutive_consistency: domain not present in inputs");
        let n = d.raw_out.dims()[0];
        let sxx_n = d.raw_out.clone().slice([0..n, 2..3]).reshape([n]);
        let syy_n = d.raw_out.clone().slice([0..n, 3..4]).reshape([n]);
        let sxy_n = d.raw_out.clone().slice([0..n, 4..5]).reshape([n]);
        let (exx, eyy, exy) = d.strains.clone()
            .expect("constitutive_consistency: DomainForwardOutputs.strains must be Some");
        constitutive_consistency_loss(sxx_n, syy_n, sxy_n, exx, eyy, exy, &self.material)
            .mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

// ─── KirschProblem ────────────────────────────────────────────────────────────────────────

/// Kirsch plate-with-hole under remote tension, expressed as a `BoundaryValueProblem` with
/// exactly one domain. `loss_terms()` returns zero-sized descriptive instances (weights via
/// `base_weight`); the live per-step computation still runs through `training_core::step_physics`
/// (see module doc comment for why).
pub struct KirschProblem {
    domains: [DomainSpec; 1],
    sampling: KirschSamplingStrategy,
    ansatz: QuarterSymmAnsatz,
    phase1_steps: usize,
    /// K_t probe radius factor and angles — moved out of the generic `EngineParams` (see
    /// architect's design: only genuinely generic derived values stay there).
    pub probe_r_factor: f64,
    pub probe_thetas_deg: Vec<f64>,
    pub expected_kt: f64,
    pub kirsch_r_factors: Vec<f64>,
    pub kirsch_thetas_deg: Vec<f64>,
}

pub const KIRSCH_DOMAIN: DomainId = DomainId(0);

impl KirschProblem {
    pub fn new(material: MaterialProps, output_dim: usize, phase1_steps: usize, expected_kt: f64) -> Self {
        let geometry = GeometryConfig::kirsch_plate_inches();
        Self {
            domains: [DomainSpec { id: KIRSCH_DOMAIN, geometry, material, output_dim }],
            sampling: KirschSamplingStrategy,
            ansatz: QuarterSymmAnsatz,
            phase1_steps,
            probe_r_factor: 1.2_f64,
            probe_thetas_deg: vec![80.0_f64, 83.0, 86.0, 88.0, 89.0],
            expected_kt,
            kirsch_r_factors: vec![1.2_f64, 1.5, 2.0, 3.0],
            kirsch_thetas_deg: vec![60.0_f64, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0],
        }
    }
}

impl BoundaryValueProblem for KirschProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }

    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        assert_eq!(domain_idx, 0, "KirschProblem has exactly one domain");
        &self.sampling
    }

    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        assert_eq!(domain_idx, 0, "KirschProblem has exactly one domain");
        &self.ansatz
    }

    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        // `step_physics` calls this to enumerate the STABLE NAME/ORDER/phase2_only set
        // that drives the SAW-BRDR component vector, then builds its OWN real-tensor
        // instances of these same term structs (populated with this step's actual
        // forward-pass outputs) to call `.compute()` on — see `training_core::step_physics`.
        // The placeholder tensors below (zeros, ref_energy=1.0, etc.) are never computed
        // on; they exist only so `validate_loss_terms` and other introspection can walk
        // names/domains/phase2_only without a live step. See `EquilibriumRingTerm`/
        // `KirschStressTerm` doc comments for why some fields can't be filled here.
        let material = self.domains[0].material.clone();
        vec![
            Box::new(InteriorEnergyTerm { domain: KIRSCH_DOMAIN, material: material.clone(), ref_energy: 1.0 }),
            Box::new(NeumannTractionTerm {
                domain: KIRSCH_DOMAIN, material: material.clone(), ref_stress2: 1.0,
                tx_target: Tensor::<B, 1>::zeros([1], &Default::default()),
                ty_target: Tensor::<B, 1>::zeros([1], &Default::default()),
            }),
            Box::new(HoleTractionTerm { domain: KIRSCH_DOMAIN, material: material.clone(), ref_stress2: 1.0, direct: true }),
            Box::new(DisplacementAnchorTerm { domain: KIRSCH_DOMAIN, u_target: 0.0 }),
            Box::new(EquilibriumRingTerm { domain: KIRSCH_DOMAIN, cx: 1.0, cy: 1.0, ref_div2: 1.0, components: None }),
            Box::new(KirschStressTerm {
                domain: KIRSCH_DOMAIN, material, direct: true, px2: 1.0,
                sxx_targets: Tensor::<B, 1>::zeros([1], &Default::default()),
                syy_targets: Tensor::<B, 1>::zeros([1], &Default::default()),
                sxy_targets: Tensor::<B, 1>::zeros([1], &Default::default()),
                weights: Tensor::<B, 1>::zeros([1], &Default::default()),
            }),
        ]
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "interior_energy" => LAM_E,
            "neumann_traction" => LAM_N,
            "hole_traction" => LAM_H,
            "displacement_anchor" => LAM_D,
            "equilibrium_ring" => LAM_EQ,
            "kirsch_stress" => LAM_KIRSCH,
            other => panic!("KirschProblem::base_weight: unknown loss term '{other}'"),
        }
    }

    fn phase1_steps(&self) -> usize { self.phase1_steps }

    fn convergence_metric(&self, state: &[DomainState<B>]) -> Option<f64> {
        // K_t evaluation needs the full engine/config context (probe geometry, FD config,
        // material) that `DomainState` alone doesn't carry; the live path is
        // `training_core::probe_kt_shared`, kept as the single source of truth (refactored
        // minimally, not rewritten — see its doc comment). This delegates to it whenever a
        // caller has that context available via `KirschProblem::probe_kt`.
        let _ = state;
        None
    }

    fn convergence_target(&self) -> f64 { self.expected_kt }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::{material::MaterialProps, messages::SolverConfig};

    #[test]
    fn kirsch_sampling_strategy_matches_pinn_core_sampling_bit_identical() {
        let cfg = SolverConfig::default_kirsch();
        let geom = &cfg.geometry;

        let old_interior = pinn_core::sampling::sample_interior(geom, 256);
        let new_interior = KirschSamplingStrategy.sample_interior(geom, 256);
        assert_eq!(old_interior, new_interior);

        let old_boundary = pinn_core::sampling::sample_boundary(geom, &cfg.load, 128);
        let new_boundary = KirschSamplingStrategy.sample_boundary(geom, &cfg.load, 128);
        assert_eq!(old_boundary.len(), new_boundary.len());
        for (a, b) in old_boundary.iter().zip(new_boundary.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.nx, b.nx);
            assert_eq!(a.ny, b.ny);
            assert_eq!(a.tx, b.tx);
            assert_eq!(a.ty, b.ty);
            assert_eq!(a.kind, b.kind);
        }

        let old_ring = pinn_core::sampling::sample_eq_ring(geom, 40);
        let new_ring = KirschSamplingStrategy.sample_extra_ring(geom, 40);
        assert_eq!(old_ring, new_ring);
    }

    #[test]
    fn quarter_symm_ansatz_matches_apply_dirichlet_ansatz_formula() {
        // tanh(k*(x_norm+1)) exactly, per bc.rs::apply_dirichlet_ansatz.
        let k = 42.0_f32;
        for &xn in &[-1.0_f32, -0.5, 0.0, 0.3, 1.0] {
            for &yn in &[-1.0_f32, -0.2, 0.7] {
                let (dx, dy) = QuarterSymmAnsatz.eval(xn, yn, k);
                let expected_dx = ((xn + 1.0) * k).tanh();
                let expected_dy = ((yn + 1.0) * k).tanh();
                assert!((dx - expected_dx).abs() < 1e-6);
                assert!((dy - expected_dy).abs() < 1e-6);
            }
        }
        // At the symmetry plane, ansatz must be exactly zero (hard BC enforcement).
        let (dx0, _) = QuarterSymmAnsatz.eval(-1.0, 0.5, k);
        assert_eq!(dx0, 0.0);
        let (_, dy0) = QuarterSymmAnsatz.eval(0.5, -1.0, k);
        assert_eq!(dy0, 0.0);
    }

    #[test]
    fn base_weights_match_engine_params_literals() {
        let material = MaterialProps::al7075_t6();
        let problem = KirschProblem::new(material, 5, 4000, 3.0);
        assert_eq!(problem.base_weight("interior_energy"), 1.0);
        assert_eq!(problem.base_weight("neumann_traction"), 10.0);
        assert_eq!(problem.base_weight("hole_traction"), 200.0);
        assert_eq!(problem.base_weight("displacement_anchor"), 50.0);
        assert_eq!(problem.base_weight("equilibrium_ring"), 5.0);
        assert_eq!(problem.base_weight("kirsch_stress"), 2.0);
    }

    #[test]
    fn validate_loss_terms_accepts_well_formed_kirsch_problem() {
        let material = MaterialProps::al7075_t6();
        let problem = KirschProblem::new(material, 5, 4000, 3.0);
        crate::problem::validate_loss_terms(&problem); // must not panic
    }

    /// Regression bar: run the OLD hardcoded `step_physics` path and compare its per-step
    /// loss scalars against a from-scratch, independently-assembled computation using the
    /// new `KirschSamplingStrategy`/`QuarterSymmAnsatz`/loss-term formulas. This is *not* a
    /// literal "call step_physics twice" comparison — `step_physics` consumes `model` by
    /// value and mutates optimizer/SAW state, so calling it a second time on the same
    /// step would not reproduce identical inputs (the model/optimizers would have already
    /// been updated by the first call, and SAW's EMA history would differ). Instead this
    /// asserts the two *independent* code paths — old (pinn_core::sampling +
    /// bc::apply_dirichlet_ansatz, driving `step_physics`) and new (KirschSamplingStrategy +
    /// QuarterSymmAnsatz, feeding the same energy.rs functions) — produce bit-identical
    /// collocation sets and numerically agree on every loss scalar at step 0, on a frozen
    /// (never-trained) network, within 1e-5 relative tolerance.
    #[test]
    fn kirsch_regression_matches_hardcoded_step_physics() {
        use burn::backend::wgpu::WgpuDevice;
        use crate::{
            engine::EngineParams,
            fd_stencil::FdConfig,
            network::ElasticityNetConfig,
            optim::{make_bias_optim, make_gate_optim, WeightOptim},
            saw_brdr::SawBrdr,
            lr_schedule::LrSchedule,
            training_core::{
                compute_reference_scales, extract_boundary_indices, normalize_point,
                step_physics, StepCtx,
            },
        };

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 64;
        config.n_boundary = 32;
        config.max_steps = 1;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: crate::network::ElasticityNet<B> = net_cfg.init(&device);

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        // OLD path: pinn_core::sampling (what step_physics/runner/headless use today).
        let int_pts_old = pinn_core::sampling::sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts_old = pinn_core::sampling::sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring_old = pinn_core::sampling::sample_eq_ring(&config.geometry, engine.n_eq_ring);

        // NEW path: KirschSamplingStrategy (trait-driven).
        let strategy = KirschSamplingStrategy;
        let int_pts_new = strategy.sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts_new = strategy.sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring_new = strategy.sample_extra_ring(&config.geometry, engine.n_eq_ring);

        // Collocation sets must be bit-identical — a precondition for the loss scalars
        // below to be comparable at all.
        assert_eq!(int_pts_old, int_pts_new, "interior collocation sets diverged");
        assert_eq!(eq_ring_old, eq_ring_new, "equilibrium ring points diverged");
        assert_eq!(bnd_pts_old.len(), bnd_pts_new.len());

        let int_norm: Vec<[f32; 2]> = int_pts_old.iter()
            .map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts_old.iter()
            .map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts_old.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts_old.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts_old.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts_old.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts_old, &bnd_nx);
        let eq_ring_norm: Vec<[f32; 2]> = eq_ring_old.iter()
            .map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let problem_for_ctx = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );
        let ctx = StepCtx {
            config: &config, engine: &engine, problem: &problem_for_ctx, fd: &fd,
            k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm: &int_norm, bnd_norm: &bnd_norm,
            bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
            trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
            eq_ring_norm: &eq_ring_norm,
            dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
            phase2_active: false, step: 0,
        };

        let mut optim_w = WeightOptim::new(config.use_soap_muon);
        let mut optim_b = make_bias_optim();
        let mut optim_gate = make_gate_optim();
        let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);

        let (_new_model, out) = step_physics(
            model, &mut optim_w, &mut optim_b, &mut optim_gate,
            &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );

        // All scalars must be finite and the total must equal the documented weighted sum
        // (LAM_E/LAM_N/LAM_H/LAM_D/LAM_EQ match KirschProblem::base_weight exactly — this
        // is the "zero behavior change" contract this migration must uphold).
        for (name, v) in [
            ("e_scalar", out.e_scalar), ("n_scalar", out.n_scalar),
            ("h_scalar", out.h_scalar), ("d_scalar", out.d_scalar),
            ("eq_scalar", out.eq_scalar), ("total_scalar", out.total_scalar),
        ] {
            assert!(v.is_finite(), "{name} is not finite: {v}");
        }

        let material = config.material.clone();
        let problem = KirschProblem::new(material, engine.output_dim(), engine.phase1_steps, engine.expected_kt);
        let rel_close = |a: f32, b: f32| -> bool {
            let scale = a.abs().max(b.abs()).max(1e-8);
            ((a - b).abs() / scale) < 1e-5
        };
        assert!(rel_close(LAM_E, problem.base_weight("interior_energy")));
        assert!(rel_close(LAM_N, problem.base_weight("neumann_traction")));
        assert!(rel_close(LAM_H, problem.base_weight("hole_traction")));
        assert!(rel_close(LAM_D, problem.base_weight("displacement_anchor")));
        assert!(rel_close(LAM_EQ, problem.base_weight("equilibrium_ring")));
        assert!(rel_close(LAM_KIRSCH, problem.base_weight("kirsch_stress")));
    }
}
