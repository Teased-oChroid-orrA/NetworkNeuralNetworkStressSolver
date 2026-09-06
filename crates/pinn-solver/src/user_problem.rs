//! User-defined problem ingestion: lets a user design a new 2D elasticity joint (a
//! rectangular plate with N circular holes, each either traction-free or fixed) from a
//! [`pinn_core::problem_spec::ProblemSpec`] instead of writing new Rust — the missing piece
//! `--headless` dispatch didn't have (it only ever routed `ProblemKind::{Kirsch,PinLug}`
//! through concrete `SolverConfig`s). Drives the SAME `BoundaryValueProblem`/`LossTerm`/
//! `DomainSamplingStrategy` trait family Kirsch/pin-lug use, through the already-generic
//! `step_physics_multi` driver (`training_core.rs`) — no new physics math, no changes to any
//! existing file's behavior.
//!
//! **Essential (Dirichlet) BCs are soft-penalty, like pin-lug's, not a hard ansatz like
//! Kirsch's.** There is no closed-form scale factor that zeroes displacement on an arbitrary
//! circle at an arbitrary position while leaving the rest of an arbitrary-hole-count domain
//! free — Kirsch's `tanh((xn+1)*k)` only works because its one hole is always centered at
//! the origin under quarter-symmetry. So this uses [`crate::pinlug_problem::IdentityAnsatz`]
//! plus a soft `mean(u^2+v^2)` anchor penalty per `Fixed` hole, exactly mirroring pin-lug's
//! own `LugShankAnchorTerm`.
//!
//! Every loss term here is a near-verbatim copy of an existing Kirsch/pin-lug term's shape,
//! reusing `energy.rs`'s already target-parametrized, geometry-agnostic functions
//! (`dem_energy_loss`/`neumann_loss`/`hole_traction_loss_direct`) — see each term's own doc
//! comment for which one it mirrors.
//!
//! Uses mDEM (`output_dim=5`: u,v,sigma_xx,sigma_yy,sigma_xy) so free-hole/anchor terms can
//! read direct network stress/displacement columns, matching pin-lug's own
//! `LugFreeEdgeTractionTerm`/`LugShankAnchorTerm` convention.

use burn::tensor::Tensor;

use pinn_core::{
    geometry::GeometryConfig,
    loading::{BoundaryKind, BoundaryPoint, LoadConfig},
    material::MaterialProps,
    problem::{
        DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec, NamedPointSet,
    },
    problem_spec::ProblemSpec,
    user_geometry::{HoleBc, UserGeometry},
};

use crate::{
    energy::{dem_energy_loss, hole_traction_loss_direct, neumann_loss},
    pinlug_problem::IdentityAnsatz,
    problem::{BoundaryValueProblem, ConflictGroup, DomainForwardOutputs, DomainState, LossTerm, B},
};

pub const USER_DOMAIN: DomainId = DomainId(0);

const LAM_INTERIOR_ENERGY: f32 = 1.0;
const LAM_OUTER_TRACTION: f32 = 10.0;
const LAM_HOLE_FREE: f32 = 100.0;
const LAM_HOLE_FIXED: f32 = 50.0;

/// Points sampled around each hole's circumference, per hole — a fixed, generous default;
/// not user-configurable in v1 (see `ProblemSpec`'s scope note).
const HOLE_RING_POINTS: usize = 64;
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
const SEED_INTERIOR: u64 = 90_210;

/// Per-domain sampling strategy driven directly by a [`UserGeometry`] — ignores the
/// `&GeometryConfig` parameter every [`DomainSamplingStrategy`] method takes (an
/// established, precedented pattern — see `FakeInterfaceSampling` in
/// `pinn-core/src/problem.rs`'s own tests) in favor of its own captured, real N-hole
/// geometry.
pub struct UserSamplingStrategy {
    pub geometry: UserGeometry,
    /// `"hole_0"`, `"hole_1"`, ... — leaked once here (not per-step) so
    /// [`Self::named_point_sets`] can return the same content-stable `&'static str` names
    /// every call without leaking memory continuously across a long training run.
    hole_names: Vec<&'static str>,
}

impl UserSamplingStrategy {
    pub fn new(geometry: UserGeometry) -> Self {
        let hole_names = (0..geometry.holes.len())
            .map(|i| -> &'static str { Box::leak(format!("hole_{i}").into_boxed_str()) })
            .collect();
        Self { geometry, hole_names }
    }
}

impl DomainSamplingStrategy for UserSamplingStrategy {
    fn sample_interior(&self, _geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;
        let mut pts = Vec::with_capacity(n);
        let mut attempts = 0usize;
        let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;
        let mut rng = LcgRng::new(SEED_INTERIOR);
        while pts.len() < n && attempts < max_attempts {
            attempts += 1;
            let x = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_w;
            let y = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_h;
            if self.geometry.contains(x, y) {
                pts.push([x, y]);
            }
        }
        pts
    }

    /// The 4 outer rectangle edges only — hole boundaries come back via
    /// [`Self::named_point_sets`] instead, one named set per hole.
    fn sample_boundary(&self, _geom: &GeometryConfig, _load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        let per_edge = (n / 4).max(1);
        let mut pts = Vec::with_capacity(per_edge * 4);
        let hw = self.geometry.half_w;
        let hh = self.geometry.half_h;
        for i in 0..per_edge {
            let frac = (i as f64 + 0.5) / per_edge as f64; // (0,1), avoids exact corners
            let along_w = -hw + 2.0 * hw * frac;
            let along_h = -hh + 2.0 * hh * frac;
            pts.push(BoundaryPoint { x: hw, y: along_h, nx: 1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad });
            pts.push(BoundaryPoint { x: -hw, y: along_h, nx: -1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad });
            pts.push(BoundaryPoint { x: along_w, y: hh, nx: 0.0, ny: 1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad });
            pts.push(BoundaryPoint { x: along_w, y: -hh, nx: 0.0, ny: -1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad });
        }
        pts
    }

    /// No AMR in v1 — never locks a cell.
    fn amr_lock_zone(&self, _geom: &GeometryConfig, _cell_center: [f64; 2]) -> bool {
        false
    }

    /// No equilibrium-ring probe in v1 (matches pin-lug, which also doesn't use one).
    fn sample_extra_ring(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> {
        Vec::new()
    }

    fn named_point_sets(&self, _bnd_pts: &[BoundaryPoint]) -> Vec<NamedPointSet> {
        self.geometry.holes.iter().zip(self.hole_names.iter()).map(|(hole, &name)| {
            let points = (0..HOLE_RING_POINTS).map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / HOLE_RING_POINTS as f64;
                let (nx, ny) = (theta.cos(), theta.sin());
                BoundaryPoint {
                    x: hole.center[0] + hole.radius * nx,
                    y: hole.center[1] + hole.radius * ny,
                    nx: -nx, // outward from the PLATE means inward toward the hole center
                    ny: -ny,
                    tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::NeumannFree,
                }
            }).collect();
            NamedPointSet { name, points }
        }).collect()
    }
}

/// Mirrors `pinlug_problem::InteriorEnergyTerm` exactly (`dem_energy_loss`, generic, no new
/// math) — the default `point_sets()` ("interior") applies unchanged.
struct InteriorEnergyTerm {
    material: MaterialProps,
    ref_energy: f32,
}
impl LossTerm for InteriorEnergyTerm {
    fn name(&self) -> &'static str { "interior_energy" }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("interior_energy: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("interior_energy: strains must be Some");
        dem_energy_loss(exx, eyy, exy, &self.material).mul_scalar(1.0 / self.ref_energy as f64)
    }
}

/// Far-field applied traction on the 4 outer edges. Mirrors `kirsch_problem::
/// NeumannTractionTerm`/`pinlug_problem::PinDrivingTractionTerm`'s shape (`neumann_loss`,
/// FD-derived strains), but computes its target INLINE from `d.normals` each call
/// (`tx_target = px * nx`, `ty_target = py * ny` — the standard Cauchy traction formula
/// `t = sigma . n` for a far-field stress state `diag(px, py)`) rather than a separately
/// pre-built Tensor field, since normals are already guaranteed fresh every step from
/// whatever points this problem's own sampling strategy just generated — simpler than
/// threading a stale target tensor through resampling, and the same physics either way.
struct OuterTractionTerm {
    material: MaterialProps,
    ref_stress2: f32,
    px: f64,
    py: f64,
}
impl LossTerm for OuterTractionTerm {
    fn name(&self) -> &'static str { "outer_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("outer_traction: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("outer_traction: strains must be Some");
        let (nx, ny) = d.normals.clone().expect("outer_traction: normals must be Some");
        let tx_target = nx.clone().mul_scalar(self.px);
        let ty_target = ny.clone().mul_scalar(self.py);
        neumann_loss(exx, eyy, exy, nx, ny, tx_target, ty_target, &self.material)
            .mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

/// One hole's boundary condition — `Free` mirrors `pinlug_problem::LugFreeEdgeTractionTerm`
/// (`hole_traction_loss_direct` on direct mDEM stress columns, implicit zero target);
/// `Fixed` mirrors `pinlug_problem::LugShankAnchorTerm` (`mean(u^2+v^2)` on direct mDEM
/// displacement columns).
struct HoleBcTerm {
    point_set: &'static str,
    bc: HoleBc,
    ref_stress2: f32,
}
impl LossTerm for HoleBcTerm {
    fn name(&self) -> &'static str {
        match self.bc { HoleBc::Free => "hole_free", HoleBc::Fixed => "hole_fixed" }
    }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec![self.point_set] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("hole_bc: domain missing");
        let n = d.raw_out.dims()[0];
        match self.bc {
            HoleBc::Free => {
                let (nx, ny) = d.normals.clone().expect("hole_bc(free): normals must be Some");
                let sxx = d.raw_out.clone().slice([0..n, 2..3]).reshape([n]);
                let syy = d.raw_out.clone().slice([0..n, 3..4]).reshape([n]);
                let sxy = d.raw_out.clone().slice([0..n, 4..5]).reshape([n]);
                hole_traction_loss_direct(sxx, syy, sxy, nx, ny).mul_scalar(1.0 / self.ref_stress2 as f64)
            }
            HoleBc::Fixed => {
                let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
                let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
                (u.clone() * u + v.clone() * v).mean()
            }
        }
    }
}

pub struct UserDefinedProblem {
    spec: ProblemSpec,
    domains: Vec<DomainSpec>,
    sampling: UserSamplingStrategy,
    ansatz: IdentityAnsatz,
    /// Same leaking convention as `UserSamplingStrategy::hole_names` — content-equal
    /// `&'static str`s independently leaked here are fine (`HashMap<&'static str, _>`
    /// lookups compare by string content, not pointer identity).
    hole_names: Vec<&'static str>,
}

impl UserDefinedProblem {
    pub fn new(spec: ProblemSpec) -> Self {
        let domain_geometry = spec.geometry.to_placeholder();
        let domains = vec![DomainSpec {
            id: USER_DOMAIN,
            geometry: domain_geometry,
            material: spec.material.clone(),
            output_dim: 5, // mDEM: u, v, sigma_xx, sigma_yy, sigma_xy
        }];
        let hole_names = (0..spec.geometry.holes.len())
            .map(|i| -> &'static str { Box::leak(format!("hole_{i}").into_boxed_str()) })
            .collect();
        let sampling = UserSamplingStrategy::new(spec.geometry.clone());
        UserDefinedProblem { spec, domains, sampling, ansatz: IdentityAnsatz, hole_names }
    }

    pub fn spec(&self) -> &ProblemSpec { &self.spec }
}

impl BoundaryValueProblem for UserDefinedProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }

    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        assert_eq!(domain_idx, 0, "UserDefinedProblem has exactly one domain");
        &self.sampling
    }

    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        assert_eq!(domain_idx, 0, "UserDefinedProblem has exactly one domain");
        &self.ansatz
    }

    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let stress_ref = self.spec.load.px.abs().max(self.spec.load.py.abs()).max(1.0);
        let ref_energy = (0.5 * stress_ref * stress_ref / self.spec.material.e).max(1.0) as f32;
        let ref_stress2 = (stress_ref * stress_ref).max(1.0) as f32;

        let mut terms: Vec<Box<dyn LossTerm>> = vec![
            Box::new(InteriorEnergyTerm { material: self.spec.material.clone(), ref_energy }),
            Box::new(OuterTractionTerm {
                material: self.spec.material.clone(),
                ref_stress2,
                px: self.spec.load.px,
                py: self.spec.load.py,
            }),
        ];
        for (hole, &name) in self.spec.geometry.holes.iter().zip(self.hole_names.iter()) {
            terms.push(Box::new(HoleBcTerm { point_set: name, bc: hole.bc, ref_stress2 }));
        }
        terms
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "interior_energy" => LAM_INTERIOR_ENERGY,
            "outer_traction" => LAM_OUTER_TRACTION,
            "hole_free" => LAM_HOLE_FREE,
            "hole_fixed" => LAM_HOLE_FIXED,
            other => panic!("UserDefinedProblem::base_weight: unknown loss term '{other}'"),
        }
    }

    fn phase1_steps(&self) -> usize { 0 }

    /// No closed-form convergence metric exists for an arbitrary user-defined geometry —
    /// unlike Kirsch's K_t, there's no analytic target to probe against.
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }

    fn convergence_target(&self) -> f64 { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::user_geometry::HoleSpec;

    fn two_hole_geometry() -> UserGeometry {
        UserGeometry {
            half_w: 0.1,
            half_h: 0.05,
            thickness: 0.005,
            holes: vec![
                HoleSpec { center: [-0.03, 0.0], radius: 0.01, bc: HoleBc::Free },
                HoleSpec { center: [0.03, 0.0], radius: 0.008, bc: HoleBc::Fixed },
            ],
        }
    }

    #[test]
    fn sample_interior_points_never_fall_inside_any_hole_or_outside_the_plate() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone());
        let placeholder = GeometryConfig::kirsch_plate_inches(); // ignored by this strategy
        let pts = strategy.sample_interior(&placeholder, 500);
        assert_eq!(pts.len(), 500, "rejection sampling must reach the requested count");
        for [x, y] in pts {
            assert!(geom.contains(x, y), "point ({x},{y}) violates plate/hole containment");
        }
    }

    #[test]
    fn sample_boundary_produces_points_on_all_four_outer_edges() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone());
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let pts = strategy.sample_boundary(&placeholder, &LoadConfig::uniaxial_x(1.0), 40);
        assert!(!pts.is_empty());
        for p in &pts {
            let on_x_edge = (p.x.abs() - geom.half_w).abs() < 1e-9;
            let on_y_edge = (p.y.abs() - geom.half_h).abs() < 1e-9;
            assert!(on_x_edge || on_y_edge, "point ({}, {}) is not on an outer edge", p.x, p.y);
        }
    }

    #[test]
    fn named_point_sets_returns_one_ring_per_hole_with_expected_point_count() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone());
        let sets = strategy.named_point_sets(&[]);
        assert_eq!(sets.len(), 2);
        let names: Vec<&str> = sets.iter().map(|s| s.name).collect();
        assert!(names.contains(&"hole_0"));
        assert!(names.contains(&"hole_1"));
        for set in &sets {
            assert_eq!(set.points.len(), HOLE_RING_POINTS);
        }
    }

    #[test]
    fn hole_ring_points_lie_on_their_hole_circle_at_the_correct_radius() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone());
        let sets = strategy.named_point_sets(&[]);
        let hole0 = geom.holes[0];
        let ring0 = &sets.iter().find(|s| s.name == "hole_0").unwrap().points;
        for p in ring0 {
            let dx = p.x - hole0.center[0];
            let dy = p.y - hole0.center[1];
            let r = (dx * dx + dy * dy).sqrt();
            assert!((r - hole0.radius).abs() < 1e-9, "ring point not on hole_0's circle: r={r}");
        }
    }

    #[test]
    fn user_defined_problem_has_one_term_per_hole_plus_interior_and_outer_traction() {
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
        };
        let problem = UserDefinedProblem::new(spec);
        let terms = problem.loss_terms();
        assert_eq!(terms.len(), 4); // interior_energy + outer_traction + 2 holes
        let names: Vec<&str> = terms.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"interior_energy"));
        assert!(names.contains(&"outer_traction"));
        assert_eq!(names.iter().filter(|&&n| n == "hole_free").count(), 1);
        assert_eq!(names.iter().filter(|&&n| n == "hole_fixed").count(), 1);
        crate::problem::validate_loss_terms(&problem);
    }
}
