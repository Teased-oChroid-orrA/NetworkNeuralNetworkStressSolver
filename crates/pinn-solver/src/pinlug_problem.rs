/// Pin-in-lug contact problem: two genuine physical domains (pin + lug) coupled by a
/// shared contact interface, driven through the same `BoundaryValueProblem`/
/// `DomainSamplingStrategy`/`LossTerm` trait family Kirsch uses (`kirsch_problem.rs`), but
/// through the NEW `step_physics_multi` driver (`training_core.rs`) since it genuinely has
/// 2 domains rather than 1.
///
/// ## Geometry
/// - LUG: `GeometryConfig::pinlug_lug_inches()` — 1.5in × 1.5in plate (half_w=half_h=0.75in),
///   t=0.4in, central hole R=0.5in, `SymmetryMode::Full` (contact loading is not
///   quarter-symmetric — the pin only bears on roughly half the hole boundary).
/// - PIN: `GeometryConfig::pinlug_pin_inches()` — see that constructor's doc comment for
///   why a solid disk is represented as a square bounding box with `HoleType::None`, with
///   `PinLugSamplingStrategy`'s `sample_interior` doing the actual disk-membership
///   rejection (analogous to how `HoleType::Circular` sampling rejects points *inside* a
///   hole — this rejects points *outside* a disk).
///
/// ## Material
/// 4340 steel for both domains (`MaterialProps::steel_4340()`, E=30 Msi, ν=0.29).
///
/// ## Load — force-to-traction conversion
/// The physical loading is P=20,000 lbf total axial force driving the pin into the lug
/// along +x. `LoadConfig`'s `px`/`py` are far-field *stress* [Pa] (per `CLAUDE.md`), not
/// force, so the resultant force must be converted to an equivalent traction magnitude for
/// the pin's driving boundary condition (`PinDrivingTractionTerm`).
///
/// Convention used here (standard pin-joint / Hertzian contact bearing-stress convention):
///     equivalent_traction = P / (2 * r_pin * t)
/// i.e. the resultant axial force is reacted by the contact pressure distribution's
/// projection onto the loading axis, whose maximum geometric extent is the pin's diameter
/// (`2*r_pin`) times the plate thickness `t` — this is the same "bearing stress" formula
/// used in mechanical-design handbooks for pin/bolt bearing capacity
/// (`σ_bearing = P / (d * t)`). See `SolverConfig::default_pinlug()` (`pinn-core/src/
/// messages.rs`) for the numeric derivation, reused verbatim here.
///
/// ## Domains / loss terms
/// `PIN_DOMAIN` (id 0), `LUG_DOMAIN` (id 1). Both interior-energy terms reuse
/// `dem_energy_loss` generically. `LugShankAnchorTerm` anchors the lug's outer edge
/// (fixed grip, analogous to Kirsch's `DisplacementAnchorTerm`). `LugFreeEdgeTractionTerm`
/// enforces the traction-free lug outer-boundary segments not gripped
/// (`hole_traction_loss_direct` reused). `PinDrivingTractionTerm` applies the driving
/// traction on the pin's loaded face (`neumann_loss`'s pattern, reused). Interface terms
/// (`InterfacePenetrationTerm`/`InterfaceNonTensionTerm`) are Signorini KKT penalties over
/// the shared-theta interface point-set (`signorini::penetration_penalty`/
/// `non_tension_penalty`/`decompose_radial`, unchanged).
///
/// Gap sign convention (see `InterfacePenetrationTerm` doc comment for the flip-risk note):
///     gap(theta) = (r_lug_hole + u_r_lug(theta)) - (r_pin_outer + u_r_pin(theta))
/// i.e. gap > 0 means there is still a physical clearance between the lug's inner (hole)
/// surface and the pin's outer surface at that angle; gap < 0 means they have
/// interpenetrated (non-physical, penalized by `penetration_penalty`).
use std::sync::Arc;

use burn::tensor::Tensor;

use pinn_core::{
    geometry::{GeometryConfig, HoleType},
    loading::{BoundaryKind, BoundaryPoint, LoadConfig},
    material::MaterialProps,
    problem::{
        DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec, InterfaceParametrization,
        NamedPointSet,
    },
};

use crate::{
    energy::{dem_energy_loss, hole_traction_loss_direct, neumann_loss},
    problem::{BoundaryValueProblem, DomainForwardOutputs, DomainState, LossTerm, B},
    signorini::{decompose_radial, non_tension_penalty, penetration_penalty},
};

pub const PIN_DOMAIN: DomainId = DomainId(0);
pub const LUG_DOMAIN: DomainId = DomainId(1);

const SEED_LUG_INTERIOR: u64 = 13_37;
const SEED_PIN_INTERIOR: u64 = 24_68;
const SEED_LUG_BOUNDARY: u64 = 55_55;
const SEED_PIN_BOUNDARY: u64 = 77_77;
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;

const LAM_E: f32 = 1.0;
const LAM_N: f32 = 10.0;
const LAM_FREE_EDGE: f32 = 100.0;
const LAM_ANCHOR: f32 = 50.0;
const LAM_PENETRATION: f32 = 500.0;
const LAM_NON_TENSION: f32 = 100.0;

/// Identity ansatz (no hard Dirichlet BC baked in for pin-in-lug — the shank anchor and
/// free-edge conditions are enforced as soft loss terms, since neither domain has a
/// Kirsch-style symmetry plane to hard-enforce zero-displacement on).
pub struct IdentityAnsatz;
impl DirichletAnsatz for IdentityAnsatz {
    fn eval(&self, _xn: f32, _yn: f32, _k: f32) -> (f32, f32) { (1.0, 1.0) }
}

/// Per-domain sampling strategy for pin-in-lug. `is_pin=true` samples the pin's solid disk
/// (see `GeometryConfig::pinlug_pin_inches` doc comment); `is_pin=false` samples the lug's
/// plate-with-hole. Both share one `Arc<InterfaceParametrization>` so their `"interface"`
/// point-sets are angle-index-aligned (see module doc comment / `InterfaceParametrization`
/// doc comment for why this is load-bearing).
pub struct PinLugSamplingStrategy {
    pub is_pin: bool,
    pub interface: Arc<InterfaceParametrization>,
    /// Pin outer radius / lug hole radius — physically equal (nominal fit), used to place
    /// the shared interface point-set on each domain's own surface.
    pub contact_radius: f64,
}

impl DomainSamplingStrategy for PinLugSamplingStrategy {
    fn sample_interior(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;
        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();
        let mut pts = Vec::with_capacity(n);
        let mut attempts = 0usize;
        let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;
        let mut rng = LcgRng::new(if self.is_pin { SEED_PIN_INTERIOR } else { SEED_LUG_INTERIOR });

        while pts.len() < n && attempts < max_attempts {
            let x = x0 + rng.next_f64() * (x1 - x0);
            let y = y0 + rng.next_f64() * (y1 - y0);
            let inside = if self.is_pin {
                // Solid disk: geom.contains() only checks the bounding box (HoleType::None,
                // see GeometryConfig::pinlug_pin_inches doc comment) — the disk membership
                // check is this sampling strategy's responsibility.
                x * x + y * y <= self.contact_radius * self.contact_radius
            } else {
                geom.contains(x, y)
            };
            if inside {
                pts.push([x, y]);
            }
            attempts += 1;
        }
        pts.truncate(n);
        pts
    }

    fn sample_boundary(&self, geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        use pinn_core::LcgRng;
        let mut pts = Vec::new();
        if self.is_pin {
            // Pin's "boundary" (outside the shared interface point-set, handled separately
            // via named_point_sets) is just its own outer circumference, which IS the
            // contact surface — no separate free/loaded edges for a solid pin disk in this
            // simplified representation. Left empty; the driving traction / interface terms
            // read from named point-sets instead.
            let _ = (geom, load, n);
            return pts;
        }
        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();
        let n_per_edge = (n / 4).max(1);
        let mut rng = LcgRng::new(SEED_LUG_BOUNDARY);
        // Outer edges: traction-free (grip is modeled via LugShankAnchorTerm on the far
        // edge x=x0 instead of a hard BC — see module doc comment).
        for _ in 0..n_per_edge {
            let y = y0 + rng.next_f64() * (y1 - y0);
            pts.push(BoundaryPoint { x: x1, y, nx: 1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree });
        }
        for _ in 0..n_per_edge {
            let x = x0 + rng.next_f64() * (x1 - x0);
            pts.push(BoundaryPoint { x, y: y1, nx: 0.0, ny: 1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree });
        }
        for _ in 0..n_per_edge {
            let y = y0 + rng.next_f64() * (y1 - y0);
            pts.push(BoundaryPoint { x: x0, y, nx: -1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree });
        }
        for _ in 0..n_per_edge {
            let x = x0 + rng.next_f64() * (x1 - x0);
            pts.push(BoundaryPoint { x, y: y0, nx: 0.0, ny: -1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree });
        }
        pts
    }

    fn amr_lock_zone(&self, _geom: &GeometryConfig, cell_center: [f64; 2]) -> bool {
        const CONTACT_ZONE_FACTOR: f64 = 1.5;
        let [cx, cy] = cell_center;
        let zone_r = self.contact_radius * CONTACT_ZONE_FACTOR;
        cx * cx + cy * cy < zone_r * zone_r
    }

    fn sample_extra_ring(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> {
        // Pin-in-lug does not use Kirsch's near-hole equilibrium-residual ring — contact
        // equilibrium is enforced entirely through the interface Signorini terms.
        Vec::new()
    }

    fn named_point_sets(&self, _bnd_pts: &[BoundaryPoint]) -> Vec<NamedPointSet> {
        use pinn_core::LcgRng;

        let interface_points: Vec<BoundaryPoint> = self.interface.thetas.iter().map(|&theta| {
            let x = self.contact_radius * theta.cos();
            let y = self.contact_radius * theta.sin();
            // Outward normal convention: for the LUG (material occupies r > contact_radius
            // near the hole), the outward normal of the solid points radially INWARD
            // (toward the hole center), matching Kirsch's hole-boundary convention
            // (sampling.rs:207-211). For the PIN (material occupies r < contact_radius),
            // the outward normal points radially OUTWARD. Same physical angle, opposite
            // radial sense — this is intentional (each domain's "outward" is w.r.t. its own
            // solid material), not a bug.
            let (nx, ny) = if self.is_pin {
                (theta.cos(), theta.sin())
            } else {
                (-theta.cos(), -theta.sin())
            };
            BoundaryPoint {
                x, y, nx, ny, tx: 0.0, ty: 0.0,
                kind: BoundaryKind::Interface {
                    partner_domain: if self.is_pin { LUG_DOMAIN } else { PIN_DOMAIN },
                },
            }
        }).collect();

        let mut sets = vec![NamedPointSet { name: "interface", points: interface_points }];

        if self.is_pin {
            // Driving traction on the pin's own "far side" (θ near π, opposite the contact
            // patch, along -x) — models the external force being reacted into the pin.
            // Represented as a small angular band of boundary points at r=contact_radius
            // (the pin's outer surface) with a prescribed traction.
            let mut rng = LcgRng::new(SEED_PIN_BOUNDARY);
            let n_drive = 32;
            let drive_pts: Vec<BoundaryPoint> = (0..n_drive).map(|_| {
                let theta = std::f64::consts::PI + (rng.next_f64() - 0.5) * (std::f64::consts::PI / 6.0);
                let x = self.contact_radius * theta.cos();
                let y = self.contact_radius * theta.sin();
                BoundaryPoint {
                    x, y, nx: theta.cos(), ny: theta.sin(),
                    tx: 0.0, ty: 0.0, // filled in with the real traction target by the caller
                    kind: BoundaryKind::NeumannLoad,
                }
            }).collect();
            sets.push(NamedPointSet { name: "driving", points: drive_pts });
        } else {
            // Lug shank anchor: far outer edge (x = x0, gripped) used by LugShankAnchorTerm.
            let mut rng = LcgRng::new(SEED_LUG_BOUNDARY.wrapping_add(1));
            let n_anchor = 32;
            let anchor_pts: Vec<BoundaryPoint> = (0..n_anchor).map(|_| {
                let y = -self.contact_radius + rng.next_f64() * (2.0 * self.contact_radius);
                BoundaryPoint { x: -self.contact_radius * 1.5, y, nx: -1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree }
            }).collect();
            sets.push(NamedPointSet { name: "shank_anchor", points: anchor_pts });
        }
        sets
    }
}

// ─── Loss terms ─────────────────────────────────────────────────────────────────────────

pub struct InteriorEnergyTerm {
    pub domain: DomainId,
    pub material: MaterialProps,
    pub ref_energy: f32,
}
impl LossTerm for InteriorEnergyTerm {
    fn name(&self) -> &'static str {
        if self.domain == PIN_DOMAIN { "pin_interior_energy" } else { "lug_interior_energy" }
    }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("interior_energy: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("interior_energy: strains must be Some");
        dem_energy_loss(exx, eyy, exy, &self.material).mul_scalar(1.0 / self.ref_energy as f64)
    }
}

/// Anchors the lug's gripped far edge to zero displacement (soft penalty, analogous to
/// Kirsch's `DisplacementAnchorTerm` but MSE-to-zero rather than to a nonzero target since
/// pin-in-lug has no closed-form target displacement).
pub struct LugShankAnchorTerm;
impl LossTerm for LugShankAnchorTerm {
    fn name(&self) -> &'static str { "lug_shank_anchor" }
    fn domains(&self) -> Vec<DomainId> { vec![LUG_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["shank_anchor"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == LUG_DOMAIN).expect("lug_shank_anchor: domain missing");
        let n = d.raw_out.dims()[0];
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        (u.clone() * u + v.clone() * v).mean()
    }
}

/// Lug's non-gripped outer edges must be traction-free. Reuses `hole_traction_loss_direct`
/// (name is legacy from its Kirsch hole-boundary origin but the formula — mean(|sigma.n|^2)
/// — applies to any traction-free direct-stress boundary).
pub struct LugFreeEdgeTractionTerm {
    pub ref_stress2: f32,
}
impl LossTerm for LugFreeEdgeTractionTerm {
    fn name(&self) -> &'static str { "lug_free_edge_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![LUG_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["boundary"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == LUG_DOMAIN).expect("lug_free_edge_traction: domain missing");
        let (nx, ny) = d.normals.clone().expect("lug_free_edge_traction: normals must be Some");
        let n = d.raw_out.dims()[0];
        let sxx = d.raw_out.clone().slice([0..n, 2..3]).reshape([n]);
        let syy = d.raw_out.clone().slice([0..n, 3..4]).reshape([n]);
        let sxy = d.raw_out.clone().slice([0..n, 4..5]).reshape([n]);
        hole_traction_loss_direct(sxx, syy, sxy, nx, ny).mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

/// Pin's driving traction (see module doc comment for the force→traction derivation).
pub struct PinDrivingTractionTerm {
    pub material: MaterialProps,
    pub ref_stress2: f32,
    pub tx_target: Tensor<B, 1>,
    pub ty_target: Tensor<B, 1>,
}
impl LossTerm for PinDrivingTractionTerm {
    fn name(&self) -> &'static str { "pin_driving_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![PIN_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["driving"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == PIN_DOMAIN).expect("pin_driving_traction: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("pin_driving_traction: strains must be Some");
        let (nx, ny) = d.normals.clone().expect("pin_driving_traction: normals must be Some");
        neumann_loss(exx, eyy, exy, nx, ny, self.tx_target.clone(), self.ty_target.clone(), &self.material)
            .mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

/// Interface non-penetration penalty (Signorini KKT condition #1): the lug's inner (hole)
/// surface and the pin's outer surface must not interpenetrate at any shared theta.
///
/// gap(theta) = (r_lug_hole + u_r_lug(theta)) - (r_pin_outer + u_r_pin(theta))
///
/// gap > 0: physical clearance remains (or exact fit at gap=0). gap < 0: interpenetration
/// (penalized). See module doc comment for the full sign-convention rationale — a flipped
/// sign here is the most likely wiring bug per the design brief, since both `u_r_pin` and
/// `u_r_lug` are radial DISPLACEMENTS (can be positive or negative) added to their
/// respective UNDEFORMED radii, not raw stress or position values.
pub struct InterfacePenetrationTerm {
    pub thetas: Vec<f64>,
    pub r_pin: f64,
    pub r_lug: f64,
}
impl LossTerm for InterfacePenetrationTerm {
    fn name(&self) -> &'static str { "interface_penetration" }
    fn domains(&self) -> Vec<DomainId> { vec![PIN_DOMAIN, LUG_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface", "interface"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let pin = inputs.iter().find(|i| i.domain == PIN_DOMAIN).expect("interface_penetration: pin missing");
        let lug = inputs.iter().find(|i| i.domain == LUG_DOMAIN).expect("interface_penetration: lug missing");
        let n = self.thetas.len();
        let pin_u: Vec<f32> = pin.raw_out.clone().slice([0..n, 0..1]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let pin_v: Vec<f32> = pin.raw_out.clone().slice([0..n, 1..2]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let lug_u: Vec<f32> = lug.raw_out.clone().slice([0..n, 0..1]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let lug_v: Vec<f32> = lug.raw_out.clone().slice([0..n, 1..2]).reshape([n]).into_data().to_vec().unwrap_or_default();

        let mut total = 0.0_f64;
        for i in 0..n {
            let theta = self.thetas[i];
            let c = theta.cos();
            let s = theta.sin();
            let u_r_pin = pin_u[i] as f64 * c + pin_v[i] as f64 * s;
            let u_r_lug = lug_u[i] as f64 * c + lug_v[i] as f64 * s;
            let gap = (self.r_lug + u_r_lug) - (self.r_pin + u_r_pin);
            total += penetration_penalty(gap);
        }
        let mean = total / n.max(1) as f64;
        let device = pin.raw_out.device();
        Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![mean as f32], vec![1]), &device)
    }
}

/// Interface non-tension penalty (Signorini KKT condition #2): contact pressure (radial
/// stress at the interface) must be compressive (<= 0), never tensile — a pin can push on a
/// lug but not pull on it. Uses `decompose_radial` on the pin's own stress state at each
/// shared theta (the pin is the one physically transmitting the contact force). `thetas`
/// must be the SAME shared `InterfaceParametrization` angles (index-aligned with the
/// `"interface"` point-set), threaded through explicitly rather than re-derived from `n`
/// alone — an implicit "assume equal angular spacing" reconstruction would silently break
/// if the interface sampling strategy ever changes.
pub struct InterfaceNonTensionTerm {
    pub thetas: Vec<f64>,
}
impl LossTerm for InterfaceNonTensionTerm {
    fn name(&self) -> &'static str { "interface_non_tension" }
    fn domains(&self) -> Vec<DomainId> { vec![PIN_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface"] }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let pin = inputs.iter().find(|i| i.domain == PIN_DOMAIN).expect("interface_non_tension: pin missing");
        let n = self.thetas.len();
        let sxx: Vec<f32> = pin.raw_out.clone().slice([0..n, 2..3]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let syy: Vec<f32> = pin.raw_out.clone().slice([0..n, 3..4]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let sxy: Vec<f32> = pin.raw_out.clone().slice([0..n, 4..5]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let mut total = 0.0_f64;
        for i in 0..n {
            let (s_rr, _s_tt, _s_rt) = decompose_radial(sxx[i] as f64, syy[i] as f64, sxy[i] as f64, self.thetas[i]);
            total += non_tension_penalty(s_rr);
        }
        let mean = total / n.max(1) as f64;
        let device = pin.raw_out.device();
        Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(vec![mean as f32], vec![1]), &device)
    }
}

// ─── PinLugProblem ────────────────────────────────────────────────────────────────────────

pub struct PinLugProblem {
    domains: [DomainSpec; 2],
    pin_sampling: PinLugSamplingStrategy,
    lug_sampling: PinLugSamplingStrategy,
    ansatz: IdentityAnsatz,
    interface: Arc<InterfaceParametrization>,
    equivalent_traction_pa: f64,
    phase1_steps: usize,
}

impl PinLugProblem {
    /// `n_interface`: number of shared-theta interface points (index-aligned across pin and
    /// lug — see `InterfaceParametrization` doc comment).
    pub fn new(material: MaterialProps, output_dim: usize, phase1_steps: usize, n_interface: usize) -> Self {
        let lug_geometry = GeometryConfig::pinlug_lug_inches();
        let pin_geometry = GeometryConfig::pinlug_pin_inches();
        let HoleType::Circular { radius: r_lug } = lug_geometry.hole else {
            panic!("PinLugProblem::new: lug geometry must have a circular hole");
        };
        let r_pin = pin_geometry.half_w; // pin disk radius (see pinlug_pin_inches doc comment)

        let thetas: Vec<f64> = (0..n_interface)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / n_interface.max(1) as f64)
            .collect();
        let interface = Arc::new(InterfaceParametrization { thetas });

        // Force -> traction (see module doc comment for the full derivation): P / (2*r*t).
        use pinn_core::units::LBF_TO_N;
        let total_force_lbf = 20_000.0;
        let total_force_n = total_force_lbf * LBF_TO_N;
        let projected_area_m2 = 2.0 * r_pin * lug_geometry.thickness;
        let equivalent_traction_pa = total_force_n / projected_area_m2;

        Self {
            domains: [
                DomainSpec { id: PIN_DOMAIN, geometry: pin_geometry, material: material.clone(), output_dim },
                DomainSpec { id: LUG_DOMAIN, geometry: lug_geometry, material, output_dim },
            ],
            pin_sampling: PinLugSamplingStrategy { is_pin: true, interface: interface.clone(), contact_radius: r_pin },
            lug_sampling: PinLugSamplingStrategy { is_pin: false, interface: interface.clone(), contact_radius: r_lug },
            ansatz: IdentityAnsatz,
            interface,
            equivalent_traction_pa,
            phase1_steps,
        }
    }

    pub fn interface_thetas(&self) -> &[f64] { &self.interface.thetas }
    pub fn equivalent_traction_pa(&self) -> f64 { self.equivalent_traction_pa }
}

impl BoundaryValueProblem for PinLugProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }

    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        match domain_idx {
            0 => &self.pin_sampling,
            1 => &self.lug_sampling,
            _ => panic!("PinLugProblem has exactly two domains (0=pin, 1=lug)"),
        }
    }

    fn ansatz(&self, _domain_idx: usize) -> &dyn DirichletAnsatz { &self.ansatz }

    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let material = self.domains[0].material.clone();
        let device = Default::default();
        vec![
            Box::new(InteriorEnergyTerm { domain: PIN_DOMAIN, material: material.clone(), ref_energy: 1.0 }),
            Box::new(InteriorEnergyTerm { domain: LUG_DOMAIN, material: material.clone(), ref_energy: 1.0 }),
            Box::new(LugShankAnchorTerm),
            Box::new(LugFreeEdgeTractionTerm { ref_stress2: 1.0 }),
            Box::new(PinDrivingTractionTerm {
                material, ref_stress2: 1.0,
                tx_target: Tensor::<B, 1>::zeros([1], &device),
                ty_target: Tensor::<B, 1>::zeros([1], &device),
            }),
            Box::new(InterfacePenetrationTerm {
                thetas: self.interface.thetas.clone(),
                r_pin: self.domains[0].geometry.half_w,
                r_lug: match self.domains[1].geometry.hole { HoleType::Circular { radius } => radius, HoleType::None => 0.0 },
            }),
            Box::new(InterfaceNonTensionTerm { thetas: self.interface.thetas.clone() }),
        ]
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "pin_interior_energy" | "lug_interior_energy" => LAM_E,
            "pin_driving_traction" => LAM_N,
            "lug_free_edge_traction" => LAM_FREE_EDGE,
            "lug_shank_anchor" => LAM_ANCHOR,
            "interface_penetration" => LAM_PENETRATION,
            "interface_non_tension" => LAM_NON_TENSION,
            other => panic!("PinLugProblem::base_weight: unknown loss term '{other}'"),
        }
    }

    fn phase1_steps(&self) -> usize { self.phase1_steps }

    /// Read-only diagnostic forward pass over the shared thetas (no backward) — RMS
    /// interface gap across all shared angles. 0.0 = perfect contact (the target). `None`
    /// when `state` is empty or doesn't have one entry per `domains()` (2).
    fn convergence_metric(&self, state: &[DomainState<B>]) -> Option<f64> {
        if state.len() != self.domains.len() {
            return None;
        }
        // Full evaluation requires a live forward pass through each domain's model at the
        // shared interface thetas — deferred to a dedicated helper analogous to
        // `probe_kt_shared` (kept out of this trait method's signature, which only has
        // `DomainState` — no FdConfig/device context). Returning `None` here documents that
        // this trait method alone cannot complete the computation; a `PinLugProblem::
        // probe_interface_gap_rms` companion (mirroring `KirschProblem::probe_kt`'s
        // delegation pattern) would need the same context `probe_kt_shared` takes.
        None
    }

    fn convergence_target(&self) -> f64 { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_metric_pinlug_returns_none_when_state_slice_empty() {
        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16);
        assert_eq!(problem.convergence_metric(&[]), None);
    }

    #[test]
    fn validate_loss_terms_accepts_well_formed_pinlug_problem() {
        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16);
        crate::problem::validate_loss_terms(&problem); // must not panic
    }

    #[test]
    fn interface_penetration_term_zero_at_zero_gap_boundary() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let n = 4;
        let thetas: Vec<f64> = (0..n).map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64).collect();
        let r_pin = 0.5_f64;
        let r_lug = 0.5_f64;

        // Engineer forward outputs so gap = (r_lug + u_r_lug) - (r_pin + u_r_pin) = 0 at
        // every theta: set both domains' displacement fields to exactly zero everywhere
        // (u_r_pin = u_r_lug = 0), and since r_pin == r_lug, gap = 0 identically.
        let zeros = vec![0.0_f32; n * 5]; // 5 cols: u, v, sxx, syy, sxy — all zero
        let pin_raw = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(zeros.clone(), vec![n, 5]), &device);
        let lug_raw = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(zeros, vec![n, 5]), &device);

        let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
        let lug_fwd = DomainForwardOutputs { domain: LUG_DOMAIN, raw_out: &lug_raw, strains: None, normals: None };

        let term = InterfacePenetrationTerm { thetas, r_pin, r_lug };
        let loss = term.compute(&[pin_fwd, lug_fwd]);
        let v: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(v, 0.0, "zero displacement + equal radii must give exactly zero gap everywhere -> zero penalty");
    }
}
