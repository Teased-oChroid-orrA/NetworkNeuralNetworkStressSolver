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
};

pub const PIN_DOMAIN: DomainId = DomainId(0);
pub const LUG_DOMAIN: DomainId = DomainId(1);

/// Which stress magnitude `PinLugProblem`'s internal `ref_energy`/`ref_stress2`/`ref_gap2`
/// normalize against. Mirrors `SolverConfig::use_ultimate_strength_scaling` for the
/// two-domain pin-lug path, which has no `SolverConfig` of its own to read the flag from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinLugScalingMode {
    /// Normalize by the driving equivalent bearing traction (existing, default behavior).
    AppliedLoad,
    /// Normalize by `material.ultimate_strength_pa` (opt-in).
    UltimateStrength,
}

const SEED_LUG_INTERIOR: u64 = 13_37;
const SEED_PIN_INTERIOR: u64 = 24_68;
const SEED_LUG_BOUNDARY: u64 = 55_55;
const SEED_PIN_BOUNDARY: u64 = 77_77;
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
/// Point count for the pin's "driving" named point-set (see `named_point_sets`) — shared
/// with `loss_terms()`'s `PinDrivingTractionTerm` construction so the target-tensor length
/// always matches the actual sampled point count.
const N_DRIVE_POINTS: usize = 32;

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
        // Only 3 of the 4 outer edges are traction-free here: the shank edge (x=x0) is
        // EXCLUDED because it is rigidly clamped (fixed grip), enforced via the dedicated
        // "shank_anchor" named point set / LugShankAnchorTerm instead — sampling it here too
        // as NeumannFree would impose a contradictory traction-free condition on the same
        // physical points LugShankAnchorTerm pins to u=v=0 (see module doc comment / Issue 1
        // in the review that prompted this fix). n_per_edge is therefore based on 3 edges,
        // not 4, to keep the total boundary point count close to the caller's requested `n`.
        let n_per_edge = (n / 3).max(1);
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
            let n_drive = N_DRIVE_POINTS;
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
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Physics }
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
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
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
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
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
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
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
    /// Normalizes the raw gap² [m²] penalty to O(1), mirroring how every other term here
    /// divides by a reference physical scale (`ref_energy`/`ref_stress2`) before SAW-BRDR
    /// weighting — without this, `gap²` at meter scale is negligible but at the sub-mm
    /// scale actually expected here would still be many orders of magnitude off from the
    /// O(1)-normalized energy/stress terms it's summed against.
    pub ref_gap2: f32,
}
/// Uploads `thetas`' cos/sin as fixed (non-learned) host constants, shared by both Signorini
/// terms below. Never carries gradient itself — only the sliced `raw_out` operands multiplied
/// against these need to stay connected to the autodiff graph.
fn theta_trig_tensors<B: burn::tensor::backend::Backend>(thetas: &[f64], device: &B::Device) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let n = thetas.len();
    let cos_v: Vec<f32> = thetas.iter().map(|t| t.cos() as f32).collect();
    let sin_v: Vec<f32> = thetas.iter().map(|t| t.sin() as f32).collect();
    let cos_theta = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(cos_v, vec![n]), device);
    let sin_theta = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(sin_v, vec![n]), device);
    (cos_theta, sin_theta)
}

impl LossTerm for InterfacePenetrationTerm {
    fn name(&self) -> &'static str { "interface_penetration" }
    fn domains(&self) -> Vec<DomainId> { vec![PIN_DOMAIN, LUG_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface", "interface"] }
    // Signorini KKT boundary/interface condition, not an interior PDE residual.
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let pin = inputs.iter().find(|i| i.domain == PIN_DOMAIN).expect("interface_penetration: pin missing");
        let lug = inputs.iter().find(|i| i.domain == LUG_DOMAIN).expect("interface_penetration: lug missing");
        let n = self.thetas.len();
        let device = pin.raw_out.device();

        let pin_u = pin.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let pin_v = pin.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let lug_u = lug.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let lug_v = lug.raw_out.clone().slice([0..n, 1..2]).reshape([n]);

        let (cos_theta, sin_theta) = theta_trig_tensors::<B>(&self.thetas, &device);

        let u_r_pin = pin_u * cos_theta.clone() + pin_v * sin_theta.clone();
        let u_r_lug = lug_u * cos_theta + lug_v * sin_theta;

        // gap = (r_lug + u_r_lug) - (r_pin + u_r_pin) == (u_r_lug - u_r_pin) + (r_lug - r_pin)
        let gap = (u_r_lug - u_r_pin).add_scalar(self.r_lug - self.r_pin);

        let penalty = gap.neg().clamp_min(0.0_f64).powf_scalar(2.0_f64);
        penalty.mean().mul_scalar(1.0 / self.ref_gap2 as f64)
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
    /// Normalizes the raw `s_rr²` [Pa²] penalty to O(1), same convention as `ref_stress2`
    /// on every other stress-based term (`LugFreeEdgeTractionTerm`, `PinDrivingTractionTerm`)
    /// — without this, raw steel-stress-scale (`s_rr` ~1e8 Pa) squared dwarfs every
    /// O(1)-normalized term it's summed against in the SAW-BRDR total.
    pub ref_stress2: f32,
}
impl LossTerm for InterfaceNonTensionTerm {
    fn name(&self) -> &'static str { "interface_non_tension" }
    fn domains(&self) -> Vec<DomainId> { vec![PIN_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface"] }
    // Same reasoning as InterfacePenetrationTerm: Signorini KKT boundary condition.
    fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let pin = inputs.iter().find(|i| i.domain == PIN_DOMAIN).expect("interface_non_tension: pin missing");
        let n = self.thetas.len();
        let device = pin.raw_out.device();

        let sxx = pin.raw_out.clone().slice([0..n, 2..3]).reshape([n]);
        let syy = pin.raw_out.clone().slice([0..n, 3..4]).reshape([n]);
        let sxy = pin.raw_out.clone().slice([0..n, 4..5]).reshape([n]);

        let (cos_theta, sin_theta) = theta_trig_tensors::<B>(&self.thetas, &device);

        let cos2 = cos_theta.clone() * cos_theta.clone();
        let sin2 = sin_theta.clone() * sin_theta.clone();
        let sin_cos_2 = (sin_theta * cos_theta).mul_scalar(2.0);

        let s_rr = sxx * cos2 + syy * sin2 + sxy * sin_cos_2;

        let penalty = s_rr.clamp_min(0.0_f64).powf_scalar(2.0_f64);
        penalty.mean().mul_scalar(1.0 / self.ref_stress2 as f64)
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
    /// O(1)-normalization reference scales, computed the same way
    /// `training_core::compute_reference_scales` does for Kirsch (`ref_energy = 0.5*P²/E`,
    /// `ref_stress2 = P²`), using `equivalent_traction_pa` as the load magnitude `P`. Both
    /// domains share one material (steel 4340) here, so one shared pair suffices — a future
    /// problem with per-domain materials would need per-domain values instead. Without this,
    /// raw-Pa-squared loss terms sit at ~1e15-1e18 (steel stresses are ~1e8-1e9 Pa), which
    /// badly conditions the SAW-BRDR weighting and optimizer step size.
    ref_energy: f32,
    ref_stress2: f32,
    /// `(equivalent_traction_pa / material.e * r_pin)²` — same `u_ref` formula Kirsch uses
    /// (`compute_reference_scales`), squared, so `InterfacePenetrationTerm`'s raw gap² [m²]
    /// normalizes to O(1) the same way `ref_stress2` normalizes raw Pa² terms.
    ref_gap2: f32,
}

impl PinLugProblem {
    /// `n_interface`: number of shared-theta interface points (index-aligned across pin and
    /// lug — see `InterfaceParametrization` doc comment). `scaling_mode` selects which
    /// stress magnitude `ref_energy`/`ref_stress2`/`ref_gap2` normalize against (see
    /// `PinLugScalingMode`).
    pub fn new(
        material: MaterialProps,
        output_dim: usize,
        phase1_steps: usize,
        n_interface: usize,
        scaling_mode: PinLugScalingMode,
    ) -> Self {
        let lug_geometry = GeometryConfig::pinlug_lug_inches();
        let pin_geometry = GeometryConfig::pinlug_pin_inches();
        let HoleType::Circular { radius: r_lug } = lug_geometry.hole else {
            panic!("PinLugProblem::new: lug geometry must have a circular hole");
        };
        let r_pin = pin_geometry.half_w; // pin disk radius (see pinlug_pin_inches doc comment)

        // Read scalar material properties BEFORE `material` gets partially moved/cloned
        // into the `DomainSpec`s below.
        let material_e = material.e;
        let material_uts = material.ultimate_strength_pa;

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

        let stress_ref = match scaling_mode {
            PinLugScalingMode::AppliedLoad => equivalent_traction_pa,
            PinLugScalingMode::UltimateStrength => material_uts,
        };

        // Same formula as `training_core::compute_reference_scales`, with `stress_ref`
        // standing in for Kirsch's far-field `config.load.px`.
        let ref_energy = (0.5 * stress_ref * stress_ref / material_e) as f32;
        let ref_stress2 = (stress_ref * stress_ref) as f32;
        let u_ref = (stress_ref / material_e) * r_pin;
        let ref_gap2 = (u_ref * u_ref) as f32;

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
            ref_energy,
            ref_stress2,
            ref_gap2,
        }
    }

    pub fn interface_thetas(&self) -> &[f64] { &self.interface.thetas }
    pub fn equivalent_traction_pa(&self) -> f64 { self.equivalent_traction_pa }

    #[cfg(test)]
    pub(crate) fn ref_stress2_for_test(&self) -> f32 { self.ref_stress2 }
    #[cfg(test)]
    pub(crate) fn ref_energy_for_test(&self) -> f32 { self.ref_energy }
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
        // Driving traction points in +x (pin pushed into the lug along the loading axis),
        // uniform magnitude `equivalent_traction_pa` — see module doc comment's force-to-
        // traction derivation. Length must match `N_DRIVE_POINTS` (the "driving" named
        // point-set's actual sampled count in `PinLugSamplingStrategy::named_point_sets`).
        let tx_target = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![self.equivalent_traction_pa as f32; N_DRIVE_POINTS], vec![N_DRIVE_POINTS]),
            &device,
        );
        let ty_target = Tensor::<B, 1>::zeros([N_DRIVE_POINTS], &device);
        vec![
            Box::new(InteriorEnergyTerm { domain: PIN_DOMAIN, material: material.clone(), ref_energy: self.ref_energy }),
            Box::new(InteriorEnergyTerm { domain: LUG_DOMAIN, material: material.clone(), ref_energy: self.ref_energy }),
            Box::new(LugShankAnchorTerm),
            Box::new(LugFreeEdgeTractionTerm { ref_stress2: self.ref_stress2 }),
            Box::new(PinDrivingTractionTerm {
                material, ref_stress2: self.ref_stress2,
                tx_target,
                ty_target,
            }),
            Box::new(InterfacePenetrationTerm {
                thetas: self.interface.thetas.clone(),
                r_pin: self.domains[0].geometry.half_w,
                r_lug: match self.domains[1].geometry.hole { HoleType::Circular { radius } => radius, HoleType::None => 0.0 },
                ref_gap2: self.ref_gap2,
            }),
            Box::new(InterfaceNonTensionTerm { thetas: self.interface.thetas.clone(), ref_stress2: self.ref_stress2 }),
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
    ///
    /// Mirrors `InterfacePenetrationTerm::compute`'s gap formula and sign convention exactly
    /// (module doc comment): `gap(theta) = (r_lug + u_r_lug) - (r_pin + u_r_pin)`. Unlike that
    /// loss term (which reads `raw_out` already produced mid-training-step by
    /// `step_physics_multi`), this diagnostic runs its own tiny forward pass per domain over
    /// just the shared interface points — cheap (`n_interface` points, no FD stencil/strains
    /// needed since only the raw displacement columns 0/1 are read).
    ///
    /// `n_fourier=0` and identity ansatz scaling (raw output x `u_ref` = physical
    /// displacement in meters) match `step_physics_multi`'s pin-in-lug convention exactly
    /// (see its doc comment: "Multi-domain problems (pin-in-lug) use plain-DEM output (no
    /// Fourier embedding...")) — NOT Kirsch's `has_hole`-gated `n_fourier=8`.
    fn convergence_metric(&self, state: &[DomainState<B>]) -> Option<f64> {
        if state.len() != self.domains.len() {
            return None;
        }
        use crate::fd_stencil::norm_pts_to_tensor;
        use crate::network::fwd;
        use crate::training_core::normalize_point;
        use pinn_core::messages::SolverConfig;

        const N_FOURIER: usize = 0; // see doc comment above

        let pin_state = state.iter().find(|s| s.id == PIN_DOMAIN)?;
        let lug_state = state.iter().find(|s| s.id == LUG_DOMAIN)?;
        let pin_geom = &self.domains[0].geometry;
        let lug_geom = &self.domains[1].geometry;
        let r_pin = pin_geom.half_w;
        let r_lug = match lug_geom.hole {
            HoleType::Circular { radius } => radius,
            HoleType::None => 0.0,
        };

        let device: burn::backend::wgpu::WgpuDevice = Default::default();

        // normalize_point only needs geometry ranges, threaded through a SolverConfig — build
        // a throwaway one per domain purely to reuse the exact normalization formula (single
        // source of truth) rather than reimplementing it.
        let cfg_for = |geom: &GeometryConfig| SolverConfig {
            geometry: geom.clone(),
            ..SolverConfig::default_pinlug()
        };
        let pin_cfg = cfg_for(pin_geom);
        let lug_cfg = cfg_for(lug_geom);

        let thetas = &self.interface.thetas;
        let pin_pts: Vec<[f32; 2]> = thetas.iter()
            .map(|&theta| normalize_point(r_pin * theta.cos(), r_pin * theta.sin(), &pin_cfg))
            .collect();
        let lug_pts: Vec<[f32; 2]> = thetas.iter()
            .map(|&theta| normalize_point(r_lug * theta.cos(), r_lug * theta.sin(), &lug_cfg))
            .collect();

        let pin_in = norm_pts_to_tensor::<B>(&pin_pts, &device);
        let lug_in = norm_pts_to_tensor::<B>(&lug_pts, &device);
        let pin_raw = fwd::<B>(&pin_state.model, pin_in, N_FOURIER, &device);
        let lug_raw = fwd::<B>(&lug_state.model, lug_in, N_FOURIER, &device);

        let n = thetas.len();
        let pin_u: Vec<f32> = pin_raw.clone().slice([0..n, 0..1]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let pin_v: Vec<f32> = pin_raw.slice([0..n, 1..2]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let lug_u: Vec<f32> = lug_raw.clone().slice([0..n, 0..1]).reshape([n]).into_data().to_vec().unwrap_or_default();
        let lug_v: Vec<f32> = lug_raw.slice([0..n, 1..2]).reshape([n]).into_data().to_vec().unwrap_or_default();

        let pin_u_ref = pin_state.u_ref as f64;
        let lug_u_ref = lug_state.u_ref as f64;

        let mut sum_sq = 0.0_f64;
        let mut finite_n = 0usize;
        for i in 0..n {
            let theta = thetas[i];
            let c = theta.cos();
            let s = theta.sin();
            let u_r_pin = (pin_u[i] as f64 * pin_u_ref) * c + (pin_v[i] as f64 * pin_u_ref) * s;
            let u_r_lug = (lug_u[i] as f64 * lug_u_ref) * c + (lug_v[i] as f64 * lug_u_ref) * s;
            let gap = (r_lug + u_r_lug) - (r_pin + u_r_pin);
            // A NaN-diverged network poisons every combined gap (both domains contribute to
            // each theta), so filter non-finite contributions rather than let a single NaN
            // silently poison the whole RMS (which would then poison `ConvergenceTracker`'s
            // internal max/min folds, breaking the crash/plateau cascade the same way an
            // unfiltered NaN reading does — see `SawBrdr::update`'s analogous fix).
            if gap.is_finite() {
                sum_sq += gap * gap;
                finite_n += 1;
            }
        }
        if finite_n == 0 {
            return None;
        }
        let rms = (sum_sq / finite_n as f64).sqrt();
        Some(rms)
    }

    fn convergence_target(&self) -> f64 { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_metric_pinlug_returns_none_when_state_slice_empty() {
        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        assert_eq!(problem.convergence_metric(&[]), None);
    }

    #[test]
    fn validate_loss_terms_accepts_well_formed_pinlug_problem() {
        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        crate::problem::validate_loss_terms(&problem); // must not panic
    }

    /// Issue 1 regression: the lug's shank edge (x = x0) must NOT appear in the general
    /// `sample_boundary` traction-free point set consumed by `LugFreeEdgeTractionTerm` — that
    /// physical edge is exclusively owned by the "shank_anchor" named point set /
    /// `LugShankAnchorTerm` (fixed-displacement). A boundary point cannot be both
    /// traction-free and rigidly clamped.
    #[test]
    fn lug_sample_boundary_excludes_shank_edge_owned_by_anchor_term() {
        let lug_geometry = GeometryConfig::pinlug_lug_inches();
        let (x0, _x1) = lug_geometry.x_range();
        let interface = Arc::new(InterfaceParametrization { thetas: vec![0.0] });
        let strategy = PinLugSamplingStrategy { is_pin: false, interface, contact_radius: 0.5 * pinn_core::units::IN_TO_M };
        let load = LoadConfig::uniaxial_x(1.0);
        let pts = strategy.sample_boundary(&lug_geometry, &load, 300);

        assert!(!pts.is_empty(), "sanity: sample_boundary should still produce points on the other 3 edges");
        const TOL: f64 = 1e-9;
        for p in &pts {
            assert!(
                (p.x - x0).abs() > TOL,
                "found a NeumannFree boundary point at x={} (shank edge x0={}) — this edge must only \
                 be constrained via the dedicated shank_anchor point set, not the general traction-free \
                 boundary loop (contradictory BCs)",
                p.x, x0,
            );
            assert_eq!(p.kind, BoundaryKind::NeumannFree);
        }
    }

    /// Real `convergence_metric` test: freshly-initialized (lazy-Param, force-materialized)
    /// 2-domain state must produce `Some(finite, non-negative)` RMS gap, not `None`/NaN/panic.
    /// A hand-computed exact value isn't practical here (`ElasticityNet`'s random init means
    /// the raw network output at the interface points is not analytically predictable), but
    /// the zero-weights case below pins down the exact formula/convention instead.
    #[test]
    fn convergence_metric_pinlug_returns_finite_nonnegative_rms_for_fresh_state() {
        use crate::network::ElasticityNetConfig;
        use crate::problem::DomainState;

        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5)
            .with_use_piratenet(false);

        let model_pin: crate::network::ElasticityNet<B> = net_cfg.init(&device);
        let model_lug: crate::network::ElasticityNet<B> = net_cfg.init(&device);

        let state = vec![
            DomainState { id: PIN_DOMAIN, model: model_pin, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
            DomainState { id: LUG_DOMAIN, model: model_lug, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
        ];

        let metric = problem.convergence_metric(&state);
        assert!(metric.is_some(), "convergence_metric must return Some for a well-formed 2-domain state");
        let rms = metric.unwrap();
        assert!(rms.is_finite(), "RMS gap must be finite, got {rms}");
        assert!(rms >= 0.0, "RMS is a root-mean-square, must be non-negative, got {rms}");
        assert!(rms > 0.0, "a freshly-initialized (non-degenerate) network's raw output is essentially \
            never exactly 0 at every interface theta, so a hard 0.0 here would suggest the forward \
            pass silently short-circuited rather than actually running");
    }

    /// Pins down the exact gap formula/sign convention `convergence_metric` must use, using
    /// zero-initialized weights+biases so every raw network output is deterministically 0 —
    /// then gap(theta) reduces to `r_lug - r_pin` at every theta (u_r terms vanish), letting
    /// the expected RMS be computed by hand instead of merely checked for finiteness.
    #[test]
    fn convergence_metric_pinlug_matches_hand_computed_gap_at_zero_displacement() {
        use burn::module::{Module, ModuleMapper, Param};
        use crate::network::ElasticityNetConfig;
        use crate::problem::DomainState;

        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5)
            .with_use_piratenet(false);

        // Zero out every float param so the network outputs exactly 0 for any input
        // (all-zero weights/biases -> every linear layer outputs 0 regardless of input).
        struct ZeroMapper;
        impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for ZeroMapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
                param.map(|t| t.zeros_like())
            }
        }
        let model_pin: crate::network::ElasticityNet<B> = net_cfg.init(&device).map(&mut ZeroMapper);
        let model_lug: crate::network::ElasticityNet<B> = net_cfg.init(&device).map(&mut ZeroMapper);

        let state = vec![
            DomainState { id: PIN_DOMAIN, model: model_pin, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
            DomainState { id: LUG_DOMAIN, model: model_lug, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
        ];

        let rms = problem.convergence_metric(&state).expect("must be Some for well-formed state");

        let r_pin = problem.domains[0].geometry.half_w;
        let r_lug = match problem.domains[1].geometry.hole { HoleType::Circular { radius } => radius, HoleType::None => 0.0 };
        let expected = (r_lug - r_pin).abs(); // gap is identical at every theta -> RMS = |gap|
        assert!(
            (rms - expected).abs() < 1e-9,
            "expected RMS gap {expected} (= |r_lug - r_pin| at zero displacement), got {rms}"
        );
    }

    #[test]
    fn convergence_metric_pinlug_returns_none_when_forward_pass_is_all_nan() {
        use burn::module::{Module, ModuleMapper, Param};
        use crate::network::ElasticityNetConfig;
        use crate::problem::DomainState;

        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5)
            .with_use_piratenet(false);

        struct NanMapper;
        impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for NanMapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
                param.map(|t| t.zeros_like().add_scalar(f32::NAN))
            }
        }
        let model_pin: crate::network::ElasticityNet<B> = net_cfg.init(&device).map(&mut NanMapper);
        let model_lug: crate::network::ElasticityNet<B> = net_cfg.init(&device).map(&mut NanMapper);

        let state = vec![
            DomainState { id: PIN_DOMAIN, model: model_pin, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
            DomainState { id: LUG_DOMAIN, model: model_lug, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
        ];

        assert_eq!(
            problem.convergence_metric(&state), None,
            "an all-NaN forward pass must yield None (every gap non-finite), not Some(NaN)"
        );
    }

    #[test]
    fn convergence_metric_pinlug_returns_none_when_only_one_domain_is_nan() {
        use burn::module::{Module, ModuleMapper, Param};
        use crate::network::ElasticityNetConfig;
        use crate::problem::DomainState;

        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(8)
            .with_n_hidden(2)
            .with_output_dim(5)
            .with_use_piratenet(false);

        struct NanMapper;
        impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for NanMapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
                param.map(|t| t.zeros_like().add_scalar(f32::NAN))
            }
        }
        // Only the PIN domain diverges to NaN; the LUG domain stays well-formed.
        let model_pin: crate::network::ElasticityNet<B> = net_cfg.init(&device).map(&mut NanMapper);
        let model_lug: crate::network::ElasticityNet<B> = net_cfg.init(&device);

        let state = vec![
            DomainState { id: PIN_DOMAIN, model: model_pin, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
            DomainState { id: LUG_DOMAIN, model: model_lug, u_ref: 1e-4, ref_energy: 1.0, ref_stress2: 1.0 },
        ];

        // Since gap(theta) combines BOTH domains' displacement, a single diverged domain must
        // still poison every combined point -> None (not silently averaged away by the
        // healthy domain).
        assert_eq!(
            problem.convergence_metric(&state), None,
            "one NaN'd domain must poison every combined gap -> None, not a partial/averaged result"
        );
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

        let term = InterfacePenetrationTerm { thetas, r_pin, r_lug, ref_gap2: 1e-8 };
        let loss = term.compute(&[pin_fwd, lug_fwd]);
        let v: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(v, 0.0, "zero displacement + equal radii must give exactly zero gap everywhere -> zero penalty");
    }

    #[test]
    fn interface_penetration_term_new_impl_matches_old_cpu_math_mixed_active_inactive() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let n = 4;
        let thetas: Vec<f64> = (0..n).map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64).collect();
        let (r_pin, r_lug) = (0.5_f64, 0.52_f64);

        let pin_u = [0.05_f32, 0.0, -0.05, 0.0];
        let pin_v = [0.0_f32, 0.0, 0.0, 0.08];
        let lug_u = [0.0_f32, 0.0, 0.0, 0.0];
        let lug_v = [0.0_f32, 0.1, 0.0, 0.0];

        let mut pin_flat = vec![0.0_f32; n * 5];
        let mut lug_flat = vec![0.0_f32; n * 5];
        for i in 0..n {
            pin_flat[i * 5] = pin_u[i];
            pin_flat[i * 5 + 1] = pin_v[i];
            lug_flat[i * 5] = lug_u[i];
            lug_flat[i * 5 + 1] = lug_v[i];
        }
        let pin_raw = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(pin_flat, vec![n, 5]), &device);
        let lug_raw = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(lug_flat, vec![n, 5]), &device);

        let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
        let lug_fwd = DomainForwardOutputs { domain: LUG_DOMAIN, raw_out: &lug_raw, strains: None, normals: None };

        let term = InterfacePenetrationTerm { thetas: thetas.clone(), r_pin, r_lug, ref_gap2: 1.0 };
        let actual: f32 = term.compute(&[pin_fwd, lug_fwd]).into_data().to_vec::<f32>().unwrap()[0];

        let mut total = 0.0_f64;
        for i in 0..n {
            let c = thetas[i].cos();
            let s = thetas[i].sin();
            let u_r_pin = pin_u[i] as f64 * c + pin_v[i] as f64 * s;
            let u_r_lug = lug_u[i] as f64 * c + lug_v[i] as f64 * s;
            let gap = (r_lug + u_r_lug) - (r_pin + u_r_pin);
            total += crate::signorini::penetration_penalty(gap);
        }
        let expected = (total / n as f64) as f32;

        assert!((expected - 0.00045).abs() < 1e-6, "fixture sanity: expected ~0.00045, got {expected}");
        let scale = actual.abs().max(expected.abs()).max(1e-8);
        assert!((actual - expected).abs() / scale < 1e-4,
            "new Tensor impl diverges from old CPU-f64 oracle: actual={actual} expected={expected}");
    }

    #[test]
    fn interface_non_tension_term_new_impl_matches_old_cpu_math_mixed_active_inactive() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let n = 4;
        let thetas: Vec<f64> = (0..n).map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64).collect();

        let sxx = [50.0_f32, 0.0, -20.0, 0.0];
        let syy = [0.0_f32, -30.0, 0.0, 10.0];
        let sxy = [0.0_f32; 4];

        let mut pin_flat = vec![0.0_f32; n * 5];
        for i in 0..n {
            pin_flat[i * 5 + 2] = sxx[i];
            pin_flat[i * 5 + 3] = syy[i];
            pin_flat[i * 5 + 4] = sxy[i];
        }
        let pin_raw = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(pin_flat, vec![n, 5]), &device);
        let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };

        let term = InterfaceNonTensionTerm { thetas: thetas.clone(), ref_stress2: 1.0 };
        let actual: f32 = term.compute(&[pin_fwd]).into_data().to_vec::<f32>().unwrap()[0];

        let mut total = 0.0_f64;
        for i in 0..n {
            let (s_rr, _, _) = crate::signorini::decompose_radial(sxx[i] as f64, syy[i] as f64, sxy[i] as f64, thetas[i]);
            total += crate::signorini::non_tension_penalty(s_rr);
        }
        let expected = (total / n as f64) as f32;

        assert!((expected - 650.0).abs() < 1e-3, "fixture sanity: expected 650.0, got {expected}");
        let scale = actual.abs().max(expected.abs()).max(1e-8);
        assert!((actual - expected).abs() / scale < 1e-4,
            "new Tensor impl diverges from old CPU-f64 oracle: actual={actual} expected={expected}");
    }

    #[test]
    fn interface_penetration_term_gradient_nonzero_when_penetrating() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let thetas = vec![0.0_f64];
        let (r_pin, r_lug) = (0.5_f64, 0.5_f64);

        let pin_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1_f32, 0.0, 0.0, 0.0, 0.0], vec![1, 5]), &device,
        ).require_grad();
        let lug_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32; 5], vec![1, 5]), &device,
        ).require_grad();

        let term = InterfacePenetrationTerm { thetas, r_pin, r_lug, ref_gap2: 1.0 };
        let loss = {
            let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
            let lug_fwd = DomainForwardOutputs { domain: LUG_DOMAIN, raw_out: &lug_raw, strains: None, normals: None };
            term.compute(&[pin_fwd, lug_fwd])
        };

        let loss_v: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!((loss_v - 0.01).abs() < 1e-5, "expected loss=0.01 (gap=-0.1), got {loss_v}");

        let grads = loss.backward();
        let pin_grad = pin_raw.grad(&grads)
            .expect("pin_raw must receive a gradient — compute() must not detach from the autodiff graph");
        let lug_grad = lug_raw.grad(&grads)
            .expect("lug_raw must receive a gradient — compute() must not detach from the autodiff graph");

        let pin_grad_v: Vec<f32> = pin_grad.into_data().to_vec().unwrap();
        let lug_grad_v: Vec<f32> = lug_grad.into_data().to_vec().unwrap();

        assert!((pin_grad_v[0] - 0.2).abs() < 1e-4, "d(loss)/d(pin_u) expected 0.2, got {}", pin_grad_v[0]);
        assert!((lug_grad_v[0] - (-0.2)).abs() < 1e-4, "d(loss)/d(lug_u) expected -0.2, got {}", lug_grad_v[0]);
        for &g in &pin_grad_v[1..] { assert!(g.abs() < 1e-6, "unrelated pin column must have ~0 gradient, got {g}"); }
        for &g in &lug_grad_v[1..] { assert!(g.abs() < 1e-6, "unrelated lug column must have ~0 gradient, got {g}"); }
    }

    #[test]
    fn interface_penetration_term_gradient_zero_when_non_penetrating() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let thetas = vec![0.0_f64];
        let (r_pin, r_lug) = (0.5_f64, 0.5_f64);

        let pin_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![-0.1_f32, 0.0, 0.0, 0.0, 0.0], vec![1, 5]), &device,
        ).require_grad();
        let lug_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32; 5], vec![1, 5]), &device,
        ).require_grad();

        let term = InterfacePenetrationTerm { thetas, r_pin, r_lug, ref_gap2: 1.0 };
        let loss = {
            let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
            let lug_fwd = DomainForwardOutputs { domain: LUG_DOMAIN, raw_out: &lug_raw, strains: None, normals: None };
            term.compute(&[pin_fwd, lug_fwd])
        };

        let loss_v: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(loss_v, 0.0, "non-penetrating gap must give exactly zero penalty");

        let grads = loss.backward();
        let pin_grad_v: Vec<f32> = pin_raw.grad(&grads)
            .expect("tensor is still part of the graph even with zero local gradient")
            .into_data().to_vec().unwrap();
        let lug_grad_v: Vec<f32> = lug_raw.grad(&grads).unwrap().into_data().to_vec().unwrap();

        for &g in &pin_grad_v { assert!(g.abs() < 1e-6, "inactive region must have ~0 gradient, got {g}"); }
        for &g in &lug_grad_v { assert!(g.abs() < 1e-6, "inactive region must have ~0 gradient, got {g}"); }
    }

    #[test]
    fn interface_non_tension_term_gradient_nonzero_when_tensile() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let thetas = vec![0.0_f64];

        let pin_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32, 0.0, 2.0, 0.0, 0.0], vec![1, 5]), &device,
        ).require_grad();

        let term = InterfaceNonTensionTerm { thetas, ref_stress2: 1.0 };
        let loss = {
            let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
            term.compute(&[pin_fwd])
        };

        let loss_v: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!((loss_v - 4.0).abs() < 1e-4, "expected loss=4.0 (s_rr=2.0), got {loss_v}");

        let grads = loss.backward();
        let pin_grad_v: Vec<f32> = pin_raw.grad(&grads)
            .expect("pin_raw must receive a gradient — compute() must not detach from the autodiff graph")
            .into_data().to_vec().unwrap();

        assert!((pin_grad_v[2] - 4.0).abs() < 1e-4, "d(loss)/d(sxx) expected 4.0, got {}", pin_grad_v[2]);
        assert!(pin_grad_v[0].abs() < 1e-6, "u column unrelated, expected ~0");
        assert!(pin_grad_v[1].abs() < 1e-6, "v column unrelated, expected ~0");
        assert!(pin_grad_v[3].abs() < 1e-6, "d(loss)/d(syy) expected ~0 at theta=0 (sin²=0)");
        assert!(pin_grad_v[4].abs() < 1e-6, "d(loss)/d(sxy) expected ~0 at theta=0 (2 sin cos=0)");
    }

    #[test]
    fn interface_non_tension_term_gradient_zero_when_compressive() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let thetas = vec![0.0_f64];

        let pin_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32, 0.0, -2.0, 0.0, 0.0], vec![1, 5]), &device,
        ).require_grad();

        let term = InterfaceNonTensionTerm { thetas, ref_stress2: 1.0 };
        let loss = {
            let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
            term.compute(&[pin_fwd])
        };

        let loss_v: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(loss_v, 0.0, "compressive s_rr must give exactly zero penalty");

        let grads = loss.backward();
        let pin_grad_v: Vec<f32> = pin_raw.grad(&grads)
            .expect("tensor is still part of the graph even with zero local gradient")
            .into_data().to_vec().unwrap();
        for &g in &pin_grad_v { assert!(g.abs() < 1e-6, "inactive region must have ~0 gradient, got {g}"); }
    }

    // All 6 pre-existing gradient/equivalence tests for InterfaceNonTensionTerm use thetas in
    // {0, pi/2, pi, 3pi/2} (sin*cos == 0 at every one) and/or sxy == 0, so none of them ever
    // exercises the `2.0 * sxy * sin(theta) * cos(theta)` shear cross-term in the s_rr radial
    // decomposition with a nonzero coefficient — a mutation dropping that factor of 2 (e.g.
    // `mul_scalar(2.0)` -> `mul_scalar(1.0)`, or the multiplication by `sxy` silently omitted)
    // would survive every one of them. theta = pi/4 makes sin*cos = 0.5 (its maximum
    // magnitude), and a pure-shear stress state (sxx = syy = 0, sxy != 0) isolates the cross
    // term from the cos^2/sin^2 terms entirely, so this test's expected values are sensitive
    // to that factor of 2 specifically.
    #[test]
    fn interface_non_tension_term_shear_cross_term_matches_oracle_at_45_degrees() {
        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let theta = std::f64::consts::FRAC_PI_4;
        let thetas = vec![theta];
        let sxy = 100.0_f32;

        let pin_raw = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32, 0.0, 0.0, 0.0, sxy], vec![1, 5]), &device,
        ).require_grad();

        let term = InterfaceNonTensionTerm { thetas: thetas.clone(), ref_stress2: 1.0 };
        let loss = {
            let pin_fwd = DomainForwardOutputs { domain: PIN_DOMAIN, raw_out: &pin_raw, strains: None, normals: None };
            term.compute(&[pin_fwd])
        };

        // Oracle: s_rr = sxx*cos^2 + syy*sin^2 + 2*sxy*sin*cos = 2*100*0.5 = 100 exactly
        // (sin(pi/4)*cos(pi/4) = 0.5), matching signorini::decompose_radial's s_rr formula.
        let (s_rr_oracle, _, _) = crate::signorini::decompose_radial(0.0, 0.0, sxy as f64, theta);
        assert!((s_rr_oracle - 100.0).abs() < 1e-9, "fixture sanity: expected s_rr=100.0, got {s_rr_oracle}");
        let expected_penalty = crate::signorini::non_tension_penalty(s_rr_oracle);
        assert!((expected_penalty - 10_000.0).abs() < 1e-6, "fixture sanity: expected penalty=10000, got {expected_penalty}");

        let loss_v: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!((loss_v - 10_000.0).abs() < 1e-2,
            "expected loss=10000 (s_rr=100 via the shear cross-term alone), got {loss_v} \
             — a missing factor of 2 in `2*sxy*sin*cos` would give s_rr=50, loss=2500");

        let grads = loss.backward();
        let pin_grad_v: Vec<f32> = pin_raw.grad(&grads)
            .expect("pin_raw must receive a gradient — compute() must not detach from the autodiff graph")
            .into_data().to_vec().unwrap();

        // d(loss)/d(sxy) = 2*s_rr * d(s_rr)/d(sxy) = 2*100*(2*sin*cos) = 2*100*1.0 = 200.
        // A missing factor of 2 would instead give 2*50*0.5 = 50 — clearly distinguishable.
        assert!((pin_grad_v[4] - 200.0).abs() < 1e-1,
            "d(loss)/d(sxy) expected 200.0, got {} — a missing factor of 2 in the shear \
             cross-term would give 50.0 instead", pin_grad_v[4]);
        assert!(pin_grad_v[0].abs() < 1e-6, "u column unrelated, expected ~0");
        assert!(pin_grad_v[1].abs() < 1e-6, "v column unrelated, expected ~0");
        // d(loss)/d(sxx) = 2*s_rr*cos^2 = 2*100*0.5 = 100, and symmetrically for syy via
        // sin^2 — nonzero even though sxx=syy=0 here, since s_rr itself is nonzero (driven
        // entirely by the cross term). These two are unaffected by the cross-term's factor of
        // 2 (cos^2/sin^2 have no such coefficient), so they serve as a control confirming the
        // 200.0 vs 50.0 distinction above is specific to the cross-term coefficient, not a
        // general scale error in the whole gradient.
        assert!((pin_grad_v[2] - 100.0).abs() < 1e-1, "d(loss)/d(sxx) expected 100.0, got {}", pin_grad_v[2]);
        assert!((pin_grad_v[3] - 100.0).abs() < 1e-1, "d(loss)/d(syy) expected 100.0, got {}", pin_grad_v[3]);
    }

    #[test]
    fn pinlug_scaling_mode_applied_load_matches_existing_traction_based_formula() {
        let material = MaterialProps::steel_4340();
        let problem = PinLugProblem::new(material.clone(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);

        let traction = problem.equivalent_traction_pa();
        let expected_ref_stress2 = (traction * traction) as f32;
        let expected_ref_energy = (0.5 * traction * traction / material.e) as f32;

        let ref_stress2 = problem.ref_stress2_for_test();
        let ref_energy = problem.ref_energy_for_test();

        assert!((ref_stress2 - expected_ref_stress2).abs() / expected_ref_stress2 < 1e-5);
        assert!((ref_energy - expected_ref_energy).abs() / expected_ref_energy < 1e-5);
    }

    #[test]
    fn pinlug_scaling_mode_ultimate_strength_diverges_from_applied_load() {
        let material = MaterialProps::steel_4340();
        let problem_applied = PinLugProblem::new(material.clone(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let problem_uts = PinLugProblem::new(material.clone(), 5, 2000, 16, PinLugScalingMode::UltimateStrength);

        let ref_stress2_applied = problem_applied.ref_stress2_for_test();
        let ref_stress2_uts = problem_uts.ref_stress2_for_test();

        assert!(
            (ref_stress2_uts - ref_stress2_applied).abs() / ref_stress2_applied > 0.01,
            "UltimateStrength mode must diverge from AppliedLoad mode by >1% relative"
        );

        let uts = material.ultimate_strength_pa;
        let expected_ref_stress2 = (uts * uts) as f32;
        let expected_ref_energy = (0.5 * uts * uts / material.e) as f32;

        assert!((ref_stress2_uts - expected_ref_stress2).abs() / expected_ref_stress2 < 1e-5);
        let ref_energy_uts = problem_uts.ref_energy_for_test();
        assert!((ref_energy_uts - expected_ref_energy).abs() / expected_ref_energy < 1e-5);
    }

    /// Pins each of PinLugProblem's 7 loss terms' expected `conflict_group()` classification
    /// AND the exact 2-Physics/5-Bc count split — catches a future term silently defaulting
    /// to `Bc` without an explicit override (see `LossTerm::conflict_group`'s default).
    #[test]
    fn pinlug_loss_terms_have_expected_conflict_group_classification() {
        use crate::problem::ConflictGroup;

        let problem = PinLugProblem::new(MaterialProps::steel_4340(), 5, 2000, 16, PinLugScalingMode::AppliedLoad);
        let terms = problem.loss_terms();
        assert_eq!(terms.len(), 7, "expected exactly 7 loss terms (2 interior-energy + 5 BC/interface)");

        let expected: &[(&str, ConflictGroup)] = &[
            ("pin_interior_energy", ConflictGroup::Physics),
            ("lug_interior_energy", ConflictGroup::Physics),
            ("lug_shank_anchor", ConflictGroup::Bc),
            ("lug_free_edge_traction", ConflictGroup::Bc),
            ("pin_driving_traction", ConflictGroup::Bc),
            ("interface_penetration", ConflictGroup::Bc),
            ("interface_non_tension", ConflictGroup::Bc),
        ];

        for term in &terms {
            let (_, expected_group) = expected.iter().find(|(name, _)| *name == term.name())
                .unwrap_or_else(|| panic!("unexpected loss term '{}' not in expected table", term.name()));
            assert_eq!(term.conflict_group(), *expected_group,
                "term '{}' has conflict_group {:?}, expected {:?}", term.name(), term.conflict_group(), expected_group);
        }

        let n_physics = terms.iter().filter(|t| t.conflict_group() == ConflictGroup::Physics).count();
        let n_bc = terms.iter().filter(|t| t.conflict_group() == ConflictGroup::Bc).count();
        assert_eq!(n_physics, 2, "expected exactly 2 Physics-group terms (pin+lug interior energy)");
        assert_eq!(n_bc, 5, "expected exactly 5 Bc-group terms");
    }
}
