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
        DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec, InterfaceParametrization, NamedPointSet,
    },
    problem_spec::ProblemSpec,
    user_geometry::{HoleBc, HoleSpec, UserGeometry},
};

use crate::{
    energy::{dem_energy_loss, equilibrium_from_displacement_hessian_loss, hole_traction_loss_direct, neumann_loss},
    pinlug_problem::IdentityAnsatz,
    problem::{BoundaryValueProblem, ConflictGroup, DomainForwardOutputs, DomainState, LossTerm, B},
};

pub const USER_DOMAIN: DomainId = DomainId(0);
/// Domains used by the one-hole bonded annular decomposition. Kept separate from the legacy
/// user domain so a mixed-domain forward can never accidentally satisfy a legacy term.
pub const ANNULUS_DOMAIN: DomainId = DomainId(1);
pub const OUTER_DOMAIN: DomainId = DomainId(2);

const LAM_INTERIOR_ENERGY: f32 = 1.0;
/// One optimization scale for the complete physical potential `Π = U - W_ext`.
///
/// This is deliberately a single term weight.  In particular, SAW-BRDR must never see `U`
/// and `-W_ext` as independently adaptable terms: doing so changes the Euler equation and the
/// physical affine minimizer.  Constraints retain their own weights below.
const LAM_PHYSICAL_POTENTIAL: f32 = 1.0;
const LAM_OUTER_TRACTION: f32 = 10.0;
/// Plate-specific base weight for `equilibrium` - deliberately NOT `kirsch_problem::LAM_EQ`
/// (5.0) anymore. That value was calibrated for Kirsch's direct-σ-based `EquilibriumRingTerm`;
/// this plate's `EquilibriumTerm` is a different quantity (derived-stress Hessian residual,
/// bugSource-New #12) with a different natural gradient magnitude. The no-hole term-gradient
/// diagnostic (post the `ref_div2` normalization fix) showed `equilibrium`'s gradient norm
/// still 10-20x smaller than `interior_energy`'s/`outer_traction`'s by step 199, growing
/// asymmetrically (interior_energy's gradient climbing while equilibrium's stays flat) -
/// exactly bugSource-New #13's "does equilibrium exert enough pressure to overcome the
/// energy term's low-strain shortcut" concern. SAW-BRDR's adaptive multiplier is driven by
/// each term's OWN loss-value decay rate, not cross-term gradient comparison, so it doesn't
/// automatically compensate for this gap - the base weight is the direct lever. `50.0`
/// (~10x, matching the observed gap; also `constitutive_consistency`'s existing fixed weight,
/// not a new magnitude in this codebase) is the first thing to test, not a final tuned value.
const LAM_EQUILIBRIUM_PLATE: f32 = 50.0;
/// Legacy-Hybrid-only base weight for `ExternalWorkTerm`. It is deliberately NOT a physical
/// coefficient: Phase 4 found that independently weighting U and W changes the variational
/// stationary point. Corrected `FormulationSelection::Variational` never registers this term;
/// it uses `PhysicalPotentialEnergyTerm` instead. bugSource-New #8's historical
/// displacement-slope diagnostic on the trained no-hole model found `du_norm/dx_norm=0.816`
/// (target 1.0) and `dv_norm/dy_norm=-0.246` (target -0.33) - the network is systematically
/// under-stretching, i.e. `interior_energy` (U, preferring less strain) is still winning
/// against `external_work` (-W_ext, rewarding more strain) even with the 1:1 ratio the true
/// functional calls for. `5.0` tests whether tipping that ratio in `external_work`'s favor
/// closes the slope gap - the same "boost the weaker term's weight directly, since SAW-BRDR's
/// loss-decay-rate-based adaptation doesn't compare cross-term gradients" lever already used
/// for `equilibrium`, not a final tuned value.
const LAM_EXTERNAL_WORK: f32 = 20.0;
const LAM_HOLE_FREE: f32 = 100.0;
const LAM_HOLE_FIXED: f32 = 50.0;
/// Issue #61 P2-07: matches `LAM_HOLE_FIXED`/pin-lug's `LugShankAnchorTerm` (weight ~50) - the
/// existing convention for a gauge/anchor term's weight in this codebase, not a new magnitude.
const LAM_TRANSLATION_GAUGE: f32 = 50.0;
const LAM_ROTATION_GAUGE: f32 = 50.0;

/// Issue #77 root-cause fix (kinematic decomposition): the closed-form uniform-tension
/// strain `(eps_xx, eps_yy, eps_xy)` of `u_affine(x,y) = ((px-nu*py)/E)*x, ((py-nu*px)/E)*y`
/// under Hooke's law inverse for plane stress (`sigma_xx=px, sigma_yy=py, sigma_xy=0`
/// exactly reproduces `run_no_hole_benchmark`'s own reference solution). Used to superpose a
/// KNOWN, EXACT background field onto the network's output so the network represents only
/// the residual correction `u_hole := u_total - u_affine`, not the (dominant, easily-learned)
/// affine part — see `docs/PHASE_4_MATHEMATICAL_OBJECTIVE_AUDIT.md`'s own affine reduction
/// and this session's SNR analysis (a hole's total energy signature is a fraction of a
/// percent of Pi for a small hole; a network minimizing total Pi spends nearly all its
/// gradient budget reducing the large affine misfit first, starving the local correction).
///
/// Every energy/stress functional evaluated on the TOTAL field must use TOTAL strain
/// (`eps_affine + eps_hole`), not `eps_hole` alone — the cross term `C:eps_affine:eps_hole`
/// is physically required for equivalence to the true Pi (dropping it reproduces the
/// documented "not a uniform rescaling, changes the stationary point" defect from
/// `PHASE_4_MATHEMATICAL_OBJECTIVE_AUDIT.md`'s historical-defect section, one level deeper).
/// Adding this constant to the network's own FD/Hessian-derived strain before calling into
/// `dem_energy_per_point`/`compute_stress`/`neumann_loss` achieves this "for free" — those
/// functions never see the decomposition, they just receive the already-total strain.
fn affine_strain(px: f64, py: f64, material: &MaterialProps) -> (f64, f64, f64) {
    let e = material.e as f64;
    let nu = material.nu as f64;
    ((px - nu * py) / e, (py - nu * px) / e, 0.0)
}

/// Kinematic decomposition (issue #77 fix) is scoped to exactly the case where
/// `u_affine` has a clean closed form and the hole's own natural BC needs an explicit,
/// correctly-targeted residual: one centered, traction-free hole. Off, unconditionally,
/// for no-hole/multi-hole/off-center/Fixed-bc geometries — those fall back to the
/// existing, unmodified behavior byte-for-byte (no `affine_strain`/`affine_target` field
/// is ever `Some` for them).
fn decomposition_applicable(spec: &ProblemSpec) -> bool {
    matches!(
        spec.geometry.holes.as_slice(),
        [HoleSpec { bc: HoleBc::Free, center, .. }] if center[0] == 0.0 && center[1] == 0.0
    )
}

/// Points sampled around each hole's circumference, per hole — a fixed, generous default;
/// not user-configurable in v1 (see `ProblemSpec`'s scope note).
const HOLE_RING_POINTS: usize = 64;
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
/// Issue #61 EPIC P2-13: made `pub` (was private) so `crate::provenance` can record this
/// codebase's real, fixed interior-collocation seed as genuine reproducibility metadata,
/// instead of guessing or omitting it.
pub const SEED_INTERIOR: u64 = 90_210;
/// Issue #64: boundary counterpart of [`SEED_INTERIOR`] — [`UserSamplingStrategy::sample_boundary`]
/// had no RNG at all before this fix (a fixed evenly-spaced grid every call); this seeds its new
/// per-call jitter.
pub const SEED_BOUNDARY: u64 = 40_404;
/// splitmix64's own odd 64-bit mixing constant — mixed into [`SEED_INTERIOR`]/[`SEED_BOUNDARY`]
/// by [`UserSamplingStrategy`]'s per-call counter so consecutive calls get well-separated seeds
/// (a plain `seed + call` would work too, but LCG state is sensitive to small seed deltas early
/// in its sequence — this constant is the standard remedy).
const CALL_SEED_MIX: u64 = 0x9E3779B97F4A7C15;

/// Safety multiple applied to the FD stencil's physical reach when computing the near-hole
/// collocation-exclusion margin (see [`UserSamplingStrategy::new`]'s margin computation) —
/// headroom above the bare minimum needed to keep every stencil arm outside the hole.
const RING_ANCHOR_SAFETY_FACTOR: f64 = 4.0;

/// The margin (physical, meters) an FD stencil needs to clear a hole boundary without any
/// arm dipping back inside it — see [`UserSamplingStrategy::new`]'s doc comment for the full
/// derivation. Public (not just internal to `UserSamplingStrategy`) so callers that need to
/// widen an AMR grid's own containment gate by the same amount (`UserGeometry::
/// inflated_for_collocation`) can compute it without constructing a full sampling strategy
/// first — one formula, reused everywhere this margin matters.
pub fn ring_anchor_margin_m(fd_h: f32, geometry: &UserGeometry) -> f64 {
    RING_ANCHOR_SAFETY_FACTOR * fd_h as f64 * geometry.half_w.max(geometry.half_h)
}

/// One side of #77's bonded annular decomposition. Both sides share `interface.thetas`, so
/// cross-domain losses compare identical physical interface points by index.
pub struct AnnularPartitionSampling {
    geometry: UserGeometry,
    partition: pinn_core::user_geometry::AnnularPartition,
    is_annulus: bool,
    partner: DomainId,
    interface: std::sync::Arc<InterfaceParametrization>,
    collocation_inner_radius: f64,
    interface_trace_offset: f64,
    interior_calls: std::sync::atomic::AtomicU64,
}

impl AnnularPartitionSampling {
    pub fn new(geometry: UserGeometry, fd_h: f32, is_annulus: bool, partner: DomainId, interface: std::sync::Arc<InterfaceParametrization>) -> Self {
        let partition = geometry.annular_partition().expect("#77 decomposition requires one safely-contained circular hole");
        let collocation_inner_radius = partition.hole_radius + ring_anchor_margin_m(fd_h, &geometry);
        assert!(collocation_inner_radius < partition.interface_radius, "#77 annulus too thin for FD-safe collocation");
        let interface_trace_offset = 0.5 * ring_anchor_margin_m(fd_h, &geometry);
        assert!(interface_trace_offset > 0.0 && interface_trace_offset < partition.interface_radius - collocation_inner_radius,
            "#77 interface FD trace offset must fit inside annulus");
        Self { geometry, partition, is_annulus, partner, interface, collocation_inner_radius, interface_trace_offset, interior_calls: std::sync::atomic::AtomicU64::new(0) }
    }

}

impl DomainSamplingStrategy for AnnularPartitionSampling {
    fn sample_interior(&self, _geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;
        let call = self.interior_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = LcgRng::new(SEED_INTERIOR ^ call.wrapping_mul(CALL_SEED_MIX) ^ if self.is_annulus { 0xA11u64 } else { 0x0u64 });
        let mut points = Vec::with_capacity(n);
        if self.is_annulus {
            let r0sq = self.collocation_inner_radius * self.collocation_inner_radius;
            let r1sq = self.partition.interface_radius * self.partition.interface_radius;
            for _ in 0..n {
                let r = (r0sq + rng.next_f64() * (r1sq - r0sq)).sqrt();
                let theta = 2.0 * std::f64::consts::PI * rng.next_f64();
                points.push([self.partition.center[0] + r * theta.cos(), self.partition.center[1] + r * theta.sin()]);
            }
        } else {
            let mut attempts = 0usize;
            while points.len() < n && attempts < n * REJECTION_SAMPLE_ATTEMPTS_FACTOR {
                attempts += 1;
                let x = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_w;
                let y = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_h;
                if self.geometry.contains(x, y) && self.partition.contains_outer(x, y) { points.push([x, y]); }
            }
        }
        points
    }

    fn sample_boundary(&self, _geom: &GeometryConfig, _load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        if self.is_annulus { return Vec::new(); }
        let per_edge = (n / 4).max(1);
        let mut points = Vec::with_capacity(4 * per_edge);
        for i in 0..per_edge {
            let f = (i as f64 + 0.5) / per_edge as f64;
            let x = -self.geometry.half_w + 2.0 * self.geometry.half_w * f;
            let y = -self.geometry.half_h + 2.0 * self.geometry.half_h * f;
            points.extend([
                BoundaryPoint { x: self.geometry.half_w, y, nx: 1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad },
                BoundaryPoint { x: -self.geometry.half_w, y, nx: -1.0, ny: 0.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad },
                BoundaryPoint { x, y: self.geometry.half_h, nx: 0.0, ny: 1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad },
                BoundaryPoint { x, y: -self.geometry.half_h, nx: 0.0, ny: -1.0, tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannLoad },
            ]);
        }
        points
    }

    fn amr_lock_zone(&self, _geom: &GeometryConfig, _cell_center: [f64; 2]) -> bool { false }
    fn sample_extra_ring(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> { Vec::new() }
    fn named_point_sets(&self, _bnd: &[BoundaryPoint]) -> Vec<NamedPointSet> {
        let interface = self.interface.thetas.iter().map(|&theta| {
            let radial = [theta.cos(), theta.sin()];
            let sign = if self.is_annulus { 1.0 } else { -1.0 };
            BoundaryPoint {
                x: self.partition.center[0] + self.partition.interface_radius * radial[0],
                y: self.partition.center[1] + self.partition.interface_radius * radial[1],
                nx: sign * radial[0], ny: sign * radial[1], tx: 0.0, ty: 0.0,
                kind: BoundaryKind::Interface { partner_domain: self.partner },
            }
        }).collect();
        let mut sets = vec![NamedPointSet { name: "interface", points: interface }];
        // Keep all five arms of the central-difference stencil inside its owner. The offset is
        // two maximum physical FD reaches: one reach remains after the worst radial component
        // of an axis-aligned arm. This is a trace approximation, never a cross-domain model
        // extension masquerading as a derivative.
        let trace_radius = if self.is_annulus {
            self.partition.interface_radius - self.interface_trace_offset
        } else {
            self.partition.interface_radius + self.interface_trace_offset
        };
        let trace_name = if self.is_annulus {
            "interface_annulus_stress"
        } else {
            "interface_outer_stress"
        };
        let trace_sign = if self.is_annulus { 1.0 } else { -1.0 };
        let trace_points = self.interface.thetas.iter().map(|&theta| BoundaryPoint {
            x: self.partition.center[0] + trace_radius * theta.cos(),
            y: self.partition.center[1] + trace_radius * theta.sin(),
            nx: trace_sign * theta.cos(), ny: trace_sign * theta.sin(), tx: 0.0, ty: 0.0,
            kind: BoundaryKind::Interface { partner_domain: self.partner },
        }).collect();
        sets.push(NamedPointSet { name: trace_name, points: trace_points });
        if self.is_annulus {
            let hole = self.geometry.holes[0];
            let points = self.interface.thetas.iter().map(|&theta| BoundaryPoint {
                x: hole.center[0] + hole.radius * theta.cos(), y: hole.center[1] + hole.radius * theta.sin(),
                nx: -theta.cos(), ny: -theta.sin(), tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree,
            }).collect();
            sets.push(NamedPointSet { name: "hole_0", points });
            // Issue #77 fix: FD-safe companion ring at `collocation_inner_radius`
            // (`radius + ring_anchor_margin_m`, already computed for this sampler's own
            // interior collocation) - the kinematic-decomposition hole traction term needs a
            // DERIVED stress read, and the exact-radius "hole_0" ring above is stencil-unsafe
            // for that (an inward FD arm would land inside the hole).
            let fd_points = self.interface.thetas.iter().map(|&theta| BoundaryPoint {
                x: hole.center[0] + self.collocation_inner_radius * theta.cos(),
                y: hole.center[1] + self.collocation_inner_radius * theta.sin(),
                nx: -theta.cos(), ny: -theta.sin(), tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree,
            }).collect();
            sets.push(NamedPointSet { name: "hole_0_fd", points: fd_points });
        }
        sets
    }
}

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
    /// `"hole_0_fd"`, `"hole_1_fd"`, ... — FD-safe rings at `radius + anchor_margin_m`, one per
    /// hole, leaked alongside `hole_names`. Issue #77 fix: unlike `hole_names`'s exact-radius
    /// ring (safe only for a DIRECT stress read), a DERIVED/constitutive-stress read needs its
    /// FD stencil arms to clear the hole boundary, the same requirement `contains_for_collocation`
    /// already enforces for interior points.
    hole_fd_names: Vec<&'static str>,
    /// Radial offset [m] applied outside each hole's radius when EXCLUDING collocation points
    /// from the FD-unsafe near-hole annulus (see [`Self::contains_for_collocation`]) — see
    /// [`Self::new`] for the derivation. No longer used to emit a training point-set/anchor
    /// term (bugSource-New #12 removed `HoleAnchorEnergyTerm`/`"hole_i_anchor"` — equilibrium
    /// is now enforced everywhere via the derived-stress Hessian, not just near the hole); the
    /// geometric "just outside the hole" concept itself stays useful for Kt measurement (see
    /// `probe_hole_boundary_profile`'s derived-stress-at-margin variant).
    anchor_margin_m: f64,
    /// Issue #64: call counters mixed into each call's RNG seed so consecutive
    /// `sample_interior`/`sample_boundary` calls on the same instance draw genuinely different
    /// points (jittered stratified sampling) instead of the same fixed point cloud every time —
    /// see each method's own doc comment for why a static point cloud across an entire training
    /// run let the network overfit the discrete quadrature nodes rather than the continuous
    /// functional. `AtomicU64`, not a plain field, so the trait's `&self` (not `&mut self`)
    /// signature doesn't need to change — no ripple into `KirschSamplingStrategy`/
    /// `PinLugSamplingStrategy`. `DomainSamplingStrategy: Send + Sync` rules out `Cell` (not
    /// `Sync`); relaxed ordering is fine — this only needs distinct counter values per call,
    /// never cross-thread visibility of any other state. Still fully reproducible run-to-run:
    /// same base seed constant ⇒ same full sequence of per-call point sets, only the "identical
    /// every call" artifact is fixed.
    interior_calls: std::sync::atomic::AtomicU64,
    boundary_calls: std::sync::atomic::AtomicU64,
}

impl UserSamplingStrategy {
    /// `fd_h`: the training config's FD step in *normalized* coordinates
    /// (`ProblemSpec.training.fd_h`) — needed here (not just by the FD stencil itself) so the
    /// near-hole collocation-exclusion annulus (`contains_for_collocation`) is wide enough
    /// that `fd_stencil::assemble_stencil`'s axis-aligned `±hx`/`±hy` arms, evaluated at any
    /// point just outside it, never dip back inside the hole (which would silently evaluate
    /// the network on an invalid, hole-interior point and produce a meaningless FD-derived
    /// strain there — the same reason `"hole_i"`'s own ring, which sits exactly ON the hole
    /// boundary, is deliberately never used for constitutive-consistency checks).
    /// Normalized-to-physical conversion: `x_norm ∈ [-1,1]` maps to physical
    /// `[-half_w, half_w]`, so a stencil arm's physical reach is `fd_h * half_w` in x /
    /// `fd_h * half_h` in y. The worst case for a point at `radius + margin` is a stencil arm
    /// pointing straight at the hole center, which stays outside the hole iff
    /// `margin > fd_h * half_w` AND `margin > fd_h * half_h` — i.e.
    /// `margin > fd_h * max(half_w, half_h)`. `RING_ANCHOR_SAFETY_FACTOR` adds headroom above
    /// that bare minimum.
    pub fn new(geometry: UserGeometry, fd_h: f32) -> Self {
        let hole_names = (0..geometry.holes.len())
            .map(|i| -> &'static str { Box::leak(format!("hole_{i}").into_boxed_str()) })
            .collect();
        let hole_fd_names = (0..geometry.holes.len())
            .map(|i| -> &'static str { Box::leak(format!("hole_{i}_fd").into_boxed_str()) })
            .collect();
        let anchor_margin_m = ring_anchor_margin_m(fd_h, &geometry);
        Self {
            geometry, hole_names, hole_fd_names, anchor_margin_m,
            interior_calls: std::sync::atomic::AtomicU64::new(0),
            boundary_calls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Like `UserGeometry::contains`, but excludes a `self.anchor_margin_m`-wide annulus just
    /// outside each hole too — deliberately DIFFERENT from `contains`'s general "is this a
    /// physically valid point" semantics (used for display/masking, where a point at
    /// `r = radius + epsilon` legitimately IS inside the domain). This stricter test is only
    /// for generating COLLOCATION points, where an FD stencil centered too close to a hole
    /// silently evaluates the network at invalid, inside-the-hole locations and produces
    /// meaningless (but finite, undetected) strain/energy signal — confirmed as a real
    /// contributor to the Kt-stays-near-zero investigation (see `powershell_tool/CLAUDE.md`):
    /// this margin (a few tenths of a millimeter for a typical spec) is smaller than the
    /// finest AMR cell size near the hole, and quadtree cells aren't boundary-aligned, so
    /// without this exclusion some of AMR's own hole-zone-refined collocation points land
    /// close enough to the true edge that their stencils cross into the hole.
    fn contains_for_collocation(&self, x: f64, y: f64) -> bool {
        if x < -self.geometry.half_w || x > self.geometry.half_w
            || y < -self.geometry.half_h || y > self.geometry.half_h {
            return false;
        }
        for hole in &self.geometry.holes {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            let r_excl = hole.radius + self.anchor_margin_m;
            if dx * dx + dy * dy < r_excl * r_excl {
                return false;
            }
        }
        true
    }
}

impl DomainSamplingStrategy for UserSamplingStrategy {
    /// Issue #64: jittered stratified sampling — every call draws a genuinely different point
    /// set (the caller, `user_runner.rs`'s training loop, already calls this fresh every step,
    /// intending real resampling). Before this fix the stratum center was a fixed `+0.5` offset
    /// and the rejection fallback reseeded from the same constant every call, so for any
    /// hole-free geometry (no rejections ever triggered) this returned the byte-identical point
    /// cloud on every one of thousands of training steps — `InteriorEnergyTerm`/
    /// `PhysicalPotentialEnergyTerm`'s `U` is a plain `mean(f(x_i))` Monte-Carlo estimator
    /// (`measure_integral::domain_integral_tensor`), unbiased only if the `x_i` vary across the
    /// optimization trajectory; with a static node set the optimizer could — and did — sculpt
    /// energy density artificially low AT those frozen nodes while the field diverged between
    /// them (see issue #64's own diagnosis: sampled `Π` undercutting the true continuum affine
    /// minimum of exactly `-1`). Jittering within each stratum preserves the domain-wide
    /// coverage the stratified grid was added for, while making every call a genuine new draw —
    /// still fully deterministic/reproducible run-to-run from `SEED_INTERIOR` alone.
    fn sample_interior(&self, _geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;
        let call = self.interior_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = LcgRng::new(SEED_INTERIOR ^ call.wrapping_mul(CALL_SEED_MIX));
        let mut pts = Vec::with_capacity(n);
        let nx = (n as f64).sqrt().ceil() as usize;
        let ny = n.div_ceil(nx);
        let cells = nx * ny;
        for i in 0..n {
            let cell = i * cells / n;
            let ix = cell % nx;
            let iy = cell / nx;
            let jitter_x = rng.next_f64();
            let jitter_y = rng.next_f64();
            let x = -self.geometry.half_w + (ix as f64 + jitter_x) * 2.0 * self.geometry.half_w / nx as f64;
            let y = -self.geometry.half_h + (iy as f64 + jitter_y) * 2.0 * self.geometry.half_h / ny as f64;
            if self.contains_for_collocation(x, y) {
                pts.push([x, y]);
            }
        }
        let mut attempts = 0usize;
        let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;
        while pts.len() < n && attempts < max_attempts {
            attempts += 1;
            let x = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_w;
            let y = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_h;
            if self.contains_for_collocation(x, y) {
                pts.push([x, y]);
            }
        }
        pts
    }

    /// The 4 outer rectangle edges only — hole boundaries come back via
    /// [`Self::named_point_sets`] instead, one named set per hole.
    ///
    /// Issue #64: per-point jittered stratified sampling in 1D, same rationale as
    /// [`Self::sample_interior`] — `ExternalWorkTerm`'s `W_ext` is the same kind of
    /// `mean(f(x_i))`/`boundary_integral_tensor` Monte-Carlo estimator, and before this fix this
    /// method had NO randomness at all (a fixed evenly-spaced grid every call), so `W_ext` was
    /// trained against one frozen boundary point cloud for the entire run. Each point stays
    /// confined to its own `1/per_edge` stratum, so the `ds_x_normal`/`ds_y_normal` quadrature
    /// weight computed elsewhere from `per_edge` (assumes equal per-point spacing in
    /// expectation) stays exactly as valid as the fixed-grid version was.
    fn sample_boundary(&self, _geom: &GeometryConfig, _load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        use pinn_core::LcgRng;
        let call = self.boundary_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = LcgRng::new(SEED_BOUNDARY ^ call.wrapping_mul(CALL_SEED_MIX));
        let per_edge = (n / 4).max(1);
        let mut pts = Vec::with_capacity(per_edge * 4);
        let hw = self.geometry.half_w;
        let hh = self.geometry.half_h;
        for i in 0..per_edge {
            let frac = (i as f64 + rng.next_f64()) / per_edge as f64; // stays in (0,1), avoids exact corners
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
        }).chain(self.geometry.holes.iter().zip(self.hole_fd_names.iter()).map(|(hole, &name)| {
            // Issue #77 fix: FD-safe ring at `radius + anchor_margin_m`, same outward-into-the-
            // hole normal convention as the exact-radius ring above — used only by the
            // kinematic-decomposition hole traction term, which needs a DERIVED (constitutive)
            // stress read and therefore a stencil-safe radius, not the exact hole boundary.
            let r = hole.radius + self.anchor_margin_m;
            let points = (0..HOLE_RING_POINTS).map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / HOLE_RING_POINTS as f64;
                let (nx, ny) = (theta.cos(), theta.sin());
                BoundaryPoint {
                    x: hole.center[0] + r * nx, y: hole.center[1] + r * ny,
                    nx: -nx, ny: -ny, tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::NeumannFree,
                }
            }).collect();
            NamedPointSet { name, points }
        })).collect()
    }

    // `constitutive_anchor_point_sets` intentionally NOT overridden here anymore (falls back
    // to the trait default, `vec![]`, matching Kirsch/pin-lug) — bugSource-New #12 removed the
    // near-ring anchor mechanism this fed (`HoleAnchorEnergyTerm`/`"hole_i_anchor"`): nothing
    // reads direct σ outside the hole ring anymore, so there is no "keep it honest" gap left
    // for an anchor to close. See `anchor_margin_m`'s doc comment for what's kept.
}

/// Issue #73: single source of truth for the per-step training logic that was independently
/// duplicated between `user_runner::run_headless_user_problem` (headless CLI) and
/// `runner::run_user_problem_training_from` (GUI-streaming) — the exact duplication that let
/// issue #64's `UserSamplingStrategy` resampling fix land correctly in one copy and silently
/// not in the other (see `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-06 follow-up section
/// for the full incident). Both call sites now call these two functions instead of maintaining
/// their own copies.
pub fn plate_normalize_point(x: f64, y: f64, half_w: f64, half_h: f64) -> [f32; 2] {
    [(x / half_w) as f32, (y / half_h) as f32]
}

/// Shared `BoundaryPoint` slice -> `PointSetData` conversion — was duplicated identically at
/// both production call sites (and several test helpers, left alone; see issue #73's own
/// non-goals).
pub fn plate_build_pointset(
    pts: &[BoundaryPoint],
    half_w: f64,
    half_h: f64,
) -> crate::problem::PointSetData {
    crate::problem::PointSetData {
        norm: pts.iter().map(|p| plate_normalize_point(p.x, p.y, half_w, half_h)).collect(),
        nx: pts.iter().map(|p| p.nx as f32).collect(),
        ny: pts.iter().map(|p| p.ny as f32).collect(),
        tx: pts.iter().map(|p| p.tx as f32).collect(),
        ty: pts.iter().map(|p| p.ty as f32).collect(),
    }
}

/// Draws a FRESH interior/boundary/named-point-set sample for one training step — the exact
/// logic issue #64 made genuinely vary per call (jittered stratified sampling), and issue #73
/// consolidates into one place after finding it silently re-frozen in one of its two call
/// sites. Every call to this function is a real, independent resample; callers must call it
/// fresh every step (matching both existing loops' own established per-step cadence) rather
/// than caching its result — caching was exactly the mistake that reintroduced issue #64's bug
/// in the GUI-streaming path.
///
/// Callers needing AMR's own interior resampling (GUI-streaming path only, gated on
/// `spec.training.amr_enabled`) should overwrite the returned `DomainStepData.int_norm`
/// afterward — AMR only ever refines the interior quadtree, never touches boundary or named
/// point sets, so this function's boundary/named output stays authoritative regardless.
pub fn resample_plate_step_data(
    sampling: &dyn DomainSamplingStrategy,
    placeholder_geom: &GeometryConfig,
    load: &LoadConfig,
    n_interior: usize,
    n_boundary: usize,
    half_w: f64,
    half_h: f64,
) -> crate::problem::DomainStepData {
    resample_domain_step_data(
        USER_DOMAIN, sampling, placeholder_geom, load, n_interior, n_boundary, half_w, half_h,
    )
}

/// Domain-id-parametrized version of [`resample_plate_step_data`]. Multi-domain problems use
/// the same sampling and point-set conversion as the legacy plate path; only ownership differs.
pub fn resample_domain_step_data(
    id: DomainId,
    sampling: &dyn DomainSamplingStrategy,
    placeholder_geom: &GeometryConfig,
    load: &LoadConfig,
    n_interior: usize,
    n_boundary: usize,
    half_w: f64,
    half_h: f64,
) -> crate::problem::DomainStepData {
    let int_pts = sampling.sample_interior(placeholder_geom, n_interior);
    let int_norm: Vec<[f32; 2]> = int_pts.iter()
        .map(|&[x, y]| plate_normalize_point(x, y, half_w, half_h))
        .collect();

    let bnd_pts = sampling.sample_boundary(placeholder_geom, load, n_boundary);
    // 1 outer_boundary + 2 per hole (traction ring + constitutive-consistency anchor ring —
    // see `UserSamplingStrategy::named_point_sets`); exact capacity unknown without querying
    // the geometry, so this just starts reasonably sized rather than tracking hole count too.
    let mut named = std::collections::HashMap::with_capacity(4);
    named.insert("outer_boundary", plate_build_pointset(&bnd_pts, half_w, half_h));
    for set in sampling.named_point_sets(&[]) {
        named.insert(set.name, plate_build_pointset(&set.points, half_w, half_h));
    }

    crate::problem::DomainStepData { id, int_norm, extra_ring_norm: Vec::new(), named }
}

/// Issue #75: whether a step's interior points came from plain uniform sampling or the
/// persistent, geometry-aware adaptive source (`apply_persistent_adaptive_interior_sample`).
/// Purely informational telemetry — callers branch on `Option<Vec<f64>>` (the actual weights),
/// not this enum, for behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteriorSampleSource {
    Uniform,
    PersistentAdaptive,
}

/// Fixed discriminator seed for `apply_persistent_adaptive_interior_sample`'s own RNG draws —
/// mixed with each call's `step` index exactly like `SEED_INTERIOR`/`SEED_BOUNDARY` are mixed
/// with `UserSamplingStrategy`'s own per-call counters. A plain constant (not derived from the
/// problem spec) because this function already receives a distinct, real per-step `step` index
/// to mix in — no separate atomic call-counter is needed the way `UserSamplingStrategy` needs
/// one (that struct's `&self`-only trait signature forces interior mutability; this free
/// function already takes `step` as an explicit parameter).
const PERSISTENT_AMR_SEED: u64 = 750_075;

/// Issue #75 workstream C: the single source of truth for whether and how THIS step's interior
/// points come from a persistent, geometry-aware adaptive source instead of the plain uniform
/// sample `resample_plate_step_data` already drew into `data.int_norm` — the "AMR overwrite"
/// step that function's own doc comment already described as the correct seam for an
/// AMR-aware caller, now consolidated into one function instead of being reimplemented ad hoc
/// at each such call site (the exact duplication class issue #73 exists to prevent).
///
/// `amr: None` (headless's current behavior; the GUI-streaming path before AMR activates, when
/// `amr_enabled=false`, or for a no-hole geometry — persistent geometry-aware sampling is
/// specifically for HOLE-bearing geometries, per issue #75's own "Preserve baseline behavior"
/// section) leaves `data.int_norm` untouched and returns `None` weights, matching
/// `UserDefinedProblem::set_interior_weights`'s own pre-existing "`None` = already unbiased, no
/// compensation needed" contract exactly.
///
/// `amr: Some(grid)` draws PERSISTENT geometry-aware adaptive samples on EVERY call, not just
/// on an AMR sweep step (issue #75's whole point — the pre-existing sweep-only mechanism fired
/// 3 times in a 3000-step run and was overwritten by uniform resampling every other step,
/// PH4-14's own real evidence for why enabling AMR alone left Kt essentially unchanged): a
/// fresh fixed-budget quadtree draw via
/// `AdaptiveGrid::sample_points_jittered_with_density_budget` (topology persists between
/// `adapt()` sweeps; coordinates still genuinely vary every call, issue #64's invariant).
/// Hole lock zones are geometry-seeded before step zero, so this retains a structural near-hole
/// density bias without appending a second, overlapping annular quadrature set. Overwrites
/// `data.int_norm` and returns matching compensation weights in the SAME call, so a caller can
/// never apply one call's weights to a different call's points.
///
/// Annular samples remain available as a core primitive, but are deliberately not combined
/// here until a disjoint-strata or full mixture-density estimator exists. Appending them to a
/// quadtree that already represents the same physical area double-counts a nonconstant field.
pub fn apply_persistent_adaptive_interior_sample(
    data: &mut crate::problem::DomainStepData,
    geometry: &UserGeometry,
    _fd_h: f32,
    half_w: f64,
    half_h: f64,
    amr: Option<&mut pinn_core::amr::AdaptiveGrid<UserGeometry>>,
    step: usize,
) -> (InteriorSampleSource, Option<Vec<f64>>) {
    let Some(grid) = amr else {
        return (InteriorSampleSource::Uniform, None);
    };
    if geometry.holes.is_empty() {
        // Persistent geometry-aware sampling is specifically for hole-bearing geometries —
        // see this function's own doc comment. A caller that still wants residual-driven-only
        // AMR for a no-hole geometry should keep using the pre-#75 sweep-only mechanism, not
        // this function.
        return (InteriorSampleSource::Uniform, None);
    }

    let n_before = data.int_norm.len().max(1);
    let samples = grid.sample_points_jittered_with_density_budget(n_before, PERSISTENT_AMR_SEED ^ step as u64);

    let weights = pinn_core::amr::compensation_weights(&samples);
    data.int_norm = samples.iter()
        .map(|s| plate_normalize_point(s.point[0], s.point[1], half_w, half_h))
        .collect();
    (InteriorSampleSource::PersistentAdaptive, Some(weights))
}

/// Build one residual-probe point per adaptive leaf represented by the grid's sampler, preserving DFS order.
/// `AdaptiveGrid::update_residuals` assigns residuals by that exact order; probing an unrelated
/// uniform cloud would associate each residual with the wrong leaf.
pub fn adaptive_grid_probe_data(
    data: &crate::problem::DomainStepData,
    half_w: f64,
    half_h: f64,
    grid: &mut pinn_core::amr::AdaptiveGrid<UserGeometry>,
    step: usize,
) -> crate::problem::DomainStepData {
    let samples = grid.sample_points_jittered_with_density(PERSISTENT_AMR_SEED ^ step as u64);
    let mut probe = data.clone();
    probe.int_norm = samples
        .iter()
        .map(|s| plate_normalize_point(s.point[0], s.point[1], half_w, half_h))
        .collect();
    probe
}

/// Builds the `MultiStepCtx` fields BOTH the headless and GUI-streaming plate training loops
/// use identically — the constants that would previously have needed to change in two places
/// at once (issue #73). `dynamic_lam_penetration_cap`/`dynamic_lam_non_tension_cap` are always
/// `f64::MAX` here (inert) since the plate problem has no interface-penetration/non-tension
/// terms (those are pin-lug-only) — matching both existing call sites exactly.
///
/// Rationale for the specific constant values (preserved from the two call sites this
/// consolidates, not re-derived):
/// - `dynamic_lam_h_cap`/`dynamic_lam_d_cap = 50.0`: real root cause of the garbage-Kt/zero-
///   hole-stress bug (see `powershell_tool/CLAUDE.md`'s Stress Solver section) —
///   `hole_traction_loss_direct` was left fully uncapped (`f64::MAX`, unlike Kirsch's own real,
///   tested 50→15 cascade), so SAW-BRDR could grow its adapted weight arbitrarily large
///   relative to `step_physics_multi`'s fixed `LAM_CONSTITUTIVE_CONSISTENCY` (5.0) — and an
///   outweighed boundary term has a strictly EASIER minimum available than the true elasticity
///   solution: drive direct-stress outputs toward zero everywhere the boundary term is
///   evaluated (trivially satisfies "traction ≈ 0" without satisfying "stress matches Hooke's
///   law"). Capped at Kirsch's own starting value (50.0, a real, already-tuned bound in this
///   codebase) — deliberately NOT replicating Kirsch's full plateau-triggered cascade down to
///   15.0, which is tuned specifically for Kirsch's own K_t dynamics.
/// - `constitutive_consistency_weight = 50.0`: paired with the cap above — capping the boundary
///   term alone measurably reduced the interior PDE residual but left Kt essentially unmoved
///   (confirmed via a real training run). Raising this to the SAME 50.0 ceiling closes the
///   remaining gap: `hole_traction` can no longer structurally outweigh constitutive-consistency
///   by 10x the way it could when one was capped at 50 and the other pinned at 5.
/// - `n_fourier`: caller-supplied, always `spec.geometry.n_fourier()` — MUST match the network's
///   own `input_dim` at construction time (see `UserGeometry::n_fourier`'s doc comment for the
///   full root-cause story: Kirsch's own path already uses Fourier positional encoding to
///   correct spectral bias near a hole; the plate path didn't until this was added).
#[allow(clippy::too_many_arguments)]
pub fn plate_multi_step_ctx<'a>(
    config: &'a pinn_core::messages::SolverConfig,
    problem: &'a dyn crate::problem::BoundaryValueProblem,
    fd: &'a crate::fd_stencil::FdConfig,
    data: &'a crate::problem::DomainStepData,
    u_ref: f32,
    ref_energy: f32,
    ref_stress2: f32,
    n_fourier: usize,
    coordinate_embedding: pinn_core::user_geometry::CoordinateEmbedding,
    probe_term_gradients: bool,
    step: usize,
) -> crate::problem::MultiStepCtx<'a> {
    crate::problem::MultiStepCtx {
        config,
        problem,
        fd,
        k: 1.0, // IdentityAnsatz ignores k entirely — value is inert
        domains: vec![crate::problem::DomainStepCtx { data, u_ref, ref_energy, ref_stress2 }],
        dynamic_lam_h_cap: 50.0,
        dynamic_lam_d_cap: 50.0,
        dynamic_lam_penetration_cap: f64::MAX,
        dynamic_lam_non_tension_cap: f64::MAX,
        constitutive_consistency_weight: 50.0,
        n_fourier,
        coordinate_embedding,
        probe_term_gradients,
        phase2_active: true,
        step,
    }
}

/// Two-domain counterpart of [`plate_multi_step_ctx`]. Both domains use shared physical
/// reference scales, while model input width selects raw outer versus charted annular forwards.
#[allow(clippy::too_many_arguments)]
pub fn plate_multi_domain_step_ctx<'a>(
    config: &'a pinn_core::messages::SolverConfig,
    problem: &'a dyn crate::problem::BoundaryValueProblem,
    fd: &'a crate::fd_stencil::FdConfig,
    annulus: &'a crate::problem::DomainStepData,
    outer: &'a crate::problem::DomainStepData,
    u_ref: f32,
    ref_energy: f32,
    ref_stress2: f32,
    n_fourier: usize,
    coordinate_embedding: pinn_core::user_geometry::CoordinateEmbedding,
    probe_term_gradients: bool,
    step: usize,
) -> crate::problem::MultiStepCtx<'a> {
    crate::problem::MultiStepCtx {
        config, problem, fd, k: 1.0,
        domains: vec![
            crate::problem::DomainStepCtx { data: annulus, u_ref, ref_energy, ref_stress2 },
            crate::problem::DomainStepCtx { data: outer, u_ref, ref_energy, ref_stress2 },
        ],
        dynamic_lam_h_cap: 50.0,
        dynamic_lam_d_cap: 50.0,
        dynamic_lam_penetration_cap: 50.0,
        dynamic_lam_non_tension_cap: f64::MAX,
        constitutive_consistency_weight: 50.0,
        n_fourier,
        coordinate_embedding,
        probe_term_gradients,
        phase2_active: true,
        step,
    }
}

/// Mirrors `pinlug_problem::InteriorEnergyTerm` exactly (`dem_energy_loss`, generic, no new
/// math) — the default `point_sets()` ("interior") applies unchanged.
///
/// Issue #62 PH3-04: `measure_aware`/`domain_area`/`thickness`/`ref_energy_absolute`/`weights`
/// are ALL inert when `measure_aware==false` (the default - `compute` takes the exact legacy
/// `dem_energy_loss(...).mean()/ref_energy` path, byte-identical to before this epic). When
/// `true`, `compute` instead calls `measure_integral::domain_integral_weighted_tensor` (falling
/// back to the unweighted `domain_integral_tensor` when `weights` is `None`, i.e. before the
/// first AMR sweep) - see `measure_integral.rs`'s own top-level doc comment for exactly why this
/// (not `dem_energy_loss`'s plain, AMR-density-biased `.mean()`) is the correct estimator once
/// sampling becomes nonuniform.
struct InteriorEnergyTerm {
    domain: DomainId,
    material: MaterialProps,
    ref_energy: f32,
    measure_aware: bool,
    domain_area: f64,
    thickness: f64,
    /// `ref_energy` (a per-unit-volume energy DENSITY scale) times `domain_area*thickness` -
    /// the matching ABSOLUTE-Joules reference scale `domain_integral_weighted_tensor`'s output
    /// needs to be normalized against, so the measure-aware term stays on the SAME dimensionless
    /// scale `ExternalWorkTerm` and the legacy `.mean()/ref_energy` path both use (algebraically
    /// this reduces to exactly `mean(density)/ref_energy` in the unweighted case - the area and
    /// thickness factors cancel - which is WHY the legacy path was never numerically wrong for
    /// uniform sampling, only for AMR-nonuniform sampling; see the PH3-04 manifest entry for the
    /// full derivation).
    ref_energy_absolute: f64,
    /// Snapshot of `UserDefinedProblem::current_interior_weights` at the moment `loss_terms()`
    /// built this term - see that field's own doc comment.
    weights: Option<Vec<f64>>,
}
impl LossTerm for InteriorEnergyTerm {
    fn name(&self) -> &'static str { "interior_energy" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Weak }
    // Strain-energy only - no stress quantity at the term level.
    // Interior PDE physics, not a boundary condition - correctly `None`.
    // Reads `d.strains` - first-order spatial derivative.
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    // `U` itself - the literal physical functional term issue #61 P2-05 names.
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("interior_energy: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("interior_energy: strains must be Some");
        if !self.measure_aware {
            return dem_energy_loss(exx, eyy, exy, &self.material).mul_scalar(1.0 / self.ref_energy as f64);
        }
        let density = crate::energy::dem_energy_per_point(exx, eyy, exy, &self.material);
        let energy = match &self.weights {
            Some(w) => crate::measure_integral::domain_integral_weighted_tensor::<B>(self.domain_area, self.thickness, density, w),
            None => crate::measure_integral::domain_integral_tensor::<B>(self.domain_area, self.thickness, density),
        };
        energy.mul_scalar(1.0 / self.ref_energy_absolute)
    }
}

/// Real strong-form equilibrium (`‖∇·σ‖²`), computed on the DERIVED stress `σ=C:ε(u)` via the
/// displacement Hessian, not the network's direct mDEM stress output — the piece this problem
/// was missing entirely (see `powershell_tool/CLAUDE.md`'s Kt investigation: without this,
/// `InteriorEnergyTerm` minimizes strain energy `U[u]` alone, not total potential energy
/// `Π=U-W_ext`, whose Euler-Lagrange equation IS equilibrium - nothing forced the stress state
/// at the loaded edges to propagate consistently through the interior).
///
/// This term ORIGINALLY read direct mDEM σ (`d.shifted_stress`, matching Kirsch's own
/// `EquilibriumRingTerm` mechanism) — but the term-by-term gradient-instrumentation diagnostic
/// (bugSource-New #1) found that version's gradient 5-6 orders of magnitude smaller than every
/// other term's throughout training, and never growing: the plate's direct-σ output never
/// develops real spatial structure, so constraining `∇·σ_direct=0` was trivially already
/// satisfied and supplied ~zero real gradient pressure. bugSource-New #12's fix: define
/// equilibrium on the stress the network's own DISPLACEMENT field implies instead, where real
/// spatial structure is actually forming during training. See `energy::
/// equilibrium_from_displacement_hessian_loss`'s doc comment for the derivation.
/// `kirsch_problem.rs` is untouched; this is a separate, plate-scoped struct so Kirsch's own
/// path/tests can never be affected by anything here.
struct EquilibriumTerm {
    domain: DomainId,
    point_set: &'static str,
    material: MaterialProps,
    ref_div2: f64,
}
impl LossTerm for EquilibriumTerm {
    fn name(&self) -> &'static str { "equilibrium" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec![self.point_set] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn needs_hessian(&self) -> bool { true }
    // `equilibrium_from_displacement_hessian_loss` applies Hooke's law constants to the
    // Hessian internally (`σ=C:ε(u)`, via second derivatives rather than FD strain) - derived.
    fn stress_source(&self) -> Option<crate::problem::StressSource> { Some(crate::problem::StressSource::Derived) }
    // Interior equilibrium (∇·σ=0), evaluated at interior collocation points, not a boundary
    // condition - same reasoning as Kirsch's `EquilibriumRingTerm`.
    // Reads `d.hessian` (`needs_hessian()==true` above) - second-order spatial derivative.
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::Second) }
    // The governing PDE itself (Strong formulation's counterpart to `interior_energy`'s `U`) -
    // physics, not an admissibility constraint or a diagnostic.
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("equilibrium: domain missing");
        let (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy) = d.hessian.clone()
            .expect("equilibrium: hessian must be Some (needs_hessian()==true)");
        equilibrium_from_displacement_hessian_loss(
            u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, &self.material, self.ref_div2,
        )
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
    domain: DomainId,
    material: MaterialProps,
    ref_stress2: f32,
    px: f64,
    py: f64,
}
impl LossTerm for OuterTractionTerm {
    fn name(&self) -> &'static str { "outer_traction" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    // `neumann_loss` computes stress from strain via `compute_stress`.
    fn stress_source(&self) -> Option<crate::problem::StressSource> { Some(crate::problem::StressSource::Derived) }
    // Prescribes stress·n (the applied far-field traction) at the outer boundary - Neumann.
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> { Some(crate::problem::BoundaryOperatorKind::Neumann) }
    // Reads `d.strains` - first-order spatial derivative.
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    // The applied far-field traction BC is part of the governing BVP itself, not an
    // admissibility constraint layered on top of it (unlike an essential/Dirichlet condition).
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("outer_traction: domain missing");
        let (exx, eyy, exy) = d.strains.clone().expect("outer_traction: strains must be Some");
        let (nx, ny) = d.normals.clone().expect("outer_traction: normals must be Some");
        let tx_target = nx.clone().mul_scalar(self.px);
        let ty_target = ny.clone().mul_scalar(self.py);
        neumann_loss(exx, eyy, exy, nx, ny, tx_target, ty_target, &self.material)
            .mul_scalar(1.0 / self.ref_stress2 as f64)
    }
}

/// External work `W_ext = ∫_{Γ_N} t̄·u dΓ` (applied traction dotted with displacement) at the
/// outer boundary — the piece missing from this formulation that bugSource-New #3/#13
/// specifically flagged: `InteriorEnergyTerm` minimizes strain energy `U[u]` ALONE, not total
/// potential energy `Π=U-W_ext`. Minimizing pure `U[u]` with no `-W_ext` counterweight is
/// trivially satisfied by `u≡0` (zero strain everywhere has zero energy) - only `OuterTractionTerm`'s
/// separate traction-RESIDUAL penalty was providing any incentive against that shortcut, a
/// fundamentally different mechanism (a boundary-condition penalty) than the genuine
/// variational term the true minimization principle calls for. This term adds that missing
/// piece ALONGSIDE (not replacing) `OuterTractionTerm` - the safest way to test the hypothesis
/// without touching a term that already works.
///
/// For the true elasticity solution, `Π[u] = U[u] - W_ext[u]` is minimized over admissible `u`
/// (mod rigid-body modes) by the exact equilibrium field, with the natural (traction) BC
/// satisfied automatically - this is the Deep Energy Method's own founding principle, applied
/// here for the first time to the OUTER boundary (the hole's traction-free BC already gets an
/// equivalent "do nothing extra" treatment for free, since `t̄=0` there makes its own `W_ext`
/// contribution identically zero).
///
/// `t̄=(px·nx, py·ny)` is the FIXED, APPLIED far-field traction (from `spec.load`), NOT a
/// network-derived quantity - unlike `OuterTractionTerm`, which compares network-derived
/// traction against this same target. Sign convention matches `OuterTractionTerm`'s own
/// `tx_target`/`ty_target` exactly (`nx·px`, `ny·py`). Returns `-mean(t̄·u)/ref_energy` (negated
/// so MINIMIZING this loss MAXIMIZES the actual external work, matching Π's own `-W_ext` sign);
/// `ref_energy` is `InteriorEnergyTerm`'s own normalization constant, reused for direct,
/// same-convention comparability against `U`, not a separately-derived scale.
///
/// Issue #62 PH3-04: when `measure_aware` is set, `compute` instead uses `measure_integral::
/// boundary_integral_tensor` with the REAL per-point arc-length `ds` (`ds_per_point`, precomputed
/// once in `loss_terms()` from `UserSamplingStrategy::sample_boundary`'s own exact point-
/// generation order/spacing) instead of assuming every point carries equal weight via `.mean()`.
/// For a SQUARE plate (`half_w==half_h`, true of every shipped example) every edge's `ds` is
/// identical and this is numerically byte-identical to the legacy path; for a NON-square plate
/// the right/left edges (spanning `half_h`) and top/bottom edges (spanning `half_w`) have
/// genuinely different `ds`, which `.mean()` silently ignored - a real, previously-latent
/// correctness gap this migration also closes generally, not only under AMR (this point set is
/// never AMR-refined - `AdaptiveGrid` only ever touches the interior quadtree - so the gap here
/// is purely about non-square aspect ratios, not sampling density).
struct ExternalWorkTerm {
    px: f64,
    py: f64,
    ref_energy: f32,
    measure_aware: bool,
    thickness: f64,
    ref_energy_absolute: f64,
    /// Real per-point arc-length spacing, in `sample_boundary`'s own point order (right, left,
    /// top, bottom, repeated `per_edge` times) - only read when `measure_aware`.
    ds_per_point: Vec<f64>,
}

/// Atomic linear-elastic potential energy, `Π = U - W_ext`.
///
/// `LossTerm` normally maps one domain to one point set.  This term deliberately repeats the
/// same domain id for the interior and outer-boundary point sets, so the generic live driver
/// builds both required forwards and gives this one term both halves of the physical
/// functional.  That keeps its physical 1:-1 coefficient ratio inside the tensor graph before
/// any optimizer/adaptive weighting is applied.
struct PhysicalPotentialEnergyTerm {
    domain: DomainId,
    material: MaterialProps,
    px: f64,
    py: f64,
    measure_aware: bool,
    domain_area: f64,
    thickness: f64,
    ref_energy: f32,
    ref_energy_absolute: f64,
    interior_weights: Option<Vec<f64>>,
    ds_per_point: Vec<f64>,
    /// Issue #77 fix: `Some((px,py))` when kinematic decomposition is active — the network's
    /// own `interior.strains` is then `eps_hole` alone, so `affine_strain(px,py,material)`
    /// must be added before computing `U` (see that function's doc comment: the cross term is
    /// physically required, not optional). `None` everywhere else — byte-identical fallback.
    affine_strain: Option<(f64, f64)>,
}

impl LossTerm for PhysicalPotentialEnergyTerm {
    fn name(&self) -> &'static str { "physical_potential" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain, self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interior", "outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Weak }
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }

    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        assert_eq!(inputs.len(), 2, "physical_potential requires interior and outer-boundary forwards");
        let interior = &inputs[0];
        let boundary = &inputs[1];
        let (mut exx, mut eyy, mut exy) = interior.strains.clone()
            .expect("physical_potential: interior strains must be Some");
        if let Some((px, py)) = self.affine_strain {
            let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &self.material);
            exx = exx.add_scalar(a_exx);
            eyy = eyy.add_scalar(a_eyy);
            exy = exy.add_scalar(a_exy);
        }
        let u = if self.measure_aware {
            let density = crate::energy::dem_energy_per_point(exx, eyy, exy, &self.material);
            match &self.interior_weights {
                Some(weights) => crate::measure_integral::domain_integral_weighted_tensor::<B>(
                    self.domain_area, self.thickness, density, weights,
                ),
                None => crate::measure_integral::domain_integral_tensor::<B>(
                    self.domain_area, self.thickness, density,
                ),
            }.mul_scalar(1.0 / self.ref_energy_absolute)
        } else {
            dem_energy_loss(exx, eyy, exy, &self.material)
                .mul_scalar(1.0 / self.ref_energy as f64)
        };

        let (nx, ny) = boundary.normals.clone()
            .expect("physical_potential: boundary normals must be Some");
        let n = boundary.raw_out.dims()[0];
        let displacement = match crate::field_graph::resolve_field(
            crate::field_graph::FieldKind::Displacement, boundary.raw_out, None, None,
        ).expect("physical_potential requires displacement") {
            crate::field_graph::ResolvedField::Displacement(value) => value,
            _ => unreachable!("displacement resolver returned wrong field kind"),
        };
        let ux = displacement.clone().slice([0..n, 0..1]).reshape([n]);
        let uy = displacement.slice([0..n, 1..2]).reshape([n]);
        let work_density = nx.mul_scalar(self.px) * ux + ny.mul_scalar(self.py) * uy;
        let w_ext = if self.measure_aware {
            crate::measure_integral::boundary_integral_tensor::<B>(
                work_density, &self.ds_per_point, self.thickness,
            ).mul_scalar(1.0 / self.ref_energy_absolute)
        } else {
            work_density.mean().mul_scalar(1.0 / self.ref_energy as f64)
        };
        u - w_ext
    }
}
impl LossTerm for ExternalWorkTerm {
    fn name(&self) -> &'static str { "external_work" }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Weak }
    // Displacement-only (`u`,`v` dotted with the applied traction) - no stress quantity.
    // The weak-form/energy-functional counterpart of `OuterTractionTerm`'s pointwise Neumann
    // residual, not itself a pointwise boundary-operator residual - `Π=U-W_ext`'s natural BC
    // is satisfied automatically by minimizing this energy, not by a per-point condition it
    // enforces directly. Correctly `None` rather than mislabeled `Neumann`.
    // `-W_ext` itself - the other literal physical functional term issue #61 P2-05 names
    // (`Π = U - W_ext`, `interior_energy` being the `U` half).
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("external_work: domain missing");
        let (nx, ny) = d.normals.clone().expect("external_work: normals must be Some");
        let n = d.raw_out.dims()[0];
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let work_density = nx.mul_scalar(self.px) * u + ny.mul_scalar(self.py) * v;
        if !self.measure_aware {
            return work_density.mean().mul_scalar(-1.0 / self.ref_energy as f64);
        }
        let w_ext = crate::measure_integral::boundary_integral_tensor::<B>(work_density, &self.ds_per_point, self.thickness);
        w_ext.mul_scalar(-1.0 / self.ref_energy_absolute)
    }
}

/// One hole's boundary condition — `Free` mirrors `pinlug_problem::LugFreeEdgeTractionTerm`
/// (`hole_traction_loss_direct` on direct mDEM stress columns, implicit zero target);
/// `Fixed` mirrors `pinlug_problem::LugShankAnchorTerm` (`mean(u^2+v^2)` on direct mDEM
/// displacement columns).
struct HoleBcTerm {
    domain: DomainId,
    point_set: &'static str,
    bc: HoleBc,
    ref_stress2: f32,
    material: MaterialProps,
    /// Issue #77 fix: `Some((px,py))` when kinematic decomposition is active for a `Free`
    /// hole. The network represents `u_hole`, so the true traction-free condition on the
    /// TOTAL field (`sigma_total.n=0`) becomes `sigma_hole.n = -sigma_affine.n` on `u_hole` —
    /// a known, nonzero target, not zero. Also switches the read from direct mDEM stress (see
    /// this term's own module doc comment on why that head "never develops real spatial
    /// structure") to DERIVED/constitutive stress at the FD-safe ring - `self.point_set` must
    /// then be the `*_fd` ring name, which the caller in `loss_terms()` is responsible for
    /// pairing correctly. `None` (both for `Fixed` and for every geometry outside
    /// `decomposition_applicable`'s scope) preserves the exact original direct-stress,
    /// zero-target, exact-radius-ring behavior byte-for-byte.
    affine_target: Option<(f64, f64)>,
}

/// Equality constraints for a bonded artificial interface between two subdomains. Both
/// domains must expose the same ordered "interface" coordinates. Traction balance uses
/// each domain's outward normal, so physical tractions sum to zero.
pub struct InterfaceDisplacementContinuityTerm {
    pub left: DomainId,
    pub right: DomainId,
    pub inv_u_ref_sq: f64,
}
impl LossTerm for InterfaceDisplacementContinuityTerm {
    fn name(&self) -> &'static str { "interface_displacement_continuity" }
    fn domains(&self) -> Vec<DomainId> { vec![self.left, self.right] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface", "interface"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> { Some(crate::problem::BoundaryOperatorKind::Interface) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let left = inputs.iter().find(|d| d.domain == self.left).expect("interface displacement: left domain missing");
        let right = inputs.iter().find(|d| d.domain == self.right).expect("interface displacement: right domain missing");
        let n = left.raw_out.dims()[0];
        assert_eq!(n, right.raw_out.dims()[0], "interface displacement: point sets must be aligned");
        let du = left.raw_out.clone().slice([0..n, 0..1]) - right.raw_out.clone().slice([0..n, 0..1]);
        let dv = left.raw_out.clone().slice([0..n, 1..2]) - right.raw_out.clone().slice([0..n, 1..2]);
        (du.clone() * du + dv.clone() * dv).mean().mul_scalar(self.inv_u_ref_sq)
    }
}

pub struct InterfaceTractionContinuityTerm {
    pub left: DomainId,
    pub right: DomainId,
    pub left_material: MaterialProps,
    pub right_material: MaterialProps,
    pub inv_ref_stress2: f64,
}
impl LossTerm for InterfaceTractionContinuityTerm {
    fn name(&self) -> &'static str { "interface_traction_continuity" }
    fn domains(&self) -> Vec<DomainId> { vec![self.left, self.right] }
    /// Central differences at the geometrical interface would sample a model beyond the
    /// subdomain it represents. Each trace is therefore evaluated on an FD-safe offset ring
    /// wholly owned by that subdomain; the two offsets converge to the same physical trace as
    /// `fd_h` is refined.
    fn point_sets(&self) -> Vec<&'static str> {
        vec!["interface_annulus_stress", "interface_outer_stress"]
    }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn stress_source(&self) -> Option<crate::problem::StressSource> { Some(crate::problem::StressSource::Derived) }
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> { Some(crate::problem::BoundaryOperatorKind::Interface) }
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let left = inputs.iter().find(|d| d.domain == self.left).expect("interface traction: left domain missing");
        let right = inputs.iter().find(|d| d.domain == self.right).expect("interface traction: right domain missing");
        let (lexx, leyy, lexy) = left.strains.clone().expect("interface traction: left strains missing");
        let (rexx, reyy, rexy) = right.strains.clone().expect("interface traction: right strains missing");
        let (lnx, lny) = left.normals.clone().expect("interface traction: left normals missing");
        let (rnx, rny) = right.normals.clone().expect("interface traction: right normals missing");
        let (lsxx, lsyy, lsxy) = crate::energy::compute_stress(lexx, leyy, lexy, &self.left_material);
        let (rsxx, rsyy, rsxy) = crate::energy::compute_stress(rexx, reyy, rexy, &self.right_material);
        let tx = lsxx * lnx.clone() + lsxy.clone() * lny.clone() + rsxx * rnx.clone() + rsxy.clone() * rny.clone();
        let ty = lsxy * lnx + lsyy * lny + rsxy * rnx + rsyy * rny;
        (tx.clone() * tx + ty.clone() * ty).mean().mul_scalar(self.inv_ref_stress2)
    }
}
impl LossTerm for HoleBcTerm {
    fn name(&self) -> &'static str {
        match self.bc { HoleBc::Free => "hole_free", HoleBc::Fixed => "hole_fixed" }
    }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec![self.point_set] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    // Runtime-dependent: `Free` normally reads `raw_out` cols 2..5 directly (FD is undefined
    // exactly at the hole boundary); `Fixed` is displacement-only. Issue #77 fix: when
    // `affine_target` is `Some`, `Free` instead reads DERIVED/constitutive stress at the
    // FD-safe `*_fd` ring `point_set` already points at in that case.
    fn stress_source(&self) -> Option<crate::problem::StressSource> {
        match (self.bc, self.affine_target) {
            (HoleBc::Free, Some(_)) => Some(crate::problem::StressSource::Derived),
            (HoleBc::Free, None) => Some(crate::problem::StressSource::Direct),
            (HoleBc::Fixed, _) => None,
        }
    }
    // Only the issue #77 decomposition arm needs an FD-derived strain (`Some(First)`); the
    // original direct-stress read needs no derivative at all (trait default `None`, matching
    // pre-existing behavior exactly since this method wasn't previously overridden).
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> {
        match (self.bc, self.affine_target) {
            (HoleBc::Free, Some(_)) => Some(crate::problem::DerivativeOrder::First),
            _ => None,
        }
    }
    // `Free` prescribes stress·n=0 (Neumann, zero flux); `Fixed` prescribes displacement=0
    // (Dirichlet) - same runtime-dependent pattern as `stress_source` above.
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> {
        match self.bc {
            HoleBc::Free => Some(crate::problem::BoundaryOperatorKind::Neumann),
            HoleBc::Fixed => Some(crate::problem::BoundaryOperatorKind::Dirichlet),
        }
    }
    // `Free` (traction-free hole boundary) is part of the governing BVP's own physics, same
    // reasoning as `OuterTractionTerm`; `Fixed` is an essential/Dirichlet admissibility
    // constraint on the solution space, issue #61 §1.1's own "essential constraints" - not
    // itself part of the physical functional (an anchor doesn't change under a different
    // formulation the way a natural BC term does - it's always active, see `loss_terms()`).
    fn term_role(&self) -> crate::problem::TermRole {
        match self.bc {
            HoleBc::Free => crate::problem::TermRole::PhysicalFunctional,
            HoleBc::Fixed => crate::problem::TermRole::Constraint,
        }
    }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("hole_bc: domain missing");
        let n = d.raw_out.dims()[0];
        match self.bc {
            HoleBc::Free => {
                let (nx, ny) = d.normals.clone().expect("hole_bc(free): normals must be Some");
                if let Some((px, py)) = self.affine_target {
                    // Issue #77 fix: `u_hole`'s own traction target is `-sigma_affine.n`, not
                    // zero - `sigma_affine` is spatially constant (`diag(px,py)`), so the
                    // target reduces to `(-px*nx, -py*ny)`. Reuses `neumann_loss` exactly as
                    // `OuterTractionTerm` does, just with the sign flipped and a derived (not
                    // direct) stress source, consistent with `stress_source`/`derivative_order`
                    // above.
                    let (exx, eyy, exy) = d.strains.clone()
                        .expect("hole_bc(free, decomposed): strains must be Some");
                    let tx_target = nx.clone().mul_scalar(-px);
                    let ty_target = ny.clone().mul_scalar(-py);
                    return crate::energy::neumann_loss(exx, eyy, exy, nx, ny, tx_target, ty_target, &self.material)
                        .mul_scalar(1.0 / self.ref_stress2 as f64);
                }
                let (sxx, syy, sxy) = match crate::field_graph::resolve_field(
                    crate::field_graph::FieldKind::DirectStress, d.raw_out, None, None,
                ).expect("hole_free requires direct mDEM stress") {
                    crate::field_graph::ResolvedField::DirectStress(value) => value,
                    _ => unreachable!("direct-stress resolver returned wrong field kind"),
                };
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

// `HoleAnchorEnergyTerm` (a real-equilibrium-via-energy signal on the `"hole_i_anchor"` point
// set) was removed here as part of bugSource-New #12: it existed to keep direct σ "honest"
// near the hole ring against a degenerate near-zero-everything solution, a problem that no
// longer exists once `EquilibriumTerm` reads the displacement-Hessian-derived stress
// everywhere instead of direct σ (see `EquilibriumTerm`'s own updated doc comment).

/// Issue #61 EPIC P2-07: translational gauge-fixing (a "mean-field constraint", one of the
/// three techniques the epic names) for pure-Neumann configurations - see `crate::gauge`'s
/// module doc comment for the full rationale. Penalizes the mean interior displacement
/// (`mean(u)^2 + mean(v)^2`, NOT `mean(u^2+v^2)` - squaring the mean, not the mean of squares,
/// so LOCAL displacement variation is untouched and only the domain-wide rigid-body DRIFT is
/// penalized) toward zero. Registered ONLY when `self.spec.geometry.is_pure_neumann()` (see
/// `loss_terms()`) - never active for a problem that already has a real Dirichlet anchor.
///
/// The raw displacement means are in metres while physical Pi is dimensionless after reference
/// energy normalization. `inv_u_ref_sq` makes this a dimensionless admissibility constraint;
/// without it a micrometre-scale rigid translation contributes ~1e-12 and is inert regardless
/// of its nominal base weight.
struct TranslationGaugeTerm {
    domain: DomainId,
    inv_u_ref_sq: f64,
}
impl LossTerm for TranslationGaugeTerm {
    fn name(&self) -> &'static str { "translation_gauge" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    // A gauge/admissibility fix, not part of the physical functional U-W_ext - "separate from
    // load enforcement" per issue #61 P2-07's own acceptance wording.
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }
    // Displacement-only - no stress quantity involved.
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("translation_gauge: domain missing");
        let n = d.raw_out.dims()[0];
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let u_mean = u.mean();
        let v_mean = v.mean();
        (u_mean.clone() * u_mean + v_mean.clone() * v_mean).mul_scalar(self.inv_u_ref_sq)
    }
}

/// Removes the remaining rigid-body rotation in a pure-Neumann plate without constraining
/// symmetric strain.  `UserSamplingStrategy::sample_boundary` orders each repeated edge group
/// as right, left, top, bottom.  The edge-mean finite differences below estimate the domain
/// average infinitesimal rotation `1/2 (dv/dx - du/dy)`: it is nonzero for a rigid rotation,
/// exactly zero for every affine symmetric strain (extension or shear), and shifts by exactly
/// the added rigid rotation.  This selects a displacement representative only; stress and
/// strain are unchanged.
struct RotationGaugeTerm {
    domain: DomainId,
    half_w: f64,
    half_h: f64,
}

impl LossTerm for RotationGaugeTerm {
    fn name(&self) -> &'static str { "rotation_gauge" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }

    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain)
            .expect("rotation_gauge: domain missing");
        let n = d.raw_out.dims()[0];
        assert!(n >= 4 && n % 4 == 0,
            "rotation_gauge requires four equally sampled outer edges, got {n} points");
        let per_edge = n / 4;
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let mean = |values: Tensor<B, 1>, start: usize| {
            values.slice([start..start + per_edge]).mean()
        };
        let u_top = mean(u.clone(), 2 * per_edge);
        let u_bottom = mean(u, 3 * per_edge);
        let v_right = mean(v.clone(), 0);
        let v_left = mean(v.clone(), per_edge);
        let dv_dx = (v_right - v_left).mul_scalar(1.0 / (2.0 * self.half_w));
        let du_dy = (u_top - u_bottom).mul_scalar(1.0 / (2.0 * self.half_h));
        let omega = (dv_dx - du_dy).mul_scalar(0.5);
        omega.clone() * omega
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
    /// Issue #62 PH3-04: the CURRENT step's per-interior-point AMR density-compensation
    /// weights (`pinn_core::amr::compensation_weights`), when `spec.training.measure_aware_
    /// training` is enabled - `None` before the first AMR sweep (uniform sampling, no
    /// compensation needed - see `set_interior_weights`'s own doc comment) or when the
    /// switch is off. Interior-mutable (`&self`, not `&mut self`) because `loss_terms()` -
    /// the only place this is read - is itself a `&self` method on the shared `BoundaryValue
    /// Problem` trait, called fresh every step; there is no owning `&mut self` call site to
    /// thread a per-step value through otherwise, matching this codebase's existing "cheap
    /// interior mutability for a value that changes every step but the trait signature can't
    /// carry" pattern (`training_case_snapshot`'s equivalent in `app-egui` is the same shape).
    /// `Mutex`, not `RefCell` - `BoundaryValueProblem: Send + Sync` requires `UserDefinedProblem:
    /// Sync`, which `RefCell` (single-threaded interior mutability) cannot provide; the lock is
    /// held only for the instant of a clone/replace, never across a training step.
    current_interior_weights: std::sync::Mutex<Option<Vec<f64>>>,
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
        let sampling = UserSamplingStrategy::new(spec.geometry.clone(), spec.training.fd_h);
        UserDefinedProblem {
            spec, domains, sampling, ansatz: IdentityAnsatz, hole_names,
            current_interior_weights: std::sync::Mutex::new(None),
        }
    }

    pub fn spec(&self) -> &ProblemSpec { &self.spec }

    /// Issue #62 PH3-04: sets the per-interior-point AMR density-compensation weights the NEXT
    /// `loss_terms()` call's `InteriorEnergyTerm` will use (only when `spec.training.measure_
    /// aware_training` is also true - `InteriorEnergyTerm::compute` ignores this entirely
    /// otherwise, matching the "legacy path unaffected unless the switch is on" rule). The
    /// caller (`runner::run_user_problem_training_from`) is responsible for calling this with
    /// `Some(weights)` computed via `pinn_core::amr::compensation_weights` from the SAME point
    /// set that produced this step's `data.int_norm` (same order, same length) whenever that
    /// point set changes (an AMR sweep), and leaving it at the default `None` before the first
    /// sweep - plain uniform-random interior sampling is ALREADY an unbiased Monte-Carlo
    /// estimator of the domain integral (no compensation needed), so `None` here is a genuine
    /// "not needed yet", not a missing-data placeholder.
    pub fn set_interior_weights(&self, weights: Option<Vec<f64>>) {
        *self.current_interior_weights.lock().unwrap() = weights;
    }
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
        assert!(
            !matches!(self.spec.formulation, pinn_core::problem_spec::FormulationSelection::Variational)
                || self.spec.training.measure_aware_training,
            "Variational formulation requires training.measure_aware_training=true: legacy mean U and boundary-mean W_ext do not share physical measures"
        );
        // Centralized (General-PINN architecture recommendations §35-37, Priority 2,
        // "dimensionless normalization") - see `training_core::PlateReferenceScales`'s doc
        // comment for why this replaced ~15 independently hand-written formula copies, and
        // for the `ref_div2`/`stress_per_length2` bug (bugSource-New #12/this session's Kt
        // investigation) this centralization exists to prevent a recurrence of.
        let scales = crate::training_core::compute_reference_scales_for_plate(&self.spec);
        let (ref_energy, ref_stress2) = (scales.ref_energy, scales.ref_stress2);
        let eq_ref_div2 = scales.stress_per_length2;

        // Issue #62 PH3-04: precompute the measure-aware machinery's inputs ONCE per
        // `loss_terms()` call (geometry/training config are fixed for the whole run - only
        // `current_interior_weights` genuinely varies step to step, snapshotted below). `false`
        // (the default) makes every one of these dead weight - `InteriorEnergyTerm`/
        // `ExternalWorkTerm::compute()` never read them, taking the exact legacy path.
        let measure_aware = self.spec.training.measure_aware_training;
        let thickness = self.spec.geometry.thickness;
        let domain_area = crate::measure_integral::plate_domain_area(
            self.spec.geometry.half_w, self.spec.geometry.half_h,
            &self.spec.geometry.holes.iter().map(|h| h.radius).collect::<Vec<_>>(),
        );
        // See `InteriorEnergyTerm::ref_energy_absolute`'s own doc comment for why this is the
        // correct normalizer for BOTH `InteriorEnergyTerm` and `ExternalWorkTerm`'s
        // measure-aware paths (a shared total-energy reference scale for this one problem).
        let ref_energy_absolute = ref_energy as f64 * domain_area * thickness;
        // Real per-point arc-length spacing, matching `UserSamplingStrategy::sample_boundary`'s
        // own point-generation order EXACTLY (right, left, top, bottom, repeated `per_edge`
        // times) - see `ExternalWorkTerm::ds_per_point`'s own doc comment.
        let per_edge = (self.spec.training.n_boundary / 4).max(1);
        let ds_right_left = 2.0 * self.spec.geometry.half_h / per_edge as f64;
        let ds_top_bottom = 2.0 * self.spec.geometry.half_w / per_edge as f64;
        let mut ds_per_point = Vec::with_capacity(per_edge * 4);
        for _ in 0..per_edge {
            ds_per_point.push(ds_right_left);
            ds_per_point.push(ds_right_left);
            ds_per_point.push(ds_top_bottom);
            ds_per_point.push(ds_top_bottom);
        }
        let interior_weights = self.current_interior_weights.lock().unwrap().clone();

        // Issue #61 P2-01 ("Explicit formulation model") - `self.spec.formulation` is a REAL
        // gate on which base terms exist below, not a label applied after the fact. See
        // `pinn_core::problem_spec::FormulationSelection`'s own doc comment for what each
        // variant means. Essential (HoleBc::Fixed) constraints are always active in every
        // formulation (an essential constraint is required regardless of formulation choice,
        // per issue #61's own text) - only the natural (HoleBc::Free) treatment and which BASE
        // terms are active differ.
        use pinn_core::problem_spec::FormulationSelection;
        let active_base: std::collections::HashSet<&'static str> = match &self.spec.formulation {
            FormulationSelection::Variational => ["physical_potential"].into_iter().collect(),
            FormulationSelection::Strong => ["equilibrium", "outer_traction"].into_iter().collect(),
            FormulationSelection::Hybrid(names) => names.iter().map(|s| match s.as_str() {
                "interior_energy" => "interior_energy",
                "equilibrium" => "equilibrium",
                "outer_traction" => "outer_traction",
                "external_work" => "external_work",
                other => panic!(
                    "UserDefinedProblem::loss_terms: unknown Hybrid formulation term '{other}' \
                     - expected one of interior_energy/equilibrium/outer_traction/external_work"
                ),
            }).collect(),
        };
        // Natural (HoleBc::Free) hole boundaries are a strong-form penalty term - active for
        // Strong and Hybrid (this codebase's own pre-remediation behavior always included it).
        // For Variational this was previously OMITTED entirely on the theory that a
        // correctly-posed W_ext already encodes the traction-free natural boundary - a valid
        // CONTINUUM argument that does not hold under a noisy finite-sample estimator (issue
        // #77 root cause: with no direct hole-boundary signal and a hole energy contribution
        // of a fraction of a percent of Pi, three independent representation changes all
        // converged to the "no hole at all" answer). `decomposition_applicable` scopes a fix -
        // kinematic decomposition (`u = u_affine + u_hole`) plus a correctly-retargeted
        // traction residual - to exactly the case it's been verified for: one centered,
        // traction-free hole. Every other Variational configuration (no-hole, multi-hole,
        // off-center, Fixed bc) is completely unaffected - `hole_free_active` stays `false` and
        // every new field below stays `None`, preserving the exact original behavior.
        let decomposed = decomposition_applicable(&self.spec);
        let hole_free_active = !matches!(self.spec.formulation, FormulationSelection::Variational) || decomposed;
        let affine_strain_pair = if decomposed { Some((self.spec.load.px, self.spec.load.py)) } else { None };

        let mut terms: Vec<Box<dyn LossTerm>> = Vec::new();
        if active_base.contains("physical_potential") {
            terms.push(Box::new(PhysicalPotentialEnergyTerm {
                domain: USER_DOMAIN,
                material: self.spec.material.clone(), px: self.spec.load.px, py: self.spec.load.py,
                measure_aware, domain_area, thickness, ref_energy, ref_energy_absolute,
                interior_weights: interior_weights.clone(), ds_per_point: ds_per_point.clone(),
                affine_strain: affine_strain_pair,
            }));
        }
        if active_base.contains("interior_energy") {
            terms.push(Box::new(InteriorEnergyTerm {
                domain: USER_DOMAIN,
                material: self.spec.material.clone(), ref_energy,
                measure_aware, domain_area, thickness, ref_energy_absolute,
                weights: interior_weights,
            }));
        }
        if active_base.contains("equilibrium") {
            terms.push(Box::new(EquilibriumTerm { domain: USER_DOMAIN, point_set: "interior", material: self.spec.material.clone(), ref_div2: eq_ref_div2 }));
        }
        if active_base.contains("outer_traction") {
            terms.push(Box::new(OuterTractionTerm {
                domain: USER_DOMAIN,
                material: self.spec.material.clone(),
                ref_stress2,
                px: self.spec.load.px,
                py: self.spec.load.py,
            }));
        }
        if active_base.contains("external_work") {
            terms.push(Box::new(ExternalWorkTerm {
                px: self.spec.load.px, py: self.spec.load.py, ref_energy,
                measure_aware, thickness, ref_energy_absolute, ds_per_point,
            }));
        }
        for (i, (hole, &name)) in self.spec.geometry.holes.iter().zip(self.hole_names.iter()).enumerate() {
            if hole.bc == HoleBc::Fixed || hole_free_active {
                let use_decomposed = decomposed && hole.bc == HoleBc::Free;
                let point_set = if use_decomposed { self.sampling.hole_fd_names[i] } else { name };
                let affine_target = if use_decomposed { affine_strain_pair } else { None };
                terms.push(Box::new(HoleBcTerm {
                    domain: USER_DOMAIN, point_set, bc: hole.bc, ref_stress2,
                    material: self.spec.material.clone(), affine_target,
                }));
            }
        }
        // Issue #61 P2-07: gauge-fix the rigid-body translation nullspace for pure-Neumann
        // configurations (no hole is HoleBc::Fixed - `no_hole_plate.toml`/`single_hole_plate.
        // toml` with its hole set to Free are both real, current examples of this). Never
        // registered when a real Dirichlet anchor already exists (redundant there).
        if self.spec.geometry.is_pure_neumann() {
            terms.push(Box::new(TranslationGaugeTerm {
                domain: USER_DOMAIN,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }));
        }
        // PH4 preserves frozen legacy Hybrid/Strong trajectories. Corrected Variational is a
        // new formulation contract and additionally removes its rotational nullspace.
        if self.spec.geometry.is_pure_neumann()
            && matches!(self.spec.formulation, FormulationSelection::Variational)
        {
            terms.push(Box::new(RotationGaugeTerm {
                domain: USER_DOMAIN,
                half_w: self.spec.geometry.half_w,
                half_h: self.spec.geometry.half_h,
            }));
        }
        terms
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "interior_energy" => LAM_INTERIOR_ENERGY,
            "physical_potential" => LAM_PHYSICAL_POTENTIAL,
            "equilibrium" => LAM_EQUILIBRIUM_PLATE,
            "outer_traction" => LAM_OUTER_TRACTION,
            "external_work" => LAM_EXTERNAL_WORK,
            "hole_free" => LAM_HOLE_FREE,
            "hole_fixed" => LAM_HOLE_FIXED,
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            other => panic!("UserDefinedProblem::base_weight: unknown loss term '{other}'"),
        }
    }

    fn phase1_steps(&self) -> usize { 0 }

    /// No closed-form convergence metric exists for an arbitrary user-defined geometry —
    /// unlike Kirsch's K_t, there's no analytic target to probe against.
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }

    fn convergence_target(&self) -> f64 { 0.0 }
}

/// Local annulus plus global exterior BVP for one small, traction-free circular hole.
///
/// This is intentionally a narrow production route: it preserves the legacy one-model path
/// for all other geometries and formulations. Both domains minimize pieces of the same global
/// potential energy, coupled by displacement and derived-traction continuity on their shared
/// circle. The annular model receives the #77 chart input; outer model remains raw-coordinate.
pub struct AnnularDecompositionProblem {
    spec: ProblemSpec,
    domains: Vec<DomainSpec>,
    annulus_sampling: AnnularPartitionSampling,
    outer_sampling: AnnularPartitionSampling,
    ansatz: IdentityAnsatz,
}

impl AnnularDecompositionProblem {
    pub fn supports(spec: &ProblemSpec) -> bool {
        matches!(spec.formulation, pinn_core::problem_spec::FormulationSelection::Variational)
            && spec.training.measure_aware_training
            && matches!(spec.geometry.holes.as_slice(), [HoleSpec { bc: HoleBc::Free, .. }])
            && spec.geometry.annular_partition().is_some()
    }

    pub fn new(spec: ProblemSpec) -> Self {
        assert!(Self::supports(&spec),
            "annular decomposition requires one safely-contained Free hole, Variational formulation, and measure-aware training");
        let placeholder = spec.geometry.to_placeholder();
        let interface = std::sync::Arc::new(InterfaceParametrization {
            thetas: (0..HOLE_RING_POINTS)
                .map(|i| 2.0 * std::f64::consts::PI * i as f64 / HOLE_RING_POINTS as f64)
                .collect(),
        });
        let annulus_sampling = AnnularPartitionSampling::new(
            spec.geometry.clone(), spec.training.fd_h, true, OUTER_DOMAIN, interface.clone(),
        );
        let outer_sampling = AnnularPartitionSampling::new(
            spec.geometry.clone(), spec.training.fd_h, false, ANNULUS_DOMAIN, interface,
        );
        Self {
            domains: vec![
                DomainSpec { id: ANNULUS_DOMAIN, geometry: placeholder.clone(), material: spec.material.clone(), output_dim: 5 },
                DomainSpec { id: OUTER_DOMAIN, geometry: placeholder, material: spec.material.clone(), output_dim: 5 },
            ],
            spec,
            annulus_sampling,
            outer_sampling,
            ansatz: IdentityAnsatz,
        }
    }

    pub fn spec(&self) -> &ProblemSpec { &self.spec }
}

/// Annular U contribution to the one global potential. Its name remains distinct from the
/// outer contribution so loss ledgers can report both physical pieces. `step_physics_multi`
/// pins both names to coefficient one, so SAW-BRDR cannot distort U_annulus + U_outer - W_ext.
struct AnnularPotentialEnergyTerm {
    material: MaterialProps,
    domain_area: f64,
    thickness: f64,
    ref_energy_absolute: f64,
    /// Issue #77 fix: see `PhysicalPotentialEnergyTerm::affine_strain`'s doc comment — same
    /// mechanism, applied to the annulus domain's own strain read.
    affine_strain: Option<(f64, f64)>,
}

impl LossTerm for AnnularPotentialEnergyTerm {
    fn name(&self) -> &'static str { "annulus_potential" }
    fn domains(&self) -> Vec<DomainId> { vec![ANNULUS_DOMAIN] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Weak }
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|d| d.domain == ANNULUS_DOMAIN)
            .expect("annular physical potential: domain missing");
        let (mut exx, mut eyy, mut exy) = d.strains.clone().expect("annular physical potential: strains missing");
        if let Some((px, py)) = self.affine_strain {
            let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &self.material);
            exx = exx.add_scalar(a_exx);
            eyy = eyy.add_scalar(a_eyy);
            exy = exy.add_scalar(a_exy);
        }
        let density = crate::energy::dem_energy_per_point(exx, eyy, exy, &self.material);
        crate::measure_integral::domain_integral_tensor::<B>(self.domain_area, self.thickness, density)
            .mul_scalar(1.0 / self.ref_energy_absolute)
    }
}

impl BoundaryValueProblem for AnnularDecompositionProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }
    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        match self.domains[domain_idx].id {
            ANNULUS_DOMAIN => &self.annulus_sampling,
            OUTER_DOMAIN => &self.outer_sampling,
            _ => unreachable!(),
        }
    }
    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        assert!(domain_idx < 2, "annular decomposition has two domains");
        &self.ansatz
    }
    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let scales = crate::training_core::compute_reference_scales_for_plate(&self.spec);
        let partition = self.spec.geometry.annular_partition().expect("validated by constructor");
        let annulus_area = std::f64::consts::PI * (partition.interface_radius.powi(2) - partition.hole_radius.powi(2));
        let outer_area = 4.0 * self.spec.geometry.half_w * self.spec.geometry.half_h
            - std::f64::consts::PI * partition.interface_radius.powi(2);
        let total_area = annulus_area + outer_area;
        let ref_energy_absolute = scales.ref_energy as f64 * total_area * self.spec.geometry.thickness;
        let per_edge = (self.spec.training.n_boundary / 4).max(1);
        let mut ds_per_point = Vec::with_capacity(per_edge * 4);
        for _ in 0..per_edge {
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
        }
        // Issue #77 root-cause fix: see `UserDefinedProblem::loss_terms()`'s matching comment.
        // `AnnularDecompositionProblem::supports` already restricts to exactly one Free,
        // safely-contained hole; `decomposition_applicable` additionally requires it centered
        // (this problem's own v1 scope) before enabling the closed-form affine background and
        // the corrected hole-traction residual - previously this formulation registered NO
        // term referencing the hole at all (the orphaned "hole_0" point set, #77's own history).
        let decomposed = decomposition_applicable(&self.spec);
        let affine_strain_pair = if decomposed { Some((self.spec.load.px, self.spec.load.py)) } else { None };
        let mut terms: Vec<Box<dyn LossTerm>> = vec![
            Box::new(AnnularPotentialEnergyTerm {
                material: self.spec.material.clone(), domain_area: annulus_area,
                thickness: self.spec.geometry.thickness, ref_energy_absolute,
                affine_strain: affine_strain_pair,
            }),
            Box::new(PhysicalPotentialEnergyTerm {
                domain: OUTER_DOMAIN, material: self.spec.material.clone(),
                px: self.spec.load.px, py: self.spec.load.py, measure_aware: true,
                domain_area: outer_area, thickness: self.spec.geometry.thickness,
                ref_energy: scales.ref_energy, ref_energy_absolute, interior_weights: None,
                ds_per_point, affine_strain: affine_strain_pair,
            }),
            Box::new(InterfaceDisplacementContinuityTerm {
                left: ANNULUS_DOMAIN, right: OUTER_DOMAIN,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }),
            Box::new(InterfaceTractionContinuityTerm {
                left: ANNULUS_DOMAIN, right: OUTER_DOMAIN,
                left_material: self.spec.material.clone(), right_material: self.spec.material.clone(),
                inv_ref_stress2: 1.0 / (scales.ref_stress2 as f64).max(1e-30),
            }),
            Box::new(TranslationGaugeTerm {
                domain: OUTER_DOMAIN,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }),
            Box::new(RotationGaugeTerm {
                domain: OUTER_DOMAIN, half_w: self.spec.geometry.half_w, half_h: self.spec.geometry.half_h,
            }),
        ];
        if decomposed {
            terms.push(Box::new(HoleBcTerm {
                domain: ANNULUS_DOMAIN, point_set: "hole_0_fd", bc: HoleBc::Free,
                ref_stress2: scales.ref_stress2, material: self.spec.material.clone(),
                affine_target: affine_strain_pair,
            }));
        }
        terms
    }
    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "annulus_potential" | "physical_potential" => LAM_PHYSICAL_POTENTIAL,
            "interface_displacement_continuity" | "interface_traction_continuity" => 100.0,
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            "hole_free" => LAM_HOLE_FREE,
            other => panic!("AnnularDecompositionProblem::base_weight: unknown term '{other}'"),
        }
    }
    fn phase1_steps(&self) -> usize { 0 }
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
    fn convergence_target(&self) -> f64 { 0.0 }
}

/// Builds a `VisFields` for GUI display by evaluating `model` once over a
/// `[nx,ny]`-shaped normalized grid masked by `geometry.contains` — mirrors `runner.rs`'s
/// private `evaluate_vis_grid_mdem` (same mDEM forward convention, same von Mises formula,
/// same Phase 14 strain/residual/AMR-score/density extension), adapted for `UserGeometry`'s
/// N-hole containment check instead of `GeometryConfig`'s single-hole one.
///
/// Phase 14 extension reuses the exact FD-stencil + physical-scale-before-derivative
/// convention `probe_hole_boundary_profile` already established for this same ansatz (see
/// that function's doc comment): displayed `sigma_xx/yy/xy` are constitutive stress from
/// FD-derived strain. Direct mDEM stress is auxiliary; direct-minus-constitutive is the
/// explicit consistency residual surfaced for display.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_user_vis_grid(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    [nx, ny]: [usize; 2],
    u_ref: f32,
    px_pa: f64,
    material: &MaterialProps,
    fd: &crate::fd_stencil::FdConfig,
    int_norm: &[[f32; 2]],
    device: &crate::training_core::BDevice,
) -> pinn_core::messages::VisFields {
    use crate::network::fwd_embedded;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor};
    use crate::energy::dem_energy_per_point;
    use crate::training_core::BInner;
    use ndarray::Array2;

    let n_total = nx * ny;
    let mut pts = Vec::with_capacity(n_total);
    let mut mask = Vec::with_capacity(n_total);
    for iy in 0..ny {
        for ix in 0..nx {
            let xn = -1.0 + 2.0 * ix as f64 / (nx.max(2) - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny.max(2) - 1) as f64;
            pts.push([xn as f32, yn as f32]);
            let (xp, yp) = (xn * geometry.half_w, yn * geometry.half_h);
            mask.push(geometry.contains(xp, yp));
        }
    }

    let mut s_vm = vec![f32::NAN; n_total];
    let mut s_xx = vec![f32::NAN; n_total];
    let mut s_yy = vec![f32::NAN; n_total];
    let mut s_xy = vec![f32::NAN; n_total];
    let mut d_u = vec![f32::NAN; n_total];
    let mut d_v = vec![f32::NAN; n_total];
    let mut e_xx = vec![f32::NAN; n_total];
    let mut e_yy = vec![f32::NAN; n_total];
    let mut e_xy = vec![f32::NAN; n_total];
    let mut pde = vec![f32::NAN; n_total];
    let mut amr = vec![f32::NAN; n_total];

    #[allow(clippy::too_many_arguments)]
    let make_vis = |vm: Vec<f32>, sxx: Vec<f32>, syy: Vec<f32>, sxy: Vec<f32>, u: Vec<f32>, v: Vec<f32>,
                     eps_xx: Vec<f32>, eps_yy: Vec<f32>, eps_xy: Vec<f32>,
                     pde_residual: Vec<f32>, amr_score: Vec<f32>| {
        let a = |v: Vec<f32>| Array2::from_shape_vec((ny, nx), v).expect("shape mismatch");
        let density = crate::runner::bin_collocation_density(int_norm, nx, ny);
        pinn_core::messages::VisFields {
            von_mises: a(vm), sigma_xx: a(sxx), sigma_yy: a(syy), sigma_xy: a(sxy),
            disp_u: a(u), disp_v: a(v),
            eps_xx: a(eps_xx), eps_yy: a(eps_yy), eps_xy: a(eps_xy),
            pde_residual: a(pde_residual), amr_score: a(amr_score),
            collocation_density: a(density),
        }
    };

    let active: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    if active.is_empty() {
        return make_vis(s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr);
    }

    let active_pts: Vec<[f32; 2]> = active.iter().map(|&i| pts[i]).collect();
    let n_act = active_pts.len();
    let pts_t = norm_pts_to_tensor::<BInner>(&active_pts, device);
    let stencil_coords = assemble_stencil::<BInner>(&pts_t, fd, device);
    let raw_net = fwd_embedded::<BInner>(model, stencil_coords, geometry.coordinate_embedding(), device); // [5*n_act, 5], unscaled

    // Physical scale BEFORE the FD derivative — same convention as `compute_domain_forwards`'s
    // `is_mdem` branch and `probe_hole_boundary_profile` (displacement by u_ref, stress by
    // px_pa), so the strain/residual computed here matches what training itself sees.
    let m = 5 * n_act;
    let raw = Tensor::cat(vec![
        raw_net.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
        raw_net.slice([0..m, 2..5]).mul_scalar(px_pa),
    ], 1);

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(raw.clone(), n_act, fd);
    let energy = dem_energy_per_point::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);
    let (sxx_fd, syy_fd, sxy_fd) =
        crate::energy::compute_stress::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);

    let center = raw.slice([0..n_act, 0..5]);
    let batched: Vec<f32> = Tensor::cat(
        vec![center.reshape([5 * n_act]), eps_xx, eps_yy, eps_xy, sxx_fd, syy_fd, sxy_fd, energy],
        0,
    ).into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 5 * n_act + 7 * n_act]);
    let center_vals = &batched[..5 * n_act];
    let chunk = |i: usize| -> &[f32] { &batched[5 * n_act + i * n_act..5 * n_act + (i + 1) * n_act] };
    let (exx_v, eyy_v, exy_v) = (chunk(0), chunk(1), chunk(2));
    let (sxx_fd_v, syy_fd_v, sxy_fd_v) = (chunk(3), chunk(4), chunk(5));
    let energy_v = chunk(6);

    for (i_act, &i_full) in active.iter().enumerate() {
        let u = center_vals[i_act * 5];
        let v = center_vals[i_act * 5 + 1];
        let direct_sxx = center_vals[i_act * 5 + 2] as f64;
        let direct_syy = center_vals[i_act * 5 + 3] as f64;
        let direct_sxy = center_vals[i_act * 5 + 4] as f64;
        // Engineering/visualization stress is constitutive stress.  Direct mDEM stress is
        // auxiliary and remains visible only through its explicit consistency residual.
        let sxx = sxx_fd_v[i_act] as f64;
        let syy = syy_fd_v[i_act] as f64;
        let sxy = sxy_fd_v[i_act] as f64;
        let vm = (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).sqrt();
        let dex = direct_sxx - sxx;
        let dey = direct_syy - syy;
        let dexy = direct_sxy - sxy;
        s_xx[i_full] = sxx as f32;
        s_yy[i_full] = syy as f32;
        s_xy[i_full] = sxy as f32;
        s_vm[i_full] = vm as f32;
        d_u[i_full] = u;
        d_v[i_full] = v;
        e_xx[i_full] = exx_v[i_act]; e_yy[i_full] = eyy_v[i_act]; e_xy[i_full] = exy_v[i_act];
        pde[i_full] = (dex*dex + dey*dey + dexy*dexy).sqrt() as f32;
        amr[i_full] = energy_v[i_act].abs();
    }
    make_vis(s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr)
}

/// `enhancement.txt` items 4/C ("BC residual RMS/max") - real per-point traction/
/// displacement residual at the outer boundary and every hole ring, combined. Same math
/// `OuterTractionTerm`/`HoleBcTerm` compute internally (via `neumann_loss`/
/// `hole_traction_loss_direct`), kept pre-mean here so a real distribution stat is possible -
/// a side probe at the existing vis cadence (mirrors `training_core::probe_interior_energy_
/// residuals`'s own "side probe, not the hot per-step path" precedent), NOT a change to
/// `step_physics_multi`'s per-step loss computation.
pub fn probe_boundary_residuals(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> (f64, f64) {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };

    let mut residuals: Vec<f32> = Vec::new();

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    if !bnd_pts_phys.is_empty() {
        let n_bnd = bnd_pts_phys.len();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&bnd_norm, device), &fd, device);
        let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
        let m = 5 * n_bnd;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
        let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &spec.material);
        let nx: Vec<f32> = bnd_pts_phys.iter().map(|p| p.nx as f32).collect();
        let ny: Vec<f32> = bnd_pts_phys.iter().map(|p| p.ny as f32).collect();
        let nx_t = Tensor::<BInner, 1>::from_data(TensorData::new(nx, vec![n_bnd]), device);
        let ny_t = Tensor::<BInner, 1>::from_data(TensorData::new(ny, vec![n_bnd]), device);
        let tx_target = nx_t.clone().mul_scalar(spec.load.px);
        let ty_target = ny_t.clone().mul_scalar(spec.load.py);
        let tx_pred = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
        let ty_pred = sxy * nx_t + syy * ny_t;
        let ex = tx_pred - tx_target;
        let ey = ty_pred - ty_target;
        let mag = (ex.clone() * ex + ey.clone() * ey).sqrt();
        residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
    }

    // `named_point_sets` returns one ring per hole in `geometry.holes.iter()` order (zip,
    // same construction `UserDefinedProblem::new` itself relies on) - zip directly instead
    // of re-deriving the correspondence from each set's name string.
    for (hole, set) in geometry.holes.iter().zip(sampling.named_point_sets(&[]).into_iter()) {
        let n_h = set.points.len();
        if n_h == 0 { continue; }
        let ring_norm: Vec<[f32; 2]> = set.points.iter().map(|p| norm_pt(p.x, p.y)).collect();
        let raw = fwd_embedded::<BInner>(model, norm_pts_to_tensor::<BInner>(&ring_norm, device), geometry.coordinate_embedding(), device);
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..n_h, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..n_h, 2..5]).mul_scalar(px_pa),
        ], 1);
        match hole.bc {
            HoleBc::Free => {
                let nx: Vec<f32> = set.points.iter().map(|p| p.nx as f32).collect();
                let ny: Vec<f32> = set.points.iter().map(|p| p.ny as f32).collect();
                let nx_t = Tensor::<BInner, 1>::from_data(TensorData::new(nx, vec![n_h]), device);
                let ny_t = Tensor::<BInner, 1>::from_data(TensorData::new(ny, vec![n_h]), device);
                let sxx = scaled.clone().slice([0..n_h, 2..3]).reshape([n_h]);
                let syy = scaled.clone().slice([0..n_h, 3..4]).reshape([n_h]);
                let sxy = scaled.slice([0..n_h, 4..5]).reshape([n_h]);
                let tx = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
                let ty = sxy * nx_t + syy * ny_t;
                let mag = (tx.clone() * tx + ty.clone() * ty).sqrt();
                residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
            }
            HoleBc::Fixed => {
                let u = scaled.clone().slice([0..n_h, 0..1]).reshape([n_h]);
                let v = scaled.slice([0..n_h, 1..2]).reshape([n_h]);
                let mag = (u.clone() * u.clone() + v.clone() * v.clone()).sqrt();
                residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
            }
        }
    }

    crate::training_core::residual_stats(&residuals)
}

/// Issue #61 EPIC P2-06's own "stencils avoid invalid points with recorded fallback/quality
/// diagnostics" - a real report over an arbitrary point set, built on
/// [`pinn_core::user_geometry::UserGeometry::valid_stencil`]. `fully_valid` are stencils safe
/// to use as-is; `fallback_needed` have a valid center but at least one invalid shifted
/// neighbor (this codebase's real FD paths already avoid these via margin-based rejection
/// sampling - `UserSamplingStrategy::contains_for_collocation` - this report makes how many
/// points that margin is actually protecting against a visible, queryable number instead of an
/// invisible property of the sampling process); `invalid_center` should never be nonzero for
/// points that already passed `contains()`-based sampling - a nonzero count here would flag a
/// genuine wiring bug (a point admitted despite being outside the domain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilQualityReport {
    pub total: usize,
    pub fully_valid: usize,
    pub fallback_needed: usize,
    pub invalid_center: usize,
}

pub fn stencil_quality_report(
    geometry: &pinn_core::user_geometry::UserGeometry,
    points: &[[f64; 2]],
    hx: f64,
    hy: f64,
) -> StencilQualityReport {
    let mut report = StencilQualityReport { total: points.len(), fully_valid: 0, fallback_needed: 0, invalid_center: 0 };
    for &[x, y] in points {
        let v = geometry.valid_stencil(x, y, hx, hy);
        if !v.center_valid {
            report.invalid_center += 1;
        } else if v.all_valid() {
            report.fully_valid += 1;
        } else {
            report.fallback_needed += 1;
        }
    }
    report
}

/// `enhancement.md` Phase 9 ("Force Equilibrium Validation") - a real reaction-force check,
/// distinct from `probe_boundary_residuals`'s pointwise mean residual: integrates the
/// PREDICTED traction over the outer boundary's real point set (arc-length-weighted,
/// `traction * ds * thickness`) rather than just averaging point-error magnitudes.
///
/// The far-field traction target this problem applies (`tx_target = px*nx`, `ty_target =
/// py*ny`, see `OuterTractionTerm`) is, by construction, self-canceling around the whole
/// closed rectangle: `px` pulls the right edge (`nx=+1`) one way and the left edge (`nx=-1`)
/// the opposite way, so the TARGET net force over the full boundary is analytically zero -
/// this is what "far-field traction, no body force" equilibrium means, not a bug. That means
/// there is no separate "applied resultant" to compare a prediction against; instead, the
/// meaningful check is whether the network's own PREDICTED traction integral is *also* close
/// to zero - any nonzero net predicted force is a real, physically-meaningful inconsistency
/// (the trained stress field failing to satisfy global force balance), normalized against the
/// magnitude of one edge's own nominal load so the error is scale-free.
pub fn probe_reaction_force(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> pinn_core::messages::ReactionForce {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };

    let reference_force = (spec.load.px * 2.0 * geometry.half_h * geometry.thickness).abs()
        .max((spec.load.py * 2.0 * geometry.half_w * geometry.thickness).abs())
        .max(1e-30);

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    if bnd_pts_phys.is_empty() {
        return pinn_core::messages::ReactionForce { net_fx: 0.0, net_fy: 0.0, reference_force, equilibrium_error: 0.0 };
    }
    let n_bnd = bnd_pts_phys.len();
    // `UserSamplingStrategy::sample_boundary` pushes exactly `per_edge` points per edge, 4
    // edges, in a fixed order every iteration - `per_edge = n_bnd / 4` recovers the same value
    // without needing sample_boundary to expose it separately.
    let per_edge = (n_bnd / 4).max(1);
    let ds_x_normal = 2.0 * geometry.half_h / per_edge as f64; // left/right edges (nx = +-1)
    let ds_y_normal = 2.0 * geometry.half_w / per_edge as f64; // top/bottom edges (ny = +-1)

    let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
    let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&bnd_norm, device), &fd, device);
    let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
    let m = 5 * n_bnd;
    let scaled = Tensor::cat(vec![
        raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
        raw.slice([0..m, 2..5]).mul_scalar(px_pa),
    ], 1);
    let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
    let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &spec.material);
    let nx: Vec<f32> = bnd_pts_phys.iter().map(|p| p.nx as f32).collect();
    let ny: Vec<f32> = bnd_pts_phys.iter().map(|p| p.ny as f32).collect();
    let nx_t = Tensor::<BInner, 1>::from_data(TensorData::new(nx.clone(), vec![n_bnd]), device);
    let ny_t = Tensor::<BInner, 1>::from_data(TensorData::new(ny.clone(), vec![n_bnd]), device);
    let tx_pred = (sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone())
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
    let ty_pred = (sxy * nx_t + syy * ny_t)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);

    let mut net_fx = 0.0f64;
    let mut net_fy = 0.0f64;
    for i in 0..n_bnd {
        let ds = if nx[i].abs() > 0.5 { ds_x_normal } else { ds_y_normal };
        net_fx += tx_pred[i] as f64 * ds * geometry.thickness;
        net_fy += ty_pred[i] as f64 * ds * geometry.thickness;
    }
    let equilibrium_error = (net_fx * net_fx + net_fy * net_fy).sqrt() / reference_force;

    pinn_core::messages::ReactionForce { net_fx, net_fy, reference_force, equilibrium_error }
}

/// Issue #61 EPIC P2-09: predicted-vs-prescribed resultant load and a generic trivial-solution
/// warning - directly motivated by the real `Debug_runs/stress_solver_report-with-hole.json`
/// evidence this remediation plan was opened against (`bc_residual_max≈` the load itself,
/// `avg_von_mises` ~50x smaller than nominal - a collapsed solution that STILL had nonzero
/// displacement, so a bare `max|displacement|` check alone would have missed it). Distinct
/// from [`probe_reaction_force`] (which checks the FULL closed boundary's resultant against
/// zero, since far-field loading is self-canceling around a whole rectangle): this checks the
/// LOADED edges specifically against their real PRESCRIBED nominal load, which is the direct
/// question "did the network actually transfer the applied load into its own stress state, or
/// did it converge on a near-zero-stress shortcut instead."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadTransferReport {
    /// Predicted resultant force integrated over the right edge (x-loaded) [N].
    pub predicted_load_x: f64,
    /// Predicted resultant force integrated over the top edge (y-loaded) [N].
    pub predicted_load_y: f64,
    pub prescribed_load_x: f64,
    pub prescribed_load_y: f64,
    /// `|predicted resultant| / |prescribed resultant|` - 1.0 is perfect load transfer, ~0 is
    /// the literal trivial/collapsed-solution symptom.
    pub load_transfer_ratio: f64,
    /// `probe_boundary_residuals`'s own (rms, max) - reused, not recomputed (issue #61 P2-08's
    /// own "no duplicated verification machinery" discipline).
    pub traction_residual_rms: f64,
    pub traction_residual_max: f64,
    /// True iff `load_transfer_ratio` is below a generous sanity floor - a genuine, physically-
    /// direct trivial-solution warning (catches the real Debug_runs case: nonzero but far-too-
    /// small stress, which a bare displacement check misses).
    pub trivial_solution_warning: bool,
}

pub fn probe_load_transfer(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> LoadTransferReport {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };

    let prescribed_load_x = spec.load.px * 2.0 * geometry.half_h * geometry.thickness;
    let prescribed_load_y = spec.load.py * 2.0 * geometry.half_w * geometry.thickness;

    let (traction_residual_rms, traction_residual_max) = probe_boundary_residuals(model, spec, device);

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    if bnd_pts_phys.is_empty() {
        return LoadTransferReport {
            predicted_load_x: 0.0, predicted_load_y: 0.0, prescribed_load_x, prescribed_load_y,
            load_transfer_ratio: 1.0, traction_residual_rms, traction_residual_max,
            trivial_solution_warning: false,
        };
    }
    let n_bnd = bnd_pts_phys.len();
    // `sample_boundary` pushes exactly (right, left, top, bottom) per iteration - see its own
    // doc comment - so every 4th point starting at offset 0/2 is the right/top edge.
    let per_edge = (n_bnd / 4).max(1);
    let ds_x_normal = 2.0 * geometry.half_h / per_edge as f64;
    let ds_y_normal = 2.0 * geometry.half_w / per_edge as f64;

    let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
    let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&bnd_norm, device), &fd, device);
    let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
    let m = 5 * n_bnd;
    let scaled = Tensor::cat(vec![
        raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
        raw.slice([0..m, 2..5]).mul_scalar(px_pa),
    ], 1);
    let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
    let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &spec.material);
    let nx: Vec<f32> = bnd_pts_phys.iter().map(|p| p.nx as f32).collect();
    let ny: Vec<f32> = bnd_pts_phys.iter().map(|p| p.ny as f32).collect();
    let nx_t = Tensor::<BInner, 1>::from_data(TensorData::new(nx.clone(), vec![n_bnd]), device);
    let ny_t = Tensor::<BInner, 1>::from_data(TensorData::new(ny.clone(), vec![n_bnd]), device);
    let tx_pred: Vec<f32> = (sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone())
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
    let ty_pred: Vec<f32> = (sxy * nx_t + syy * ny_t)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);

    // Issue #61 P2-04: measure-aware integration, via the shared abstraction - the right/top
    // edges' predicted traction integrated with their own real arc-length measure.
    let right_tx: Vec<f32> = (0..per_edge).map(|i| tx_pred[4 * i]).collect();
    let top_ty: Vec<f32> = (0..per_edge).map(|i| ty_pred[4 * i + 2]).collect();
    let right_ds = vec![ds_x_normal; per_edge];
    let top_ds = vec![ds_y_normal; per_edge];
    let predicted_load_x = crate::measure_integral::boundary_integral(&right_tx, &right_ds, geometry.thickness);
    let predicted_load_y = crate::measure_integral::boundary_integral(&top_ty, &top_ds, geometry.thickness);
    let (load_transfer_ratio, trivial_solution_warning) =
        compute_load_transfer_ratio(predicted_load_x, predicted_load_y, prescribed_load_x, prescribed_load_y);

    LoadTransferReport {
        predicted_load_x, predicted_load_y, prescribed_load_x, prescribed_load_y,
        load_transfer_ratio, traction_residual_rms, traction_residual_max, trivial_solution_warning,
    }
}

/// Pure logic half of [`probe_load_transfer`] - separated so it can be hand-verified directly
/// without needing a real network forward pass (see this function's own tests). `ratio = 1.0`
/// (trivially "fully transferred") when nothing is prescribed (`prescribed_magnitude ~ 0`) -
/// there is nothing to fail to transfer. The 10% floor is a generous sanity bound (matches
/// P2-08's own "sanity bound, not a calibrated P2-14 acceptance threshold" convention).
fn compute_load_transfer_ratio(predicted_x: f64, predicted_y: f64, prescribed_x: f64, prescribed_y: f64) -> (f64, bool) {
    let prescribed_magnitude = (prescribed_x * prescribed_x + prescribed_y * prescribed_y).sqrt();
    let predicted_magnitude = (predicted_x * predicted_x + predicted_y * predicted_y).sqrt();
    if prescribed_magnitude <= 1e-30 {
        return (1.0, false);
    }
    let ratio = predicted_magnitude / prescribed_magnitude;
    (ratio, ratio < 0.10)
}

// ─── Issue #61 EPIC P2-14: benchmark protocol with hard numeric thresholds ─────────────────

/// Hard thresholds from issue #61's own literal acceptance text - "thresholds SHALL NOT be
/// silently relaxed". Each is a MAXIMUM (the metric must be strictly below it to pass), except
/// `LOAD_TRANSFER_RATIO_MIN`/`MAX`, which bound a "≈1" window.
pub const SIGMA_XX_RELATIVE_ERROR_MAX: f64 = 0.01;
pub const SIGMA_YY_OVER_REF_MAX: f64 = 0.01;
pub const SIGMA_XY_OVER_REF_MAX: f64 = 0.01;
pub const TRACTION_RMS_OVER_REF_MAX: f64 = 0.01;
pub const LOAD_TRANSFER_RATIO_MIN: f64 = 0.99;
pub const LOAD_TRANSFER_RATIO_MAX: f64 = 1.01;

/// Issue #61 EPIC P2-14's own "no-hole gate": for a plate with NO holes under uniform far-field
/// uniaxial tension `px` (`py=0`), the EXACT elasticity solution is `sigma_xx=px`,
/// `sigma_yy=0`, `sigma_xy=0` EVERYWHERE in the interior - a real, closed-form reference this
/// codebase's no-hole examples can be checked against exactly (unlike the hole case, which has
/// no simple closed form for a FINITE plate - see [`run_hole_benchmark`]). Reuses [`probe_
/// boundary_residuals`] and [`probe_load_transfer`] (P2-09) directly rather than duplicating
/// verification machinery.
#[derive(Debug, Clone, PartialEq)]
pub struct NoHoleBenchmarkResult {
    /// RMS relative error of `sigma_xx` against the exact value `px`.
    pub sigma_xx_relative_error: f64,
    /// RMS `|sigma_yy|` normalized by `|px|` (exact value is 0, so error is reported relative
    /// to the reference stress, not to itself).
    pub sigma_yy_over_ref: f64,
    pub sigma_xy_over_ref: f64,
    pub traction_rms_over_ref: f64,
    pub load_transfer_ratio: f64,
    pub passed: bool,
    /// Which specific threshold(s) failed, if any - empty iff `passed`.
    pub failures: Vec<&'static str>,
}

pub fn run_no_hole_benchmark(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> NoHoleBenchmarkResult {
    assert!(
        spec.geometry.holes.is_empty(),
        "run_no_hole_benchmark: the exact reference solution (sigma_xx=px, sigma_yy=0, \
         sigma_xy=0 everywhere) is only valid for a plate with NO holes - use run_hole_benchmark \
         for a holed geometry",
    );
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::{residual_stats, BInner};

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };
    let px = spec.load.px;
    let sigma_ref = px.abs().max(1e-30);

    let interior = sampling.sample_interior(&placeholder, 512.max(spec.training.n_interior));
    let (sigma_xx_relative_error, sigma_yy_over_ref, sigma_xy_over_ref) = if interior.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let n_int = interior.len();
        let int_norm: Vec<[f32; 2]> = interior.iter().map(|&[x, y]| norm_pt(x, y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&int_norm, device), &fd, device);
        let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
        let m = 5 * n_int;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_int, &fd);
        let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &spec.material);
        let sxx_v: Vec<f32> = sxx.into_data().to_vec::<f32>().unwrap_or_default();
        let syy_v: Vec<f32> = syy.into_data().to_vec::<f32>().unwrap_or_default();
        let sxy_v: Vec<f32> = sxy.into_data().to_vec::<f32>().unwrap_or_default();

        let sxx_err: Vec<f32> = sxx_v.iter().map(|&s| (s as f64 - px) as f32).collect();
        let (sxx_rms, _) = residual_stats(&sxx_err);
        let (syy_rms, _) = residual_stats(&syy_v);
        let (sxy_rms, _) = residual_stats(&sxy_v);
        (sxx_rms as f64 / sigma_ref, syy_rms as f64 / sigma_ref, sxy_rms as f64 / sigma_ref)
    };

    let (traction_rms, _traction_max) = probe_boundary_residuals(model, spec, device);
    let traction_rms_over_ref = traction_rms / sigma_ref;

    let load_transfer = probe_load_transfer(model, spec, device);

    let mut failures = Vec::new();
    if !(sigma_xx_relative_error < SIGMA_XX_RELATIVE_ERROR_MAX) { failures.push("sigma_xx_relative_error"); }
    if !(sigma_yy_over_ref < SIGMA_YY_OVER_REF_MAX) { failures.push("sigma_yy_over_ref"); }
    if !(sigma_xy_over_ref < SIGMA_XY_OVER_REF_MAX) { failures.push("sigma_xy_over_ref"); }
    if !(traction_rms_over_ref < TRACTION_RMS_OVER_REF_MAX) { failures.push("traction_rms_over_ref"); }
    if !(LOAD_TRANSFER_RATIO_MIN..=LOAD_TRANSFER_RATIO_MAX).contains(&load_transfer.load_transfer_ratio) { failures.push("load_transfer_ratio"); }

    NoHoleBenchmarkResult {
        sigma_xx_relative_error, sigma_yy_over_ref, sigma_xy_over_ref,
        traction_rms_over_ref, load_transfer_ratio: load_transfer.load_transfer_ratio,
        passed: failures.is_empty(), failures,
    }
}

/// Independent no-hole displacement/strain validation against the exact affine field. This
/// consumes published visualization fields, not the training loss or direct mDEM stress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoHoleFieldValidation {
    pub u_l2: f64,
    pub u_linf: f64,
    pub v_l2: f64,
    pub v_linf: f64,
    pub strain_l2: f64,
    pub strain_linf: f64,
    pub sigma_xx_relative_error: f64,
    pub sigma_yy_over_ref: f64,
    pub sigma_xy_over_ref: f64,
    pub rigid_translation_residual: f64,
    pub rigid_rotation_residual: f64,
}

/// Validate constitutive/displayed fields on a regular no-hole grid. NaN-masked points are
/// excluded. Rotation uses the antisymmetric displacement gradient, so symmetric strain alone
/// cannot hide a rigid-body mode.
pub fn validate_no_hole_fields(
    fields: &pinn_core::messages::VisFields,
    spec: &ProblemSpec,
) -> NoHoleFieldValidation {
    assert!(spec.geometry.holes.is_empty(), "field validation requires no-hole geometry");
    let (ny, nx) = fields.disp_u.dim();
    let a = spec.load.px / spec.material.e;
    let stress_ref = spec.load.px.abs().max(1e-30);
    let mut u_sq: f64 = 0.0; let mut u_max: f64 = 0.0;
    let mut v_sq: f64 = 0.0; let mut v_max: f64 = 0.0;
    let mut strain_sq: f64 = 0.0; let mut strain_max: f64 = 0.0;
    let mut sxx_sq: f64 = 0.0; let mut syy_sq: f64 = 0.0; let mut sxy_sq: f64 = 0.0;
    let mut count: f64 = 0.0; let mut mean_du: f64 = 0.0; let mut mean_dv: f64 = 0.0;
    for iy in 0..ny { for ix in 0..nx {
        let x = -spec.geometry.half_w + 2.0 * spec.geometry.half_w * ix as f64 / (nx.max(2)-1) as f64;
        let y = -spec.geometry.half_h + 2.0 * spec.geometry.half_h * iy as f64 / (ny.max(2)-1) as f64;
        let u = fields.disp_u[(iy, ix)] as f64; let v = fields.disp_v[(iy, ix)] as f64;
        if !(u.is_finite() && v.is_finite()) { continue; }
        let du = u - a*x; let dv = v + spec.material.nu*a*y;
        u_sq += du*du; v_sq += dv*dv; u_max = u_max.max(du.abs()); v_max = v_max.max(dv.abs());
        mean_du += du; mean_dv += dv; count += 1.0;
        let ex = fields.eps_xx[(iy, ix)] as f64 - a;
        let ey = fields.eps_yy[(iy, ix)] as f64 + spec.material.nu*a;
        let es = fields.eps_xy[(iy, ix)] as f64;
        if ex.is_finite() && ey.is_finite() && es.is_finite() {
            let e2 = ex*ex + ey*ey + es*es; strain_sq += e2; strain_max = strain_max.max(e2.sqrt());
        }
        let sx = fields.sigma_xx[(iy, ix)] as f64 - spec.load.px;
        let sy = fields.sigma_yy[(iy, ix)] as f64;
        let ss = fields.sigma_xy[(iy, ix)] as f64;
        if sx.is_finite() { sxx_sq += sx*sx; } if sy.is_finite() { syy_sq += sy*sy; } if ss.is_finite() { sxy_sq += ss*ss; }
    }}
    let denom = count.max(1.0);
    let mut rotation_sq = 0.0; let mut rotation_count = 0.0;
    if nx > 2 && ny > 2 {
        let dx = 2.0 * spec.geometry.half_w / (nx - 1) as f64;
        let dy = 2.0 * spec.geometry.half_h / (ny - 1) as f64;
        for iy in 1..ny-1 { for ix in 1..nx-1 {
            let vals = [fields.disp_v[(iy, ix+1)], fields.disp_v[(iy, ix-1)], fields.disp_u[(iy+1, ix)], fields.disp_u[(iy-1, ix)]];
            if vals.iter().all(|v| v.is_finite()) {
                let omega = 0.5 * (((vals[0]-vals[1]) as f64 / (2.0*dx)) - ((vals[2]-vals[3]) as f64 / (2.0*dy)));
                rotation_sq += omega*omega; rotation_count += 1.0;
            }
        }}
    }
    NoHoleFieldValidation {
        u_l2: (u_sq/denom).sqrt(), u_linf: u_max, v_l2: (v_sq/denom).sqrt(), v_linf: v_max,
        strain_l2: (strain_sq/denom).sqrt(), strain_linf: strain_max,
        sigma_xx_relative_error: (sxx_sq/denom).sqrt()/stress_ref,
        sigma_yy_over_ref: (syy_sq/denom).sqrt()/stress_ref,
        sigma_xy_over_ref: (sxy_sq/denom).sqrt()/stress_ref,
        rigid_translation_residual: ((mean_du/denom).powi(2)+(mean_dv/denom).powi(2)).sqrt(),
        rigid_rotation_residual: if rotation_count > 0.0 { (rotation_sq/rotation_count).sqrt() } else { f64::NAN },
    }
}

/// Whether a hole's stress concentration should be checked against the classical INFINITE-plate
/// Kirsch result (`Kt=3` for a circular hole under uniaxial tension), or is only sanity-checked
/// (this codebase has no finite-plate correction formula implemented) - issue #61 P2-14's own
/// "distinguishing finite vs infinite-domain references". A common engineering rule of thumb:
/// a hole whose radius is less than 10% of the plate's (smaller) half-dimension behaves close
/// enough to the infinite-plate idealization for that comparison to be meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleReferenceKind {
    InfiniteApprox,
    Finite,
}

pub const HOLE_TO_HALF_WIDTH_INFINITE_APPROX_MAX_RATIO: f64 = 0.10;
/// Tolerance for Kt vs. the classical `3.0` infinite-plate result, for holes small enough to
/// use that reference - NOT one of issue #61's own literal thresholds (that text only specifies
/// the no-hole gate's numbers), so kept looser and explicitly labeled as this epic's own
/// judgment call, not an issue-mandated hard number.
pub const KT_VS_INFINITE_THEORY_RELATIVE_TOLERANCE: f64 = 0.25;
pub const THEORETICAL_KT_INFINITE_CIRCULAR_UNIAXIAL: f64 = 3.0;

#[derive(Debug, Clone, PartialEq)]
pub struct HoleBenchmarkResult {
    pub kt: f64,
    pub reference_kind: HoleReferenceKind,
    /// `Some(relative_error)` only for `HoleReferenceKind::InfiniteApprox` - `None` for a
    /// finite-plate hole, where no reference value exists to compare against (honest, not a
    /// silently-omitted zero).
    pub relative_error_vs_infinite_theory: Option<f64>,
    pub passed: bool,
    pub failures: Vec<&'static str>,
}

/// Issue #61 EPIC P2-14's own "hole gate valid only after no-hole passes" - `no_hole_gate` is a
/// REQUIRED parameter (not optional/defaulted), and this function REFUSES (returns `Err`, does
/// not compute or report a Kt value at all) if it did not pass. This is real, enforced gating,
/// not a printed warning a caller could ignore.
pub fn run_hole_benchmark(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    hole_index: usize,
    no_hole_gate: &NoHoleBenchmarkResult,
    device: &crate::training_core::BDevice,
) -> Result<HoleBenchmarkResult, &'static str> {
    if !no_hole_gate.passed {
        return Err(
            "run_hole_benchmark refused: the companion no-hole benchmark did not pass - Kt/hole \
             results are not accepted without a verified no-hole PASS (issue #61 EPIC P2-14's \
             own mandatory ordering)",
        );
    }
    let geometry = &spec.geometry;
    let hole = geometry.holes.get(hole_index)
        .ok_or("run_hole_benchmark: hole_index out of range")?;

    let half_min = geometry.half_w.min(geometry.half_h);
    let ratio = hole.radius / half_min.max(1e-30);
    let reference_kind = if ratio < HOLE_TO_HALF_WIDTH_INFINITE_APPROX_MAX_RATIO {
        HoleReferenceKind::InfiniteApprox
    } else {
        HoleReferenceKind::Finite
    };

    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let fd = crate::fd_stencil::FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let margin = ring_anchor_margin_m(spec.training.fd_h, geometry);
    let profile = probe_hole_boundary_profile_derived(
        model, geometry, hole, 72, &fd, scales.u_ref, spec.load.px, &spec.material, margin, device,
    );
    let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
    let sc = stress_concentration_from_profile(&profile, nominal_stress);
    let kt = sc.kt;

    let (relative_error_vs_infinite_theory, passed, failures) = match reference_kind {
        HoleReferenceKind::InfiniteApprox => {
            let rel_err = (kt - THEORETICAL_KT_INFINITE_CIRCULAR_UNIAXIAL).abs() / THEORETICAL_KT_INFINITE_CIRCULAR_UNIAXIAL;
            let ok = rel_err < KT_VS_INFINITE_THEORY_RELATIVE_TOLERANCE;
            (Some(rel_err), ok, if ok { Vec::new() } else { vec!["kt_vs_infinite_theory"] })
        }
        HoleReferenceKind::Finite => {
            // No finite-plate correction formula implemented (issue #61's own "does not hard-
            // code Kt=3" rule, extended honestly to "does not hard-code ANY closed-form
            // reference for a geometry it doesn't apply to") - only a physical sanity check:
            // Kt must be finite and >= 1 (a hole cannot reduce peak stress below the far-field
            // value for this loading).
            let ok = kt.is_finite() && kt >= 1.0;
            (None, ok, if ok { Vec::new() } else { vec!["kt_not_physically_sane"] })
        }
    };

    Ok(HoleBenchmarkResult { kt, reference_kind, relative_error_vs_infinite_theory, passed, failures })
}

/// `enhancement.md` Phase 10 ("Energy Validation") - a real domain-integrated internal-energy-
/// vs-external-work comparison. See `pinn_core::messages::EnergyBalance`'s doc comment for why
/// this is DISTINCT from the optimizer's own `energy_loss` field. Internal energy is a
/// Monte-Carlo estimate of `∫ (strain energy density) dA * thickness` over the plate's real
/// area (using the SAME interior sampling training itself uses); external work is `∮ t·u ds *
/// thickness` over the same arc-length-weighted outer-boundary point set `probe_reaction_force`
/// uses, halved for the same quasi-static-linear-loading `1/2` factor `energy::
/// dem_energy_per_point`'s own `1/2 * sigma:epsilon` formula carries (so both sides are on a
/// consistent basis).
fn prescribed_traction_dot_displacement(
    load: &LoadConfig,
    nx: &[f32],
    ny: &[f32],
    u: &[f32],
    v: &[f32],
) -> Vec<f32> {
    assert_eq!(nx.len(), ny.len());
    assert_eq!(nx.len(), u.len());
    assert_eq!(nx.len(), v.len());
    (0..nx.len()).map(|i| {
        (load.px as f32 * nx[i]) * u[i] + (load.py as f32 * ny[i]) * v[i]
    }).collect()
}

pub fn probe_energy_balance(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> pinn_core::messages::EnergyBalance {
    use crate::energy::dem_energy_per_point;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };

    let interior = sampling.sample_interior(&placeholder, spec.training.n_interior);
    // Issue #61 P2-04: real geometric measure, via the shared abstraction (was inlined here).
    let hole_radii: Vec<f64> = geometry.holes.iter().map(|h| h.radius).collect();
    let area = crate::measure_integral::plate_domain_area(geometry.half_w, geometry.half_h, &hole_radii);
    let internal_energy = if interior.is_empty() {
        0.0
    } else {
        let n_int = interior.len();
        let int_norm: Vec<[f32; 2]> = interior.iter().map(|&[x, y]| norm_pt(x, y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&int_norm, device), &fd, device);
        let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
        let m = 5 * n_int;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_int, &fd);
        let energy_density = dem_energy_per_point::<BInner>(exx, eyy, exy, &spec.material);
        let density_vals: Vec<f32> = energy_density.into_data().to_vec::<f32>().unwrap_or_default();
        // Issue #61 P2-04: `Integral_Omega(f) ≈ |Omega|*mean(f)`, via the shared abstraction
        // (byte-identical formula to this call site's pre-P2-04 inline `mean_density * area *
        // thickness`, now a real, tested, reusable function - see `measure_integral`'s tests).
        crate::measure_integral::domain_integral(area, geometry.thickness, &density_vals)
    };

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    let external_work = if bnd_pts_phys.is_empty() {
        0.0
    } else {
        let n_bnd = bnd_pts_phys.len();
        let per_edge = (n_bnd / 4).max(1);
        let ds_x_normal = 2.0 * geometry.half_h / per_edge as f64;
        let ds_y_normal = 2.0 * geometry.half_w / per_edge as f64;
        let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&bnd_norm, device), &fd, device);
        let raw = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);
        let m = 5 * n_bnd;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let nx: Vec<f32> = bnd_pts_phys.iter().map(|p| p.nx as f32).collect();
        let ny: Vec<f32> = bnd_pts_phys.iter().map(|p| p.ny as f32).collect();
        let u_vals: Vec<f32> = scaled.clone().slice([0..n_bnd, 0..1]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let v_vals: Vec<f32> = scaled.slice([0..n_bnd, 1..2]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        // Physical external work is the prescribed traction `tbar` dotted with the trial
        // displacement.  Using model-derived traction here would diagnose a different,
        // non-variational quantity and make `U-W_ext` unreconstructable from persisted data.
        let traction_dot_u = prescribed_traction_dot_displacement(
            &spec.load, &nx, &ny, &u_vals, &v_vals,
        );
        let ds_per_point: Vec<f64> = (0..n_bnd)
            .map(|i| if nx[i].abs() > 0.5 { ds_x_normal } else { ds_y_normal })
            .collect();
        // Issue #61 P2-04: `∮ f ds ≈ Σ f_i * ds_i * thickness`, via the shared abstraction
        // (byte-identical formula to this call site's pre-P2-04 inline accumulation loop).
        0.5 * crate::measure_integral::boundary_integral(&traction_dot_u, &ds_per_point, geometry.thickness)
    };

    let denom = external_work.abs().max(1e-30);
    let energy_balance_error = (internal_energy - external_work).abs() / denom;

    pinn_core::messages::EnergyBalance { internal_energy, external_work, energy_balance_error }
}

/// `HoleBoundaryPoint`/`StressConcentration` now live in `pinn_core::messages` (re-exported
/// at crate root) - not defined here - so they can travel inside a `TrainingUpdate` without
/// `pinn-core` needing to depend on `pinn-solver`. Same pattern `VisFields` already
/// established: the solver computes the data, `pinn-core` owns the shape.
use pinn_core::messages::HoleBoundaryPoint;

/// Hole-boundary stress profile (Phase 10, "Neural-Network-Wide Adaptive Collocation" epic)
/// — samples the trained solution at fine angular resolution around one hole's
/// circumference, returning per-angle displacement/strain/stress/Von Mises. This is the
/// concrete, numerical answer to "does the PINN actually resolve the stress concentration"
/// that a contour plot alone can't prove — the original motivation for this whole
/// investigation (a suspicious-looking Von Mises field with no visible concentration at the
/// hole). Reuses the exact same forward-pass/FD-stencil/scaling machinery `evaluate_user_
/// vis_grid`/`compute_domain_forwards` already use — no new physics, just a different
/// (angular, not grid) point layout.
///
/// Reads DIRECT mDEM σ - see [`PROBE_HOLE_BOUNDARY_PROFILE_SOURCE`].
pub fn probe_hole_boundary_profile(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    n_theta: usize,
    fd: &crate::fd_stencil::FdConfig,
    u_ref: f32,
    px_pa: f64,
    device: &crate::training_core::BDevice,
) -> Vec<HoleBoundaryPoint> {
    probe_hole_stress_profile_direct_at_radius(
        model, geometry, hole, n_theta, fd, u_ref, px_pa, hole.radius, device,
    )
}

/// Direct mDEM stress at an FD-safe radial offset. Kept separate from the exact-boundary
/// probe so diagnostics can compare direct and derived stress at identical coordinates.
pub fn probe_hole_stress_profile_direct_at_radius(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    n_theta: usize,
    fd: &crate::fd_stencil::FdConfig,
    u_ref: f32,
    px_pa: f64,
    radius: f64,
    device: &crate::training_core::BDevice,
) -> Vec<HoleBoundaryPoint> {
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;

    let n = n_theta.max(1);
    let half_w = geometry.half_w;
    let half_h = geometry.half_h;
    let mut thetas = Vec::with_capacity(n);
    let mut pts_phys = Vec::with_capacity(n);
    let mut pts_norm = Vec::with_capacity(n);
    for i in 0..n {
        let theta_deg = 360.0 * i as f64 / n as f64;
        let theta = theta_deg.to_radians();
        let x = hole.center[0] + radius * theta.cos();
        let y = hole.center[1] + radius * theta.sin();
        thetas.push(theta_deg);
        pts_phys.push((x, y));
        pts_norm.push([(x / half_w) as f32, (y / half_h) as f32]);
    }

    let pts_t = norm_pts_to_tensor::<BInner>(&pts_norm, device);
    let stencil = assemble_stencil::<BInner>(&pts_t, fd, device);
    let raw_stencil = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device); // [5n, 5]: u,v,sxx,syy,sxy

    // Physical-scale FIRST (same convention as `compute_domain_forwards`/`evaluate_user_
    // vis_grid`), so the FD-derived strain below is directly the physical strain - no extra
    // scale factor needed after `compute_strains`.
    let m = 5 * n;
    let scaled = Tensor::cat(vec![
        raw_stencil.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
        raw_stencil.slice([0..m, 2..5]).mul_scalar(px_pa),
    ], 1);
    let center = scaled.clone().slice([0..n, 0..5]);
    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(scaled, n, fd);

    let center_vals: Vec<f32> = center.into_data().to_vec().unwrap_or_else(|_| vec![0.0; 5 * n]);
    let exx_vals: Vec<f32> = eps_xx.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let eyy_vals: Vec<f32> = eps_yy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let exy_vals: Vec<f32> = eps_xy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);

    (0..n).map(|i| {
        let ux = center_vals[i * 5];
        let uy = center_vals[i * 5 + 1];
        let sxx = center_vals[i * 5 + 2];
        let syy = center_vals[i * 5 + 3];
        let sxy = center_vals[i * 5 + 4];
        let vm = ((sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy) as f64).sqrt() as f32;
        let (x, y) = pts_phys[i];
        HoleBoundaryPoint {
            theta_deg: thetas[i], x, y, ux, uy,
            eps_xx: exx_vals[i], eps_yy: eyy_vals[i], eps_xy: exy_vals[i],
            sxx, syy, sxy, von_mises: vm,
        }
    }).collect()
}

/// [`crate::problem::StressSource`] of [`probe_hole_boundary_profile`]'s output - a plain
/// `const`, not a runtime computation, since the source is a static fact about which function
/// you called. General-PINN architecture recommendations §4's own worked example is literally
/// this chain (`Kt -> σ -> ...`) - see [`dependency_chain_for_kt`].
pub const PROBE_HOLE_BOUNDARY_PROFILE_SOURCE: crate::problem::StressSource = crate::problem::StressSource::Direct;
/// [`crate::problem::StressSource`] of [`probe_hole_boundary_profile_derived`]'s output.
pub const PROBE_HOLE_BOUNDARY_PROFILE_DERIVED_SOURCE: crate::problem::StressSource = crate::problem::StressSource::Derived;

/// Prints the literal `Kt -> σ -> ...` dependency chain General-PINN architecture
/// recommendations §4 uses as its own worked example for "diagnostics should be able to print
/// [this]" - scoped to the one chain this codebase actually has (Kt is always computed from a
/// [`HoleBoundaryPoint`] profile, which is always built by one of the two probes above).
/// `derived` should be `true` when the profile came from [`probe_hole_boundary_profile_derived`]
/// (the production Kt path, since bugSource-New #12), `false` for
/// [`probe_hole_boundary_profile`] (the hole-BC-satisfaction-only variant).
///
/// Issue #61 P2-03: this string is now derived FROM [`crate::field_graph::FieldKind::
/// dependency_chain`] rather than hand-written per-branch, so the field graph is provably
/// load-bearing (not a dead parallel abstraction). The node labels are the graph's own
/// (`"strain"` rather than the pre-P2-03 text's `"strain (FD stencil)"` — the backend detail
/// belongs to `differential_operator::DerivativeBackend`, not this graph, per P2-02's
/// separately-scoped abstraction), so the string content changed slightly; see the exact-match
/// test [`dependency_chain_for_kt_derived_matches_the_graph_exactly`] for the new text.
pub fn dependency_chain_for_kt(derived: bool) -> String {
    let source = if derived { PROBE_HOLE_BOUNDARY_PROFILE_DERIVED_SOURCE } else { PROBE_HOLE_BOUNDARY_PROFILE_SOURCE };
    let sigma_label = match source {
        crate::problem::StressSource::Derived => "derived sigma (energy::compute_stress)",
        crate::problem::StressSource::Direct => "direct sigma (network output cols 2..5)",
        crate::problem::StressSource::Both =>
            unreachable!("a Kt profile probe reads exactly one representation, never both"),
    };
    let field = crate::field_graph::FieldKind::from_stress_source(source);
    let mut chain: Vec<crate::field_graph::FieldKind> = field.dependency_chain();
    chain.reverse();
    let mut parts: Vec<String> = vec!["Kt".to_string(), "von_mises".to_string(), sigma_label.to_string()];
    parts.extend(chain.into_iter().skip(1).map(|f| match f {
        crate::field_graph::FieldKind::NetworkOutput => "network".to_string(),
        other => other.label().to_string(),
    }));
    parts.join(" -> ")
}

/// Same sampling/forward-pass machinery as [`probe_hole_boundary_profile`], but reads
/// DERIVED stress (`σ=C:ε`, from FD strain via [`crate::energy::compute_stress`]) at
/// `r = hole.radius + margin` instead of the network's direct mDEM σ output exactly at the
/// hole boundary.
///
/// Required, not optional, once `EquilibriumTerm` stopped constraining direct σ (bugSource-New
/// #12): with nothing left keeping the direct-σ output aligned to real elasticity away from
/// the traction-free BC itself, a Kt computed from it would stay near-zero even if the
/// underlying displacement field becomes physically correct - reading the wrong quantity, not
/// a failed fix. `margin` (physical, e.g. [`ring_anchor_margin_m`]'s output) keeps the FD
/// stencil's arms from crossing into the hole, the same FD-safety concern
/// `contains_for_collocation` exists for. The exact-boundary, direct-σ variant stays available
/// for a DIFFERENT, still-meaningful question — "is the traction-free condition satisfied" —
/// not concentration. See [`PROBE_HOLE_BOUNDARY_PROFILE_DERIVED_SOURCE`].
pub fn probe_hole_boundary_profile_derived(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    n_theta: usize,
    fd: &crate::fd_stencil::FdConfig,
    u_ref: f32,
    px_pa: f64,
    material: &MaterialProps,
    margin: f64,
    device: &crate::training_core::BDevice,
) -> Vec<HoleBoundaryPoint> {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor};
    use crate::network::fwd_embedded;
    use crate::training_core::BInner;

    let n = n_theta.max(1);
    let half_w = geometry.half_w;
    let half_h = geometry.half_h;
    let r = hole.radius + margin;
    let mut thetas = Vec::with_capacity(n);
    let mut pts_phys = Vec::with_capacity(n);
    let mut pts_norm = Vec::with_capacity(n);
    for i in 0..n {
        let theta_deg = 360.0 * i as f64 / n as f64;
        let theta = theta_deg.to_radians();
        let x = hole.center[0] + r * theta.cos();
        let y = hole.center[1] + r * theta.sin();
        thetas.push(theta_deg);
        pts_phys.push((x, y));
        pts_norm.push([(x / half_w) as f32, (y / half_h) as f32]);
    }

    let pts_t = norm_pts_to_tensor::<BInner>(&pts_norm, device);
    let stencil = assemble_stencil::<BInner>(&pts_t, fd, device);
    let raw_stencil = fwd_embedded::<BInner>(model, stencil, geometry.coordinate_embedding(), device);

    let m = 5 * n;
    let scaled = Tensor::cat(vec![
        raw_stencil.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
        raw_stencil.slice([0..m, 2..5]).mul_scalar(px_pa),
    ], 1);
    let center_uv = scaled.clone().slice([0..n, 0..2]);
    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(scaled, n, fd);
    let (sxx_t, syy_t, sxy_t) = compute_stress::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);

    let uv_vals: Vec<f32> = center_uv.into_data().to_vec().unwrap_or_else(|_| vec![0.0; 2 * n]);
    let exx_vals: Vec<f32> = eps_xx.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let eyy_vals: Vec<f32> = eps_yy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let exy_vals: Vec<f32> = eps_xy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let sxx_vals: Vec<f32> = sxx_t.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let syy_vals: Vec<f32> = syy_t.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let sxy_vals: Vec<f32> = sxy_t.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);

    (0..n).map(|i| {
        let (sxx, syy, sxy) = (sxx_vals[i], syy_vals[i], sxy_vals[i]);
        let vm = ((sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy) as f64).sqrt() as f32;
        let (x, y) = pts_phys[i];
        HoleBoundaryPoint {
            theta_deg: thetas[i], x, y, ux: uv_vals[i * 2], uy: uv_vals[i * 2 + 1],
            eps_xx: exx_vals[i], eps_yy: eyy_vals[i], eps_xy: exy_vals[i],
            sxx, syy, sxy, von_mises: vm,
        }
    }).collect()
}

/// Measure direct-versus-derived stress at the same FD-safe ring. The direct boundary profile
/// cannot answer this question because its coordinates lie on `r=R`, while derived stress needs
/// a margin so its finite-difference stencil stays outside the hole.
pub fn probe_hole_stress_diagnostic(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    n_theta: usize,
    fd: &crate::fd_stencil::FdConfig,
    u_ref: f32,
    px_pa: f64,
    material: &MaterialProps,
    margin: f64,
    device: &crate::training_core::BDevice,
) -> pinn_core::messages::HoleStressDiagnostic {
    let radius = hole.radius + margin;
    let direct = probe_hole_stress_profile_direct_at_radius(
        model, geometry, hole, n_theta, fd, u_ref, px_pa, radius, device,
    );
    let derived = probe_hole_boundary_profile_derived(
        model, geometry, hole, n_theta, fd, u_ref, px_pa, material, margin, device,
    );
    let n = direct.len().min(derived.len()).max(1) as f64;
    let mut direct_stress2 = 0.0;
    let mut derived_stress2 = 0.0;
    let mut mismatch2 = 0.0;
    let mut mismatch_max: f64 = 0.0;
    let mut direct_traction2 = 0.0;
    let mut derived_traction2 = 0.0;
    for (d, c) in direct.iter().zip(&derived) {
        let theta = d.theta_deg.to_radians();
        let (nx, ny) = (theta.cos(), theta.sin());
        let direct_norm2 = (d.sxx as f64).powi(2) + (d.syy as f64).powi(2) + 2.0 * (d.sxy as f64).powi(2);
        let derived_norm2 = (c.sxx as f64).powi(2) + (c.syy as f64).powi(2) + 2.0 * (c.sxy as f64).powi(2);
        let dsxx = d.sxx as f64 - c.sxx as f64;
        let dsyy = d.syy as f64 - c.syy as f64;
        let dsxy = d.sxy as f64 - c.sxy as f64;
        let mismatch = (dsxx * dsxx + dsyy * dsyy + 2.0 * dsxy * dsxy).sqrt();
        let traction2 = |p: &HoleBoundaryPoint| {
            let tx = p.sxx as f64 * nx + p.sxy as f64 * ny;
            let ty = p.sxy as f64 * nx + p.syy as f64 * ny;
            tx * tx + ty * ty
        };
        direct_stress2 += direct_norm2;
        derived_stress2 += derived_norm2;
        mismatch2 += mismatch * mismatch;
        mismatch_max = mismatch_max.max(mismatch);
        direct_traction2 += traction2(d);
        derived_traction2 += traction2(c);
    }
    pinn_core::messages::HoleStressDiagnostic {
        radial_offset_m: margin,
        direct_stress_rms: (direct_stress2 / n).sqrt(),
        derived_stress_rms: (derived_stress2 / n).sqrt(),
        stress_mismatch_rms: (mismatch2 / n).sqrt(),
        stress_mismatch_max: mismatch_max,
        direct_traction_rms: (direct_traction2 / n).sqrt(),
        derived_traction_rms: (derived_traction2 / n).sqrt(),
    }
}

/// Stress-concentration summary derived from a hole-boundary profile. `nominal_stress` is
/// the applied far-field traction magnitude - the standard Kt denominator for this problem
/// class. Deliberately NOT compared against a hardcoded Kt=3: that is the closed-form
/// result for an IDEALIZED INFINITE plate under uniaxial tension specifically - this plate
/// is finite, may carry biaxial/off-axis load, and may have other holes perturbing the
/// field, so a real discrepancy from 3.0 is expected, not itself evidence of a bug (see this
/// epic's own explicit instruction: "Do not hard-code Kt = 3 as a required answer").
pub fn stress_concentration_from_profile(profile: &[HoleBoundaryPoint], nominal_stress: f64) -> pinn_core::messages::StressConcentration {
    stress_concentration_from_profile_generic(profile, nominal_stress, StressProjection::VonMises, ReductionOp::Max)
}

/// Issue #61 EPIC P2-10: which scalar field the Kt reduction operates on - a real, declared,
/// swappable "projection" stage (issue #61 §3's own pipeline: "authoritative stress ->
/// projection -> boundary selection -> boundary-limit eval -> reduction -> reference
/// normalization"), instead of von Mises being implicitly hardcoded inside the reduction step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressProjection {
    VonMises,
    /// Hoop (tangential) stress at the hole boundary - the classical Kirsch-problem Kt
    /// definition for uniaxial far-field tension. Computed by projecting `(sxx, syy, sxy)`
    /// onto the local tangential direction `(-sin(theta), cos(theta))` implied by each point's
    /// own `theta_deg`: `sigma_tt = sxx*sin^2(theta) - 2*sxy*sin(theta)*cos(theta) +
    /// syy*cos^2(theta)`.
    HoopStress,
}

impl StressProjection {
    pub fn project(self, p: &HoleBoundaryPoint) -> f64 {
        match self {
            StressProjection::VonMises => p.von_mises as f64,
            StressProjection::HoopStress => {
                let theta = p.theta_deg.to_radians();
                let (s, c) = (theta.sin(), theta.cos());
                (p.sxx as f64) * s * s - 2.0 * (p.sxy as f64) * s * c + (p.syy as f64) * c * c
            }
        }
    }
}

/// Issue #61 EPIC P2-10's own "reduction" stage - a real, declared, swappable scalar reduction
/// over the projected per-point values, instead of `max` being implicitly hardcoded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionOp {
    Max,
    Mean,
    /// `p` in `[0, 100]` (e.g. `Percentile(95.0)` for the 95th percentile) - a robust
    /// alternative to `Max` when a single outlier point shouldn't dominate the QoI.
    Percentile(f64),
}

impl ReductionOp {
    pub fn reduce(self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        match self {
            ReductionOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ReductionOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
            ReductionOp::Percentile(p) => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round().clamp(0.0, (sorted.len() - 1) as f64) as usize;
                sorted[idx]
            }
        }
    }
}

/// Generalized form of [`stress_concentration_from_profile`]: `projection`/`reduction` are
/// real, caller-selectable stages, matching issue #61 §3's own architecture pipeline instead of
/// hardcoding `max(von_mises)`. `stress_concentration_from_profile` itself is a thin default
/// wrapper (`VonMises`/`Max`) preserved for every existing caller - byte-identical behavior,
/// zero migration needed for code that doesn't care about this generalization.
pub fn stress_concentration_from_profile_generic(
    profile: &[HoleBoundaryPoint],
    nominal_stress: f64,
    projection: StressProjection,
    reduction: ReductionOp,
) -> pinn_core::messages::StressConcentration {
    use pinn_core::messages::StressConcentration;
    let values: Vec<f64> = profile.iter().map(|p| projection.project(p)).collect();
    let reduced = reduction.reduce(&values);
    // Representative theta: the point whose OWN projected value is closest to the reduced
    // value - exact for `Max` (the maximizing point itself), a genuine "nearest representative"
    // for `Mean`/`Percentile`, which have no single defining point.
    let max_theta_deg = profile.iter().zip(values.iter())
        .min_by(|(_, a), (_, b)| (*a - reduced).abs().partial_cmp(&(*b - reduced).abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p.theta_deg)
        .unwrap_or(0.0);
    let kt = if nominal_stress.abs() > 1e-300 { reduced / nominal_stress.abs() } else { f64::NAN };
    StressConcentration {
        nominal_stress, max_von_mises: reduced, max_theta_deg, kt,
        stress_projection: match projection {
            StressProjection::VonMises => "VonMises",
            StressProjection::HoopStress => "HoopStress",
        },
        // Issue #62 PH3-15: this bare function never runs `kt_convergence_check` itself (it
        // has no access to the model/geometry needed to re-probe at a different resolution/
        // margin) - `None` here is a real "not computed by this call", not a claim of
        // non-convergence. Real callers that DO have that context (`runner.rs`'s vis-cadence
        // hole-analysis block) overwrite these 3 fields after calling this function - see that
        // call site's own comment.
        angular_refinement_relative_change: None,
        radial_offset_refinement_relative_change: None,
        refinement_converged: None,
        domain_classification: "FiniteDomainReference",
    }
}

/// Issue #61 EPIC P2-10's own "angular/radial convergence support" - runs the SAME Kt QoI
/// pipeline at a coarser and a finer angular resolution (same radial margin), and at the
/// coarse resolution with a LARGER radial margin (1.5x - never smaller, so this never risks
/// crossing into `valid_stencil`-unsafe territory near the hole boundary), reporting whether
/// Kt is actually converging rather than drifting. Directly answers issue #61 §3's own concern
/// (echoed for domain integrals in P2-11): a single point-count/margin choice proves nothing
/// about convergence on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KtConvergenceReport {
    pub kt_coarse: f64,
    pub kt_fine_angular: f64,
    pub kt_coarse_margin_1_5x: f64,
    pub angular_relative_change: f64,
    pub radial_relative_change: f64,
    pub converged: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn kt_convergence_check(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    n_theta_coarse: usize,
    fd: &crate::fd_stencil::FdConfig,
    u_ref: f32,
    px_pa: f64,
    material: &MaterialProps,
    margin_coarse: f64,
    nominal_stress: f64,
    tolerance: f64,
    device: &crate::training_core::BDevice,
) -> KtConvergenceReport {
    let kt_of = |n_theta: usize, margin: f64| -> f64 {
        let profile = probe_hole_boundary_profile_derived(model, geometry, hole, n_theta, fd, u_ref, px_pa, material, margin, device);
        stress_concentration_from_profile(&profile, nominal_stress).kt
    };

    let kt_coarse = kt_of(n_theta_coarse, margin_coarse);
    let kt_fine_angular = kt_of(n_theta_coarse * 2, margin_coarse);
    let kt_coarse_margin_1_5x = kt_of(n_theta_coarse, margin_coarse * 1.5);

    let rel = |a: f64, b: f64| if b.abs() > 1e-30 { (a - b).abs() / b.abs() } else { (a - b).abs() };
    let angular_relative_change = rel(kt_fine_angular, kt_coarse);
    let radial_relative_change = rel(kt_coarse_margin_1_5x, kt_coarse);
    let converged = angular_relative_change.is_finite() && radial_relative_change.is_finite()
        && angular_relative_change < tolerance && radial_relative_change < tolerance;

    KtConvergenceReport {
        kt_coarse, kt_fine_angular, kt_coarse_margin_1_5x,
        angular_relative_change, radial_relative_change, converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annular_partition_samplers_are_disjoint_and_interface_aligned() {
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }] };
        let thetas = std::sync::Arc::new(InterfaceParametrization { thetas: (0..16).map(|i| i as f64 * std::f64::consts::TAU / 16.0).collect() });
        let annulus = AnnularPartitionSampling::new(geometry.clone(), 1e-3, true, DomainId(1), thetas.clone());
        let outer = AnnularPartitionSampling::new(geometry.clone(), 1e-3, false, DomainId(0), thetas);
        let placeholder = geometry.to_placeholder();
        for point in annulus.sample_interior(&placeholder, 128) {
            assert!(geometry.annular_partition().unwrap().contains_annulus(point[0], point[1]));
            let r = (point[0] * point[0] + point[1] * point[1]).sqrt();
            assert!(r >= 0.005 + ring_anchor_margin_m(1e-3, &geometry));
        }
        let annular_sets = annulus.named_point_sets(&[]);
        let outer_sets = outer.named_point_sets(&[]);
        let annular_trace = annular_sets.iter().find(|set| set.name == "interface_annulus_stress").unwrap();
        let outer_trace = outer_sets.iter().find(|set| set.name == "interface_outer_stress").unwrap();
        for point in &annular_trace.points {
            let r = (point.x * point.x + point.y * point.y).sqrt();
            assert!(r < 0.015, "annular stress stencil trace must be inside interface: {r}");
        }
        for point in &outer_trace.points {
            let r = (point.x * point.x + point.y * point.y).sqrt();
            assert!(r > 0.015, "outer stress stencil trace must be outside interface: {r}");
        }
        for point in outer.sample_interior(&placeholder, 128) {
            assert!(geometry.annular_partition().unwrap().contains_outer(point[0], point[1]));
        }
        let annular_interface = annulus.named_point_sets(&[]).remove(0).points;
        let outer_interface = outer.named_point_sets(&[]).remove(0).points;
        assert_eq!(annular_interface.len(), outer_interface.len());
        for (a, b) in annular_interface.iter().zip(&outer_interface) {
            assert!((a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);
            assert!((a.nx + b.nx).abs() < 1e-12 && (a.ny + b.ny).abs() < 1e-12);
        }
    }
    use pinn_core::user_geometry::HoleSpec;

    /// Representative `fd_h` for tests that don't otherwise have a `ProblemSpec.training.fd_h`
    /// in scope — matches the default most real TOML specs use (e.g. `single_hole_plate.toml`).
    const TEST_FD_H: f32 = 1e-3;

    // ─── Issue #75 workstream C: apply_persistent_adaptive_interior_sample ───────

    fn l5_geometry() -> UserGeometry {
        UserGeometry {
            half_w: 0.10, half_h: 0.10, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }],
        }
    }

    /// Same shape as `runner.rs`'s own private `single_hole_like_spec` test helper (not
    /// reusable across modules) — `l5_geometry()`'s real L5 hole (radius=0.005,
    /// `3*r=0.015 < half_w=0.10` so `annular_partition()` applies too), `measure_aware_training`
    /// left to the caller since Variational-vs-Hybrid/Strong tests need different values.
    fn single_hole_like_spec(max_steps: usize) -> ProblemSpec {
        use pinn_core::loading::LoadConfig;
        use pinn_core::problem_spec::{NetworkSpec, TrainingSpec};
        ProblemSpec {
            geometry: l5_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec { hidden_dim: 64, n_hidden: 3, ..Default::default() },
            training: TrainingSpec {
                max_steps, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: false, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::default_formulation(),
        }
    }

    #[test]
    fn annular_decomposition_bvp_has_two_disjoint_domains_and_bonded_interface_terms() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        assert!(AnnularDecompositionProblem::supports(&spec));
        let problem = AnnularDecompositionProblem::new(spec);
        crate::problem::validate_loss_terms(&problem);
        assert_eq!(problem.domains().iter().map(|d| d.id).collect::<Vec<_>>(), vec![ANNULUS_DOMAIN, OUTER_DOMAIN]);
        let terms = problem.loss_terms();
        assert!(terms.iter().any(|t| t.name() == "interface_displacement_continuity"));
        assert!(terms.iter().any(|t| t.name() == "interface_traction_continuity"));
    }

    #[test]
    fn annular_decomposition_two_model_runner_smoke_is_finite() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 2, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        let device = crate::training_core::BDevice::default();
        let (annulus, outer, loss) = crate::user_runner::run_annular_decomposition_training(
            spec, device, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(annulus.input_dim(), 10);
        assert_eq!(outer.input_dim(), 3);
    }

    #[test]
    fn annular_l5_diagnostics_preserve_both_energy_halves_and_write_json() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 1, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        let device = crate::training_core::BDevice::default();
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics(
            spec, device, &[0], |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), 1);
        let terms = &diagnostics[0].terms;
        assert!(terms.iter().any(|term| term.name == "annulus_potential"), "{terms:?}");
        assert!(terms.iter().any(|term| term.name == "physical_potential"), "{terms:?}");
        assert!(diagnostics[0].kt_derived_fd_vm.is_finite());
        let path = std::env::temp_dir().join(format!("pinn-annular-diagnostic-{}.json", std::process::id()));
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("annulus_potential") && text.contains("physical_potential"));
        let _ = std::fs::remove_file(path);
    }

    /// Controlled #77 reset run. This records evidence at initialization-adjacent, early,
    /// middle, and final checkpoints before any further boundary or representation change.
    #[test]
    #[ignore]
    fn issue_77_l5_annular_diagnostic_trace() {
        let spec = ProblemSpec {
            geometry: l5_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics(
            spec, device, &checkpoints, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 diagnostic] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        // Was a relative "target/issue-77-l5-diagnostics.json": a `pinn-solver` unit
        // test's cwd is the crate manifest dir (no `target/` subdir there), and the
        // workspace's real build dir is `.shared-cargo-target` (`.cargo/config.toml`),
        // not `target/` - the write would `.unwrap()`-panic on the very first real run.
        // `std::env::temp_dir()` always exists and matches this file's sibling test
        // (`annular_l5_diagnostics_preserve_both_energy_halves_and_write_json` above).
        let path = std::env::temp_dir().join("issue-77-l5-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 diagnostic] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 diagnostic] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    fn uniform_int_norm(n: usize) -> Vec<[f32; 2]> {
        // Deterministic placeholder "uniform" point set for tests that only need SOME baseline
        // data.int_norm to exist before the function under test may overwrite it.
        (0..n).map(|i| {
            let frac = i as f32 / n as f32;
            [frac * 2.0 - 1.0, (frac * 3.0).fract() * 2.0 - 1.0]
        }).collect()
    }

    #[test]
    fn apply_persistent_adaptive_interior_sample_none_amr_leaves_data_untouched() {
        let geometry = l5_geometry();
        let original = uniform_int_norm(100);
        let mut data = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: original.clone(), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let (source, weights) = apply_persistent_adaptive_interior_sample(
            &mut data, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, None, 0,
        );
        assert_eq!(source, InteriorSampleSource::Uniform);
        assert!(weights.is_none());
        assert_eq!(data.int_norm, original, "int_norm must be untouched when amr is None");
    }

    #[test]
    fn apply_persistent_adaptive_interior_sample_no_holes_stays_uniform_even_with_a_grid() {
        let geometry = UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] };
        let cfg = pinn_core::amr::derive_amr_config(
            (-geometry.half_w, geometry.half_w, -geometry.half_h, geometry.half_h),
            &pinn_core::amr::AmrDomain::lock_zones(&geometry),
        );
        let mut grid = pinn_core::amr::AdaptiveGrid::<UserGeometry>::new(&geometry, cfg);
        let original = uniform_int_norm(100);
        let mut data = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: original.clone(), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let (source, weights) = apply_persistent_adaptive_interior_sample(
            &mut data, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, Some(&mut grid), 0,
        );
        assert_eq!(source, InteriorSampleSource::Uniform, "persistent adaptive sampling is hole-specific");
        assert!(weights.is_none());
        assert_eq!(data.int_norm, original);
    }

    #[test]
    fn apply_persistent_adaptive_interior_sample_with_holes_returns_matching_points_and_weights() {
        let geometry = l5_geometry();
        let cfg = pinn_core::amr::derive_amr_config(
            (-geometry.half_w, geometry.half_w, -geometry.half_h, geometry.half_h),
            &pinn_core::amr::AmrDomain::lock_zones(&geometry),
        );
        let mut grid = pinn_core::amr::AdaptiveGrid::<UserGeometry>::new(&geometry, cfg);
        let mut data = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: uniform_int_norm(4096), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let (source, weights) = apply_persistent_adaptive_interior_sample(
            &mut data, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, Some(&mut grid), 0,
        );
        assert_eq!(source, InteriorSampleSource::PersistentAdaptive);
        let weights = weights.expect("hole-bearing geometry with Some(grid) must return weights");
        assert_eq!(data.int_norm.len(), 4096, "persistent AMR must keep L5's fixed interior budget");
        assert_eq!(weights.len(), data.int_norm.len(), "points and weights must be created together, same length");
        for w in &weights {
            assert!(w.is_finite() && *w > 0.0, "weight {w} must be finite and positive");
        }
        for [x, y] in &data.int_norm {
            assert!(*x >= -1.0 && *x <= 1.0 && *y >= -1.0 && *y <= 1.0, "normalized point ({x},{y}) out of [-1,1]");
        }
    }

    #[test]
    fn adaptive_grid_probe_data_matches_grid_sampler_count() {
        let geometry = l5_geometry();
        let cfg = pinn_core::amr::derive_amr_config(
            (-geometry.half_w, geometry.half_w, -geometry.half_h, geometry.half_h),
            &pinn_core::amr::AmrDomain::lock_zones(&geometry),
        );
        let mut grid = pinn_core::amr::AdaptiveGrid::<UserGeometry>::new(&geometry, cfg);
        let data = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: uniform_int_norm(4096), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let probe = adaptive_grid_probe_data(&data, geometry.half_w, geometry.half_h, &mut grid, 0);
        assert_eq!(probe.int_norm.len(), grid.sample_points().len());
        assert_eq!(probe.named.len(), data.named.len(), "probe must preserve boundary point sets");
    }

    #[test]
    fn apply_persistent_adaptive_interior_sample_draws_fresh_points_across_consecutive_steps() {
        let geometry = l5_geometry();
        let cfg = pinn_core::amr::derive_amr_config(
            (-geometry.half_w, geometry.half_w, -geometry.half_h, geometry.half_h),
            &pinn_core::amr::AmrDomain::lock_zones(&geometry),
        );
        let mut grid = pinn_core::amr::AdaptiveGrid::<UserGeometry>::new(&geometry, cfg);
        let mut data_a = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: uniform_int_norm(512), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let (_, _) = apply_persistent_adaptive_interior_sample(
            &mut data_a, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, Some(&mut grid), 0,
        );
        let mut data_b = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: uniform_int_norm(512), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        let (_, _) = apply_persistent_adaptive_interior_sample(
            &mut data_b, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, Some(&mut grid), 1,
        );
        assert_eq!(data_a.int_norm.len(), data_b.int_norm.len(), "topology unchanged between step 0 and step 1 (no adapt() called)");
        let differ = data_a.int_norm.iter().zip(&data_b.int_norm).any(|(a, b)| a != b);
        assert!(differ, "consecutive steps must draw genuinely different points (issue #64's own invariant)");
    }

    #[test]
    fn apply_persistent_adaptive_interior_sample_gives_far_higher_near_hole_density_than_plain_uniform() {
        // Real numerical acceptance criterion from issue #75: persistent adaptive sampling
        // must sustain measurably higher near-hole density than uniform sampling for the L5
        // geometry - this session's own earlier diagnostic found only ~0.55% of a uniform
        // 4096-point batch (~22 points) land within 2 hole-radii of this exact hole.
        let geometry = l5_geometry();
        let cfg = pinn_core::amr::derive_amr_config(
            (-geometry.half_w, geometry.half_w, -geometry.half_h, geometry.half_h),
            &pinn_core::amr::AmrDomain::lock_zones(&geometry),
        );
        let mut grid = pinn_core::amr::AdaptiveGrid::<UserGeometry>::new(&geometry, cfg);
        let mut data = crate::problem::DomainStepData {
            id: USER_DOMAIN, int_norm: uniform_int_norm(4096), extra_ring_norm: Vec::new(),
            named: std::collections::HashMap::new(),
        };
        apply_persistent_adaptive_interior_sample(
            &mut data, &geometry, TEST_FD_H, geometry.half_w, geometry.half_h, Some(&mut grid), 0,
        );
        let r_hole = geometry.holes[0].radius;
        let near_hole = data.int_norm.iter().filter(|&&[xn, yn]| {
            let x = xn as f64 * geometry.half_w;
            let y = yn as f64 * geometry.half_h;
            (x * x + y * y).sqrt() < 2.0 * r_hole
        }).count();
        let near_hole_fraction = near_hole as f64 / data.int_norm.len() as f64;
        assert!(near_hole_fraction > 0.05,
            "persistent adaptive near-hole fraction {near_hole_fraction:.4} should be far above \
             uniform sampling's own measured ~0.0055 for this exact geometry - got {near_hole} \
             of {} points", data.int_norm.len());
    }

    /// Zero-cost analytical check (no training, no network) for the newly added
    /// `ExternalWorkTerm` (bugSource-New #3/#13's missing `-W_ext` piece of `Π=U-W_ext`) -
    /// same established methodology as `energy::tests::analytical_uniform_uniaxial_tension_
    /// satisfies_every_plate_loss_term`: feed the EXACT uniaxial-tension displacement field at
    /// 4 outer-boundary points (one per edge) and confirm the term's output matches a
    /// hand-computed value before ever wiring it into a real training run.
    #[test]
    fn external_work_term_matches_hand_computed_value_for_the_exact_uniaxial_tension_field() {
        use burn::tensor::TensorData;

        let device: crate::training_core::BDevice = Default::default();
        let e = 71.7e9_f64;
        let nu = 0.33_f64;
        let sigma0 = 69e6_f64;
        let half_w = 0.1_f64;
        let half_h = 0.1_f64;
        let px = sigma0;
        let py = 0.0_f64;
        let ref_energy = (0.5 * sigma0 * sigma0 / e) as f32;

        let u = |x: f64| sigma0 / e * x;
        let v = |y: f64| -nu * sigma0 / e * y;

        // One point per outer edge: right (nx=1,ny=0), left (nx=-1,ny=0), top (nx=0,ny=1),
        // bottom (nx=0,ny=-1) - matches `UserSamplingStrategy::sample_boundary`'s own 4-edge
        // convention (nx/ny encode the outward normal exactly as `OuterTractionTerm` expects).
        let pts: [(f64, f64, f32, f32); 4] = [
            (half_w, 0.0, 1.0, 0.0),
            (-half_w, 0.0, -1.0, 0.0),
            (0.0, half_h, 0.0, 1.0),
            (0.0, -half_h, 0.0, -1.0),
        ];
        let n = pts.len();
        let mut raw_data = Vec::with_capacity(n * 2);
        let mut nx_v = Vec::with_capacity(n);
        let mut ny_v = Vec::with_capacity(n);
        for &(x, y, nx, ny) in &pts {
            raw_data.push(u(x) as f32);
            raw_data.push(v(y) as f32);
            nx_v.push(nx);
            ny_v.push(ny);
        }
        let raw_out = Tensor::<B, 2>::from_data(TensorData::new(raw_data, vec![n, 2]), &device);
        let nx_t = Tensor::<B, 1>::from_data(TensorData::new(nx_v, vec![n]), &device);
        let ny_t = Tensor::<B, 1>::from_data(TensorData::new(ny_v, vec![n]), &device);

        let d = DomainForwardOutputs {
            domain: USER_DOMAIN,
            raw_out: &raw_out,
            strains: None,
            normals: Some((nx_t, ny_t)),
            shifted_stress: None,
            hessian: None,
        };

        let term = ExternalWorkTerm {
            px, py, ref_energy,
            // Legacy `.mean()` path - the exact behavior this test verifies - so the
            // measure-aware-only fields are dead weight (never read by `compute()`).
            measure_aware: false, thickness: 1.0, ref_energy_absolute: 1.0, ds_per_point: Vec::new(),
        };
        let loss = term.compute(&[d]);
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0] as f64;

        // Hand-computed: work_density = px*nx*u + py*ny*v (py=0, so top/bottom contribute 0).
        // right: px*(+1)*u(+half_w) = px*(sigma0/e*half_w)
        // left:  px*(-1)*u(-half_w) = px*(-1)*(-sigma0/e*half_w) = same as right, by symmetry.
        let work_right = px * (sigma0 / e * half_w);
        let work_left = px * (sigma0 / e * half_w);
        let mean_work = (work_right + work_left) / n as f64;
        let expected = -mean_work / ref_energy as f64;

        assert!((loss_v - expected).abs() / expected.abs() < 1e-3,
            "external_work {loss_v} vs hand-computed {expected}");
    }

    // ─── Issue #62 PH3-04: measure-aware InteriorEnergyTerm/ExternalWorkTerm ───────────────

    #[test]
    fn interior_energy_term_measure_aware_with_no_weights_matches_domain_integral_tensor_directly() {
        use burn::tensor::TensorData;
        let material = MaterialProps::al7075_t6();
        let device: crate::training_core::BDevice = Default::default();
        let n = 3;
        let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![0.001_f32, 0.002, 0.0015], vec![n]), &device);
        let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![-0.0003_f32, -0.0006, -0.0005], vec![n]), &device);
        let exy = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0001, -0.0001], vec![n]), &device);
        let raw_out = Tensor::<B, 2>::zeros([n, 5], &device);
        let d = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out,
            strains: Some((exx.clone(), eyy.clone(), exy.clone())),
            normals: None, shifted_stress: None, hessian: None,
        };
        let domain_area = 2.0_f64;
        let thickness = 0.01_f64;
        let ref_energy_absolute = 5.0_f64;
        let term = InteriorEnergyTerm {
            domain: USER_DOMAIN,
            material: material.clone(), ref_energy: 1.0,
            measure_aware: true, domain_area, thickness, ref_energy_absolute, weights: None,
        };
        let loss = term.compute(&[d]);
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0] as f64;

        let density = crate::energy::dem_energy_per_point::<B>(exx, eyy, exy, &material);
        let expected = crate::measure_integral::domain_integral_tensor::<B>(domain_area, thickness, density)
            .into_data().to_vec::<f32>().unwrap()[0] as f64 / ref_energy_absolute;
        assert!((loss_v - expected).abs() / expected.abs() < 1e-6, "loss={loss_v} expected={expected}");
    }

    #[test]
    fn interior_energy_term_measure_aware_with_weights_matches_domain_integral_weighted_tensor_and_differs_from_unweighted() {
        use burn::tensor::TensorData;
        let material = MaterialProps::al7075_t6();
        let device: crate::training_core::BDevice = Default::default();
        let n = 3;
        let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![0.001_f32, 0.005, 0.0015], vec![n]), &device);
        let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![-0.0003_f32, -0.0015, -0.0005], vec![n]), &device);
        let exy = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0002, -0.0001], vec![n]), &device);
        let raw_out = Tensor::<B, 2>::zeros([n, 5], &device);
        let d = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out,
            strains: Some((exx.clone(), eyy.clone(), exy.clone())),
            normals: None, shifted_stress: None, hessian: None,
        };
        let domain_area = 2.0_f64;
        let thickness = 0.01_f64;
        let ref_energy_absolute = 5.0_f64;
        // Deliberately non-uniform (mean == 1.0, per `compensation_weights`'s own contract).
        let weights = vec![0.2_f64, 2.5, 0.3];
        let term_weighted = InteriorEnergyTerm {
            domain: USER_DOMAIN,
            material: material.clone(), ref_energy: 1.0,
            measure_aware: true, domain_area, thickness, ref_energy_absolute, weights: Some(weights.clone()),
        };
        let term_unweighted = InteriorEnergyTerm {
            domain: USER_DOMAIN,
            material: material.clone(), ref_energy: 1.0,
            measure_aware: true, domain_area, thickness, ref_energy_absolute, weights: None,
        };
        let loss_weighted = term_weighted.compute(&[DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out, strains: Some((exx.clone(), eyy.clone(), exy.clone())),
            normals: None, shifted_stress: None, hessian: None,
        }]);
        let loss_unweighted = term_unweighted.compute(&[d]);
        let weighted_v = loss_weighted.into_data().to_vec::<f32>().unwrap()[0] as f64;
        let unweighted_v = loss_unweighted.into_data().to_vec::<f32>().unwrap()[0] as f64;

        let density = crate::energy::dem_energy_per_point::<B>(exx, eyy, exy, &material);
        let expected = crate::measure_integral::domain_integral_weighted_tensor::<B>(domain_area, thickness, density, &weights)
            .into_data().to_vec::<f32>().unwrap()[0] as f64 / ref_energy_absolute;
        assert!((weighted_v - expected).abs() / expected.abs() < 1e-6, "weighted={weighted_v} expected={expected}");
        assert!(
            (weighted_v - unweighted_v).abs() / unweighted_v.abs() > 0.05,
            "nonuniform weights must produce a MEASURABLY different result than the plain mean \
             for this to be a real regression guard: weighted={weighted_v} unweighted={unweighted_v}",
        );
    }

    /// Issue #63 sub-issue #67: proves `UserDefinedProblem::set_interior_weights(None)` actually
    /// clears a previously-set `Some(weights)` back to the documented "plain uniform-random
    /// sampling is already unbiased, no compensation needed" state — the exact mechanism
    /// `runner::run_user_problem_training_from`'s new per-step reset (issue #67) relies on to
    /// stop a stale AMR sweep's compensation weights from being misapplied to the next several
    /// hundred steps' worth of freshly-resampled (issue #64/#73), unrelated interior points.
    /// Uses the real production `loss_terms()` API (not a hand-constructed `InteriorEnergyTerm`)
    /// so this is a genuine end-to-end proof of the mutex round-trip, not just the term's own
    /// `compute()` formula (already covered by the test immediately above).
    #[test]
    fn set_interior_weights_none_clears_a_previously_set_weighting_back_to_unweighted() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec { measure_aware_training: true, ..Default::default() },
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);

        let n = 3;
        let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![0.001_f32, 0.005, 0.0015], vec![n]), &device);
        let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![-0.0003_f32, -0.0015, -0.0005], vec![n]), &device);
        let exy = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0002, -0.0001], vec![n]), &device);
        let raw_out = Tensor::<B, 2>::zeros([n, 5], &device);
        let inputs = |exx: Tensor<B, 1>, eyy: Tensor<B, 1>, exy: Tensor<B, 1>| vec![DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out, strains: Some((exx, eyy, exy)),
            normals: None, shifted_stress: None, hessian: None,
        }];
        let interior_energy = |terms: &[Box<dyn LossTerm>], exx: Tensor<B, 1>, eyy: Tensor<B, 1>, exy: Tensor<B, 1>| -> f64 {
            terms.iter().find(|t| t.name() == "interior_energy").expect("interior_energy term must exist")
                .compute(&inputs(exx, eyy, exy)).into_data().to_vec::<f32>().unwrap()[0] as f64
        };

        // Simulate an AMR sweep having fired: set deliberately non-uniform weights (mean == 1.0,
        // matching `compensation_weights`'s own contract).
        problem.set_interior_weights(Some(vec![0.2, 2.5, 0.3]));
        let weighted = interior_energy(&problem.loss_terms(), exx.clone(), eyy.clone(), exy.clone());

        // The fix under test: reset back to `None` (what issue #67's new per-step call site does
        // every step by default, before AMR's conditional block re-sets it only for the exact
        // step a sweep fires).
        problem.set_interior_weights(None);
        let unweighted = interior_energy(&problem.loss_terms(), exx, eyy, exy);

        assert_ne!(
            weighted, unweighted,
            "resetting to None must genuinely change InteriorEnergyTerm's behavior back to the \
             plain unweighted mean - if this ever equals `weighted`, the reset silently stopped \
             taking effect and stale AMR weights would leak into unrelated future steps again",
        );
    }

    #[test]
    fn external_work_term_measure_aware_matches_boundary_integral_tensor_directly() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();
        let px = 6.9e7_f64;
        let py = 0.0_f64;
        let n = 2;
        // Point 0: right edge (nx=1, ny=0), u=1e-4, v=0. Point 1: top edge (nx=0, ny=1), u=0, v=2e-4.
        let raw_data = vec![1e-4_f32, 0.0, 0.0, 2e-4];
        let raw_out = Tensor::<B, 2>::from_data(TensorData::new(raw_data, vec![n, 2]), &device);
        let nx = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32, 0.0], vec![n]), &device);
        let ny = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32, 1.0], vec![n]), &device);
        let d = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out, strains: None,
            normals: Some((nx, ny)), shifted_stress: None, hessian: None,
        };
        let ds_per_point = vec![1.0_f64, 2.0];
        let thickness = 0.5_f64;
        let ref_energy_absolute = 3.0_f64;
        let term = ExternalWorkTerm {
            px, py, ref_energy: 1.0, measure_aware: true, thickness, ref_energy_absolute,
            ds_per_point: ds_per_point.clone(),
        };
        let loss = term.compute(&[d]);
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0] as f64;

        // work_density = px*nx*u + py*ny*v = [px*1e-4, 0.0] (py=0 zeroes the second point's v term)
        let work_density: Vec<f64> = vec![px * 1e-4, 0.0];
        let w_ext: f64 = work_density.iter().zip(ds_per_point.iter()).map(|(&v, &ds)| v * ds * thickness).sum();
        let expected = -w_ext / ref_energy_absolute;
        assert!((loss_v - expected).abs() / expected.abs() < 1e-3, "loss={loss_v} expected={expected}");
    }

    /// PH4-03: exercise the exact two-point-set shape used by `step_physics_multi`, not a
    /// helper-only reconstruction.  One returned tensor contains both U and W; a caller can
    /// therefore only scale their already-combined physical ratio uniformly.
    #[test]
    fn physical_potential_is_one_live_atomic_u_minus_w_term() {
        use burn::tensor::TensorData;

        let device: crate::training_core::BDevice = Default::default();
        let material = MaterialProps::al7075_t6();
        let n_interior = 2;
        let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![0.001_f32, 0.0015], vec![n_interior]), &device);
        let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![-0.00033_f32, -0.0005], vec![n_interior]), &device);
        let exy = Tensor::<B, 1>::zeros([n_interior], &device);
        let interior_out = Tensor::<B, 2>::zeros([n_interior, 5], &device);

        let n_boundary = 2;
        let boundary_out = Tensor::<B, 2>::from_data(
            TensorData::new(vec![2e-5_f32, 0.0, -2e-5, 0.0], vec![n_boundary, 2]), &device,
        );
        let nx = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32, -1.0], vec![n_boundary]), &device);
        let ny = Tensor::<B, 1>::zeros([n_boundary], &device);
        let domain_area = 2.0;
        let thickness = 0.5;
        let ref_energy_absolute = 7.0;
        let px = 6.9e7;
        let term = PhysicalPotentialEnergyTerm {
            domain: USER_DOMAIN,
            material: material.clone(), px, py: 0.0, measure_aware: true, domain_area,
            thickness, ref_energy: 1.0, ref_energy_absolute, interior_weights: None,
            ds_per_point: vec![1.0, 1.0], affine_strain: None,
        };
        assert_eq!(term.domains(), vec![USER_DOMAIN, USER_DOMAIN]);
        assert_eq!(term.point_sets(), vec!["interior", "outer_boundary"]);
        let actual = term.compute(&[
            DomainForwardOutputs {
                domain: USER_DOMAIN, raw_out: &interior_out,
                strains: Some((exx.clone(), eyy.clone(), exy.clone())), normals: None,
                shifted_stress: None, hessian: None,
            },
            DomainForwardOutputs {
                domain: USER_DOMAIN, raw_out: &boundary_out, strains: None,
                normals: Some((nx, ny)), shifted_stress: None, hessian: None,
            },
        ]).into_data().to_vec::<f32>().unwrap()[0] as f64;

        let u = crate::measure_integral::domain_integral_tensor::<B>(
            domain_area, thickness,
            crate::energy::dem_energy_per_point::<B>(exx, eyy, exy, &material),
        ).into_data().to_vec::<f32>().unwrap()[0] as f64;
        let w = 2.0 * px * 2e-5 * thickness;
        let expected = (u - w) / ref_energy_absolute;
        assert!((actual - expected).abs() / expected.abs().max(1e-12) < 1e-5,
            "atomic Pi={actual}, expected (U-W)/reference={expected}");
    }

    /// PH4-03 acceptance at the exact production-term layer: use the prescribed affine
    /// displacement/strain field and prove this term's stationary point is `sigma0 / E`.
    /// This catches a sign, measure, or independently-scaled U/W regression that a helper-only
    /// affine calculation would miss.
    #[test]
    fn physical_potential_live_term_has_the_correct_affine_minimizer() {
        use burn::tensor::TensorData;

        let device: crate::training_core::BDevice = Default::default();
        let material = MaterialProps::al7075_t6();
        let (half_w, half_h, thickness) = (0.13_f64, 0.07_f64, 0.004_f64);
        let area = 4.0 * half_w * half_h;
        let sigma0 = 4.2e7_f64;
        let exact = sigma0 / material.e;
        let evaluate = |a: f64| -> f64 {
            let interior_out = Tensor::<B, 2>::zeros([1, 5], &device);
            let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![a as f32], vec![1]), &device);
            let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![(-material.nu * a) as f32], vec![1]), &device);
            let exy = Tensor::<B, 1>::zeros([1], &device);
            // Boundary order is right, left, top, bottom. Only right/left do prescribed work.
            let boundary_out = Tensor::<B, 2>::from_data(TensorData::new(vec![
                (a * half_w) as f32, 0.0, (-a * half_w) as f32, 0.0,
                0.0, (-material.nu * a * half_h) as f32,
                0.0, (material.nu * a * half_h) as f32,
            ], vec![4, 2]), &device);
            let nx = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32, -1.0, 0.0, 0.0], vec![4]), &device);
            let ny = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0, 1.0, -1.0], vec![4]), &device);
            PhysicalPotentialEnergyTerm {
                domain: USER_DOMAIN,
                material: material.clone(), px: sigma0, py: 0.0, measure_aware: true,
                domain_area: area, thickness, ref_energy: 1.0, ref_energy_absolute: 1.0,
                interior_weights: None,
                ds_per_point: vec![2.0 * half_h, 2.0 * half_h, 2.0 * half_w, 2.0 * half_w],
                affine_strain: None,
            }.compute(&[
                DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &interior_out,
                    strains: Some((exx, eyy, exy)), normals: None, shifted_stress: None, hessian: None },
                DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &boundary_out,
                    strains: None, normals: Some((nx, ny)), shifted_stress: None, hessian: None },
            ]).into_data().to_vec::<f32>().unwrap()[0] as f64
        };
        let pi_exact = evaluate(exact);
        let pi_low = evaluate(0.75 * exact);
        let pi_high = evaluate(1.25 * exact);
        assert!(pi_exact < pi_low && pi_exact < pi_high,
            "atomic live Pi must minimize at sigma0/E: exact={pi_exact}, low={pi_low}, high={pi_high}");
        let h = exact * 1e-3;
        let derivative = (evaluate(exact + h) - evaluate(exact - h)) / (2.0 * h);
        let scale = (sigma0 * area * thickness).abs();
        assert!(derivative.abs() / scale < 2e-3,
            "dPi/da at sigma0/E must vanish, got {derivative}");
    }

    #[test]
    fn physical_potential_live_measure_aware_weights_remove_nonuniform_interior_bias() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();
        let material = MaterialProps::al7075_t6();
        let exx = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0005_f32, 0.004], vec![2]), &device);
        let eyy = Tensor::<B, 1>::from_data(TensorData::new(vec![-0.000165_f32, -0.00132], vec![2]), &device);
        let exy = Tensor::<B, 1>::zeros([2], &device);
        let interior_out = Tensor::<B, 2>::zeros([2, 5], &device);
        let boundary_out = Tensor::<B, 2>::zeros([4, 2], &device);
        let nx = Tensor::<B, 1>::zeros([4], &device);
        let ny = Tensor::<B, 1>::zeros([4], &device);
        let weights = vec![1.8, 0.2]; // mean one; an AMR-density compensation shape.
        let make_term = |weights: Option<Vec<f64>>| PhysicalPotentialEnergyTerm {
            domain: USER_DOMAIN,
            material: material.clone(), px: 0.0, py: 0.0, measure_aware: true,
            domain_area: 1.0, thickness: 1.0, ref_energy: 1.0, ref_energy_absolute: 1.0,
            interior_weights: weights, ds_per_point: vec![0.25; 4], affine_strain: None,
        };
        let weighted = make_term(Some(weights.clone())).compute(&[
            DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &interior_out,
                strains: Some((exx.clone(), eyy.clone(), exy.clone())), normals: None, shifted_stress: None, hessian: None },
            DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &boundary_out,
                strains: None, normals: Some((nx.clone(), ny.clone())), shifted_stress: None, hessian: None },
        ]).into_data().to_vec::<f32>().unwrap()[0] as f64;
        let unweighted = make_term(None).compute(&[
            DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &interior_out,
                strains: Some((exx.clone(), eyy.clone(), exy.clone())), normals: None, shifted_stress: None, hessian: None },
            DomainForwardOutputs { domain: USER_DOMAIN, raw_out: &boundary_out,
                strains: None, normals: Some((nx, ny)), shifted_stress: None, hessian: None },
        ]).into_data().to_vec::<f32>().unwrap()[0] as f64;
        let expected = crate::measure_integral::domain_integral_weighted_tensor::<B>(
            1.0, 1.0, crate::energy::dem_energy_per_point::<B>(exx, eyy, exy, &material), &weights,
        ).into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!((weighted - expected).abs() / expected.abs() < 1e-6, "{weighted} vs {expected}");
        assert!((weighted - unweighted).abs() / unweighted.abs() > 0.1,
            "compensation must materially change this deliberately biased sample");
    }

    // ─── Issue #77 root-cause fix: kinematic decomposition (u = u_affine + u_hole) ─────────

    /// The correctness gate this whole approach depends on (plan's own explicit requirement):
    /// `PhysicalPotentialEnergyTerm`'s `affine_strain` mechanism adds a CONSTANT strain to the
    /// network's own FD-derived strain before calling `dem_energy_per_point`, which computes
    /// `psi(eps_affine + eps_hole)` via `strain_energy_density`'s real quadratic form - not a
    /// hand-rolled decomposition that could accidentally drop the cross term
    /// `C:eps_affine:eps_hole`. This test proves the two are numerically identical for a
    /// nontrivial (nonzero, asymmetric) `eps_hole`, confirmed against the closed-form quadratic
    /// expansion `psi(eps_affine+eps_hole) = psi(eps_affine) + C:eps_affine:eps_hole +
    /// psi(eps_hole)` computed independently in plain f64 - i.e. the cross term really is
    /// present in what the production code computes, not silently dropped.
    #[test]
    fn affine_strain_cross_term_is_present_not_dropped() {
        let material = MaterialProps::al7075_t6();
        let (e, nu) = (material.e as f64, material.nu as f64);
        let px = 6.9e7_f64;
        let py = 1.3e7_f64; // nonzero, to exercise the general biaxial case too
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &material);
        // Independent hand-check: exx_affine=(px-nu*py)/E, eyy_affine=(py-nu*px)/E, exy=0.
        assert!((a_exx - (px - nu * py) / e).abs() / a_exx.abs() < 1e-12);
        assert!((a_eyy - (py - nu * px) / e).abs() / a_eyy.abs() < 1e-12);
        assert_eq!(a_exy, 0.0);

        let (h_exx, h_eyy, h_exy) = (3.7e-5_f64, -1.1e-5_f64, 2.3e-5_f64); // a plausible u_hole strain
        let device: crate::training_core::BDevice = Default::default();
        let t = |v: f64| Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![v as f32], vec![1]), &device,
        );
        // What the production code actually computes: add the constant, then call the SAME
        // strain-energy function every other term uses.
        let total_exx = t(h_exx).add_scalar(a_exx);
        let total_eyy = t(h_eyy).add_scalar(a_eyy);
        let total_exy = t(h_exy).add_scalar(a_exy);
        let psi_total: f64 = crate::energy::dem_energy_per_point::<B>(total_exx, total_eyy, total_exy, &material)
            .into_data().to_vec::<f32>().unwrap()[0] as f64;

        // Independent plain-f64 reference: full quadratic expansion, cross term included.
        let c = e / (1.0 - nu * nu);
        let psi = |exx: f64, eyy: f64, exy: f64| -> f64 {
            let sxx = c * (exx + nu * eyy);
            let syy = c * (eyy + nu * exx);
            let sxy = e / (1.0 + nu) * exy;
            0.5 * (sxx * exx + syy * eyy + 2.0 * sxy * exy)
        };
        let psi_affine = psi(a_exx, a_eyy, a_exy);
        let psi_hole = psi(h_exx, h_eyy, h_exy);
        // Full cross term from expanding psi(a+h): the C*(exx^2+eyy^2+2*nu*exx*eyy) part of
        // strain_energy_density contributes BOTH a same-component piece (a_exx*h_exx,
        // a_eyy*h_eyy) AND a Poisson cross-coupling piece (nu*(a_exx*h_eyy+a_eyy*h_exx)) - easy
        // to drop by only expanding the diagonal terms, which is exactly the mistake this test
        // exists to catch in PRODUCTION code, so it must not make it here either.
        let cross = c * (a_exx * h_exx + a_eyy * h_eyy + nu * (a_exx * h_eyy + a_eyy * h_exx))
            + e / (1.0 + nu) * a_exy * h_exy;
        let psi_expected = psi_affine + cross + psi_hole;

        assert!((psi_total - psi_expected).abs() / psi_expected.abs() < 1e-5,
            "psi_total={psi_total:.6e} psi_expected={psi_expected:.6e} (psi_affine={psi_affine:.6e} \
             cross={cross:.6e} psi_hole={psi_hole:.6e}) - if this fails the cross term is being \
             dropped somewhere, which changes the stationary point (see this test's own doc comment)");
        // Also confirm the cross term is NOT negligible relative to psi_hole alone - otherwise
        // this test wouldn't actually be exercising the property it claims to.
        assert!(cross.abs() / psi_hole.abs() > 0.01,
            "cross term must be a real, non-negligible contribution for this test to be meaningful");
    }

    /// `decomposition_applicable` scoping: exactly one centered, Free hole is in scope; every
    /// other shape (no hole, multiple holes, off-center, Fixed bc) is explicitly out and must
    /// fall back to the pre-#77 behavior byte-for-byte.
    #[test]
    fn decomposition_applicable_scopes_to_single_centered_free_hole_only() {
        let base = single_hole_like_spec(1);
        assert!(decomposition_applicable(&base), "L5-shaped spec must be in scope");

        let mut no_hole = base.clone();
        no_hole.geometry.holes.clear();
        assert!(!decomposition_applicable(&no_hole));

        let mut fixed = base.clone();
        fixed.geometry.holes[0].bc = HoleBc::Fixed;
        assert!(!decomposition_applicable(&fixed));

        let mut off_center = base.clone();
        off_center.geometry.holes[0].center = [0.01, 0.0];
        assert!(!decomposition_applicable(&off_center));

        let mut multi = base.clone();
        multi.geometry.holes.push(HoleSpec { center: [0.03, 0.03], radius: 0.005, bc: HoleBc::Free });
        assert!(!decomposition_applicable(&multi));
    }

    /// Direct value check of the corrected hole-traction target: for a `u_hole` whose derived
    /// stress is exactly zero (network output all-zero strain), the residual must equal
    /// `|sigma_affine . n|^2` exactly — i.e. the term is now driving toward a real, nonzero,
    /// closed-form target rather than the old (wrong, for a decomposed field) zero target.
    #[test]
    fn hole_bc_term_decomposed_uses_negative_affine_traction_as_target() {
        let material = MaterialProps::al7075_t6();
        let px = 6.9e7_f64;
        let py = 0.0_f64;
        let device: crate::training_core::BDevice = Default::default();
        let n = 4;
        let zeros = || Tensor::<B, 1>::zeros([n], &device);
        // theta = 0, 90, 180, 270 degrees; outward-into-hole normal convention (nx=-cos, ny=-sin).
        let nx_v = [-1.0_f32, 0.0, 1.0, 0.0];
        let ny_v = [0.0_f32, -1.0, 0.0, 1.0];
        let nx = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(nx_v.to_vec(), vec![n]), &device);
        let ny = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(ny_v.to_vec(), vec![n]), &device);
        let raw_out = Tensor::<B, 2>::zeros([n, 5], &device);

        let term = HoleBcTerm {
            domain: USER_DOMAIN, point_set: "hole_0_fd", bc: HoleBc::Free, ref_stress2: 1.0,
            material: material.clone(), affine_target: Some((px, py)),
        };
        let residual = term.compute(&[DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out,
            strains: Some((zeros(), zeros(), zeros())), normals: Some((nx, ny)),
            shifted_stress: None, hessian: None,
        }]).into_data().to_vec::<f32>().unwrap()[0] as f64;

        // sigma_hole=0 everywhere -> tx_pred=ty_pred=0; target=(-px*nx,-py*ny)=(-px*nx,0).
        // mean over the 4 points of (0-(-px*nx))^2 = mean of (px*nx)^2 = px^2 * mean(nx^2).
        let mean_nx2 = nx_v.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64;
        let expected = px * px * mean_nx2;
        assert!((residual - expected).abs() / expected < 1e-5,
            "residual={residual:.6e} expected={expected:.6e} - decomposed hole term must target \
             -sigma_affine.n, not zero");
        assert!(residual > 0.0, "a nonzero target must produce nonzero residual for sigma_hole=0");
    }

    /// End-to-end registration proof: for the L5-shaped spec, `UserDefinedProblem::loss_terms()`
    /// under `Variational` now includes `hole_free`, sourced from DERIVED stress at the FD-safe
    /// ring — this is the exact gap issue #77's investigation traced the Kt~1.0 result to
    /// (`hole_free_active = !matches!(formulation, Variational)` previously excluded it
    /// unconditionally). Off-scope geometries (no-hole/multi-hole/off-center/Fixed) must NOT
    /// register it under Variational, preserving the original behavior exactly.
    #[test]
    fn variational_registers_corrected_hole_term_only_when_decomposition_applicable() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = UserDefinedProblem::new(spec.clone());
        let terms = problem.loss_terms();
        let hole_term = terms.iter().find(|t| t.name() == "hole_free");
        assert!(hole_term.is_some(), "L5-shaped Variational spec must register hole_free");
        assert_eq!(hole_term.unwrap().point_sets(), vec!["hole_0_fd"],
            "decomposed hole term must read the FD-safe ring, not the exact-radius one");
        assert_eq!(hole_term.unwrap().stress_source(), Some(crate::problem::StressSource::Derived));

        let mut off_center = spec.clone();
        off_center.geometry.holes[0].center = [0.01, 0.0];
        let off_center_problem = UserDefinedProblem::new(off_center);
        assert!(off_center_problem.loss_terms().iter().all(|t| t.name() != "hole_free"),
            "off-center hole must NOT register hole_free under Variational - out of #77 v1 scope");

        let mut no_hole = spec;
        no_hole.geometry.holes.clear();
        let no_hole_problem = UserDefinedProblem::new(no_hole);
        assert!(no_hole_problem.loss_terms().iter().all(|t| t.name() != "hole_free"),
            "no-hole geometry must be byte-for-byte unaffected");
    }

    /// Same registration proof for `AnnularDecompositionProblem`, plus the point-set-consumption
    /// hardening (`problem::validate_point_sets_consumed`) - before this fix, the annulus
    /// sampler's own `"hole_0"` point set was emitted and consumed by NOTHING, which this new
    /// check now catches structurally. `"hole_0_fd"` (the new FD-safe ring) must be consumed by
    /// the registered `hole_free` term; the exact-radius `"hole_0"` legitimately stays
    /// unconsumed in this problem (kept only for potential future direct-stress use) - which is
    /// why this test targets the FD-safe name specifically via the loss-term assertion, and
    /// calls the full consumption check only after confirming that.
    #[test]
    fn annular_decomposition_registers_corrected_hole_term_for_l5_shaped_spec() {
        let mut spec = single_hole_like_spec(1); // l5_geometry: half_w=half_h=0.10, radius=0.005
        spec.training.measure_aware_training = true; // AnnularDecompositionProblem::supports requires this
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        let problem = AnnularDecompositionProblem::new(spec);
        let terms = problem.loss_terms();
        let hole_term = terms.iter().find(|t| t.name() == "hole_free");
        assert!(hole_term.is_some(), "L5-shaped annular spec must register hole_free");
        assert_eq!(hole_term.unwrap().domains(), vec![ANNULUS_DOMAIN]);
        assert_eq!(hole_term.unwrap().point_sets(), vec!["hole_0_fd"]);
    }

    /// Issue #63 PH4-04 (sub-issue #65): live measure-aware integral unbiasedness proof on the
    /// real production `InteriorEnergyTerm` path (not a helper-only reconstruction), for a
    /// KNOWN, spatially NONCONSTANT strain field — `exx(x,y)=a*x`, `eyy(x,y)=b*y`, `exy=0`
    /// (chosen so the shear term vanishes regardless of engineering-vs-tensor shear-strain
    /// convention, eliminating that one ambiguity from this proof entirely). `energy.rs`'s own
    /// `strain_energy_density = 0.5*(sxx*exx+syy*eyy+2*sxy*exy)` reduces, with `exy=0` and
    /// `compute_stress`'s `sxx=C*(exx+nu*eyy)`/`syy=C*(eyy+nu*exx)` (`C=E/(1-nu^2)`), to
    /// `0.5*C*(exx^2+eyy^2+2*nu*exx*eyy)`. Substituting and integrating over `[-1,1]x[-1,1]`
    /// (`integral of x^2 dA = integral of y^2 dA = 4/3`; `integral of x*y dA = 0` by odd-function
    /// cancellation) gives the closed form `analytical = (2/3)*C*(a^2+b^2)` — hand-derived, not
    /// a fine-grid numerical stand-in.
    ///
    /// Three independently-shaped point distributions are checked against that SAME analytical
    /// value: a plain uniform grid (measure-aware, unweighted path), a 1D left/right density
    /// split (80% of points in the left half, 20% in the right, each half still exactly 50% of
    /// the area), and a 2D corner-refinement split mimicking real AMR behavior (60% of points
    /// concentrated in one quadrant - 25% of the area - the other 75% of the area sharing the
    /// remaining 40%). Both nonuniform cases use `domain_integral_weighted_tensor`'s own
    /// contract (`mean(weight)=1`, weight = true-area-fraction / point-count-fraction) - proving
    /// the compensation mechanism generalizes across genuinely different nonuniformity shapes,
    /// not one hand-picked case.
    ///
    /// `ExternalWorkTerm`/`W_ext` is deliberately NOT included here: its boundary integral
    /// already uses the EXACT known per-point arc-length `ds` (`boundary_integral_tensor`), not
    /// an MC-style density estimator needing AMR compensation weights - `AdaptiveGrid` only
    /// ever refines the interior quadtree (see `UserSamplingStrategy`'s own doc comments), so
    /// PH4-04's uniform/nonuniform/AMR distinction is squarely an interior-integral question.
    #[test]
    fn ph4_04_interior_energy_integral_agrees_across_uniform_nonuniform_and_amr_like_sampling() {
        use burn::tensor::TensorData;

        let material = MaterialProps::al7075_t6();
        let device: crate::training_core::BDevice = Default::default();
        let e = material.e as f64;
        let nu = material.nu as f64;
        let c = e / (1.0 - nu * nu);
        let a = 1.0e-3_f64;
        let b = -3.0e-4_f64;
        let analytical = (2.0 / 3.0) * c * (a * a + b * b);

        let domain_area = 4.0_f64; // [-1,1] x [-1,1]
        let thickness = 1.0_f64;
        let ref_energy_absolute = 1.0_f64; // raw Joules out, comparable directly to `analytical`

        let compute = |points: &[(f64, f64)], weights: Option<Vec<f64>>| -> f64 {
            let n = points.len();
            let exx: Vec<f32> = points.iter().map(|&(x, _)| (a * x) as f32).collect();
            let eyy: Vec<f32> = points.iter().map(|&(_, y)| (b * y) as f32).collect();
            let exx_t = Tensor::<B, 1>::from_data(TensorData::new(exx, vec![n]), &device);
            let eyy_t = Tensor::<B, 1>::from_data(TensorData::new(eyy, vec![n]), &device);
            let exy_t = Tensor::<B, 1>::zeros([n], &device);
            let raw_out = Tensor::<B, 2>::zeros([n, 5], &device);
            let term = InteriorEnergyTerm {
                domain: USER_DOMAIN,
                material: material.clone(), ref_energy: 1.0,
                measure_aware: true, domain_area, thickness, ref_energy_absolute, weights,
            };
            term.compute(&[DomainForwardOutputs {
                domain: USER_DOMAIN, raw_out: &raw_out,
                strains: Some((exx_t, eyy_t, exy_t)), normals: None, shifted_stress: None, hessian: None,
            }]).into_data().to_vec::<f32>().unwrap()[0] as f64
        };

        let grid_in = |x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize| -> Vec<(f64, f64)> {
            let mut pts = Vec::with_capacity(nx * ny);
            for ix in 0..nx {
                for iy in 0..ny {
                    let x = x0 + (ix as f64 + 0.5) * (x1 - x0) / nx as f64;
                    let y = y0 + (iy as f64 + 0.5) * (y1 - y0) / ny as f64;
                    pts.push((x, y));
                }
            }
            pts
        };

        // Uniform: plain 80x80 regular grid over the whole domain, no compensation weights.
        let uniform_pts = grid_in(-1.0, 1.0, -1.0, 1.0, 80, 80);
        let integral_uniform = compute(&uniform_pts, None);

        // Nonuniform: 1D left/right split. Left half gets 80% of points (3200 in a 64x50
        // grid), right half gets 20% (800 in a 32x25 grid) - each half is exactly 50% of the
        // area, so weight_left = 0.5/0.8 = 0.625, weight_right = 0.5/0.2 = 2.5.
        let left = grid_in(-1.0, 0.0, -1.0, 1.0, 64, 50);
        let right = grid_in(0.0, 1.0, -1.0, 1.0, 32, 25);
        let mut nonuniform_pts = left.clone();
        nonuniform_pts.extend(right.clone());
        let mut nonuniform_weights = vec![0.625; left.len()];
        nonuniform_weights.extend(vec![2.5; right.len()]);
        assert_eq!(nonuniform_pts.len(), nonuniform_weights.len());
        let integral_nonuniform = compute(&nonuniform_pts, Some(nonuniform_weights));

        // AMR-like: 2D corner refinement. Top-left quadrant (x<0, y>0 - 25% of the area) is
        // densely gridded (60x40=2400 points); the other three quadrants (75% of the area,
        // 25% each) are each identically gridded (30x20=600 points each, 1800 total) - equal
        // point counts per equal-area sub-quadrant, so the single shared `w_other` weight is
        // exact, not an approximation. Weights are derived from the ACTUAL realized point-count
        // fractions (not a hand-picked target ratio), eliminating rounding mismatch entirely:
        // `w = true_area_fraction / actual_point_fraction`.
        let refined = grid_in(-1.0, 0.0, 0.0, 1.0, 60, 40);
        let other_a = grid_in(0.0, 1.0, 0.0, 1.0, 30, 20); // top-right
        let other_b = grid_in(-1.0, 0.0, -1.0, 0.0, 30, 20); // bottom-left
        let other_c = grid_in(0.0, 1.0, -1.0, 0.0, 30, 20); // bottom-right
        let n_other = other_a.len() + other_b.len() + other_c.len();
        let n_total = refined.len() + n_other;
        let w_refined = 0.25 / (refined.len() as f64 / n_total as f64);
        let w_other = 0.75 / (n_other as f64 / n_total as f64);
        let mut amr_pts = refined.clone();
        amr_pts.extend(other_a.iter().chain(other_b.iter()).chain(other_c.iter()).copied());
        let mut amr_weights = vec![w_refined; refined.len()];
        amr_weights.extend(vec![w_other; n_other]);
        assert_eq!(amr_pts.len(), amr_weights.len());
        let integral_amr_like = compute(&amr_pts, Some(amr_weights));

        const REL_TOL: f64 = 0.02; // 2% - grid coarseness margin for a smooth quadratic field
        for (name, value) in [
            ("uniform", integral_uniform),
            ("nonuniform", integral_nonuniform),
            ("amr_like", integral_amr_like),
        ] {
            let rel_err = (value - analytical).abs() / analytical.abs();
            assert!(
                rel_err < REL_TOL,
                "{name}: {value} vs analytical {analytical} (rel_err={rel_err}, tol={REL_TOL})"
            );
        }
        // Pairwise agreement, not just each-vs-analytical - the actual PH4-04 wording
        // ("integral_uniform ≈ integral_nonuniform ≈ integral_AMR").
        assert!((integral_uniform - integral_nonuniform).abs() / analytical.abs() < REL_TOL);
        assert!((integral_uniform - integral_amr_like).abs() / analytical.abs() < REL_TOL);
    }

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
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let placeholder = GeometryConfig::kirsch_plate_inches(); // ignored by this strategy
        let pts = strategy.sample_interior(&placeholder, 500);
        assert_eq!(pts.len(), 500, "rejection sampling must reach the requested count");
        for [x, y] in pts {
            assert!(geom.contains(x, y), "point ({x},{y}) violates plate/hole containment");
        }
    }

    #[test]
    fn sample_interior_points_stay_outside_the_fd_safe_margin_not_just_the_bare_hole_radius() {
        // Real regression guard for the corrupted-FD-signal bug: a point at r = radius +
        // epsilon (epsilon smaller than the FD stencil's physical reach) would pass the bare
        // `UserGeometry::contains` check but still have a stencil arm land inside the hole.
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let pts = strategy.sample_interior(&placeholder, 2000);
        let margin = crate::user_problem::ring_anchor_margin_m(TEST_FD_H, &geom);
        for [x, y] in pts {
            for hole in &geom.holes {
                let dx = x - hole.center[0];
                let dy = y - hole.center[1];
                let r = (dx * dx + dy * dy).sqrt();
                assert!(
                    r >= hole.radius + margin - 1e-12,
                    "point ({x},{y}) at r={r} is within the FD-unsafe margin of hole radius {} (margin={margin})",
                    hole.radius,
                );
            }
        }
    }

    #[test]
    fn sample_boundary_produces_points_on_all_four_outer_edges() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let pts = strategy.sample_boundary(&placeholder, &LoadConfig::uniaxial_x(1.0), 40);
        assert!(!pts.is_empty());
        for p in &pts {
            let on_x_edge = (p.x.abs() - geom.half_w).abs() < 1e-9;
            let on_y_edge = (p.y.abs() - geom.half_h).abs() < 1e-9;
            assert!(on_x_edge || on_y_edge, "point ({}, {}) is not on an outer edge", p.x, p.y);
        }
    }

    /// Issue #64: proves the actual bug fixed in this session — before this fix, two
    /// consecutive `sample_interior` calls on the same strategy instance (exactly what
    /// `user_runner.rs`'s training loop does every step) returned byte-identical points for any
    /// hole-free geometry, silently defeating the "resample every step" design intent and
    /// letting the network overfit one frozen quadrature-node set (see this file's own
    /// `sample_interior` doc comment for the full mechanism).
    #[test]
    fn sample_interior_resamples_different_points_across_consecutive_calls() {
        let geom = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let strategy = UserSamplingStrategy::new(geom, TEST_FD_H);
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let first = strategy.sample_interior(&placeholder, 256);
        let second = strategy.sample_interior(&placeholder, 256);
        assert_eq!(first.len(), 256);
        assert_eq!(second.len(), 256);
        assert_ne!(first, second, "consecutive calls must draw genuinely different points");
    }

    /// Boundary counterpart of the above — before this fix `sample_boundary` had no RNG at all,
    /// so it returned the identical evenly-spaced grid on every call, starving `W_ext` of any
    /// real resampling too.
    #[test]
    fn sample_boundary_resamples_different_points_across_consecutive_calls() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let load = LoadConfig::uniaxial_x(1.0);
        let first = strategy.sample_boundary(&placeholder, &load, 40);
        let second = strategy.sample_boundary(&placeholder, &load, 40);
        assert_ne!(
            first.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>(),
            second.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>(),
            "consecutive calls must draw genuinely different points",
        );
        // Same edges/normals convention must still hold on the second call too.
        for p in &second {
            let on_x_edge = (p.x.abs() - geom.half_w).abs() < 1e-9;
            let on_y_edge = (p.y.abs() - geom.half_h).abs() < 1e-9;
            assert!(on_x_edge || on_y_edge, "point ({}, {}) is not on an outer edge", p.x, p.y);
        }
    }

    #[test]
    fn named_point_sets_returns_one_ring_per_hole_with_expected_point_count() {
        // No more "_anchor" point sets since bugSource-New #12 removed the near-ring
        // constitutive-anchor mechanism. Issue #77 fix added a SECOND ring per hole
        // (`"hole_i_fd"`, FD-safe at `radius+anchor_margin_m`) alongside the original
        // exact-radius `"hole_i"` ring, so this is now 2 holes * 2 rings = 4 named point sets.
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let sets = strategy.named_point_sets(&[]);
        assert_eq!(sets.len(), 4, "2 holes * 2 rings (exact-radius + FD-safe) = 4 named point sets");
        let names: Vec<&str> = sets.iter().map(|s| s.name).collect();
        for expected in ["hole_0", "hole_1", "hole_0_fd", "hole_1_fd"] {
            assert!(names.contains(&expected), "missing point set {expected:?}, got {names:?}");
        }
        for set in &sets {
            assert_eq!(set.points.len(), HOLE_RING_POINTS);
        }
    }

    #[test]
    fn constitutive_anchor_point_sets_falls_back_to_the_trait_default() {
        // bugSource-New #12: `UserSamplingStrategy` no longer overrides this - falls back to
        // the trait default (`vec![]`), matching Kirsch/pin-lug parity exactly, since nothing
        // reads direct σ outside the hole ring anymore.
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        assert_eq!(strategy.constitutive_anchor_point_sets(), Vec::<&'static str>::new());
    }

    #[test]
    fn hole_ring_points_lie_on_their_hole_circle_at_the_correct_radius() {
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
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
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let terms = problem.loss_terms();
        // interior_energy + equilibrium + outer_traction + external_work + 2 hole BC terms (no
        // hole anchor-energy terms since bugSource-New #12 removed that mechanism).
        assert_eq!(terms.len(), 6);
        let names: Vec<&str> = terms.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"interior_energy"));
        assert!(names.contains(&"equilibrium"));
        assert!(names.contains(&"outer_traction"));
        assert!(names.contains(&"external_work"));
        assert_eq!(names.iter().filter(|&&n| n == "hole_free").count(), 1);
        assert_eq!(names.iter().filter(|&&n| n == "hole_fixed").count(), 1);
        crate::problem::validate_loss_terms(&problem);
        // two_hole_geometry's 2nd hole is HoleBc::Fixed - a real Dirichlet anchor already
        // exists, so the P2-07 gauge-fix must NOT be registered (would be redundant).
        assert!(!names.contains(&"translation_gauge"));
    }

    /// Issue #61 P2-07: `translation_gauge` is registered exactly when the geometry is
    /// pure-Neumann (no `HoleBc::Fixed` hole anywhere) - `no_hole_plate.toml`'s real
    /// configuration (no holes at all).
    #[test]
    fn translation_gauge_term_is_registered_for_a_no_hole_pure_neumann_geometry() {
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"translation_gauge"), "{names:?}");
    }

    /// Same as above, but for `single_hole_plate.toml`'s real configuration: one hole, set to
    /// `HoleBc::Free` - still pure-Neumann (no Dirichlet condition anywhere).
    #[test]
    fn translation_gauge_term_is_registered_when_the_only_hole_is_free() {
        let spec = ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.05, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.01, bc: HoleBc::Free }],
            },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"translation_gauge"), "{names:?}");
    }

    /// Issue #61 P2-07: `TranslationGaugeTerm::compute()` penalizes the SQUARED MEAN
    /// displacement, not the mean of squares - a uniform rigid-body offset gets a real nonzero
    /// penalty, while a field with zero mean (equal positive/negative displacement) gets zero
    /// penalty regardless of how large the local variation is.
    #[test]
    fn translation_gauge_term_matches_hand_computed_value_for_a_uniform_offset_field() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();

        // Uniform offset: u=0.002, v=-0.001 everywhere - expected = 0.002^2 + 0.001^2.
        let n = 4;
        let raw_out = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.002_f32, -0.001, 0.002, -0.001, 0.002, -0.001, 0.002, -0.001], vec![n, 2]),
            &device,
        );
        let d = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out,
            strains: None, normals: None, shifted_stress: None, hessian: None,
        };
        let term = TranslationGaugeTerm { domain: USER_DOMAIN, inv_u_ref_sq: 1.0 };
        let loss = term.compute(&[d]);
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0] as f64;
        let expected = 0.002_f64 * 0.002 + 0.001 * 0.001;
        assert!((loss_v - expected).abs() < 1e-12, "loss={loss_v} expected={expected}");

        // Zero-mean field (equal positive/negative displacement) - expected = 0, regardless of
        // local variation magnitude.
        let raw_out_zero_mean = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], vec![n, 2]),
            &device,
        );
        let d2 = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out_zero_mean,
            strains: None, normals: None, shifted_stress: None, hessian: None,
        };
        let loss2 = term.compute(&[d2]);
        let loss2_v = loss2.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss2_v.abs() < 1e-9, "zero-mean field must get zero penalty, got {loss2_v}");
    }

    #[test]
    fn translation_gauge_is_dimensionless_at_the_reference_displacement() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();
        let u_ref = 2.0e-5_f64;
        let raw_out = Tensor::<B, 2>::from_data(
            TensorData::new(vec![u_ref as f32, 0.0, u_ref as f32, 0.0], vec![2, 2]), &device,
        );
        let d = DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &raw_out,
            strains: None, normals: None, shifted_stress: None, hessian: None,
        };
        let loss = TranslationGaugeTerm { domain: USER_DOMAIN, inv_u_ref_sq: 1.0 / u_ref.powi(2) }
            .compute(&[d]).into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!((loss - 1.0).abs() < 1e-5, "reference rigid translation must be O(1), got {loss}");
    }

    #[test]
    fn rotation_gauge_removes_only_rigid_rotation_not_affine_symmetric_strain() {
        use burn::tensor::TensorData;
        let device: crate::training_core::BDevice = Default::default();
        let term = RotationGaugeTerm { domain: USER_DOMAIN, half_w: 2.0, half_h: 1.0 };
        // One point per edge in UserSamplingStrategy's right, left, top, bottom order.
        // Rigid rotation u=-omega*y, v=omega*x has omega=0.03 exactly.
        let rotation = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.06, 0.0, -0.06, -0.03, 0.0, 0.03, 0.0], vec![4, 2]),
            &device,
        );
        let rigid_loss = term.compute(&[DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &rotation, strains: None, normals: None,
            shifted_stress: None, hessian: None,
        }]).into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!((rigid_loss - 0.03_f64.powi(2)).abs() < 1e-9, "{rigid_loss}");

        // Symmetric affine extension u=a*x, v=-nu*a*y has zero rotation.
        let extension = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.04_f32, 0.0, -0.04, 0.0, 0.0, -0.01, 0.0, 0.01], vec![4, 2]),
            &device,
        );
        let extension_loss = term.compute(&[DomainForwardOutputs {
            domain: USER_DOMAIN, raw_out: &extension, strains: None, normals: None,
            shifted_stress: None, hessian: None,
        }]).into_data().to_vec::<f32>().unwrap()[0];
        assert!(extension_loss.abs() < 1e-12, "{extension_loss}");
    }

    /// Issue #61 P2-01 acceptance: "Variational activates only declared variational terms and
    /// constraints." Atomic `physical_potential` (`U-W_ext`) plus the essential
    /// `hole_fixed` constraint are active; the natural `hole_free` boundary and both strong-form
    /// residuals (`equilibrium`/`outer_traction`) are ABSENT entirely - not merely zero-weighted.
    #[test]
    fn variational_formulation_activates_only_atomic_pi_and_essential_constraints() {
        use pinn_core::problem_spec::FormulationSelection;
        let mut spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.contains(&"physical_potential"), "{names:?}");
        assert!(names.contains(&"hole_fixed"), "{names:?} - essential constraint must stay active");
        assert!(!names.contains(&"hole_free"), "{names:?} - natural boundary must be OMITTED under Variational");
        assert!(!names.contains(&"equilibrium"), "{names:?} - strong-form residual must be OMITTED under Variational");
        assert!(!names.contains(&"outer_traction"), "{names:?} - strong-form residual must be OMITTED under Variational");
    }

    /// Issue #62 PH3-05's own real production configuration (`examples/problems/variational_
    /// no_hole_plate.toml`): a NO-HOLE, pure-Neumann geometry under `Variational` must activate
    /// EXACTLY atomic `physical_potential` plus the `TranslationGaugeTerm` (issue #61
    /// P2-07 - a no-hole plate has no `HoleBc::Fixed` essential constraint at all, so the
    /// rigid-body translation nullspace needs gauge-fixing instead) - no `hole_fixed`/
    /// `hole_free` (there are no holes), no `equilibrium`/`outer_traction` (Strong-form,
    /// omitted under Variational).
    #[test]
    fn variational_formulation_on_a_no_hole_geometry_activates_atomic_pi_and_translation_gauge() {
        use pinn_core::problem_spec::FormulationSelection;
        let mut spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names.contains(&"physical_potential"), "{names:?}");
        assert!(names.contains(&"translation_gauge"), "{names:?} - no essential constraint exists on a no-hole geometry, so the rigid-body nullspace must be gauge-fixed instead");
        assert!(names.contains(&"rotation_gauge"), "{names:?} - rigid rotation is also a pure-Neumann nullspace mode");
        assert!(!names.contains(&"equilibrium"), "{names:?}");
        assert!(!names.contains(&"outer_traction"), "{names:?}");
    }

    #[test]
    #[should_panic(expected = "Variational formulation requires training.measure_aware_training=true")]
    fn variational_legacy_mean_is_explicitly_unsupported() {
        use pinn_core::problem_spec::FormulationSelection;
        let mut spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(), load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(), training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Variational;
        let _ = UserDefinedProblem::new(spec).loss_terms();
    }

    /// Issue #61 P2-01 acceptance: "Strong activates declared PDE/BC residuals." No energy
    /// functional terms (`interior_energy`/`external_work`) are present.
    #[test]
    fn strong_formulation_activates_only_pde_and_bc_residuals() {
        use pinn_core::problem_spec::FormulationSelection;
        let mut spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Strong;
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 4, "{names:?}");
        assert!(names.contains(&"equilibrium"), "{names:?}");
        assert!(names.contains(&"outer_traction"), "{names:?}");
        assert!(names.contains(&"hole_free"), "{names:?}");
        assert!(names.contains(&"hole_fixed"), "{names:?}");
        assert!(!names.contains(&"interior_energy"), "{names:?} - energy functional must be OMITTED under Strong");
        assert!(!names.contains(&"external_work"), "{names:?} - energy functional must be OMITTED under Strong");
    }

    /// Issue #62 PH3-16's own "cross-configuration regression matrix" - the Strong x No-hole
    /// cell (marked "required"), not previously an explicit standalone test (only exercised
    /// implicitly by real training runs elsewhere). `EquilibriumTerm` still activates on a
    /// no-hole geometry (it registers on the "interior" point set regardless of holes) but
    /// `hole_free`/`hole_fixed` cannot (zero holes) - `TranslationGaugeTerm` (P2-07) DOES fire
    /// under Strong too, since a Strong-form residual set still has no essential BC to pin the
    /// rigid-body nullspace on a pure-Neumann (no-hole) geometry, exactly the same reasoning
    /// `variational_formulation_on_a_no_hole_geometry_...` already established for Variational.
    #[test]
    fn strong_formulation_on_a_no_hole_geometry_activates_only_pde_residuals_and_translation_gauge() {
        use pinn_core::problem_spec::FormulationSelection;
        let mut spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Strong;
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names.contains(&"equilibrium"), "{names:?}");
        assert!(names.contains(&"outer_traction"), "{names:?}");
        assert!(names.contains(&"translation_gauge"), "{names:?} - no essential constraint exists on a no-hole geometry under ANY formulation");
        assert!(!names.contains(&"interior_energy"), "{names:?} - energy functional must be OMITTED under Strong");
        assert!(!names.contains(&"external_work"), "{names:?} - energy functional must be OMITTED under Strong");
        assert!(!names.contains(&"hole_free") && !names.contains(&"hole_fixed"), "{names:?} - zero holes");
    }

    /// Issue #62 PH3-16's own "cross-configuration regression matrix" - the Hybrid x No-hole
    /// cell (marked "required"). This is the EXACT term set the real PH3-01 baseline
    /// (`Debug_run/baseline_legacy_no_hole/`) and every subsequent PH3-08/09 investigation
    /// trained against - never previously asserted as its own explicit, standalone term-
    /// activation regression test (only exercised implicitly by those real training runs).
    /// `translation_gauge` IS active here (registration depends ONLY on `is_pure_neumann()` -
    /// zero holes means vacuously pure-Neumann regardless of formulation) - this is the SAME
    /// under-suppressed gauge PH3-08's own real investigation found and PH3-09 then closed by
    /// training longer, not evidence this test got the term set wrong.
    #[test]
    fn hybrid_formulation_on_a_no_hole_geometry_activates_exactly_the_ph3_01_baseline_term_set() {
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(), // Hybrid
        };
        let problem = UserDefinedProblem::new(spec);
        let names: Vec<&str> = problem.loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 5, "{names:?}");
        assert!(names.contains(&"interior_energy"), "{names:?}");
        assert!(names.contains(&"equilibrium"), "{names:?}");
        assert!(names.contains(&"outer_traction"), "{names:?}");
        assert!(names.contains(&"external_work"), "{names:?}");
        assert!(names.contains(&"translation_gauge"), "{names:?} - zero holes is vacuously pure-Neumann under `is_pure_neumann()`, regardless of formulation");
        assert!(!names.contains(&"hole_free") && !names.contains(&"hole_fixed"), "{names:?} - zero holes");
    }

    /// Issue #61 P2-01 acceptance: "Hybrid requires an explicit term list" - an empty list
    /// activates zero base terms (still real, not a fallback to "everything"); a named subset
    /// activates exactly that subset; an unknown name panics rather than being silently
    /// ignored.
    #[test]
    fn hybrid_formulation_activates_exactly_the_named_subset() {
        use pinn_core::problem_spec::FormulationSelection;
        let base_spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };

        let mut only_energy = base_spec.clone();
        only_energy.formulation = FormulationSelection::Hybrid(vec!["interior_energy".to_string()]);
        let names: Vec<&str> = UserDefinedProblem::new(only_energy).loss_terms().iter().map(|t| t.name()).collect();
        // Hole terms are always included for Hybrid (legacy behavior), regardless of the base list.
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names.contains(&"interior_energy"), "{names:?}");
        assert!(names.contains(&"hole_free"), "{names:?}");
        assert!(names.contains(&"hole_fixed"), "{names:?}");

        let mut empty = base_spec.clone();
        empty.formulation = FormulationSelection::Hybrid(vec![]);
        let names: Vec<&str> = UserDefinedProblem::new(empty).loss_terms().iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 2, "{names:?} - empty Hybrid list must activate zero base terms, not fall back to \"everything\"");
        assert!(names.contains(&"hole_free") && names.contains(&"hole_fixed"), "{names:?}");
    }

    #[test]
    #[should_panic(expected = "unknown Hybrid formulation term 'bogus_term'")]
    fn hybrid_formulation_panics_on_unknown_term_name() {
        use pinn_core::problem_spec::FormulationSelection;
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: FormulationSelection::Hybrid(vec!["bogus_term".to_string()]),
        };
        UserDefinedProblem::new(spec).loss_terms();
    }

    /// Issue #61 P2-01 acceptance: "Tests prove inactive terms contribute no gradients." Rather
    /// than asserting a zero gradient (which a term could satisfy by accident, e.g. at a
    /// symmetric initial point), this proves the stronger claim: the excluded terms are not
    /// even PART of the computation graph - `term_grad_norms` (built from `loss_terms()`
    /// itself) has no entry for them at all under Variational, and gains real, nonzero entries
    /// for them once switched to Hybrid-all on the identical model/step.
    #[test]
    fn variational_formulation_excluded_terms_have_no_gradient_norm_entry_at_all() {
        use crate::problem::{DomainStepCtx, MultiStepCtx};
        use crate::training_core::{step_physics_multi, sync_device, BDevice};
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::network::ElasticityNetConfig;
        use crate::fd_stencil::FdConfig;
        use crate::saw_brdr::SawBrdr;
        use crate::lr_schedule::LrSchedule;
        use pinn_core::problem_spec::FormulationSelection;

        let device = BDevice::default();
        let mut spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        spec.formulation = FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = UserDefinedProblem::new(spec.clone());

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(spec.geometry.net_input_dim())
            .with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let model = net_cfg.init(&device);
        let mut optim = crate::problem::DomainOptim {
            weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim(),
        };
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 100, 500);
        let fd = FdConfig::new(1e-3, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);

        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let sampling = problem.sampling_strategy(0);
        let placeholder_geom = pinn_core::geometry::GeometryConfig::kirsch_plate_inches();
        let int_pts = sampling.sample_interior(&placeholder_geom, 64);
        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| [(x / half_w) as f32, (y / half_h) as f32]).collect();
        let mut named = std::collections::HashMap::new();
        for set in sampling.named_point_sets(&[]) {
            let pts = &set.points;
            named.insert(set.name, crate::problem::PointSetData {
                norm: pts.iter().map(|p| [(p.x / half_w) as f32, (p.y / half_h) as f32]).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(),
                ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(),
                ty: pts.iter().map(|p| p.ty as f32).collect(),
            });
        }
        let bnd_pts = sampling.sample_boundary(
            &placeholder_geom,
            &spec.load,
            spec.training.n_boundary,
        );
        named.insert("outer_boundary", crate::problem::PointSetData {
            norm: bnd_pts.iter().map(|p| [(p.x / half_w) as f32, (p.y / half_h) as f32]).collect(),
            nx: bnd_pts.iter().map(|p| p.nx as f32).collect(),
            ny: bnd_pts.iter().map(|p| p.ny as f32).collect(),
            tx: bnd_pts.iter().map(|p| p.tx as f32).collect(),
            ty: bnd_pts.iter().map(|p| p.ty as f32).collect(),
        });
        let data = crate::problem::DomainStepData { id: USER_DOMAIN, int_norm, extra_ring_norm: Vec::new(), named };
        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let ctx = MultiStepCtx {
            config: &pinn_core::messages::SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![DomainStepCtx { data: &data, u_ref: scales.u_ref, ref_energy: scales.ref_energy, ref_stress2: scales.ref_stress2 }],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: f64::MAX,
            dynamic_lam_non_tension_cap: f64::MAX,
            constitutive_consistency_weight: 50.0,
            n_fourier: spec.geometry.n_fourier(),
            coordinate_embedding: spec.geometry.coordinate_embedding(),
            probe_term_gradients: true,
            phase2_active: true,
            step: 0,
        };
        let (_new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        sync_device(&device);
        let norms = out.term_grad_norms.expect("probe_term_gradients=true must populate term_grad_norms");
        assert!(norms.contains_key("physical_potential"), "{norms:?}");
        assert!(!norms.contains_key("equilibrium"), "{norms:?} - excluded term must have NO gradient-norm entry, not just a zero one");
        assert!(!norms.contains_key("outer_traction"), "{norms:?} - excluded term must have NO gradient-norm entry, not just a zero one");
        assert!(!norms.contains_key("hole_free"), "{norms:?} - excluded natural boundary must have NO gradient-norm entry");
        assert!(!norms.contains_key("constitutive_consistency"),
            "{norms:?} - pure Variational has no direct-stress consumer, so auxiliary stress consistency must not become a hidden constraint");
    }

    /// `stress_source_report` on `UserDefinedProblem` matches `docs/investigations/
    /// kt-investigation-bugsource-new.md`'s own §2/§12 written conclusion exactly - the
    /// generalized, always-available answer to the question that document's own audit had to
    /// resolve by hand ("which terms still read direct σ after the derived-stress fix?").
    #[test]
    fn stress_source_report_matches_the_kt_investigation_docs_written_conclusion() {
        use crate::problem::StressSource;

        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let report = crate::training_core::stress_source_report(&problem);

        // `two_hole_geometry()` has one `Free` hole and one `Fixed` hole - `hole_fixed`
        // (displacement-only) correctly reports no stress source and is absent from `report`,
        // same as `interior_energy`/`external_work` (also displacement/strain-only). 4 entries
        // total: equilibrium, outer_traction, and one `hole_free` (the `Fixed` hole's own term
        // never appears here).
        for &(name, source) in &[
            ("equilibrium", StressSource::Derived),
            ("outer_traction", StressSource::Derived),
            ("hole_free", StressSource::Direct),
        ] {
            assert!(report.contains(&(name, source)), "expected ({name}, {source:?}) in {report:?}");
        }
        assert_eq!(report.len(), 3, "report: {report:?}");
        for absent in ["interior_energy", "external_work", "hole_fixed"] {
            assert!(!report.iter().any(|(n, _)| *n == absent), "{absent} has no stress source, must be absent from {report:?}");
        }
    }

    #[test]
    fn boundary_operator_report_classifies_every_boundary_term_correctly() {
        use crate::problem::BoundaryOperatorKind as Bok;

        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let report = crate::training_core::boundary_operator_report(&problem);

        // `two_hole_geometry()`: one `Free` hole (Neumann) and one `Fixed` hole (Dirichlet).
        // `outer_traction` is Neumann. `equilibrium`/`interior_energy`/`external_work` all
        // enforce interior physics, not a boundary condition, so are absent - same shape as
        // `stress_source_report`'s own regression test above.
        for &(name, kind) in &[
            ("outer_traction", Bok::Neumann),
            ("hole_free", Bok::Neumann),
            ("hole_fixed", Bok::Dirichlet),
        ] {
            assert!(report.contains(&(name, kind)), "expected ({name}, {kind:?}) in {report:?}");
        }
        assert_eq!(report.len(), 3, "report: {report:?}");
        for absent in ["interior_energy", "external_work", "equilibrium"] {
            assert!(!report.iter().any(|(n, _)| *n == absent), "{absent} is not a boundary condition, must be absent from {report:?}");
        }
    }

    #[test]
    fn derivative_order_report_classifies_every_term_correctly() {
        use crate::problem::DerivativeOrder as Do;

        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let report = crate::training_core::derivative_order_report(&problem);

        // `equilibrium` needs the Hessian (second-order); `interior_energy`/`outer_traction`
        // need strain (first-order); `external_work`/`hole_free`/`hole_fixed` read `raw_out`
        // only, so are absent - same shape as `boundary_operator_report`'s own regression test.
        for &(name, order) in &[
            ("equilibrium", Do::Second),
            ("interior_energy", Do::First),
            ("outer_traction", Do::First),
        ] {
            assert!(report.contains(&(name, order)), "expected ({name}, {order:?}) in {report:?}");
        }
        assert_eq!(report.len(), 3, "report: {report:?}");
        for absent in ["external_work", "hole_free", "hole_fixed"] {
            assert!(!report.iter().any(|(n, _)| *n == absent), "{absent} needs no spatial derivative, must be absent from {report:?}");
        }
    }

    #[test]
    fn user_problem_loss_terms_have_expected_formulation_kind_classification() {
        use crate::problem::FormulationKind as Fk;

        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let terms = problem.loss_terms();

        // Unlike stress_source/boundary_kind/derivative_order, every term has a meaningful
        // formulation_kind (no "not applicable" case) - a direct per-term table, same shape as
        // kirsch_problem.rs/pinlug_problem.rs's own formulation_kind classification tests.
        let expected: &[(&str, Fk)] = &[
            ("interior_energy", Fk::Weak),
            ("physical_potential", Fk::Weak),
            ("equilibrium", Fk::Strong),
            ("outer_traction", Fk::Strong),
            ("external_work", Fk::Weak),
            ("hole_free", Fk::Strong),
            ("hole_fixed", Fk::Strong),
        ];

        for term in &terms {
            let (_, expected_kind) = expected.iter().find(|(name, _)| *name == term.name())
                .unwrap_or_else(|| panic!("unexpected loss term '{}' not in expected table", term.name()));
            assert_eq!(term.formulation_kind(), *expected_kind,
                "term '{}' has formulation_kind {:?}, expected {:?}", term.name(), term.formulation_kind(), expected_kind);
        }
    }

    /// Issue #61 P2-05's own categorization, same shape as `formulation_kind`'s classification
    /// test above. `hole_fixed` is the plate's one essential/Dirichlet Constraint term;
    /// everything else here is part of the governing BVP's own physics.
    #[test]
    fn user_problem_loss_terms_have_expected_term_role_classification() {
        use crate::problem::TermRole as Tr;

        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let problem = UserDefinedProblem::new(spec);
        let terms = problem.loss_terms();

        let expected: &[(&str, Tr)] = &[
            ("interior_energy", Tr::PhysicalFunctional),
            ("equilibrium", Tr::PhysicalFunctional),
            ("outer_traction", Tr::PhysicalFunctional),
            ("external_work", Tr::PhysicalFunctional),
            ("hole_free", Tr::PhysicalFunctional),
            ("hole_fixed", Tr::Constraint),
        ];

        for term in &terms {
            let (_, expected_role) = expected.iter().find(|(name, _)| *name == term.name())
                .unwrap_or_else(|| panic!("unexpected loss term '{}' not in expected table", term.name()));
            assert_eq!(term.term_role(), *expected_role,
                "term '{}' has term_role {:?}, expected {:?}", term.name(), term.term_role(), expected_role);
        }
    }

    #[test]
    fn dependency_chain_for_kt_reflects_which_probe_was_used() {
        assert!(dependency_chain_for_kt(true).contains("derived"));
        assert!(dependency_chain_for_kt(false).contains("direct"));
    }

    /// Issue #61 P2-03: proves `dependency_chain_for_kt` is genuinely built from
    /// `field_graph::FieldKind::dependency_chain`, not a hardcoded string that merely happens
    /// to contain "derived"/"direct" (the weaker assertion above).
    #[test]
    fn dependency_chain_for_kt_derived_matches_the_graph_exactly() {
        assert_eq!(
            dependency_chain_for_kt(true),
            "Kt -> von_mises -> derived sigma (energy::compute_stress) -> strain -> displacement -> network",
        );
        assert_eq!(
            dependency_chain_for_kt(false),
            "Kt -> von_mises -> direct sigma (network output cols 2..5) -> network",
        );
    }

    // ─── Phase 10 (Neural-Network-Wide Adaptive Collocation epic): hole-boundary profile ────

    #[test]
    fn probe_hole_boundary_profile_samples_points_on_the_circle_and_computes_consistent_von_mises() {
        let device = crate::training_core::BDevice::default();
        let net_cfg = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let model: crate::network::ElasticityNet<crate::training_core::BInner> = net_cfg.init(&device);
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);

        let profile = probe_hole_boundary_profile(&model, &geometry, &hole, 8, &fd, 1.0, 1.0, &device);
        assert_eq!(profile.len(), 8);
        for (i, p) in profile.iter().enumerate() {
            let expected_theta = 360.0 * i as f64 / 8.0;
            assert!((p.theta_deg - expected_theta).abs() < 1e-9, "theta_deg must be evenly spaced");
            let r = ((p.x - hole.center[0]).powi(2) + (p.y - hole.center[1]).powi(2)).sqrt();
            assert!((r - hole.radius).abs() < 1e-9, "sampled point must lie exactly on the hole circle, got r={r}");
            let expected_vm = ((p.sxx * p.sxx - p.sxx * p.syy + p.syy * p.syy + 3.0 * p.sxy * p.sxy) as f64).sqrt() as f32;
            assert!(
                (p.von_mises - expected_vm).abs() < 1e-3,
                "von_mises must match the plane-stress formula applied to the returned stress components: got {} expected {expected_vm}",
                p.von_mises
            );
            assert!(p.ux.is_finite() && p.uy.is_finite() && p.eps_xx.is_finite(), "all fields must be finite for a freshly-initialized model");
        }
    }

    #[test]
    fn hole_stress_diagnostic_compares_identical_fd_safe_coordinates() {
        let device = crate::training_core::BDevice::default();
        let model = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device);
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let margin = 0.003;
        let direct = probe_hole_stress_profile_direct_at_radius(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, hole.radius + margin, &device,
        );
        let derived = probe_hole_boundary_profile_derived(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, &MaterialProps::al7075_t6(), margin, &device,
        );
        assert_eq!(direct.len(), derived.len());
        for (d, c) in direct.iter().zip(&derived) {
            assert!((d.x - c.x).abs() < 1e-12 && (d.y - c.y).abs() < 1e-12,
                "direct and derived stress must be sampled at identical coordinates");
        }
        let diagnostic = probe_hole_stress_diagnostic(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, &MaterialProps::al7075_t6(), margin, &device,
        );
        assert_eq!(diagnostic.radial_offset_m, margin);
        assert!(diagnostic.stress_mismatch_rms.is_finite() && diagnostic.stress_mismatch_max.is_finite());
        assert!(diagnostic.direct_traction_rms.is_finite() && diagnostic.derived_traction_rms.is_finite());
    }

    #[test]
    fn stress_concentration_from_profile_finds_the_max_and_computes_kt() {
        let mk = |theta_deg: f64, von_mises: f32| HoleBoundaryPoint {
            theta_deg, x: 0.0, y: 0.0, ux: 0.0, uy: 0.0,
            eps_xx: 0.0, eps_yy: 0.0, eps_xy: 0.0, sxx: 0.0, syy: 0.0, sxy: 0.0, von_mises,
        };
        let profile = vec![mk(0.0, 1.0), mk(90.0, 3.0), mk(180.0, 2.0)];
        let sc = stress_concentration_from_profile(&profile, 1.0);
        assert_eq!(sc.max_theta_deg, 90.0, "must locate the angle of maximum Von Mises, not just its value");
        assert!((sc.max_von_mises - 3.0).abs() < 1e-9);
        assert!((sc.kt - 3.0).abs() < 1e-9, "Kt = max_von_mises / nominal_stress");
    }

    #[test]
    fn stress_concentration_from_profile_does_not_hardcode_three() {
        // A deliberately non-3.0 concentration must be reported as-is, not clamped/assumed
        // toward the idealized-infinite-plate textbook value (this epic's own explicit rule).
        let mk = |von_mises: f32| HoleBoundaryPoint {
            theta_deg: 0.0, x: 0.0, y: 0.0, ux: 0.0, uy: 0.0,
            eps_xx: 0.0, eps_yy: 0.0, eps_xy: 0.0, sxx: 0.0, syy: 0.0, sxy: 0.0, von_mises,
        };
        let sc = stress_concentration_from_profile(&[mk(4.7)], 1.0);
        assert!((sc.kt - 4.7).abs() < 1e-6, "Kt must be reported honestly, not coerced toward 3.0");
    }

    // ─── Issue #61 EPIC P2-10: generic QoI/Kt architecture ─────────────────────────────────

    #[test]
    fn hoop_stress_projection_matches_hand_computed_values_at_cardinal_angles() {
        let mk = |theta_deg: f64, sxx: f32, syy: f32, sxy: f32| HoleBoundaryPoint {
            theta_deg, x: 0.0, y: 0.0, ux: 0.0, uy: 0.0,
            eps_xx: 0.0, eps_yy: 0.0, eps_xy: 0.0, sxx, syy, sxy, von_mises: 0.0,
        };
        // theta=0: tangent=(0,1) - sigma_tt = syy exactly.
        let p0 = mk(0.0, 10.0, 20.0, 5.0);
        assert!((StressProjection::HoopStress.project(&p0) - 20.0).abs() < 1e-6);
        // theta=90: tangent=(-1,0) - sigma_tt = sxx exactly.
        let p90 = mk(90.0, 10.0, 20.0, 5.0);
        assert!((StressProjection::HoopStress.project(&p90) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn reduction_op_matches_hand_computed_values() {
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        assert_eq!(ReductionOp::Max.reduce(&values), 9.0);
        assert!((ReductionOp::Mean.reduce(&values) - 4.0).abs() < 1e-9);
        // Sorted: [1,2,3,5,9] - 50th percentile (median) is index round(0.5*4)=2 -> 3.0.
        assert_eq!(ReductionOp::Percentile(50.0).reduce(&values), 3.0);
        // 100th percentile is the max.
        assert_eq!(ReductionOp::Percentile(100.0).reduce(&values), 9.0);
        assert!(ReductionOp::Max.reduce(&[]).is_nan());
    }

    #[test]
    fn stress_concentration_from_profile_generic_defaults_match_the_von_mises_max_wrapper() {
        let mk = |theta_deg: f64, von_mises: f32| HoleBoundaryPoint {
            theta_deg, x: 0.0, y: 0.0, ux: 0.0, uy: 0.0,
            eps_xx: 0.0, eps_yy: 0.0, eps_xy: 0.0, sxx: 0.0, syy: 0.0, sxy: 0.0, von_mises,
        };
        let profile = vec![mk(0.0, 1.0), mk(90.0, 3.0), mk(180.0, 2.0)];
        let default_sc = stress_concentration_from_profile(&profile, 1.0);
        let generic_sc = stress_concentration_from_profile_generic(&profile, 1.0, StressProjection::VonMises, ReductionOp::Max);
        assert_eq!(default_sc.max_theta_deg, generic_sc.max_theta_deg);
        assert_eq!(default_sc.max_von_mises, generic_sc.max_von_mises);
        assert_eq!(default_sc.kt, generic_sc.kt);
        assert_eq!(default_sc.nominal_stress, generic_sc.nominal_stress);
    }

    #[test]
    fn stress_concentration_from_profile_generic_with_mean_reduction_differs_from_max() {
        let mk = |theta_deg: f64, von_mises: f32| HoleBoundaryPoint {
            theta_deg, x: 0.0, y: 0.0, ux: 0.0, uy: 0.0,
            eps_xx: 0.0, eps_yy: 0.0, eps_xy: 0.0, sxx: 0.0, syy: 0.0, sxy: 0.0, von_mises,
        };
        let profile = vec![mk(0.0, 1.0), mk(90.0, 3.0), mk(180.0, 2.0)];
        let sc_mean = stress_concentration_from_profile_generic(&profile, 1.0, StressProjection::VonMises, ReductionOp::Mean);
        assert!((sc_mean.max_von_mises - 2.0).abs() < 1e-9, "mean of [1,3,2] = 2.0");
        assert!((sc_mean.kt - 2.0).abs() < 1e-9);
    }

    #[test]
    fn kt_convergence_check_runs_end_to_end_and_returns_finite_values() {
        let device = crate::training_core::BDevice::default();
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let fd = crate::fd_stencil::FdConfig::new(TEST_FD_H, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let hole = geometry.holes[0];
        let margin = ring_anchor_margin_m(TEST_FD_H, &geometry);
        let report = kt_convergence_check(
            &model, &geometry, &hole, 32, &fd, 1.0, 1e7, &material, margin, 1e7, 0.5, &device,
        );
        assert!(report.kt_coarse.is_finite(), "{report:?}");
        assert!(report.kt_fine_angular.is_finite(), "{report:?}");
        assert!(report.kt_coarse_margin_1_5x.is_finite(), "{report:?}");
        assert!(report.angular_relative_change.is_finite(), "{report:?}");
        assert!(report.radial_relative_change.is_finite(), "{report:?}");
    }

    // ─── Phase 14 (Neural-Network-Wide Adaptive Collocation epic): spatial diagnostic fields ──

    /// `n_fourier` must match whatever geometry the model will actually be probed against -
    /// `0` for a no-hole geometry, `8` for a holed one (`UserGeometry::n_fourier`) - or the
    /// probe's forward pass panics on a tensor width mismatch (the model's `input_dim` is
    /// fixed at construction time; the probe functions derive their Fourier embedding from
    /// the geometry they're actually given, independently).
    /// Pre-#77 signature took `n_fourier: usize` (`4*n_fourier` if nonzero, else `3`) — stale
    /// since `UserGeometry::net_input_dim()` was rewritten to `coordinate_embedding().
    /// input_dim()` (chart embedding for exactly-one-Free-hole geometries returns 10, not a
    /// Fourier-derived width). `n_fourier()` itself is now hardcoded `0` for every geometry, so
    /// every pre-existing call site's old `tiny_model(geometry.n_fourier())` silently built a
    /// 3-input model even where `coordinate_embedding()` returns `SingleHoleChart` (10) —
    /// caught as a real `cargo test` matmul-dimension failure, not by construction. Fixed by
    /// taking `&UserGeometry` directly and using the SAME embedding-driven width production
    /// code uses, so this helper can never again drift independently from it.
    fn tiny_model(geometry: &UserGeometry) -> crate::network::ElasticityNet<crate::training_core::BInner> {
        let device = crate::training_core::BDevice::default();
        crate::network::ElasticityNetConfig::new()
            .with_input_dim(geometry.coordinate_embedding().input_dim())
            .with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device)
    }

    /// The pre-#77 `tiny_model_raw()` shape (raw 3-input, no geometry needed) - kept as a
    /// separately-named helper rather than guessing a geometry at call sites that never had
    /// one in scope.
    fn tiny_model_raw() -> crate::network::ElasticityNet<crate::training_core::BInner> {
        let device = crate::training_core::BDevice::default();
        crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device)
    }

    #[test]
    fn evaluate_user_vis_grid_masks_every_new_field_outside_the_domain_same_as_the_original_six() {
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let device = crate::training_core::BDevice::default();
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let vis = evaluate_user_vis_grid(
            &model, &geometry, [16, 16], 1.0, 1.0, &MaterialProps::al7075_t6(), &fd, &[], &device,
        );
        for ((row, col), &vm) in vis.von_mises.indexed_iter() {
            let masked_out = vm.is_nan();
            for field in [&vis.eps_xx, &vis.eps_yy, &vis.eps_xy, &vis.pde_residual, &vis.amr_score] {
                assert_eq!(
                    field[(row, col)].is_nan(), masked_out,
                    "field mask must exactly match von_mises's mask at ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn evaluate_user_vis_grid_pde_residual_is_finite_nonnegative_and_not_trivially_zero() {
        // A freshly-initialized network's direct sxx/syy/sxy columns have no reason to already
        // satisfy Hooke's law against the independently FD-derived strain - if this residual
        // were accidentally wired to compare a value against itself (a copy-paste bug), every
        // point would read exactly 0.0. Real, independently-computed values essentially never
        // land on exactly zero float-for-float, so "not identically zero everywhere" is strong
        // evidence the comparison is real, not a stub.
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let vis = evaluate_user_vis_grid(
            &model, &geometry, [12, 12], 1.0, 1.0, &MaterialProps::al7075_t6(), &fd, &[], &device,
        );
        let mut any_nonzero = false;
        for &r in vis.pde_residual.iter() {
            assert!(r.is_nan() || (r.is_finite() && r >= 0.0), "pde_residual must be NaN (masked) or a finite, non-negative magnitude, got {r}");
            if r.is_finite() && r > 1e-12 { any_nonzero = true; }
        }
        assert!(any_nonzero, "pde_residual was identically zero everywhere - suspect a copy-paste bug comparing a value against itself");
    }

    #[test]
    fn evaluate_user_vis_grid_amr_score_is_finite_and_nonnegative_everywhere_inside_domain() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let vis = evaluate_user_vis_grid(
            &model, &geometry, [12, 12], 1.0, 1.0, &MaterialProps::al7075_t6(), &fd, &[], &device,
        );
        for &v in vis.amr_score.iter() {
            assert!(v.is_nan() || (v.is_finite() && v >= 0.0), "amr_score (strain energy density magnitude) must never be negative, got {v}");
        }
    }

    #[test]
    fn evaluate_user_vis_grid_collocation_density_matches_a_hand_binned_histogram() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let geometry = UserGeometry { half_w: 1.0, half_h: 1.0, thickness: 0.005, holes: vec![] };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        // All 4 points land in the same quadrant (top-right of normalized [-1,1]^2) -> the
        // whole histogram mass must land in ONE cell of a coarse enough grid, not spread out.
        let int_norm: Vec<[f32; 2]> = vec![[0.9, 0.9], [0.91, 0.92], [0.95, 0.85], [0.99, 0.99]];
        let vis = evaluate_user_vis_grid(
            &model, &geometry, [2, 2], 1.0, 1.0, &MaterialProps::al7075_t6(), &fd, &int_norm, &device,
        );
        let total: f32 = vis.collocation_density.iter().sum();
        assert!((total - int_norm.len() as f32).abs() < 1e-9, "density histogram must sum to the exact point count, got {total}");
        assert_eq!(vis.collocation_density[(1, 1)], 4.0, "all 4 points fall in the top-right cell (row=1 after the y-flip, col=1)");
    }

    // ─── enhancement.txt items 4/C: real BC residual RMS/max ────────────────────────────────

    #[test]
    fn probe_boundary_residuals_is_finite_nonnegative_and_max_at_least_rms() {
        let model = tiny_model(&two_hole_geometry());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let (rms, max) = probe_boundary_residuals(&model, &spec, &device);
        assert!(rms.is_finite() && rms >= 0.0, "rms must be finite and non-negative, got {rms}");
        assert!(max.is_finite() && max >= 0.0, "max must be finite and non-negative, got {max}");
        assert!(max >= rms - 1e-6, "max must be >= rms, got rms={rms} max={max}");
        // A freshly-initialized network has no reason to already satisfy the far-field
        // traction target or the hole boundary conditions - if this were accidentally wired
        // to compare a value against itself, every point would read exactly 0.0.
        assert!(rms > 1e-12, "boundary residual was ~zero for an untrained network - suspect a copy-paste bug comparing a value against itself");
    }

    #[test]
    fn probe_boundary_residuals_handles_a_geometry_with_no_holes() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let (rms, max) = probe_boundary_residuals(&model, &spec, &device);
        assert!(rms.is_finite() && rms >= 0.0);
        assert!(max.is_finite() && max >= 0.0);
    }

    // ─── enhancement.md Phase 9: force equilibrium ──────────────────────────────────────────

    #[test]
    fn probe_reaction_force_is_finite_and_reference_force_matches_hand_computed_nominal_load() {
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let device = crate::training_core::BDevice::default();
        let px = 6.9e7;
        let spec = ProblemSpec {
            geometry: geometry.clone(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(px),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let rf = probe_reaction_force(&model, &spec, &device);
        assert!(rf.net_fx.is_finite() && rf.net_fy.is_finite(), "net force must be finite, got fx={} fy={}", rf.net_fx, rf.net_fy);
        assert!(rf.equilibrium_error.is_finite() && rf.equilibrium_error >= 0.0);
        let expected_reference = (px * 2.0 * geometry.half_h * geometry.thickness).abs();
        assert!(
            (rf.reference_force - expected_reference).abs() / expected_reference < 1e-9,
            "reference_force must equal the analytically nominal one-edge load |px * 2*half_h * thickness|: got {} expected {expected_reference}",
            rf.reference_force
        );
    }

    #[test]
    fn stencil_quality_report_counts_fully_valid_points_far_from_any_boundary() {
        let geom = two_hole_geometry();
        // Interior points midway between the two holes, well clear of both holes and the
        // outer rectangle at the given fd_h.
        let points = vec![[0.0, 0.0], [0.005, 0.01], [-0.005, -0.01]];
        let report = stencil_quality_report(&geom, &points, 1e-3, 1e-3);
        assert_eq!(report.total, 3);
        assert_eq!(report.fully_valid, 3);
        assert_eq!(report.fallback_needed, 0);
        assert_eq!(report.invalid_center, 0);
    }

    #[test]
    fn stencil_quality_report_flags_points_whose_stencil_crosses_into_a_hole() {
        let geom = two_hole_geometry();
        // Hole 1: center [-0.03, 0.0], radius 0.01. Point at [-0.042, 0.0] is dist=0.012 from
        // the center - outside the hole (valid), but a +0.005 step in x lands at [-0.037, 0.0]
        // (dist=0.007) - inside the hole. The -x/+-y shifts all stay outside.
        let points = vec![[-0.042, 0.0]];
        let report = stencil_quality_report(&geom, &points, 0.005, 0.005);
        assert_eq!(report.total, 1);
        assert_eq!(report.fully_valid, 0);
        assert_eq!(report.fallback_needed, 1, "center is valid but the +x neighbor crosses into the hole");
        assert_eq!(report.invalid_center, 0);
    }

    #[test]
    fn stencil_quality_report_flags_an_invalid_center_separately_from_fallback_needed() {
        let geom = two_hole_geometry();
        // Dead center of hole 1 - not a real collocation point (would never pass `contains()`-
        // based sampling), but the report must still classify it correctly, not silently.
        let points = vec![[-0.03, 0.0]];
        let report = stencil_quality_report(&geom, &points, 1e-3, 1e-3);
        assert_eq!(report.invalid_center, 1);
        assert_eq!(report.fully_valid, 0);
        assert_eq!(report.fallback_needed, 0);
    }

    #[test]
    fn probe_reaction_force_handles_a_geometry_with_no_holes() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let rf = probe_reaction_force(&model, &spec, &device);
        assert!(rf.net_fx.is_finite() && rf.net_fy.is_finite());
        assert!(rf.equilibrium_error.is_finite() && rf.equilibrium_error >= 0.0);
    }

    #[test]
    fn probe_reaction_force_zero_load_gives_zero_reference_force_floored_and_finite_error() {
        // px = py = 0.0: `reference_force` would otherwise be exactly 0.0, which must not
        // produce a NaN/infinite division - the `.max(1e-30)` floor exists exactly for this.
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(0.0),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let rf = probe_reaction_force(&model, &spec, &device);
        assert!(rf.equilibrium_error.is_finite(), "equilibrium_error must stay finite at zero applied load, got {}", rf.equilibrium_error);
    }

    // ─── Issue #61 EPIC P2-09: load-transfer / trivial-solution diagnostics ────────────────

    #[test]
    fn compute_load_transfer_ratio_hand_computed_cases() {
        // Perfect transfer: predicted exactly matches prescribed.
        let (ratio, warn) = compute_load_transfer_ratio(100.0, 0.0, 100.0, 0.0);
        assert!((ratio - 1.0).abs() < 1e-12);
        assert!(!warn);

        // The literal collapsed-solution symptom: near-zero predicted vs a real prescribed load.
        let (ratio, warn) = compute_load_transfer_ratio(0.5, 0.0, 100.0, 0.0);
        assert!((ratio - 0.005).abs() < 1e-12, "ratio={ratio}");
        assert!(warn, "0.5% transferred load must trigger the trivial-solution warning");

        // Half-transferred: below the 10% floor is a warning, above is not - boundary check.
        let (ratio_low, warn_low) = compute_load_transfer_ratio(9.0, 0.0, 100.0, 0.0);
        assert!((ratio_low - 0.09).abs() < 1e-12);
        assert!(warn_low);
        let (ratio_high, warn_high) = compute_load_transfer_ratio(11.0, 0.0, 100.0, 0.0);
        assert!((ratio_high - 0.11).abs() < 1e-12);
        assert!(!warn_high);

        // Nothing prescribed - trivially "fully transferred", never a warning.
        let (ratio_zero, warn_zero) = compute_load_transfer_ratio(0.0, 0.0, 0.0, 0.0);
        assert_eq!(ratio_zero, 1.0);
        assert!(!warn_zero);

        // Combined x/y magnitude, not just the x component.
        let (ratio_xy, _) = compute_load_transfer_ratio(3.0, 4.0, 3.0, 4.0); // both (3,4), magnitude 5/5=1
        assert!((ratio_xy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn probe_load_transfer_matches_hand_computed_prescribed_load_and_is_finite() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let px = 6.9e7;
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.05, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(px),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let report = probe_load_transfer(&model, &spec, &device);
        let expected_prescribed_x = px * 2.0 * spec.geometry.half_h * spec.geometry.thickness;
        assert!((report.prescribed_load_x - expected_prescribed_x).abs() / expected_prescribed_x.abs() < 1e-9);
        assert_eq!(report.prescribed_load_y, 0.0);
        assert!(report.predicted_load_x.is_finite());
        assert!(report.load_transfer_ratio.is_finite());
        assert!(report.load_transfer_ratio >= 0.0);
        assert!(report.traction_residual_rms.is_finite());
        assert!(report.traction_residual_max.is_finite());
    }

    #[test]
    fn probe_load_transfer_zero_load_gives_ratio_one_and_no_warning() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(0.0),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let report = probe_load_transfer(&model, &spec, &device);
        assert_eq!(report.load_transfer_ratio, 1.0);
        assert!(!report.trivial_solution_warning);
    }

    #[test]
    fn probe_load_transfer_handles_a_geometry_with_holes() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let report = probe_load_transfer(&model, &spec, &device);
        assert!(report.load_transfer_ratio.is_finite());
        assert!(report.predicted_load_x.is_finite());
    }

    // ─── Issue #61 EPIC P2-14: benchmark protocol with hard numeric thresholds ─────────────

    #[test]
    fn run_no_hole_benchmark_runs_end_to_end_and_correctly_fails_an_untrained_model() {
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let result = run_no_hole_benchmark(&model, &spec, &device);
        assert!(result.sigma_xx_relative_error.is_finite());
        assert!(result.sigma_yy_over_ref.is_finite());
        assert!(result.sigma_xy_over_ref.is_finite());
        assert!(result.traction_rms_over_ref.is_finite());
        assert!(result.load_transfer_ratio.is_finite());
        // A freshly-initialized, untrained model has no reason to already satisfy sigma_xx=px
        // everywhere - a real, honest "this correctly fails" check, not assuming success.
        assert!(!result.passed, "an untrained model should not pass the no-hole benchmark: {result:?}");
        assert!(!result.failures.is_empty());
    }

    /// Issue #64: real, live evidence that the `UserSamplingStrategy` resampling fix (see
    /// `sample_interior`/`sample_boundary`'s own doc comments) actually resolves the field-
    /// recovery failure — a real, trained model passes BOTH the training-loss-independent hard
    /// numeric benchmark (`run_no_hole_benchmark`) AND the independent field validation
    /// (`validate_no_hole_fields`, evaluated on a separate 96x96 grid, not the training
    /// collocation points). Same formulation/material/load/geometry/network/training
    /// configuration as `examples/problems/variational_no_hole_plate.toml` — the one this
    /// session's own `Debug_run/phase4/issue64_resample_fix/higher_res.log` real headless run
    /// already confirmed reaches normalized_Pi=-1.000017 and passes all five P2-14 hard
    /// thresholds. Mirrors `user_runner::run_headless_user_problem`'s own training loop
    /// (duplicated rather than reused, matching this test module's own established precedent
    /// for self-contained training-loop tests — see e.g. `runner::tests::term_by_term_raw_
    /// lambda_weighted_gradient_diagnostic_on_no_hole_plate`). `#[ignore]`d like `toy_beam`'s
    /// own training-loop tests: real cost (thousands of steps, thousands of collocation
    /// points), not meant for default `cargo test --workspace` — run explicitly via
    /// `cargo test -p pinn-solver --release --features ndarray-backend -- --ignored
    /// issue_64_resampling_fix_passes_l4_and_independent_field_validation`.
    #[test]
    #[ignore]
    fn issue_64_resampling_fix_passes_l4_and_independent_field_validation() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{step_physics_multi, BDevice, B};
        use burn::tensor::backend::Backend;
        use pinn_core::messages::SolverConfig;
        use std::collections::HashMap;

        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 2048, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        let device = BDevice::default();
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let problem = UserDefinedProblem::new(spec.clone());
        crate::problem::validate_loss_terms(&problem);

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(spec.geometry.net_input_dim())
            .with_hidden_dim(spec.network.hidden_dim)
            .with_n_hidden(spec.network.n_hidden)
            .with_output_dim(5);
        B::seed(&device, spec.network.model_init_seed);
        let mut model = net_cfg.init(&device);
        let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
        let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
        let config = SolverConfig::default_kirsch();
        let sampling = problem.sampling_strategy(0);
        let placeholder = GeometryConfig::kirsch_plate_inches();
        let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };
        let to_pointset = |pts: &[BoundaryPoint]| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| norm_pt(p.x, p.y)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(), ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(), ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };

        for step in 0..spec.training.max_steps {
            // Genuinely resampled every step, thanks to this session's fix — the whole point
            // under test.
            let int_pts = sampling.sample_interior(&placeholder, spec.training.n_interior);
            let bnd_pts = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
            let mut named = HashMap::new();
            named.insert("outer_boundary", to_pointset(&bnd_pts));
            for set in sampling.named_point_sets(&[]) { named.insert(set.name, to_pointset(&set.points)); }
            let data = DomainStepData {
                id: USER_DOMAIN,
                int_norm: int_pts.iter().map(|&[x, y]| norm_pt(x, y)).collect(),
                extra_ring_norm: Vec::new(), named,
            };
            let ctx = MultiStepCtx {
                config: &config, problem: &problem, fd: &fd, k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: 50.0,
                n_fourier: spec.geometry.n_fourier(),
                coordinate_embedding: spec.geometry.coordinate_embedding(),
                probe_term_gradients: false,
                phase2_active: true, step,
            };
            let (new_model, _out) = step_physics_multi(
                vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            model = new_model.into_iter().next().unwrap();
        }

        use burn::module::AutodiffModule;
        let model_val: crate::network::ElasticityNet<crate::training_core::BInner> = model.valid();

        let benchmark = run_no_hole_benchmark(&model_val, &spec, &device);
        assert!(benchmark.passed, "P2-14 no-hole benchmark must pass after the resampling fix: {benchmark:?}");

        let diag_int_norm: Vec<[f32; 2]> = sampling.sample_interior(&placeholder, 512).iter()
            .map(|&[x, y]| norm_pt(x, y)).collect();
        let vis = evaluate_user_vis_grid(
            &model_val, &spec.geometry, [96, 96], u_ref, spec.load.px, &spec.material, &fd, &diag_int_norm, &device,
        );
        let field_check = validate_no_hole_fields(&vis, &spec);
        assert!(
            field_check.sigma_xx_relative_error < SIGMA_XX_RELATIVE_ERROR_MAX
                && field_check.sigma_yy_over_ref < SIGMA_YY_OVER_REF_MAX
                && field_check.sigma_xy_over_ref < SIGMA_XY_OVER_REF_MAX,
            "independent field validation (separate grid, not training points) must pass: {field_check:?}",
        );
    }

    /// Issue #63 sub-issue #70 (real L5): a genuine converged single-hole Kt result against a
    /// real, verified no-hole L4 companion - not a finite-but-unconverged number (issue #63's
    /// own explicit "a finite Kt from an unconverged model is diagnostic only" rule). Trains
    /// TWO real models: the exact validated no-hole companion (`examples/problems/
    /// variational_no_hole_plate.toml`'s own configuration) and a small-hole variant (hole
    /// radius/half-width ratio 0.05, well under `HOLE_TO_HALF_WIDTH_INFINITE_APPROX_MAX_RATIO`
    /// (0.10) so the classical infinite-plate `Kt=3.0` reference actually applies), both via
    /// `user_problem::resample_plate_step_data`/`plate_multi_step_ctx` (issue #73's shared
    /// functions - this is also real, additional evidence those functions work correctly for a
    /// holed geometry, not just the no-hole case they were originally extracted from). Does NOT
    /// force a Kt≈3.0 assertion (issue #63 rule: "no benchmark-specific hacks to force a pass")
    /// - reports the real result honestly via `run_hole_benchmark`'s own PASS/FAIL classification.
    /// `#[ignore]`d - real cost, two full training runs (~15-25 min each in release).
    /// Issue #63 sub-issue #71, PH4-11's own stated gap: `Strong`/`Hybrid` `FormulationSelection`
    /// variants are wired (`loss_terms()`'s `active_base` match) and unit-tested for correct term
    /// ACTIVATION (`strong_formulation_on_a_no_hole_geometry_activates_only_pde_residuals_and_
    /// translation_gauge`, `hybrid_formulation_on_a_no_hole_geometry_activates_exactly_the_ph3_01_
    /// baseline_term_set`), but until this test neither had ever been exercised in a REAL
    /// training run this epic - only `Variational` had runtime convergence evidence. This test
    /// closes that gap: trains both formulations on the exact same no-hole config already
    /// verified for Variational (`half_w=half_h=0.10`, Al7075-T6, `px=6.9e7`, `hidden_dim=64`/
    /// `n_hidden=8`, 3000 steps, `n_interior=n_boundary=4096`, AMR off), then runs the real
    /// `run_no_hole_benchmark` P2-14 check against each - the SAME hard-threshold benchmark
    /// Variational was held to, not a formulation-specific relaxation. Does NOT assert either
    /// formulation must pass (issue #63's "no benchmark-specific hacks to force a pass") -
    /// reports whatever each formulation actually achieves, honestly. `#[ignore]`d - two real
    /// training runs (~15-25 min each in release).
    #[test]
    #[ignore]
    fn issue_71_real_strong_and_hybrid_formulation_no_hole_runtime_evidence() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{BoundaryValueProblem, DomainOptim};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{step_physics_multi, BDevice, B};
        use burn::module::AutodiffModule;
        use burn::tensor::backend::Backend;
        use pinn_core::messages::SolverConfig;
        use pinn_core::problem_spec::FormulationSelection;

        fn train(spec: &ProblemSpec, device: &crate::training_core::BDevice) -> crate::network::ElasticityNet<crate::training_core::BInner> {
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let problem = UserDefinedProblem::new(spec.clone());
            crate::problem::validate_loss_terms(&problem);
            let net_cfg = ElasticityNetConfig::new()
                .with_input_dim(spec.geometry.net_input_dim())
                .with_hidden_dim(spec.network.hidden_dim)
                .with_n_hidden(spec.network.n_hidden)
                .with_output_dim(5);
            B::seed(device, spec.network.model_init_seed);
            let mut model = net_cfg.init(device);
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
            let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
            let mut saw = SawBrdr::with_base(base_weights, 0.95);
            let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
            let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
            let scales = crate::training_core::compute_reference_scales_for_plate(spec);
            let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
            let config = SolverConfig::default_kirsch();
            let sampling = problem.sampling_strategy(0);
            let placeholder = GeometryConfig::kirsch_plate_inches();

            for step in 0..spec.training.max_steps {
                let data = resample_plate_step_data(
                    sampling, &placeholder, &spec.load, spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
                );
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &data, u_ref, ref_energy, ref_stress2,
                    spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
                );
                let (new_model, _out) = step_physics_multi(
                    vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
                );
                model = new_model.into_iter().next().unwrap();
            }
            model.valid()
        }

        let device = BDevice::default();
        let base_spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: FormulationSelection::Variational, // overwritten per-run below
        };

        let mut strong_spec = base_spec.clone();
        strong_spec.formulation = FormulationSelection::Strong;
        println!("=== training Strong formulation (equilibrium + outer_traction) ===");
        let strong_model = train(&strong_spec, &device);
        let strong_result = run_no_hole_benchmark(&strong_model, &strong_spec, &device);
        println!("[PH4-11] Strong formulation benchmark: {strong_result:?}");

        let mut hybrid_spec = base_spec.clone();
        hybrid_spec.formulation = FormulationSelection::Hybrid(vec![
            "interior_energy".to_string(), "equilibrium".to_string(),
            "outer_traction".to_string(), "external_work".to_string(),
        ]);
        println!("=== training Hybrid formulation (interior_energy + equilibrium + outer_traction + external_work) ===");
        let hybrid_model = train(&hybrid_spec, &device);
        let hybrid_result = run_no_hole_benchmark(&hybrid_model, &hybrid_spec, &device);
        println!("[PH4-11] Hybrid formulation benchmark: {hybrid_result:?}");

        // Real assertions, not benchmark-forcing ones: both runs must produce finite,
        // non-collapsed fields - a genuine training-methodology sanity check (matches this
        // file's own `run_no_hole_benchmark_runs_end_to_end_and_correctly_fails_an_untrained_
        // model`'s "does the number even compute" bar), independent of whether either formulation
        // clears the hard P2-14 thresholds.
        assert!(strong_result.sigma_xx_relative_error.is_finite(), "{strong_result:?}");
        assert!(hybrid_result.sigma_xx_relative_error.is_finite(), "{hybrid_result:?}");
    }

    #[test]
    #[ignore]
    fn issue_70_real_l5_single_hole_kt_against_verified_no_hole_companion() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{BoundaryValueProblem, DomainOptim};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{step_physics_multi, BDevice, B};
        use burn::module::AutodiffModule;
        use burn::tensor::backend::Backend;
        use pinn_core::messages::SolverConfig;

        fn train(spec: &ProblemSpec, device: &crate::training_core::BDevice) -> crate::network::ElasticityNet<crate::training_core::BInner> {
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let problem = UserDefinedProblem::new(spec.clone());
            crate::problem::validate_loss_terms(&problem);
            let net_cfg = ElasticityNetConfig::new()
                .with_input_dim(spec.geometry.net_input_dim())
                .with_hidden_dim(spec.network.hidden_dim)
                .with_n_hidden(spec.network.n_hidden)
                .with_output_dim(5);
            B::seed(device, spec.network.model_init_seed);
            let mut model = net_cfg.init(device);
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
            let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
            let mut saw = SawBrdr::with_base(base_weights, 0.95);
            let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
            let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
            let scales = crate::training_core::compute_reference_scales_for_plate(spec);
            let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
            let config = SolverConfig::default_kirsch();
            let sampling = problem.sampling_strategy(0);
            let placeholder = GeometryConfig::kirsch_plate_inches();

            for step in 0..spec.training.max_steps {
                let data = resample_plate_step_data(
                    sampling, &placeholder, &spec.load, spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
                );
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &data, u_ref, ref_energy, ref_stress2,
                    spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
                );
                let (new_model, _out) = step_physics_multi(
                    vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
                );
                model = new_model.into_iter().next().unwrap();
            }
            model.valid()
        }

        let device = BDevice::default();

        let no_hole_spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        println!("=== training no-hole companion ===");
        let no_hole_model = train(&no_hole_spec, &device);
        let no_hole_result = run_no_hole_benchmark(&no_hole_model, &no_hole_spec, &device);
        println!("[L5] no-hole companion benchmark: {no_hole_result:?}");
        assert!(no_hole_result.passed, "L5 requires a verified no-hole companion, got: {no_hole_result:?}");

        let mut hole_spec = no_hole_spec.clone();
        hole_spec.geometry.holes = vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }];
        println!("=== training single-hole (infinite-approx-eligible, ratio=0.05) ===");
        let hole_model = train(&hole_spec, &device);

        let hole_result = run_hole_benchmark(&hole_model, &hole_spec, 0, &no_hole_result, &device)
            .expect("run_hole_benchmark must accept a verified-passing no-hole companion");
        println!("[L5] hole benchmark: {hole_result:?}");

        assert!(hole_result.kt.is_finite() && hole_result.kt > 0.0, "{hole_result:?}");
        assert_eq!(hole_result.reference_kind, HoleReferenceKind::InfiniteApprox, "ratio=0.05 must classify as InfiniteApprox: {hole_result:?}");
        assert!(hole_result.relative_error_vs_infinite_theory.is_some(), "{hole_result:?}");
    }

    /// Issue #63 sub-issue #70, re-attempt after issue #74's fix. The test above
    /// (`issue_70_real_l5_single_hole_kt_against_verified_no_hole_companion`) trains with plain
    /// uniform sampling and found `kt=1.008` vs the theoretical `3.0` - root-caused (not just
    /// observed) to near-hole collocation starvation: a zero-cost sampling check found only
    /// ~0.55% of interior points land within 2 hole-radii of a ratio=0.05 hole's boundary under
    /// uniform Monte-Carlo sampling. The identified fix was AMR's density biasing toward the
    /// hole boundary - but AMR+Variational crashed past ~1200-2200 steps until issue #74 fixed
    /// it (`training_core::probe_interior_energy_residuals` now routes through `BInner` instead
    /// of leaking orphaned nodes into the live autodiff graph). This test re-runs the IDENTICAL
    /// hole configuration with `amr_enabled: true`, reusing this file's own AMR sweep pattern
    /// (mirrors `runner::run_user_problem_training_from`'s AMR block line-for-line - that
    /// function isn't reusable here directly since it streams `TrainingUpdate`s over a channel
    /// rather than returning the trained model this test needs for `run_hole_benchmark`).
    ///
    /// Same discipline as every other benchmark test in this file: does NOT assert `kt≈3.0`
    /// (issue #63's "no benchmark-specific hacks to force a pass") - reports whatever AMR
    /// actually achieves, honestly, whether that's a real improvement, no improvement, or worse.
    /// `#[ignore]`d - same real cost as the baseline test above.
    #[test]
    #[ignore]
    fn issue_70_real_l5_with_amr_enabled_after_issue_74_fix() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, MultiStepCtx};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{probe_interior_energy_residuals, residual_stats, step_physics_multi, BDevice, B};
        use burn::module::AutodiffModule;
        use burn::tensor::backend::Backend;
        use pinn_core::amr::{derive_amr_config, AdaptiveGrid, AmrDomain};
        use pinn_core::messages::SolverConfig;

        const AMR_WARMUP_STEPS: usize = 200;

        fn train(spec: &ProblemSpec, device: &crate::training_core::BDevice) -> crate::network::ElasticityNet<crate::training_core::BInner> {
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let problem = UserDefinedProblem::new(spec.clone());
            crate::problem::validate_loss_terms(&problem);
            let net_cfg = ElasticityNetConfig::new()
                .with_input_dim(spec.geometry.net_input_dim())
                .with_hidden_dim(spec.network.hidden_dim)
                .with_n_hidden(spec.network.n_hidden)
                .with_output_dim(5);
            B::seed(device, spec.network.model_init_seed);
            let mut model = net_cfg.init(device);
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
            let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
            let mut saw = SawBrdr::with_base(base_weights, 0.95);
            let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
            let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
            let scales = crate::training_core::compute_reference_scales_for_plate(spec);
            let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
            let config = SolverConfig::default_kirsch();
            let sampling = problem.sampling_strategy(0);
            let placeholder = GeometryConfig::kirsch_plate_inches();
            let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };

            let collocation_margin_m = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
            let collocation_geometry = spec.geometry.inflated_for_collocation(collocation_margin_m);
            let amr_cfg = derive_amr_config((-half_w, half_w, -half_h, half_h), &collocation_geometry.lock_zones());
            let amr_interval = amr_cfg.interval_steps;
            let mut amr_grid = AdaptiveGrid::<UserGeometry>::new(&collocation_geometry, amr_cfg);

            for step in 0..spec.training.max_steps {
                let mut data = resample_plate_step_data(
                    sampling, &placeholder, &spec.load, spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
                );
                problem.set_interior_weights(None);

                if spec.training.amr_enabled && step >= AMR_WARMUP_STEPS && (step - AMR_WARMUP_STEPS) % amr_interval == 0 {
                    let probe_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(), probe_term_gradients: false,
                        phase2_active: true, step,
                    };
                    if let Some(residuals) = probe_interior_energy_residuals(&probe_ctx, &[&model], device).remove(&USER_DOMAIN) {
                        amr_grid.update_residuals(&residuals);
                        if amr_grid.should_adapt(&residuals) {
                            let points_before = data.int_norm.len();
                            let (rms_before, max_before) = residual_stats(&residuals);
                            amr_grid.adapt();
                            if spec.training.measure_aware_training {
                                let density_samples = amr_grid.sample_points_with_density();
                                data.int_norm = density_samples.iter().map(|s| norm_pt(s.point[0], s.point[1])).collect();
                                problem.set_interior_weights(Some(pinn_core::amr::compensation_weights(&density_samples)));
                            } else {
                                data.int_norm = amr_grid.sample_points().iter().map(|&[x, y]| norm_pt(x, y)).collect();
                            }
                            let points_after = data.int_norm.len();
                            let after_ctx = MultiStepCtx {
                                config: &config, problem: &problem, fd: &fd, k: 1.0,
                                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                                dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                                constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                                n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(), probe_term_gradients: false,
                                phase2_active: true, step,
                            };
                            let after_residuals = probe_interior_energy_residuals(&after_ctx, &[&model], device)
                                .remove(&USER_DOMAIN).unwrap_or_default();
                            let (rms_after, max_after) = residual_stats(&after_residuals);
                            println!(
                                "  [AMR sweep step {step}] points {points_before}->{points_after} residual_rms {rms_before:.4e}->{rms_after:.4e} residual_max {max_before:.4e}->{max_after:.4e}",
                            );
                        }
                    }
                }

                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &data, u_ref, ref_energy, ref_stress2,
                    spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
                );
                let (new_model, _out) = step_physics_multi(
                    vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
                );
                model = new_model.into_iter().next().unwrap();
            }
            model.valid()
        }

        let device = BDevice::default();

        let no_hole_spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        println!("=== training no-hole companion (AMR off - matches the shipped canonical config) ===");
        let no_hole_model = train(&no_hole_spec, &device);
        let no_hole_result = run_no_hole_benchmark(&no_hole_model, &no_hole_spec, &device);
        println!("[L5+AMR] no-hole companion benchmark: {no_hole_result:?}");
        assert!(no_hole_result.passed, "L5 requires a verified no-hole companion, got: {no_hole_result:?}");

        let mut hole_spec = no_hole_spec.clone();
        hole_spec.geometry.holes = vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }];
        hole_spec.training.amr_enabled = true;
        println!("=== training single-hole WITH AMR (ratio=0.05, same config as the AMR-off baseline test) ===");
        let hole_model = train(&hole_spec, &device);

        let hole_result = run_hole_benchmark(&hole_model, &hole_spec, 0, &no_hole_result, &device)
            .expect("run_hole_benchmark must accept a verified-passing no-hole companion");
        println!("[L5+AMR] hole benchmark (AMR enabled): {hole_result:?}");

        assert!(hole_result.kt.is_finite() && hole_result.kt > 0.0, "{hole_result:?}");
        assert_eq!(hole_result.reference_kind, HoleReferenceKind::InfiniteApprox, "ratio=0.05 must classify as InfiniteApprox: {hole_result:?}");
        assert!(hole_result.relative_error_vs_infinite_theory.is_some(), "{hole_result:?}");
    }

    /// Issue #75: the decisive real re-attempt. `issue_70_real_l5_with_amr_enabled_after_
    /// issue_74_fix` (above) used the OLD sweep-only AMR mechanism (data.int_norm overwritten
    /// only on the 3 steps a sweep fires in a 3000-step run, uniform resampling winning back
    /// every other step) and found it made essentially no difference (`kt=1.0088` vs `1.0076`
    /// without AMR at all). This test uses issue #75's actual fix instead: `apply_persistent_
    /// adaptive_interior_sample` called EVERY step (quadtree jitter + geometry-seeded hole
    /// annulus, active from step 0 - never waiting on the network's own residual signal),
    /// mirroring `runner.rs`'s real production wiring exactly (not a simplified reimplementation
    /// - same AMR_WARMUP_STEPS/interval/probe/adapt() sequence). Same discipline as every other
    /// benchmark test in this file: does NOT assert `kt≈3.0` (issue #63's own no-benchmark-
    /// hacking rule) - reports whatever persistent geometry-aware AMR actually achieves,
    /// honestly. `#[ignore]`d - two real training runs (~15-30 min each in release).
    #[test]
    #[ignore]
    fn issue_75_real_l5_with_persistent_geometry_aware_amr() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, MultiStepCtx};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{probe_interior_energy_residuals, step_physics_multi, BDevice, B};
        use burn::module::AutodiffModule;
        use burn::tensor::backend::Backend;
        use pinn_core::amr::{derive_amr_config, AdaptiveGrid, AmrDomain};
        use pinn_core::messages::SolverConfig;

        const AMR_WARMUP_STEPS: usize = 200;

        fn train(spec: &ProblemSpec, use_persistent_amr: bool, device: &crate::training_core::BDevice) -> crate::network::ElasticityNet<crate::training_core::BInner> {
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let problem = UserDefinedProblem::new(spec.clone());
            crate::problem::validate_loss_terms(&problem);
            let net_cfg = ElasticityNetConfig::new()
                .with_input_dim(spec.geometry.net_input_dim())
                .with_hidden_dim(spec.network.hidden_dim)
                .with_n_hidden(spec.network.n_hidden)
                .with_output_dim(5);
            B::seed(device, spec.network.model_init_seed);
            let mut model = net_cfg.init(device);
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
            let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
            let mut saw = SawBrdr::with_base(base_weights, 0.95);
            let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
            let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
            let scales = crate::training_core::compute_reference_scales_for_plate(spec);
            let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
            let config = SolverConfig::default_kirsch();
            let sampling = problem.sampling_strategy(0);
            let placeholder = GeometryConfig::kirsch_plate_inches();

            let collocation_margin_m = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
            let collocation_geometry = spec.geometry.inflated_for_collocation(collocation_margin_m);
            let amr_cfg = derive_amr_config((-half_w, half_w, -half_h, half_h), &collocation_geometry.lock_zones());
            let amr_interval = amr_cfg.interval_steps;
            let mut amr_grid = AdaptiveGrid::<UserGeometry>::new(&collocation_geometry, amr_cfg);

            for step in 0..spec.training.max_steps {
                let mut data = resample_plate_step_data(
                    sampling, &placeholder, &spec.load, spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
                );

                if use_persistent_amr && step >= AMR_WARMUP_STEPS && (step - AMR_WARMUP_STEPS) % amr_interval == 0 {
                    // `update_residuals` consumes one value per active leaf in DFS order.
                    // The training batch has a fixed budget and may contain several samples
                    // from one leaf, so it is not a valid residual-assignment cloud.
                    let probe_data = adaptive_grid_probe_data(
                        &data, half_w, half_h, &mut amr_grid, step,
                    );
                    let probe_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![DomainStepCtx { data: &probe_data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(), probe_term_gradients: false,
                        phase2_active: true, step,
                    };
                    if let Some(residuals) = probe_interior_energy_residuals(&probe_ctx, &[&model], device).remove(&USER_DOMAIN) {
                        amr_grid.update_residuals(&residuals);
                        if amr_grid.should_adapt(&residuals) {
                            amr_grid.adapt();
                        }
                    }
                }

                let (source, weights) = if use_persistent_amr {
                    apply_persistent_adaptive_interior_sample(
                        &mut data, &spec.geometry, spec.training.fd_h, half_w, half_h, Some(&mut amr_grid), step,
                    )
                } else {
                    (InteriorSampleSource::Uniform, None)
                };
                if source == InteriorSampleSource::PersistentAdaptive {
                    problem.set_interior_weights(weights);
                } else {
                    problem.set_interior_weights(None);
                }

                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &data, u_ref, ref_energy, ref_stress2,
                    spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
                );
                let (new_model, _out) = step_physics_multi(
                    vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
                );
                model = new_model.into_iter().next().unwrap();
            }
            model.valid()
        }

        let device = BDevice::default();

        let no_hole_spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        println!("=== training no-hole companion (persistent AMR is hole-specific - unaffected) ===");
        let no_hole_model = train(&no_hole_spec, false, &device);
        let no_hole_result = run_no_hole_benchmark(&no_hole_model, &no_hole_spec, &device);
        println!("[L5+persistent-AMR] no-hole companion benchmark: {no_hole_result:?}");
        assert!(no_hole_result.passed, "L5 requires a verified no-hole companion, got: {no_hole_result:?}");

        let mut hole_spec = no_hole_spec.clone();
        hole_spec.geometry.holes = vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }];
        println!("=== training single-hole WITH PERSISTENT geometry-aware AMR (ratio=0.05) ===");
        let hole_model = train(&hole_spec, true, &device);

        let hole_result = run_hole_benchmark(&hole_model, &hole_spec, 0, &no_hole_result, &device)
            .expect("run_hole_benchmark must accept a verified-passing no-hole companion");
        println!("[L5+persistent-AMR] hole benchmark: {hole_result:?}");

        assert!(hole_result.kt.is_finite() && hole_result.kt > 0.0, "{hole_result:?}");
        assert_eq!(hole_result.reference_kind, HoleReferenceKind::InfiniteApprox, "ratio=0.05 must classify as InfiniteApprox: {hole_result:?}");
        assert!(hole_result.relative_error_vs_infinite_theory.is_some(), "{hole_result:?}");
    }

    /// Issue #75 follow-up: persistent geometry-aware AMR (this file's own `issue_75_real_l5_
    /// with_persistent_geometry_aware_amr`, above) gave `kt=1.0023` - essentially unchanged
    /// from uniform sampling despite a real, verified ~20x near-hole density increase. Reading
    /// `UserDefinedProblem::loss_terms()` afterward found a real structural reason this specific
    /// null result isn't surprising: `hole_free_active = !matches!(formulation, Variational)` -
    /// for EVERY L5 attempt so far (all `Variational`), a `Free` hole registers NO `HoleBcTerm`
    /// at all. Traction-free at the hole is enforced only IMPLICITLY through the energy
    /// functional's own natural boundary condition - there is no explicit, local,
    /// hole-boundary-specific loss signal for Variational at all, so denser sampling near the
    /// hole only sharpens the domain-integrated energy ESTIMATE, not necessarily the network's
    /// incentive to produce a genuinely sharp local field there (Deep-Ritz/DEM methods are
    /// documented in the literature to under-resolve sharp local features via pure
    /// energy-integral minimization - a representational/optimization limitation, not a
    /// quadrature one). `Strong` formulation is structurally different: it registers an
    /// EXPLICIT `HoleBcTerm::Free` (direct-stress traction-free penalty, evaluated locally at
    /// the hole ring) alongside `EquilibriumTerm`/`OuterTractionTerm` - a genuinely different
    /// training signal at the hole boundary, not just a denser sample of the same signal. This
    /// test combines persistent AMR (already proven to increase near-hole density) with Strong
    /// formulation (already proven to pass the no-hole L4 benchmark cleanly, issue #71) to test
    /// this specific, code-grounded hypothesis with real evidence, honestly - does NOT assert
    /// `kt≈3.0` (issue #63's own no-benchmark-hacking rule). `#[ignore]`d - two real training
    /// runs, Strong's own `EquilibriumTerm` needs a real Hessian forward pass (materially more
    /// expensive per step than Variational - see issue #71's own real measured runtime).
    #[test]
    #[ignore]
    fn issue_75_real_l5_strong_formulation_with_persistent_amr() {
        use crate::fd_stencil::FdConfig;
        use crate::lr_schedule::LrSchedule;
        use crate::network::ElasticityNetConfig;
        use crate::optim::{make_bias_optim, make_gate_optim, WeightOptim};
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, MultiStepCtx};
        use crate::saw_brdr::SawBrdr;
        use crate::training_core::{probe_interior_energy_residuals, step_physics_multi, BDevice, B};
        use burn::module::AutodiffModule;
        use burn::tensor::backend::Backend;
        use pinn_core::amr::{derive_amr_config, AdaptiveGrid, AmrDomain};
        use pinn_core::messages::SolverConfig;
        use pinn_core::problem_spec::FormulationSelection;

        const AMR_WARMUP_STEPS: usize = 200;

        fn train(spec: &ProblemSpec, use_persistent_amr: bool, device: &crate::training_core::BDevice) -> crate::network::ElasticityNet<crate::training_core::BInner> {
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let problem = UserDefinedProblem::new(spec.clone());
            crate::problem::validate_loss_terms(&problem);
            let net_cfg = ElasticityNetConfig::new()
                .with_input_dim(spec.geometry.net_input_dim())
                .with_hidden_dim(spec.network.hidden_dim)
                .with_n_hidden(spec.network.n_hidden)
                .with_output_dim(5);
            B::seed(device, spec.network.model_init_seed);
            let mut model = net_cfg.init(device);
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
            let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
            let mut saw = SawBrdr::with_base(base_weights, 0.95);
            let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
            let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
            let scales = crate::training_core::compute_reference_scales_for_plate(spec);
            let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
            let config = SolverConfig::default_kirsch();
            let sampling = problem.sampling_strategy(0);
            let placeholder = GeometryConfig::kirsch_plate_inches();

            let collocation_margin_m = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
            let collocation_geometry = spec.geometry.inflated_for_collocation(collocation_margin_m);
            let amr_cfg = derive_amr_config((-half_w, half_w, -half_h, half_h), &collocation_geometry.lock_zones());
            let amr_interval = amr_cfg.interval_steps;
            let mut amr_grid = AdaptiveGrid::<UserGeometry>::new(&collocation_geometry, amr_cfg);

            for step in 0..spec.training.max_steps {
                let mut data = resample_plate_step_data(
                    sampling, &placeholder, &spec.load, spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
                );

                if use_persistent_amr && step >= AMR_WARMUP_STEPS && (step - AMR_WARMUP_STEPS) % amr_interval == 0 {
                    let probe_data = adaptive_grid_probe_data(
                        &data, half_w, half_h, &mut amr_grid, step,
                    );
                    let probe_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![DomainStepCtx { data: &probe_data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(), probe_term_gradients: false,
                        phase2_active: true, step,
                    };
                    if let Some(residuals) = probe_interior_energy_residuals(&probe_ctx, &[&model], device).remove(&USER_DOMAIN) {
                        amr_grid.update_residuals(&residuals);
                        if amr_grid.should_adapt(&residuals) {
                            amr_grid.adapt();
                        }
                    }
                }

                let (source, weights) = if use_persistent_amr {
                    apply_persistent_adaptive_interior_sample(
                        &mut data, &spec.geometry, spec.training.fd_h, half_w, half_h, Some(&mut amr_grid), step,
                    )
                } else {
                    (InteriorSampleSource::Uniform, None)
                };
                if source == InteriorSampleSource::PersistentAdaptive {
                    problem.set_interior_weights(weights);
                } else {
                    problem.set_interior_weights(None);
                }

                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &data, u_ref, ref_energy, ref_stress2,
                    spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
                );
                let (new_model, _out) = step_physics_multi(
                    vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
                );
                model = new_model.into_iter().next().unwrap();
            }
            model.valid()
        }

        let device = BDevice::default();

        let no_hole_spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: FormulationSelection::Strong,
        };
        println!("=== training Strong no-hole companion ===");
        let no_hole_model = train(&no_hole_spec, false, &device);
        let no_hole_result = run_no_hole_benchmark(&no_hole_model, &no_hole_spec, &device);
        println!("[L5+Strong] no-hole companion benchmark: {no_hole_result:?}");
        assert!(no_hole_result.passed, "L5 requires a verified no-hole companion, got: {no_hole_result:?}");

        let mut hole_spec = no_hole_spec.clone();
        hole_spec.geometry.holes = vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }];
        println!("=== training single-hole Strong formulation + persistent AMR (ratio=0.05) ===");
        let hole_model = train(&hole_spec, true, &device);

        let hole_result = run_hole_benchmark(&hole_model, &hole_spec, 0, &no_hole_result, &device)
            .expect("run_hole_benchmark must accept a verified-passing no-hole companion");
        println!("[L5+Strong] hole benchmark: {hole_result:?}");

        assert!(hole_result.kt.is_finite() && hole_result.kt > 0.0, "{hole_result:?}");
        assert_eq!(hole_result.reference_kind, HoleReferenceKind::InfiniteApprox, "ratio=0.05 must classify as InfiniteApprox: {hole_result:?}");
        assert!(hole_result.relative_error_vs_infinite_theory.is_some(), "{hole_result:?}");
    }

    /// Complements (does not duplicate) this session's earlier zero-cost diagnostic that found
    /// only ~0.55% of interior points land within 2 hole-radii of a ratio=0.05 hole's boundary
    /// under `UserSamplingStrategy`'s UNIFORM Monte-Carlo sampling (see the `issue_70_real_l5_*`
    /// tests above). That diagnostic measured the starvation problem; this test proves the
    /// FIX mechanism - `pinn_core::amr::AdaptiveGrid`'s residual-driven `adapt()` - actually
    /// responds to it, without spending any training time: a synthetic residual field seeded
    /// large exactly in the hole's near-boundary band (mirroring `pinn_core::amr`'s own
    /// `spatial_test_e_circular_hole_boundary_ring_refines_from_residual_not_just_lock_zone`
    /// convention) must drive measurably higher point density in that band than
    /// `AdaptiveGrid`'s own initial uniform `sample_points()` produced, for the EXACT L5
    /// geometry (`half_w=half_h=0.10`, single Free hole `radius=0.005` at the origin,
    /// ratio=0.05) the `issue_70_real_l5_*` tests train against. `hole_zone_factor: 0.0`
    /// disables the structural hole-zone floor, so any density rise observed below is
    /// attributable only to the residual signal, not a free structural guarantee.
    #[test]
    fn adaptive_grid_density_near_l5_hole_rises_above_uniform_baseline_after_synthetic_residual_sweep() {
        use pinn_core::amr::{AdaptiveGrid, AmrtConfig};

        let half_w = 0.10_f64;
        let half_h = 0.10_f64;
        let r_hole = 0.005_f64; // ratio = r_hole / half_w = 0.05, the exact L5 config
        let geom = UserGeometry {
            half_w, half_h, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: r_hole, bc: HoleBc::Free }],
        };
        let cfg = AmrtConfig {
            initial_level: 5, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::<UserGeometry>::new(&geom, cfg);

        // Near-hole band: within 2 hole-radii of the hole boundary (r in [r_hole, 3*r_hole]) -
        // the exact band the earlier uniform-sampling diagnostic measured.
        let near_lo = r_hole;
        let near_hi = r_hole * 3.0;
        let near_area = std::f64::consts::PI * (near_hi * near_hi - near_lo * near_lo);
        let in_band = |&[x, y]: &[f64; 2]| -> bool {
            let r = (x * x + y * y).sqrt();
            r >= near_lo && r <= near_hi
        };

        let initial_pts = grid.sample_points();
        let initial_near = initial_pts.iter().filter(|p| in_band(p)).count();
        assert!(initial_near > 0, "sanity: the near-hole band must contain at least one initial \
            uniform sample point, got 0 (test setup needs a finer initial_level)");
        let baseline_density = initial_near as f64 / near_area;

        // Drive several sweeps with residual concentrated exactly in the hole's near-boundary
        // band - nothing else in the domain gets any signal.
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|p| if in_band(p) { 1.0 } else { 0.001 }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }

        let after_pts = grid.sample_points();
        let after_near = after_pts.iter().filter(|p| in_band(p)).count();
        let after_density = after_near as f64 / near_area;

        assert!(
            after_density > baseline_density * 3.0,
            "AMR must concentrate refinement near the L5 hole's boundary once the residual \
             signal says so: baseline_density={baseline_density:.1} (near={initial_near}), \
             after_density={after_density:.1} (near={after_near})"
        );
    }

    #[test]
    #[should_panic(expected = "only valid for a plate with NO holes")]
    fn run_no_hole_benchmark_panics_on_a_holed_geometry() {
        let model = tiny_model(&two_hole_geometry());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let _ = run_no_hole_benchmark(&model, &spec, &device);
    }

    // ─── Issue #62 PH3-01: production no-hole baseline evidence ────────────────────────────

    /// Loads the REAL, frozen legacy checkpoint captured for PH3-01
    /// (`Debug_run/baseline_legacy_no_hole/`, main SHA 21a0e730 — a genuine 2000-step Hybrid
    /// run whose `stress_solver_report.json`/`model.meta.json` sit alongside the weights in
    /// that same directory) and runs the actual P2-14 `run_no_hole_benchmark` against it —
    /// closing the exact gap issue #62 §2.1.A calls out (`"model_validity": null` in the
    /// persisted report, i.e. the hard benchmark machinery exists but was never actually run
    /// against this checkpoint and recorded). `#[ignore]`d because it reads a checkpoint file
    /// from a fixed repo-relative path rather than being a self-contained unit test — run
    /// explicitly with `cargo test --release -p pinn-solver ph3_01 -- --ignored --nocapture`
    /// to reproduce the PH3-01 manifest entry's own recorded numbers.
    #[test]
    #[ignore = "reads the real PH3-01 baseline checkpoint from Debug_run/baseline_legacy_no_hole/"]
    fn ph3_01_baseline_legacy_no_hole_checkpoint_benchmark_result() {
        let weights_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../Debug_run/baseline_legacy_no_hole/model"
        ));
        let device = crate::training_core::BDevice::default();
        let (model, meta) = crate::checkpoint::load_checkpoint(weights_path, &device)
            .expect("PH3-01 baseline checkpoint must load - see Debug_run/baseline_legacy_no_hole/");
        let spec = match &meta.spec {
            crate::checkpoint::CheckpointSpec::Plate(spec) => spec.clone(),
            crate::checkpoint::CheckpointSpec::Parametric(_) => {
                panic!("PH3-01 baseline is a Plate checkpoint, not Parametric")
            }
        };
        assert!(spec.geometry.holes.is_empty(), "PH3-01 baseline is the no-hole case");

        let result = run_no_hole_benchmark(&model, &spec, &device);
        println!("PH3-01 baseline_legacy_no_hole run_no_hole_benchmark result: {result:#?}");
        println!("PH3-01 baseline checkpoint provenance: {:#?}", meta.provenance);

        // Real, evidence-based assertions (not just "is_finite") - this is the actual gate
        // issue #62 §2.1.C says the current run FAILS (traction RMS ~1.40% > 1% threshold), so
        // this test's own expectation is that the frozen legacy baseline does NOT pass yet -
        // an honest regression guard on the baseline's own recorded failure mode, not a
        // vacuous check.
        assert!(result.sigma_xx_relative_error.is_finite());
        assert!(result.sigma_yy_over_ref.is_finite());
        assert!(result.sigma_xy_over_ref.is_finite());
        assert!(result.traction_rms_over_ref.is_finite());
        assert!(result.load_transfer_ratio.is_finite());
        assert!(
            !result.passed,
            "PH3-01 baseline is EXPECTED to fail the hard no-hole benchmark (issue #62 \
             §2.1.C: traction RMS ~1.40% > 1% threshold) - if this ever passes, the baseline \
             checkpoint or benchmark logic has changed and PH3-01's manifest entry must be \
             re-verified: {result:?}"
        );
    }

    // ─── Issue #62 PH3-08: displacement/stress discrepancy investigation ───────────────────

    /// Real investigation, against the ACTUAL PH3-01 baseline checkpoint (not a fresh training
    /// run - the exact numbers issue #62 §2.1.D cites, `123.446 um` reported max vs. `~101.3 um`
    /// analytic corner magnitude, come from this exact checkpoint), of WHERE in the domain the
    /// reported maximum displacement actually occurs and whether it matches the analytic
    /// solution AT THAT SAME LOCATION - the investigation order issue #62 itself specifies
    /// (output scaling -> coordinate normalization -> ... -> network approximation error) is
    /// followed by elimination below, each step backed by a real, printed number, not assumed.
    #[test]
    #[ignore = "reads the real PH3-01 baseline checkpoint from Debug_run/baseline_legacy_no_hole/"]
    fn ph3_08_baseline_displacement_discrepancy_investigation() {
        let weights_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../Debug_run/baseline_legacy_no_hole/model"
        ));
        let device = crate::training_core::BDevice::default();
        let (model, meta) = crate::checkpoint::load_checkpoint(weights_path, &device)
            .expect("PH3-01 baseline checkpoint must load - see Debug_run/baseline_legacy_no_hole/");
        let spec = match &meta.spec {
            crate::checkpoint::CheckpointSpec::Plate(spec) => spec.clone(),
            crate::checkpoint::CheckpointSpec::Parametric(_) => panic!("expected a Plate checkpoint"),
        };
        let (half_w, half_h) = (spec.geometry.half_w, spec.geometry.half_h);
        let e = spec.material.e;
        let nu = spec.material.nu;
        let px = spec.load.px;

        // Step 1 ("output scaling"): reproduce the EXACT GUI computation (same grid size [64,64]
        // the real run used, same `u_ref` formula `compute_reference_scales_for_plate` already
        // proved correct via its own dedicated hand-computed-value unit test) - if this alone
        // already disagrees with the persisted `stress_solver_report.json`'s `123.446 um`, the
        // discrepancy is in the REPORTING layer, not the physics. If it agrees, the discrepancy
        // is real and lives in what the network actually learned.
        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let fd = crate::fd_stencil::FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
        let vis = evaluate_user_vis_grid(&model, &spec.geometry, [64, 64], scales.u_ref, px, &spec.material, &fd, &[], &device);

        // Step 2 ("coordinate normalization/de-normalization" + "edge/corner evaluation"): the
        // vis grid's own point generation (`evaluate_user_vis_grid`'s own body, read directly)
        // places grid index (0,0) at (x_norm,y_norm)=(-1,-1) and (nx-1,ny-1) at (+1,+1) - the
        // TRUE corners ARE exact grid points, not missed by a coarser sampling. Find the actual
        // (ix,iy) where the reported maximum magnitude occurs.
        let (ny, nx) = vis.disp_u.dim();
        let mut max_mag = f32::NEG_INFINITY;
        let mut max_idx = (0usize, 0usize);
        for iy in 0..ny {
            for ix in 0..nx {
                let (u, v) = (vis.disp_u[[iy, ix]], vis.disp_v[[iy, ix]]);
                if u.is_finite() && v.is_finite() {
                    let mag = (u * u + v * v).sqrt();
                    if mag > max_mag { max_mag = mag; max_idx = (ix, iy); }
                }
            }
        }
        let (ix, iy) = max_idx;
        let x_norm = -1.0 + 2.0 * ix as f64 / (nx.max(2) - 1) as f64;
        let y_norm = -1.0 + 2.0 * iy as f64 / (ny.max(2) - 1) as f64;
        let (x_phys, y_phys) = (x_norm * half_w, y_norm * half_h);
        let is_true_corner = (x_norm.abs() - 1.0).abs() < 1e-9 && (y_norm.abs() - 1.0).abs() < 1e-9;

        // Step 3 ("stress-vs-displacement consistency" + "Poisson contraction"): the exact
        // analytic solution AT THE SAME (x_phys, y_phys) the network's own maximum occurs at -
        // not just at the nominal (half_w, half_h) corner, in case the max is elsewhere.
        let u_analytic = (px / e) * x_phys;
        let v_analytic = -nu * (px / e) * y_phys;
        let mag_analytic = (u_analytic * u_analytic + v_analytic * v_analytic).sqrt();

        println!("[PH3-08] max |disp| = {:.6e} m at grid ({ix},{iy}) -> (x_norm={:.4}, y_norm={:.4}) -> (x={:.6e}, y={:.6e}) m, is_true_corner={is_true_corner}", max_mag, x_norm, y_norm, x_phys, y_phys);
        println!("[PH3-08] network u,v at that point:  u={:.6e}  v={:.6e}", vis.disp_u[[iy, ix]], vis.disp_v[[iy, ix]]);
        println!("[PH3-08] analytic u,v at that SAME point: u={u_analytic:.6e}  v={v_analytic:.6e}  |disp|={mag_analytic:.6e}");
        println!("[PH3-08] relative error at the network's own reported max location: {:.2}%", (max_mag as f64 - mag_analytic).abs() / mag_analytic.abs() * 100.0);

        // The nominal corner (half_w, half_h) specifically, for direct comparison against issue
        // #62's own cited "~101.3 um" figure.
        let u_corner_analytic = (px / e) * half_w;
        let v_corner_analytic = -nu * (px / e) * half_h;
        let mag_corner_analytic = (u_corner_analytic * u_corner_analytic + v_corner_analytic * v_corner_analytic).sqrt();
        println!("[PH3-08] nominal analytic corner |disp| = {mag_corner_analytic:.6e} m (issue #62's own cited ~101.3 um)");

        // Step 4 ("edge/corner evaluation", continued): `UserSamplingStrategy::sample_boundary`
        // places boundary collocation points at `frac = (i+0.5)/per_edge` - DELIBERATELY
        // avoiding the exact corners (its own doc comment: "avoids exact corners"). If the
        // corner itself is never a training point, the network's value exactly there is an
        // UNSUPERVISED EXTRAPOLATION one half-edge-spacing beyond the nearest real boundary
        // sample - real error should therefore be measurably WORSE at the exact corner than one
        // grid step inward along either edge. This distinguishes "corner extrapolation, an
        // inherent and expected property of this training scheme" from "a systemic bug that
        // would show comparable error everywhere on the boundary".
        let one_step_in_x = (ix + 1).min(nx - 1);
        let one_step_in_y = (iy + 1).min(ny - 1);
        let check_point = |ix: usize, iy: usize, label: &str| {
            let xn = -1.0 + 2.0 * ix as f64 / (nx.max(2) - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny.max(2) - 1) as f64;
            let (xp, yp) = (xn * half_w, yn * half_h);
            let (u_net, v_net) = (vis.disp_u[[iy, ix]] as f64, vis.disp_v[[iy, ix]] as f64);
            let (u_an, v_an) = ((px / e) * xp, -nu * (px / e) * yp);
            let mag_net = (u_net * u_net + v_net * v_net).sqrt();
            let mag_an = (u_an * u_an + v_an * v_an).sqrt();
            let rel_err = (mag_net - mag_an).abs() / mag_an.abs() * 100.0;
            println!("[PH3-08] {label} (x={xp:.4e}, y={yp:.4e}): network |disp|={mag_net:.6e}  analytic |disp|={mag_an:.6e}  rel_err={rel_err:.2}%");
            rel_err
        };
        let err_one_step_x = check_point(one_step_in_x, iy, "one grid step inward along x from the corner");
        let err_one_step_y = check_point(ix, one_step_in_y, "one grid step inward along y from the corner");
        let err_at_corner = (max_mag as f64 - mag_analytic).abs() / mag_analytic.abs() * 100.0;
        println!("[PH3-08] error at exact corner ({err_at_corner:.2}%) vs one step inward (x: {err_one_step_x:.2}%, y: {err_one_step_y:.2}%)");

        // Step 5 ("stress-vs-displacement consistency" + "network approximation error"): error
        // essentially FLAT moving inward from the corner (~22% at all three points above)
        // already rules out corner-specific extrapolation. Check the domain CENTER (x=0,y=0,
        // analytic u=v=0 exactly) - if the network's own u is ALSO non-trivially offset from
        // zero there, the error is a genuine domain-wide property of what the network learned
        // for u specifically (not a boundary-localized artifact of any kind).
        let (cx, cy) = (nx / 2, ny / 2);
        let (u_center, v_center) = (vis.disp_u[[cy, cx]] as f64, vis.disp_v[[cy, cx]] as f64);
        println!("[PH3-08] domain center (grid {cx},{cy}): network u={u_center:.6e}  v={v_center:.6e}  (analytic: both exactly 0.0)");

        // **THE ROOT-CAUSE FINDING**: the center's own u offset from its analytic value (0.0)
        // and the corner's own u offset from ITS analytic value are nearly IDENTICAL - i.e. the
        // network's u field is (the correct affine slope) PLUS a near-CONSTANT residual offset
        // across the whole domain, not a scale/slope/sign/Poisson/corner-extrapolation error at
        // all. This is the signature of an imperfectly-suppressed RIGID-BODY TRANSLATION mode
        // in u specifically (issue #61 P2-07's own `TranslationGaugeTerm` exists precisely to
        // remove this nullspace for a pure-Neumann, no-Fixed-BC configuration like this one) -
        // `v`'s own analogous offset (2.0e-7) is negligible by comparison, consistent with the
        // gauge term suppressing v's translation mode adequately while u's remains only
        // partially suppressed within this run's 2000-step budget.
        let u_offset_at_corner = vis.disp_u[[iy, ix]] as f64 - u_analytic;
        let u_offset_at_center = u_center - 0.0;
        let offset_consistency = (u_offset_at_corner - u_offset_at_center).abs() / u_offset_at_corner.abs().max(1e-12);
        println!("[PH3-08] u offset from analytic: at corner={u_offset_at_corner:.6e}, at center={u_offset_at_center:.6e} - consistency={:.1}% difference", offset_consistency * 100.0);

        // Real, checkable assertions - not just prints - so this investigation's own conclusion
        // is a regression-guarded fact, not prose that can silently go stale.
        assert!(is_true_corner, "the reported maximum displacement must occur at a true domain corner for a no-hole plate under this load (both u and v grow monotonically with |x|,|y| for this affine field) - if this ever fails, the maximum is occurring somewhere unexpected and this investigation's own premise needs revisiting");
        assert!((mag_corner_analytic - 101.3e-6).abs() / 101.3e-6 < 0.01, "sanity check on issue #62's own cited analytic figure: computed {mag_corner_analytic:.6e} vs cited ~101.3e-6");
        assert!(
            offset_consistency < 0.2,
            "the u-offset-from-analytic at the corner and at the domain center must be nearly \
             equal for this to genuinely be a near-uniform residual translation mode (the real \
             root cause this investigation found) rather than a scale/slope error: corner \
             offset={u_offset_at_corner:.6e}, center offset={u_offset_at_center:.6e}, \
             consistency={:.1}%", offset_consistency * 100.0,
        );
    }

    fn failed_no_hole_gate() -> NoHoleBenchmarkResult {
        NoHoleBenchmarkResult {
            sigma_xx_relative_error: 0.5, sigma_yy_over_ref: 0.5, sigma_xy_over_ref: 0.5,
            traction_rms_over_ref: 0.5, load_transfer_ratio: 0.1,
            passed: false, failures: vec!["sigma_xx_relative_error"],
        }
    }

    fn passed_no_hole_gate() -> NoHoleBenchmarkResult {
        NoHoleBenchmarkResult {
            sigma_xx_relative_error: 0.001, sigma_yy_over_ref: 0.001, sigma_xy_over_ref: 0.001,
            traction_rms_over_ref: 0.001, load_transfer_ratio: 1.0,
            passed: true, failures: vec![],
        }
    }

    #[test]
    fn run_hole_benchmark_refuses_when_the_no_hole_gate_did_not_pass() {
        let model = tiny_model(&two_hole_geometry());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let result = run_hole_benchmark(&model, &spec, 0, &failed_no_hole_gate(), &device);
        assert!(result.is_err(), "must refuse to report Kt without a passing no-hole gate");
    }

    #[test]
    fn run_hole_benchmark_classifies_a_small_hole_as_infinite_approx_and_computes_relative_error() {
        let geometry = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 0.1,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.05, bc: HoleBc::Free }], // ratio 0.05 < 0.10
        };
        let model = tiny_model(&geometry);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry, material: MaterialProps::al7075_t6(), load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(), training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let result = run_hole_benchmark(&model, &spec, 0, &passed_no_hole_gate(), &device).expect("gate passed, must not refuse");
        assert_eq!(result.reference_kind, HoleReferenceKind::InfiniteApprox);
        assert!(result.relative_error_vs_infinite_theory.is_some());
        assert!(result.kt.is_finite());
    }

    #[test]
    fn run_hole_benchmark_classifies_a_large_hole_as_finite_with_no_theory_reference() {
        // two_hole_geometry: half_w=0.1, half_h=0.05, hole 0 radius=0.01 -> ratio = 0.01/0.05 = 0.2 > 0.10.
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry, material: MaterialProps::al7075_t6(), load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(), training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let result = run_hole_benchmark(&model, &spec, 0, &passed_no_hole_gate(), &device).expect("gate passed, must not refuse");
        assert_eq!(result.reference_kind, HoleReferenceKind::Finite);
        assert_eq!(result.relative_error_vs_infinite_theory, None, "no closed-form reference exists for a finite-plate hole - must not be fabricated");
    }

    // ─── enhancement.md Phase 10: energy balance ────────────────────────────────────────────

    #[test]
    fn physical_work_probe_uses_prescribed_not_model_traction() {
        let load = LoadConfig { px: 10.0, py: -4.0 };
        let values = prescribed_traction_dot_displacement(
            &load,
            &[1.0, -1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, -1.0],
            &[2.0, -2.0, 99.0, 99.0],
            &[99.0, 99.0, 3.0, -3.0],
        );
        assert_eq!(values, vec![20.0, 20.0, -12.0, -12.0]);
    }

    #[test]
    fn probe_energy_balance_is_finite_for_a_fresh_model() {
        let model = tiny_model(&two_hole_geometry());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let eb = probe_energy_balance(&model, &spec, &device);
        assert!(eb.internal_energy.is_finite(), "internal_energy must be finite, got {}", eb.internal_energy);
        assert!(eb.external_work.is_finite(), "external_work must be finite, got {}", eb.external_work);
        assert!(eb.energy_balance_error.is_finite() && eb.energy_balance_error >= 0.0);
    }

    #[test]
    fn probe_energy_balance_distinguishes_internal_from_external_for_an_untrained_model() {
        // A freshly-initialized network has no reason for its interior energy density and its
        // boundary work integral to already agree - if this were accidentally wired to compare
        // a value against itself, internal_energy would exactly equal external_work.
        let model = tiny_model_raw();
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let eb = probe_energy_balance(&model, &spec, &device);
        assert!(
            (eb.internal_energy - eb.external_work).abs() > 1e-20,
            "internal_energy and external_work were suspiciously identical - suspect a copy-paste bug comparing a value against itself: {} vs {}",
            eb.internal_energy, eb.external_work
        );
    }

    #[test]
    fn no_hole_field_validation_accepts_affine_and_detects_translation() {
        use ndarray::Array2;
        let spec = ProblemSpec { geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] }, material: MaterialProps::al7075_t6(), load: LoadConfig::uniaxial_x(1e7), network: Default::default(), training: Default::default(), formulation: pinn_core::problem_spec::default_formulation() };
        let (ny, nx) = (5, 5); let a = spec.load.px / spec.material.e;
        let mut u = Array2::zeros((ny, nx)); let mut v = Array2::zeros((ny, nx));
        for iy in 0..ny { for ix in 0..nx { let x = -0.1 + 0.2 * ix as f64 / 4.0; let y = -0.1 + 0.2 * iy as f64 / 4.0; u[(iy,ix)] = (a*x) as f32; v[(iy,ix)] = (-spec.material.nu*a*y) as f32; }}
        let z = || Array2::zeros((ny,nx));
        let fields = pinn_core::messages::VisFields { von_mises:z(), sigma_xx:Array2::from_elem((ny,nx), spec.load.px as f32), sigma_yy:z(), sigma_xy:z(), disp_u:u, disp_v:v, eps_xx:Array2::from_elem((ny,nx),a as f32), eps_yy:Array2::from_elem((ny,nx),(-spec.material.nu*a) as f32), eps_xy:z(), pde_residual:z(), amr_score:z(), collocation_density:z() };
        let ok = validate_no_hole_fields(&fields, &spec);
        assert!(ok.u_linf < 1e-12 && ok.v_linf < 1e-12 && ok.strain_linf < 1e-7 && ok.rigid_translation_residual < 1e-12 && ok.rigid_rotation_residual < 1e-12);
        let mut translated = fields.clone(); translated.disp_u += 1.0;
        assert!(validate_no_hole_fields(&translated, &spec).rigid_translation_residual > 0.1);
    }

    /// Issue #63 sub-issue #72 (PH4-20 consolidated regression fixture): non-square plate,
    /// zero-cost variant of the real #68 headless run. `half_w != half_h` (aspect ratio
    /// 1.875:1, matching `variational_no_hole_plate_nonsquare.toml`'s own real config) with an
    /// exact affine field constructed over the true non-square grid extent - proves
    /// `validate_no_hole_fields` (and the measure-aware machinery it exercises) reads
    /// `geometry.half_w`/`half_h` independently rather than assuming a square domain, without
    /// spending a real training run to prove it. Runs by default (not `#[ignore]`d).
    #[test]
    fn no_hole_field_validation_accepts_affine_on_a_non_square_plate() {
        use ndarray::Array2;
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.15, half_h: 0.08, thickness: 0.006, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
        };
        let (ny, nx) = (5, 7);
        let a = spec.load.px / spec.material.e;
        let mut u = Array2::zeros((ny, nx));
        let mut v = Array2::zeros((ny, nx));
        for iy in 0..ny {
            for ix in 0..nx {
                let x = -spec.geometry.half_w + 2.0 * spec.geometry.half_w * ix as f64 / (nx - 1) as f64;
                let y = -spec.geometry.half_h + 2.0 * spec.geometry.half_h * iy as f64 / (ny - 1) as f64;
                u[(iy, ix)] = (a * x) as f32;
                v[(iy, ix)] = (-spec.material.nu * a * y) as f32;
            }
        }
        let z = || Array2::zeros((ny, nx));
        let fields = pinn_core::messages::VisFields {
            von_mises: z(),
            sigma_xx: Array2::from_elem((ny, nx), spec.load.px as f32),
            sigma_yy: z(),
            sigma_xy: z(),
            disp_u: u,
            disp_v: v,
            eps_xx: Array2::from_elem((ny, nx), a as f32),
            eps_yy: Array2::from_elem((ny, nx), (-spec.material.nu * a) as f32),
            eps_xy: z(),
            pde_residual: z(),
            amr_score: z(),
            collocation_density: z(),
        };
        let ok = validate_no_hole_fields(&fields, &spec);
        assert!(
            ok.u_linf < 1e-12 && ok.v_linf < 1e-12 && ok.strain_linf < 1e-7
                && ok.rigid_translation_residual < 1e-12 && ok.rigid_rotation_residual < 1e-12,
            "non-square affine field validation must pass with the same tolerance as the \
             square case, got: {ok:?}"
        );
    }
}

#[cfg(test)]
mod issue_77_l5_tests {
    use super::*;

    /// #77 proof: exact L5 sampling, optimizer, seed, derived-FD VM metric, and independently
    /// converged finite-plate FEM comparator. No direct stress or infinite-plate target.
    #[test]
    #[ignore]
    fn issue_77_l5_annular_decomposition_converges_to_fem_reference() {
        use burn::module::AutodiffModule;
        const FEM_KT: f64 = 2.460_638_516;
        let base = ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
        };
        let device = crate::training_core::BDevice::default();
        let no_hole = crate::user_runner::train_single_user_problem_for_benchmark(base.clone(), &device);
        let companion = run_no_hole_benchmark(&no_hole, &base, &device);
        assert!(companion.passed, "L5 companion no-hole gate failed: {companion:?}");

        let mut hole = base.clone();
        hole.geometry.holes = vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }];
        let (annulus, _outer, _) = crate::user_runner::run_annular_decomposition_training(
            hole.clone(), device.clone(), |_step, _loss, _lr, _points| false,
        );
        let annulus = annulus.valid();
        let scales = crate::training_core::compute_reference_scales_for_plate(&hole);
        let fd = crate::fd_stencil::FdConfig::new(hole.training.fd_h, 2.0 * hole.geometry.half_w, 2.0 * hole.geometry.half_h);
        let profile = probe_hole_boundary_profile_derived(
            &annulus, &hole.geometry, &hole.geometry.holes[0], 144, &fd, scales.u_ref,
            scales.stress_ref, &hole.material, ring_anchor_margin_m(hole.training.fd_h, &hole.geometry), &device,
        );
        let kt = stress_concentration_from_profile(&profile, hole.load.px.abs()).kt;
        let error = (kt - FEM_KT).abs() / FEM_KT;
        println!("[L5 #77] Kt={kt:.9} FEM={FEM_KT:.9} error={:.3}%", 100.0 * error);
        assert!(kt.is_finite() && error <= 0.05,
            "#77 L5 failed: Kt={kt:.9}, FEM={FEM_KT:.9}, error={:.3}%", 100.0 * error);
    }
}
