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
    user_geometry::{HoleBc, HoleSpec, UserGeometry},
};

use crate::{
    energy::{dem_energy_loss, equilibrium_from_displacement_hessian_loss, hole_traction_loss_direct, neumann_loss},
    pinlug_problem::IdentityAnsatz,
    problem::{BoundaryValueProblem, ConflictGroup, DomainForwardOutputs, DomainState, LossTerm, B},
};

pub const USER_DOMAIN: DomainId = DomainId(0);

const LAM_INTERIOR_ENERGY: f32 = 1.0;
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
/// Base weight for `ExternalWorkTerm` - deliberately NOT tied to `LAM_INTERIOR_ENERGY` (the
/// TRUE `Π=U-W_ext` 1:1 ratio `ExternalWorkTerm` was originally given). bugSource-New #8's
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

/// Points sampled around each hole's circumference, per hole — a fixed, generous default;
/// not user-configurable in v1 (see `ProblemSpec`'s scope note).
const HOLE_RING_POINTS: usize = 64;
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
const SEED_INTERIOR: u64 = 90_210;

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
    /// Radial offset [m] applied outside each hole's radius when EXCLUDING collocation points
    /// from the FD-unsafe near-hole annulus (see [`Self::contains_for_collocation`]) — see
    /// [`Self::new`] for the derivation. No longer used to emit a training point-set/anchor
    /// term (bugSource-New #12 removed `HoleAnchorEnergyTerm`/`"hole_i_anchor"` — equilibrium
    /// is now enforced everywhere via the derived-stress Hessian, not just near the hole); the
    /// geometric "just outside the hole" concept itself stays useful for Kt measurement (see
    /// `probe_hole_boundary_profile`'s derived-stress-at-margin variant).
    anchor_margin_m: f64,
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
        let anchor_margin_m = ring_anchor_margin_m(fd_h, &geometry);
        Self { geometry, hole_names, anchor_margin_m }
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
            if self.contains_for_collocation(x, y) {
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

    // `constitutive_anchor_point_sets` intentionally NOT overridden here anymore (falls back
    // to the trait default, `vec![]`, matching Kirsch/pin-lug) — bugSource-New #12 removed the
    // near-ring anchor mechanism this fed (`HoleAnchorEnergyTerm`/`"hole_i_anchor"`): nothing
    // reads direct σ outside the hole ring anymore, so there is no "keep it honest" gap left
    // for an anchor to close. See `anchor_margin_m`'s doc comment for what's kept.
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
    point_set: &'static str,
    material: MaterialProps,
    ref_div2: f64,
}
impl LossTerm for EquilibriumTerm {
    fn name(&self) -> &'static str { "equilibrium" }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec![self.point_set] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn needs_hessian(&self) -> bool { true }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("equilibrium: domain missing");
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
struct ExternalWorkTerm {
    px: f64,
    py: f64,
    ref_energy: f32,
}
impl LossTerm for ExternalWorkTerm {
    fn name(&self) -> &'static str { "external_work" }
    fn domains(&self) -> Vec<DomainId> { vec![USER_DOMAIN] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["outer_boundary"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == USER_DOMAIN).expect("external_work: domain missing");
        let (nx, ny) = d.normals.clone().expect("external_work: normals must be Some");
        let n = d.raw_out.dims()[0];
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let work_density = nx.mul_scalar(self.px) * u + ny.mul_scalar(self.py) * v;
        work_density.mean().mul_scalar(-1.0 / self.ref_energy as f64)
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

// `HoleAnchorEnergyTerm` (a real-equilibrium-via-energy signal on the `"hole_i_anchor"` point
// set) was removed here as part of bugSource-New #12: it existed to keep direct σ "honest"
// near the hole ring against a degenerate near-zero-everything solution, a problem that no
// longer exists once `EquilibriumTerm` reads the displacement-Hessian-derived stress
// everywhere instead of direct σ (see `EquilibriumTerm`'s own updated doc comment).

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
        let sampling = UserSamplingStrategy::new(spec.geometry.clone(), spec.training.fd_h);
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

        // `ref_div2` normalizes `equilibrium`'s residual (units Pa/m - stress per length, from
        // `factor·u_xx`-type terms in `equilibrium_from_displacement_hessian_loss`) to O(1).
        //
        // NOT the `(px*cx)^2` formula every OTHER `equilibrium_*_loss` caller uses (Kirsch's
        // `step_physics`, `headless.rs`, `runner.rs`'s Kirsch path, and this term's own
        // earlier direct-σ version) - that `cx = sx/(2·fd_h·domain_width)` was calibrated for
        // a genuinely different quantity, the FIRST-derivative FD-divergence-of-stress
        // residual, whose own natural scale really does grow as `1/fd_h`. Reusing it here was
        // a real bug, found via the term-gradient diagnostic (bugSource-New #1/#12): at
        // production's `fd_h=1e-3`, `cx ~ 1/(fd_h·domain_width)` is astronomically large
        // (~1e23 once squared), so dividing by it crushed `equilibrium`'s reported loss AND
        // gradient to near-zero regardless of the real underlying residual - on BOTH the
        // no-hole case (genuinely near-zero curvature at the true solution, a red herring) AND
        // the real single-hole case (where curvature is definitely NOT near-zero at
        // convergence), the diagnostic showed the identical ~1e-7-1e-6 grad_norm, which is the
        // signature of an `fd_h`-independent quantity being divided by an `fd_h`-DEPENDENT
        // constant, not of a physically inert term.
        //
        // The correct scale has nothing to do with `fd_h` (the Hessian residual is a converged
        // FD approximation of a smooth quantity, not itself proportional to `1/fd_h`) -
        // `px / half_w` (Pa/m, matching the residual's own units) is the natural characteristic
        // scale instead.
        let char_length = self.spec.geometry.half_w.max(self.spec.geometry.half_h).max(1e-9);
        let eq_ref_div2 = (self.spec.load.px / char_length).powi(2).max(1.0);

        let mut terms: Vec<Box<dyn LossTerm>> = vec![
            Box::new(InteriorEnergyTerm { material: self.spec.material.clone(), ref_energy }),
            Box::new(EquilibriumTerm { point_set: "interior", material: self.spec.material.clone(), ref_div2: eq_ref_div2 }),
            Box::new(OuterTractionTerm {
                material: self.spec.material.clone(),
                ref_stress2,
                px: self.spec.load.px,
                py: self.spec.load.py,
            }),
            Box::new(ExternalWorkTerm { px: self.spec.load.px, py: self.spec.load.py, ref_energy }),
        ];
        for (hole, &name) in self.spec.geometry.holes.iter().zip(self.hole_names.iter()) {
            terms.push(Box::new(HoleBcTerm { point_set: name, bc: hole.bc, ref_stress2 }));
        }
        terms
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "interior_energy" => LAM_INTERIOR_ENERGY,
            "equilibrium" => LAM_EQUILIBRIUM_PLATE,
            "outer_traction" => LAM_OUTER_TRACTION,
            "external_work" => LAM_EXTERNAL_WORK,
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

/// Builds a `VisFields` for GUI display by evaluating `model` once over a
/// `[nx,ny]`-shaped normalized grid masked by `geometry.contains` — mirrors `runner.rs`'s
/// private `evaluate_vis_grid_mdem` (same mDEM direct-column read, same von Mises formula,
/// same Phase 14 strain/residual/AMR-score/density extension), adapted for `UserGeometry`'s
/// N-hole containment check instead of `GeometryConfig`'s single-hole one.
///
/// Phase 14 extension reuses the exact FD-stencil + physical-scale-before-derivative
/// convention `probe_hole_boundary_profile` already established for this same ansatz (see
/// that function's doc comment): `sigma_xx/yy/xy` are direct network outputs here, so an
/// independent FD-derived strain estimate is a genuine second measurement, and comparing it
/// against the direct stress via the material's constitutive law is exactly the
/// `constitutive_consistency` training term's per-point residual, now surfaced for display.
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
    use crate::network::fwd;
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor};
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
    let raw_net = fwd::<BInner>(model, stencil_coords, geometry.n_fourier(), device); // [5*n_act, 5], unscaled

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
        let sxx = center_vals[i_act * 5 + 2] as f64;
        let syy = center_vals[i_act * 5 + 3] as f64;
        let sxy = center_vals[i_act * 5 + 4] as f64;
        let vm = (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).sqrt();
        let dex = sxx - sxx_fd_v[i_act] as f64;
        let dey = syy - syy_fd_v[i_act] as f64;
        let dexy = sxy - sxy_fd_v[i_act] as f64;
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
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let stress_ref = spec.load.px.abs().max(spec.load.py.abs()).max(1.0);
    let u_ref = ((stress_ref / spec.material.e) * geometry.half_w) as f32;
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };
    let n_fourier = geometry.n_fourier();

    let mut residuals: Vec<f32> = Vec::new();

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    if !bnd_pts_phys.is_empty() {
        let n_bnd = bnd_pts_phys.len();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&bnd_norm, device), &fd, device);
        let raw = fwd::<BInner>(model, stencil, n_fourier, device);
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
        let raw = fwd::<BInner>(model, norm_pts_to_tensor::<BInner>(&ring_norm, device), n_fourier, device);
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
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let stress_ref = spec.load.px.abs().max(spec.load.py.abs()).max(1.0);
    let u_ref = ((stress_ref / spec.material.e) * geometry.half_w) as f32;
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
    let raw = fwd::<BInner>(model, stencil, geometry.n_fourier(), device);
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

/// `enhancement.md` Phase 10 ("Energy Validation") - a real domain-integrated internal-energy-
/// vs-external-work comparison. See `pinn_core::messages::EnergyBalance`'s doc comment for why
/// this is DISTINCT from the optimizer's own `energy_loss` field. Internal energy is a
/// Monte-Carlo estimate of `∫ (strain energy density) dA * thickness` over the plate's real
/// area (using the SAME interior sampling training itself uses); external work is `∮ t·u ds *
/// thickness` over the same arc-length-weighted outer-boundary point set `probe_reaction_force`
/// uses, halved for the same quasi-static-linear-loading `1/2` factor `energy::
/// dem_energy_per_point`'s own `1/2 * sigma:epsilon` formula carries (so both sides are on a
/// consistent basis).
pub fn probe_energy_balance(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
) -> pinn_core::messages::EnergyBalance {
    use crate::energy::{compute_stress, dem_energy_per_point};
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
    use crate::network::fwd;
    use crate::training_core::BInner;
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let stress_ref = spec.load.px.abs().max(spec.load.py.abs()).max(1.0);
    let u_ref = ((stress_ref / spec.material.e) * geometry.half_w) as f32;
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };

    let interior = sampling.sample_interior(&placeholder, spec.training.n_interior);
    let area = 4.0 * geometry.half_w * geometry.half_h
        - geometry.holes.iter().map(|h| std::f64::consts::PI * h.radius * h.radius).sum::<f64>();
    let internal_energy = if interior.is_empty() {
        0.0
    } else {
        let n_int = interior.len();
        let int_norm: Vec<[f32; 2]> = interior.iter().map(|&[x, y]| norm_pt(x, y)).collect();
        let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&int_norm, device), &fd, device);
        let raw = fwd::<BInner>(model, stencil, geometry.n_fourier(), device);
        let m = 5 * n_int;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_int, &fd);
        let energy_density = dem_energy_per_point::<BInner>(exx, eyy, exy, &spec.material);
        let mean_density: f64 = energy_density.into_data().to_vec::<f32>().unwrap_or_default()
            .iter().map(|&v| v as f64).sum::<f64>() / n_int as f64;
        mean_density * area * geometry.thickness
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
        let raw = fwd::<BInner>(model, stencil, geometry.n_fourier(), device);
        let m = 5 * n_bnd;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(px_pa),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled.clone(), n_bnd, &fd);
        let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &spec.material);
        let nx: Vec<f32> = bnd_pts_phys.iter().map(|p| p.nx as f32).collect();
        let ny: Vec<f32> = bnd_pts_phys.iter().map(|p| p.ny as f32).collect();
        let nx_t = Tensor::<BInner, 1>::from_data(TensorData::new(nx.clone(), vec![n_bnd]), device);
        let ny_t = Tensor::<BInner, 1>::from_data(TensorData::new(ny.clone(), vec![n_bnd]), device);
        let tx_pred = (sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone())
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let ty_pred = (sxy * nx_t + syy * ny_t)
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let u_vals: Vec<f32> = scaled.clone().slice([0..n_bnd, 0..1]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let v_vals: Vec<f32> = scaled.slice([0..n_bnd, 1..2]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let mut work = 0.0f64;
        for i in 0..n_bnd {
            let ds = if nx[i].abs() > 0.5 { ds_x_normal } else { ds_y_normal };
            work += (tx_pred[i] as f64 * u_vals[i] as f64 + ty_pred[i] as f64 * v_vals[i] as f64) * ds * geometry.thickness;
        }
        0.5 * work
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
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor};
    use crate::network::fwd;
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
        let x = hole.center[0] + hole.radius * theta.cos();
        let y = hole.center[1] + hole.radius * theta.sin();
        thetas.push(theta_deg);
        pts_phys.push((x, y));
        pts_norm.push([(x / half_w) as f32, (y / half_h) as f32]);
    }

    let pts_t = norm_pts_to_tensor::<BInner>(&pts_norm, device);
    let stencil = assemble_stencil::<BInner>(&pts_t, fd, device);
    let raw_stencil = fwd::<BInner>(model, stencil, geometry.n_fourier(), device); // [5n, 5]: u,v,sxx,syy,sxy

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
/// not concentration.
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
    use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor};
    use crate::network::fwd;
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
    let raw_stencil = fwd::<BInner>(model, stencil, geometry.n_fourier(), device);

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

/// Stress-concentration summary derived from a hole-boundary profile. `nominal_stress` is
/// the applied far-field traction magnitude - the standard Kt denominator for this problem
/// class. Deliberately NOT compared against a hardcoded Kt=3: that is the closed-form
/// result for an IDEALIZED INFINITE plate under uniaxial tension specifically - this plate
/// is finite, may carry biaxial/off-axis load, and may have other holes perturbing the
/// field, so a real discrepancy from 3.0 is expected, not itself evidence of a bug (see this
/// epic's own explicit instruction: "Do not hard-code Kt = 3 as a required answer").
pub fn stress_concentration_from_profile(profile: &[HoleBoundaryPoint], nominal_stress: f64) -> pinn_core::messages::StressConcentration {
    use pinn_core::messages::StressConcentration;
    let (max_theta_deg, max_von_mises) = profile.iter()
        .map(|p| (p.theta_deg, p.von_mises as f64))
        .fold((0.0, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
    let kt = if nominal_stress.abs() > 1e-300 { max_von_mises / nominal_stress.abs() } else { f64::NAN };
    StressConcentration { nominal_stress, max_von_mises, max_theta_deg, kt }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::user_geometry::HoleSpec;

    /// Representative `fd_h` for tests that don't otherwise have a `ProblemSpec.training.fd_h`
    /// in scope — matches the default most real TOML specs use (e.g. `single_hole_plate.toml`).
    const TEST_FD_H: f32 = 1e-3;

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

        let term = ExternalWorkTerm { px, py, ref_energy };
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

    #[test]
    fn named_point_sets_returns_one_ring_per_hole_with_expected_point_count() {
        // No more "_anchor" point sets since bugSource-New #12 removed the near-ring
        // constitutive-anchor mechanism - only one ring per hole now, matching Kirsch/pin-lug's
        // own hole/interface point-set shape.
        let geom = two_hole_geometry();
        let strategy = UserSamplingStrategy::new(geom.clone(), TEST_FD_H);
        let sets = strategy.named_point_sets(&[]);
        assert_eq!(sets.len(), 2, "2 holes * 1 ring = 2 named point sets");
        let names: Vec<&str> = sets.iter().map(|s| s.name).collect();
        for expected in ["hole_0", "hole_1"] {
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

    // ─── Phase 14 (Neural-Network-Wide Adaptive Collocation epic): spatial diagnostic fields ──

    /// `n_fourier` must match whatever geometry the model will actually be probed against -
    /// `0` for a no-hole geometry, `8` for a holed one (`UserGeometry::n_fourier`) - or the
    /// probe's forward pass panics on a tensor width mismatch (the model's `input_dim` is
    /// fixed at construction time; the probe functions derive their Fourier embedding from
    /// the geometry they're actually given, independently).
    fn tiny_model(n_fourier: usize) -> crate::network::ElasticityNet<crate::training_core::BInner> {
        let device = crate::training_core::BDevice::default();
        let input_dim = if n_fourier > 0 { 4 * n_fourier } else { 3 };
        crate::network::ElasticityNetConfig::new()
            .with_input_dim(input_dim).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device)
    }

    #[test]
    fn evaluate_user_vis_grid_masks_every_new_field_outside_the_domain_same_as_the_original_six() {
        let geometry = two_hole_geometry();
        let model = tiny_model(geometry.n_fourier());
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
        let model = tiny_model(0);
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
        let model = tiny_model(0);
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
        let model = tiny_model(0);
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
        let model = tiny_model(two_hole_geometry().n_fourier());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
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
        let model = tiny_model(0);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
        };
        let (rms, max) = probe_boundary_residuals(&model, &spec, &device);
        assert!(rms.is_finite() && rms >= 0.0);
        assert!(max.is_finite() && max >= 0.0);
    }

    // ─── enhancement.md Phase 9: force equilibrium ──────────────────────────────────────────

    #[test]
    fn probe_reaction_force_is_finite_and_reference_force_matches_hand_computed_nominal_load() {
        let geometry = two_hole_geometry();
        let model = tiny_model(geometry.n_fourier());
        let device = crate::training_core::BDevice::default();
        let px = 6.9e7;
        let spec = ProblemSpec {
            geometry: geometry.clone(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(px),
            network: Default::default(),
            training: Default::default(),
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
    fn probe_reaction_force_handles_a_geometry_with_no_holes() {
        let model = tiny_model(0);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
        };
        let rf = probe_reaction_force(&model, &spec, &device);
        assert!(rf.net_fx.is_finite() && rf.net_fy.is_finite());
        assert!(rf.equilibrium_error.is_finite() && rf.equilibrium_error >= 0.0);
    }

    #[test]
    fn probe_reaction_force_zero_load_gives_zero_reference_force_floored_and_finite_error() {
        // px = py = 0.0: `reference_force` would otherwise be exactly 0.0, which must not
        // produce a NaN/infinite division - the `.max(1e-30)` floor exists exactly for this.
        let model = tiny_model(0);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(0.0),
            network: Default::default(),
            training: Default::default(),
        };
        let rf = probe_reaction_force(&model, &spec, &device);
        assert!(rf.equilibrium_error.is_finite(), "equilibrium_error must stay finite at zero applied load, got {}", rf.equilibrium_error);
    }

    // ─── enhancement.md Phase 10: energy balance ────────────────────────────────────────────

    #[test]
    fn probe_energy_balance_is_finite_for_a_fresh_model() {
        let model = tiny_model(two_hole_geometry().n_fourier());
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: two_hole_geometry(),
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
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
        let model = tiny_model(0);
        let device = crate::training_core::BDevice::default();
        let spec = ProblemSpec {
            geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] },
            material: MaterialProps::al7075_t6(),
            load: pinn_core::loading::LoadConfig::uniaxial_x(6.9e7),
            network: Default::default(),
            training: Default::default(),
        };
        let eb = probe_energy_balance(&model, &spec, &device);
        assert!(
            (eb.internal_energy - eb.external_work).abs() > 1e-20,
            "internal_energy and external_work were suspiciously identical - suspect a copy-paste bug comparing a value against itself: {} vs {}",
            eb.internal_energy, eb.external_work
        );
    }
}
