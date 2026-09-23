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

/// Issue #78 second root-cause fix: how many "envelope's own physical transition widths" of
/// clearance to require between the envelope's characteristic length scale and the real FD
/// stencil step, when deriving [`target_phi_at_margin`] below. This is the ONE remaining
/// hand-picked number in the whole `saturation_scale` derivation chain - by design, not an
/// oversight: `target_phi_at_margin`'s own derivation (see its doc comment) proves the target
/// phi value that keeps the envelope FD-resolvable is a fixed multiple of `RING_ANCHOR_SAFETY_
/// FACTOR` (since `margin` and the real physical FD step are ALWAYS related by that exact
/// factor in this codebase - `ring_anchor_margin_m`'s own definition), so there is no further
/// geometry/discretization quantity left to derive it FROM without inventing one. Matches this
/// codebase's own existing precedent for this exact category of constant -
/// `fd_stencil::HESSIAN_FD_SAFETY_MULT` (10.0) is the same kind of "one empirically-verified
/// safety multiple, everything else derived" number, chosen the same way: `2.0` was picked
/// because it reproduces (see the worked check in `target_phi_at_margin`'s doc comment) almost
/// exactly the `target≈0.98` implied by this session's own best-measured trial (`triple_hole_
/// plate.toml` at a hand-picked `saturation_scale=30`, Kt=2.973/2.956, closest of every trial
/// to FEM ground truth ≈2.98-3.06/2.97-3.02) - not an independent guess, a value chosen to
/// match already-collected real evidence. `HOLE_MARGIN_FRACTION`'s own doc comment (below)
/// candidly draws the identical distinction ("deliberately conservative... a real, working
/// ratio already exercised elsewhere," not derived from first principles) - this codebase
/// consistently keeps exactly one disclosed safety multiple per FD-accuracy tradeoff rather
/// than pretending every constant can be eliminated.
///
/// **A more aggressive first attempt was tried and rejected during this derivation**: reusing
/// `RING_ANCHOR_SAFETY_FACTOR` itself as this factor (i.e. requiring the envelope's transition
/// width to equal exactly the training-safety margin) collapses `target_phi_at_margin` to the
/// fixed value `1 - exp(-1) ≈ 0.632` for every geometry - which is mathematically the WORST
/// possible choice: `traction_free_envelope_scaled`'s own `d(phi)/dr` at the margin is MAXIMIZED
/// near `target≈0.632` (a separate derivation, see `CLAUDE.md`'s "Issue #78" section), not
/// minimized. `RING_ANCHOR_SAFETY_FACTOR` answers a different question (how far to keep
/// COLLOCATION POINTS from the hole) and reusing its value for an unrelated FD-truncation-
/// accuracy question was a category error, caught before landing.
const ENVELOPE_FD_RESOLUTION_FACTOR: f64 = 2.0;

/// Issue #78 second root-cause fix: replaces the former hand-picked `TARGET_PHI_AT_MARGIN`
/// constant with a real, closed-form derivation - the dimensionless `phi` target `AnnulusAnsatz::
/// MultiHoleHardConstraint` should reach AT the real Kt-measurement point (`hole.radius +
/// margin`), DERIVED from a genuine FD-accuracy constraint rather than picked empirically.
///
/// **The constraint**: `traction_free_envelope_scaled`'s envelope has its own characteristic
/// physical transition length `L = hole_radius / saturation_scale` (the `hole_radius` cancels
/// out of this expression once `saturation_scale` is itself expressed via `margin` and
/// `target_phi` - see the worked algebra below - so `L` depends only on `margin` and
/// `target_phi`, never on hole radius). If `L` shrinks below the real physical FD stencil step
/// (`fd_step`), the central-difference strain computed AT the margin starts reflecting FD
/// truncation error from the envelope's own curvature, not real physics - the exact, now-
/// understood mechanism behind this session's earlier empirical finding that the constitutive
/// residual rose alongside Kt accuracy at higher hand-picked scales. Requiring
/// `L >= ENVELOPE_FD_RESOLUTION_FACTOR * fd_step` and solving for `target_phi` at equality
/// (the tightest value that still satisfies the constraint, i.e. maximum genuine gradient
/// signal) gives this function's formula.
///
/// **Worked algebra**: `multi_hole_saturation_scale`'s own definition gives `scale =
/// sqrt(-ln(1-target)) * hole_radius / margin`, so `L = hole_radius/scale = margin /
/// sqrt(-ln(1-target))` (hole_radius cancels). Setting `L = ENVELOPE_FD_RESOLUTION_FACTOR *
/// fd_step` and solving for `target`: `target = 1 - exp(-(margin / (ENVELOPE_FD_RESOLUTION_
/// FACTOR * fd_step))^2)`. `fd_step` is the real physical FD stencil step
/// (`fd_stencil::FdConfig`'s `hx`/`hy` converted to physical units - `fd_h * half_w.max
/// (half_h)`, the SAME quantity `ring_anchor_margin_m` is itself built from, before applying
/// `RING_ANCHOR_SAFETY_FACTOR`) - not a second, independently-guessed step size.
///
/// Because `margin = RING_ANCHOR_SAFETY_FACTOR * fd_step` always holds in this codebase
/// (`ring_anchor_margin_m`'s own definition), this reduces to a single closed-form constant
/// per fixed `RING_ANCHOR_SAFETY_FACTOR`/`ENVELOPE_FD_RESOLUTION_FACTOR` pair, independent of
/// plate size, hole radius, or `fd_h`'s own raw value - a genuine mathematical fact (not a
/// simplification chosen for convenience): both `margin` and `fd_step` scale together with
/// `fd_h`/plate size, so their ratio is fixed by construction. Kept as a function of `(margin,
/// fd_step)` rather than a hardcoded literal so it stays correct and traceable if either
/// upstream convention ever changes, and so the derivation is visible in one place rather than
/// re-derived from a bare number.
fn target_phi_at_margin(margin: f64, fd_step: f64) -> f64 {
    let ratio = margin / (ENVELOPE_FD_RESOLUTION_FACTOR * fd_step);
    1.0 - (-(ratio * ratio)).exp()
}

/// The real physical FD stencil step at a hole's own location - `fd_stencil::FdConfig`'s
/// normalized `hx`/`hy` (both equal to the caller's raw `fd_h`, see `FdConfig::new`) converted
/// to physical meters via that config's own `domain_width/2 = half_w`/`domain_height/2 =
/// half_h` normalization (every `FdConfig::new` call site in this codebase uses `domain_width =
/// 2*half_w`, `domain_height = 2*half_h` - confirmed, not assumed). Takes the larger of the two
/// axes (worst case), matching `ring_anchor_margin_m`'s own established convention exactly -
/// this IS the same quantity that function multiplies by `RING_ANCHOR_SAFETY_FACTOR`.
fn physical_fd_step_m(fd_h: f32, geometry: &UserGeometry) -> f64 {
    fd_h as f64 * geometry.half_w.max(geometry.half_h)
}

/// See [`target_phi_at_margin`]'s own doc comment. Solves `traction_free_envelope_scaled`'s
/// formula (`phi = 1 - exp(-(scale*u)^2)`, `u = margin/hole_radius`) for the `scale` that makes
/// `phi` reach the DERIVED `target_phi_at_margin(margin, fd_step)` at `r = hole_radius +
/// margin` - `scale = sqrt(-ln(1-target_phi)) / u`. `margin` is the SAME `ring_anchor_margin_m`
/// the real Kt diagnostic and every FD-safety exclusion already use - not a second,
/// independently-guessed distance.
fn multi_hole_saturation_scale(hole_radius: f64, margin: f64, fd_step: f64) -> f64 {
    let target_phi = target_phi_at_margin(margin, fd_step);
    let u = margin / hole_radius;
    (-(1.0 - target_phi).ln()).sqrt() / u
}

/// Issue #78 item 3: the closed-form-derived `saturation_scale` for every Free hole in `spec`,
/// IN THE SAME ORDER `free_holes(spec)`/`new_with_hard_constraint_ansatz`'s own `holes.into_
/// iter()` produces (load-bearing - a caller uses this to seed a trainable model's own
/// `hole_scales: Vec<Param<Tensor<B,1>>>`, which is indexed positionally against the ansatz's
/// `trainable_envelope_holes()` output, itself built from the same `free_holes` order).
/// Returns an empty `Vec` when N≤1 (no seed needed - `trainable_saturation_scale` never
/// applies there, see `ArchitectureSpec`'s own doc comment) - callers should treat an empty
/// result as "nothing to seed," not an error.
pub(crate) fn trainable_hole_scale_seeds(spec: &ProblemSpec) -> Vec<f64> {
    let holes = free_holes(spec);
    if holes.len() <= 1 {
        return Vec::new();
    }
    let margin = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    let fd_step = physical_fd_step_m(spec.training.fd_h, &spec.geometry);
    holes.into_iter().map(|hole| multi_hole_saturation_scale(hole.radius, margin, fd_step)).collect()
}

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
pub(crate) fn decomposition_applicable(spec: &ProblemSpec) -> bool {
    matches!(
        spec.geometry.holes.as_slice(),
        [HoleSpec { bc: HoleBc::Free, center, .. }] if center[0] == 0.0 && center[1] == 0.0
    )
}

/// Issue #78 (multi-hole Kt): the N-hole generalization of `decomposition_applicable`'s own
/// eligibility check, but for `AnnulusAnsatz::MultiHoleHardConstraint`'s construction gate
/// specifically — every hole with `bc == HoleBc::Free` is eligible for its own closed-form
/// correction, with NO count or centering restriction.
/// `kirsch_hole_correction::HoleTractionFreeAnsatz::physical_xy` already translates by
/// `hole_center` per-hole; centering was never a mathematical requirement of that math, only
/// `decomposition_applicable`'s own deliberate first-verified-case scope narrowing for the
/// SEPARATE kinematic-decomposition mechanism (`affine_strain_pair`/`use_decomposed` in
/// `UserDefinedProblem::loss_terms()`), which this function does not touch or replace.
fn free_holes(spec: &ProblemSpec) -> Vec<&HoleSpec> {
    spec.geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect()
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

/// Issue #77 Step 2 fix: `ring_anchor_margin_m` above is PLATE-scaled (`fd_h *
/// max(half_w,half_h)`), not hole-scaled — for a small hole this makes the exclusion
/// margin/radius ratio proportionally WORSE as the hole shrinks (confirmed real: at L5's
/// `radius=0.005`, the default margin is `4e-4 m = 0.08*radius`, i.e. the ring already sits
/// at `1.08*radius` — not "far" in absolute plate terms, but the wrong scale to reason about
/// for a hole whose own stress concentration lives entirely within a few hole-radii). This
/// constant instead defines the margin as a FIXED FRACTION of the hole's own radius,
/// independent of plate size — the ring sits at `(1+HOLE_MARGIN_FRACTION)*radius` for every
/// hole size, not just the ones that happen to be large relative to the plate.
///
/// Deliberately conservative (not pushed to the theoretical FD-safety minimum): `0.02` keeps
/// the ring's `r/a` ratio at `1.02`, matching what `single_hole_plate.toml`'s larger
/// `radius=0.02` case already got "for free" under the old plate-scaled formula (`4e-4/0.02 =
/// 0.02`) — this is a real, working ratio already exercised elsewhere in this codebase, not a
/// speculative new value.
const HOLE_MARGIN_FRACTION: f64 = 0.02;

/// The margin (physical, meters) used ONLY for the decomposition hole term's own FD-safe ring
/// (`"hole_i_fd"`/`"hole_0_fd"`, added in issue #77 Step 1) — see [`HOLE_MARGIN_FRACTION`]'s
/// doc comment. Deliberately a SEPARATE function from [`ring_anchor_margin_m`]: the interior
/// collocation exclusion (`UserSamplingStrategy::contains_for_collocation`,
/// `AnnularPartitionSampling`'s own inner sampling radius) is tied to the GLOBAL `fd_h` every
/// interior/boundary point's own stencil uses and must stay unchanged by this fix — only the
/// hole ring's OWN, separately-configurable FD step ([`hole_ring_fd_config`]) shrinks.
pub fn hole_ring_margin_m(radius: f64) -> f64 {
    (HOLE_MARGIN_FRACTION * radius).max(1e-9)
}

/// The (normalized-step) `FdConfig` the hole ring's OWN stencil must use so its physical reach
/// stays within [`hole_ring_margin_m`]'s tighter margin — inverts
/// [`ring_anchor_margin_m`]'s derivation (`margin = RING_ANCHOR_SAFETY_FACTOR * fd_h *
/// max(half_w,half_h)`) to solve for the `fd_h` a margin this small requires, rather than
/// reusing the training config's own (larger, plate-scaled) `fd_h`. Same `domain_width`/
/// `domain_height` (the real plate dimensions) as every other point set's `FdConfig` — only
/// the step size `h` differs, so `FdConfig::sx`/`sy` (used to convert the resulting FD
/// derivative back to physical units) stay correct.
pub fn hole_ring_fd_config(radius: f64, geometry: &UserGeometry) -> crate::fd_stencil::FdConfig {
    let margin = hole_ring_margin_m(radius);
    let h = (margin / (RING_ANCHOR_SAFETY_FACTOR * geometry.half_w.max(geometry.half_h))) as f32;
    crate::fd_stencil::FdConfig::new(h, 2.0 * geometry.half_w, 2.0 * geometry.half_h)
}

/// Convenience for `plate_multi_step_ctx`/`plate_multi_domain_step_ctx`'s callers: the tighter
/// [`hole_ring_fd_config`] only means anything for exactly the single-hole geometries
/// [`decomposition_applicable`]-style logic targets — every other case (no-hole, multi-hole)
/// has no `"_i_fd"` point set for it to ever apply to, so falling back to the caller's own
/// (unchanged) `fd` there is exactly as inert as it needs to be, not a special case to track.
pub fn hole_fd_config_for_geometry(fd: &crate::fd_stencil::FdConfig, geometry: &UserGeometry) -> crate::fd_stencil::FdConfig {
    match geometry.holes.as_slice() {
        [hole] => hole_ring_fd_config(hole.radius, geometry),
        _ => *fd,
    }
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
            // Issue #77 Step 1: FD-safe companion ring - the kinematic-decomposition hole
            // traction term needs a DERIVED stress read, and the exact-radius "hole_0" ring
            // above is stencil-unsafe for that (an inward FD arm would land inside the hole).
            // Issue #77 Step 2: radius uses `hole_ring_margin_m` (hole-relative, `0.02*radius`)
            // rather than `self.collocation_inner_radius` (plate-scaled, this sampler's own
            // interior-collocation inner bound) - a deliberately SEPARATE, tighter margin, paired
            // with `hole_ring_fd_config`'s own smaller FD step at the call site that reads this
            // point set's strains (see `MultiStepCtx::hole_fd`'s doc comment).
            let fd_radius = hole.radius + hole_ring_margin_m(hole.radius);
            let fd_points = self.interface.thetas.iter().map(|&theta| BoundaryPoint {
                x: hole.center[0] + fd_radius * theta.cos(),
                y: hole.center[1] + fd_radius * theta.sin(),
                nx: -theta.cos(), ny: -theta.sin(), tx: 0.0, ty: 0.0, kind: BoundaryKind::NeumannFree,
            }).collect();
            sets.push(NamedPointSet { name: "hole_0_fd", points: fd_points });
        }
        sets
    }
}

/// Issue #78 item 4: the OUTER domain's own sampling strategy for the N-hole generalization of
/// #77's annular decomposition (`MultiAnnularDecompositionProblem`) - the multi-hole analogue
/// of `AnnularPartitionSampling(is_annulus=false)`, generalized to exclude EVERY Free hole's own
/// interface circle (not just one) and to expose N sets of interface points (one per hole,
/// named via [`occurrence_suffixed_name`] so hole 0's own names stay byte-identical to the
/// original single-hole path - load-bearing for the N=1 regression proof).
///
/// Each ANNULUS domain, by contrast, reuses [`AnnularPartitionSampling`] completely UNCHANGED
/// (constructed with a synthetic single-hole [`UserGeometry`] for just that one hole) - only
/// the shared OUTER domain genuinely needs new multi-hole-aware logic, since it's the only
/// domain that must know about every hole at once.
pub struct MultiAnnularOuterSampling {
    /// The FULL, real N-hole geometry (every hole, Free and Fixed) - needed for `contains`/
    /// the plate boundary, unlike each annulus domain's own synthetic single-hole view.
    geometry: UserGeometry,
    /// One entry per Free hole, in the SAME order `annulus_domain_ids`/`interfaces` use.
    partitions: Vec<pinn_core::user_geometry::AnnularPartition>,
    /// The partner `DomainId` for each Free hole's own annulus domain, same order as `partitions`.
    annulus_domain_ids: Vec<DomainId>,
    /// Shared across every hole (same `HOLE_RING_POINTS` angle set each - geometry-independent).
    interface: std::sync::Arc<InterfaceParametrization>,
    /// Same derivation as `AnnularPartitionSampling::interface_trace_offset` (plate-scaled via
    /// `ring_anchor_margin_m`, not hole-scaled) - a single shared value is correct, not a
    /// per-hole simplification, since the ORIGINAL single-hole field is plate-scaled too.
    interface_trace_offset: f64,
    interior_calls: std::sync::atomic::AtomicU64,
}

impl MultiAnnularOuterSampling {
    pub fn new(
        geometry: UserGeometry,
        fd_h: f32,
        partitions: Vec<pinn_core::user_geometry::AnnularPartition>,
        annulus_domain_ids: Vec<DomainId>,
        interface: std::sync::Arc<InterfaceParametrization>,
    ) -> Self {
        assert_eq!(partitions.len(), annulus_domain_ids.len(),
            "one partition per annulus domain id - they're indexed together positionally");
        let interface_trace_offset = 0.5 * ring_anchor_margin_m(fd_h, &geometry);
        for p in &partitions {
            assert!(interface_trace_offset > 0.0 && interface_trace_offset < p.interface_radius - p.hole_radius,
                "#78 interface FD trace offset must fit inside every annulus");
        }
        Self { geometry, partitions, annulus_domain_ids, interface, interface_trace_offset, interior_calls: std::sync::atomic::AtomicU64::new(0) }
    }
}

impl DomainSamplingStrategy for MultiAnnularOuterSampling {
    fn sample_interior(&self, _geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
        use pinn_core::LcgRng;
        let call = self.interior_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = LcgRng::new(SEED_INTERIOR ^ call.wrapping_mul(CALL_SEED_MIX));
        let mut points = Vec::with_capacity(n);
        let mut attempts = 0usize;
        while points.len() < n && attempts < n * REJECTION_SAMPLE_ATTEMPTS_FACTOR {
            attempts += 1;
            let x = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_w;
            let y = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_h;
            if self.geometry.contains(x, y) && self.partitions.iter().all(|p| p.contains_outer(x, y)) {
                points.push([x, y]);
            }
        }
        points
    }

    fn sample_boundary(&self, _geom: &GeometryConfig, _load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
        // Identical to `AnnularPartitionSampling`'s own outer-boundary sampling - the plate's
        // own outer edge doesn't depend on how many holes it has.
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
        let mut sets = Vec::with_capacity(self.partitions.len() * 2);
        for (i, (partition, &partner)) in self.partitions.iter().zip(self.annulus_domain_ids.iter()).enumerate() {
            let interface_pts = self.interface.thetas.iter().map(|&theta| {
                let radial = [theta.cos(), theta.sin()];
                BoundaryPoint {
                    x: partition.center[0] + partition.interface_radius * radial[0],
                    y: partition.center[1] + partition.interface_radius * radial[1],
                    nx: -radial[0], ny: -radial[1], tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::Interface { partner_domain: partner },
                }
            }).collect();
            sets.push(NamedPointSet { name: occurrence_suffixed_name("interface", i), points: interface_pts });

            let trace_radius = partition.interface_radius + self.interface_trace_offset;
            let trace_pts = self.interface.thetas.iter().map(|&theta| BoundaryPoint {
                x: partition.center[0] + trace_radius * theta.cos(),
                y: partition.center[1] + trace_radius * theta.sin(),
                nx: -theta.cos(), ny: -theta.sin(), tx: 0.0, ty: 0.0,
                kind: BoundaryKind::Interface { partner_domain: partner },
            }).collect();
            sets.push(NamedPointSet { name: occurrence_suffixed_name("interface_outer_stress", i), points: trace_pts });
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
    /// Issue #77 Phase 1 architectural redesign (single-domain hard-constraint): fraction of
    /// `sample_interior`'s budget drawn from a near-hole-biased stratum instead of the plain
    /// whole-plate draw — see [`Self::sample_interior`]'s own doc comment for the mechanism.
    /// `0.0` via [`Self::new`] is byte-identical to every pre-existing caller; set only via
    /// [`Self::with_hole_bias`] (builder-style, so none of this struct's ~14 existing
    /// `UserSamplingStrategy::new` call sites need to change). Issue #78: generalized from
    /// exactly one centered hole to EVERY `HoleBc::Free` hole (any count, any position) — the
    /// total fraction is split evenly across however many Free holes exist; a geometry with
    /// zero Free holes makes this an unconditional no-op regardless of the fraction, same as
    /// `fraction=0.0` (nothing to bias toward).
    hole_bias_fraction: f64,
}

/// Outer radius of the near-hole-biased sampling stratum, as a multiple of hole radius — reuses
/// the SAME `3*radius` convention `AnnularPartitionSampling`'s own `interface_radius` and every
/// PH4-24..37 comparison's hard-constraint envelope saturation point already established, not a
/// newly chosen constant.
const HOLE_BIAS_RADIUS_MULTIPLIER: f64 = 3.0;

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
            hole_bias_fraction: 0.0,
        }
    }

    /// See [`Self::hole_bias_fraction`]'s own doc comment. Builder-style (consumes `self`) so
    /// every existing `UserSamplingStrategy::new(...)` call site stays byte-identical unless it
    /// explicitly opts in by chaining this.
    pub fn with_hole_bias(mut self, fraction: f64) -> Self {
        assert!((0.0..=1.0).contains(&fraction), "hole_bias fraction must be in [0,1], got {fraction}");
        self.hole_bias_fraction = fraction;
        self
    }

    /// See [`Self::hole_bias_fraction`]'s own doc comment — exposed so a caller can compute
    /// matching quadrature weights via [`hole_bias_quadrature_weights`] without duplicating
    /// this struct's own bias-fraction/geometry state.
    pub fn hole_bias_fraction(&self) -> f64 { self.hole_bias_fraction }

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

/// Issue #77 PH4-41 fix (finding 2): per-point quadrature weights compensating for
/// `UserSamplingStrategy::sample_interior`'s own hole-biased stratified sampling
/// (`hole_bias_fraction>0`, Phase 1's own addition). Reuses `pinn_core::amr::
/// compensation_weights` — the SAME "leaf_area*n/total_area" mechanism AMR already uses to
/// correct for its own non-uniform sampling density — rather than inventing a second
/// integration-weight scheme. Classifies each point as inside/outside the bias stratum purely
/// from its own distance to the hole center (the SAME `HOLE_BIAS_RADIUS_MULTIPLIER*hole.radius`
/// boundary `sample_interior` itself uses), so it works on any point set that sampler produced
/// without needing `sample_interior` to expose stratum membership directly.
///
/// `hole_bias_fraction<=0.0`, or a geometry with no `HoleBc::Free` holes at all, returns
/// all-`1.0` weights — byte-identical to `PhysicalPotentialEnergyTerm`'s existing
/// `domain_integral_tensor` (unweighted mean) path, since `domain_integral_weighted_tensor_
/// matches_unweighted_tensor_when_weights_are_uniform` already proves uniform weights of `1.0`
/// reduce to the same result.
///
/// Issue #78: generalized from one bias disk to N (one per `HoleBc::Free` hole, matching
/// `UserSamplingStrategy::sample_interior`'s own N-hole generalization) — a point is "in bias"
/// if it falls within ANY Free hole's own `HOLE_BIAS_RADIUS_MULTIPLIER*radius` disk. Assumes
/// (asserted, loudly) that no two Free holes' bias disks overlap — an untested edge case for
/// very closely-spaced holes, flagged rather than silently producing wrong quadrature weights
/// via double-counted area, per this codebase's "loud not silent" convention.
pub(crate) fn hole_bias_quadrature_weights(
    points_norm: &[[f32; 2]],
    geometry: &UserGeometry,
    hole_bias_fraction: f64,
) -> Vec<f64> {
    let free: Vec<&HoleSpec> = geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
    if free.is_empty() || hole_bias_fraction <= 0.0 {
        return vec![1.0; points_norm.len()];
    }
    for i in 0..free.len() {
        for j in (i + 1)..free.len() {
            let (dx, dy) = (free[i].center[0] - free[j].center[0], free[i].center[1] - free[j].center[1]);
            let sep = (dx * dx + dy * dy).sqrt();
            let sum_bias_r = HOLE_BIAS_RADIUS_MULTIPLIER * (free[i].radius + free[j].radius);
            assert!(sep >= sum_bias_r,
                "hole_bias_quadrature_weights: Free holes {i} and {j} have overlapping bias \
                 disks (separation={sep}, sum of bias radii={sum_bias_r}) - not handled, see \
                 this function's own doc comment");
        }
    }
    let points: Vec<[f64; 2]> = points_norm.iter()
        .map(|p| [p[0] as f64 * geometry.half_w, p[1] as f64 * geometry.half_h])
        .collect();
    let bias_r2: Vec<f64> = free.iter().map(|h| (HOLE_BIAS_RADIUS_MULTIPLIER * h.radius).powi(2)).collect();
    let in_bias = |p: &[f64; 2]| -> bool {
        free.iter().zip(&bias_r2).any(|(hole, &r2)| {
            let (dx, dy) = (p[0] - hole.center[0], p[1] - hole.center[1]);
            dx * dx + dy * dy <= r2
        })
    };
    let n_biased = points.iter().filter(|p| in_bias(p)).count();
    let n_remaining = points.len() - n_biased;
    // Physical area each stratum represents - the SAME areas `sample_interior`'s own uniform-
    // in-r^2 draw (biased strata) and whole-plate-minus-those-disks draw (remainder) are meant
    // to cover, independent of how many points actually landed in each this particular call.
    let all_hole_area: f64 = geometry.holes.iter().map(|h| std::f64::consts::PI * h.radius * h.radius).sum();
    let bias_area: f64 = free.iter().zip(&bias_r2).map(|(h, &r2)| {
        (std::f64::consts::PI * r2 - std::f64::consts::PI * h.radius * h.radius).max(0.0)
    }).sum();
    let plate_area = 4.0 * geometry.half_w * geometry.half_h - all_hole_area;
    let remaining_area = (plate_area - bias_area).max(0.0);
    let samples: Vec<pinn_core::amr::DensitySample> = points.iter().map(|&p| {
        let leaf_area = if in_bias(&p) {
            if n_biased > 0 { bias_area / n_biased as f64 } else { 0.0 }
        } else if n_remaining > 0 {
            remaining_area / n_remaining as f64
        } else {
            0.0
        };
        pinn_core::amr::DensitySample { point: p, leaf_area }
    }).collect();
    pinn_core::amr::compensation_weights(&samples)
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

        // Issue #77 Phase 1 architectural redesign: near-hole-biased stratum, opt-in via
        // `hole_bias_fraction` (0.0 = every pre-existing caller, making this block a complete
        // no-op — `bias_holes` is empty). Issue #78: generalized from exactly one hole to
        // EVERY `HoleBc::Free` hole, splitting the total fraction evenly across them (N=1
        // reduces byte-identically to the original single-hole formula — same `n_biased`,
        // same `max_attempts`, same RNG draw order, since the loop runs exactly once). Draws
        // uniform-in-r^2 over `[hole.radius + anchor_margin_m, HOLE_BIAS_RADIUS_MULTIPLIER *
        // hole.radius]` per hole — the SAME formula `AnnularPartitionSampling::sample_interior`'s
        // own `is_annulus` branch already uses and is already tested, adapted to a
        // single-domain sampler instead of a second domain. `contains_for_collocation` (not a
        // bespoke check) enforces the same FD-safety/plate-bounds/other-hole-exclusion
        // invariants every other draw in this function already relies on.
        let bias_holes: Vec<HoleSpec> = if self.hole_bias_fraction > 0.0 {
            self.geometry.holes.iter().copied().filter(|h| h.bc == HoleBc::Free).collect()
        } else {
            Vec::new()
        };
        let per_hole_fraction = if bias_holes.is_empty() { 0.0 } else { self.hole_bias_fraction / bias_holes.len() as f64 };
        for hole in &bias_holes {
            let n_biased = (n as f64 * per_hole_fraction).round() as usize;
            let r0 = hole.radius + self.anchor_margin_m;
            let r1 = HOLE_BIAS_RADIUS_MULTIPLIER * hole.radius;
            let (r0sq, r1sq) = (r0 * r0, r1 * r1);
            let mut attempts = 0usize;
            let max_attempts = n_biased.max(1) * REJECTION_SAMPLE_ATTEMPTS_FACTOR;
            let mut drawn = 0usize;
            while drawn < n_biased && attempts < max_attempts {
                attempts += 1;
                let r = (r0sq + rng.next_f64() * (r1sq - r0sq)).sqrt();
                let theta = 2.0 * std::f64::consts::PI * rng.next_f64();
                let x = hole.center[0] + r * theta.cos();
                let y = hole.center[1] + r * theta.sin();
                if self.contains_for_collocation(x, y) {
                    pts.push([x, y]);
                    drawn += 1;
                }
            }
        }
        // The bias strata's own outer radii, excluded from the draw below so the biased
        // regions' density isn't further inflated by double-counting — an empty `bias_holes`
        // (every pre-existing caller, or a geometry with no Free hole) makes `excludes_bias`
        // an unconditional `true` (`.all()` over an empty iterator), byte-identical.
        let bias_r1sq: Vec<(HoleSpec, f64)> = bias_holes.iter()
            .map(|&h| (h, (HOLE_BIAS_RADIUS_MULTIPLIER * h.radius).powi(2)))
            .collect();
        let excludes_bias = |x: f64, y: f64| -> bool {
            bias_r1sq.iter().all(|(h, r1sq)| {
                let (dx, dy) = (x - h.center[0], y - h.center[1]);
                dx * dx + dy * dy >= *r1sq
            })
        };

        let n_remaining = n.saturating_sub(pts.len());
        let nx = (n_remaining as f64).sqrt().ceil().max(1.0) as usize;
        let ny = n_remaining.div_ceil(nx.max(1)).max(1);
        let cells = nx * ny;
        for i in 0..n_remaining {
            let cell = i * cells / n_remaining;
            let ix = cell % nx;
            let iy = cell / nx;
            let jitter_x = rng.next_f64();
            let jitter_y = rng.next_f64();
            let x = -self.geometry.half_w + (ix as f64 + jitter_x) * 2.0 * self.geometry.half_w / nx as f64;
            let y = -self.geometry.half_h + (iy as f64 + jitter_y) * 2.0 * self.geometry.half_h / ny as f64;
            if self.contains_for_collocation(x, y) && excludes_bias(x, y) {
                pts.push([x, y]);
            }
        }
        let mut attempts = 0usize;
        let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;
        while pts.len() < n && attempts < max_attempts {
            attempts += 1;
            let x = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_w;
            let y = (rng.next_f64() * 2.0 - 1.0) * self.geometry.half_h;
            if self.contains_for_collocation(x, y) && excludes_bias(x, y) {
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
            // Issue #77 Step 1: FD-safe ring, same outward-into-the-hole normal convention as
            // the exact-radius ring above — used only by the kinematic-decomposition hole
            // traction term, which needs a DERIVED (constitutive) stress read and therefore a
            // stencil-safe radius, not the exact hole boundary.
            // Issue #77 Step 2: radius uses `hole_ring_margin_m` (hole-relative, `0.02*radius`)
            // rather than `self.anchor_margin_m` (plate-scaled, this sampler's own interior-
            // collocation exclusion) - deliberately separate, tighter, paired with
            // `hole_ring_fd_config`'s own smaller FD step at the call site.
            let r = hole.radius + hole_ring_margin_m(hole.radius);
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
    hole_fd: &'a crate::fd_stencil::FdConfig,
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
        hole_fd,
        per_domain_lr: None,
        k: 1.0, // IdentityAnsatz ignores k entirely — value is inert
        domains: vec![crate::problem::DomainStepCtx { data, u_ref, ref_energy, ref_stress2 }],
        dynamic_lam_h_cap: 50.0,
        dynamic_lam_d_cap: 50.0,
        dynamic_lam_penetration_cap: f64::MAX,
        dynamic_lam_non_tension_cap: f64::MAX,
        constitutive_consistency_weight: 50.0,
        n_fourier,
        coordinate_embedding,
        // Only `run_multi_annular_decomposition_training`'s own inline `MultiStepCtx` literal
        // ever needs a per-domain override (N different hole-relative charts) - neither of
        // this file's two shared single/two-domain builders is used by that N-hole path, so
        // `None` (the shared `coordinate_embedding` above, applied to every domain) is always
        // correct here.
        domain_coordinate_embeddings: None,
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
    hole_fd: &'a crate::fd_stencil::FdConfig,
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
        config, problem, fd, hole_fd,
        // Issue #77 Step 4: defaults to `None` here (every domain shares `lr_sched`'s single
        // LR, this function's own pre-Step-4 behavior). `run_annular_decomposition_training_
        // inner`, the one real caller that needs per-domain LR, sets `ctx.per_domain_lr`
        // AFTER calling this builder (a simple post-construction field mutation) rather than
        // this function growing yet another parameter only one caller would ever use non-None.
        per_domain_lr: None,
        k: 1.0,
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
        domain_coordinate_embeddings: None,
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

/// Issue #78 Stage 1.1: `HoleBcTerm::name` for the `occurrence`-th hole (0-indexed among holes
/// sharing this `bc`) - the FIRST hole of a given BC keeps the exact pre-#78 constant name
/// (`"hole_free"`/`"hole_fixed"`), so every existing single-hole or mixed-BC (e.g. one Free +
/// one Fixed) spec's term names, `base_weight` lookups, and diagnostics stay byte-identical.
/// Only the 2nd+ hole of the SAME bc (the actual colliding case) gets a numeric suffix. Leaks
/// the formatted string, matching `UserSamplingStrategy::new`'s own `hole_names`/`hole_fd_names`
/// per-index-leaked-`&'static str` convention for the identical reason: `LossTerm::name()`'s
/// trait contract is `&'static str`, and these names are computed once per problem (not
/// per-step), so the one-time leak is bounded and never repeats within a training run.
fn hole_bc_term_name(bc: HoleBc, occurrence: usize) -> &'static str {
    let base = match bc { HoleBc::Free => "hole_free", HoleBc::Fixed => "hole_fixed" };
    if occurrence == 0 {
        base
    } else {
        Box::leak(format!("{base}_{occurrence}").into_boxed_str())
    }
}

/// Issue #78 item 4: the SAME "first occurrence unsuffixed, 2nd+ suffixed" convention
/// `hole_bc_term_name` established, generalized for any `base` name - used for the OUTER
/// domain's own per-hole interface point-set names (`"interface"`/`"interface_outer_stress"`
/// for the first Free hole, `"interface_1"`/`"interface_outer_stress_1"` for the second, ...).
/// Load-bearing for the N=1 regression proof: at `occurrence=0` this is byte-identical to the
/// literal `"interface"`/`"interface_outer_stress"` strings `AnnularPartitionSampling` itself
/// already hardcodes, so a single-Free-hole `MultiAnnularOuterSampling` produces IDENTICAL
/// point-set names to the original single-hole path.
fn occurrence_suffixed_name(base: &'static str, occurrence: usize) -> &'static str {
    if occurrence == 0 {
        base
    } else {
        Box::leak(format!("{base}_{occurrence}").into_boxed_str())
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
    /// Issue #78 Stage 1.1 fix: this term's own identity for SAW-BRDR/diagnostics lookup
    /// (`lam_by_name`/`raw_scalar_by_name`/`term_grad_norms`, all `HashMap<&str, _>` keyed by
    /// `name()`). Before this field existed, `name()` returned the CONSTANT `"hole_free"`/
    /// `"hole_fixed"` regardless of which hole this term belonged to - two holes sharing a BC
    /// (e.g. `triple_hole_plate.toml`'s two Free holes) collided in every one of those maps,
    /// silently discarding one hole's own SAW-adapted weight and diagnostics in favor of
    /// whichever hole's entry was inserted last. Computed once in `loss_terms()` (see
    /// `hole_bc_term_name`'s own doc comment for the exact naming rule - unsuffixed for the
    /// FIRST hole of a given BC, matching every pre-#78 spec's term name byte-for-byte, so
    /// only the actual colliding case's behavior changes at all).
    name: &'static str,
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
/// Issue #78 item 4: the N-hole generalization of `InterfaceDisplacementContinuityTerm` - the
/// ONLY reason a new struct is needed (not a parameter added to the existing one) is that the
/// original hardcodes `point_sets() -> ["interface", "interface"]` (the SAME name on both
/// sides, correct for exactly one annulus/outer pair). Once the outer domain is SHARED across
/// N holes, each hole's own interface points need a per-hole-unique name on the outer side
/// (`occurrence_suffixed_name`) while the annulus side stays unsuffixed "interface" (each
/// annulus domain is independently scoped, never needs distinguishing). `right_point_set` is
/// the only real addition; `compute()` is copied verbatim (byte-identical math).
pub struct MultiInterfaceDisplacementContinuityTerm {
    pub left: DomainId,
    pub right: DomainId,
    pub right_point_set: &'static str,
    pub inv_u_ref_sq: f64,
}
impl LossTerm for MultiInterfaceDisplacementContinuityTerm {
    fn name(&self) -> &'static str { "interface_displacement_continuity" }
    fn domains(&self) -> Vec<DomainId> { vec![self.left, self.right] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface", self.right_point_set] }
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

/// Issue #78 item 4: the N-hole generalization of `InterfaceTractionContinuityTerm` - see
/// `MultiInterfaceDisplacementContinuityTerm`'s own doc comment for why a new struct (not a
/// modified existing one) is the right shape. `right_point_set` replaces the hardcoded
/// `"interface_outer_stress"` with a per-hole-suffixed name; the annulus side stays
/// unsuffixed `"interface_annulus_stress"` (domain-scoped, needs no distinguishing).
pub struct MultiInterfaceTractionContinuityTerm {
    pub left: DomainId,
    pub right: DomainId,
    pub right_point_set: &'static str,
    pub left_material: MaterialProps,
    pub right_material: MaterialProps,
    pub inv_ref_stress2: f64,
}
impl LossTerm for MultiInterfaceTractionContinuityTerm {
    fn name(&self) -> &'static str { "interface_traction_continuity" }
    fn domains(&self) -> Vec<DomainId> { vec![self.left, self.right] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface_annulus_stress", self.right_point_set] }
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
    fn name(&self) -> &'static str { self.name }
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
    /// Issue #77 Phase 1 architectural redesign: `Identity` (every pre-existing caller,
    /// byte-identical) or the SAME exact closed-form hard-constraint ansatz
    /// `AnnularDecompositionProblem::annulus_ansatz` uses — reused directly, not a new type,
    /// since it's already general enough (see `kirsch_hole_correction::AnnulusAnsatz`'s own doc
    /// comment). Set only via [`UserDefinedProblem::new_with_hard_constraint_ansatz`].
    ansatz: crate::kirsch_hole_correction::AnnulusAnsatz,
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
            spec, domains, sampling, ansatz: crate::kirsch_hole_correction::AnnulusAnsatz::Identity, hole_names,
            current_interior_weights: std::sync::Mutex::new(None),
        }
    }

    /// Issue #77 Phase 1 architectural redesign: single-domain hard-constraint. `false` produces
    /// the byte-identical `Self::new` result. Also opts into
    /// [`UserSamplingStrategy::with_hole_bias`] at `hole_bias_fraction` (`0.0` = today's plain
    /// whole-plate draw, matching `AnnularDecompositionProblem`'s own `n_annulus = n_interior/2`
    /// convention when set to `0.5`) — bundled into one constructor since the hard constraint's
    /// own effectiveness depends on the network actually seeing enough near-hole collocation
    /// density to learn the finite-plate correction beyond the exact infinite-plate closed form.
    ///
    /// Issue #78 (multi-hole Kt): generalized from requiring exactly one centered `Free` hole
    /// (`decomposition_applicable`) to [`free_holes`]'s own broader eligibility — EVERY
    /// `HoleBc::Free` hole, any count, any position, each gets its own closed-form correction
    /// (`AnnulusAnsatz::MultiHoleHardConstraint`). Requires at least one Free hole (asserted, not
    /// silently ignored — a hard constraint requested on a geometry with no eligible hole at all
    /// is a real misconfiguration, not a graceful fallback). A single-element case is exactly
    /// the old `HardConstraint` computation, wrapped in the new variant — see
    /// `multi_hole_reduces_to_single_hole_hard_constraint_when_n_equals_one`
    /// (`kirsch_hole_correction.rs`) for the byte-identical-output proof.
    pub fn new_with_hard_constraint_ansatz(spec: ProblemSpec, use_hard_constraint: bool, hole_bias_fraction: f64) -> Self {
        let mut problem = Self::new(spec.clone());
        if hole_bias_fraction > 0.0 {
            problem.sampling = problem.sampling.with_hole_bias(hole_bias_fraction);
        }
        if use_hard_constraint {
            let holes = free_holes(&spec);
            assert!(!holes.is_empty(),
                "hard-constraint ansatz requires at least one Free hole - see free_holes");
            let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
            // Issue #78 root-cause fix: `saturation_scale` - see `traction_free_envelope_
            // scaled`'s own doc comment for the full derivation. STRICTLY `1.0` when exactly
            // one Free hole is eligible (`holes.len() == 1`, load-bearing for byte-identical
            // behavior with `AnnulusAnsatz::HardConstraint` - proven by `multi_hole_reduces_to_
            // single_hole_hard_constraint_when_n_equals_one`), which keeps every single-hole
            // spec through this exact constructor - INCLUDING PH4-42's own real, already-
            // verified L5 result (0.67-1.23% of FEM), which goes through this identical function
            // - completely untouched by this fix. Once `holes.len() > 1`, each hole gets its OWN
            // DERIVED scale from `multi_hole_saturation_scale(hole.radius, margin, fd_step)` - no
            // hand-picked raw magnitude, self-adjusting to that hole's own radius/margin ratio AND
            // (via `target_phi_at_margin`) to the real FD-resolvability constraint (see its own
            // doc comment).
            let margin = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
            let fd_step = physical_fd_step_m(spec.training.fd_h, &spec.geometry);
            let multi_hole = holes.len() > 1;
            // Issue #78 item 3: `trainable_saturation_scale` only ever applies once N>1 (see
            // `ArchitectureSpec.trainable_saturation_scale`'s own doc comment - N=1's
            // `saturation_scale=1.0` is exact by construction, nothing to learn). The closed-
            // form-derived value is STILL computed and stored either way - it's the SEED a
            // trainable model's own `hole_scales` Param is initialized from (see
            // `trainable_hole_scale_seeds`), never discarded.
            let trainable = multi_hole && spec.architecture.trainable_saturation_scale;
            let sub_ansatzes = holes.into_iter().map(|hole| {
                let saturation_scale = if multi_hole { multi_hole_saturation_scale(hole.radius, margin, fd_step) } else { 1.0 };
                crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                    hole_center: hole.center, hole_radius: hole.radius,
                    half_w: spec.geometry.half_w, half_h: spec.geometry.half_h,
                    px: spec.load.px, py: spec.load.py,
                    e: spec.material.e as f64, nu: spec.material.nu as f64,
                    u_ref: scales.u_ref as f64, saturation_scale, trainable,
                }
            }).collect();
            problem.ansatz = crate::kirsch_hole_correction::AnnulusAnsatz::MultiHoleHardConstraint(sub_ansatzes);
        }
        problem
    }

    /// `true` iff `ansatz` is `AnnulusAnsatz::HardConstraint` or `MultiHoleHardConstraint` — the
    /// single source of truth for "should the soft `hole_free` penalty be skipped" and "should
    /// the affine background be subtracted from the network's own learning target" (see
    /// `loss_terms()`'s `affine_strain_pair` gate), mirroring `AnnularDecompositionProblem::
    /// hard_constraint_active`'s exact same pattern (that struct stays single-hole-only,
    /// untouched by issue #78). `pub(crate)` so `user_runner.rs`'s own Kt-diagnostic call sites
    /// can key their ansatz/affine reconstruction off the SAME source of truth `loss_terms()`
    /// used during training, instead of independently guessing - see this session's real,
    /// previously-unfixed `run_headless_user_problem` bug this generalization surfaced
    /// (`docs/multi-hole-fem-ground-truth-investigation.md`'s own write-up).
    pub(crate) fn hard_constraint_active(&self) -> bool {
        matches!(self.ansatz,
            crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(_)
                | crate::kirsch_hole_correction::AnnulusAnsatz::MultiHoleHardConstraint(_))
    }

    pub fn spec(&self) -> &ProblemSpec { &self.spec }

    /// See `UserSamplingStrategy::hole_bias_fraction`'s own doc comment - exposed so a caller
    /// can compute matching quadrature weights (`hole_bias_quadrature_weights`) without
    /// reaching into this struct's own private `sampling` field.
    pub fn hole_bias_fraction(&self) -> f64 { self.sampling.hole_bias_fraction() }

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
        // Issue #78 (multi-hole Kt): `affine_strain_pair`'s own activation generalizes from
        // `decomposed` alone (`decomposition_applicable`'s single-centered-hole scope) to
        // `decomposed || self.hard_constraint_active()` - a strict OR, so every pre-existing
        // `decomposed=true` case (with or without the hard constraint) keeps its exact prior
        // `Some((px,py))` value, byte-for-byte. The NEW case this adds is `decomposed=false,
        // hard_constraint_active()=true` (an off-center or multi-hole `MultiHoleHardConstraint`
        // spec): `PhysicalPotentialEnergyTerm`'s `affine_strain` field adds a spatially UNIFORM
        // constant strain - independent of hole count/position by construction - so there is no
        // reason this relief should stay scoped to the narrower kinematic-decomposition
        // eligibility. Without this, the network's own (ansatz-suppressed) output has to learn
        // the ENTIRE affine far-field background from scratch on top of refining the hole
        // correction - the exact gradient-competition failure mode the original L5 fix existed
        // to eliminate, just reintroduced for every off-center/multi-hole hard-constraint spec.
        // Confirmed as the real root cause of this session's own measured Kt≈0.46-0.59 (vs FEM's
        // ≈2.9-3.1) on `triple_hole_plate.toml`'s real off-center 2-Free-hole geometry before
        // this fix - not a training-hyperparameter/undersampling issue (a 33% larger
        // n_interior/n_boundary run reproduced the same stuck Kt).
        let affine_active = decomposed || self.hard_constraint_active();
        let affine_strain_pair = affine_active.then_some((self.spec.load.px, self.spec.load.py));

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
        // Issue #78 Stage 1.1: per-BC occurrence counters feeding `hole_bc_term_name` - see
        // that function's own doc comment for why only the 2nd+ same-BC hole's name changes.
        let (mut free_occurrence, mut fixed_occurrence) = (0usize, 0usize);
        for (i, (hole, &name)) in self.spec.geometry.holes.iter().zip(self.hole_names.iter()).enumerate() {
            if hole.bc == HoleBc::Fixed || hole_free_active {
                let use_decomposed = decomposed && hole.bc == HoleBc::Free;
                // Issue #77 Phase 1: under the hard-constraint ansatz the traction-free
                // condition is already exact by construction (`kirsch_hole_correction`'s own
                // module doc comment) - registering the soft `hole_free` penalty on top would
                // be redundant at best and reintroduce the exact gradient-competition problem
                // this ansatz exists to eliminate. Issue #78: generalized from "this IS the
                // one decomposed hole" (`use_decomposed`, which stays scoped to
                // `decomposition_applicable`'s single-centered-hole case) to "this Free hole is
                // covered by the active hard-constraint ansatz" - `MultiHoleHardConstraint`
                // covers every `HoleBc::Free` hole regardless of `decomposed`, so the older,
                // narrower `use_decomposed &&` prefix is dropped; `Fixed` holes are never
                // matched here (`hole.bc == HoleBc::Free` guards it) so Stage 1's own
                // already-correct N-hole soft-penalty handling for them is unaffected.
                if hole.bc == HoleBc::Free && self.hard_constraint_active() {
                    continue;
                }
                let point_set = if use_decomposed { self.sampling.hole_fd_names[i] } else { name };
                let affine_target = if use_decomposed { affine_strain_pair } else { None };
                let occurrence = match hole.bc {
                    HoleBc::Free => { let o = free_occurrence; free_occurrence += 1; o }
                    HoleBc::Fixed => { let o = fixed_occurrence; fixed_occurrence += 1; o }
                };
                terms.push(Box::new(HoleBcTerm {
                    domain: USER_DOMAIN, point_set, bc: hole.bc, ref_stress2,
                    material: self.spec.material.clone(), affine_target,
                    name: hole_bc_term_name(hole.bc, occurrence),
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
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            // Issue #78 Stage 1.1: `hole_bc_term_name` suffixes the 2nd+ hole sharing a BC
            // (e.g. "hole_free_1") - every such hole gets the SAME base weight as the first
            // (there's no per-hole-specific weight tuning, only per-BC-type), so match by
            // prefix rather than requiring a second per-index match arm.
            other if other == "hole_free" || other.starts_with("hole_free_") => LAM_HOLE_FREE,
            other if other == "hole_fixed" || other.starts_with("hole_fixed_") => LAM_HOLE_FIXED,
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
    /// The outer domain's own ansatz — always `IdentityAnsatz`, unaffected by anything below.
    outer_ansatz: IdentityAnsatz,
    /// The annulus domain's own ansatz — `Identity` (byte-identical to every pre-PH4-35
    /// caller) unless `new_with_hard_constraint_ansatz` selected the hard-constraint mode.
    /// See `kirsch_hole_correction::AnnulusAnsatz`'s own doc comment.
    annulus_ansatz: crate::kirsch_hole_correction::AnnulusAnsatz,
    /// Issue #77 candidate (b): base weight for BOTH `interface_displacement_continuity` and
    /// `interface_traction_continuity` (previously a hardcoded `100.0` literal in
    /// `base_weight()`). Defaults to `100.0` via `new()` - byte-identical to every pre-#77-
    /// candidate-(b) caller. `new_with_interface_weight` exists solely to run a real,
    /// controlled A/B comparison testing whether this weight over-constrains the annulus
    /// field toward the outer field's smoothness at the interface - see
    /// `PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-28 open-question list.
    interface_weight: f32,
    /// Issue #77 next candidate: when `true`, adds a strong-form `EquilibriumTerm` (the SAME
    /// struct/loss the single-domain `UserDefinedProblem`'s `Strong`/`Hybrid` formulations
    /// already use, reused verbatim here - not a new implementation) on the ANNULUS domain's
    /// own "interior" point set. Every axis raised so far (representation, collocation
    /// margin, sampling variance, training-dynamics/LR, interface-continuity weight) has been
    /// tested with real evidence and ruled out or fixed without closing the Kt gap - this
    /// tests the working hypothesis that pure variational/DEM energy minimization
    /// under-resolves a concentration this sharp because a domain-INTEGRATED energy term's
    /// gradient at any one point is diluted by the whole domain's integral, while a strong-
    /// form residual supplies gradient pressure LOCALLY (`PHASE_4_IMPLEMENTATION_MANIFEST.md`'s
    /// PH4-29). Defaults to `false` via `new()` - byte-identical to every existing caller.
    include_annulus_equilibrium: bool,
    /// Issue #77 gradient-share hypothesis (PH4-34): real diagnostic evidence collected across
    /// every prior comparison run in this investigation shows `hole_free`'s RAW loss converges
    /// to a tiny residual (~1e-4, effectively satisfied) by step ~1500, yet its GRADIENT SHARE
    /// climbs back up to 70-90% of the entire optimization's gradient budget by step 2999 —
    /// while `physical_potential` (raw ~1.0, far from converged) gets only ~15% and
    /// `annulus_potential` ~3%. The optimizer spends most of its late-training gradient budget
    /// re-polishing an already-satisfied boundary condition instead of the actual energy
    /// functional that shapes the stress field. Defaults to `LAM_HOLE_FREE` (100.0) via `new()`
    /// - byte-identical to every pre-PH4-34 caller. `new_with_hole_free_weight` exists to test
    /// whether reducing this weight frees gradient budget for the energy terms without
    /// un-satisfying the (already nearly-exact) traction-free condition.
    hole_free_weight: f32,
    /// Issue #77 Phase 3 architectural redesign: `false` (every pre-existing caller) keeps the
    /// annulus domain's existing `SingleHoleChart` embedding, byte-identical. `true` (via
    /// `new_with_log_polar_embedding`) switches it to `CoordinateEmbedding::LogPolar` instead —
    /// see that embedding's own doc comment (`pinn_core::user_geometry`) for the full rationale.
    /// Orthogonal to `annulus_use_hard_constraint`/`annulus_n_fourier`/`annulus_use_siren` —
    /// combinable with the hard-constraint ansatz (representation and traction-free enforcement
    /// are independent axes), though not exercised combined with Fourier features or SIREN in
    /// this pass (log-polar is itself the representation change; stacking it with another one
    /// would confound which change caused any observed effect).
    use_log_polar_embedding: bool,
}

impl AnnularDecompositionProblem {
    pub fn supports(spec: &ProblemSpec) -> bool {
        matches!(spec.formulation, pinn_core::problem_spec::FormulationSelection::Variational)
            && spec.training.measure_aware_training
            && matches!(spec.geometry.holes.as_slice(), [HoleSpec { bc: HoleBc::Free, .. }])
            && spec.geometry.annular_partition().is_some()
    }

    pub fn new(spec: ProblemSpec) -> Self {
        Self::new_experimental(spec, 100.0, false, LAM_HOLE_FREE, false, false)
    }

    /// See `interface_weight`'s own doc comment.
    pub fn new_with_interface_weight(spec: ProblemSpec, interface_weight: f32) -> Self {
        Self::new_experimental(spec, interface_weight, false, LAM_HOLE_FREE, false, false)
    }

    /// See `include_annulus_equilibrium`'s own doc comment.
    pub fn new_with_annulus_equilibrium(spec: ProblemSpec, include_annulus_equilibrium: bool) -> Self {
        Self::new_experimental(spec, 100.0, include_annulus_equilibrium, LAM_HOLE_FREE, false, false)
    }

    /// See `hole_free_weight`'s own doc comment (PH4-34).
    pub fn new_with_hole_free_weight(spec: ProblemSpec, hole_free_weight: f32) -> Self {
        Self::new_experimental(spec, 100.0, false, hole_free_weight, false, false)
    }

    /// Issue #77 PH4-35: the exact, closed-form hard-constraint hole ansatz — see
    /// `kirsch_hole_correction`'s own module doc comment for the full design and rationale.
    /// When `true`, the annulus domain's `"hole_free"` soft-penalty term is not registered at
    /// all (see `loss_terms()`'s own comment) — it would be redundant with an already-exact
    /// constraint, and penalizing an already-exact residual serves no purpose.
    pub fn new_with_hard_constraint_ansatz(spec: ProblemSpec, use_hard_constraint: bool) -> Self {
        Self::new_experimental(spec, 100.0, false, LAM_HOLE_FREE, use_hard_constraint, false)
    }

    /// Issue #77 Phase 3 architectural redesign: see `use_log_polar_embedding`'s own doc
    /// comment. Combinable with the hard-constraint ansatz (`use_hard_constraint`) since
    /// representation and traction-free enforcement are independent axes.
    pub fn new_with_log_polar_embedding(spec: ProblemSpec, use_log_polar_embedding: bool, use_hard_constraint: bool) -> Self {
        Self::new_experimental(spec, 100.0, false, LAM_HOLE_FREE, use_hard_constraint, use_log_polar_embedding)
    }

    /// Issue #77 Phase 3: `true` iff `use_log_polar_embedding` was set — the CALLER
    /// (`user_runner::run_annular_decomposition_training_inner`) is the single source of truth
    /// for the actual embedding value (it also owns `annulus_n_fourier`, which this flag is
    /// deliberately independent of — see `use_log_polar_embedding`'s own doc comment on why
    /// the two aren't combined), so this struct only exposes the flag, not a computed
    /// embedding, to avoid two places that could silently disagree on how `n_fourier` combines.
    pub fn use_log_polar_embedding(&self) -> bool { self.use_log_polar_embedding }

    pub(crate) fn new_experimental(
        spec: ProblemSpec, interface_weight: f32, include_annulus_equilibrium: bool, hole_free_weight: f32,
        use_hard_constraint: bool, use_log_polar_embedding: bool,
    ) -> Self {
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
        let hole = &spec.geometry.holes[0];
        let annulus_ansatz = if use_hard_constraint {
            let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
            crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(
                crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                    hole_center: hole.center,
                    hole_radius: hole.radius,
                    half_w: spec.geometry.half_w,
                    half_h: spec.geometry.half_h,
                    px: spec.load.px,
                    py: spec.load.py,
                    e: spec.material.e as f64,
                    nu: spec.material.nu as f64,
                    u_ref: scales.u_ref as f64,
                    saturation_scale: 1.0,
                    trainable: false,
                },
            )
        } else {
            crate::kirsch_hole_correction::AnnulusAnsatz::Identity
        };
        Self {
            domains: vec![
                DomainSpec { id: ANNULUS_DOMAIN, geometry: placeholder.clone(), material: spec.material.clone(), output_dim: 5 },
                DomainSpec { id: OUTER_DOMAIN, geometry: placeholder, material: spec.material.clone(), output_dim: 5 },
            ],
            spec,
            annulus_sampling,
            outer_sampling,
            outer_ansatz: IdentityAnsatz,
            annulus_ansatz,
            interface_weight,
            include_annulus_equilibrium,
            hole_free_weight,
            use_log_polar_embedding,
        }
    }

    pub fn spec(&self) -> &ProblemSpec { &self.spec }

    /// `true` iff `annulus_ansatz` is `AnnulusAnsatz::HardConstraint` — the single source of
    /// truth for "should the soft `hole_free` penalty be skipped" (`loss_terms()`) rather than
    /// a second, independently-set boolean that could drift out of sync with the ansatz.
    fn hard_constraint_active(&self) -> bool {
        matches!(self.annulus_ansatz, crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(_))
    }
}

/// Annular U contribution to the one global potential. Its name remains distinct from the
/// outer contribution so loss ledgers can report both physical pieces. `step_physics_multi`
/// pins both names to coefficient one, so SAW-BRDR cannot distort U_annulus + U_outer - W_ext.
struct AnnularPotentialEnergyTerm {
    /// Issue #78 item 4: which annulus domain this term integrates over - was hardcoded to
    /// the single frozen `ANNULUS_DOMAIN` constant (correct when there's only ever one), now a
    /// real field so `MultiAnnularDecompositionProblem` can construct one instance per Free
    /// hole's own annulus domain. `AnnularDecompositionProblem`'s own N=1 construction sites
    /// set this to `ANNULUS_DOMAIN` explicitly - byte-identical behavior, not a default.
    domain: DomainId,
    /// Issue #78 item 4: was hardcoded `"annulus_potential"` - a real field so N annulus
    /// domains' own energy terms get DISTINCT names (`occurrence_suffixed_name`), avoiding the
    /// exact SAME silent-`HashMap`-overwrite bug class `hole_bc_term_name` was already fixed
    /// for (`HoleBcTerm`, issue #78 Stage 1) - unnamed/duplicate-named terms would silently
    /// collide in `training_core.rs`'s `lam_by_name`/`term_grad_norms` bookkeeping. Every
    /// N=1 construction site sets this to the literal `"annulus_potential"` - byte-identical.
    name: &'static str,
    material: MaterialProps,
    domain_area: f64,
    thickness: f64,
    ref_energy_absolute: f64,
    /// Issue #77 fix: see `PhysicalPotentialEnergyTerm::affine_strain`'s doc comment — same
    /// mechanism, applied to the annulus domain's own strain read.
    affine_strain: Option<(f64, f64)>,
}

impl LossTerm for AnnularPotentialEnergyTerm {
    fn name(&self) -> &'static str { self.name }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Physics }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Weak }
    fn derivative_order(&self) -> Option<crate::problem::DerivativeOrder> { Some(crate::problem::DerivativeOrder::First) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::PhysicalFunctional }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|d| d.domain == self.domain)
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
        match self.domains[domain_idx].id {
            ANNULUS_DOMAIN => &self.annulus_ansatz,
            OUTER_DOMAIN => &self.outer_ansatz,
            _ => unreachable!(),
        }
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
                domain: ANNULUS_DOMAIN, name: "annulus_potential",
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
        // PH4-35: under the hard-constraint ansatz, the traction-free condition is already
        // exact by construction (see `kirsch_hole_correction`'s module doc comment) -
        // registering the soft `hole_free` penalty on top would be redundant at best (its
        // target is already satisfied to FD-truncation precision) and would reintroduce
        // exactly the gradient-competition problem this ansatz exists to eliminate.
        if decomposed && !self.hard_constraint_active() {
            terms.push(Box::new(HoleBcTerm {
                domain: ANNULUS_DOMAIN, point_set: "hole_0_fd", bc: HoleBc::Free,
                ref_stress2: scales.ref_stress2, material: self.spec.material.clone(),
                affine_target: affine_strain_pair,
                // Single-hole-only scope (`decomposition_applicable` requires exactly one
                // centered Free hole) - always the first (and only) hole of its BC.
                name: hole_bc_term_name(HoleBc::Free, 0),
            }));
        }
        if self.include_annulus_equilibrium {
            // See `include_annulus_equilibrium`'s own doc comment (PH4-29's working
            // hypothesis) - the SAME `EquilibriumTerm` struct the single-domain
            // `UserDefinedProblem`'s Strong/Hybrid formulations already use, reused verbatim
            // (it's already generic over `domain`/`point_set`, no new struct needed), scoped
            // to the annulus domain's own "interior" point set. `ref_div2` reuses
            // `stress_per_length2`, the SAME normalization `UserDefinedProblem::loss_terms()`
            // derives for its own `EquilibriumTerm` - not a new formula.
            terms.push(Box::new(EquilibriumTerm {
                domain: ANNULUS_DOMAIN, point_set: "interior",
                material: self.spec.material.clone(), ref_div2: scales.stress_per_length2,
            }));
        }
        terms
    }
    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "annulus_potential" | "physical_potential" => LAM_PHYSICAL_POTENTIAL,
            "interface_displacement_continuity" | "interface_traction_continuity" => self.interface_weight,
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            "hole_free" => self.hole_free_weight,
            "equilibrium" => LAM_EQUILIBRIUM_PLATE,
            other => panic!("AnnularDecompositionProblem::base_weight: unknown term '{other}'"),
        }
    }
    fn phase1_steps(&self) -> usize { 0 }
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
    fn convergence_target(&self) -> f64 { 0.0 }
}

/// Issue #78 item 4: the N-hole generalization of `AnnularDecompositionProblem` - N annulus
/// domains (each single-hole-scoped exactly like `AnnularDecompositionProblem`'s own one,
/// reusing `AnnularPartitionSampling`/`AnnulusAnsatz::HardConstraint` completely UNCHANGED via
/// a synthetic single-hole `UserGeometry` per Free hole) sharing ONE outer domain/model
/// (`MultiAnnularOuterSampling`, the one genuinely new sampling strategy this needs). This is
/// the key architectural choice that keeps the blast radius small: log-polar embedding needs
/// NO change at all (each annulus model stays relative to exactly one hole), and
/// `AnnularDecompositionProblem` itself is completely untouched (a real, deliberate parallel
/// struct, not a refactor of the frozen single-hole path - matching this codebase's own
/// `step_physics`/`step_physics_multi` precedent for exactly this situation).
///
/// **Deliberately NOT wired into `TrainingProcedure::SequentialTwoStage`** - that path's own
/// N-hole generalization (Stage A needs N `OuterInterfaceAnchorTerm`s, Stage B needs N annulus
/// models trained as one `step_physics_multi` call) is real, separate, additional scope, not
/// attempted in this pass - a genuine, disclosed follow-up, not silently glossed over. Only
/// `TrainingProcedure::Joint` (the default) dispatches through this struct for N>1 Free holes.
pub struct MultiAnnularDecompositionProblem {
    spec: ProblemSpec,
    domains: Vec<DomainSpec>,
    /// One id per Free hole's own annulus domain, in the SAME order `annulus_samplings`/
    /// `annulus_ansatzes`/`spec.geometry.holes.iter().filter(Free)` all use.
    annulus_domain_ids: Vec<DomainId>,
    outer_domain_id: DomainId,
    annulus_samplings: Vec<AnnularPartitionSampling>,
    outer_sampling: MultiAnnularOuterSampling,
    outer_ansatz: IdentityAnsatz,
    /// One ansatz per Free hole - `AnnulusAnsatz::HardConstraint` (built from a real, per-hole
    /// `HoleTractionFreeAnsatz` - NOT `MultiHoleHardConstraint`, since these holes are
    /// geometrically SEPARATE domains here, not sharing one model) or `Identity`, matching
    /// `use_hard_constraint`.
    annulus_ansatzes: Vec<crate::kirsch_hole_correction::AnnulusAnsatz>,
    interface_weight: f32,
}

/// Shared by `MultiAnnularDecompositionProblem::new` (builds each annulus domain's sampling/
/// ansatz), `domain_coordinate_embeddings` (derives each annulus model's own input width), and
/// `multi_annular_hole_kt_diagnostics` (must probe each trained model through the SAME geometry
/// it was trained relative to, or `embedding_for_model` panics on a width it doesn't recognize -
/// a real bug this factoring-out fixes, see that diagnostic function's own doc comment). A
/// single source of truth for "what geometry does hole i's own annulus model see" - three
/// independently-written copies previously risked silently diverging.
fn single_hole_geometry_for(full: &UserGeometry, hole: &HoleSpec) -> UserGeometry {
    UserGeometry {
        half_w: full.half_w, half_h: full.half_h, thickness: full.thickness,
        holes: vec![HoleSpec { center: hole.center, radius: hole.radius, bc: HoleBc::Free }],
    }
}

impl MultiAnnularDecompositionProblem {
    /// The N-hole analogue of `AnnularDecompositionProblem::supports` - every hole must be
    /// `Free` (a `Fixed` hole has no annulus-decomposition machinery - it stays in the outer
    /// domain via the existing soft-penalty `hole_fixed` term, unaffected), and
    /// `UserGeometry::annular_partitions` must succeed (every interface circle fits inside the
    /// plate AND no two overlap - see that function's own doc comment).
    pub fn supports(spec: &ProblemSpec) -> bool {
        matches!(spec.formulation, pinn_core::problem_spec::FormulationSelection::Variational)
            && spec.training.measure_aware_training
            && !spec.geometry.holes.is_empty()
            && spec.geometry.holes.iter().all(|h| h.bc == HoleBc::Free)
            && spec.geometry.annular_partitions().is_some()
    }

    pub fn new(spec: ProblemSpec, use_hard_constraint: bool) -> Self {
        assert!(Self::supports(&spec),
            "multi-hole annular decomposition requires every hole Free, non-overlapping interface circles, Variational formulation, and measure-aware training");
        let placeholder = spec.geometry.to_placeholder();
        let free_holes: Vec<&HoleSpec> = spec.geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
        let n = free_holes.len();
        let outer_domain_id = DomainId(9);
        let annulus_domain_ids: Vec<DomainId> = (0..n).map(|i| DomainId(10 + i as u32)).collect();

        let interface = phase2_interface_parametrization();
        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);

        let mut domains = Vec::with_capacity(n + 1);
        let mut annulus_samplings = Vec::with_capacity(n);
        let mut annulus_ansatzes = Vec::with_capacity(n);
        for (hole, &annulus_id) in free_holes.iter().zip(annulus_domain_ids.iter()) {
            // Each annulus domain reuses `AnnularPartitionSampling`/`AnnulusAnsatz::
            // HardConstraint` COMPLETELY UNCHANGED via a synthetic single-hole `UserGeometry`
            // (same plate dimensions, exactly this one hole) - the existing, already-proven
            // single-hole machinery never needs to know it's one of several.
            let single_hole_geometry = single_hole_geometry_for(&spec.geometry, hole);
            annulus_samplings.push(AnnularPartitionSampling::new(
                single_hole_geometry, spec.training.fd_h, true, outer_domain_id, interface.clone(),
            ));
            let ansatz = if use_hard_constraint {
                crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(
                    crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                        hole_center: hole.center, hole_radius: hole.radius,
                        half_w: spec.geometry.half_w, half_h: spec.geometry.half_h,
                        px: spec.load.px, py: spec.load.py,
                        e: spec.material.e as f64, nu: spec.material.nu as f64,
                        u_ref: scales.u_ref as f64, saturation_scale: 1.0, trainable: false,
                    },
                )
            } else {
                crate::kirsch_hole_correction::AnnulusAnsatz::Identity
            };
            annulus_ansatzes.push(ansatz);
            domains.push(DomainSpec { id: annulus_id, geometry: placeholder.clone(), material: spec.material.clone(), output_dim: 5 });
        }
        domains.push(DomainSpec { id: outer_domain_id, geometry: placeholder, material: spec.material.clone(), output_dim: 5 });

        let partitions = spec.geometry.annular_partitions().expect("validated by Self::supports above");
        let outer_sampling = MultiAnnularOuterSampling::new(
            spec.geometry.clone(), spec.training.fd_h, partitions, annulus_domain_ids.clone(), interface,
        );

        Self {
            spec, domains, annulus_domain_ids, outer_domain_id,
            annulus_samplings, outer_sampling, outer_ansatz: IdentityAnsatz,
            annulus_ansatzes, interface_weight: 100.0,
        }
    }

    fn hard_constraint_active(&self, i: usize) -> bool {
        matches!(self.annulus_ansatzes[i], crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(_))
    }

    /// Issue #78 item 4 follow-up: closes the `MultiStepCtx.coordinate_embedding`-is-one-
    /// shared-value gap this problem's own doc comment (and `run_multi_annular_decomposition_
    /// training`'s) previously disclosed as a real, unfixed accuracy limitation. One
    /// `CoordinateEmbedding` per domain, in `self.domains()` order (annulus domains first, then
    /// the outer domain) - each annulus domain gets a genuine `SingleHoleChart` relative to ITS
    /// OWN hole, recomputed from the SAME synthetic single-hole `UserGeometry` shape `new()`
    /// already builds that hole's sampling/ansatz from above (never a second, potentially-
    /// diverging geometry - if `new()`'s own construction ever changes, this must change with
    /// it). The outer domain stays `Raw`, matching the original single-hole
    /// `AnnularDecompositionProblem`'s own convention (annulus gets chart features, outer
    /// doesn't - the outer domain represents only a small residual correction post-kinematic-
    /// decomposition, the same rationale that keeps it plain-tanh there too). The caller MUST
    /// build each annulus model with the matching `input_dim()` and populate `MultiStepCtx::
    /// domain_coordinate_embeddings` with this Vec, in this exact order, or `compute_domain_
    /// forwards` panics on a width mismatch (loud, not silent - see that function's own
    /// model-input-width dispatch).
    pub fn domain_coordinate_embeddings(&self) -> Vec<pinn_core::user_geometry::CoordinateEmbedding> {
        let free_holes: Vec<&HoleSpec> = self.spec.geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
        let mut out = Vec::with_capacity(self.domains.len());
        for hole in &free_holes {
            out.push(single_hole_geometry_for(&self.spec.geometry, hole).coordinate_embedding());
        }
        out.push(pinn_core::user_geometry::CoordinateEmbedding::Raw);
        out
    }

    /// Issue #78 item 4 follow-up: the exact synthetic single-hole `UserGeometry` free-hole `i`'s
    /// own annulus model was trained relative to (same one `new()`/`domain_coordinate_
    /// embeddings()` use internally). Diagnostics that probe that model directly (`multi_
    /// annular_hole_kt_diagnostics`) MUST pass this, not `spec.geometry` (the full N-hole
    /// geometry) - `embedding_for_model` derives its embedding from whatever `UserGeometry` it's
    /// given, and the full geometry's own `coordinate_embedding()` is `MultiHoleChart` (a
    /// DIFFERENT width) for N>1, which the annulus model was never built with.
    pub fn free_hole_geometry(&self, i: usize) -> UserGeometry {
        let free_holes: Vec<&HoleSpec> = self.spec.geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
        single_hole_geometry_for(&self.spec.geometry, free_holes[i])
    }
}

impl BoundaryValueProblem for MultiAnnularDecompositionProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }

    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        let id = self.domains[domain_idx].id;
        if let Some(i) = self.annulus_domain_ids.iter().position(|&d| d == id) {
            &self.annulus_samplings[i]
        } else if id == self.outer_domain_id {
            &self.outer_sampling
        } else {
            unreachable!("domain id {id:?} not owned by this problem")
        }
    }

    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        let id = self.domains[domain_idx].id;
        if let Some(i) = self.annulus_domain_ids.iter().position(|&d| d == id) {
            &self.annulus_ansatzes[i]
        } else if id == self.outer_domain_id {
            &self.outer_ansatz
        } else {
            unreachable!("domain id {id:?} not owned by this problem")
        }
    }

    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let scales = crate::training_core::compute_reference_scales_for_plate(&self.spec);
        let partitions = self.spec.geometry.annular_partitions().expect("validated by constructor");
        let annulus_areas: Vec<f64> = partitions.iter()
            .map(|p| std::f64::consts::PI * (p.interface_radius.powi(2) - p.hole_radius.powi(2)))
            .collect();
        let interface_area_total: f64 = partitions.iter().map(|p| std::f64::consts::PI * p.interface_radius.powi(2)).sum();
        let outer_area = 4.0 * self.spec.geometry.half_w * self.spec.geometry.half_h - interface_area_total;
        let total_area = annulus_areas.iter().sum::<f64>() + outer_area;
        let ref_energy_absolute = scales.ref_energy as f64 * total_area * self.spec.geometry.thickness;
        let per_edge = (self.spec.training.n_boundary / 4).max(1);
        let mut ds_per_point = Vec::with_capacity(per_edge * 4);
        for _ in 0..per_edge {
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
        }

        // Issue #78 item 4: the SAME affine-relief mechanism `AnnularDecompositionProblem`
        // uses, generalized - the closed-form uniform-tension background is independent of
        // hole count/position by construction (see `UserDefinedProblem::loss_terms()`'s own
        // matching comment on this exact generalization for the single-domain path), so it
        // applies unconditionally here (every hole is Free by `Self::supports`'s own gate).
        let affine_strain_pair = Some((self.spec.load.px, self.spec.load.py));

        let mut terms: Vec<Box<dyn LossTerm>> = vec![
            Box::new(PhysicalPotentialEnergyTerm {
                domain: self.outer_domain_id, material: self.spec.material.clone(),
                px: self.spec.load.px, py: self.spec.load.py, measure_aware: true,
                domain_area: outer_area, thickness: self.spec.geometry.thickness,
                ref_energy: scales.ref_energy, ref_energy_absolute, interior_weights: None,
                ds_per_point, affine_strain: affine_strain_pair,
            }),
            Box::new(TranslationGaugeTerm {
                domain: self.outer_domain_id,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }),
            Box::new(RotationGaugeTerm {
                domain: self.outer_domain_id, half_w: self.spec.geometry.half_w, half_h: self.spec.geometry.half_h,
            }),
        ];

        let free_holes: Vec<&HoleSpec> = self.spec.geometry.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
        for (i, (&annulus_id, hole)) in self.annulus_domain_ids.iter().zip(free_holes.iter()).enumerate() {
            terms.push(Box::new(AnnularPotentialEnergyTerm {
                domain: annulus_id, name: occurrence_suffixed_name("annulus_potential", i),
                material: self.spec.material.clone(), domain_area: annulus_areas[i],
                thickness: self.spec.geometry.thickness, ref_energy_absolute,
                affine_strain: affine_strain_pair,
            }));
            terms.push(Box::new(MultiInterfaceDisplacementContinuityTerm {
                left: annulus_id, right: self.outer_domain_id,
                right_point_set: occurrence_suffixed_name("interface", i),
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }));
            terms.push(Box::new(MultiInterfaceTractionContinuityTerm {
                left: annulus_id, right: self.outer_domain_id,
                right_point_set: occurrence_suffixed_name("interface_outer_stress", i),
                left_material: self.spec.material.clone(), right_material: self.spec.material.clone(),
                inv_ref_stress2: 1.0 / (scales.ref_stress2 as f64).max(1e-30),
            }));
            if !self.hard_constraint_active(i) {
                terms.push(Box::new(HoleBcTerm {
                    domain: annulus_id, point_set: "hole_0_fd", bc: HoleBc::Free,
                    ref_stress2: scales.ref_stress2, material: self.spec.material.clone(),
                    affine_target: affine_strain_pair,
                    // Each annulus domain's own sampling always emits "hole_0"/"hole_0_fd"
                    // (domain-scoped, built from a synthetic single-hole geometry) - but the
                    // TERM's own name must still be unique across domains for SAW-BRDR
                    // bookkeeping, same `hole_bc_term_name` convention every other multi-hole
                    // BC term in this codebase already uses.
                    name: hole_bc_term_name(HoleBc::Free, i),
                }));
            }
            let _ = hole;
        }
        terms
    }

    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "physical_potential" => LAM_PHYSICAL_POTENTIAL,
            other if other == "annulus_potential" || other.starts_with("annulus_potential_") => LAM_PHYSICAL_POTENTIAL,
            "interface_displacement_continuity" | "interface_traction_continuity" => self.interface_weight,
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            other if other == "hole_free" || other.starts_with("hole_free_") => LAM_HOLE_FREE,
            other => panic!("MultiAnnularDecompositionProblem::base_weight: unknown term '{other}'"),
        }
    }
    fn phase1_steps(&self) -> usize { 0 }
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
    fn convergence_target(&self) -> f64 { 0.0 }
}

/// Issue #77 Phase 2 architectural redesign: sequential two-stage training with ONE-DIRECTIONAL
/// domain coupling, replacing simultaneous joint optimization's symmetric interface-continuity
/// terms — see the investigation-branch plan's own Phase 2 design for the full rationale (PH4-37
/// found that a symmetric two-way interface constraint lets two independently-parameterized
/// networks drift toward a jointly-cheaper-but-wrong configuration once gradient budget is
/// redirected away from it; a one-directional anchor cannot exhibit that failure mode, since
/// only one side is ever free to move to satisfy it).
///
/// Builds the SAME `InterfaceParametrization` shape `AnnularDecompositionProblem::new_experimental`
/// uses (`HOLE_RING_POINTS` angles) — deliberately NOT shared/reused from that function (frozen-
/// adjacent, heavily tested code this module's own convention treats as risk-averse to touch);
/// a few duplicated lines here is the smaller-blast-radius choice.
pub(crate) fn phase2_interface_parametrization() -> std::sync::Arc<InterfaceParametrization> {
    std::sync::Arc::new(InterfaceParametrization {
        thetas: (0..HOLE_RING_POINTS)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / HOLE_RING_POINTS as f64)
            .collect(),
    })
}

/// Stage A's own loss term: anchors the OUTER domain's interface trace to the EXACT closed-form
/// field (`u_affine + kirsch_hole_displacement`, evaluated once per point at construction time,
/// not a live annulus network — there isn't one during Stage A) instead of a symmetric
/// consistency constraint against a second network. Same normalization convention
/// (`inv_u_ref_sq`) as `InterfaceDisplacementContinuityTerm`, whose raw-column-read pattern this
/// mirrors — `raw_out`'s `u,v` columns are already PHYSICAL displacement (meters) by the time
/// `compute()` sees them (the ansatz's multiplicative/additive contribution and the `u_ref`
/// rescale both happen earlier, in `compute_domain_forwards`).
struct OuterInterfaceAnchorTerm {
    domain: DomainId,
    target_u: Vec<f32>,
    target_v: Vec<f32>,
    inv_u_ref_sq: f64,
}
impl LossTerm for OuterInterfaceAnchorTerm {
    fn name(&self) -> &'static str { "outer_interface_anchor" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> { Some(crate::problem::BoundaryOperatorKind::Dirichlet) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("outer_interface_anchor: domain missing");
        let n = d.raw_out.dims()[0];
        let device = d.raw_out.device();
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let target_u = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(self.target_u.clone(), [n]), &device);
        let target_v = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(self.target_v.clone(), [n]), &device);
        let du = u - target_u;
        let dv = v - target_v;
        (du.clone() * du + dv.clone() * dv).mean().mul_scalar(self.inv_u_ref_sq)
    }
}

/// Stage B's own loss term: anchors the ANNULUS domain's interface trace to Stage A's now-
/// FROZEN outer model's own forward-pass output at the interface points — a fixed target
/// recomputed fresh each step from the frozen model (which is in inference mode, no gradient
/// graph), so this is structurally a one-directional Dirichlet-style anchor, not a symmetric
/// constraint: only the annulus model's own parameters ever receive a gradient from this term.
struct FrozenInterfaceAnchorTerm {
    domain: DomainId,
    target_u: Vec<f32>,
    target_v: Vec<f32>,
    inv_u_ref_sq: f64,
}
impl LossTerm for FrozenInterfaceAnchorTerm {
    fn name(&self) -> &'static str { "frozen_interface_anchor" }
    fn domains(&self) -> Vec<DomainId> { vec![self.domain] }
    fn point_sets(&self) -> Vec<&'static str> { vec!["interface"] }
    fn conflict_group(&self) -> ConflictGroup { ConflictGroup::Bc }
    fn formulation_kind(&self) -> crate::problem::FormulationKind { crate::problem::FormulationKind::Strong }
    fn boundary_kind(&self) -> Option<crate::problem::BoundaryOperatorKind> { Some(crate::problem::BoundaryOperatorKind::Dirichlet) }
    fn term_role(&self) -> crate::problem::TermRole { crate::problem::TermRole::Constraint }
    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
        let d = inputs.iter().find(|i| i.domain == self.domain).expect("frozen_interface_anchor: domain missing");
        let n = d.raw_out.dims()[0];
        let device = d.raw_out.device();
        let u = d.raw_out.clone().slice([0..n, 0..1]).reshape([n]);
        let v = d.raw_out.clone().slice([0..n, 1..2]).reshape([n]);
        let target_u = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(self.target_u.clone(), [n]), &device);
        let target_v = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(self.target_v.clone(), [n]), &device);
        let du = u - target_u;
        let dv = v - target_v;
        (du.clone() * du + dv.clone() * dv).mean().mul_scalar(self.inv_u_ref_sq)
    }
}

/// Stage A BVP: the OUTER domain alone, anchored to the exact closed-form field at the
/// interface instead of a live annulus network. Reuses `AnnularPartitionSampling` (the SAME
/// sampler `AnnularDecompositionProblem` already uses for its own outer domain) and
/// `PhysicalPotentialEnergyTerm`/gauge terms unchanged — only the interface term differs.
pub struct OuterStageProblem {
    spec: ProblemSpec,
    domains: Vec<DomainSpec>,
    sampling: AnnularPartitionSampling,
    ansatz: IdentityAnsatz,
}
impl OuterStageProblem {
    pub fn new(spec: ProblemSpec) -> Self {
        assert!(AnnularDecompositionProblem::supports(&spec),
            "sequential two-stage training requires the same scope AnnularDecompositionProblem does");
        let placeholder = spec.geometry.to_placeholder();
        let interface = phase2_interface_parametrization();
        let sampling = AnnularPartitionSampling::new(spec.geometry.clone(), spec.training.fd_h, false, ANNULUS_DOMAIN, interface);
        Self {
            domains: vec![DomainSpec { id: OUTER_DOMAIN, geometry: placeholder, material: spec.material.clone(), output_dim: 5 }],
            spec, sampling, ansatz: IdentityAnsatz,
        }
    }
}
impl BoundaryValueProblem for OuterStageProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }
    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        assert_eq!(domain_idx, 0);
        &self.sampling
    }
    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        assert_eq!(domain_idx, 0);
        &self.ansatz
    }
    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let scales = crate::training_core::compute_reference_scales_for_plate(&self.spec);
        let partition = self.spec.geometry.annular_partition().expect("validated by constructor");
        let outer_area = 4.0 * self.spec.geometry.half_w * self.spec.geometry.half_h
            - std::f64::consts::PI * partition.interface_radius.powi(2);
        let ref_energy_absolute = scales.ref_energy as f64 * outer_area * self.spec.geometry.thickness;
        let per_edge = (self.spec.training.n_boundary / 4).max(1);
        let mut ds_per_point = Vec::with_capacity(per_edge * 4);
        for _ in 0..per_edge {
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_h / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
            ds_per_point.push(2.0 * self.spec.geometry.half_w / per_edge as f64);
        }
        let hole = &self.spec.geometry.holes[0];
        // Issue #77 PH4-41 fix (finding 3): `PhysicalPotentialEnergyTerm`'s own `affine_strain`
        // is now `Some(...)`, matching `AnnularDecompositionProblem`'s own established outer-
        // domain convention exactly - Stage A's model represents `u_hole` (the correction)
        // alone, same as Stage B's `AnnulusStageProblem`, so `evaluate_frozen_outer_interface_
        // displacement`'s direct raw-output read (no affine addition of its own) is now
        // comparing like-for-like against Stage B's own `u_hole`-only output via
        // `FrozenInterfaceAnchorTerm`, instead of the field-convention mismatch (`u_total` vs
        // `u_hole`) this fix closes. The interface anchor TARGET below is therefore also
        // corrected to `hx, hy` alone (the closed-form hole correction only, no `a_exx*x`/
        // `a_eyy*y` term) - it must target the SAME `u_hole` convention the now-decomposed
        // energy term trains the model toward, or Stage A's own energy and boundary-condition
        // terms would disagree on what the model output represents.
        let affine_strain_pair = Some((self.spec.load.px, self.spec.load.py));
        let interface_pts = phase2_interface_parametrization();
        let (target_u, target_v): (Vec<f32>, Vec<f32>) = interface_pts.thetas.iter().map(|&theta| {
            let (x, y) = (partition.interface_radius * theta.cos(), partition.interface_radius * theta.sin());
            let (hx, hy) = crate::kirsch_hole_correction::kirsch_hole_displacement(
                x, y, hole.radius, self.spec.material.e as f64, self.spec.material.nu as f64,
                self.spec.load.px, self.spec.load.py,
            );
            (hx as f32, hy as f32)
        }).unzip();
        vec![
            Box::new(PhysicalPotentialEnergyTerm {
                domain: OUTER_DOMAIN, material: self.spec.material.clone(),
                px: self.spec.load.px, py: self.spec.load.py, measure_aware: true,
                domain_area: outer_area, thickness: self.spec.geometry.thickness,
                ref_energy: scales.ref_energy, ref_energy_absolute, interior_weights: None,
                ds_per_point, affine_strain: affine_strain_pair,
            }),
            Box::new(OuterInterfaceAnchorTerm {
                domain: OUTER_DOMAIN, target_u, target_v,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }),
            Box::new(TranslationGaugeTerm {
                domain: OUTER_DOMAIN,
                inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
            }),
            Box::new(RotationGaugeTerm {
                domain: OUTER_DOMAIN, half_w: self.spec.geometry.half_w, half_h: self.spec.geometry.half_h,
            }),
        ]
    }
    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "physical_potential" => LAM_PHYSICAL_POTENTIAL,
            "outer_interface_anchor" => LAM_PHYSICAL_POTENTIAL * 10.0, // real, ongoing BC - same order as interface_weight's own default (100.0/10 relative to unit Pi), not a redundant penalty
            "translation_gauge" => LAM_TRANSLATION_GAUGE,
            "rotation_gauge" => LAM_ROTATION_GAUGE,
            other => panic!("OuterStageProblem::base_weight: unknown term '{other}'"),
        }
    }
    fn phase1_steps(&self) -> usize { 0 }
    fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
    fn convergence_target(&self) -> f64 { 0.0 }
}

/// Stage B BVP: the ANNULUS domain alone, anchored to Stage A's frozen outer model's interface
/// trace instead of a live, jointly-trained outer network. `frozen_target` is recomputed fresh
/// each step by the caller (`user_runner::run_annular_decomposition_training_sequential`) from
/// the frozen model's own forward pass and threaded in via `set_frozen_interface_target` —
/// interior mutability for the same reason `UserDefinedProblem::current_interior_weights` uses
/// it (a per-step value the shared `&self` `loss_terms()` trait method can't otherwise carry).
pub struct AnnulusStageProblem {
    spec: ProblemSpec,
    domains: Vec<DomainSpec>,
    sampling: AnnularPartitionSampling,
    ansatz: crate::kirsch_hole_correction::AnnulusAnsatz,
    frozen_target: std::sync::Mutex<Option<(Vec<f32>, Vec<f32>)>>,
}
impl AnnulusStageProblem {
    pub fn new(spec: ProblemSpec, use_hard_constraint: bool) -> Self {
        assert!(AnnularDecompositionProblem::supports(&spec),
            "sequential two-stage training requires the same scope AnnularDecompositionProblem does");
        let placeholder = spec.geometry.to_placeholder();
        let interface = phase2_interface_parametrization();
        let sampling = AnnularPartitionSampling::new(spec.geometry.clone(), spec.training.fd_h, true, OUTER_DOMAIN, interface);
        let ansatz = if use_hard_constraint {
            let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
            let hole = &spec.geometry.holes[0];
            crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(
                crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                    hole_center: hole.center, hole_radius: hole.radius,
                    half_w: spec.geometry.half_w, half_h: spec.geometry.half_h,
                    px: spec.load.px, py: spec.load.py,
                    e: spec.material.e as f64, nu: spec.material.nu as f64,
                    u_ref: scales.u_ref as f64, saturation_scale: 1.0, trainable: false,
                },
            )
        } else {
            crate::kirsch_hole_correction::AnnulusAnsatz::Identity
        };
        Self {
            domains: vec![DomainSpec { id: ANNULUS_DOMAIN, geometry: placeholder, material: spec.material.clone(), output_dim: 5 }],
            spec, sampling, ansatz, frozen_target: std::sync::Mutex::new(None),
        }
    }
    fn hard_constraint_active(&self) -> bool {
        matches!(self.ansatz, crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(_))
    }
    /// Called by the training loop, once per step, with the frozen outer model's OWN forward
    /// pass evaluated at the same interface points `loss_terms()` will read — see this struct's
    /// own doc comment.
    pub fn set_frozen_interface_target(&self, target_u: Vec<f32>, target_v: Vec<f32>) {
        *self.frozen_target.lock().unwrap() = Some((target_u, target_v));
    }
}
impl BoundaryValueProblem for AnnulusStageProblem {
    fn domains(&self) -> &[DomainSpec] { &self.domains }
    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy {
        assert_eq!(domain_idx, 0);
        &self.sampling
    }
    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz {
        assert_eq!(domain_idx, 0);
        &self.ansatz
    }
    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
        let scales = crate::training_core::compute_reference_scales_for_plate(&self.spec);
        let partition = self.spec.geometry.annular_partition().expect("validated by constructor");
        let annulus_area = std::f64::consts::PI * (partition.interface_radius.powi(2) - partition.hole_radius.powi(2));
        let ref_energy_absolute = scales.ref_energy as f64 * annulus_area * self.spec.geometry.thickness;
        let decomposed = decomposition_applicable(&self.spec);
        let affine_strain_pair = if decomposed { Some((self.spec.load.px, self.spec.load.py)) } else { None };
        let mut terms: Vec<Box<dyn LossTerm>> = vec![
            Box::new(AnnularPotentialEnergyTerm {
                domain: ANNULUS_DOMAIN, name: "annulus_potential",
                material: self.spec.material.clone(), domain_area: annulus_area,
                thickness: self.spec.geometry.thickness, ref_energy_absolute,
                affine_strain: affine_strain_pair,
            }),
        ];
        let (target_u, target_v) = self.frozen_target.lock().unwrap().clone()
            .expect("AnnulusStageProblem::loss_terms called before set_frozen_interface_target");
        terms.push(Box::new(FrozenInterfaceAnchorTerm {
            domain: ANNULUS_DOMAIN, target_u, target_v,
            inv_u_ref_sq: 1.0 / (scales.u_ref as f64).powi(2).max(1e-30),
        }));
        if decomposed && !self.hard_constraint_active() {
            terms.push(Box::new(HoleBcTerm {
                domain: ANNULUS_DOMAIN, point_set: "hole_0_fd", bc: HoleBc::Free,
                ref_stress2: scales.ref_stress2, material: self.spec.material.clone(),
                affine_target: affine_strain_pair,
                // Single-hole-only scope, same rationale as `AnnularDecompositionProblem`'s
                // own identical construction site.
                name: hole_bc_term_name(HoleBc::Free, 0),
            }));
        }
        terms
    }
    fn base_weight(&self, term_name: &str) -> f32 {
        match term_name {
            "annulus_potential" => LAM_PHYSICAL_POTENTIAL,
            "frozen_interface_anchor" => LAM_PHYSICAL_POTENTIAL * 10.0,
            "hole_free" => LAM_HOLE_FREE,
            other => panic!("AnnulusStageProblem::base_weight: unknown term '{other}'"),
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
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> pinn_core::messages::VisFields {
    use crate::differential_operator::production_strain as compute_strains;
    use crate::energy::dem_energy_per_point;
    use crate::training_core::{stencil_forward_with_ansatz, BInner};
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
    // Issue #77 PH4-41/45: same ansatz-application + u_ref/px scaling `compute_domain_forwards`
    // itself uses for training (via the shared `stencil_forward_with_ansatz` helper) - a bare
    // `fwd_embedded` forward pass here would report `u_hole`/`eps_hole` alone under kinematic
    // decomposition, mislabeled as the total field, exactly the bug PH4-41 fixed in
    // `probe_hole_boundary_profile_derived`.
    let (raw, _model_embedding) = stencil_forward_with_ansatz::<BInner>(
        model, ansatz, &active_pts, fd, 1.0, u_ref as f64, px_pa, true,
        embedding_for_model(model, geometry), None, device,
    );

    let (mut eps_xx, mut eps_yy, mut eps_xy) = compute_strains::<BInner>(raw.clone(), n_act, fd);
    // Second half of the PH4-41 gap: add the affine background strain directly to the
    // FD-derived strain, matching `PhysicalPotentialEnergyTerm`/`AnnularPotentialEnergyTerm`'s
    // own `exx.add_scalar(a_exx)` convention.
    let affine_uv: Option<(f64, f64, f64)> = affine_strain_pair.map(|(px, py)| {
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, material);
        eps_xx = eps_xx.clone().add_scalar(a_exx);
        eps_yy = eps_yy.clone().add_scalar(a_eyy);
        eps_xy = eps_xy.clone().add_scalar(a_exy);
        (a_exx, a_eyy, a_exy)
    });
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
        let mut u = center_vals[i_act * 5];
        let mut v = center_vals[i_act * 5 + 1];
        // Reported displacement is also total (affine + ansatz-transformed correction) when
        // decomposed - `decomposition_applicable` requires the hole centered at the origin, so
        // the plate-center-relative and hole-center-relative affine offsets coincide.
        if let Some((a_exx, a_eyy, a_exy)) = affine_uv {
            let (xp, yp) = (pts[i_full][0] as f64 * geometry.half_w, pts[i_full][1] as f64 * geometry.half_h);
            u += (a_exx * xp + a_exy * yp) as f32;
            v += (a_exy * xp + a_eyy * yp) as f32;
        }
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

/// Issue #77 PH4-45: the two-domain analogue of [`evaluate_user_vis_grid`] for
/// `AnnularDecompositionProblem`'s bonded annulus/outer model pair - `run_training_
/// annular_decomposition` deliberately sent `vis: None` before this existed ("no correct
/// two-model field evaluator existed" - see that call site's own doc comment) rather than
/// splicing raw, un-reconstructed fields the way this whole investigation's own PH4-41 finding
/// warns against.
///
/// Evaluates each model over the FULL grid with its own real training-time ansatz (each
/// domain's own `affine_strain_pair` is identical for `AnnularDecompositionProblem` - both
/// `AnnularPotentialEnergyTerm`/`PhysicalPotentialEnergyTerm(OUTER_DOMAIN)` share the same
/// `decomposition_applicable`-gated pair, confirmed in `loss_terms()`), then splices per-cell:
/// the annulus model's own field where a cell's physical distance from the hole center is
/// less than `interface_radius`, the outer model's elsewhere. `collocation_density` is
/// model-independent (a pure function of `int_norm`) - either call's copy is used, not merged.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_annular_vis_grid(
    annulus_model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    outer_model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
    grid: [usize; 2],
    u_ref: f32,
    px_pa: f64,
    material: &MaterialProps,
    fd: &crate::fd_stencil::FdConfig,
    int_norm: &[[f32; 2]],
    device: &crate::training_core::BDevice,
    annulus_ansatz: &dyn DirichletAnsatz,
    outer_ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
    hole_center: [f64; 2],
    interface_radius: f64,
) -> pinn_core::messages::VisFields {
    use ndarray::Array2;

    let [nx, ny] = grid;
    let annulus_vis = evaluate_user_vis_grid(
        annulus_model, geometry, grid, u_ref, px_pa, material, fd, int_norm, device,
        annulus_ansatz, affine_strain_pair,
    );
    let outer_vis = evaluate_user_vis_grid(
        outer_model, geometry, grid, u_ref, px_pa, material, fd, int_norm, device,
        outer_ansatz, affine_strain_pair,
    );

    // Same normalized-grid-coordinate convention `evaluate_user_vis_grid`'s own point loop
    // uses (row=iy, col=ix), so the splice boundary lands exactly where each model's own
    // domain (`AnnularDecompositionProblem::sampling_strategy`) actually trained.
    let use_annulus = |row: usize, col: usize| -> bool {
        let xn = -1.0 + 2.0 * col as f64 / (nx.max(2) - 1) as f64;
        let yn = -1.0 + 2.0 * row as f64 / (ny.max(2) - 1) as f64;
        let (xp, yp) = (xn * geometry.half_w, yn * geometry.half_h);
        let (dx, dy) = (xp - hole_center[0], yp - hole_center[1]);
        (dx * dx + dy * dy).sqrt() < interface_radius
    };
    let splice = |a: &Array2<f32>, o: &Array2<f32>| -> Array2<f32> {
        Array2::from_shape_fn((ny, nx), |(row, col)| {
            if use_annulus(row, col) { a[(row, col)] } else { o[(row, col)] }
        })
    };

    pinn_core::messages::VisFields {
        von_mises: splice(&annulus_vis.von_mises, &outer_vis.von_mises),
        sigma_xx: splice(&annulus_vis.sigma_xx, &outer_vis.sigma_xx),
        sigma_yy: splice(&annulus_vis.sigma_yy, &outer_vis.sigma_yy),
        sigma_xy: splice(&annulus_vis.sigma_xy, &outer_vis.sigma_xy),
        disp_u: splice(&annulus_vis.disp_u, &outer_vis.disp_u),
        disp_v: splice(&annulus_vis.disp_v, &outer_vis.disp_v),
        eps_xx: splice(&annulus_vis.eps_xx, &outer_vis.eps_xx),
        eps_yy: splice(&annulus_vis.eps_yy, &outer_vis.eps_yy),
        eps_xy: splice(&annulus_vis.eps_xy, &outer_vis.eps_xy),
        pde_residual: splice(&annulus_vis.pde_residual, &outer_vis.pde_residual),
        amr_score: splice(&annulus_vis.amr_score, &outer_vis.amr_score),
        collocation_density: annulus_vis.collocation_density,
    }
}

/// `enhancement.txt` items 4/C ("BC residual RMS/max") - real per-point traction/
/// displacement residual at the outer boundary and every hole ring, combined. Same math
/// `OuterTractionTerm`/`HoleBcTerm` compute internally (via `neumann_loss`/
/// `hole_traction_loss_direct`), kept pre-mean here so a real distribution stat is possible -
/// a side probe at the existing vis cadence (mirrors `training_core::probe_interior_energy_
/// residuals`'s own "side probe, not the hot per-step path" precedent), NOT a change to
/// `step_physics_multi`'s per-step loss computation.
/// Issue #78 real bug fix: `ansatz`/`affine_strain_pair` parameters, threaded through exactly
/// like `probe_hole_boundary_profile_derived`'s own PH4-41 fix - this function used to read the
/// model via a bare `fwd_embedded` forward, completely bypassing any active `AnnulusAnsatz`
/// (multiplicative envelope + additive closed-form correction) and any active kinematic-
/// decomposition affine background. Under a hard-constraint-ansatz model this made the outer-
/// boundary residual (and therefore `probe_load_transfer`/`probe_reaction_force`, which both
/// call this for their own traction numbers) read the RAW NETWORK'S OWN residual output as if
/// it WERE the physical field - a small quantity by design (the ansatz suppresses/supplies most
/// of the real field), producing a spurious near-zero "trivial collapse" reading even on a
/// genuinely well-trained hard-constraint model. Found via this session's own real off-center
/// multi-hole `--headless` verification run, which printed a P2-09 trivial-solution warning
/// (2.3% load transfer) on a model whose OWN Kt read correctly at ~2.5 (close to FEM's ~2.9-3.1)
/// once `probe_hole_boundary_profile_derived`'s already-correct ansatz/affine wiring was used -
/// the mismatch traced directly to this function's stale reconstruction.
pub fn probe_boundary_residuals(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> (f64, f64) {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::{norm_pts_to_tensor, FdConfig};
    use crate::network::fwd_embedded;
    use crate::training_core::{stencil_forward_with_ansatz, BInner};
    use burn::tensor::TensorData;

    let geometry = &spec.geometry;
    let sampling = UserSamplingStrategy::new(geometry.clone(), spec.training.fd_h);
    let placeholder = geometry.to_placeholder();
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let (stress_ref, u_ref) = (scales.stress_ref, scales.u_ref);
    let px_pa = stress_ref;
    let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32] };
    let add_affine = |exx: Tensor<BInner, 1>, eyy: Tensor<BInner, 1>, exy: Tensor<BInner, 1>| -> (Tensor<BInner, 1>, Tensor<BInner, 1>, Tensor<BInner, 1>) {
        match affine_strain_pair {
            Some((px, py)) => {
                let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &spec.material);
                (exx.add_scalar(a_exx), eyy.add_scalar(a_eyy), exy.add_scalar(a_exy))
            }
            None => (exx, eyy, exy),
        }
    };

    let mut residuals: Vec<f32> = Vec::new();

    let bnd_pts_phys = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
    if !bnd_pts_phys.is_empty() {
        let n_bnd = bnd_pts_phys.len();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts_phys.iter().map(|p| norm_pt(p.x, p.y)).collect();
        let (scaled, _embedding) = stencil_forward_with_ansatz::<BInner>(
            model, ansatz, &bnd_norm, &fd, 1.0, u_ref as f64, px_pa, true,
            embedding_for_model(model, geometry), None, device,
        );
        let (exx, eyy, exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
        let (exx, eyy, exy) = add_affine(exx, eyy, exy);
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
    // of re-deriving the correspondence from each set's name string. Direct-stress-column
    // read (not FD-derived) - unaffected by `affine_strain_pair` (that mechanism only ever
    // corrects FD-derived strain; a direct stress-column read has no strain step to correct),
    // but the `u`/`v` columns DO need the ansatz's own transform for the Fixed-hole branch
    // below, which is why `fwd_embedded` is still replaced with the ansatz-aware forward.
    for (hole, set) in geometry.holes.iter().zip(sampling.named_point_sets(&[]).into_iter()) {
        let n_h = set.points.len();
        if n_h == 0 { continue; }
        let ring_norm: Vec<[f32; 2]> = set.points.iter().map(|p| norm_pt(p.x, p.y)).collect();
        // Issue #78: `embedding_for_model` (model-aware), not a bare `geometry.coordinate_
        // embedding()` - the latter assumes the PASSED-IN model was built with the geometry's
        // own "default" embedding, which stopped holding the instant `coordinate_embedding()`
        // started returning `MultiHoleChart` for N>1 holes (a real shape-mismatch panic this
        // exact call site hit against a `tiny_model_raw()`-style narrower test model, on a
        // multi-hole geometry - the same model/geometry-embedding-mismatch class of bug
        // `embedding_for_model` itself exists to prevent everywhere else in this file).
        let raw = fwd_embedded::<BInner>(model, norm_pts_to_tensor::<BInner>(&ring_norm, device), embedding_for_model(model, geometry), device);
        let (dx_v, dy_v, add_x_v, add_y_v): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = {
            let mut dx_v = Vec::with_capacity(n_h);
            let mut dy_v = Vec::with_capacity(n_h);
            let mut add_x_v = Vec::with_capacity(n_h);
            let mut add_y_v = Vec::with_capacity(n_h);
            for p in &ring_norm {
                let (dx, dy) = ansatz.eval(p[0], p[1], 1.0);
                dx_v.push(dx);
                dy_v.push(dy);
                let (ax, ay) = ansatz.additive(p[0], p[1]);
                add_x_v.push(ax);
                add_y_v.push(ay);
            }
            (dx_v, dy_v, add_x_v, add_y_v)
        };
        let dx_t = Tensor::<BInner, 2>::from_data(TensorData::new(dx_v, vec![n_h, 1]), device);
        let dy_t = Tensor::<BInner, 2>::from_data(TensorData::new(dy_v, vec![n_h, 1]), device);
        let add_x_t = Tensor::<BInner, 2>::from_data(TensorData::new(add_x_v, vec![n_h, 1]), device);
        let add_y_t = Tensor::<BInner, 2>::from_data(TensorData::new(add_y_v, vec![n_h, 1]), device);
        let u_col = (raw.clone().slice([0..n_h, 0..1]) * dx_t + add_x_t).mul_scalar(u_ref as f64);
        let v_col = (raw.clone().slice([0..n_h, 1..2]) * dy_t + add_y_t).mul_scalar(u_ref as f64);
        let scaled = Tensor::cat(vec![u_col, v_col, raw.slice([0..n_h, 2..5]).mul_scalar(px_pa)], 1);
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
/// Issue #78 real bug fix: `ansatz`/`affine_strain_pair` parameters - see
/// `probe_boundary_residuals`'s own doc comment for the shared root cause and evidence; this
/// function had the identical bare-`fwd_embedded` staleness for its own independent outer-
/// boundary forward pass.
pub fn probe_reaction_force(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> pinn_core::messages::ReactionForce {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::FdConfig;
    use crate::training_core::{stencil_forward_with_ansatz, BInner};
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
    let (scaled, _embedding) = stencil_forward_with_ansatz::<BInner>(
        model, ansatz, &bnd_norm, &fd, 1.0, u_ref as f64, px_pa, true,
        embedding_for_model(model, geometry), None, device,
    );
    let (mut exx, mut eyy, mut exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
    if let Some((px, py)) = affine_strain_pair {
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &spec.material);
        exx = exx.add_scalar(a_exx);
        eyy = eyy.add_scalar(a_eyy);
        exy = exy.add_scalar(a_exy);
    }
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

/// Issue #78 real bug fix: `ansatz`/`affine_strain_pair` parameters - see
/// `probe_boundary_residuals`'s own doc comment for the shared root cause and evidence. This is
/// the function whose stale reconstruction directly produced this session's own real, false
/// "P2-09 trivial-solution warning: only 2.3% of load transferred" on a hard-constraint model
/// that had actually trained well (its own correctly-reconstructed Kt read ~2.5, close to FEM's
/// ~2.9-3.1, before this fix).
pub fn probe_load_transfer(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    device: &crate::training_core::BDevice,
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> LoadTransferReport {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::fd_stencil::FdConfig;
    use crate::training_core::{stencil_forward_with_ansatz, BInner};
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

    let (traction_residual_rms, traction_residual_max) = probe_boundary_residuals(model, spec, device, ansatz, affine_strain_pair);

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
    let (scaled, _embedding) = stencil_forward_with_ansatz::<BInner>(
        model, ansatz, &bnd_norm, &fd, 1.0, u_ref as f64, px_pa, true,
        embedding_for_model(model, geometry), None, device,
    );
    let (mut exx, mut eyy, mut exy) = compute_strains::<BInner>(scaled, n_bnd, &fd);
    if let Some((px, py)) = affine_strain_pair {
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &spec.material);
        exx = exx.add_scalar(a_exx);
        eyy = eyy.add_scalar(a_eyy);
        exy = exy.add_scalar(a_exy);
    }
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

    // `run_no_hole_benchmark` only ever runs against a no-hole geometry (asserted - see
    // `run_no_hole_benchmark_panics_on_a_holed_geometry`) - `IdentityAnsatz`/no affine is the
    // exact, correct reconstruction there, not a stopgap.
    let (traction_rms, _traction_max) = probe_boundary_residuals(model, spec, device, &crate::pinlug_problem::IdentityAnsatz, None);
    let traction_rms_over_ref = traction_rms / sigma_ref;

    let load_transfer = probe_load_transfer(model, spec, device, &crate::pinlug_problem::IdentityAnsatz, None);

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
    // Issue #77 PH4-41: `run_hole_benchmark` only ever validates the plain `UserDefinedProblem::
    // new(spec)` path (no hard-constraint ansatz concept here) - `IdentityAnsatz` matches that
    // model's own training convention exactly. `decomposition_applicable` is the SAME gate
    // `UserDefinedProblem::loss_terms()` itself uses to decide whether the model was trained
    // under kinematic decomposition.
    let affine = decomposition_applicable(spec).then_some((spec.load.px, spec.load.py));
    let profile = probe_hole_boundary_profile_derived(
        model, geometry, hole, 72, &fd, scales.u_ref, spec.load.px, &spec.material, margin, device,
        &IdentityAnsatz, affine,
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

/// Issue #77 PH4-31 fix: derives the `CoordinateEmbedding` that actually matches `model`'s
/// own saved input width, rather than assuming `geometry.coordinate_embedding()` (which is
/// always the plain, `n_fourier=0` embedding). Mirrors `training_core::compute_domain_
/// forwards`'s identical "the model's saved architecture is authoritative" dispatch, needed
/// here because the model this is called on (the annulus model, at diagnostic checkpoints)
/// may have been built with a Fourier-augmented embedding the caller doesn't otherwise pass
/// in. A real bug this fixed: `issue_77_annulus_fourier_l5_trace`'s first run panicked with
/// `IncompatibleShapes { left: [720, 10], right: [26, 64] }` - the probe was embedding at the
/// plain 10-column width while the model's first layer expected 26 (`10 + 4*4` for
/// `n_fourier=4`).
fn embedding_for_model(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &UserGeometry,
) -> pinn_core::user_geometry::CoordinateEmbedding {
    use pinn_core::user_geometry::CoordinateEmbedding;
    let dim = model.input_dim();
    if dim == 3 {
        return CoordinateEmbedding::Raw;
    }
    // Issue #77 Phase 3: `LogPolar`'s own fixed width (6) - checked BEFORE the plain-chart
    // comparison below so a log-polar model (dim=6) never falls through toward the Fourier
    // branch (which only ever matches dim>10) and panics, the exact bug class PH4-31's own
    // fix here exists to prevent for a different embedding variant.
    let log_polar = geometry.log_polar_embedding();
    if dim == log_polar.input_dim() && matches!(log_polar, CoordinateEmbedding::LogPolar { .. }) {
        return log_polar;
    }
    let plain = geometry.coordinate_embedding();
    if dim == plain.input_dim() {
        return plain;
    }
    if dim > 10 && (dim - 10) % 4 == 0 {
        let with_fourier = geometry.coordinate_embedding_with_fourier((dim - 10) / 4);
        if dim == with_fourier.input_dim() {
            return with_fourier;
        }
    }
    panic!("embedding_for_model: model input_dim {dim} matches no known embedding for this geometry");
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
    let raw_stencil = fwd_embedded::<BInner>(model, stencil, embedding_for_model(model, geometry), device); // [5n, 5]: u,v,sxx,syy,sxy

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
///
/// **Issue #77 PH4-41 fix**: this function used to read the network's own raw output, scaled
/// only by `u_ref`/`px_pa` (unit conversion) — bypassing BOTH the domain's `DirichletAnsatz`
/// (the hard-constraint ansatz's own envelope suppression and exact closed-form Kirsch
/// correction) AND `affine_strain`'s own contribution (added only inside training's loss terms,
/// never exposed on the displacement itself under kinematic decomposition). That meant every
/// real Kt/stress number this whole investigation reported (`annular_l5_diagnostic`,
/// `user_problem_l5_diagnostic`, and everything downstream of them) measured a PARTIAL field,
/// not total physical displacement/stress — confirmed by direct comparison against
/// `training_core::compute_domain_forwards`'s own `u_col = raw_net*ansatz.eval() +
/// ansatz.additive()` convention, which this function never applied. Now takes the SAME
/// `ansatz`/`affine_strain` the caller's own problem uses for training (no independent
/// convention of its own to drift out of sync again), reusing `training_core::
/// stencil_forward_with_ansatz` — the single shared implementation `compute_domain_forwards`
/// itself now also calls, so there is exactly one "how ansatz + u_ref combine" in this crate.
/// `ansatz` and `affine_strain` MUST be the exact same values the caller's `loss_terms()` used
/// to train `model` - passing `&IdentityAnsatz`/`None` for a model actually trained under the
/// hard-constraint ansatz or kinematic decomposition would silently reintroduce this same bug
/// in a new place.
#[allow(clippy::too_many_arguments)]
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
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> Vec<HoleBoundaryPoint> {
    use crate::energy::compute_stress;
    use crate::differential_operator::production_strain as compute_strains;
    use crate::training_core::{stencil_forward_with_ansatz, BInner};

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

    // Same ansatz-application + u_ref/px scaling `compute_domain_forwards` itself uses for
    // training - `scaled` here is genuinely total (ansatz-transformed) displacement/direct-
    // stress, not raw network output, closing the first half of the PH4-41 gap.
    let (scaled, _embedding) = stencil_forward_with_ansatz::<BInner>(
        model, ansatz, &pts_norm, fd, 1.0, u_ref as f64, px_pa, true,
        embedding_for_model(model, geometry), None, device,
    );
    let center_uv = scaled.clone().slice([0..n, 0..2]);
    let (mut eps_xx, mut eps_yy, mut eps_xy) = compute_strains::<BInner>(scaled, n, fd);
    // Second half of the PH4-41 gap: `affine_strain`'s own contribution is added directly to
    // the FD-derived strain, exactly matching `PhysicalPotentialEnergyTerm`/
    // `AnnularPotentialEnergyTerm::compute()`'s own `exx.add_scalar(a_exx)` convention (the
    // network under kinematic decomposition represents `eps_hole`/`u_hole` alone; total strain
    // requires this addition, which training already does internally but this diagnostic
    // never did).
    let affine_uv: Option<(f64, f64, f64, f64, f64)> = affine_strain_pair.map(|(px, py)| {
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, material);
        eps_xx = eps_xx.clone().add_scalar(a_exx);
        eps_yy = eps_yy.clone().add_scalar(a_eyy);
        eps_xy = eps_xy.clone().add_scalar(a_exy);
        (a_exx, a_eyy, a_exy, px, py)
    });
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
        // Reported displacement: also total (affine + ansatz-transformed correction) when
        // decomposed - a plain point-value addition (not FD-differenced, since it's reported
        // for inspection, not itself differentiated further).
        let (mut ux, mut uy) = (uv_vals[i * 2], uv_vals[i * 2 + 1]);
        if let Some((a_exx, a_eyy, a_exy, _px, _py)) = affine_uv {
            let (dx, dy) = (x - hole.center[0], y - hole.center[1]);
            ux += (a_exx * dx + a_exy * dy) as f32;
            uy += (a_exy * dx + a_eyy * dy) as f32;
        }
        HoleBoundaryPoint {
            theta_deg: thetas[i], x, y, ux, uy,
            eps_xx: exx_vals[i], eps_yy: eyy_vals[i], eps_xy: exy_vals[i],
            sxx, syy, sxy, von_mises: vm,
        }
    }).collect()
}

/// Measure direct-versus-derived stress at the same FD-safe ring. The direct boundary profile
/// cannot answer this question because its coordinates lie on `r=R`, while derived stress needs
/// a margin so its finite-difference stencil stays outside the hole.
///
/// **Issue #77 PH4-41**: `ansatz` is threaded through to `probe_hole_boundary_profile_derived`
/// so `derived`'s displacement (and thus its FD strain/stress) reflects the real ansatz
/// transform (needed for a meaningful reading under the hard-constraint ansatz). `affine_strain`
/// is deliberately NOT added here (`None` is always passed downstream) - `direct` reads the raw
/// mDEM stress columns, which NEVER include an affine contribution by construction anywhere in
/// this codebase (nothing adds affine to the direct stress output), so adding it only to
/// `derived` would make this specific direct-vs-derived CONSISTENCY check compare two
/// deliberately mismatched conventions - the opposite of what it's measuring. This is a
/// narrower, self-consistency diagnostic, not the total-field Kt measurement
/// `probe_hole_boundary_profile_derived`'s own callers use.
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
    ansatz: &dyn DirichletAnsatz,
) -> pinn_core::messages::HoleStressDiagnostic {
    let radius = hole.radius + margin;
    let direct = probe_hole_stress_profile_direct_at_radius(
        model, geometry, hole, n_theta, fd, u_ref, px_pa, radius, device,
    );
    let derived = probe_hole_boundary_profile_derived(
        model, geometry, hole, n_theta, fd, u_ref, px_pa, material, margin, device, ansatz, None,
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

/// Issue #78 second root-cause fix (Kt convergence check): how saturated `phi` must be at the
/// SECOND radial probe point (`kt_coarse_margin_1_5x` below) - deliberately close to 1
/// ("essentially at asymptote"), so this probe sits safely past the envelope's own steep
/// transition zone regardless of the training-time `saturation_scale` in use. Deliberately NOT
/// the same target as `target_phi_at_margin`: that target answers a different question ("how
/// much raw gradient signal survives at the real training margin"), and reusing its value here
/// would put this probe at the exact SAME radius as the first probe (since that margin is, by
/// construction, where `target_phi_at_margin`'s own target is reached) - collapsing the
/// comparison to zero by definition rather than measuring anything. This is a genuinely
/// separate, disclosed constant for a genuinely separate purpose.
const ENVELOPE_MEASUREMENT_SATURATED_TARGET: f64 = 0.999;

/// The margin (physical, `r - hole_radius`) at which `traction_free_envelope_scaled` reaches
/// `target`, for a GIVEN `scale` - the inverse of `multi_hole_saturation_scale`'s own solve
/// (that function solves for `scale` given a target margin and radius; this solves for the
/// margin given a target and a scale, i.e. `margin = hole_radius * sqrt(-ln(1-target)) / scale`).
fn envelope_margin_for_target(hole_radius: f64, scale: f64, target: f64) -> f64 {
    hole_radius * (-(1.0 - target).ln()).sqrt() / scale
}

/// Extracted from `kt_convergence_check` so the second-radial-probe margin-selection logic
/// (see `KtConvergenceReport`'s own doc comment) is independently unit-testable without a full
/// model forward pass. `hole_center`/`hole_radius` identify the hole to `ansatz.saturation_
/// scale_near`; `margin_coarse` is the first probe's own (already FD-safety-derived) margin.
fn kt_convergence_radial_probe_margin(
    ansatz: &dyn DirichletAnsatz,
    hole_center: [f64; 2],
    hole_radius: f64,
    margin_coarse: f64,
) -> f64 {
    let margin_1_5x = margin_coarse * 1.5;
    match ansatz.saturation_scale_near(hole_center) {
        Some(scale) if scale.is_finite() && scale > 1.0 => {
            envelope_margin_for_target(hole_radius, scale, ENVELOPE_MEASUREMENT_SATURATED_TARGET)
                .max(margin_1_5x)
        }
        _ => margin_1_5x,
    }
}

/// Issue #61 EPIC P2-10's own "angular/radial convergence support" - runs the SAME Kt QoI
/// pipeline at a coarser and a finer angular resolution (same radial margin), and at the
/// coarse resolution with a LARGER radial margin, reporting whether Kt is actually converging
/// rather than drifting. Directly answers issue #61 §3's own concern (echoed for domain
/// integrals in P2-11): a single point-count/margin choice proves nothing about convergence on
/// its own.
///
/// **Issue #78 second root-cause fix**: the second radial probe's margin used to be a flat
/// `margin_coarse * 1.5` - fine when `saturation_scale=1.0` (N=1, byte-identical to before this
/// fix), but once `saturation_scale` grows for N>1 (issue #78's own root-cause fix), both
/// `margin_coarse` and `margin_coarse*1.5` can land INSIDE the envelope's own steep transition
/// zone, so `radial_relative_change` partly measures the ansatz's own known, deterministic
/// curvature rather than genuine training non-convergence. Now: if `ansatz.saturation_scale_
/// near(hole.center)` reports an active envelope, the second probe's margin is instead set to
/// wherever `phi` reaches [`ENVELOPE_MEASUREMENT_SATURATED_TARGET`] for that hole's real scale
/// (never smaller than the original `1.5x`, so this never risks crossing into `valid_stencil`-
/// unsafe territory near the hole boundary) - guaranteeing both radial probes sit past the
/// envelope's own steep region. Gated to `scale > 1.0` specifically (not merely "an envelope is
/// active"): at `scale=1.0` (N=1, or any pre-#78 caller) the envelope saturates so slowly that
/// solving for `phi=0.999` would land far outside any sensible probe radius, so at `scale=1.0`
/// the ORIGINAL `1.5x` margin is used EXACTLY, not approximately - a true zero-regression
/// guarantee for N=1 (verified - see `kt_convergence_check_radial_probe_is_byte_identical_at_
/// unit_scale`).
/// `radial_residual_kt_delta`: Issue #78 second root-cause fix - the ABSOLUTE (not relative -
/// the residual can be near zero, making a ratio numerically unstable) change in the network's
/// OWN Kt contribution (`kt_measured - closed_form_only_kt`) between the two radial probes.
/// `None` when no closed-form baseline applies at all (`IdentityAnsatz` with no affine
/// background - baseline is identically zero everywhere, so there is nothing to subtract and
/// this residual would just equal the raw values, not a meaningful decomposition). This does
/// NOT change `converged`'s own existing gate (kept exactly as before - backward compatible for
/// every existing caller/threshold) - it's ADDITIVE diagnostic context surfacing how much of
/// the raw `radial_relative_change` is real, expected closed-form field curvature versus the
/// network's own still-changing correction, for a human reader to judge, not a new pass/fail
/// rule silently redefining what "converged" means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KtConvergenceReport {
    pub kt_coarse: f64,
    pub kt_fine_angular: f64,
    pub kt_coarse_margin_1_5x: f64,
    pub angular_relative_change: f64,
    pub radial_relative_change: f64,
    pub radial_residual_kt_delta: Option<f64>,
    pub converged: bool,
}

/// Issue #78 second root-cause fix, decisive supporting evidence: the Kt implied by ONLY the
/// ansatz's own closed-form baseline (`additive()`, zero network contribution) at `r = hole.
/// radius + margin` - generic over any `DirichletAnsatz` (`IdentityAnsatz`'s `additive()` is
/// `(0,0)` everywhere, so with no affine background this returns `0.0`, a real "no baseline"
/// signal, not a crash). Pure host math, no `model`/`device`/burn tensor involved at all - a
/// genuinely independent computation from the trained-model path, deliberately so (this is
/// meant to isolate what the CLOSED FORM ALONE predicts, uncontaminated by anything the model
/// might be doing).
///
/// Exists because a real diagnostic test (`kirsch_hole_correction::closed_form_only_kt_
/// varies_meaningfully_between_the_two_radial_probe_points_at_real_scale`) found the PURE
/// closed-form baseline alone already accounts for a 7.35% Kt change between `margin_coarse`
/// and `margin_coarse*1.5` for `triple_hole_plate.toml`'s real geometry at the item-2-derived
/// scale - most of a real trained run's own observed ~11% raw radial Δ. `kt_convergence_check`'s
/// raw comparison was always going to partly (mostly) reflect real, EXPECTED analytic field
/// curvature this close to a hole boundary, not training non-convergence - no choice of SECOND
/// probe radius alone (`kt_convergence_radial_probe_margin`, this fix's first attempt) can fully
/// separate the two without this kind of baseline subtraction. `KtConvergenceReport.radial_
/// residual_kt_delta` uses this to report the network's OWN residual-correction drift between
/// the two radii, separately from the raw (baseline-contaminated) `radial_relative_change`.
#[allow(clippy::too_many_arguments)]
fn closed_form_only_kt_at_margin(
    ansatz: &dyn DirichletAnsatz,
    geometry: &UserGeometry,
    hole: &HoleSpec,
    margin: f64,
    u_ref: f64,
    material: &MaterialProps,
    affine_strain_pair: Option<(f64, f64)>,
    nominal_stress: f64,
    n_theta: usize,
) -> f64 {
    let half_w = geometry.half_w;
    let half_h = geometry.half_h;
    let r = hole.radius + margin;
    // Tiny physical FD step, independent of this codebase's own `fd_h` training convention -
    // matches `kirsch_hole_correction.rs`'s own `total_field_is_traction_free_at_hole_
    // boundary_numerically` precedent for this exact kind of closed-form-only check.
    let h = 1e-7_f64;
    let (a_exx, a_eyy, a_exy) = affine_strain_pair
        .map(|(px, py)| affine_strain(px, py, material))
        .unwrap_or((0.0, 0.0, 0.0));
    let total_u = |x: f64, y: f64| -> (f64, f64) {
        let xn = (x / half_w) as f32;
        let yn = (y / half_h) as f32;
        let (ax, ay) = ansatz.additive(xn, yn);
        (a_exx * x + a_exy * y + ax as f64 * u_ref, a_exy * x + a_eyy * y + ay as f64 * u_ref)
    };
    let e = material.e as f64;
    let nu = material.nu as f64;
    let n = n_theta.max(1);
    let mut max_vm = 0.0_f64;
    for i in 0..n {
        let theta = i as f64 * std::f64::consts::TAU / n as f64;
        let x0 = hole.center[0] + r * theta.cos();
        let y0 = hole.center[1] + r * theta.sin();
        let (u_xp, v_xp) = total_u(x0 + h, y0);
        let (u_xm, v_xm) = total_u(x0 - h, y0);
        let (u_yp, v_yp) = total_u(x0, y0 + h);
        let (u_ym, v_ym) = total_u(x0, y0 - h);
        let exx = (u_xp - u_xm) / (2.0 * h);
        let eyy = (v_yp - v_ym) / (2.0 * h);
        let exy = 0.5 * ((u_yp - u_ym) / (2.0 * h) + (v_xp - v_xm) / (2.0 * h));
        let sxx = e / (1.0 - nu * nu) * (exx + nu * eyy);
        let syy = e / (1.0 - nu * nu) * (eyy + nu * exx);
        let sxy = e / (2.0 * (1.0 + nu)) * (2.0 * exy);
        let vm = (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).sqrt();
        if vm > max_vm { max_vm = vm; }
    }
    max_vm / nominal_stress.abs()
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
    ansatz: &dyn DirichletAnsatz,
    affine_strain_pair: Option<(f64, f64)>,
) -> KtConvergenceReport {
    let kt_of = |n_theta: usize, margin: f64| -> f64 {
        let profile = probe_hole_boundary_profile_derived(
            model, geometry, hole, n_theta, fd, u_ref, px_pa, material, margin, device, ansatz, affine_strain_pair,
        );
        stress_concentration_from_profile(&profile, nominal_stress).kt
    };

    let kt_coarse = kt_of(n_theta_coarse, margin_coarse);
    let kt_fine_angular = kt_of(n_theta_coarse * 2, margin_coarse);

    // Issue #78 second root-cause fix: pick the second radial probe's margin from the
    // envelope's own saturation curve when an envelope is active, instead of a flat 1.5x -
    // see `KtConvergenceReport`'s own doc comment for the full rationale.
    let margin_radial_2 = kt_convergence_radial_probe_margin(ansatz, hole.center, hole.radius, margin_coarse);
    let kt_coarse_margin_1_5x = kt_of(n_theta_coarse, margin_radial_2);

    let rel = |a: f64, b: f64| if b.abs() > 1e-30 { (a - b).abs() / b.abs() } else { (a - b).abs() };
    let angular_relative_change = rel(kt_fine_angular, kt_coarse);
    let radial_relative_change = rel(kt_coarse_margin_1_5x, kt_coarse);

    // Issue #78 second root-cause fix: decompose the raw radial Δ into "expected closed-form
    // curvature" vs. "the network's own residual still moving" - see `KtConvergenceReport`'s
    // own doc comment. `None` when there's no real ansatz baseline to subtract (would just
    // reproduce the raw values, not a meaningful decomposition).
    let has_baseline = affine_strain_pair.is_some() || ansatz.saturation_scale_near(hole.center).is_some();
    let radial_residual_kt_delta = has_baseline.then(|| {
        let kt_baseline_1 = closed_form_only_kt_at_margin(
            ansatz, geometry, hole, margin_coarse, u_ref as f64, material, affine_strain_pair, nominal_stress, n_theta_coarse,
        );
        let kt_baseline_2 = closed_form_only_kt_at_margin(
            ansatz, geometry, hole, margin_radial_2, u_ref as f64, material, affine_strain_pair, nominal_stress, n_theta_coarse,
        );
        ((kt_coarse - kt_baseline_1) - (kt_coarse_margin_1_5x - kt_baseline_2)).abs()
    });

    // Issue #78 item 1 (final fix): the radial half of `converged` now gates on the DECOMPOSED
    // residual (relative to `kt_coarse`, the same units/scale `tolerance` already means for the
    // angular check) whenever a baseline exists, instead of the raw, curvature-contaminated
    // `radial_relative_change` - the raw number was proven (see this function's own doc comment
    // and `closed_form_only_kt_varies_meaningfully_between_the_two_radial_probe_points_at_real_
    // scale`) to be dominated by real, EXPECTED closed-form field curvature that has nothing to
    // do with training convergence, so gating pass/fail on it was measuring the wrong thing. No
    // baseline (`IdentityAnsatz`, no affine background - e.g. the no-hole variational benchmark)
    // falls back to the ORIGINAL raw check exactly as before - zero regression for that case,
    // since there's no known curvature to subtract there in the first place.
    let radial_ok = match radial_residual_kt_delta {
        Some(residual) => {
            let residual_relative = if kt_coarse.abs() > 1e-30 { residual / kt_coarse.abs() } else { residual };
            residual_relative.is_finite() && residual_relative < tolerance
        }
        None => radial_relative_change.is_finite() && radial_relative_change < tolerance,
    };
    let converged = angular_relative_change.is_finite() && angular_relative_change < tolerance && radial_ok;

    KtConvergenceReport {
        kt_coarse, kt_fine_angular, kt_coarse_margin_1_5x,
        angular_relative_change, radial_relative_change, radial_residual_kt_delta, converged,
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
            architecture: Default::default(),
        }
    }

    /// Issue #77 PH4-45: real regression guard for a genuine pre-existing bug this session's
    /// own headless smoke test caught - `run_headless_user_problem`'s "not a trivial collapse"
    /// diagnostic used to call `model.forward` directly on a bare 3-column tensor, bypassing
    /// the embedding transform entirely, panicking on ANY single-hole geometry (the model is
    /// built with `net_input_dim()`=10 for `SingleHoleChart`, not 3). This never had a test
    /// exercising `run_headless_user_problem` itself against a real hole - every PH4-24..44 run
    /// trained through a different function. `single_hole_like_spec` is `measure_aware_
    /// training: false`, so this exercises the plain single-domain branch specifically (not
    /// the annular-decomposition one, which has its own separate code path).
    #[test]
    fn run_headless_user_problem_completes_on_a_real_single_hole_geometry_without_panicking() {
        let mut spec = single_hole_like_spec(3);
        spec.training.n_interior = 64;
        spec.training.n_boundary = 32;
        assert!(
            crate::user_runner::run_headless_user_problem(spec),
            "headless run against a real single-hole geometry must complete with a finite final loss"
        );
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
            architecture: Default::default(),
        };
        assert!(AnnularDecompositionProblem::supports(&spec));
        let problem = AnnularDecompositionProblem::new(spec);
        crate::problem::validate_loss_terms(&problem);
        assert_eq!(problem.domains().iter().map(|d| d.id).collect::<Vec<_>>(), vec![ANNULUS_DOMAIN, OUTER_DOMAIN]);
        let terms = problem.loss_terms();
        assert!(terms.iter().any(|t| t.name() == "interface_displacement_continuity"));
        assert!(terms.iter().any(|t| t.name() == "interface_traction_continuity"));
    }

    /// Issue #78 item 4: the load-bearing N=1 regression proof - `MultiAnnularDecompositionProblem`
    /// with exactly one Free hole must produce STRUCTURALLY IDENTICAL output to the frozen,
    /// already-proven `AnnularDecompositionProblem` for the SAME spec: same domain count/output
    /// widths, bit-identical sampled interior points (both reuse `AnnularPartitionSampling`'s
    /// own seeding unchanged), bit-identical named interface/hole point sets, and the same set
    /// of loss-term names with matching `base_weight`s. This is the single check that makes
    /// trusting the new N-hole code for N=1 defensible - not just "it compiles."
    #[test]
    fn multi_annular_decomposition_matches_annular_decomposition_at_n_equals_one() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let original = AnnularDecompositionProblem::new_with_hard_constraint_ansatz(spec.clone(), true);
        let multi = MultiAnnularDecompositionProblem::new(spec, true);

        // Same domain count and per-domain output width.
        assert_eq!(original.domains().len(), multi.domains().len());
        for (o, m) in original.domains().iter().zip(multi.domains().iter()) {
            assert_eq!(o.output_dim, m.output_dim);
        }

        // Bit-identical sampled interior points, both domains, several calls (proves the RNG
        // seed streams genuinely match, not just "both non-empty").
        // Both samplers ignore this parameter entirely (they carry their own captured
        // `UserGeometry`) - any real `GeometryConfig` works, matching the established
        // `FakeInterfaceSampling` test precedent.
        let placeholder = GeometryConfig::kirsch_plate_inches();
        for domain_idx in 0..2 {
            let orig_s = original.sampling_strategy(domain_idx);
            let multi_s = multi.sampling_strategy(domain_idx);
            for _ in 0..3 {
                let o_pts = orig_s.sample_interior(&placeholder, 64);
                let m_pts = multi_s.sample_interior(&placeholder, 64);
                assert_eq!(o_pts.len(), m_pts.len(), "domain {domain_idx}: point count must match");
                for (o, m) in o_pts.iter().zip(m_pts.iter()) {
                    assert!((o[0] - m[0]).abs() < 1e-15 && (o[1] - m[1]).abs() < 1e-15,
                        "domain {domain_idx}: sampled point mismatch: {o:?} vs {m:?}");
                }
            }
        }

        // Bit-identical named point sets (interface/hole rings) on both domains.
        for domain_idx in 0..2 {
            let o_sets = original.sampling_strategy(domain_idx).named_point_sets(&[]);
            let m_sets = multi.sampling_strategy(domain_idx).named_point_sets(&[]);
            let mut o_names: Vec<&str> = o_sets.iter().map(|s| s.name).collect();
            let mut m_names: Vec<&str> = m_sets.iter().map(|s| s.name).collect();
            o_names.sort(); m_names.sort();
            assert_eq!(o_names, m_names, "domain {domain_idx}: named point-set names must match exactly");
            for o_set in &o_sets {
                let m_set = m_sets.iter().find(|s| s.name == o_set.name).unwrap();
                assert_eq!(o_set.points.len(), m_set.points.len(), "point-set '{}' length mismatch", o_set.name);
                for (o, m) in o_set.points.iter().zip(m_set.points.iter()) {
                    assert!((o.x - m.x).abs() < 1e-12 && (o.y - m.y).abs() < 1e-12,
                        "point-set '{}': point mismatch ({},{}) vs ({},{})", o_set.name, o.x, o.y, m.x, m.y);
                }
            }
        }

        // Same set of loss-term names, same base_weight per name.
        let mut o_terms: Vec<&str> = original.loss_terms().iter().map(|t| t.name()).collect();
        let mut m_terms: Vec<&str> = multi.loss_terms().iter().map(|t| t.name()).collect();
        o_terms.sort(); m_terms.sort();
        assert_eq!(o_terms, m_terms, "loss-term name set must match exactly at N=1");
        for name in &o_terms {
            assert_eq!(original.base_weight(name), multi.base_weight(name), "base_weight mismatch for '{name}'");
        }
    }

    /// Issue #78 item 4 follow-up: `domain_coordinate_embeddings` closes the real, previously-
    /// disclosed "every domain uses plain Raw" limitation - one `SingleHoleChart` per annulus
    /// domain, each carrying its OWN hole's center (not a shared/first-hole value silently
    /// reused for every domain, the exact failure mode this test exists to rule out), Raw for
    /// the shared outer domain, in `self.domains()` order.
    #[test]
    fn multi_annular_domain_coordinate_embeddings_are_genuinely_per_hole() {
        let spec = ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.15, half_h: 0.06, thickness: 0.005,
                holes: vec![
                    HoleSpec { center: [-0.06, 0.0], radius: 0.009, bc: HoleBc::Free },
                    HoleSpec { center: [0.06, 0.0], radius: 0.009, bc: HoleBc::Free },
                ],
            },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec { measure_aware_training: true, ..Default::default() },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let problem = MultiAnnularDecompositionProblem::new(spec, true);
        let embeddings = problem.domain_coordinate_embeddings();
        assert_eq!(embeddings.len(), 3, "2 annulus domains + 1 outer domain");

        let center_of = |e: &pinn_core::user_geometry::CoordinateEmbedding| match e {
            pinn_core::user_geometry::CoordinateEmbedding::SingleHoleChart { center_norm, .. } => *center_norm,
            other => panic!("expected SingleHoleChart for an annulus domain, got {other:?}"),
        };
        let c0 = center_of(&embeddings[0]);
        let c1 = center_of(&embeddings[1]);
        assert!((c0[0] - c1[0]).abs() > 0.1, "the two annulus domains must NOT share the same hole center: {c0:?} vs {c1:?}");
        assert!(c0[0] < 0.0, "hole 0 is at x=-0.06, its own chart center must reflect that (negative x)");
        assert!(c1[0] > 0.0, "hole 1 is at x=+0.06, its own chart center must reflect that (positive x)");
        assert_eq!(embeddings[2], pinn_core::user_geometry::CoordinateEmbedding::Raw, "the shared outer domain must stay Raw");

        // Real, decisive proof this isn't just correct in isolation but actually reaches the
        // model: `run_multi_annular_decomposition_training` must build each annulus model with
        // THIS embedding's own input width (10 for SingleHoleChart, n_fourier=0), not the
        // uniform Raw(3) every domain used before this fix.
        assert_eq!(embeddings[0].input_dim(), 10);
        assert_eq!(embeddings[1].input_dim(), 10);
        assert_eq!(embeddings[2].input_dim(), 3);
    }

    /// Issue #77 Phase 2 architectural redesign: `OuterStageProblem` is a genuine single-domain
    /// BVP (only `OUTER_DOMAIN`), registers the new one-directional `outer_interface_anchor`
    /// term instead of a symmetric interface-continuity pair, and never registers `hole_free`
    /// (there's no hole-adjacent term in the outer domain at all, same as
    /// `AnnularDecompositionProblem`'s own outer-domain term set).
    #[test]
    fn outer_stage_problem_is_single_domain_with_one_directional_interface_anchor() {
        let spec = single_hole_like_spec(1);
        let mut spec = spec;
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = OuterStageProblem::new(spec);
        crate::problem::validate_loss_terms(&problem);
        assert_eq!(problem.domains().iter().map(|d| d.id).collect::<Vec<_>>(), vec![OUTER_DOMAIN]);
        let terms = problem.loss_terms();
        assert!(terms.iter().any(|t| t.name() == "outer_interface_anchor"),
            "Stage A must register the one-directional anchor term");
        assert!(terms.iter().all(|t| t.name() != "interface_displacement_continuity"
            && t.name() != "interface_traction_continuity"),
            "Stage A must never register the SYMMETRIC interface terms - that's the whole point of the redesign");
        assert!(terms.iter().all(|t| t.name() != "hole_free"));
    }

    /// Same registration proof for `AnnulusStageProblem`: single domain (`ANNULUS_DOMAIN`
    /// only), registers `frozen_interface_anchor` (never the symmetric pair), and its
    /// `hole_free`/hard-constraint gating mirrors `AnnularDecompositionProblem`'s own exactly.
    /// Requires `set_frozen_interface_target` to have been called first (asserted, not
    /// silently defaulted to zero - a real, deliberate misuse-prevention gate, matching this
    /// codebase's own "fail loudly on misconfiguration" discipline).
    #[test]
    fn annulus_stage_problem_is_single_domain_with_one_directional_interface_anchor() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;

        let soft = AnnulusStageProblem::new(spec.clone(), false);
        soft.set_frozen_interface_target(vec![0.0; HOLE_RING_POINTS], vec![0.0; HOLE_RING_POINTS]);
        crate::problem::validate_loss_terms(&soft);
        assert_eq!(soft.domains().iter().map(|d| d.id).collect::<Vec<_>>(), vec![ANNULUS_DOMAIN]);
        let soft_terms = soft.loss_terms();
        assert!(soft_terms.iter().any(|t| t.name() == "frozen_interface_anchor"));
        assert!(soft_terms.iter().all(|t| t.name() != "interface_displacement_continuity"
            && t.name() != "interface_traction_continuity"));
        assert!(soft_terms.iter().any(|t| t.name() == "hole_free"), "soft mode must still register hole_free");

        let hard = AnnulusStageProblem::new(spec, true);
        hard.set_frozen_interface_target(vec![0.0; HOLE_RING_POINTS], vec![0.0; HOLE_RING_POINTS]);
        let hard_terms = hard.loss_terms();
        assert!(hard_terms.iter().all(|t| t.name() != "hole_free"), "hard-constraint mode must NOT register hole_free");
    }

    #[test]
    #[should_panic(expected = "set_frozen_interface_target")]
    fn annulus_stage_problem_panics_if_loss_terms_called_before_target_set() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        let problem = AnnulusStageProblem::new(spec, false);
        let _ = problem.loss_terms(); // no set_frozen_interface_target call - must panic, not silently use a zero target
    }

    /// Issue #77 Phase 2: real, tiny end-to-end sequential run (Stage A then Stage B) proving
    /// the full pipeline - outer training, freezing, frozen-target evaluation, annulus training
    /// against that frozen target - trains without panicking and stays finite in both stages.
    #[test]
    fn sequential_two_stage_smoke_is_finite() {
        let mut spec = single_hole_like_spec(2);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.training.n_interior = 64;
        spec.training.n_boundary = 32;
        spec.network = pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() };
        let device = crate::training_core::BDevice::default();
        let mut stage_a_losses = Vec::new();
        let mut stage_b_losses = Vec::new();
        let (_outer, _annulus, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_sequential(
            spec, device, 2, 2, true, &[0, 1],
            |stage, _step, loss| {
                match stage { "stage_a" => stage_a_losses.push(loss), "stage_b" => stage_b_losses.push(loss), _ => unreachable!() }
                false
            },
            &mut |_d, _a, _o| {},
        );
        assert!(loss.is_finite(), "Stage B final loss must be finite, got {loss}");
        assert_eq!(stage_a_losses.len(), 2, "Stage A must run its own requested step count");
        assert_eq!(stage_b_losses.len(), 2, "Stage B must run its own requested step count");
        assert!(stage_a_losses.iter().all(|l| l.is_finite()));
        assert!(stage_b_losses.iter().all(|l| l.is_finite()));
        assert_eq!(diagnostics.len(), 2, "both diagnostic checkpoints must fire during Stage B");
    }

    /// Issue #77 candidate (b): `new_with_interface_weight`'s default arm (`new()`) must be
    /// byte-identical to the pre-#77-candidate-(b) hardcoded `100.0` - a real override must
    /// actually change what `base_weight` returns for both interface terms, and nothing else.
    #[test]
    fn interface_weight_override_changes_only_the_two_interface_terms_base_weight() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let default_problem = AnnularDecompositionProblem::new(spec.clone());
        assert_eq!(default_problem.base_weight("interface_displacement_continuity"), 100.0);
        assert_eq!(default_problem.base_weight("interface_traction_continuity"), 100.0);

        let overridden = AnnularDecompositionProblem::new_with_interface_weight(spec, 10.0);
        assert_eq!(overridden.base_weight("interface_displacement_continuity"), 10.0);
        assert_eq!(overridden.base_weight("interface_traction_continuity"), 10.0);
        // Every other term's base weight must be completely unaffected by the override.
        assert_eq!(overridden.base_weight("annulus_potential"), default_problem.base_weight("annulus_potential"));
        assert_eq!(overridden.base_weight("physical_potential"), default_problem.base_weight("physical_potential"));
        assert_eq!(overridden.base_weight("translation_gauge"), default_problem.base_weight("translation_gauge"));
        assert_eq!(overridden.base_weight("rotation_gauge"), default_problem.base_weight("rotation_gauge"));
    }

    /// PH4-34 gate test: `new()`'s default `hole_free_weight` must equal the pre-PH4-34
    /// hardcoded `LAM_HOLE_FREE` constant exactly, and a real override must change ONLY
    /// `hole_free`'s own base weight, nothing else.
    #[test]
    fn hole_free_weight_override_changes_only_the_hole_free_term_base_weight() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let default_problem = AnnularDecompositionProblem::new(spec.clone());
        assert_eq!(default_problem.base_weight("hole_free"), LAM_HOLE_FREE);

        let overridden = AnnularDecompositionProblem::new_with_hole_free_weight(spec, 10.0);
        assert_eq!(overridden.base_weight("hole_free"), 10.0);
        // Every other term's base weight must be completely unaffected by the override.
        assert_eq!(overridden.base_weight("annulus_potential"), default_problem.base_weight("annulus_potential"));
        assert_eq!(overridden.base_weight("physical_potential"), default_problem.base_weight("physical_potential"));
        assert_eq!(overridden.base_weight("translation_gauge"), default_problem.base_weight("translation_gauge"));
        assert_eq!(overridden.base_weight("rotation_gauge"), default_problem.base_weight("rotation_gauge"));
        assert_eq!(
            overridden.base_weight("interface_displacement_continuity"),
            default_problem.base_weight("interface_displacement_continuity"),
        );
    }

    /// PH4-35 gate test: `new()`'s default (`use_hard_constraint=false`) must register
    /// `"hole_free"` exactly as before (byte-identical), and `ansatz(ANNULUS_DOMAIN)` must be
    /// the `Identity` variant. `new_with_hard_constraint_ansatz(true)` must do the opposite:
    /// no `"hole_free"` term at all, and `ansatz(ANNULUS_DOMAIN)` must be the `HardConstraint`
    /// variant with a NONZERO `additive()` (proving the closed-form correction is actually
    /// wired in, not silently falling back to `(0,0)`) and an `eval()` in `[0,1]` (the
    /// envelope, not left at the pre-existing constant `1.0`). The outer domain's own ansatz
    /// must be `Identity` in BOTH modes - this feature is annulus-only by design.
    #[test]
    fn hard_constraint_ansatz_default_false_is_byte_identical_and_true_wires_in_the_exact_correction() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let default_problem = AnnularDecompositionProblem::new(spec.clone());
        assert!(default_problem.loss_terms().iter().any(|t| t.name() == "hole_free"),
            "default (hard-constraint disabled) must still register the soft hole_free term");
        let default_annulus_ansatz = default_problem.ansatz(0);
        // domains()[0] is ANNULUS_DOMAIN - confirmed by
        // annular_decomposition_bvp_has_two_disjoint_domains_and_bonded_interface_terms above.
        let (dx, dy) = default_annulus_ansatz.eval(0.06, 0.02, 1.0);
        assert_eq!((dx, dy), (1.0, 1.0), "default annulus ansatz must be the pre-existing Identity");
        assert_eq!(default_annulus_ansatz.additive(0.06, 0.02), (0.0, 0.0));

        let hard_problem = AnnularDecompositionProblem::new_with_hard_constraint_ansatz(spec, true);
        assert!(hard_problem.loss_terms().iter().all(|t| t.name() != "hole_free"),
            "hard-constraint mode must NOT register the now-redundant soft hole_free term");
        let hard_annulus_ansatz = hard_problem.ansatz(0);
        // A point just inside the annulus (not exactly on r=a, so the envelope is meaningfully
        // between 0 and 1, not degenerately equal to the boundary value).
        let (dx2, dy2) = hard_annulus_ansatz.eval(0.06, 0.02, 1.0);
        assert!((0.0..=1.0).contains(&dx2) && dx2 < 1.0, "envelope must suppress below 1.0 near the hole, got {dx2}");
        assert_eq!(dx2, dy2, "envelope is a single scalar applied to both u,v");
        let (ax, ay) = hard_annulus_ansatz.additive(0.06, 0.02);
        assert!(ax != 0.0 || ay != 0.0, "hard-constraint additive correction must be nonzero away from the hole center");

        // Outer domain's ansatz must stay Identity in BOTH modes - annulus-only feature.
        let outer_default = default_problem.ansatz(1).eval(0.5, 0.5, 1.0);
        let outer_hard = hard_problem.ansatz(1).eval(0.5, 0.5, 1.0);
        assert_eq!(outer_default, (1.0, 1.0));
        assert_eq!(outer_hard, (1.0, 1.0));
    }

    /// PH4-29's next candidate: `include_annulus_equilibrium=false` (the `new()` default)
    /// must never register `"equilibrium"`; `true` must register it on `ANNULUS_DOMAIN`'s
    /// own "interior" point set with `needs_hessian()==true` (proving it will actually get a
    /// real Hessian forward pass, not silently read `None`) - and every other registered term
    /// must be completely unaffected either way.
    #[test]
    fn annulus_equilibrium_registers_only_when_explicitly_enabled() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let default_problem = AnnularDecompositionProblem::new(spec.clone());
        let default_terms = default_problem.loss_terms();
        assert!(default_terms.iter().all(|t| t.name() != "equilibrium"),
            "equilibrium must NOT register by default: {:?}", default_terms.iter().map(|t| t.name()).collect::<Vec<_>>());

        let enabled_problem = AnnularDecompositionProblem::new_with_annulus_equilibrium(spec, true);
        let enabled_terms = enabled_problem.loss_terms();
        let eq = enabled_terms.iter().find(|t| t.name() == "equilibrium")
            .expect("equilibrium must register when include_annulus_equilibrium=true");
        assert_eq!(eq.domains(), vec![ANNULUS_DOMAIN]);
        assert_eq!(eq.point_sets(), vec!["interior"]);
        assert!(eq.needs_hessian(), "equilibrium term must request a real Hessian forward pass");
        assert_eq!(enabled_problem.base_weight("equilibrium"), LAM_EQUILIBRIUM_PLATE);
        // Every other term is present, unchanged, in both configurations.
        for name in ["annulus_potential", "physical_potential", "interface_displacement_continuity",
                     "interface_traction_continuity", "translation_gauge", "rotation_gauge"] {
            assert!(default_terms.iter().any(|t| t.name() == name), "missing in default: {name}");
            assert!(enabled_terms.iter().any(|t| t.name() == name), "missing when enabled: {name}");
        }
        assert_eq!(default_terms.len() + 1, enabled_terms.len(),
            "enabling must add EXACTLY one term (equilibrium), nothing else");
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let (annulus, outer, loss) = crate::user_runner::run_annular_decomposition_training(
            spec, device, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(annulus.input_dim(), 10);
        assert_eq!(outer.input_dim(), 3);
    }

    /// Issue #77 root-cause fix, Step 1: `grad_norm_rescale_period=1` forces the recalibration
    /// path to run on EVERY step of a tiny (3-step) real training run — a genuine end-to-end
    /// smoke test that `probe_term_gradients` gets set, `StepOutput.term_grad_norms` gets
    /// populated, `grad_norm_damping_factors` runs against it, and `saw.set_base_weights`
    /// doesn't panic or produce a non-finite loss, not just a static wiring check.
    /// `grad_norm_rescale_period=0` in a second run proves the disabled default path is
    /// unaffected (still trains to a finite loss).
    #[test]
    fn grad_norm_rescale_period_recalibrates_every_step_without_panicking_or_going_nonfinite() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let (_annulus, _outer, loss, _diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
            spec.clone(), device.clone(), &[], 1, 0.1, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite(), "loss must stay finite with grad-norm rescale active every step, got {loss}");

        let (_annulus0, _outer0, loss0, _diagnostics0) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
            spec, device, &[], 0, 0.1, |_step, _loss, _lr, _points| false,
        );
        assert!(loss0.is_finite(), "period=0 (disabled) must still train to a finite loss, got {loss0}");
    }

    /// Issue #77 spectral-bias fix (PH4-31): the annulus model's real, constructed
    /// `input_dim()` must reflect `annulus_n_fourier` (`10 + 4*n_fourier`), and the outer
    /// model must be COMPLETELY unaffected (stays raw 3-input) - a real, tiny end-to-end
    /// smoke run, not just a static dimension-formula check, so it also proves the forward
    /// pass doesn't panic on a width mismatch between the constructed model and
    /// `MultiStepCtx.coordinate_embedding` (see `run_annular_decomposition_training_inner`'s
    /// own comment on why those two must always agree).
    #[test]
    fn annulus_fourier_embedding_changes_only_the_annulus_models_input_dim() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 2, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        // `diagnostic_steps=&[0]` (NOT `&[]`) is load-bearing: it's the diagnostic ledger path
        // (`annular_l5_diagnostic` -> `probe_hole_boundary_profile_derived`/`probe_hole_stress_
        // diagnostic`) that actually exercises the annulus model's Fourier-augmented forward
        // pass outside the main training step - the real bug this test caught
        // (`IncompatibleShapes { left: [720, 10], right: [26, 64] }`) only fired on a
        // diagnostic checkpoint step, so a smoke test with no checkpoints at all would have
        // passed right past it.
        let (annulus, outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_annulus_fourier(
            spec, device, &[0], 4, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), 1, "diagnostic checkpoint must actually fire and succeed");
        assert!(diagnostics[0].kt_derived_fd_vm.is_finite());
        assert_eq!(annulus.input_dim(), 10 + 4 * 4, "annulus model must use the Fourier-augmented width");
        assert_eq!(outer.input_dim(), 3, "outer model must be completely unaffected");
    }

    /// `embedding_for_model` unit test (issue #77 PH4-31's own bugfix): must correctly derive
    /// Raw / plain-chart / Fourier-augmented-chart from a model's saved `input_dim()` alone,
    /// for every width this codebase actually produces.
    #[test]
    fn embedding_for_model_derives_correct_embedding_from_saved_input_width() {
        let no_hole_geom = UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] };
        let raw_model = tiny_model(&no_hole_geom);
        assert_eq!(embedding_for_model(&raw_model, &no_hole_geom), pinn_core::user_geometry::CoordinateEmbedding::Raw);

        let hole_geom = l5_geometry();
        let plain_chart_model = tiny_model(&hole_geom);
        assert_eq!(embedding_for_model(&plain_chart_model, &hole_geom), hole_geom.coordinate_embedding());

        let fourier_cfg = crate::network::ElasticityNetConfig::new()
            .with_input_dim(hole_geom.coordinate_embedding_with_fourier(4).input_dim())
            .with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let fourier_model = fourier_cfg.init(&crate::training_core::BDevice::default());
        assert_eq!(embedding_for_model(&fourier_model, &hole_geom), hole_geom.coordinate_embedding_with_fourier(4));

        // Issue #77 Phase 3: a log-polar model (input_dim=6) must resolve to LogPolar, not
        // fall through toward the Fourier branch and panic - the same bug class PH4-31's own
        // fix prevented for a different embedding variant, now covered for this one too.
        let log_polar_cfg = crate::network::ElasticityNetConfig::new()
            .with_input_dim(hole_geom.log_polar_embedding().input_dim())
            .with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let log_polar_model = log_polar_cfg.init(&crate::training_core::BDevice::default());
        assert_eq!(embedding_for_model(&log_polar_model, &hole_geom), hole_geom.log_polar_embedding());
    }

    /// Issue #77 Phase 3 architectural redesign: same discipline as
    /// `annulus_fourier_embedding_changes_only_the_annulus_models_input_dim` - real, tiny
    /// end-to-end smoke run with `diagnostic_steps=&[0]` (load-bearing: exercises the
    /// diagnostic-ledger forward pass, not just the main training step, which is exactly where
    /// `embedding_for_model`'s own LogPolar gap would have panicked before the fix above).
    /// Annulus model must use the log-polar width; outer model must be completely unaffected.
    #[test]
    fn annulus_log_polar_embedding_changes_only_the_annulus_models_input_dim() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 2, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let (annulus, outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_log_polar_embedding(
            spec, device, &[0], true, false, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), 1, "diagnostic checkpoint must actually fire and succeed");
        assert!(diagnostics[0].kt_derived_fd_vm.is_finite());
        assert_eq!(annulus.input_dim(), 6, "annulus model must use the log-polar embedding's width");
        assert_eq!(outer.input_dim(), 3, "outer model must be completely unaffected");
    }

    /// `use_log_polar=false` must be byte-identical to the pre-Phase-3 default - the annulus
    /// model keeps its `SingleHoleChart` width.
    #[test]
    fn log_polar_embedding_disabled_by_default_keeps_chart_embedding() {
        let spec = ProblemSpec {
            geometry: l5_geometry(), material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 1, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: false,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let (annulus, _outer, loss, _diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_log_polar_embedding(
            spec, device, &[], false, false, |_step, _loss, _lr, _points| false,
        );
        assert!(loss.is_finite());
        assert_eq!(annulus.input_dim(), 10, "disabled must keep the pre-existing SingleHoleChart width");
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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

    /// Issue #77 candidate (b): controlled A/B against `issue_77_l5_annular_diagnostic_trace`'s
    /// own real run (identical geometry/material/load/network/training config, identical
    /// checkpoints, identical seed - ONLY `interface_weight` differs: `10.0` here vs the
    /// default `100.0`). Tests whether `interface_displacement_continuity`/
    /// `interface_traction_continuity` over-constrain the annulus field toward the outer
    /// field's smoothness at `r=3a` - the one remaining candidate after LR/training-dynamics
    /// was ruled out with real evidence (`PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-28).
    #[test]
    #[ignore]
    fn issue_77_interface_weight_reduced_l5_trace() {
        const REDUCED_INTERFACE_WEIGHT: f32 = 10.0;
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_interface_weight(
            spec, device, &checkpoints, REDUCED_INTERFACE_WEIGHT, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 interface-weight] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-interface-weight-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 interface-weight] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 interface-weight] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-29 working hypothesis: controlled A/B against the same baseline config
    /// (identical geometry/material/load/network/training/checkpoints/seed) - ONLY
    /// `include_annulus_equilibrium=true` differs. Tests whether a strong-form residual
    /// (the SAME `EquilibriumTerm` the single-domain path already uses, reused verbatim) on
    /// the annulus domain's own "interior" points closes or narrows the Kt gap that every
    /// prior axis (representation, margin, sampling variance, LR/training-dynamics,
    /// interface-continuity weight) failed to move.
    #[test]
    #[ignore]
    fn issue_77_annulus_equilibrium_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_annulus_equilibrium(
            spec, device, &checkpoints, true, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 equilibrium] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-equilibrium-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 equilibrium] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 equilibrium] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-31 spectral-bias hypothesis: controlled A/B against the same baseline
    /// config (identical geometry/material/load/network/training/checkpoints/seed as every
    /// PH4-28/29/30 comparison) - ONLY `annulus_n_fourier=4` differs (16 extra hole-relative
    /// Fourier features on the annulus domain's own embedding). Tests whether the network's
    /// spectral bias (a well-established property of standard MLPs: fast at learning smooth,
    /// low-frequency structure, slow/unable to represent sharp local features even with
    /// correct, present gradient signal - exactly what six other ruled-out/fixed mechanisms
    /// this session pointed at) is the actual remaining bottleneck.
    #[test]
    #[ignore]
    fn issue_77_annulus_fourier_l5_trace() {
        const N_FOURIER: usize = 4;
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_annulus_fourier(
            spec, device, &checkpoints, N_FOURIER, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 fourier] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-fourier-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 fourier] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 fourier] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-31 follow-up: `n_fourier=4`'s real result was flat (Kt=1.238 vs
    /// baseline 1.231 - statistically indistinguishable), but PH4-31's own writeup flagged
    /// this as inconclusive on spectral bias generally, since `n_fourier=4` (max frequency
    /// `8*pi`) is a modest range - an under-powered frequency range would look identical to a
    /// genuinely absent effect. This doubles the range to `n_fourier=8` (max frequency
    /// `128*pi`) as the real sweep point needed before ruling spectral bias out, identical
    /// config/seed/checkpoints to every prior comparison in this investigation.
    #[test]
    #[ignore]
    fn issue_77_annulus_fourier_n8_l5_trace() {
        const N_FOURIER: usize = 8;
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_annulus_fourier(
            spec, device, &checkpoints, N_FOURIER, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 fourier-n8] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-fourier-n8-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 fourier-n8] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 fourier-n8] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 investigation follow-up: `issue_77_l5_annular_diagnostic_trace`'s real run
    /// (3000 steps) showed loss AND Kt both still moving at the final checkpoint (Kt
    /// 0.255->0.344->1.141->1.338 at steps 0/300/1500/2999, decelerating but not plateaued) -
    /// i.e. every L5 Kt number measured so far in this investigation (1.065, 1.213, 1.207) may
    /// be an UNDERTRAINED snapshot, not a converged-but-wrong one. This test extends training
    /// 4x (12000 steps) with checkpoints spread across the full range to get real evidence on
    /// whether Kt keeps climbing meaningfully toward the FEM reference or plateaus well short
    /// of it. Separate test, NOT modifying the canonical 3000-step
    /// `issue_77_l5_annular_diagnostic_trace` - that test's own checkpoint convention stays
    /// exactly as-is for future comparability.
    #[test]
    #[ignore]
    fn issue_77_l5_extended_convergence_trend_trace() {
        let spec = ProblemSpec {
            geometry: l5_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 12000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 1500, 3000, 6000, 9000, 11999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics(
            spec, device, &checkpoints, |step, loss, _lr, _points| {
                if step % 500 == 0 { println!("[#77 extended] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-extended-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 extended] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 extended] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77, user-requested isolation test: a quasi-infinite-plate geometry variant of
    /// `l5_geometry()` — hole radius 0.125in (0.003175m), plate 10in x 10in
    /// (half_w=half_h=0.127m), thickness 0.125in (0.003175m). L5's hole/half-width ratio is
    /// 0.05 (0.005/0.10); this geometry's ratio is 0.025 (0.003175/0.127) — half of L5's,
    /// closer to the idealized-infinite-plate regime the Kt=3.0 Kirsch solution assumes.
    /// `3*radius=0.009525 < half_w=0.127`, so `annular_partition()`/`decomposition_applicable`
    /// still validate — same centered single Free hole, same kinematic-decomposition/
    /// hole-correction machinery `l5_geometry()` uses, per the user's own "maintain spatial
    /// compatibility" instruction. Material/load unchanged from L5 (E=71.7e9 Pa, nu=0.33,
    /// far-field uniaxial traction 6.9e7 Pa) — the far-field BC is already applied at the
    /// plate edges, and enlarging the plate relative to the hole is itself what pushes that
    /// edge further (in ratio terms) from the hole, without needing a load change.
    fn quasi_infinite_geometry() -> UserGeometry {
        UserGeometry {
            half_w: 0.127, half_h: 0.127, thickness: 0.003175,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.003175, bc: HoleBc::Free }],
        }
    }

    #[test]
    fn quasi_infinite_geometry_still_supports_annular_decomposition() {
        let geometry = quasi_infinite_geometry();
        let partition = geometry.annular_partition().expect("3*radius must be < half_w/half_h");
        assert!((partition.interface_radius - 3.0 * 0.003175).abs() < 1e-12);
        let spec = ProblemSpec {
            geometry, material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(6.9e7), network: Default::default(),
            training: pinn_core::problem_spec::TrainingSpec {
                measure_aware_training: true, ..Default::default()
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        assert!(AnnularDecompositionProblem::supports(&spec));
    }

    /// Issue #77, user-requested isolation test: is the ~1.0-1.3 Kt plateau (against FEM
    /// 2.4606 for L5's own geometry) partly a finite-domain/finite-width artifact, rather than
    /// purely a training/representation problem? Same baseline architecture as
    /// `issue_77_l5_annular_diagnostic_trace` (kinematic decomposition, corrected hole term,
    /// annular decomposition, per-domain LR — no Fourier features, no interface-weight
    /// override, no equilibrium term: those axes are independently tested elsewhere in this
    /// investigation and are not the variable under test here), identical network/training
    /// hyperparameters and checkpoints for direct comparability — ONLY the geometry changes to
    /// `quasi_infinite_geometry()`. The correct comparison target is a FRESH FEM reference
    /// computed for THIS geometry (`tools/finite_plate_reference.py`), not L5's 2.4606 — a
    /// smaller hole/half-width ratio has its own, closer-to-3.0, mesh-converged finite-plate
    /// Kt, and comparing the network's result to the wrong target would misattribute a real
    /// geometry difference as more or less error than the network actually has.
    #[test]
    #[ignore]
    fn issue_77_quasi_infinite_plate_l5_trace() {
        let spec = ProblemSpec {
            geometry: quasi_infinite_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 3000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics(
            spec, device, &checkpoints, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 quasi-infinite] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-quasi-infinite-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 quasi-infinite] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 quasi-infinite] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 SIREN hypothesis test: controlled A/B against the same baseline config
    /// (identical geometry/material/load/network/training/checkpoints/seed as every PH4-28..33
    /// comparison) - ONLY `use_siren=true` differs on the annulus domain's own network. Unlike
    /// the Fourier-feature candidate (an input-encoding change), this replaces the activation
    /// function itself (`sin(omega_0*z)` instead of `tanh`, Sitzmann et al. 2020) — a
    /// network-wide representational change with its own specific weight-init scheme
    /// (`ElasticityNetConfig::init`'s `use_siren` branch). Tests whether the network's spectral
    /// bias is an activation-function property (this candidate) rather than an input-encoding
    /// gap (the already-rejected Fourier-feature candidate, PH4-31/32).
    #[test]
    #[ignore]
    fn issue_77_annulus_siren_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_siren(
            spec, device, &checkpoints, true, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 siren] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-siren-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 siren] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 siren] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-36 Step 1: controlled A/B against the same baseline config (identical
    /// geometry/material/load/network/training/checkpoints/seed as every PH4-28..35
    /// comparison) - ONLY `grad_norm_rescale_period` differs: `150` here (a real periodic
    /// gradient-norm-aware SAW-BRDR base-weight recalibration, `saw_brdr::grad_norm_damping_
    /// factors`) vs `0` (disabled) for every prior run. Baseline Kt at step 2999 is `1.231`
    /// (PH4-28's own real number, same config) - see PH4-36's own manifest entry for the full
    /// mechanism rationale. The plan's own Step 3 calls for running this BEFORE combining with
    /// a gentler hard-constraint envelope (Step 2), to isolate whether gradient-norm balancing
    /// alone - with the low, already-characterized variance of the soft-`hole_free` baseline -
    /// can already redirect gradient share toward `physical_potential`/`annulus_potential` and
    /// move Kt.
    #[test]
    #[ignore]
    fn issue_77_grad_norm_rescale_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
            spec, device, &checkpoints, 150, 0.1, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 grad-norm-rescale] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-grad-norm-rescale-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 grad-norm-rescale] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 grad-norm-rescale] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 grad-norm-rescale] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 PH4-36 Step 1 reproducibility: exact repeat of
    /// [`issue_77_grad_norm_rescale_l5_trace`] (same config, same seed, genuinely independent
    /// process invocation) - PH4-35's own history is the reason this exists from the start
    /// rather than being added only after a surprising result: a single favorable run must
    /// never be reported as a breakthrough without a repeat first.
    #[test]
    #[ignore]
    fn issue_77_grad_norm_rescale_l5_trace_repeat() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
            spec, device, &checkpoints, 150, 0.1, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 grad-norm-rescale-repeat] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-grad-norm-rescale-repeat-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 grad-norm-rescale-repeat] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 grad-norm-rescale-repeat] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-36 Step 1 escalation: `issue_77_grad_norm_rescale_l5_trace`'s real result
    /// (`period=150, floor=0.1`) showed the mechanism genuinely activating - `hole_free`'s
    /// weight damped `100.0 -> 39.62` by the final checkpoint, combined `physical_potential`+
    /// `annulus_potential` gradient share rising from PH4-34b's baseline ~18% to ~30% - but
    /// `hole_free` STILL dominated at 64.2% share and Kt (1.261) did not meaningfully move past
    /// baseline (1.231). This tests a genuinely more aggressive configuration - `period=75`
    /// (twice as responsive to hole_free's own real-time gradient dominance) and `floor=0.02`
    /// (allows a much larger nominal-weight cut than `0.1`'s floor did) - before concluding
    /// Step 1 alone cannot move Kt and escalating to Step 2 (a gentler hard-constraint envelope,
    /// combined with this mechanism).
    #[test]
    #[ignore]
    fn issue_77_grad_norm_rescale_aggressive_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
            spec, device, &checkpoints, 75, 0.02, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 grad-norm-rescale-aggressive] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-grad-norm-rescale-aggressive-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 grad-norm-rescale-aggressive] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 grad-norm-rescale-aggressive] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 grad-norm-rescale-aggressive] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 Phase 1 architectural redesign (PH4-38): single-domain hard-constraint. NOT a
    /// reweighting scheme (per the redesign plan's own explicit "no more reweighting"
    /// constraint) - `UserDefinedProblem`'s single network spans the WHOLE plate, so there is no
    /// second domain and no interface term for `interface_traction_continuity` to become or
    /// need to be protected from (PH4-37's actual failure mode structurally cannot occur here).
    /// Combines the exact hard-constraint ansatz (`use_hard_constraint=true`) with hole-biased
    /// sampling (`hole_bias_fraction=0.5`, matching `AnnularDecompositionProblem`'s own
    /// `n_annulus = n_interior/2` convention) so the network still sees dense near-hole
    /// collocation despite having no dedicated annulus domain. Same geometry/material/load/
    /// network/training/checkpoints as every PH4-28..37 comparison, so this Kt number is
    /// directly comparable (baseline 1.231; PH4-35 hard-constraint mean 1.2825, CV 43.9%).
    #[test]
    #[ignore]
    fn issue_77_single_domain_hard_constraint_l5_trace() {
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
            architecture: Default::default(),
        };
        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec.clone(), true, 0.5);
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_model, loss, diagnostics) = crate::user_runner::run_user_problem_training_with_diagnostics(
            problem, spec, device, &checkpoints, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 single-domain hard-constraint] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-single-domain-hard-constraint-diagnostics.json");
        let text = serde_json::to_string_pretty(&diagnostics).expect("UserProblemL5Diagnostic must serialize");
        std::fs::write(&path, text).unwrap();
        println!("[#77 single-domain hard-constraint] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 single-domain hard-constraint] step={} kt={:.9}", diagnostic.step, diagnostic.kt_derived_fd_vm);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 single-domain hard-constraint] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 Phase 1 reproducibility: exact repeat of
    /// [`issue_77_single_domain_hard_constraint_l5_trace`] - same discipline PH4-35's own
    /// history established (a single favorable run is never reported as a breakthrough without
    /// a repeat first).
    #[test]
    #[ignore]
    fn issue_77_single_domain_hard_constraint_l5_trace_repeat() {
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
            architecture: Default::default(),
        };
        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec.clone(), true, 0.5);
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_model, loss, diagnostics) = crate::user_runner::run_user_problem_training_with_diagnostics(
            problem, spec, device, &checkpoints, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 single-domain hard-constraint repeat] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        for diagnostic in &diagnostics {
            println!("[#77 single-domain hard-constraint repeat] step={} kt={:.9}", diagnostic.step, diagnostic.kt_derived_fd_vm);
        }
    }

    /// Issue #77 Phase 2 architectural redesign (PH4-39): sequential two-stage training with
    /// one-directional domain coupling. Same total step budget (3000) as every PH4-28..38
    /// comparison, split 1500/1500 between Stage A (outer alone) and Stage B (annulus alone,
    /// hard-constraint ansatz active) as the plan's own first real split. Same
    /// geometry/material/load/network/checkpoints, so directly comparable (baseline 1.231;
    /// PH4-35 hard-constraint mean 1.2825 CV 43.9%; PH4-38 single-domain result — see that
    /// entry). Checkpoints are relative to Stage B's own step counter (Stage B is where the
    /// hole-adjacent field, and therefore Kt, is actually represented).
    #[test]
    #[ignore]
    fn issue_77_sequential_two_stage_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1499];
        let (_outer, _annulus, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_sequential(
            spec, device, 1500, 1500, true, &checkpoints,
            |stage, step, loss| {
                if step % 300 == 0 { println!("[#77 sequential] {stage} step={step} loss={loss:.6e}"); }
                false
            },
            &mut |_d, _a, _o| {},
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-sequential-two-stage-diagnostics.json");
        let text = serde_json::to_string_pretty(&diagnostics).expect("UserProblemL5Diagnostic must serialize");
        std::fs::write(&path, text).unwrap();
        println!("[#77 sequential] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 sequential] stage_b_step={} kt={:.9}", diagnostic.step, diagnostic.kt_derived_fd_vm);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 sequential] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 Phase 2 reproducibility: exact repeat of
    /// [`issue_77_sequential_two_stage_l5_trace`] - same discipline PH4-35's own history
    /// established.
    #[test]
    #[ignore]
    fn issue_77_sequential_two_stage_l5_trace_repeat() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1499];
        let (_outer, _annulus, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_sequential(
            spec, device, 1500, 1500, true, &checkpoints,
            |stage, step, loss| {
                if step % 300 == 0 { println!("[#77 sequential repeat] {stage} step={step} loss={loss:.6e}"); }
                false
            },
            &mut |_d, _a, _o| {},
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        for diagnostic in &diagnostics {
            println!("[#77 sequential repeat] stage_b_step={} kt={:.9}", diagnostic.step, diagnostic.kt_derived_fd_vm);
        }
    }

    /// Issue #77 Phase 3 architectural redesign (PH4-40): log-polar coordinate embedding for
    /// the annulus domain, combined with the hard-constraint ansatz (Phase 1/PH4-35 - the two
    /// are orthogonal axes). Same geometry/material/load/network/training/checkpoints as every
    /// PH4-28..39 comparison, so directly comparable (baseline 1.231; PH4-35 hard-constraint
    /// mean 1.2825 CV 43.9%; PH4-38/39 — see those entries).
    #[test]
    #[ignore]
    fn issue_77_annulus_log_polar_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_log_polar_embedding(
            spec, device, &checkpoints, true, true, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 log-polar] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-log-polar-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 log-polar] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 log-polar] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 log-polar] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 Phase 3 reproducibility: exact repeat of
    /// [`issue_77_annulus_log_polar_l5_trace`] - same discipline PH4-35's own history
    /// established.
    #[test]
    #[ignore]
    fn issue_77_annulus_log_polar_l5_trace_repeat() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_log_polar_embedding(
            spec, device, &checkpoints, true, true, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 log-polar repeat] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        for diagnostic in &diagnostics {
            println!("[#77 log-polar repeat] step={} kt={:.9}", diagnostic.step, diagnostic.kt_derived_fd_vm);
        }
    }

    /// Issue #77 PH4-35: controlled A/B against the same baseline config (identical geometry/
    /// material/load/network/training/checkpoints/seed as every PH4-28..34 comparison) - ONLY
    /// the ansatz differs: the exact, closed-form hard-constraint hole ansatz
    /// (`kirsch_hole_correction`) instead of the soft `hole_free` penalty. Ten independent
    /// prior candidates (representation, sampling, LR, interface weight, strong-form residual,
    /// spectral bias x2, finite-domain scale, SIREN, gradient-share rebalancing) all failed or
    /// made things worse - every one of them left the SOFT penalty/energy gradient competition
    /// intact and tried to rebalance it. This removes the competition entirely: traction-free
    /// is exact by construction, so 100% of the annulus domain's gradient budget goes to
    /// `annulus_potential`/interface continuity, with nothing left to "compete" against.
    #[test]
    #[ignore]
    fn issue_77_annulus_hard_constraint_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_hard_constraint(
            spec, device, &checkpoints, true, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 hard-constraint] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-hard-constraint-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 hard-constraint] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 hard-constraint] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 PH4-36 Step 2: hard-constraint ansatz (PH4-35) COMBINED with grad-norm rescale
    /// (Step 1). PH4-35's own real diagnostic showed removing `hole_free` redirects gradient
    /// share to `interface_traction_continuity` (62.8%), NOT to `physical_potential`/
    /// `annulus_potential` (never over ~20% combined) - an unaddressed finding this test
    /// targets directly: `grad_norm_damping_factors` applies automatically to whatever term is
    /// NOT `physical_potential`/`annulus_potential`, so with `hole_free` absent (hard-constraint
    /// active) it should now damp `interface_traction_continuity` instead. `period=150,
    /// floor=0.1` - the SAME moderate settings as the first (non-escalated) Step 1 test, so this
    /// result is directly comparable to both PH4-35's own hard-constraint-alone numbers and
    /// `issue_77_grad_norm_rescale_l5_trace`'s own grad-norm-alone number.
    #[test]
    #[ignore]
    fn issue_77_hard_constraint_and_grad_norm_rescale_l5_trace() {
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_hard_constraint_and_grad_norm_rescale(
            spec, device, &checkpoints, true, 150, 0.1, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 hard-constraint+grad-norm] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-hard-constraint-and-grad-norm-rescale-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 hard-constraint+grad-norm] wrote {}", path.display());
        for diagnostic in &diagnostics {
            println!("[#77 hard-constraint+grad-norm] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
        if let Some(last) = diagnostics.last() {
            for t in &last.terms {
                println!("[#77 hard-constraint+grad-norm] final term={} raw={:.6e} weight={:.4} grad_norm={:?} grad_share={:?}",
                    t.name, t.raw, t.effective_weight, t.gradient_norm, t.gradient_share);
            }
        }
    }

    /// Issue #77 PH4-35 follow-up: the real 3000-step hard-constraint run showed Kt STILL
    /// RISING at the final checkpoint (1.196 -> 2.121 between steps 1500 and 2999, the largest
    /// single-interval jump of the whole trajectory) - unlike every prior candidate, which had
    /// plateaued or declined by 3000 steps. Extends training 4x (12000 steps), mirroring
    /// `issue_77_l5_extended_convergence_trend_trace`'s own checkpoint convention exactly, to
    /// find out where Kt actually converges rather than reading a still-moving snapshot as if
    /// it were final. Separate test, NOT modifying the canonical 3000-step
    /// `issue_77_annulus_hard_constraint_l5_trace` - that test's own checkpoints stay as-is for
    /// future comparability.
    #[test]
    #[ignore]
    fn issue_77_annulus_hard_constraint_extended_l5_trace() {
        let spec = ProblemSpec {
            geometry: l5_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 12000, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 1500, 3000, 6000, 9000, 11999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_hard_constraint(
            spec, device, &checkpoints, true, |step, loss, _lr, _points| {
                if step % 500 == 0 { println!("[#77 hard-constraint-extended] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-hard-constraint-extended-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 hard-constraint-extended] wrote {}", path.display());
        for diagnostic in diagnostics {
            println!("[#77 hard-constraint-extended] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms);
        }
    }

    /// Issue #77 gradient-share hypothesis (PH4-34): controlled A/B against the same baseline
    /// config (identical geometry/material/load/network/training/checkpoints/seed as every
    /// PH4-28..33 comparison) - ONLY `hole_free_weight` differs: `10.0` here vs the default
    /// `100.0`. Real diagnostic evidence (`AnnularTermDiagnostic.gradient_share` across every
    /// prior comparison run) shows `hole_free`'s raw loss is already tiny (~1e-4, effectively
    /// satisfied) by step ~1500, yet its gradient share climbs to 70-90% of the ENTIRE
    /// optimization's gradient budget by step 2999 - while `physical_potential` (raw ~1.0, far
    /// from converged) gets only ~15%. Tests whether a 10x weight reduction frees gradient
    /// budget for the energy terms that shape the stress field, without meaningfully
    /// un-satisfying the already-small traction residual.
    #[test]
    #[ignore]
    fn issue_77_hole_free_weight_reduced_l5_trace() {
        const REDUCED_HOLE_FREE_WEIGHT: f32 = 10.0;
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
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        let checkpoints = [0, 300, 1500, 2999];
        let (_annulus, _outer, loss, diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics_and_hole_free_weight(
            spec, device, &checkpoints, REDUCED_HOLE_FREE_WEIGHT, |step, loss, _lr, _points| {
                if step % 300 == 0 { println!("[#77 hole-free-weight] step={step} loss={loss:.6e}"); }
                false
            },
        );
        assert!(loss.is_finite());
        assert_eq!(diagnostics.len(), checkpoints.len(), "missing checkpoints: {diagnostics:?}");
        let path = std::env::temp_dir().join("issue-77-l5-hole-free-weight-diagnostics.json");
        crate::user_runner::write_annular_l5_diagnostics_json(&path, &diagnostics).unwrap();
        println!("[#77 hole-free-weight] wrote {}", path.display());
        for diagnostic in diagnostics {
            let hole = diagnostic.terms.iter().find(|t| t.name == "hole_free");
            println!("[#77 hole-free-weight] step={} kt={:.9} derived_traction_rms={:.6e} mismatch_rms={:.6e} hole_free_raw={:?} hole_free_grad_share={:?}",
                diagnostic.step, diagnostic.kt_derived_fd_vm, diagnostic.derived_traction_rms,
                diagnostic.direct_derived_mismatch_rms, hole.map(|t| t.raw), hole.and_then(|t| t.gradient_share));
        }
    }

    /// Issue #77 training-time research (`docs/L5_TRAINING_PERFORMANCE_RESEARCH.md`,
    /// candidate 2): backend-agnostic steady-state per-step timing on the real L5 annular
    /// config. `BDevice`/`B` are compile-time-selected by the `ndarray-backend` feature
    /// (`training_core`'s own doc comment) - this same test measures whichever backend the
    /// crate was built with, so the comparison is `cargo test ... issue_77_backend_comparison_
    /// timing -- --ignored --nocapture` run twice: once with the default Wgpu build, once with
    /// `--features ndarray-backend`. 300 steps: the first 50 are discarded as warmup (Phase 2's
    /// own real measurement found Wgpu's one-time kernel-compile cost concentrated in the first
    /// handful of steps - step 0 forward=264ms vs step 4 forward=35ms), the remaining 250 give
    /// the steady-state per-step number that's actually comparable across backends.
    #[test]
    #[ignore]
    fn issue_77_backend_comparison_timing() {
        let spec = ProblemSpec {
            geometry: l5_geometry(),
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() },
            training: pinn_core::problem_spec::TrainingSpec {
                max_steps: 300, n_interior: 4096, n_boundary: 4096, fd_h: 1e-3, lr: 1e-3,
                measure_aware_training: true, derivative_operator_diagnostic: false, amr_enabled: true,
            },
            formulation: pinn_core::problem_spec::FormulationSelection::Variational,
            architecture: Default::default(),
        };
        let device = crate::training_core::BDevice::default();
        const WARMUP_STEPS: usize = 50;
        let mut warmup_start: Option<std::time::Instant> = None;
        let mut steady_start: Option<std::time::Instant> = None;
        let mut steady_end: Option<std::time::Instant> = None;
        let (_annulus, _outer, loss, _diagnostics) = crate::user_runner::run_annular_decomposition_training_with_diagnostics(
            spec, device, &[], |step, _loss, _lr, _points| {
                let now = std::time::Instant::now();
                if step == 0 { warmup_start = Some(now); }
                if step == WARMUP_STEPS { steady_start = Some(now); }
                steady_end = Some(now);
                false
            },
        );
        assert!(loss.is_finite());
        let warmup_s = steady_start.unwrap().duration_since(warmup_start.unwrap()).as_secs_f64();
        let steady_s = steady_end.unwrap().duration_since(steady_start.unwrap()).as_secs_f64();
        let steady_steps = 300 - WARMUP_STEPS;
        println!("[#77 backend-timing] backend={} warmup({WARMUP_STEPS} steps)={warmup_s:.3}s \
            steady_state({steady_steps} steps)={steady_s:.3}s ({:.4} s/step)",
            if cfg!(feature = "ndarray-backend") { "ndarray" } else { "wgpu" },
            steady_s / steady_steps as f64);
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
            architecture: Default::default(),
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
            name: hole_bc_term_name(HoleBc::Free, 0),
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

    /// Issue #77 Phase 1 architectural redesign: `new_with_hard_constraint_ansatz(spec, false,
    /// 0.0)` must be byte-identical to `new()` — `hole_free` still registers, ansatz stays
    /// `Identity`. `true` must register the exact hard-constraint ansatz and NOT register
    /// `hole_free`, mirroring `AnnularDecompositionProblem`'s own
    /// `hard_constraint_ansatz_default_false_is_byte_identical_...` test exactly, adapted to
    /// this single-domain problem.
    #[test]
    fn single_domain_hard_constraint_default_false_is_byte_identical_and_true_wires_in_the_exact_correction() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;

        let default_problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec.clone(), false, 0.0);
        assert!(default_problem.loss_terms().iter().any(|t| t.name() == "hole_free"),
            "default (hard-constraint disabled) must still register the soft hole_free term");
        let default_ansatz = default_problem.ansatz(0);
        let (dx, dy) = default_ansatz.eval(0.06, 0.02, 1.0);
        assert_eq!((dx, dy), (1.0, 1.0), "default ansatz must be the pre-existing Identity");
        assert_eq!(default_ansatz.additive(0.06, 0.02), (0.0, 0.0));

        let hard_problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec, true, 0.0);
        assert!(hard_problem.loss_terms().iter().all(|t| t.name() != "hole_free"),
            "hard-constraint mode must NOT register the now-redundant soft hole_free term");
        let hard_ansatz = hard_problem.ansatz(0);
        let (dx2, dy2) = hard_ansatz.eval(0.06, 0.02, 1.0);
        assert!((0.0..=1.0).contains(&dx2) && dx2 < 1.0, "envelope must suppress below 1.0 near the hole, got {dx2}");
        assert_eq!(dx2, dy2);
        let (ax, ay) = hard_ansatz.additive(0.06, 0.02);
        assert!(ax != 0.0 || ay != 0.0, "hard-constraint additive correction must be nonzero away from the hole center");
    }

    /// Issue #78: dropping `decomposition_applicable`'s centering restriction for the hard-
    /// constraint gate is a real behavior change from the pre-#78 test above (which asserted
    /// this exact off-center case PANICS) - an off-center single Free hole is now a genuine,
    /// supported single-element `MultiHoleHardConstraint`, not a misconfiguration. Only a
    /// geometry with NO eligible (`HoleBc::Free`) hole at all must still panic.
    #[test]
    fn single_domain_hard_constraint_accepts_an_off_center_hole() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.geometry.holes[0].center = [0.01, 0.0]; // off-center - decomposition_applicable is false
        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec, true, 0.0);
        assert!(problem.hard_constraint_active(), "off-center single Free hole must still wire in the hard constraint");
        assert!(problem.loss_terms().iter().all(|t| t.name() != "hole_free"),
            "off-center hard-constraint hole must NOT register the redundant soft hole_free term");
    }

    /// A geometry with no `HoleBc::Free` hole at all (every hole `Fixed`, or no holes) must
    /// PANIC when the hard constraint is requested, not silently ignore it - a real
    /// misconfiguration, not a graceful fallback (matches this codebase's own "fail loudly"
    /// discipline elsewhere).
    #[test]
    #[should_panic(expected = "hard-constraint ansatz requires at least one Free hole")]
    fn single_domain_hard_constraint_panics_when_no_free_hole_exists() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.geometry.holes[0].bc = HoleBc::Fixed;
        let _ = UserDefinedProblem::new_with_hard_constraint_ansatz(spec, true, 0.0);
    }

    /// Issue #78 (multi-hole Kt), the load-bearing registration proof for N>1: on
    /// `triple_hole_plate.toml`'s real shipped geometry (two off-center `Free` holes flanking
    /// one `Fixed` hole), hard-constraint mode must suppress BOTH `Free` holes' soft `hole_free*`
    /// terms while leaving the `Fixed` hole's own soft `hole_fixed` penalty completely
    /// unaffected - proves the generalized `loss_terms()` gate is keyed on `HoleBc::Free`, not
    /// on `decomposition_applicable`'s narrower single-centered-hole scope.
    #[test]
    fn multi_hole_hard_constraint_suppresses_every_free_holes_term_but_not_fixed() {
        let mut spec = single_hole_like_spec(1);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![
                HoleSpec { center: [-0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
                HoleSpec { center: [0.0, -0.02], radius: 0.007, bc: HoleBc::Fixed },
                HoleSpec { center: [0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
            ],
        };

        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec, true, 0.0);
        assert!(matches!(problem.ansatz, crate::kirsch_hole_correction::AnnulusAnsatz::MultiHoleHardConstraint(ref v) if v.len() == 2),
            "two off-center Free holes must produce a 2-element MultiHoleHardConstraint");
        let terms = problem.loss_terms();
        assert!(terms.iter().all(|t| !t.name().starts_with("hole_free")),
            "every Free hole's soft term must be suppressed under the multi-hole hard constraint");
        assert!(terms.iter().any(|t| t.name() == "hole_fixed"),
            "the Fixed hole's own soft penalty must be completely unaffected");
    }

    /// Issue #78 second root-cause fix: `target_phi_at_margin` replaces the former hand-picked
    /// `TARGET_PHI_AT_MARGIN=0.9` constant with a derivation from `margin`/`fd_step`/
    /// `ENVELOPE_FD_RESOLUTION_FACTOR`. Because `margin = RING_ANCHOR_SAFETY_FACTOR * fd_step`
    /// always holds via `ring_anchor_margin_m`/`physical_fd_step_m`'s own definitions, the real
    /// call-site value collapses to a fixed constant - this test asserts that specific worked
    /// value (`1 - exp(-(RING_ANCHOR_SAFETY_FACTOR / ENVELOPE_FD_RESOLUTION_FACTOR)^2)` =
    /// `1 - exp(-4) ≈ 0.9817` at the real `4.0`/`2.0` values) so a future change to either
    /// constant is caught here, not silently.
    #[test]
    fn target_phi_at_margin_matches_the_worked_closed_form_value_at_real_ring_anchor_ratio() {
        let fd_step = 1.5e-4_f64;
        let margin = RING_ANCHOR_SAFETY_FACTOR * fd_step;
        let target = target_phi_at_margin(margin, fd_step);
        let expected = 1.0 - (-(RING_ANCHOR_SAFETY_FACTOR / ENVELOPE_FD_RESOLUTION_FACTOR).powi(2)).exp();
        assert!((target - expected).abs() < 1e-12, "target={target} expected={expected}");
        assert!((target - 0.9816843).abs() < 1e-6, "worked value drifted: target={target}");
    }

    /// `physical_fd_step_m` must match the SAME `fd_h * half_w.max(half_h)` quantity
    /// `ring_anchor_margin_m` builds its own margin from (before applying `RING_ANCHOR_SAFETY_
    /// FACTOR`) - confirmed by construction here, not assumed, since a future change to either
    /// function independently would silently break `target_phi_at_margin`'s own "margin =
    /// RING_ANCHOR_SAFETY_FACTOR * fd_step always" premise.
    #[test]
    fn physical_fd_step_m_times_ring_anchor_safety_factor_equals_ring_anchor_margin_m() {
        let geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.009, bc: HoleBc::Free }],
        };
        let fd_h = 1e-3_f32;
        let fd_step = physical_fd_step_m(fd_h, &geometry);
        let margin = ring_anchor_margin_m(fd_h, &geometry);
        assert!((margin - RING_ANCHOR_SAFETY_FACTOR * fd_step).abs() < 1e-15,
            "margin={margin} fd_step={fd_step}");
    }

    /// `multi_hole_saturation_scale`'s own derived scale must genuinely make `phi` reach the
    /// DERIVED `target_phi_at_margin` value at `r = hole_radius + margin` - the same closed-loop
    /// consistency proof the earlier (hand-picked-target) version of this derivation had.
    #[test]
    fn multi_hole_saturation_scale_derived_from_target_reaches_that_target_at_the_margin() {
        let hole_radius = 0.009_f64;
        let margin = 6e-4_f64;
        let fd_step = 1.5e-4_f64;
        let scale = multi_hole_saturation_scale(hole_radius, margin, fd_step);
        let expected_target = target_phi_at_margin(margin, fd_step);
        let r = hole_radius + margin;
        let phi = crate::kirsch_hole_correction::traction_free_envelope_scaled(r, 0.0, hole_radius, scale);
        assert!((phi - expected_target).abs() < 1e-9, "phi={phi} expected_target={expected_target}");
    }

    /// Issue #78 root-cause fix: `new_with_hard_constraint_ansatz` must pick `saturation_scale`
    /// based on how many Free holes are actually eligible, NOT unconditionally - `1.0` at N=1
    /// (load-bearing for PH4-42's own real, already-verified L5 result, which is constructed
    /// through this exact function), the geometry-DERIVED `multi_hole_saturation_scale` (see its
    /// own doc comment) only once N>1 - never a flat hand-picked constant.
    #[test]
    fn new_with_hard_constraint_ansatz_only_raises_saturation_scale_when_multiple_free_holes_exist() {
        use crate::kirsch_hole_correction::AnnulusAnsatz;

        let mut single = single_hole_like_spec(1);
        single.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        single.training.measure_aware_training = true;
        let single_problem = UserDefinedProblem::new_with_hard_constraint_ansatz(single, true, 0.0);
        match &single_problem.ansatz {
            AnnulusAnsatz::MultiHoleHardConstraint(holes) => {
                assert_eq!(holes.len(), 1);
                assert_eq!(holes[0].saturation_scale, 1.0, "N=1 must stay at the original saturation rate - byte-identical to HardConstraint");
            }
            _ => panic!("expected MultiHoleHardConstraint"),
        }

        let mut multi = single_hole_like_spec(1);
        multi.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        multi.training.measure_aware_training = true;
        multi.geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![
                HoleSpec { center: [-0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
                HoleSpec { center: [0.0, -0.02], radius: 0.007, bc: HoleBc::Fixed },
                HoleSpec { center: [0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
            ],
        };
        let margin = ring_anchor_margin_m(multi.training.fd_h, &multi.geometry);
        let fd_step = physical_fd_step_m(multi.training.fd_h, &multi.geometry);
        let expected_scale = multi_hole_saturation_scale(0.009, margin, fd_step);
        let multi_problem = UserDefinedProblem::new_with_hard_constraint_ansatz(multi, true, 0.0);
        match &multi_problem.ansatz {
            AnnulusAnsatz::MultiHoleHardConstraint(holes) => {
                assert_eq!(holes.len(), 2);
                for hole in holes {
                    assert!((hole.saturation_scale - expected_scale).abs() < 1e-9,
                        "N>1 must use the geometry-derived saturation rate for every Free hole (got {}, expected {})",
                        hole.saturation_scale, expected_scale);
                    assert!(hole.saturation_scale > 1.0,
                        "the derived scale must genuinely raise phi at the margin above the N=1 rate");
                }
            }
            _ => panic!("expected MultiHoleHardConstraint"),
        }
    }

    /// Issue #78: `UserSamplingStrategy`'s hole-biased sampling must generalize to bias EVERY
    /// `HoleBc::Free` hole (not just a single centered one), splitting the total fraction evenly
    /// - checked on `triple_hole_plate.toml`'s real geometry (two off-center Free holes, one
    /// Fixed). Every biased point must land near ONE of the two Free holes (never the Fixed
    /// one, which gets no bias budget), and both Free holes' own near-hole shares should be
    /// comparable (evenly split, not all budget going to one hole).
    #[test]
    fn hole_bias_generalizes_to_every_free_hole_and_splits_the_budget_evenly() {
        let geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![
                HoleSpec { center: [-0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
                HoleSpec { center: [0.0, -0.02], radius: 0.007, bc: HoleBc::Fixed },
                HoleSpec { center: [0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
            ],
        };
        let placeholder = geometry.to_placeholder();
        let n = 4096;
        let sampler = UserSamplingStrategy::new(geometry.clone(), 1e-3).with_hole_bias(0.6);
        let pts = sampler.sample_interior(&placeholder, n);
        assert_eq!(pts.len(), n);

        let near = |p: &[f64; 2], hole: &HoleSpec| {
            let (dx, dy) = (p[0] - hole.center[0], p[1] - hole.center[1]);
            (dx * dx + dy * dy).sqrt() <= HOLE_BIAS_RADIUS_MULTIPLIER * hole.radius
        };
        let hole0 = geometry.holes[0];
        let hole1_fixed = geometry.holes[1];
        let hole2 = geometry.holes[2];
        let n0 = pts.iter().filter(|p| near(p, &hole0)).count();
        let n2 = pts.iter().filter(|p| near(p, &hole2)).count();
        let n1 = pts.iter().filter(|p| near(p, &hole1_fixed)).count();

        let share0 = n0 as f64 / n as f64;
        let share2 = n2 as f64 / n as f64;
        assert!(share0 > 0.15 && share2 > 0.15,
            "both Free holes must get a real, comparable share of the bias budget: share0={share0} share2={share2}");
        assert!((share0 - share2).abs() < 0.1,
            "the 0.6 budget must split roughly evenly across the two Free holes: share0={share0} share2={share2}");
        // The Fixed hole gets no bias budget of its own, but its own exclusion-margin ring can
        // still coincidentally overlap the near-hole-radius test above by chance from ordinary
        // whole-plate draws - only assert it's not systematically over-represented like the two
        // Free holes are.
        assert!((n1 as f64 / n as f64) < share0.min(share2),
            "the Fixed hole must not receive a Free-hole-sized share of the bias budget");
    }

    /// Issue #77 Phase 1: `hole_bias_fraction=0.0` (every pre-existing `UserSamplingStrategy`
    /// caller) must draw the byte-identical point count/distribution as before this existed -
    /// proven here by an exact-count check (the real behavioral guarantee; the RNG SEQUENCE
    /// itself is unchanged because the biased-stratum block is structurally skipped, not fed a
    /// zero-sized loop). A nonzero fraction must land the expected SHARE of points within the
    /// bias radius, and every point (biased or not) must still respect the FD-safety margin.
    #[test]
    fn hole_bias_fraction_zero_is_unbiased_and_nonzero_concentrates_near_the_hole() {
        let geometry = l5_geometry();
        let placeholder = geometry.to_placeholder();
        let n = 2048;

        let unbiased = UserSamplingStrategy::new(geometry.clone(), 1e-3);
        let pts_unbiased = unbiased.sample_interior(&placeholder, n);
        assert_eq!(pts_unbiased.len(), n, "unbiased draw must hit the exact requested count");

        let biased = UserSamplingStrategy::new(geometry.clone(), 1e-3).with_hole_bias(0.5);
        let pts_biased = biased.sample_interior(&placeholder, n);
        assert_eq!(pts_biased.len(), n, "biased draw must still hit the exact requested count");

        let hole = geometry.holes[0];
        let bias_r = super::HOLE_BIAS_RADIUS_MULTIPLIER * hole.radius;
        let near_hole = |p: &[f64; 2]| {
            let (dx, dy) = (p[0] - hole.center[0], p[1] - hole.center[1]);
            (dx * dx + dy * dy).sqrt() <= bias_r
        };
        let share_unbiased = pts_unbiased.iter().filter(|p| near_hole(p)).count() as f64 / n as f64;
        let share_biased = pts_biased.iter().filter(|p| near_hole(p)).count() as f64 / n as f64;
        // Near-hole area is a small fraction of the whole plate for L5's geometry - the
        // unbiased share should be well under the biased target (~0.5), and the biased share
        // should land close to the requested 0.5 fraction (within stratified-sampling slop).
        assert!(share_unbiased < 0.1, "unbiased near-hole share should be small, got {share_unbiased}");
        assert!(share_biased > 0.4, "biased near-hole share should approach the 0.5 target, got {share_biased}");

        // FD-safety: every point, biased or not, must clear the anchor margin around the hole.
        let margin = ring_anchor_margin_m(1e-3, &geometry);
        for p in pts_biased.iter().chain(pts_unbiased.iter()) {
            let (dx, dy) = (p[0] - hole.center[0], p[1] - hole.center[1]);
            let r = (dx * dx + dy * dy).sqrt();
            assert!(r >= hole.radius + margin - 1e-9, "point at r={r} violates FD-safety margin");
        }
    }

    /// Issue #77 PH4-41 fix proof (finding 2): `hole_bias_quadrature_weights` must recover the
    /// TRUE domain integral of a NON-constant field (`f(x,y)=x^2+y^2`, closed-form integral
    /// `(4ab/3)(a^2+b^2)` over the square minus `pi*hole.radius^4/2` for the centered hole
    /// disk) under BOTH `fraction=0.0` (unbiased) and `fraction=0.5` (hole-biased) sampling - a
    /// constant integrand could not catch this bug (uniform-vs-biased density is invisible to a
    /// constant function), which is why this test deliberately uses a spatially-varying one.
    /// Also proves the UNWEIGHTED mean (the pre-fix behavior) is measurably WRONG under bias,
    /// making this a genuine regression proof, not just a "close enough" sanity check.
    #[test]
    fn hole_bias_quadrature_weights_recovers_true_integral_under_biased_and_unbiased_sampling() {
        let geometry = l5_geometry();
        let placeholder = geometry.to_placeholder();
        let n = 8192;
        let hole = geometry.holes[0];
        let (a, b) = (geometry.half_w, geometry.half_h);
        let square_integral = (4.0 * a * b / 3.0) * (a * a + b * b);
        let hole_integral = std::f64::consts::PI * hole.radius.powi(4) / 2.0;
        let exact = square_integral - hole_integral;
        let plate_area = 4.0 * a * b - std::f64::consts::PI * hole.radius * hole.radius;

        for fraction in [0.0_f64, 0.5] {
            let sampler = UserSamplingStrategy::new(geometry.clone(), 1e-3).with_hole_bias(fraction);
            let pts = sampler.sample_interior(&placeholder, n);
            let pts_norm: Vec<[f32; 2]> = pts.iter().map(|p| [(p[0] / a) as f32, (p[1] / b) as f32]).collect();
            let weights = hole_bias_quadrature_weights(&pts_norm, &geometry, fraction);
            let f = |p: &[f64; 2]| p[0] * p[0] + p[1] * p[1];

            let weighted_mean: f64 = pts.iter().zip(&weights).map(|(p, &w)| f(p) * w).sum::<f64>() / pts.len() as f64;
            let weighted_estimate = weighted_mean * plate_area;
            let rel_err = (weighted_estimate - exact).abs() / exact.abs();
            assert!(rel_err < 0.08,
                "fraction={fraction}: weighted estimate={weighted_estimate:e} exact={exact:e} rel_err={rel_err}");

            if fraction > 0.0 {
                let unweighted_mean: f64 = pts.iter().map(f).sum::<f64>() / pts.len() as f64;
                let unweighted_estimate = unweighted_mean * plate_area;
                let unweighted_rel_err = (unweighted_estimate - exact).abs() / exact.abs();
                assert!(unweighted_rel_err > 0.15,
                    "the UNWEIGHTED mean under bias should be measurably wrong (this is the bug \
                     the fix closes) - got rel_err={unweighted_rel_err}, expected a real, large error");
            }
        }
    }

    /// Issue #77 Phase 1: real, tiny end-to-end training run proving the combined path
    /// (hard-constraint ansatz + hole-biased sampling) trains without panicking and stays
    /// finite - same discipline as `annular_decomposition_two_model_runner_smoke_is_finite`
    /// and this branch's own grad-norm-rescale smoke test.
    #[test]
    fn single_domain_hard_constraint_and_hole_bias_smoke_is_finite() {
        let mut spec = single_hole_like_spec(3);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.training.n_interior = 64;
        spec.training.n_boundary = 32;
        spec.network = pinn_core::problem_spec::NetworkSpec { hidden_dim: 12, n_hidden: 2, ..Default::default() };

        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec.clone(), true, 0.5);
        let device = crate::training_core::BDevice::default();
        let (_model, loss) = crate::user_runner::train_user_problem_for_benchmark(&problem, spec, &device);
        assert!(loss.is_finite(), "single-domain hard-constraint + hole-bias must train to a finite loss, got {loss}");
    }

    /// Issue #78 root-cause investigation (Step 1, decisive diagnostic): does `traction_free_
    /// envelope`'s value AT the real Kt-measurement point (`hole.radius + ring_anchor_margin_m`,
    /// the tiny FD-safety margin, NOT a large offset) suppress the trained network's own
    /// contribution to near-zero - meaning Kt reads almost purely the closed-form baseline
    /// regardless of what the network has learned? Real, precise, numerically confirmed lead
    /// (`docs/multi-hole-fem-ground-truth-investigation.md`'s own "Third pass" section): for
    /// `triple_hole_plate.toml`'s real geometry, `phi ≈ 0.44%` at that exact margin (vs. `phi >
    /// 0.97` by `3*radius`) - a genuine, severe local vanishing-gradient bottleneck for the
    /// specific weights that would need to shape the field there, independent of every
    /// hyperparameter axis already falsified (none of those change what phi IS at that radius).
    /// This test measures Kt via the SAME trained model at the real margin AND at several
    /// larger radii (where phi has actually saturated), on the SAME real geometry, to confirm
    /// or refute this mechanism directly before any production fix is attempted.
    #[test]
    #[ignore]
    fn issue_78_kt_varies_with_probe_radius_matching_envelope_saturation_not_yet_a_fix() {
        let mut spec = single_hole_like_spec(600);
        spec.formulation = pinn_core::problem_spec::FormulationSelection::Variational;
        spec.training.measure_aware_training = true;
        spec.training.n_interior = 4096;
        spec.training.n_boundary = 3072;
        spec.training.amr_enabled = true;
        spec.geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![
                HoleSpec { center: [-0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
                HoleSpec { center: [0.0, -0.02], radius: 0.007, bc: HoleBc::Fixed },
                HoleSpec { center: [0.06, 0.02], radius: 0.009, bc: HoleBc::Free },
            ],
        };
        spec.network = pinn_core::problem_spec::NetworkSpec { hidden_dim: 64, n_hidden: 8, ..Default::default() };

        let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(spec.clone(), true, 0.4);
        let device = crate::training_core::BDevice::default();
        let (model, loss) = crate::user_runner::train_user_problem_for_benchmark(&problem, spec.clone(), &device);
        assert!(loss.is_finite());

        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let fd = crate::fd_stencil::FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
        let nominal_stress = spec.load.px.abs();
        let real_margin = ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);

        for &hole_idx in &[0usize, 2] {
            let hole = &spec.geometry.holes[hole_idx];
            println!("=== hole {hole_idx} (radius={}) ===", hole.radius);
            println!("  real FD-safety margin={real_margin:.6e} m ({:.4}% of radius)", 100.0 * real_margin / hole.radius);
            for margin_mult in [1.0_f64, 5.0, 20.0, 50.0, 100.0, 300.0] {
                let margin = real_margin * margin_mult;
                let phi = crate::kirsch_hole_correction::traction_free_envelope(hole.radius + margin, 0.0, hole.radius);
                let profile = probe_hole_boundary_profile_derived(
                    &model, &spec.geometry, hole, 72, &fd, scales.u_ref, spec.load.px, &spec.material,
                    margin, &device, problem.ansatz(0), (decomposition_applicable(&spec) || problem.hard_constraint_active()).then_some((spec.load.px, spec.load.py)),
                );
                let sc = stress_concentration_from_profile(&profile, nominal_stress);
                println!("  margin_mult={margin_mult:>6.1}  margin={margin:.6e} m  phi={phi:.6}  Kt_vm={:.4}", sc.kt);
            }
        }
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

    // ─── Issue #77 Step 2: hole-relative (not plate-relative) ring margin ───────────────────

    /// `hole_ring_margin_m` must scale with the HOLE's own radius, not the plate size — the
    /// direct falsification of the pre-Step-2 defect (`ring_anchor_margin_m` gives a
    /// margin/radius ratio that gets proportionally worse as the hole shrinks). Same plate,
    /// three different hole sizes: the ratio `margin/radius` must be IDENTICAL (hole-relative),
    /// unlike `ring_anchor_margin_m`'s ratio, which must DIFFER across the same three holes
    /// (plate-relative) - both properties checked together so this test cannot pass by
    /// accident.
    #[test]
    fn hole_ring_margin_scales_with_hole_radius_not_plate_size() {
        let make = |radius: f64| UserGeometry {
            half_w: 0.10, half_h: 0.10, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius, bc: HoleBc::Free }],
        };
        let radii = [0.005, 0.01, 0.02];
        let hole_ratios: Vec<f64> = radii.iter().map(|&r| hole_ring_margin_m(r) / r).collect();
        for &ratio in &hole_ratios {
            assert!((ratio - HOLE_MARGIN_FRACTION).abs() / HOLE_MARGIN_FRACTION < 1e-9,
                "hole_ring_margin_m/radius must be the fixed fraction {HOLE_MARGIN_FRACTION} \
                 for every hole size, got {ratio}");
        }
        let plate_ratios: Vec<f64> = radii.iter()
            .map(|&r| ring_anchor_margin_m(TEST_FD_H, &make(r)) / r)
            .collect();
        assert!(plate_ratios[0] > plate_ratios[1] * 1.9 && plate_ratios[1] > plate_ratios[2] * 1.9,
            "sanity check on the OLD plate-scaled margin: its margin/radius ratio must roughly \
             halve each time radius doubles (it's independent of radius), got {plate_ratios:?} - \
             if this assertion itself fails, `ring_anchor_margin_m` changed and this test's own \
             premise needs revisiting");
    }

    /// The FD-safety property Step 2 must preserve at the NEW, tighter margin: every stencil
    /// arm reachable by `hole_ring_fd_config`'s own (smaller) step, evaluated from a point on
    /// the "hole_i_fd" ring, must stay outside the hole. Direct geometric check (not a training
    /// run) - the same style as `sample_interior_points_stay_outside_the_fd_safe_margin_...`
    /// above, adapted to the ring's own margin/FD-step pairing instead of the interior
    /// sampler's plate-scaled one.
    #[test]
    fn hole_ring_fd_config_keeps_every_stencil_arm_outside_the_hole() {
        for radius in [0.002, 0.005, 0.02, 0.05] {
            let geom = UserGeometry {
                half_w: 0.10, half_h: 0.10, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius, bc: HoleBc::Free }],
            };
            let fd = hole_ring_fd_config(radius, &geom);
            let ring_r = radius + hole_ring_margin_m(radius);
            // Physical reach of the ring's own (smaller) FD step, worst case (arm pointing
            // straight at the hole center): `fd.hx * half_w` / `fd.hy * half_h`.
            let reach_x = fd.hx as f64 * geom.half_w;
            let reach_y = fd.hy as f64 * geom.half_h;
            let worst_case_r = ring_r - reach_x.max(reach_y);
            assert!(worst_case_r > radius,
                "radius={radius}: ring at r={ring_r} with FD reach {reach_x}/{reach_y} leaves \
                 worst-case stencil arm at r={worst_case_r}, inside or on the hole (radius={radius})");
        }
    }

    /// `hole_fd_config_for_geometry` must fall back to the caller's own `fd` (byte-identical,
    /// no shrinking) for every geometry outside single-centered-hole scope — no-hole and
    /// multi-hole must be completely unaffected by Step 2, matching Step 1's own scope
    /// discipline.
    #[test]
    fn hole_fd_config_for_geometry_falls_back_to_fd_outside_single_hole_scope() {
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 0.2, 0.2);
        let no_hole = UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] };
        assert_eq!(hole_fd_config_for_geometry(&fd, &no_hole).hx, fd.hx);

        let multi = two_hole_geometry();
        assert_eq!(hole_fd_config_for_geometry(&fd, &multi).hx, fd.hx);

        let single = UserGeometry {
            half_w: 0.10, half_h: 0.10, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.005, bc: HoleBc::Free }],
        };
        assert_ne!(hole_fd_config_for_geometry(&fd, &single).hx, fd.hx,
            "single-hole geometry must get a genuinely different (smaller) hole_fd");
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
            architecture: Default::default(),
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

    /// Issue #78 Stage 1.1 fix proof: two holes sharing a BC (mirrors `triple_hole_plate.
    /// toml`'s real two-Free-hole shape) must produce two DISTINCT term names, not a collision
    /// that would collapse in `training_core`'s `lam_by_name`/`raw_scalar_by_name`/`term_grad_
    /// norms` `HashMap<&str, _>`s. Before this fix, both terms returned the literal constant
    /// `"hole_free"` here - this test fails against the pre-fix code (both entries equal) and
    /// passes against the fixed one.
    #[test]
    fn hole_bc_terms_get_distinct_names_when_two_holes_share_a_bc() {
        let spec = ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.05, thickness: 0.005,
                holes: vec![
                    HoleSpec { center: [-0.03, 0.0], radius: 0.005, bc: HoleBc::Free },
                    HoleSpec { center: [0.03, 0.0], radius: 0.005, bc: HoleBc::Free },
                    HoleSpec { center: [0.0, 0.02], radius: 0.004, bc: HoleBc::Fixed },
                ],
            },
            material: MaterialProps::al7075_t6(),
            load: LoadConfig::uniaxial_x(1e7),
            network: Default::default(),
            training: Default::default(),
            formulation: pinn_core::problem_spec::default_formulation(),
            architecture: Default::default(),
        };
        let problem = UserDefinedProblem::new(spec);
        let terms = problem.loss_terms();
        let hole_names: Vec<&str> = terms.iter().map(|t| t.name()).filter(|n| n.starts_with("hole_")).collect();
        assert_eq!(hole_names.len(), 3, "one term per hole: {hole_names:?}");
        // The two Free holes must NOT share a name - this is the actual bug this fix closes.
        let free_names: Vec<&&str> = hole_names.iter().filter(|n| n.starts_with("hole_free")).collect();
        assert_eq!(free_names.len(), 2);
        assert_ne!(free_names[0], free_names[1], "two Free holes must get distinct term names, not collide");
        // First hole of its BC keeps the exact pre-#78 constant name - every existing
        // single/mixed-BC spec's term names, base_weight lookups, and diagnostics stay
        // byte-identical.
        assert!(hole_names.contains(&"hole_free"), "first Free hole must keep the unsuffixed name");
        assert!(hole_names.contains(&"hole_fixed"), "the lone Fixed hole must keep the unsuffixed name");
        // Every hole term's name must still resolve through base_weight without panicking -
        // the actual reader this fix had to keep working for suffixed names too.
        for name in &hole_names {
            let _ = problem.base_weight(name); // panics on an unrecognized name - the assertion IS not panicking
        }
    }

    /// Issue #78 Stage 1.1: `hole_bc_term_name` itself - the pure naming rule, independent of
    /// `loss_terms()`'s own wiring.
    #[test]
    fn hole_bc_term_name_suffixes_only_the_second_and_later_occurrence() {
        assert_eq!(hole_bc_term_name(HoleBc::Free, 0), "hole_free");
        assert_eq!(hole_bc_term_name(HoleBc::Free, 1), "hole_free_1");
        assert_eq!(hole_bc_term_name(HoleBc::Free, 2), "hole_free_2");
        assert_eq!(hole_bc_term_name(HoleBc::Fixed, 0), "hole_fixed");
        assert_eq!(hole_bc_term_name(HoleBc::Fixed, 1), "hole_fixed_1");
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            hole_scale: make_gate_optim(),
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
            fd: &fd, hole_fd: &fd, per_domain_lr: None,
            k: 1.0,
            domains: vec![DomainStepCtx { data: &data, u_ref: scales.u_ref, ref_energy: scales.ref_energy, ref_stress2: scales.ref_stress2 }],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: f64::MAX,
            dynamic_lam_non_tension_cap: f64::MAX,
            constitutive_consistency_weight: 50.0,
            n_fourier: spec.geometry.n_fourier(),
            coordinate_embedding: spec.geometry.coordinate_embedding(),
            domain_coordinate_embeddings: None,
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            &IdentityAnsatz, None,
        );
        assert_eq!(direct.len(), derived.len());
        for (d, c) in direct.iter().zip(&derived) {
            assert!((d.x - c.x).abs() < 1e-12 && (d.y - c.y).abs() < 1e-12,
                "direct and derived stress must be sampled at identical coordinates");
        }
        let diagnostic = probe_hole_stress_diagnostic(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, &MaterialProps::al7075_t6(), margin, &device, &IdentityAnsatz,
        );
        assert_eq!(diagnostic.radial_offset_m, margin);
        assert!(diagnostic.stress_mismatch_rms.is_finite() && diagnostic.stress_mismatch_max.is_finite());
        assert!(diagnostic.direct_traction_rms.is_finite() && diagnostic.derived_traction_rms.is_finite());
    }

    /// Issue #77 PH4-41 fix proof (finding 1, affine half): `probe_hole_boundary_profile_derived`
    /// with `affine_strain_pair=Some((px,py))` must differ from the SAME call with `None` by
    /// EXACTLY `affine_strain(px,py,material)`'s own constant values, for ANY network - this is
    /// an unconditional, network-independent arithmetic identity (the fix adds a known constant
    /// to FD-derived strain, `eps_xx.add_scalar(a_exx)`, not something that depends on what the
    /// network predicts), so a real (not zeroed/mocked) freshly-initialized network is a valid
    /// probe. Before this fix, both calls would have been byte-identical (the function never
    /// read `affine_strain_pair` at all) - this test fails against the pre-fix function and
    /// passes against the fixed one, making it a genuine regression proof, not just a shape check.
    #[test]
    fn probe_hole_boundary_profile_derived_adds_affine_strain_exactly() {
        let device = crate::training_core::BDevice::default();
        let model = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device);
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let margin = 0.003;
        let (px, py) = (6.9e7_f64, 2.0e7_f64);

        let without_affine = probe_hole_boundary_profile_derived(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, &material, margin, &device, &IdentityAnsatz, None,
        );
        let with_affine = probe_hole_boundary_profile_derived(
            &model, &geometry, &hole, 16, &fd, 1.0, 1.0, &material, margin, &device, &IdentityAnsatz, Some((px, py)),
        );
        let (a_exx, a_eyy, a_exy) = affine_strain(px, py, &material);
        assert!(a_exx != 0.0 && a_eyy != 0.0, "test fixture must use a genuinely nonzero affine strain, got ({a_exx}, {a_eyy})");

        for (w, wo) in with_affine.iter().zip(&without_affine) {
            assert!((w.eps_xx as f64 - wo.eps_xx as f64 - a_exx).abs() < 1e-6 * a_exx.abs().max(1.0),
                "eps_xx must shift by exactly a_exx={a_exx:e}: with={} without={}", w.eps_xx, wo.eps_xx);
            assert!((w.eps_yy as f64 - wo.eps_yy as f64 - a_eyy).abs() < 1e-6 * a_eyy.abs().max(1.0),
                "eps_yy must shift by exactly a_eyy={a_eyy:e}: with={} without={}", w.eps_yy, wo.eps_yy);
            assert!((w.eps_xy as f64 - wo.eps_xy as f64 - a_exy).abs() < 1e-9,
                "eps_xy must shift by exactly a_exy={a_exy:e} (0 for this shear-free load): with={} without={}", w.eps_xy, wo.eps_xy);
            // sxx/syy/sxy must therefore also differ (Hooke's law is linear, so a nonzero
            // strain shift produces a nonzero stress shift) - a real, decisive symptom of the
            // pre-fix bug: before this fix, `with_affine`/`without_affine` were byte-identical.
            assert!(w.sxx != wo.sxx || w.syy != wo.syy,
                "stress must differ once affine strain is included - pre-fix bug would make these identical");
        }
    }

    /// Issue #77 PH4-41 fix proof (finding 1, ansatz half): under the hard-constraint ansatz,
    /// the reconstructed displacement very close to the hole boundary must be DOMINATED by the
    /// exact closed-form Kirsch correction (`kirsch_hole_correction::kirsch_hole_displacement`),
    /// not by the network's own (small, freshly-initialized) raw output - because
    /// `traction_free_envelope` suppresses the network's contribution toward zero at `r=a`
    /// while the additive correction is NOT suppressed. Before this fix,
    /// `probe_hole_boundary_profile_derived` never called `ansatz.eval()`/`ansatz.additive()`
    /// at all, so switching from `Identity` to `HardConstraint` would have changed NOTHING -
    /// this test fails against the pre-fix function (both ansatz choices give byte-identical
    /// `ux`/`uy`) and passes against the fixed one.
    #[test]
    fn probe_hole_boundary_profile_derived_reflects_hard_constraint_ansatz_near_hole_boundary() {
        let device = crate::training_core::BDevice::default();
        let model = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5)
            .init(&device);
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        // A small margin (hole-relative, matching hole_ring_margin_m's own 0.02*radius
        // convention) so the envelope is still close to its r=a value (heavily suppressing the
        // network), making the closed-form correction's dominance a decisive, not marginal,
        // effect.
        let margin = 0.02 * hole.radius;
        let (px, py) = (6.9e7_f64, 0.0_f64);
        let u_ref = 1e-3_f32; // a physically plausible displacement reference scale

        let identity_ansatz = IdentityAnsatz;
        let hard_ansatz = crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(
            crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                hole_center: hole.center, hole_radius: hole.radius,
                half_w: geometry.half_w, half_h: geometry.half_h,
                px, py, e: material.e as f64, nu: material.nu as f64, u_ref: u_ref as f64, saturation_scale: 1.0,
                trainable: false,
            },
        );
        let via_identity = probe_hole_boundary_profile_derived(
            &model, &geometry, &hole, 8, &fd, u_ref, px, &material, margin, &device, &identity_ansatz, None,
        );
        let via_hard_constraint = probe_hole_boundary_profile_derived(
            &model, &geometry, &hole, 8, &fd, u_ref, px, &material, margin, &device, &hard_ansatz, None,
        );

        let r = hole.radius + margin;
        // RMS comparison across all probed points, not a per-point relative error (which is
        // ill-conditioned wherever the closed-form correction's own component crosses zero by
        // symmetry - e.g. `u_hole_x` vanishes at certain angles for a uniaxial load; comparing
        // a tiny "got" against a near-zero "expected" there would blow up a naive relative
        // error despite both being genuinely, correctly small). RMS magnitude is well-
        // conditioned everywhere and still a real, decisive proof.
        let mut sq_diff = 0.0_f64;
        let mut sq_expected = 0.0_f64;
        for (via_id, via_hard) in via_identity.iter().zip(&via_hard_constraint) {
            assert!((via_id.ux - via_hard.ux).abs() > 1e-9 || (via_id.uy - via_hard.uy).abs() > 1e-9,
                "hard-constraint ansatz must change the reconstructed displacement - pre-fix bug \
                 would make these byte-identical regardless of which ansatz is passed");
            let (x, y) = (via_hard.x, via_hard.y);
            let (expected_ux, expected_uy) = crate::kirsch_hole_correction::kirsch_hole_displacement(
                x, y, hole.radius, material.e as f64, material.nu as f64, px, py,
            );
            let (dx, dy) = (via_hard.ux as f64 - expected_ux, via_hard.uy as f64 - expected_uy);
            sq_diff += dx * dx + dy * dy;
            sq_expected += expected_ux * expected_ux + expected_uy * expected_uy;
        }
        // Envelope at this small margin: phi(r) = 1 - exp(-((r-a)/a)^2) with (r-a)/a=0.02 -
        // phi ~ 4e-4, so the network's own (bounded, freshly-initialized) contribution is
        // suppressed to well under 1% of a typical correction magnitude - the reconstructed
        // displacement should closely track the closed form in aggregate, not equal it exactly
        // (the network's suppressed-but-nonzero contribution is real and expected).
        let rms_rel_err = (sq_diff / sq_expected.max(1e-30)).sqrt();
        assert!(rms_rel_err < 0.1,
            "RMS reconstructed displacement at r={r:.6} should closely track the closed-form \
             Kirsch correction under the hard-constraint ansatz: rms_rel_err={rms_rel_err}");
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

    /// `envelope_margin_for_target` must be the exact inverse of `multi_hole_saturation_scale`
    /// (round-trip: derive a scale for a target margin, then solve back for the margin at that
    /// same target, recover the original margin).
    #[test]
    fn envelope_margin_for_target_inverts_multi_hole_saturation_scale() {
        let hole_radius = 0.009_f64;
        let margin = 6e-4_f64;
        let fd_step = 1.5e-4_f64;
        let target = target_phi_at_margin(margin, fd_step);
        let scale = multi_hole_saturation_scale(hole_radius, margin, fd_step);
        let recovered_margin = envelope_margin_for_target(hole_radius, scale, target);
        assert!((recovered_margin - margin).abs() < 1e-12,
            "recovered={recovered_margin} expected={margin}");
    }

    /// Issue #78 second root-cause fix: at `scale=1.0` (N=1, or any pre-#78 ansatz), the
    /// second radial probe's margin must be EXACTLY `margin_coarse * 1.5` - the original,
    /// byte-identical formula - never the envelope-saturated one (which would be nonsensically
    /// large at `scale=1.0`, see `KtConvergenceReport`'s own doc comment).
    #[test]
    fn kt_convergence_radial_probe_margin_is_exactly_1_5x_at_unit_scale() {
        let margin_coarse = 6e-4_f64;
        let ansatz = crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
            hole_center: [0.0, 0.0], hole_radius: 0.009, half_w: 0.15, half_h: 0.06,
            px: 6.9e7, py: 0.0, e: 71.7e9, nu: 0.33, u_ref: 1e-6, saturation_scale: 1.0,
            trainable: false,
        };
        let margin = kt_convergence_radial_probe_margin(&ansatz, [0.0, 0.0], 0.009, margin_coarse);
        assert!((margin - margin_coarse * 1.5).abs() < 1e-15, "margin={margin}");

        // Also exactly 1.5x when the ansatz has no envelope at all (Identity/no match).
        let margin_identity = kt_convergence_radial_probe_margin(&IdentityAnsatz, [0.0, 0.0], 0.009, margin_coarse);
        assert!((margin_identity - margin_coarse * 1.5).abs() < 1e-15, "margin_identity={margin_identity}");
    }

    /// Once `saturation_scale > 1.0` (the real N>1 regime), the second radial probe must move
    /// FARTHER out than the original `1.5x` margin - never smaller (FD-stencil safety), and
    /// genuinely different (proving the fix actually engages, not a no-op).
    #[test]
    fn kt_convergence_radial_probe_margin_exceeds_1_5x_once_scale_exceeds_one() {
        let margin_coarse = 6e-4_f64;
        let hole_radius = 0.009_f64;
        let ansatz = crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
            hole_center: [0.0, 0.0], hole_radius, half_w: 0.15, half_h: 0.06,
            px: 6.9e7, py: 0.0, e: 71.7e9, nu: 0.33, u_ref: 1e-6, saturation_scale: 22.75,
            trainable: false,
        };
        let margin = kt_convergence_radial_probe_margin(&ansatz, [0.0, 0.0], hole_radius, margin_coarse);
        assert!(margin > margin_coarse * 1.5, "expected margin > 1.5x margin_coarse, got {margin}");
        // Sanity bound: this must still stay a small fraction of the hole's own radius, not
        // blow up to plate scale - confirms `ENVELOPE_MEASUREMENT_SATURATED_TARGET`'s own
        // choice stays physically sensible at a real derived scale.
        assert!(margin < hole_radius, "margin={margin} should stay well under the hole radius at this scale");
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
            &IdentityAnsatz, None,
        );
        assert!(report.kt_coarse.is_finite(), "{report:?}");
        assert!(report.kt_fine_angular.is_finite(), "{report:?}");
        assert!(report.kt_coarse_margin_1_5x.is_finite(), "{report:?}");
        assert!(report.angular_relative_change.is_finite(), "{report:?}");
        assert!(report.radial_relative_change.is_finite(), "{report:?}");
        // IdentityAnsatz + no affine background = no closed-form baseline to subtract at all.
        assert_eq!(report.radial_residual_kt_delta, None, "{report:?}");
    }

    /// `closed_form_only_kt_at_margin`, computed generically via `ansatz.additive()`, must
    /// numerically agree with `kirsch_hole_correction`'s own independent, hand-derived
    /// reimplementation (`closed_form_only_kt_varies_meaningfully_between_the_two_radial_
    /// probe_points_at_real_scale`) for the SAME geometry/scale/margins - two independently
    /// written computations of the same real quantity agreeing is real evidence neither has a
    /// transcription bug, not merely "it compiles."
    /// Debug isolation: does `ansatz.additive(xn,yn)` (via `AnnulusAnsatz::MultiHoleHardConstraint`)
    /// return the SAME physical displacement as directly calling `kirsch_hole_displacement`
    /// per-hole and summing, at ONE specific point? Narrows whether a mismatch is in the ansatz
    /// plumbing or in `closed_form_only_kt_at_margin`'s own FD/stress math.
    #[test]
    fn debug_ansatz_additive_matches_direct_kirsch_hole_displacement_sum_at_one_point() {
        let a = 0.009_f64;
        let px = 6.9e7_f64;
        let half_w = 0.15_f64;
        let half_h = 0.06_f64;
        let centers = [[-0.06_f64, 0.02_f64], [0.06_f64, 0.02_f64]];
        let ansatz = crate::kirsch_hole_correction::AnnulusAnsatz::MultiHoleHardConstraint(
            centers.iter().map(|&c| crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                hole_center: c, hole_radius: a, half_w, half_h,
                px, py: 0.0, e: 71.7e9, nu: 0.33, u_ref: 1.0, saturation_scale: 30.0,
                trainable: false,
            }).collect()
        );
        // A point on hole0's own boundary ring.
        let x0 = centers[0][0] + (a + 6e-4) * 0.7_f64.cos();
        let y0 = centers[0][1] + (a + 6e-4) * 0.7_f64.sin();
        use pinn_core::problem::DirichletAnsatz;
        let (ax, ay) = ansatz.additive((x0 / half_w) as f32, (y0 / half_h) as f32);

        let mut ex = 0.0_f64;
        let mut ey = 0.0_f64;
        for &c in &centers {
            let (hx, hy) = crate::kirsch_hole_correction::kirsch_hole_displacement(x0 - c[0], y0 - c[1], a, 71.7e9, 0.33, px, 0.0);
            ex += hx;
            ey += hy;
        }
        println!("ansatz.additive -> ({ax}, {ay})   direct sum -> ({ex}, {ey})");
        // f32 precision (`additive`'s own return type), not f64 - a tight-but-realistic
        // tolerance given the magnitudes involved here (~1e-5).
        assert!((ax as f64 - ex).abs() < 1e-6 * ex.abs().max(1e-12), "ux mismatch: ansatz={ax} direct={ex}");
        assert!((ay as f64 - ey).abs() < 1e-6 * ey.abs().max(1e-12), "uy mismatch: ansatz={ay} direct={ey}");
    }

    #[test]
    fn closed_form_only_kt_at_margin_matches_the_independent_kirsch_hole_correction_reimplementation() {
        let a = 0.009_f64;
        let px = 6.9e7_f64;
        let margin_coarse = 6e-4_f64;
        let scale = 30.0_f64;
        let geometry = UserGeometry {
            half_w: 0.15, half_h: 0.06, thickness: 0.006,
            holes: vec![
                HoleSpec { center: [-0.06, 0.02], radius: a, bc: HoleBc::Free },
                HoleSpec { center: [0.0, -0.02], radius: 0.007, bc: HoleBc::Fixed },
                HoleSpec { center: [0.06, 0.02], radius: a, bc: HoleBc::Free },
            ],
        };
        let material = MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 };
        let ansatz = crate::kirsch_hole_correction::AnnulusAnsatz::MultiHoleHardConstraint(vec![
            crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                hole_center: [-0.06, 0.02], hole_radius: a, half_w: 0.15, half_h: 0.06,
                px, py: 0.0, e: material.e as f64, nu: material.nu as f64, u_ref: 1.0, saturation_scale: scale,
                trainable: false,
            },
            crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                hole_center: [0.06, 0.02], hole_radius: a, half_w: 0.15, half_h: 0.06,
                px, py: 0.0, e: material.e as f64, nu: material.nu as f64, u_ref: 1.0, saturation_scale: scale,
                trainable: false,
            },
        ]);
        let hole0 = geometry.holes[0];
        let nominal_stress = px;
        // The independent reference test includes the affine uniaxial background
        // (`a_exx*x + a_eyy*y`, `py=0.0`) - must pass the matching `Some((px, 0.0))` here, not
        // `None`, or this cross-check compares two different physical fields.
        let affine = Some((px, 0.0));
        let kt_1 = closed_form_only_kt_at_margin(&ansatz, &geometry, &hole0, margin_coarse, 1.0, &material, affine, nominal_stress, 72);
        let kt_1_5 = closed_form_only_kt_at_margin(&ansatz, &geometry, &hole0, margin_coarse * 1.5, 1.0, &material, affine, nominal_stress, 72);
        // From the independent kirsch_hole_correction.rs test: margin=2.5075, margin*1.5=2.3231.
        // 0.5% relative tolerance, not an exact match - this function round-trips the FD-
        // perturbed point through `additive`'s `f32` (xn,yn) contract, while the reference
        // test in `kirsch_hole_correction.rs` computes everything in pure `f64`; a small,
        // expected precision difference between the two independent implementations, not a
        // logic error (confirmed by `debug_ansatz_additive_matches_direct_kirsch_hole_
        // displacement_sum_at_one_point`'s own tight single-point agreement).
        assert!((kt_1 - 2.5075).abs() / 2.5075 < 5e-3, "kt_1={kt_1}, expected ~2.5075");
        assert!((kt_1_5 - 2.3231).abs() / 2.3231 < 5e-3, "kt_1_5={kt_1_5}, expected ~2.3231");
    }

    /// End-to-end: `radial_residual_kt_delta` must be `Some` and finite for a real multi-hole
    /// hard-constraint ansatz (a genuine baseline exists to subtract), unlike the `IdentityAnsatz`
    /// case above.
    #[test]
    fn kt_convergence_check_reports_a_residual_when_a_real_baseline_ansatz_is_active() {
        let device = crate::training_core::BDevice::default();
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let fd = crate::fd_stencil::FdConfig::new(TEST_FD_H, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let hole = geometry.holes[0];
        let margin = ring_anchor_margin_m(TEST_FD_H, &geometry);
        let ansatz = crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
            hole_center: hole.center, hole_radius: hole.radius, half_w: geometry.half_w, half_h: geometry.half_h,
            px: 1e7, py: 0.0, e: material.e as f64, nu: material.nu as f64, u_ref: 1.0, saturation_scale: 20.0,
            trainable: false,
        };
        let report = kt_convergence_check(
            &model, &geometry, &hole, 32, &fd, 1.0, 1e7, &material, margin, 1e7, 0.5, &device,
            &ansatz, None,
        );
        assert!(report.radial_residual_kt_delta.is_some_and(f64::is_finite), "{report:?}");
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

    /// Issue #77 PH4-45 fix proof (mirrors `probe_hole_boundary_profile_derived_adds_affine_
    /// strain_exactly`): `evaluate_user_vis_grid`'s heatmap must add the affine background
    /// strain exactly, not silently report `u_hole`/`eps_hole` alone as if it were the total
    /// field - the bug this fix closes. Before this fix, `evaluate_user_vis_grid` had no
    /// `affine_strain_pair` parameter at all, so this comparison would be impossible to write
    /// against the pre-fix signature (a real, decisive difference from the fixed one, not just
    /// a stronger assertion on the same behavior).
    #[test]
    fn evaluate_user_vis_grid_adds_affine_strain_exactly() {
        let device = crate::training_core::BDevice::default();
        let model = tiny_model_raw();
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let (px, py) = (6.9e7_f64, 2.0e7_f64);

        let without_affine = evaluate_user_vis_grid(
            &model, &geometry, [6, 6], 1.0, 1.0, &material, &fd, &[], &device, &IdentityAnsatz, None,
        );
        let with_affine = evaluate_user_vis_grid(
            &model, &geometry, [6, 6], 1.0, 1.0, &material, &fd, &[], &device, &IdentityAnsatz, Some((px, py)),
        );
        let (a_exx, a_eyy, _a_exy) = affine_strain(px, py, &material);
        assert!(a_exx != 0.0 && a_eyy != 0.0, "test fixture must use a genuinely nonzero affine strain");

        let mut checked = 0;
        for ((row, col), &wo) in without_affine.eps_xx.indexed_iter() {
            if wo.is_nan() { continue; }
            let w = with_affine.eps_xx[(row, col)];
            assert!((w as f64 - wo as f64 - a_exx).abs() < 1e-6 * a_exx.abs().max(1.0),
                "eps_xx must shift by exactly a_exx={a_exx:e} at ({row},{col}): with={w} without={wo}");
            let (w_yy, wo_yy) = (with_affine.eps_yy[(row, col)], without_affine.eps_yy[(row, col)]);
            assert!((w_yy as f64 - wo_yy as f64 - a_eyy).abs() < 1e-6 * a_eyy.abs().max(1.0),
                "eps_yy must shift by exactly a_eyy={a_eyy:e} at ({row},{col})");
            let (w_sxx, wo_sxx) = (with_affine.sigma_xx[(row, col)], without_affine.sigma_xx[(row, col)]);
            assert!(w_sxx != wo_sxx, "stress must differ once affine strain is included at ({row},{col})");
            checked += 1;
        }
        assert!(checked > 0, "no unmasked grid cells were compared - test fixture is broken");
    }

    /// Issue #77 PH4-45 fix proof (mirrors `probe_hole_boundary_profile_derived_reflects_hard_
    /// constraint_ansatz_near_hole_boundary`): `evaluate_user_vis_grid`'s heatmap must actually
    /// apply the passed `ansatz` (not just unit-scale it) - before this fix, the function did a
    /// bare `fwd_embedded` forward pass with no ansatz application at all, so switching
    /// `Identity` -> `HardConstraint` would change nothing.
    #[test]
    fn evaluate_user_vis_grid_reflects_hard_constraint_ansatz_near_hole_boundary() {
        let device = crate::training_core::BDevice::default();
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free };
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![hole] };
        let model = tiny_model(&geometry);
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let (px, py) = (6.9e7_f64, 0.0_f64);
        let u_ref = 1e-3_f32;

        let identity_ansatz = IdentityAnsatz;
        let hard_ansatz = crate::kirsch_hole_correction::AnnulusAnsatz::HardConstraint(
            crate::kirsch_hole_correction::HoleTractionFreeAnsatz {
                hole_center: geometry.holes[0].center, hole_radius: geometry.holes[0].radius,
                half_w: geometry.half_w, half_h: geometry.half_h,
                px, py, e: material.e as f64, nu: material.nu as f64, u_ref: u_ref as f64, saturation_scale: 1.0,
                trainable: false,
            },
        );
        let via_identity = evaluate_user_vis_grid(
            &model, &geometry, [24, 24], u_ref, px, &material, &fd, &[], &device, &identity_ansatz, None,
        );
        let via_hard_constraint = evaluate_user_vis_grid(
            &model, &geometry, [24, 24], u_ref, px, &material, &fd, &[], &device, &hard_ansatz, None,
        );

        let mut any_differs = false;
        for ((row, col), &u_id) in via_identity.disp_u.indexed_iter() {
            let u_hc = via_hard_constraint.disp_u[(row, col)];
            if u_id.is_nan() || u_hc.is_nan() { continue; }
            if (u_id - u_hc).abs() > 1e-9 { any_differs = true; }
        }
        assert!(any_differs, "hard-constraint ansatz must change disp_u somewhere on the grid - \
                 pre-fix bug would make every unmasked cell byte-identical regardless of ansatz");
    }

    /// Issue #77 PH4-45: `evaluate_annular_vis_grid` must actually splice - the annulus
    /// model's own field strictly inside `interface_radius`, the outer model's strictly
    /// outside - not silently return one model's field everywhere. Uses two models with
    /// different seeds (genuinely different raw output) so a real splice produces a real,
    /// decisive difference between the two regions; a broken splice (e.g. always returning the
    /// annulus model's field) would make `outer_only` region cells match `annulus_vis` instead
    /// of `outer_vis`, caught by the assertions below.
    #[test]
    fn evaluate_annular_vis_grid_splices_at_the_interface_radius_not_one_model_everywhere() {
        let device = crate::training_core::BDevice::default();
        let hole = HoleSpec { center: [0.0, 0.0], radius: 0.01, bc: HoleBc::Free };
        let geometry = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![hole] };
        let interface_radius = 0.03;
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let material = MaterialProps::al7075_t6();
        let px = 6.9e7_f64;

        use burn::tensor::backend::Backend;
        crate::training_core::B::seed(&device, 111);
        let annulus_model = crate::network::ElasticityNetConfig::new()
            .with_input_dim(geometry.coordinate_embedding().input_dim())
            .with_hidden_dim(8).with_n_hidden(2).with_output_dim(5).init(&device);
        crate::training_core::B::seed(&device, 222);
        let outer_model = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5).init(&device);

        let annulus_vis = evaluate_user_vis_grid(
            &annulus_model, &geometry, [24, 24], 1.0, 1.0, &material, &fd, &[], &device, &IdentityAnsatz, None,
        );
        let outer_vis = evaluate_user_vis_grid(
            &outer_model, &geometry, [24, 24], 1.0, 1.0, &material, &fd, &[], &device, &IdentityAnsatz, None,
        );
        let spliced = evaluate_annular_vis_grid(
            &annulus_model, &outer_model, &geometry, [24, 24], 1.0, px, &material, &fd, &[], &device,
            &IdentityAnsatz, &IdentityAnsatz, None, hole.center, interface_radius,
        );

        let mut checked_inside = 0;
        let mut checked_outside = 0;
        for ((row, col), &spliced_val) in spliced.disp_u.indexed_iter() {
            if spliced_val.is_nan() { continue; }
            let xn = -1.0 + 2.0 * col as f64 / 23.0;
            let yn = -1.0 + 2.0 * row as f64 / 23.0;
            let (xp, yp) = (xn * geometry.half_w, yn * geometry.half_h);
            let r = (xp * xp + yp * yp).sqrt();
            if r < interface_radius {
                assert_eq!(spliced_val, annulus_vis.disp_u[(row, col)], "inside interface_radius must equal the annulus model's own field at ({row},{col})");
                checked_inside += 1;
            } else {
                assert_eq!(spliced_val, outer_vis.disp_u[(row, col)], "outside interface_radius must equal the outer model's own field at ({row},{col})");
                checked_outside += 1;
            }
        }
        assert!(checked_inside > 0 && checked_outside > 0, "test grid must cover both sides of the interface radius: inside={checked_inside} outside={checked_outside}");
    }

    #[test]
    fn evaluate_user_vis_grid_masks_every_new_field_outside_the_domain_same_as_the_original_six() {
        let geometry = two_hole_geometry();
        let model = tiny_model(&geometry);
        let device = crate::training_core::BDevice::default();
        let fd = crate::fd_stencil::FdConfig::new(1e-3, 2.0 * geometry.half_w, 2.0 * geometry.half_h);
        let vis = evaluate_user_vis_grid(
            &model, &geometry, [16, 16], 1.0, 1.0, &MaterialProps::al7075_t6(), &fd, &[], &device,
            &IdentityAnsatz, None,
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
            &IdentityAnsatz, None,
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
            &IdentityAnsatz, None,
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
            &IdentityAnsatz, None,
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
            architecture: Default::default(),
        };
        let (rms, max) = probe_boundary_residuals(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let (rms, max) = probe_boundary_residuals(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let rf = probe_reaction_force(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let rf = probe_reaction_force(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let rf = probe_reaction_force(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let report = probe_load_transfer(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let report = probe_load_transfer(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
        };
        let report = probe_load_transfer(&model, &spec, &device, &crate::pinlug_problem::IdentityAnsatz, None);
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
        let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                config: &config, problem: &problem, fd: &fd, hole_fd: &fd, per_domain_lr: None, k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: 50.0,
                n_fourier: spec.geometry.n_fourier(),
                coordinate_embedding: spec.geometry.coordinate_embedding(),
                domain_coordinate_embeddings: None,
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
        let affine = decomposition_applicable(&spec).then_some((spec.load.px, spec.load.py));
        let vis = evaluate_user_vis_grid(
            &model_val, &spec.geometry, [96, 96], u_ref, spec.load.px, &spec.material, &fd, &diag_int_norm, &device,
            problem.ansatz(0), affine,
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
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
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
            architecture: Default::default(),
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
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
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
            architecture: Default::default(),
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
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                        config: &config, problem: &problem, fd: &fd, hole_fd: &fd, per_domain_lr: None, k: 1.0,
                        domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(),
                        domain_coordinate_embeddings: None, probe_term_gradients: false,
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
                                config: &config, problem: &problem, fd: &fd, hole_fd: &fd, per_domain_lr: None, k: 1.0,
                                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                                dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                                constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                                n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(),
                                domain_coordinate_embeddings: None, probe_term_gradients: false,
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

                let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
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
            architecture: Default::default(),
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
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                        config: &config, problem: &problem, fd: &fd, hole_fd: &fd, per_domain_lr: None, k: 1.0,
                        domains: vec![DomainStepCtx { data: &probe_data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(),
                        domain_coordinate_embeddings: None, probe_term_gradients: false,
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

                let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
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
            architecture: Default::default(),
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
            let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim(), hole_scale: make_gate_optim() };
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
                        config: &config, problem: &problem, fd: &fd, hole_fd: &fd, per_domain_lr: None, k: 1.0,
                        domains: vec![DomainStepCtx { data: &probe_data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(), coordinate_embedding: spec.geometry.coordinate_embedding(),
                        domain_coordinate_embeddings: None, probe_term_gradients: false,
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

                let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
                let ctx = plate_multi_step_ctx(
                    &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
        let vis = evaluate_user_vis_grid(
            &model, &spec.geometry, [64, 64], scales.u_ref, px, &spec.material, &fd, &[], &device,
            &IdentityAnsatz, None,
        );

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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
        let spec = ProblemSpec { geometry: UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] }, material: MaterialProps::al7075_t6(), load: LoadConfig::uniaxial_x(1e7), network: Default::default(), training: Default::default(), formulation: pinn_core::problem_spec::default_formulation(),
            architecture: Default::default(),
};
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
            architecture: Default::default(),
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
            architecture: Default::default(),
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
        // Issue #77 PH4-41: `run_annular_decomposition_training` always uses `Identity` (never
        // the hard-constraint ansatz), and this geometry is exactly the `decomposition_
        // applicable` case (one centered Free hole) - matches training's own real convention.
        let affine = decomposition_applicable(&hole).then_some((hole.load.px, hole.load.py));
        let profile = probe_hole_boundary_profile_derived(
            &annulus, &hole.geometry, &hole.geometry.holes[0], 144, &fd, scales.u_ref,
            scales.stress_ref, &hole.material, ring_anchor_margin_m(hole.training.fd_h, &hole.geometry), &device,
            &IdentityAnsatz, affine,
        );
        let kt = stress_concentration_from_profile(&profile, hole.load.px.abs()).kt;
        let error = (kt - FEM_KT).abs() / FEM_KT;
        println!("[L5 #77] Kt={kt:.9} FEM={FEM_KT:.9} error={:.3}%", 100.0 * error);
        assert!(kt.is_finite() && error <= 0.05,
            "#77 L5 failed: Kt={kt:.9}, FEM={FEM_KT:.9}, error={:.3}%", 100.0 * error);
    }
}
