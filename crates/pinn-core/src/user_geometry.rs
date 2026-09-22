//! Geometry representation for user-defined problems (the "design a joint from scratch"
//! ingestion path) — deliberately **independent** of [`crate::geometry::GeometryConfig`],
//! which hardcodes exactly one hole (`hole: HoleType`, singular) and is load-bearing for the
//! frozen Kirsch/pin-lug paths. Supporting N holes through the shared struct would risk
//! those paths for no benefit; this type exists so user-defined problems can have an
//! arbitrary hole count without touching `GeometryConfig` at all.
//!
//! A [`UserGeometry`]'s real geometry is read directly by
//! `pinn_solver::user_problem::UserSamplingStrategy` — it does NOT get encoded into a
//! `GeometryConfig`. The [`DomainSamplingStrategy`](crate::problem::DomainSamplingStrategy)
//! trait's methods take a `&GeometryConfig` parameter that concrete strategies are free to
//! ignore in favor of their own captured state (an established, precedented pattern — see
//! `FakeInterfaceSampling` in `pinn-core/src/problem.rs`'s own tests). [`to_placeholder`]
//! builds the inert `GeometryConfig` `DomainSpec.geometry` still requires as a field, sized
//! to match this geometry's real bounding box (so any code that innocently reads
//! `geom.x_range()`/`y_range()`/`half_w`/`half_h` for pure bounding-box purposes — e.g.
//! normalization — still gets a correct answer), with `hole: HoleType::None` (there is no
//! single hole to encode) and `symmetry: SymmetryMode::Full` (v1 user-defined problems don't
//! support symmetry reduction).

use serde::{Deserialize, Serialize};

use crate::geometry::{GeometryConfig, HoleType, SymmetryMode};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HoleBc {
    /// Traction-free hole boundary (natural/Neumann, zero prescribed traction).
    Free,
    /// Zero-displacement hole boundary (soft Dirichlet penalty — see
    /// `pinn_solver::user_problem`'s module doc for why this is soft, not a hard ansatz).
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HoleSpec {
    /// Hole center, physical coordinates [m], relative to the plate's own center.
    pub center: [f64; 2],
    /// Hole radius [m].
    pub radius: f64,
    pub bc: HoleBc,
}

/// Edge-referenced hole placement (TOML input layer only — see `UserGeometry`'s custom
/// `Deserialize` impl, which is the only consumer of this type). `HoleSpec.center` already
/// accepts arbitrary off-center coordinates; this exists purely so a TOML author can place a
/// hole by distance from a plate edge (the engineering-drawing convention) instead of computing
/// a raw `[x, y]` relative to the plate centroid by hand. Every field optional so `serde` can
/// deserialize whichever convention a given hole entry actually used; [`resolve_hole`]
/// validates that exactly one convention was used per hole and resolves it into a real
/// `HoleSpec` before `UserGeometry` is ever constructed — this type itself never appears
/// outside a TOML load, is never constructed directly by production code, and carries no
/// physics of its own.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
struct HoleSpecRaw {
    #[serde(default)]
    center: Option<[f64; 2]>,
    /// Distance [m] from the plate's left edge (`x = -half_w`) to the hole center.
    #[serde(default)]
    from_left: Option<f64>,
    /// Distance [m] from the plate's right edge (`x = +half_w`) to the hole center.
    #[serde(default)]
    from_right: Option<f64>,
    /// Distance [m] from the plate's top edge (`y = +half_h`) to the hole center.
    #[serde(default)]
    from_top: Option<f64>,
    /// Distance [m] from the plate's bottom edge (`y = -half_h`) to the hole center.
    #[serde(default)]
    from_bottom: Option<f64>,
    radius: f64,
    bc: HoleBc,
}

/// Resolves one [`HoleSpecRaw`] into a real [`HoleSpec`], given the plate's own `half_w`/
/// `half_h` (only known once the REST of `UserGeometry` has been deserialized — the reason this
/// resolution happens in `UserGeometry`'s own custom `Deserialize` impl, not in a per-field
/// `HoleSpec`/`HoleSpecRaw` deserializer, which has no access to sibling fields). Accepts
/// exactly one placement convention: the existing `center = [x, y]` with every `from_*` field
/// absent, OR exactly one X-axis reference (`from_left` XOR `from_right`) plus exactly one
/// Y-axis reference (`from_top` XOR `from_bottom`) with `center` absent. Every other
/// combination — `center` mixed with any `from_*`, two references on the same axis, zero
/// references on an axis when `center` is absent — is a clear, named error rather than a
/// silent fallback to `[0, 0]` or an arbitrarily-chosen reference, matching this codebase's
/// "loud, not silent" failure convention (e.g. `base_weight`'s own `panic!` on an unknown loss
/// term, `--problem-spec`'s own `anyhow::bail!` on a non-finite result).
fn resolve_hole(index: usize, raw: HoleSpecRaw, half_w: f64, half_h: f64) -> Result<HoleSpec, String> {
    let x_refs = raw.from_left.is_some() as u8 + raw.from_right.is_some() as u8;
    let y_refs = raw.from_top.is_some() as u8 + raw.from_bottom.is_some() as u8;

    let center = match (raw.center, x_refs, y_refs) {
        (Some(center), 0, 0) => center,
        (Some(_), _, _) => {
            return Err(format!(
                "geometry.holes[{index}]: specify EITHER `center = [x, y]` OR edge-reference \
                 fields (`from_left`/`from_right`/`from_top`/`from_bottom`), never both"
            ));
        }
        (None, 1, 1) => {
            let x = match (raw.from_left, raw.from_right) {
                (Some(fl), None) => -half_w + fl,
                (None, Some(fr)) => half_w - fr,
                _ => unreachable!("x_refs == 1 guarantees exactly one of from_left/from_right is Some"),
            };
            let y = match (raw.from_bottom, raw.from_top) {
                (Some(fb), None) => -half_h + fb,
                (None, Some(ft)) => half_h - ft,
                _ => unreachable!("y_refs == 1 guarantees exactly one of from_bottom/from_top is Some"),
            };
            [x, y]
        }
        (None, xr, yr) => {
            return Err(format!(
                "geometry.holes[{index}]: no `center` given, so exactly one X-axis reference \
                 (`from_left` XOR `from_right`) and exactly one Y-axis reference (`from_top` \
                 XOR `from_bottom`) are required — found {xr} X-axis and {yr} Y-axis reference(s)"
            ));
        }
    };
    Ok(HoleSpec { center, radius: raw.radius, bc: raw.bc })
}

/// First #77 annular-decomposition interface: three hole radii from the center. This retains
/// two radii of material beyond the free boundary while leaving a substantial outer domain for
/// the L5 small-hole geometry. It is geometry-owned, not duplicated in sampler/runner code.
pub const ANNULAR_INTERFACE_RADIUS_FACTOR: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnularPartition {
    pub center: [f64; 2],
    pub hole_radius: f64,
    pub interface_radius: f64,
}

impl AnnularPartition {
    pub fn contains_annulus(self, x: f64, y: f64) -> bool {
        let dx = x - self.center[0];
        let dy = y - self.center[1];
        let r2 = dx * dx + dy * dy;
        r2 >= self.hole_radius * self.hole_radius && r2 <= self.interface_radius * self.interface_radius
    }

    pub fn contains_outer(self, x: f64, y: f64) -> bool {
        let dx = x - self.center[0];
        let dy = y - self.center[1];
        dx * dx + dy * dy >= self.interface_radius * self.interface_radius
    }
}

/// Deterministic coordinate representation used by user-defined plate networks.
///
/// `SingleHoleChart` keeps raw normalized coordinates and appends dimensionless local-hole
/// chart and bounded-envelope features. `inv_radius` converts independently normalized x/y coordinates
/// back to radius units, preserving circular physical geometry on non-square plates.
///
/// Issue #78: no longer `Copy` - `MultiHoleChart`'s own `Vec<HoleChartParams>` field (N holes,
/// runtime-variable length) cannot be. Every pre-existing call site that relied on implicit
/// `Copy` now needs an explicit `.clone()` (cheap - a small `Vec` of two-`[f32;2]`-pair
/// structs, never on a per-step hot path) - the compiler enumerates every one of them
/// exhaustively, the same discipline this codebase's own doc comments describe for every prior
/// exhaustive-match migration (`HoleBcTerm::name()`'s `hole_bc_term_name`, `AnnulusAnsatz`'s
/// own `MultiHoleHardConstraint` addition, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateEmbedding {
    Raw,
    SingleHoleChart {
        center_norm: [f32; 2],
        inv_radius: [f32; 2],
        epsilon: f32,
        /// Issue #77 spectral-bias fix: multi-scale Fourier feature band count, applied to
        /// the hole-relative `(qx,qy)` coordinates already computed for the chart embedding
        /// (NOT raw x,y - encoding frequency content tuned to the hole's own radius avoids
        /// needing very high absolute frequencies to resolve a small hole in a large plate).
        /// `0` (every pre-existing call site of `UserGeometry::coordinate_embedding()`) is
        /// byte-identical to the pre-#77-spectral-bias-fix embedding - only
        /// `coordinate_embedding_with_fourier` ever sets this nonzero. See
        /// `network::chart_embed`'s doc comment for the exact feature formula and
        /// `PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-31 for why this exists (six other
        /// candidate mechanisms tested and ruled out or fixed without closing the Kt gap;
        /// this tests the spectral-bias hypothesis directly).
        n_fourier: usize,
    },
    /// Issue #77 Phase 3 architectural redesign: a true reparameterization of the network's own
    /// coordinate input to `(ξ, cosθ, sinθ)` — `ξ = ln(r/hole_radius)`, hole-relative log-polar —
    /// appended to the raw `(x,y,z)` prefix (mirrors `SingleHoleChart`'s own "raw prefix + derived
    /// features" convention exactly, see `network::chart_embed`'s doc comment). Unlike
    /// `SingleHoleChart`'s additive FEATURES on top of raw coordinates (already tried as Fourier
    /// features, PH4-31/32, and rejected), this asks the network to condition on a coordinate
    /// system whose natural symmetry actually matches the field near the hole — no near-hole
    /// singularity in the INPUT space itself (`r→a` is `ξ→-∞`, but training only ever samples a
    /// bounded, FD-safe range of `ξ`, same as every other embedding's own near-hole exclusion
    /// margin). `cosθ,sinθ` (not raw `θ`) avoids the angular wraparound discontinuity at `θ=±π`.
    /// FD stencils remain in PHYSICAL `(x,y)` space, unaffected — this embedding is transparent
    /// to the existing FD/derivative machinery exactly like `SingleHoleChart` already is (no
    /// stencil/differential-operator changes needed, confirmed in `network.rs`'s own doc comment
    /// for `log_polar_embed`).
    LogPolar {
        center_norm: [f32; 2],
        inv_radius: [f32; 2],
        epsilon: f32,
    },
    /// Issue #78: the N-hole generalization of `SingleHoleChart` - the same 7 hole-relative
    /// features (`r, log_r, c2, s2, psi, psi*c2, psi*s2` - see `network::chart_embed`'s own doc
    /// comment for the exact formula), computed independently for EVERY hole in this geometry
    /// (not just `Free` ones - the network needs geometric awareness of a `Fixed` hole's own
    /// local stress concentration too, via `interior_energy`/`equilibrium`/`hole_fixed`, none
    /// of which are Free-only) and concatenated after the raw 3-column prefix. `holes.len()==1`
    /// is deliberately NOT routed through this variant - `UserGeometry::coordinate_embedding`
    /// keeps returning the exact pre-existing `SingleHoleChart` for that case, so every single-
    /// hole spec's embedding, and every test/call site that pattern-matches `SingleHoleChart`
    /// directly, stays byte-identical. This variant exists purely to close the gap `SingleHole
    /// Chart`'s own doc comment already flagged ("Multi-hole enrichment is intentionally
    /// deferred until it has an unambiguous benchmark") - the benchmark is this session's own
    /// real finding: three independent hyperparameter experiments (collocation density,
    /// network capacity, learning rate) each reproduced an IDENTICAL trained Kt for a real
    /// multi-hole hard-constraint spec, ruling out optimization/data/capacity as the cause and
    /// pointing at a representation-level gap instead - the only accuracy-relevant machinery
    /// still gated to `holes.len()==1` after issue #78 Stage 2's own N-hole ansatz/affine work.
    /// `n_fourier` always `0` here (no per-hole-Fourier call site exists yet, matching
    /// `SingleHoleChart`'s own real, disclosed negative result for the single-hole case -
    /// PH4-31 found it made the interior residual ~28x WORSE - so this is deliberately NOT
    /// wired up for multi-hole either without new evidence).
    MultiHoleChart {
        holes: Vec<HoleChartParams>,
        epsilon: f32,
    },
}

/// Per-hole chart parameters for [`CoordinateEmbedding::MultiHoleChart`] - the same
/// `center_norm`/`inv_radius` pair [`CoordinateEmbedding::SingleHoleChart`] carries for its one
/// hole, one instance per hole in geometry order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoleChartParams {
    pub center_norm: [f32; 2],
    pub inv_radius: [f32; 2],
}

impl CoordinateEmbedding {
    pub fn input_dim(&self) -> usize {
        match self {
            Self::Raw => 3,
            // 10 base chart columns + 4 columns (sin/cos of qx, sin/cos of qy) per frequency
            // band - same `4*n_fourier` convention `network::fourier_embed` already uses.
            Self::SingleHoleChart { n_fourier, .. } => 10 + 4 * n_fourier,
            // 3 raw prefix columns + (ξ, cosθ, sinθ).
            Self::LogPolar { .. } => 6,
            // 3 raw prefix columns + 7 chart columns per hole (no Fourier - see this variant's
            // own doc comment).
            Self::MultiHoleChart { holes, .. } => 3 + 7 * holes.len(),
        }
    }
}

/// Issue #61 EPIC P2-06: identifies one boundary component of a [`UserGeometry`] - an outer
/// rectangle edge, or a specific hole by index. See [`UserGeometry::nearest_boundary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryRef {
    OuterLeft,
    OuterRight,
    OuterTop,
    OuterBottom,
    /// Index into [`UserGeometry::holes`].
    Hole(usize),
}

/// Issue #61 EPIC P2-06: per-direction validity of a 5-point central-difference FD stencil
/// centered at some point - see [`UserGeometry::valid_stencil`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilValidity {
    pub center_valid: bool,
    pub x_plus_valid: bool,
    pub x_minus_valid: bool,
    pub y_plus_valid: bool,
    pub y_minus_valid: bool,
}

impl StencilValidity {
    /// True iff the center AND all 4 shifted neighbors are valid - a real, usable FD stencil.
    pub fn all_valid(&self) -> bool {
        self.center_valid && self.x_plus_valid && self.x_minus_valid && self.y_plus_valid && self.y_minus_valid
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UserGeometry {
    /// Plate half-width in x [m] — full width = 2*half_w.
    pub half_w: f64,
    /// Plate half-height in y [m] — full height = 2*half_h.
    pub half_h: f64,
    /// Thickness [m] (for plane-stress scaling).
    pub thickness: f64,
    pub holes: Vec<HoleSpec>,
}

/// Deliberately hand-written, not derived — see [`HoleSpecRaw`]/[`resolve_hole`]'s own doc
/// comments. Every hole's `[x, y]` center may be given directly OR via edge-referenced
/// distances; resolving the latter needs `half_w`/`half_h`, which are only available once
/// deserialized here alongside `holes`, not inside a per-hole `Deserialize` impl. `Serialize`
/// stays derived and unaffected — round-tripping a parsed spec always re-emits `center =
/// [x, y]`, never edge references (this is a convenience input format, not a storage format).
impl<'de> Deserialize<'de> for UserGeometry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct UserGeometryRaw {
            half_w: f64,
            half_h: f64,
            thickness: f64,
            #[serde(default)]
            holes: Vec<HoleSpecRaw>,
        }

        let raw = UserGeometryRaw::deserialize(deserializer)?;
        let holes = raw
            .holes
            .into_iter()
            .enumerate()
            .map(|(i, h)| resolve_hole(i, h, raw.half_w, raw.half_h))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        Ok(UserGeometry { half_w: raw.half_w, half_h: raw.half_h, thickness: raw.thickness, holes })
    }
}

impl UserGeometry {
    /// Issue #78 Stage 1.2: rejects a geometrically-nonsensical spec before training ever
    /// starts - a hole (partly or wholly) outside the plate, a non-positive radius, or two
    /// holes overlapping. Returns the first violation found (not every violation collected) -
    /// matches this codebase's own established "loud, not silent" convention (e.g.
    /// `--problem-spec`'s own `anyhow::bail!` on a non-finite result, `UserDefinedProblem::
    /// base_weight`'s `panic!` on an unrecognized term name) of failing fast with a clear
    /// message rather than continuing with silently-wrong geometry.
    ///
    /// Deliberately a pure GEOMETRIC check, not an FD-safety-margin check: `pinn-core` has no
    /// dependency on `pinn-solver` (an established architectural rule - see this crate's own
    /// module docs), where the real per-run FD-stencil-safety margin formula
    /// (`ring_anchor_margin_m`, a function of `training.fd_h` and this geometry's own
    /// dimensions) actually lives. This catches the unambiguous case - literal overlap or
    /// out-of-bounds - not "too close for a numerically safe FD stencil at this run's specific
    /// `fd_h`," which stays an open, narrower gap (see issue #78's own tracked doc).
    pub fn validate(&self) -> Result<(), String> {
        for (i, hole) in self.holes.iter().enumerate() {
            if !(hole.radius > 0.0) {
                return Err(format!("geometry.holes[{i}]: radius must be positive, got {}", hole.radius));
            }
            let (cx, cy) = (hole.center[0], hole.center[1]);
            if cx - hole.radius < -self.half_w
                || cx + hole.radius > self.half_w
                || cy - hole.radius < -self.half_h
                || cy + hole.radius > self.half_h
            {
                return Err(format!(
                    "geometry.holes[{i}]: hole (center=[{cx}, {cy}], radius={}) extends outside \
                     the plate (half_w={}, half_h={})",
                    hole.radius, self.half_w, self.half_h
                ));
            }
        }
        for i in 0..self.holes.len() {
            for j in (i + 1)..self.holes.len() {
                let (a, b) = (&self.holes[i], &self.holes[j]);
                let dx = a.center[0] - b.center[0];
                let dy = a.center[1] - b.center[1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < a.radius + b.radius {
                    return Err(format!(
                        "geometry.holes[{i}] and geometry.holes[{j}] overlap: center distance \
                         {dist} is less than the sum of their radii ({} + {} = {})",
                        a.radius, b.radius, a.radius + b.radius
                    ));
                }
            }
        }
        Ok(())
    }

    /// #77's first decomposition supports exactly one circular hole. Multi-hole partition
    /// ownership is deliberately deferred rather than silently assigning overlap regions.
    pub fn annular_partition(&self) -> Option<AnnularPartition> {
        let [hole] = self.holes.as_slice() else { return None };
        let interface_radius = ANNULAR_INTERFACE_RADIUS_FACTOR * hole.radius;
        if interface_radius >= self.half_w.min(self.half_h) {
            return None;
        }
        Some(AnnularPartition { center: hole.center, hole_radius: hole.radius, interface_radius })
    }

    /// Issue #78 item 4: the N-hole generalization `annular_partition`'s own doc comment
    /// flagged as deliberately deferred - one `AnnularPartition` per `HoleBc::Free` hole
    /// (`Fixed` holes get no annulus domain at all; they stay in the outer domain via the
    /// existing soft-penalty `hole_fixed` term, unaffected by this). `annular_partition`'s own
    /// per-hole math (`interface_radius = 3·hole.radius`, centered on that hole) was ALREADY
    /// hole-local, not plate-wide - this function just calls it once per Free hole and adds
    /// the ONE thing genuinely missing for N>1: validating no two interface circles overlap
    /// each other (a real, previously-nonexistent check - `UserGeometry::validate()` only ever
    /// checked the HOLES themselves don't overlap, not their much-larger interface circles).
    /// Returns `None` (loud, not silent) if ANY interface circle doesn't fit inside the plate
    /// OR any two interface circles overlap - a caller must not silently fall back to a subset
    /// of holes or an arbitrary ownership assignment for the overlapping region.
    pub fn annular_partitions(&self) -> Option<Vec<AnnularPartition>> {
        let free_holes: Vec<&HoleSpec> = self.holes.iter().filter(|h| h.bc == HoleBc::Free).collect();
        if free_holes.is_empty() {
            return None;
        }
        let mut partitions = Vec::with_capacity(free_holes.len());
        for hole in &free_holes {
            let interface_radius = ANNULAR_INTERFACE_RADIUS_FACTOR * hole.radius;
            if interface_radius >= self.half_w.min(self.half_h) {
                return None;
            }
            partitions.push(AnnularPartition { center: hole.center, hole_radius: hole.radius, interface_radius });
        }
        for i in 0..partitions.len() {
            for j in (i + 1)..partitions.len() {
                let (a, b) = (partitions[i], partitions[j]);
                let dx = a.center[0] - b.center[0];
                let dy = a.center[1] - b.center[1];
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < a.interface_radius + b.interface_radius {
                    return None;
                }
            }
        }
        Some(partitions)
    }
    /// True if `(x, y)` is inside the plate's rectangular bound and outside every hole.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        if x < -self.half_w || x > self.half_w || y < -self.half_h || y > self.half_h {
            return false;
        }
        for hole in &self.holes {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            if dx * dx + dy * dy < hole.radius * hole.radius {
                return false;
            }
        }
        true
    }

    /// Returns a clone with every hole's radius inflated by `margin_m` — used ONLY to build
    /// an `AdaptiveGrid<UserGeometry>`'s own containment gate (via `amr::AmrDomain::contains`)
    /// for collocation purposes, NEVER for physics (hole boundary-condition terms), display,
    /// or the real `contains`/masking semantics above. AMR's leaf-cell-center containment
    /// check has no margin of its own, and quadtree cells aren't boundary-aligned, so without
    /// this a hole-zone-refined cell's center can legitimately satisfy `contains()` (r >
    /// radius) while still being close enough to the true edge that an FD stencil there
    /// crosses back inside the hole — silently corrupting the interior-energy/constitutive-
    /// consistency signal exactly where AMR concentrates the most collocation density. See
    /// `powershell_tool/CLAUDE.md`'s Kt investigation for the real, measured margin-vs-cell-
    /// size comparison that confirmed this as a genuine (not merely theoretical) gap.
    pub fn inflated_for_collocation(&self, margin_m: f64) -> Self {
        let mut inflated = self.clone();
        for hole in &mut inflated.holes {
            hole.radius += margin_m;
        }
        inflated
    }

    /// Positional-Fourier-feature count the network's input should use for this geometry.
    /// Currently always `0` (raw x,y,z, no embedding) - see below for why, despite a real
    /// attempt to enable it.
    ///
    /// This mechanism exists (and every consumer downstream of it is real, generalized
    /// infrastructure, not dead code) because of a genuine, tested hypothesis: Kirsch's own
    /// problem uses positional-Fourier embedding specifically for its hole
    /// (`pinn_solver::engine::EngineParams::analyze`: `let n_fourier = if has_hole { 8_usize }
    /// else { 0 };`, commented "corrects spectral bias near hole") and achieves real Kt
    /// convergence; the generalized N-hole path never had it
    /// (`training_core::compute_domain_forwards` hardcoded `n_fourier = 0` unconditionally,
    /// with a comment explaining only why it skips the *hard Dirichlet ansatz* Fourier
    /// embedding was originally paired with there - not a deliberate decision to omit it for
    /// holed geometries). After ruling out weighting (capping `dynamic_lam_h_cap`, matching
    /// `constitutive_consistency_weight` to it) and sampling density (a 2.47x guaranteed
    /// hole-zone collocation increase) as the cause of a plate-with-hole run's Kt staying ~0,
    /// porting Kirsch's own Fourier fix was the next well-motivated, evidence-driven step.
    ///
    /// Measured result, not assumed: enabling `n_fourier = 8` for a holed geometry made the
    /// interior PDE (constitutive-consistency) residual RMS **~28x WORSE** (2.80e7 Pa vs.
    /// 9.85e5 Pa without it, same 8000-step config) while total_loss converged FASTER and Kt
    /// stayed at ~0 either way. Interpretation: the higher-frequency basis let the network fit
    /// the boundary/traction collocation points more precisely while oscillating wildly
    /// between them - a well-known Fourier-feature pitfall when embedding frequency outstrips
    /// collocation density, and a straightforward port of Kirsch's own working fix without
    /// re-deriving whether its frequency/density balance holds for a different point-sampling
    /// scheme. A real negative result, kept disabled (not deleted) so a future attempt (e.g.
    /// paired with denser boundary-adjacent sampling, or a lower `n_fourier`) doesn't have to
    /// re-build this plumbing from scratch or re-discover this pitfall blind.
    pub fn n_fourier(&self) -> usize {
        let _ = &self.holes; // kept as a parameter for when this is revisited - see doc comment
        0
    }

    /// Network input dimension implied by [`Self::n_fourier`] — `3` (raw x,y,z) when there's
    /// no Fourier embedding, `4 * n_fourier` when there is. Mirrors `pinn_solver::engine::
    /// EngineParams::net_input_dim`'s identical formula.
    pub fn net_input_dim(&self) -> usize {
        self.coordinate_embedding().input_dim()
    }

    /// Geometry-owned input representation. One hole gets a local radial/angular chart with a
    /// bounded far-field envelope (`SingleHoleChart`); N>1 holes get the same per-hole chart
    /// features, one set per hole, concatenated (`MultiHoleChart` - issue #78, see that
    /// variant's own doc comment for why the earlier "multi-hole enrichment is intentionally
    /// deferred until it has an unambiguous benchmark" deferral is now resolved); no-hole
    /// problems remain a raw-coordinate model.
    pub fn coordinate_embedding(&self) -> CoordinateEmbedding {
        self.coordinate_embedding_with_fourier(0)
    }

    /// Issue #77 spectral-bias fix: same as [`Self::coordinate_embedding`], but with
    /// `n_fourier` set on the resulting `SingleHoleChart` (a no-op — byte-identical to
    /// `coordinate_embedding()` — for a no-hole/multi-hole geometry, which always returns
    /// `Raw` regardless of this argument). See `CoordinateEmbedding::SingleHoleChart::
    /// n_fourier`'s own doc comment.
    ///
    /// Issue #78: `n_fourier` only ever applies to the exact single-hole case
    /// (`SingleHoleChart`) - `MultiHoleChart` (N>1 holes) never carries a Fourier band count
    /// (see that variant's own doc comment for why: PH4-31's real negative result for the
    /// single-hole case, no new evidence justifying it for N holes either).
    pub fn coordinate_embedding_with_fourier(&self, n_fourier: usize) -> CoordinateEmbedding {
        match self.holes.as_slice() {
            [] => CoordinateEmbedding::Raw,
            [hole] => {
                let radius = hole.radius as f32;
                CoordinateEmbedding::SingleHoleChart {
                    center_norm: [
                        (hole.center[0] / self.half_w) as f32,
                        (hole.center[1] / self.half_h) as f32,
                    ],
                    inv_radius: [(self.half_w as f32) / radius, (self.half_h as f32) / radius],
                    // Only protection at r=0. Valid plate points are outside r=a, so this
                    // cannot alter physical features at collocation or FD-stencil points.
                    epsilon: 1e-4,
                    n_fourier,
                }
            }
            holes => CoordinateEmbedding::MultiHoleChart {
                holes: holes.iter().map(|hole| {
                    let radius = hole.radius as f32;
                    HoleChartParams {
                        center_norm: [
                            (hole.center[0] / self.half_w) as f32,
                            (hole.center[1] / self.half_h) as f32,
                        ],
                        inv_radius: [(self.half_w as f32) / radius, (self.half_h as f32) / radius],
                    }
                }).collect(),
                epsilon: 1e-4,
            },
        }
    }

    /// Issue #77 Phase 3 architectural redesign: the log-polar embedding, opt-in (never the
    /// result of [`Self::coordinate_embedding`]/[`Self::coordinate_embedding_with_fourier`],
    /// which stay `SingleHoleChart`/`Raw` unchanged). Same no-hole/multi-hole fallback to `Raw`
    /// as [`Self::coordinate_embedding_with_fourier`] — this representation only makes sense
    /// relative to exactly one hole's own center/radius.
    pub fn log_polar_embedding(&self) -> CoordinateEmbedding {
        let [hole] = self.holes.as_slice() else {
            return CoordinateEmbedding::Raw;
        };
        let radius = hole.radius as f32;
        CoordinateEmbedding::LogPolar {
            center_norm: [
                (hole.center[0] / self.half_w) as f32,
                (hole.center[1] / self.half_h) as f32,
            ],
            inv_radius: [(self.half_w as f32) / radius, (self.half_h as f32) / radius],
            epsilon: 1e-4,
        }
    }

    /// Issue #61 EPIC P2-07: true iff NO hole is [`HoleBc::Fixed`] - the plate's only essential/
    /// Dirichlet mechanism. Without ANY Dirichlet condition anywhere (this codebase's real
    /// `no_hole_plate.toml`, and `single_hole_plate.toml` with its hole set to `HoleBc::Free`,
    /// are BOTH this case), the elasticity BVP is only determined up to an additive rigid-body
    /// motion - translation is a genuine mathematical nullspace: the network could add ANY
    /// constant `(u0, v0)` offset to its output with zero effect on strain energy, traction
    /// residuals, or equilibrium, since none of those quantities depend on absolute position.
    /// See `pinn_solver::gauge`'s module doc comment for the gauge-fixing mechanism this drives.
    pub fn is_pure_neumann(&self) -> bool {
        !self.holes.iter().any(|h| h.bc == HoleBc::Fixed)
    }

    /// Issue #61 EPIC P2-06: signed distance to the domain boundary (positive = inside the
    /// valid domain - inside the outer rectangle AND outside every hole; negative = outside).
    /// `min(rect_sdf, hole_sdfs...)` is an approximate (not exact) SDF for a rectangle-minus-
    /// circles domain - exact everywhere except where a hole boundary and the outer rectangle
    /// boundary are close enough that their influence regions overlap (this codebase's real
    /// hole radii are always small relative to the plate, so this never matters in practice,
    /// but is not claimed exact in general). Generic over any number of holes - not hardcoded
    /// to one, unlike `crate::geometry::GeometryConfig`'s single `HoleType`.
    pub fn signed_distance(&self, x: f64, y: f64) -> f64 {
        let rect_sdf = (self.half_w - x.abs()).min(self.half_h - y.abs());
        self.holes.iter().fold(rect_sdf, |sdf, hole| {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            sdf.min((dx * dx + dy * dy).sqrt() - hole.radius)
        })
    }

    /// Which boundary component (an outer edge or a specific hole) is closest to `(x, y)` -
    /// the generic identifier [`Self::boundary_normal`]/[`Self::boundary_tangent`]/
    /// [`Self::boundary_measure`] key off, instead of each re-deriving "which edge" ad hoc.
    pub fn nearest_boundary(&self, x: f64, y: f64) -> BoundaryRef {
        let mut best = BoundaryRef::OuterLeft;
        let mut best_dist = (x + self.half_w).abs();
        for (candidate, dist) in [
            (BoundaryRef::OuterRight, (self.half_w - x).abs()),
            (BoundaryRef::OuterTop, (self.half_h - y).abs()),
            (BoundaryRef::OuterBottom, (y + self.half_h).abs()),
        ] {
            if dist < best_dist {
                best_dist = dist;
                best = candidate;
            }
        }
        for (i, hole) in self.holes.iter().enumerate() {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            let dist = ((dx * dx + dy * dy).sqrt() - hole.radius).abs();
            if dist < best_dist {
                best_dist = dist;
                best = BoundaryRef::Hole(i);
            }
        }
        best
    }

    /// Outward unit normal for the given boundary component, evaluated at `(x, y)` - constant
    /// per outer edge (axis-aligned), radial (from the hole's own center through `(x,y)`) for
    /// a hole. `(x, y)` need not lie exactly ON the boundary (the radial direction is still
    /// well-defined for any point other than a hole's exact center).
    pub fn boundary_normal_for(&self, boundary: BoundaryRef, x: f64, y: f64) -> (f64, f64) {
        match boundary {
            BoundaryRef::OuterLeft => (-1.0, 0.0),
            BoundaryRef::OuterRight => (1.0, 0.0),
            BoundaryRef::OuterTop => (0.0, 1.0),
            BoundaryRef::OuterBottom => (0.0, -1.0),
            BoundaryRef::Hole(i) => {
                let hole = &self.holes[i];
                let dx = x - hole.center[0];
                let dy = y - hole.center[1];
                let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                (dx / len, dy / len)
            }
        }
    }

    /// [`Self::boundary_normal_for`] at whichever boundary [`Self::nearest_boundary`] finds
    /// closest to `(x, y)` - the common case of "what's the normal AT this point".
    pub fn boundary_normal(&self, x: f64, y: f64) -> (f64, f64) {
        self.boundary_normal_for(self.nearest_boundary(x, y), x, y)
    }

    /// Unit tangent (90° counter-clockwise rotation of the outward normal) at `(x, y)`'s
    /// nearest boundary.
    pub fn boundary_tangent(&self, x: f64, y: f64) -> (f64, f64) {
        let (nx, ny) = self.boundary_normal(x, y);
        (-ny, nx)
    }

    /// Total arc length / edge length of the given boundary component (m) - the real
    /// geometric measure [`crate`]-level `BoundaryIntegral`-style consumers (see
    /// `pinn_solver::measure_integral`) need for a specific component, generalizing
    /// `pinn_solver::measure_integral::plate_outer_perimeter`'s outer-only formula to also
    /// cover individual holes.
    pub fn boundary_measure(&self, boundary: BoundaryRef) -> f64 {
        match boundary {
            BoundaryRef::OuterLeft | BoundaryRef::OuterRight => 2.0 * self.half_h,
            BoundaryRef::OuterTop | BoundaryRef::OuterBottom => 2.0 * self.half_w,
            BoundaryRef::Hole(i) => 2.0 * std::f64::consts::PI * self.holes[i].radius,
        }
    }

    /// Whether a 5-point central-difference FD stencil centered at `(x, y)` with half-steps
    /// `(hx, hy)` stays entirely within the valid domain (issue #61 EPIC P2-06's own "stencils
    /// avoid invalid points with recorded fallback/quality diagnostics" - this is the
    /// diagnostic; [`StencilValidity::all_valid`] is the yes/no answer, the per-direction
    /// fields are the "which neighbor(s) failed" detail a fallback strategy would need).
    /// Generalizes the ad hoc margin-based checks this codebase's real sampling strategies
    /// already perform (e.g. `pinn_solver::user_problem::UserSamplingStrategy::contains_for_
    /// collocation`) into a declared, reusable, geometry-level primitive.
    pub fn valid_stencil(&self, x: f64, y: f64, hx: f64, hy: f64) -> StencilValidity {
        StencilValidity {
            center_valid: self.contains(x, y),
            x_plus_valid: self.contains(x + hx, y),
            x_minus_valid: self.contains(x - hx, y),
            y_plus_valid: self.contains(x, y + hy),
            y_minus_valid: self.contains(x, y - hy),
        }
    }

    /// Inert placeholder `GeometryConfig` sized to this geometry's real bounding box — see
    /// the module doc comment for why this is safe and what it's actually used for.
    pub fn to_placeholder(&self) -> GeometryConfig {
        GeometryConfig {
            half_w: self.half_w,
            half_h: self.half_h,
            thickness: self.thickness,
            hole: HoleType::None,
            symmetry: SymmetryMode::Full,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_hole_geometry() -> UserGeometry {
        UserGeometry {
            half_w: 1.0,
            half_h: 1.0,
            thickness: 0.1,
            holes: vec![
                HoleSpec { center: [-0.5, 0.0], radius: 0.1, bc: HoleBc::Free },
                HoleSpec { center: [0.5, 0.0], radius: 0.1, bc: HoleBc::Fixed },
            ],
        }
    }

    /// Stage 0 (issue #78): the existing `center = [x, y]` convention must round-trip through
    /// `UserGeometry`'s new custom `Deserialize` impl byte-identically to before this change.
    #[test]
    fn deserialize_center_convention_is_unchanged() {
        let toml_str = r#"
            half_w = 0.1
            half_h = 0.2
            thickness = 0.005

            [[holes]]
            center = [0.03, -0.02]
            radius = 0.01
            bc = "Free"
        "#;
        let geom: UserGeometry = toml::from_str(toml_str).expect("existing center convention must still parse");
        assert_eq!(geom.holes, vec![HoleSpec { center: [0.03, -0.02], radius: 0.01, bc: HoleBc::Free }]);
    }

    /// Stage 0 (issue #78): each of the 4 valid edge-reference combinations resolves to the
    /// exact closed-form coordinate — not just "parses without error".
    #[test]
    fn deserialize_resolves_every_valid_edge_reference_combination() {
        // half_w=0.10, half_h=0.05: x in [-0.10, 0.10], y in [-0.05, 0.05].
        let cases: [(&str, [f64; 2]); 4] = [
            ("from_left = 0.03\nfrom_bottom = 0.01", [-0.10 + 0.03, -0.05 + 0.01]),
            ("from_left = 0.03\nfrom_top = 0.01", [-0.10 + 0.03, 0.05 - 0.01]),
            ("from_right = 0.03\nfrom_bottom = 0.01", [0.10 - 0.03, -0.05 + 0.01]),
            ("from_right = 0.03\nfrom_top = 0.01", [0.10 - 0.03, 0.05 - 0.01]),
        ];
        for (refs, expected_center) in cases {
            let toml_str = format!(
                "half_w = 0.10\nhalf_h = 0.05\nthickness = 0.005\n\n[[holes]]\n{refs}\nradius = 0.005\nbc = \"Free\"\n"
            );
            let geom: UserGeometry = toml::from_str(&toml_str)
                .unwrap_or_else(|e| panic!("valid edge-reference combination must parse ({refs:?}): {e}"));
            assert_eq!(geom.holes.len(), 1);
            let center = geom.holes[0].center;
            assert!(
                (center[0] - expected_center[0]).abs() < 1e-12 && (center[1] - expected_center[1]).abs() < 1e-12,
                "combination {refs:?}: expected center {expected_center:?}, got {center:?}"
            );
        }
    }

    /// Stage 0 (issue #78): every invalid combination must produce a clear parse error, never
    /// a silent fallback to `[0, 0]` or an arbitrarily-picked reference.
    #[test]
    fn deserialize_rejects_every_invalid_hole_placement_combination() {
        let base = |holes_body: &str| format!(
            "half_w = 0.10\nhalf_h = 0.05\nthickness = 0.005\n\n[[holes]]\n{holes_body}\nradius = 0.005\nbc = \"Free\"\n"
        );
        let invalid = [
            // center mixed with a reference field.
            "center = [0.0, 0.0]\nfrom_left = 0.01",
            // two references on the same axis (X).
            "from_left = 0.01\nfrom_right = 0.01\nfrom_bottom = 0.01",
            // two references on the same axis (Y).
            "from_left = 0.01\nfrom_top = 0.01\nfrom_bottom = 0.01",
            // only an X-axis reference, no Y-axis reference, no center.
            "from_left = 0.01",
            // no center and no reference fields at all.
            "",
        ];
        for body in invalid {
            let toml_str = base(body);
            let result: Result<UserGeometry, _> = toml::from_str(&toml_str);
            assert!(result.is_err(), "expected a parse error for hole body {body:?}, got {result:?}");
        }
    }

    /// Stage 0 (issue #78): a 3-hole spec mixing all three placement styles (explicit center,
    /// and two different edge-reference combinations) resolves every hole correctly and
    /// independently — the resolution is per-hole, not spec-wide.
    #[test]
    fn deserialize_resolves_mixed_placement_styles_across_multiple_holes() {
        let toml_str = r#"
            half_w = 0.10
            half_h = 0.05
            thickness = 0.005

            [[holes]]
            center = [0.0, 0.0]
            radius = 0.005
            bc = "Free"

            [[holes]]
            from_left = 0.02
            from_bottom = 0.01
            radius = 0.004
            bc = "Fixed"

            [[holes]]
            from_right = 0.02
            from_top = 0.01
            radius = 0.004
            bc = "Free"
        "#;
        let geom: UserGeometry = toml::from_str(toml_str).expect("mixed placement styles must parse");
        assert_eq!(geom.holes.len(), 3);
        assert_eq!(geom.holes[0].center, [0.0, 0.0]);
        assert!((geom.holes[1].center[0] - (-0.10 + 0.02)).abs() < 1e-12);
        assert!((geom.holes[1].center[1] - (-0.05 + 0.01)).abs() < 1e-12);
        assert!((geom.holes[2].center[0] - (0.10 - 0.02)).abs() < 1e-12);
        assert!((geom.holes[2].center[1] - (0.05 - 0.01)).abs() < 1e-12);
    }

    /// Issue #78 Stage 1.2: `validate()` accepts every real shipped multi-hole example's own
    /// geometry (real, well-separated, in-bounds holes).
    #[test]
    fn validate_accepts_a_sane_multi_hole_geometry() {
        assert!(two_hole_geometry().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_hole_extending_past_the_left_edge() {
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![HoleSpec { center: [-0.095, 0.0], radius: 0.01, bc: HoleBc::Free }],
        };
        assert!(geom.validate().is_err());
    }

    #[test]
    fn validate_rejects_a_hole_extending_past_every_edge_direction() {
        let cases = [
            [-0.095, 0.0],  // past left
            [0.095, 0.0],   // past right
            [0.0, 0.095],   // past top
            [0.0, -0.095],  // past bottom
        ];
        for center in cases {
            let geom = UserGeometry {
                half_w: 0.1, half_h: 0.1, thickness: 0.005,
                holes: vec![HoleSpec { center, radius: 0.01, bc: HoleBc::Free }],
            };
            assert!(geom.validate().is_err(), "expected rejection for hole at {center:?}");
        }
    }

    #[test]
    fn validate_rejects_a_non_positive_radius() {
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.0, bc: HoleBc::Free }],
        };
        assert!(geom.validate().is_err());
    }

    #[test]
    fn validate_rejects_two_overlapping_holes() {
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![
                HoleSpec { center: [-0.01, 0.0], radius: 0.02, bc: HoleBc::Free },
                HoleSpec { center: [0.01, 0.0], radius: 0.02, bc: HoleBc::Fixed },
            ],
        };
        assert!(geom.validate().is_err());
    }

    #[test]
    fn validate_accepts_two_holes_exactly_tangent() {
        // Distance between centers == sum of radii exactly - touching, not overlapping.
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![
                HoleSpec { center: [-0.01, 0.0], radius: 0.01, bc: HoleBc::Free },
                HoleSpec { center: [0.01, 0.0], radius: 0.01, bc: HoleBc::Fixed },
            ],
        };
        assert!(geom.validate().is_ok());
    }

    #[test]
    fn validate_reports_the_first_violation_for_a_three_hole_spec_with_one_bad_hole() {
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.1, thickness: 0.005,
            holes: vec![
                HoleSpec { center: [-0.05, 0.0], radius: 0.01, bc: HoleBc::Free },
                HoleSpec { center: [0.095, 0.0], radius: 0.01, bc: HoleBc::Fixed }, // out of bounds
                HoleSpec { center: [0.0, 0.05], radius: 0.01, bc: HoleBc::Free },
            ],
        };
        let err = geom.validate().expect_err("hole 1 is out of bounds");
        assert!(err.contains("holes[1]"), "error should identify the offending hole: {err}");
    }

    #[test]
    fn contains_rejects_points_outside_the_rectangle() {
        let geom = two_hole_geometry();
        assert!(!geom.contains(1.5, 0.0));
        assert!(!geom.contains(0.0, -1.5));
    }

    #[test]
    fn contains_rejects_points_inside_either_hole() {
        let geom = two_hole_geometry();
        assert!(!geom.contains(-0.5, 0.0)); // center of first hole
        assert!(!geom.contains(0.55, 0.02)); // inside second hole, off-center
    }

    #[test]
    fn contains_accepts_a_point_in_the_plate_between_holes() {
        let geom = two_hole_geometry();
        assert!(geom.contains(0.0, 0.0));
    }

    #[test]
    fn is_pure_neumann_true_with_no_holes_or_all_free_holes() {
        let no_holes = UserGeometry { half_w: 1.0, half_h: 1.0, thickness: 0.1, holes: vec![] };
        assert!(no_holes.is_pure_neumann());

        let mut all_free = two_hole_geometry();
        all_free.holes[1].bc = HoleBc::Free; // two_hole_geometry's 2nd hole is Fixed by default
        assert!(all_free.is_pure_neumann());
    }

    #[test]
    fn is_pure_neumann_false_when_any_hole_is_fixed() {
        // two_hole_geometry's 2nd hole is HoleBc::Fixed by construction.
        let geom = two_hole_geometry();
        assert!(!geom.is_pure_neumann());
    }

    #[test]
    fn coordinate_embedding_preserves_raw_no_hole_and_generalizes_multi_hole_inputs() {
        let no_holes = UserGeometry { half_w: 1.0, half_h: 1.0, thickness: 0.1, holes: vec![] };
        assert_eq!(no_holes.coordinate_embedding(), CoordinateEmbedding::Raw);
        assert_eq!(no_holes.net_input_dim(), 3);
        // Issue #78: multi-hole geometries used to fall back to `Raw` ("multi-hole enrichment
        // is intentionally deferred until it has an unambiguous benchmark" - `SingleHoleChart`'s
        // own now-resolved doc comment). They now get `MultiHoleChart`: the same 7 hole-
        // relative features `SingleHoleChart` computes for one hole, computed independently for
        // EACH hole and concatenated - `3 + 7*2 = 17` for this 2-hole fixture.
        let multi = two_hole_geometry();
        assert_eq!(multi.coordinate_embedding(), CoordinateEmbedding::MultiHoleChart {
            holes: vec![
                HoleChartParams { center_norm: [-0.5, 0.0], inv_radius: [10.0, 10.0] },
                HoleChartParams { center_norm: [0.5, 0.0], inv_radius: [10.0, 10.0] },
            ],
            epsilon: 1e-4,
        });
        assert_eq!(multi.net_input_dim(), 17);
    }

    #[test]
    fn coordinate_embedding_normalizes_single_hole_in_physical_radius_units() {
        let geometry = UserGeometry {
            half_w: 2.0,
            half_h: 1.0,
            thickness: 0.1,
            holes: vec![HoleSpec { center: [0.5, -0.25], radius: 0.1, bc: HoleBc::Free }],
        };
        assert_eq!(geometry.net_input_dim(), 10);
        assert_eq!(geometry.coordinate_embedding(), CoordinateEmbedding::SingleHoleChart {
            center_norm: [0.25, -0.25],
            inv_radius: [20.0, 10.0],
            epsilon: 1e-4,
            n_fourier: 0,
        });
    }

    #[test]
    fn annular_partition_is_disjoint_and_covers_the_exterior_of_one_hole() {
        let geometry = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 0.1,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.1, bc: HoleBc::Free }],
        };
        let partition = geometry.annular_partition().expect("small single hole must partition");
        assert!((partition.interface_radius - 0.3).abs() < 1e-12);
        assert!(partition.contains_annulus(0.2, 0.0));
        assert!(!partition.contains_outer(0.2, 0.0));
        assert!(partition.contains_outer(0.5, 0.0));
        assert!(!partition.contains_annulus(0.5, 0.0));
        assert!(!partition.contains_annulus(0.0, 0.0));
    }

    /// Issue #78 item 4: N=1 reduction - `annular_partitions` on a single-Free-hole geometry
    /// must produce the SAME single `AnnularPartition` `annular_partition` (singular) does.
    #[test]
    fn annular_partitions_reduces_to_annular_partition_at_n_equals_one() {
        let geometry = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 0.1,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.1, bc: HoleBc::Free }],
        };
        let single = geometry.annular_partition().expect("single hole must partition");
        let plural = geometry.annular_partitions().expect("single Free hole must partition");
        assert_eq!(plural.len(), 1);
        assert_eq!(plural[0], single);
    }

    /// Two well-separated Free holes must each get their own real, correctly-centered
    /// partition - `Fixed` holes are excluded entirely (no partition for them).
    #[test]
    fn annular_partitions_returns_one_partition_per_free_hole_and_skips_fixed_holes() {
        let geometry = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 0.1,
            holes: vec![
                HoleSpec { center: [-0.5, 0.0], radius: 0.05, bc: HoleBc::Free },
                HoleSpec { center: [0.0, 0.0], radius: 0.05, bc: HoleBc::Fixed },
                HoleSpec { center: [0.5, 0.0], radius: 0.05, bc: HoleBc::Free },
            ],
        };
        let partitions = geometry.annular_partitions().expect("two well-separated Free holes must partition");
        assert_eq!(partitions.len(), 2, "exactly the two Free holes, Fixed excluded");
        assert_eq!(partitions[0].center, [-0.5, 0.0]);
        assert_eq!(partitions[1].center, [0.5, 0.0]);
    }

    /// Real overlap validation, previously nonexistent: two Free holes close enough that their
    /// INTERFACE circles (not just the holes themselves) overlap must return `None`, loudly.
    #[test]
    fn annular_partitions_returns_none_when_interface_circles_overlap() {
        // radius=0.1 => interface_radius=0.3 each; centers 0.5 apart => circles overlap
        // (0.3+0.3=0.6 > 0.5) even though the holes themselves (0.1+0.1=0.2 < 0.5) do not.
        let geometry = UserGeometry {
            half_w: 2.0, half_h: 2.0, thickness: 0.1,
            holes: vec![
                HoleSpec { center: [-0.25, 0.0], radius: 0.1, bc: HoleBc::Free },
                HoleSpec { center: [0.25, 0.0], radius: 0.1, bc: HoleBc::Free },
            ],
        };
        assert!(geometry.validate().is_ok(), "the holes themselves must not overlap, only their interface circles");
        assert!(geometry.annular_partitions().is_none(), "overlapping interface circles must be rejected, not silently assigned");
    }

    /// No Free holes at all (every hole Fixed, or no holes) - nothing to decompose.
    #[test]
    fn annular_partitions_returns_none_when_no_free_holes_exist() {
        let geometry = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 0.1,
            holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.1, bc: HoleBc::Fixed }],
        };
        assert!(geometry.annular_partitions().is_none());
    }

    #[test]
    fn signed_distance_matches_hand_computed_values() {
        let geom = two_hole_geometry();
        // Interior point midway between holes: rect_sdf=1.0, both hole sdfs=0.5-0.1=0.4.
        assert!((geom.signed_distance(0.0, 0.0) - 0.4).abs() < 1e-12);
        // Exactly at hole 1's center: inside the hole, sdf = 0 - radius = -0.1 (the min term).
        assert!((geom.signed_distance(-0.5, 0.0) - (-0.1)).abs() < 1e-12);
        // Outside the outer rectangle: rect_sdf = 1.0 - 1.5 = -0.5, dominates the min.
        assert!((geom.signed_distance(1.5, 0.0) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn nearest_boundary_identifies_the_closest_outer_edge_or_hole() {
        let geom = two_hole_geometry();
        assert_eq!(geom.nearest_boundary(0.99, 0.0), BoundaryRef::OuterRight);
        assert_eq!(geom.nearest_boundary(-0.99, 0.0), BoundaryRef::OuterLeft);
        assert_eq!(geom.nearest_boundary(0.0, 0.99), BoundaryRef::OuterTop);
        assert_eq!(geom.nearest_boundary(0.0, -0.99), BoundaryRef::OuterBottom);
        // Just outside hole 1's boundary (radius 0.1, center -0.5) - much closer to that hole
        // than to any outer edge.
        assert_eq!(geom.nearest_boundary(-0.39, 0.0), BoundaryRef::Hole(0));
        assert_eq!(geom.nearest_boundary(0.39, 0.0), BoundaryRef::Hole(1));
    }

    #[test]
    fn boundary_normal_is_axis_aligned_on_outer_edges_and_radial_on_holes() {
        let geom = two_hole_geometry();
        assert_eq!(geom.boundary_normal(0.99, 0.0), (1.0, 0.0));
        assert_eq!(geom.boundary_normal(-0.99, 0.0), (-1.0, 0.0));
        assert_eq!(geom.boundary_normal(0.0, 0.99), (0.0, 1.0));
        assert_eq!(geom.boundary_normal(0.0, -0.99), (0.0, -1.0));
        // Point just outside hole 1, to its right - radial direction points away from the
        // hole's center, i.e. in +x.
        let (nx, ny) = geom.boundary_normal(-0.39, 0.0);
        assert!((nx - 1.0).abs() < 1e-9, "{nx}");
        assert!(ny.abs() < 1e-9, "{ny}");
    }

    #[test]
    fn boundary_tangent_is_perpendicular_to_the_normal() {
        let geom = two_hole_geometry();
        assert_eq!(geom.boundary_tangent(0.99, 0.0), (0.0, 1.0));
        let (nx, ny) = geom.boundary_normal(0.99, 0.0);
        let (tx, ty) = geom.boundary_tangent(0.99, 0.0);
        assert!((nx * tx + ny * ty).abs() < 1e-12, "normal and tangent must be perpendicular");
    }

    #[test]
    fn boundary_measure_matches_hand_computed_lengths() {
        let geom = two_hole_geometry();
        assert!((geom.boundary_measure(BoundaryRef::OuterLeft) - 2.0).abs() < 1e-12);
        assert!((geom.boundary_measure(BoundaryRef::OuterTop) - 2.0).abs() < 1e-12);
        let expected_circumference = 2.0 * std::f64::consts::PI * 0.1;
        assert!((geom.boundary_measure(BoundaryRef::Hole(0)) - expected_circumference).abs() < 1e-12);
    }

    #[test]
    fn valid_stencil_is_fully_valid_far_from_any_boundary() {
        let geom = two_hole_geometry();
        let v = geom.valid_stencil(0.0, 0.0, 0.05, 0.05);
        assert!(v.all_valid());
    }

    #[test]
    fn valid_stencil_flags_exactly_the_direction_that_crosses_into_a_hole() {
        let geom = two_hole_geometry();
        // (-0.65, 0) is outside hole 1 (dist to center 0.15 > radius 0.1) - a valid center.
        // Shifting +hx=0.1 lands at (-0.55, 0): dist to hole 1 center = 0.05 < radius 0.1 -
        // INSIDE the hole. Shifting -hx lands at (-0.75, 0): still well outside. y-shifts stay
        // at x=-0.65, also well outside either hole.
        let v = geom.valid_stencil(-0.65, 0.0, 0.1, 0.1);
        assert!(v.center_valid);
        assert!(!v.x_plus_valid, "x+ neighbor crosses into hole 1, must be flagged invalid");
        assert!(v.x_minus_valid);
        assert!(v.y_plus_valid);
        assert!(v.y_minus_valid);
        assert!(!v.all_valid());
    }

    #[test]
    fn to_placeholder_preserves_bounding_box_and_has_no_hole() {
        let geom = two_hole_geometry();
        let placeholder = geom.to_placeholder();
        assert_eq!(placeholder.half_w, geom.half_w);
        assert_eq!(placeholder.half_h, geom.half_h);
        assert_eq!(placeholder.thickness, geom.thickness);
        assert_eq!(placeholder.hole, HoleType::None);
        assert_eq!(placeholder.symmetry, SymmetryMode::Full);
    }

    #[test]
    fn inflated_for_collocation_grows_every_hole_radius_by_the_margin_and_nothing_else() {
        let geom = two_hole_geometry();
        let margin = 0.003;
        let inflated = geom.inflated_for_collocation(margin);
        assert_eq!(inflated.half_w, geom.half_w);
        assert_eq!(inflated.half_h, geom.half_h);
        assert_eq!(inflated.thickness, geom.thickness);
        assert_eq!(inflated.holes.len(), geom.holes.len());
        for (orig, grown) in geom.holes.iter().zip(inflated.holes.iter()) {
            assert_eq!(grown.center, orig.center);
            assert_eq!(grown.bc, orig.bc);
            assert!((grown.radius - (orig.radius + margin)).abs() < 1e-15);
        }
    }

    #[test]
    fn inflated_for_collocation_rejects_points_the_original_geometry_would_accept() {
        let geom = two_hole_geometry();
        let margin = 0.02; // deliberately large relative to this fixture's 0.1 m radius
        let inflated = geom.inflated_for_collocation(margin);
        let hole = geom.holes[0];
        let just_outside_true_radius = (hole.center[0] + hole.radius + margin * 0.5, hole.center[1]);
        assert!(geom.contains(just_outside_true_radius.0, just_outside_true_radius.1));
        assert!(!inflated.contains(just_outside_true_radius.0, just_outside_true_radius.1));
    }
}
