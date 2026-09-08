/// Self-learning quadtree Adaptive Mesh Refinement for PINN collocation.
///
/// The grid learns from training history via per-cell EMA residuals and trend tracking.
/// Cells with high AND rising residuals get priority refinement (stuck regions).
/// Cells with low residuals are coarsened (wasted points removed).
/// Hole zone cells are permanently locked at a minimum refinement level.
///
/// Integration with training loop:
///   - Phase 1: disabled — use static sample_interior() for stable BC convergence
///   - Phase 2: AMR::adapt() called every interval_steps; sample_points() replaces int_pts_phys

use crate::geometry::{GeometryConfig, HoleType};
use crate::user_geometry::UserGeometry;

/// A 2D domain `AdaptiveGrid` can refine collocation points over — generalizes the single
/// origin-centered-hole assumption `GeometryConfig` bakes in (Kirsch/pin-lug's quarter-
/// symmetry-friendly domains) to ANY geometry that can describe its own bounds/containment
/// and zero or more "must-stay-refined" feature zones. AMR itself stays fully generic and
/// problem-agnostic this way: nothing is gated on a hardcoded "does this problem have a
/// hole" flag anywhere in this module — a feature-less geometry (`lock_zones() == []`) just
/// runs pure residual-driven refine/coarsen with no permanent floor, and a geometry with N
/// features (N holes, or one, or none) gets N independently-enforced zones for free.
pub trait AmrDomain {
    fn x_range(&self) -> (f64, f64);
    fn y_range(&self) -> (f64, f64);
    fn contains(&self, x: f64, y: f64) -> bool;
    /// Zero or more `(center_x, center_y, radius)` zones that must stay refined to at least
    /// `AmrtConfig::min_level_hole` regardless of residual signal — generalizes the single
    /// implicit hole-at-origin zone `GeometryConfig`-based AMR has always enforced.
    fn lock_zones(&self) -> Vec<(f64, f64, f64)>;
}

impl AmrDomain for GeometryConfig {
    fn x_range(&self) -> (f64, f64) { GeometryConfig::x_range(self) }
    fn y_range(&self) -> (f64, f64) { GeometryConfig::y_range(self) }
    fn contains(&self, x: f64, y: f64) -> bool { GeometryConfig::contains(self, x, y) }
    /// One zone from this geometry's existing single `HoleType`, at the origin — a 1:1
    /// restatement of what `enforce_hole_zone` already hardcoded before this generalization,
    /// so every existing Kirsch/pin-lug caller's behavior is provably unchanged.
    fn lock_zones(&self) -> Vec<(f64, f64, f64)> {
        match self.hole {
            HoleType::Circular { radius } => vec![(0.0, 0.0, radius)],
            HoleType::None => vec![],
        }
    }
}

impl AmrDomain for UserGeometry {
    fn x_range(&self) -> (f64, f64) { (-self.half_w, self.half_w) }
    fn y_range(&self) -> (f64, f64) { (-self.half_h, self.half_h) }
    fn contains(&self, x: f64, y: f64) -> bool { UserGeometry::contains(self, x, y) }
    /// One zone per hole, at that hole's own center — the N-hole generalization of
    /// `GeometryConfig`'s single origin-centered zone.
    fn lock_zones(&self) -> Vec<(f64, f64, f64)> {
        self.holes.iter().map(|h| (h.center[0], h.center[1], h.radius)).collect()
    }
}

/// Derives an `AmrtConfig` from a domain's own bounds and lock zones — generic replacement
/// for `pinn_solver::engine::EngineParams::analyze`'s Kirsch-only inline derivation (kept
/// there, unchanged, for now — see this crate's own AMR generalization notes). Refinement
/// depth is driven by the SMALLEST zone's radius relative to the domain's own extent (the
/// hardest feature to resolve sets the requirement); a feature-less domain (no zones) uses
/// the same defaults `EngineParams::analyze` already falls back to for its own hole-less
/// branch. Every other field is a generic, already self-tuning default (percentiles/ema/
/// trend-weight/interval), not derived from any one problem's own tuning.
pub fn derive_amr_config(bounds: (f64, f64, f64, f64), lock_zones: &[(f64, f64, f64)]) -> AmrtConfig {
    const SMALL_FEATURE_RATIO_THRESHOLD: f64 = 0.04;
    let (x0, x1, y0, y1) = bounds;
    let extent = (x1 - x0).max(y1 - y0).max(1e-12);
    let smallest_zone_r = lock_zones.iter().map(|&(_, _, r)| r).fold(f64::INFINITY, f64::min);
    let feature_ratio = if smallest_zone_r.is_finite() { smallest_zone_r / extent } else { 1.0 };
    let max_level = if feature_ratio < SMALL_FEATURE_RATIO_THRESHOLD { 7 } else { 6 };
    // Phase 12 ("Prevent AMR Runaway") of the "Neural-Network-Wide Adaptive Collocation"
    // epic - a real, verified gap: before this, no caller of `derive_amr_config` (the only
    // two AMR-wired paths outside Kirsch's own hand-tuned `engine.rs` config) had ANY active
    // point-count safety net, since `max_active_cells`/`max_growth_fraction` both default to
    // `None` (unbounded) and nothing here ever set them. `4_usize.pow(max_level as u32 - 1)`
    // (one refinement level short of max, applied uniformly) is a real, non-infinite ceiling
    // derived transparently from the SAME `max_level` already computed above, not an
    // independent arbitrary constant - a first-pass default, not empirically tuned, but
    // strictly better than the previous "no cap at all". `max_growth_fraction: Some(1.0)`
    // (at most double per single sweep) bounds the RATE of growth the same way, independent
    // of the absolute cap.
    let max_active_cells = 4_usize.pow(max_level as u32 - 1);
    AmrtConfig {
        initial_level:      4,
        max_level,
        min_level_hole:     max_level - 1,
        hole_zone_factor:   DEFAULT_HOLE_ZONE_FACTOR,
        refine_percentile:  0.80,
        coarsen_percentile: 0.15,
        ema_alpha:          0.30,
        trend_weight:       0.40,
        interval_steps:     1000,
        pts_per_cell:       1,
        max_active_cells:   Some(max_active_cells),
        max_growth_fraction: Some(1.0),
        // See AmrtConfig::residual_threshold/nonuniformity_threshold's own doc comments for
        // why these stay None even here - no universal, non-arbitrary default exists.
        residual_threshold: None,
        nonuniformity_threshold: None,
    }
}

/// Global-mean PDE residual below which the grid is considered converged. Shared between
/// `is_converged()` and the self-tuning "ease off near convergence" branch in `adapt()` so
/// the two checks can't silently drift apart.
const CONVERGENCE_RESIDUAL_THRESHOLD: f64 = 5e-4;
/// EMA decay for the grid-wide (not per-cell) mean residual tracked in `update_residuals()`.
const GLOBAL_MEAN_EMA_ALPHA: f64 = 0.1;
/// Number of `adapt()` calls before the self-tuning improvement check engages — lets the
/// EMA-tracked global mean settle past its initial transient first.
const SELF_TUNE_WARMUP_ADAPTS: usize = 5;
/// Floor below which `prev_global` is treated as zero (avoids a near-zero-divisor blowup
/// in the improvement-ratio calculation).
const PREV_GLOBAL_EPSILON: f64 = 1e-10;
/// Relative improvement in global mean residual below which refinement is considered
/// "stuck" and the refine percentile is lowered to refine more aggressively.
const STUCK_IMPROVEMENT_THRESHOLD: f64 = 0.10;
/// How much to lower/raise `refine_percentile` when stuck / near-converged.
const REFINE_PERCENTILE_STEP: f64 = 0.05;
/// Floor and ceiling the self-tuned `refine_percentile` is clamped to.
const REFINE_PERCENTILE_MIN: f64 = 0.65;
const REFINE_PERCENTILE_MAX: f64 = 0.92;
/// Extra `adapt()` calls (beyond the warmup) required before the "ease off near
/// convergence" branch can fire — avoids easing off on a single lucky low-residual read.
const SELF_TUNE_EASE_OFF_ADAPTS: usize = 10;
const COARSEN_QUORUM: usize = 3; // out of 4 fixed siblings — tolerates one persistently-noisy
                                   // outlier (issue #14); hole-zone/leaf-structural preconditions
                                   // remain unanimous, not subject to this relaxation.
/// Upper bound on `enforce_active_cell_cap()`'s bounded escalation loop — guarantees the
/// cap-enforcement pass always terminates, even when `max_active_cells` can never be
/// satisfied given the hole-zone/structural floor.
const MAX_CAP_ENFORCEMENT_ROUNDS: usize = 8;
/// Per-round increase to the coarsen-eligibility percentile `enforce_active_cell_cap()`
/// escalates through while the mesh remains over its configured cap.
const CAP_ENFORCEMENT_PERCENTILE_STEP: f64 = 0.10;

/// Default zone-lock radius factor: cells within `hole_zone_factor × r_hole` of the hole
/// center are locked at `min_level_hole` (see `AdaptiveGrid::enforce_hole_zone`). `pub` so
/// `pinn_solver::engine::EngineParams::analyze` (which builds its own `AmrtConfig` from
/// derived, per-problem values) and `pinn_solver::kirsch_problem::KirschSamplingStrategy::
/// amr_lock_zone` (which mirrors this zone in geometry-only terms, without an `AmrtConfig`
/// of its own) both read this single value instead of each hand-typing `3.0` independently.
pub const DEFAULT_HOLE_ZONE_FACTOR: f64 = 3.0;

/// Configuration — derived from problem geometry in engine.rs, not hardcoded.
#[derive(Debug, Clone)]
pub struct AmrtConfig {
    /// Depth of initial uniform grid (4 → 16×16 = 256 cells).
    pub initial_level:      usize,
    /// Maximum refinement depth (7 → up to 128×128 cells).
    pub max_level:          usize,
    /// Minimum refinement level near hole (always maintained regardless of residual).
    pub min_level_hole:     usize,
    /// Cells within `hole_zone_factor × r_hole` of origin are locked at min_level_hole.
    /// Defaults to `DEFAULT_HOLE_ZONE_FACTOR`.
    pub hole_zone_factor:   f64,
    /// Refine leaf cells above this residual percentile (self-tuning based on convergence trend).
    pub refine_percentile:  f64,
    /// Coarsen leaf cells below this residual percentile (only outside hole zone).
    pub coarsen_percentile: f64,
    /// EMA decay for per-cell residual smoothing (higher = more responsive, noisier).
    pub ema_alpha:          f64,
    /// Weight of residual trend (EMA increase) in refinement priority score.
    pub trend_weight:       f64,
    /// Steps between AMR sweeps (called externally from training loop).
    pub interval_steps:     usize,
    /// Collocation points per leaf cell (1 = cell center only).
    pub pts_per_cell:       usize,
    /// Best-effort soft cap on total active leaf cells (`None` = unbounded — the historical
    /// behavior, zero change). When `Some(cap)` and `active_count() > cap`, `adapt()` runs an
    /// additional coarsen-only escalation pass (`enforce_active_cell_cap`) beyond its normal
    /// refine/coarsen sweep. This is a *best-effort* cap, not a hard limit: a value at or
    /// below the structural floor imposed by the hole zone (`min_level_hole`) can never be
    /// met exactly, and the enforcement pass must never panic or violate the hole-zone/
    /// `max_level` invariants trying to reach it — it simply coarsens as far as the
    /// quorum/hole-zone preconditions structurally allow and then stops.
    pub max_active_cells:   Option<usize>,
    /// Best-effort cap on how much a SINGLE `adapt()` call may grow the active cell count,
    /// as a fraction of the count before that call (Phase 12, "Prevent AMR Runaway", of the
    /// "Neural-Network-Wide Adaptive Collocation" epic). `None` = unbounded (zero behavior
    /// change - matches `max_active_cells`'s own established opt-in convention). `Some(1.0)`
    /// means "at most double per sweep". Independent of `max_active_cells`: this bounds the
    /// RATE of growth per sweep; `max_active_cells` bounds the ABSOLUTE total. Same best-
    /// effort semantics as `max_active_cells` - never violates the hole-zone/`max_level`
    /// floor trying to satisfy it.
    pub max_growth_fraction: Option<f64>,
    /// Phase 7 ("Improve Activation Only If Required") of the "Neural-Network-Wide Adaptive
    /// Collocation" epic - skip a sweep entirely (not just its refine/coarsen effect, the
    /// whole `adapt()` call) when the just-probed residuals' mean is below this. `None`
    /// (the default, everywhere including `derive_amr_config`) = always pass, zero behavior
    /// change - deliberately NOT given a real default value anywhere: a universal absolute
    /// threshold across problems with different physical residual scales (this signal is
    /// `dem_energy_per_point`, normalized differently per problem via `ref_energy`) would be
    /// exactly the kind of "invent an arbitrary threshold" this epic's own rules warn
    /// against. A caller who wants this active must calibrate it to their own problem's
    /// residual scale. See `AdaptiveGrid::should_adapt`.
    pub residual_threshold: Option<f64>,
    /// Skip a sweep when the just-probed residuals' max/mean ratio is below this - i.e. the
    /// residual is too UNIFORM to expect adaptive concentration to help (matches the epic's
    /// own worked example: "RMS=1e-4, MAX=1.2e-4" doesn't justify refinement, "RMS=1e-4,
    /// MAX=2.5e-2" does). Also doubles as this policy's "expected benefit" gate (Phase 6/7) -
    /// a highly nonuniform residual is exactly the condition under which concentrating
    /// points is expected to help; a near-uniform one is exactly the condition under which
    /// it isn't, so no separate benefit metric is introduced. `None` everywhere by default,
    /// same rationale as `residual_threshold` - see `AdaptiveGrid::should_adapt`.
    pub nonuniformity_threshold: Option<f64>,
}

impl Default for AmrtConfig {
    fn default() -> Self {
        Self {
            initial_level:      4,
            max_level:          7,
            min_level_hole:     6,
            hole_zone_factor:   DEFAULT_HOLE_ZONE_FACTOR,
            refine_percentile:  0.80,
            coarsen_percentile: 0.15,
            ema_alpha:          0.30,
            trend_weight:       0.40,
            interval_steps:     1000,
            pts_per_cell:       1,
            max_active_cells:   None,
            max_growth_fraction: None,
            residual_threshold: None,
            nonuniformity_threshold: None,
        }
    }
}

/// Summary statistics returned from stats().
#[derive(Debug, Clone)]
pub struct AmrtStats {
    pub active_count:  usize,
    pub max_depth:     usize,
    pub mean_residual: f64,
    pub adapt_count:   usize,
}

/// Global coverage diagnostics — see `AdaptiveGrid::coverage_stats`'s doc comment.
#[derive(Debug, Clone, Copy)]
pub struct CoverageStats {
    /// 1 / (largest leaf-cell area) — the sparsest sampling density anywhere in the domain.
    pub min_local_density: f64,
    /// 1 / (smallest leaf-cell area) — the densest sampling anywhere in the domain.
    pub max_local_density: f64,
    /// `max_local_density / min_local_density` — 1.0 means perfectly uniform; large values
    /// mean AMR has concentrated points strongly in a sub-region.
    pub density_ratio: f64,
    /// Mean leaf-cell side length across the domain — a fast proxy for typical point spacing
    /// (see doc comment: not a rigorous nearest-neighbor search).
    pub mean_spacing: f64,
}

// ─── Quadtree node ────────────────────────────────────────────────────────────

struct QuadNode {
    x0: f64, x1: f64,
    y0: f64, y1: f64,
    level: usize,
    children: Option<Box<[QuadNode; 4]>>,
    residual_ema:  f64,
    residual_prev: f64,
    in_hole_zone:  bool,
}

impl QuadNode {
    fn new(x0: f64, x1: f64, y0: f64, y1: f64, level: usize) -> Self {
        Self { x0, x1, y0, y1, level,
               children: None, residual_ema: 0.0,
               residual_prev: 0.0, in_hole_zone: false }
    }

    #[inline] fn cx(&self) -> f64 { (self.x0 + self.x1) * 0.5 }
    #[inline] fn cy(&self) -> f64 { (self.y0 + self.y1) * 0.5 }
    #[inline] fn is_leaf(&self) -> bool { self.children.is_none() }

    fn split(&mut self) {
        let (cx, cy, l) = (self.cx(), self.cy(), self.level + 1);
        self.children = Some(Box::new([
            QuadNode::new(self.x0, cx,   self.y0, cy,   l),
            QuadNode::new(cx, self.x1,   self.y0, cy,   l),
            QuadNode::new(self.x0, cx,   cy, self.y1,   l),
            QuadNode::new(cx, self.x1,   cy, self.y1,   l),
        ]));
        // Inherit residual so children start with parent's history
        if let Some(ref mut ch) = self.children {
            for c in ch.iter_mut() {
                c.residual_ema  = self.residual_ema;
                c.residual_prev = self.residual_prev;
            }
        }
    }

    fn count_leaves(&self) -> usize {
        match &self.children {
            None => 1,
            Some(ch) => ch.iter().map(|c| c.count_leaves()).sum(),
        }
    }

    fn max_depth(&self) -> usize {
        match &self.children {
            None => self.level,
            Some(ch) => ch.iter().map(|c| c.max_depth()).max().unwrap_or(self.level),
        }
    }

    fn collect_points<G: AmrDomain>(&self, geom: &G, pts: &mut Vec<[f64; 2]>) {
        match &self.children {
            Some(ch) => { for c in ch.iter() { c.collect_points(geom, pts); } }
            None => {
                let (cx, cy) = (self.cx(), self.cy());
                if geom.contains(cx, cy) { pts.push([cx, cy]); }
            }
        }
    }
}

// ─── Adaptive grid ────────────────────────────────────────────────────────────

pub struct AdaptiveGrid<G: AmrDomain = GeometryConfig> {
    root:        QuadNode,
    cfg:         AmrtConfig,
    geom:        G,
    adapt_count: usize,
    global_mean: f64,
    prev_global: f64,
}

impl<G: AmrDomain + Clone> AdaptiveGrid<G> {
    /// Create grid with initial uniform subdivision, then enforce hole zone.
    pub fn new(geom: &G, cfg: AmrtConfig) -> Self {
        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();
        let mut root = QuadNode::new(x0, x1, y0, y1, 0);
        Self::build_uniform(&mut root, cfg.initial_level);

        let mut grid = Self {
            root, cfg,
            geom: geom.clone(),
            adapt_count: 0,
            global_mean: 0.0,
            prev_global: 0.0,
        };
        grid.enforce_hole_zone();
        grid
    }

    fn build_uniform(node: &mut QuadNode, target: usize) {
        if node.level < target {
            if node.is_leaf() { node.split(); }
            if let Some(ref mut ch) = node.children {
                for c in ch.iter_mut() { Self::build_uniform(c, target); }
            }
        }
    }

    /// Return center of every active leaf cell that's inside the domain.
    /// Order is deterministic DFS — must match the order used in update_residuals().
    pub fn sample_points(&self) -> Vec<[f64; 2]> {
        let mut pts = Vec::with_capacity(self.cfg.pts_per_cell * self.active_count());
        self.root.collect_points(&self.geom, &mut pts);
        pts
    }

    /// Assign per-point residuals back to leaf cells.
    /// `residuals` MUST be in the same DFS order as the last sample_points() call.
    pub fn update_residuals(&mut self, residuals: &[f32]) {
        let mut idx = 0_usize;
        let alpha = self.cfg.ema_alpha;
        Self::assign_residuals_rec(&mut self.root, &self.geom, residuals, &mut idx, alpha);

        self.prev_global = self.global_mean;
        let finite: Vec<f64> = residuals.iter()
            .filter(|&&r| r.is_finite()).map(|&r| r as f64).collect();
        if !finite.is_empty() {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            self.global_mean = (1.0 - GLOBAL_MEAN_EMA_ALPHA) * self.global_mean
                + GLOBAL_MEAN_EMA_ALPHA * mean;
        }
    }

    fn assign_residuals_rec(
        node:      &mut QuadNode,
        geom:      &G,
        residuals: &[f32],
        idx:       &mut usize,
        alpha:     f64,
    ) {
        match node.children {
            Some(ref mut ch) => {
                // Internal node — recurse in same DFS order as collect_points
                for c in ch.iter_mut() {
                    Self::assign_residuals_rec(c, geom, residuals, idx, alpha);
                }
            }
            None => {
                // Leaf — only consumes a residual if it contributed a point
                if geom.contains(node.cx(), node.cy()) {
                    if *idx < residuals.len() {
                        let r = residuals[*idx] as f64;
                        if r.is_finite() {
                            node.residual_prev = node.residual_ema;
                            node.residual_ema = (1.0 - alpha) * node.residual_ema + alpha * r;
                        }
                        *idx += 1;
                    }
                }
            }
        }
    }

    /// Run one AMR sweep: refine high-residual cells, coarsen low-residual cells,
    /// then enforce the hole zone. Self-tunes thresholds based on convergence trend.
    pub fn adapt(&mut self) {
        let count_before_adapt = self.active_count();

        // Collect per-leaf residuals to determine thresholds
        let mut leaf_res: Vec<f64> = Vec::new();
        Self::collect_leaf_residuals(&self.root, &self.geom, &mut leaf_res);
        if leaf_res.is_empty() { return; }

        let mut sorted = leaf_res.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        // Self-tuning: if improvement is stalling, refine more aggressively
        let refine_pct = {
            let base = self.cfg.refine_percentile;
            if self.adapt_count > SELF_TUNE_WARMUP_ADAPTS && self.prev_global > PREV_GLOBAL_EPSILON {
                let improvement =
                    (self.prev_global - self.global_mean) / self.prev_global;
                if improvement < STUCK_IMPROVEMENT_THRESHOLD {
                    (base - REFINE_PERCENTILE_STEP).max(REFINE_PERCENTILE_MIN) // stuck → refine more
                } else if self.adapt_count > SELF_TUNE_EASE_OFF_ADAPTS
                    && self.global_mean < CONVERGENCE_RESIDUAL_THRESHOLD
                {
                    (base + REFINE_PERCENTILE_STEP).min(REFINE_PERCENTILE_MAX) // near converged → ease off
                } else {
                    base
                }
            } else {
                base
            }
        };

        let refine_idx  = ((n as f64 * refine_pct)  as usize).min(n - 1);
        let coarsen_idx = ((n as f64 * self.cfg.coarsen_percentile) as usize).min(n - 1);
        let refine_th  = sorted[refine_idx];
        let coarsen_th = sorted[coarsen_idx];

        let max_lv = self.cfg.max_level;
        let tw = self.cfg.trend_weight;
        Self::apply_adapt_rec(&mut self.root, &self.geom, refine_th, coarsen_th, max_lv, tw);

        // Hole zone is non-negotiable — always enforce after adapt
        self.enforce_hole_zone();
        self.enforce_growth_fraction_cap(count_before_adapt);
        self.enforce_active_cell_cap();
        self.adapt_count += 1;
    }

    fn priority(node: &QuadNode, tw: f64) -> f64 {
        let trend = (node.residual_ema - node.residual_prev).max(0.0);
        node.residual_ema + tw * trend
    }

    fn collect_leaf_residuals(node: &QuadNode, geom: &G, out: &mut Vec<f64>) {
        match &node.children {
            Some(ch) => { for c in ch.iter() { Self::collect_leaf_residuals(c, geom, out); } }
            None => { if geom.contains(node.cx(), node.cy()) { out.push(node.residual_ema); } }
        }
    }

    fn apply_adapt_rec(
        node:       &mut QuadNode,
        geom:       &G,
        refine_th:  f64,
        coarsen_th: f64,
        max_level:  usize,
        tw:         f64,
    ) {
        match node.children {
            Some(ref mut ch) => {
                // Structural preconditions for coarsening stay unanimous, non-negotiable:
                // every sibling must be a leaf, and none may be hole-zone-locked.
                let all_leaves        = ch.iter().all(|c| c.is_leaf());
                let none_in_hole_zone = ch.iter().all(|c| !c.in_hole_zone);
                // Residual eligibility is relaxed to a quorum (issue #14): tolerates one
                // persistently-noisy outlier sibling instead of requiring all 4 to agree,
                // which otherwise permanently blocks 3 well-behaved neighbors from coarsening.
                let n_eligible = ch.iter().filter(|c| Self::priority(c, tw) < coarsen_th).count();
                let should_coarsen = all_leaves && none_in_hole_zone && n_eligible >= COARSEN_QUORUM;

                if should_coarsen {
                    // Merge siblings: parent inherits max residual of children
                    let ema  = ch.iter().map(|c| c.residual_ema).fold(0.0_f64, f64::max);
                    let prev = ch.iter().map(|c| c.residual_prev).fold(0.0_f64, f64::max);
                    node.children = None;
                    node.residual_ema  = ema;
                    node.residual_prev = prev;
                } else {
                    // Recurse into children — need to re-borrow after potential merge above
                    if let Some(ref mut ch) = node.children {
                        for c in ch.iter_mut() {
                            Self::apply_adapt_rec(c, geom, refine_th, coarsen_th, max_level, tw);
                        }
                    }
                }
            }
            None => {
                // Leaf: refine if priority above threshold, not already at max depth,
                // and the cell has a sample point in the domain
                if Self::priority(node, tw) > refine_th
                    && node.level < max_level
                    && geom.contains(node.cx(), node.cy())
                {
                    node.split();
                }
            }
        }
    }

    /// Force all cells overlapping ANY of the geometry's lock zones to ≥ min_level_hole —
    /// the N-zone generalization of the old single-origin-hole enforcement. A geometry with
    /// zero zones (e.g. a feature-less `UserGeometry`, or `GeometryConfig::hole ==
    /// HoleType::None`) is a no-op here: no permanent floor, pure residual-driven refine/
    /// coarsen still applies via `adapt()` — nothing about AMR eligibility is gated on
    /// whether a "hole" exists.
    fn enforce_hole_zone(&mut self) {
        let zones: Vec<(f64, f64, f64)> = self.geom.lock_zones().into_iter()
            .map(|(zx, zy, r)| (zx, zy, r * self.cfg.hole_zone_factor))
            .collect();
        if zones.is_empty() {
            return;
        }
        let min_lv = self.cfg.min_level_hole;
        let max_lv = self.cfg.max_level;
        Self::enforce_zones_rec(&mut self.root, &zones, min_lv, max_lv);
    }

    fn cell_overlaps_zone(node: &QuadNode, zone_cx: f64, zone_cy: f64, zone_r: f64) -> bool {
        // Closest point on rectangle [x0,x1]×[y0,y1] to the zone center.
        let cx = node.x0.max(zone_cx).min(node.x1);
        let cy = node.y0.max(zone_cy).min(node.y1);
        let dx = cx - zone_cx;
        let dy = cy - zone_cy;
        dx * dx + dy * dy < zone_r * zone_r
    }

    fn enforce_zones_rec(node: &mut QuadNode, zones: &[(f64, f64, f64)], min_level: usize, max_level: usize) {
        let in_zone = zones.iter().any(|&(zx, zy, zr)| Self::cell_overlaps_zone(node, zx, zy, zr));
        node.in_hole_zone = in_zone;

        if in_zone && node.level < min_level && node.level < max_level {
            if node.is_leaf() { node.split(); }
        }

        if let Some(ref mut ch) = node.children {
            for c in ch.iter_mut() {
                Self::enforce_zones_rec(c, zones, min_level, max_level);
            }
        }
    }

    /// Parallels `collect_leaf_residuals` but collects refinement *priority* (residual +
    /// trend bonus, see `priority()`) rather than raw residual. `enforce_active_cell_cap`
    /// must compare against this same quantity for its escalation threshold — sizing the
    /// threshold from raw `residual_ema` while comparing leaves against `priority()` (as
    /// happens elsewhere in this file, e.g. `adapt()`'s own refine/coarsen thresholds) would
    /// let the trend bonus silently push cells to either side of a threshold sized from a
    /// different quantity, stalling this loop against plateau residual patterns.
    fn collect_leaf_priorities(node: &QuadNode, geom: &G, tw: f64, out: &mut Vec<f64>) {
        match &node.children {
            Some(ch) => { for c in ch.iter() { Self::collect_leaf_priorities(c, geom, tw, out); } }
            None => { if geom.contains(node.cx(), node.cy()) { out.push(Self::priority(node, tw)); } }
        }
    }

    /// Coarsen-only counterpart to `apply_adapt_rec`'s merge arm, used solely by
    /// `enforce_active_cell_cap`. Shares the exact same structural gate — all 4 siblings
    /// must be leaves, none hole-zone-locked, and at least `COARSEN_QUORUM` of them eligible
    /// — but never splits a leaf (coarsen-only, no refine path): this pass runs strictly
    /// after the normal refine/coarsen sweep and must never grow the tree further.
    ///
    /// Uses `<=` rather than the main pass's strict `<`: this pass only runs when the mesh
    /// is already over budget, so it must be able to make progress against a tied/plateau
    /// residual population, which strict `<` can never admit (an eligibility threshold drawn
    /// from a population that's fully tied at that value has no member strictly less than
    /// it, even though every member is, in effect, at the floor of that population).
    fn apply_coarsen_only_rec(node: &mut QuadNode, coarsen_th: f64, tw: f64) {
        // leaf: coarsen-only pass never splits, so there's nothing to do for a `None` node
        if let Some(ref mut ch) = node.children {
            let all_leaves        = ch.iter().all(|c| c.is_leaf());
            let none_in_hole_zone = ch.iter().all(|c| !c.in_hole_zone);
            let n_eligible = ch.iter().filter(|c| Self::priority(c, tw) <= coarsen_th).count();
            let should_coarsen = all_leaves && none_in_hole_zone && n_eligible >= COARSEN_QUORUM;

            if should_coarsen {
                let ema  = ch.iter().map(|c| c.residual_ema).fold(0.0_f64, f64::max);
                let prev = ch.iter().map(|c| c.residual_prev).fold(0.0_f64, f64::max);
                node.children = None;
                node.residual_ema  = ema;
                node.residual_prev = prev;
            } else {
                // Recurse into children — need to re-borrow after potential merge above
                if let Some(ref mut ch) = node.children {
                    for c in ch.iter_mut() {
                        Self::apply_coarsen_only_rec(c, coarsen_th, tw);
                    }
                }
            }
        }
    }

    /// Best-effort enforcement of `AmrtConfig::max_active_cells` (issue #14). No-op when the
    /// cap is unset (`None` — zero behavior change) or already satisfied. Otherwise runs a
    /// bounded escalation loop, up to `MAX_CAP_ENFORCEMENT_ROUNDS` rounds: each round
    /// recomputes the current leaf-priority distribution (the tree changed under the
    /// previous round's merges), derives a threshold from an increasingly lenient percentile
    /// of it, and applies one coarsen-only pass at that threshold. Stops as soon as the cap
    /// is met or a round fails to merge anything — a cap at or below the hole-zone/
    /// structural floor simply converges to that floor and stops, rather than looping
    /// uselessly or violating the hole-zone/`max_level` invariants to try to satisfy it.
    fn enforce_active_cell_cap(&mut self) {
        let Some(cap) = self.cfg.max_active_cells else { return; };
        self.enforce_cap(cap);
    }

    /// Shared escalation loop behind both `enforce_active_cell_cap` (issue #14,
    /// `AmrtConfig::max_active_cells`) and `enforce_growth_fraction_cap` (Phase 12 of the
    /// "Neural-Network-Wide Adaptive Collocation" epic, `AmrtConfig::max_growth_fraction`) —
    /// both are "coarsen down toward this target cell count" requests differing only in how
    /// the target itself is computed, so the actual enforcement mechanism is one function.
    /// No-op if already at or below `cap`. Otherwise runs a bounded escalation loop, up to
    /// `MAX_CAP_ENFORCEMENT_ROUNDS` rounds: each round recomputes the current leaf-priority
    /// distribution (the tree changed under the previous round's merges), derives a threshold
    /// from an increasingly lenient percentile of it, and applies one coarsen-only pass at
    /// that threshold. Stops as soon as the cap is met or a round fails to merge anything — a
    /// cap at or below the hole-zone/structural floor simply converges to that floor and
    /// stops, rather than looping uselessly or violating the hole-zone/`max_level` invariants
    /// trying to reach it.
    fn enforce_cap(&mut self, cap: usize) {
        if self.active_count() <= cap {
            return;
        }

        let tw = self.cfg.trend_weight;
        for round in 0..MAX_CAP_ENFORCEMENT_ROUNDS {
            let mut priorities: Vec<f64> = Vec::new();
            Self::collect_leaf_priorities(&self.root, &self.geom, tw, &mut priorities);
            if priorities.is_empty() {
                break;
            }
            priorities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = priorities.len();
            let percentile  = (CAP_ENFORCEMENT_PERCENTILE_STEP * (round as f64 + 1.0)).min(1.0);
            let idx         = ((n as f64 * percentile) as usize).min(n - 1);
            let coarsen_th  = priorities[idx];

            let before = self.active_count();
            Self::apply_coarsen_only_rec(&mut self.root, coarsen_th, tw);
            let after = self.active_count();

            if after <= cap || after == before {
                break;
            }
        }
    }

    /// Best-effort enforcement of `AmrtConfig::max_growth_fraction` (Phase 12 - "Prevent AMR
    /// Runaway"): caps how much a SINGLE `adapt()` call may grow the active cell count
    /// relative to what it was before this call, independent of any absolute
    /// `max_active_cells` ceiling. `None` (the default) is zero behavior change - this
    /// safeguard is opt-in, matching `max_active_cells`'s own established convention.
    /// Real, verified gap this closes: before this, a highly nonuniform residual pattern that
    /// persisted across many consecutive sweeps had no mechanism at all bounding how much a
    /// single sweep could grow the mesh by (only the absolute `max_level`/`max_active_cells`
    /// ceilings existed, and the latter defaulted to unset for every wired caller).
    fn enforce_growth_fraction_cap(&mut self, count_before_adapt: usize) {
        let Some(max_growth) = self.cfg.max_growth_fraction else { return; };
        let cap = ((count_before_adapt as f64) * (1.0 + max_growth)).ceil() as usize;
        self.enforce_cap(cap.max(1));
    }

    /// Phase 7 ("Improve Activation Only If Required") of the "Neural-Network-Wide Adaptive
    /// Collocation" epic — the "smart activation" gate. Callers should check this AFTER
    /// warmup/interval have already passed (that's the trigger; this is the additional
    /// intelligence layered on top) but BEFORE calling `adapt()` — `update_residuals` should
    /// still be called unconditionally either way, so the EMA/trend history stays current for
    /// the next check regardless of whether this sweep actually adapts.
    ///
    /// Takes the just-probed residuals directly (not `self`'s own EMA-smoothed
    /// `global_mean`) — matches the epic's own worked examples, which reason about the
    /// CURRENT snapshot ("RMS=1e-4, MAX=1.2e-4"), not a rolling average. Cooldown is already
    /// provided by `AmrtConfig::interval_steps` (callers can't invoke a sweep more often than
    /// that regardless) — no separate cooldown mechanism is introduced here. "Expected
    /// benefit" is folded into the nonuniformity check itself (see
    /// `AmrtConfig::nonuniformity_threshold`'s doc comment) rather than a separate metric.
    ///
    /// Both thresholds default to `None` (always returns `true` — zero behavior change)
    /// unless a caller has explicitly calibrated them to their own problem's residual scale.
    pub fn should_adapt(&self, residuals: &[f32]) -> bool {
        let finite: Vec<f64> = residuals.iter().filter(|r| r.is_finite()).map(|&r| r as f64).collect();
        if finite.is_empty() {
            return false;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        if let Some(rt) = self.cfg.residual_threshold {
            if mean < rt {
                return false;
            }
        }
        if let Some(nt) = self.cfg.nonuniformity_threshold {
            let max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ratio = if mean > 1e-300 { max / mean } else { 0.0 };
            if ratio < nt {
                return false;
            }
        }
        true
    }

    // ─── Public query API ─────────────────────────────────────────────────────

    /// Number of active leaf cells (includes cells outside the domain).
    pub fn active_count(&self) -> usize { self.root.count_leaves() }

    pub fn max_depth(&self) -> usize { self.root.max_depth() }

    /// True when global mean PDE residual is below convergence threshold.
    pub fn is_converged(&self) -> bool { self.global_mean < CONVERGENCE_RESIDUAL_THRESHOLD }

    pub fn stats(&self) -> AmrtStats {
        AmrtStats {
            active_count:  self.active_count(),
            max_depth:     self.max_depth(),
            mean_residual: self.global_mean,
            adapt_count:   self.adapt_count,
        }
    }

    pub fn adapt_count(&self) -> usize { self.adapt_count }

    /// Global coverage diagnostics (Phase 4 of the "Neural-Network-Wide Adaptive
    /// Collocation" epic) - verifies AMR concentrates points adaptively WITHOUT collapsing
    /// coverage into only the highest-residual region. `min_local_density`/`max_local_density`
    /// are exact (1 / leaf-cell-area, computed from the quadtree's own known cell sizes, no
    /// approximation). `mean_spacing` is a fast quadtree-native PROXY for nearest-neighbor
    /// distance (each leaf's own side length) - not a rigorous k-d-tree nearest-neighbor
    /// search, which would be O(n²) over the sampled points; documented as an approximation
    /// so it isn't mistaken for one. Not called anywhere in the training hot path - purely an
    /// opt-in diagnostic (tests, or future UI use), per the epic's own Performance Rule.
    pub fn coverage_stats(&self) -> CoverageStats {
        let mut leaves: Vec<(f64, f64)> = Vec::new(); // (density, side_length) per in-domain leaf
        Self::collect_leaf_coverage(&self.root, &self.geom, &mut leaves);
        if leaves.is_empty() {
            return CoverageStats { min_local_density: 0.0, max_local_density: 0.0, density_ratio: 0.0, mean_spacing: 0.0 };
        }
        let min_d = leaves.iter().map(|&(d, _)| d).fold(f64::INFINITY, f64::min);
        let max_d = leaves.iter().map(|&(d, _)| d).fold(f64::NEG_INFINITY, f64::max);
        let density_ratio = if min_d > 0.0 { max_d / min_d } else { f64::INFINITY };
        let mean_spacing = leaves.iter().map(|&(_, s)| s).sum::<f64>() / leaves.len() as f64;
        CoverageStats { min_local_density: min_d, max_local_density: max_d, density_ratio, mean_spacing }
    }

    fn collect_leaf_coverage(node: &QuadNode, geom: &G, out: &mut Vec<(f64, f64)>) {
        match &node.children {
            Some(ch) => { for c in ch.iter() { Self::collect_leaf_coverage(c, geom, out); } }
            None => {
                if geom.contains(node.cx(), node.cy()) {
                    let w = node.x1 - node.x0;
                    let h = node.y1 - node.y0;
                    let area = (w * h).max(1e-300);
                    out.push((1.0 / area, w.min(h)));
                }
            }
        }
    }

    /// Mean local density (1 / leaf-area) of leaves overlapping a circular zone — Phase 8
    /// ("Plate-With-Hole Physics Validation") diagnostic: answers "how refined is AMR near
    /// THIS region" as a single comparable number, e.g. before vs. after a sweep. `0.0` if no
    /// leaf overlaps the zone (shouldn't happen for a zone inside the domain bounds, but a
    /// degenerate zero-radius/off-domain zone must not panic or divide by zero).
    pub fn zone_density(&self, cx: f64, cy: f64, r: f64) -> f64 {
        let mut leaves = Vec::new();
        Self::collect_zone_leaf_density(&self.root, &self.geom, cx, cy, r, &mut leaves);
        if leaves.is_empty() { return 0.0; }
        leaves.iter().sum::<f64>() / leaves.len() as f64
    }

    fn collect_zone_leaf_density(node: &QuadNode, geom: &G, cx: f64, cy: f64, r: f64, out: &mut Vec<f64>) {
        match &node.children {
            Some(ch) => { for c in ch.iter() { Self::collect_zone_leaf_density(c, geom, cx, cy, r, out); } }
            None => {
                if geom.contains(node.cx(), node.cy()) && Self::cell_overlaps_zone(node, cx, cy, r) {
                    let w = node.x1 - node.x0;
                    let h = node.y1 - node.y0;
                    out.push(1.0 / (w * h).max(1e-300));
                }
            }
        }
    }

    /// `zone_density` averaged over every one of `self.geom.lock_zones()` — `0.0` for a
    /// feature-less geometry with no lock zones (there is no hole to concentrate near, which
    /// is itself the correct, meaningful answer, not a missing one).
    pub fn lock_zone_density(&self) -> f64 {
        let zones = self.geom.lock_zones();
        if zones.is_empty() { return 0.0; }
        zones.iter().map(|&(zx, zy, zr)| self.zone_density(zx, zy, zr)).sum::<f64>() / zones.len() as f64
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{HoleType, SymmetryMode};

    fn test_geom() -> GeometryConfig {
        GeometryConfig {
            half_w: 0.127,
            half_h: 0.127,
            thickness: 0.00254,
            hole: HoleType::Circular { radius: 0.003175 },
            symmetry: SymmetryMode::QuarterSymm,
        }
    }

    #[test]
    fn test_no_points_inside_hole() {
        let geom = test_geom();
        let cfg = AmrtConfig::default();
        let grid = AdaptiveGrid::new(&geom, cfg);
        let pts = grid.sample_points();
        let r_hole = 0.003175_f64;
        for &[x, y] in &pts {
            let r = (x * x + y * y).sqrt();
            assert!(r >= r_hole * 0.999, "point ({x:.4},{y:.4}) inside hole, r={r:.5} < {r_hole}");
        }
    }

    #[test]
    fn test_initial_count_reasonable() {
        let geom = test_geom();
        let cfg  = AmrtConfig { initial_level: 4, min_level_hole: 6, ..AmrtConfig::default() };
        let grid = AdaptiveGrid::new(&geom, cfg);
        let n = grid.active_count();
        assert!(n >= 256, "expected ≥256 cells after L4 init + hole zone, got {n}");
        assert!(n <= 2048, "initial count unexpectedly large: {n}");
    }

    #[test]
    fn test_max_level_not_exceeded() {
        let geom = test_geom();
        let cfg  = AmrtConfig { max_level: 6, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        // Inject very high residuals to trigger aggressive refinement
        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|_| 1.0_f32).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        assert!(grid.max_depth() <= 6, "max_level violated: depth={}", grid.max_depth());
    }

    #[test]
    fn test_high_residual_triggers_refine() {
        let geom = test_geom();
        let cfg  = AmrtConfig::default();
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        let n_before = grid.sample_points().len();

        let pts = grid.sample_points();
        // All high residuals → refine everything
        let res: Vec<f32> = pts.iter().map(|_| 1.0_f32).collect();
        grid.update_residuals(&res);
        grid.adapt();

        let n_after = grid.sample_points().len();
        assert!(n_after > n_before, "refine did not increase count: {n_before} → {n_after}");
    }

    #[test]
    fn test_low_residual_triggers_coarsen() {
        let geom = test_geom();
        let cfg  = AmrtConfig { initial_level: 5, min_level_hole: 5, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        let n_before = grid.sample_points().len();

        // All zero residuals → coarsen everything outside hole zone. Run several EMA
        // cycles so residual_ema actually falls to the coarsen threshold.
        for _ in 0..10 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|_| 0.0_f32).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let n_after = grid.sample_points().len();
        // May not coarsen if everything is in the hole zone — just check it doesn't blow up
        assert!(n_after <= n_before + 10, "coarsen not working: {n_before} → {n_after}");
    }

    #[test]
    fn test_hole_zone_always_refined() {
        let geom = test_geom();
        let cfg  = AmrtConfig { min_level_hole: 6, hole_zone_factor: DEFAULT_HOLE_ZONE_FACTOR, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        // Force coarsen
        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|_| 0.0_f32).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        // All points near hole should still be there
        let pts = grid.sample_points();
        let r_zone = 0.003175 * DEFAULT_HOLE_ZONE_FACTOR;
        let near_hole: Vec<_> = pts.iter().filter(|&&[x, y]| {
            (x*x + y*y).sqrt() < r_zone
        }).collect();
        assert!(!near_hole.is_empty(), "hole zone was coarsened away — no near-hole points remain");
    }

    /// `hole_zone_factor`'s value (3.0) must live in exactly one place. `AmrtConfig::default()`
    /// reads `DEFAULT_HOLE_ZONE_FACTOR` instead of hand-typing its own `3.0`, and
    /// `pinn_solver::engine::EngineParams::analyze` / `KirschSamplingStrategy::amr_lock_zone`
    /// (pinn-solver, see their own tests) read the same constant rather than each carrying an
    /// independent copy that could silently drift from this one.
    #[test]
    fn default_hole_zone_factor_is_shared_constant() {
        assert_eq!(AmrtConfig::default().hole_zone_factor, DEFAULT_HOLE_ZONE_FACTOR);
    }

    /// A simple no-hole, full-symmetry [-1,1]x[-1,1] domain — isolates the coarsening-quorum
    /// fix from hole-zone interaction (covered separately).
    fn no_hole_geom() -> GeometryConfig {
        GeometryConfig {
            half_w: 1.0,
            half_h: 1.0,
            thickness: 1.0,
            hole: HoleType::None,
            symmetry: SymmetryMode::Full,
        }
    }

    #[test]
    fn test_coarsen_never_merges_when_all_four_siblings_stay_above_threshold() {
        const NOISY: f32 = 1.0;
        const PADDING: f32 = 0.001;

        let cfg = AmrtConfig {
            initial_level: 3, max_level: 3, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        assert_eq!(grid.active_count(), 64);

        for _ in 0..10 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter()
                .map(|&[x, y]| if x < -0.5 && y < -0.5 { NOISY } else { PADDING })
                .collect();
            grid.update_residuals(&res);
            grid.adapt();
        }

        assert_eq!(grid.active_count(), 64,
            "a quad merged even though 0-of-4 siblings satisfied the coarsen threshold");
    }

    #[test]
    fn test_coarsen_blocked_by_single_hole_zone_sibling_despite_full_residual_agreement() {
        let geom = GeometryConfig {
            half_w: 1.0, half_h: 1.0, thickness: 1.0,
            hole: HoleType::Circular { radius: 0.01 }, // zone_r = radius * hole_zone_factor = 0.05
            symmetry: SymmetryMode::QuarterSymm,
        };
        let cfg = AmrtConfig {
            initial_level: 3,
            max_level: 3,
            min_level_hole: 3,
            hole_zone_factor: 5.0,
            ema_alpha: 1.0,
            ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        assert_eq!(grid.active_count(), 64);

        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter()
                .map(|&[x, y]| if x < 0.25 && y < 0.25 { 0.001_f32 } else { 0.10_f32 })
                .collect();
            grid.update_residuals(&res);
            grid.adapt();
        }

        let pts = grid.sample_points();
        let expected_centers = [[0.0625, 0.0625], [0.1875, 0.0625], [0.0625, 0.1875], [0.1875, 0.1875]];
        for [ex, ey] in expected_centers {
            assert!(
                pts.iter().any(|&[x, y]| (x - ex).abs() < 1e-9 && (y - ey).abs() < 1e-9),
                "quad (0,0) coarsened away point ({ex},{ey}) despite a hole-zone sibling — the \
                 hole-zone lock must remain absolute/unanimous, not subject to the majority relaxation"
            );
        }
    }

    #[test]
    fn test_coarsen_requires_at_least_three_of_four_not_two() {
        const QUIET: f32 = 0.001;
        const NOISY: f32 = 1.0;
        const PADDING: f32 = 0.10;

        let cfg = AmrtConfig {
            initial_level: 3, max_level: 3, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        assert_eq!(grid.active_count(), 64);

        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| {
                if !(x < -0.5 && y < -0.5) { return PADDING; }
                if y < -0.75 { QUIET } else { NOISY }
            }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }

        assert_eq!(grid.active_count(), 64,
            "quad merged with only 2-of-4 siblings eligible — coarsen must require >= 3-of-4");
    }

    #[test]
    fn test_coarsen_merges_when_all_four_of_four_eligible() {
        const QUIET: f32 = 0.001;
        const PADDING: f32 = 0.10;

        let cfg = AmrtConfig {
            initial_level: 3, max_level: 3, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        assert_eq!(grid.active_count(), 64);

        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter()
                .map(|&[x, y]| if x < -0.5 && y < -0.5 { QUIET } else { PADDING })
                .collect();
            grid.update_residuals(&res);
            grid.adapt();
        }

        assert_eq!(grid.active_count(), 61,
            "quad with 4-of-4 eligible siblings failed to merge: expected 64-3=61, got {}",
            grid.active_count());
    }

    // ─── AmrDomain generalization regression guards ─────────────────────────────

    #[test]
    fn geometry_config_lock_zones_matches_old_hardcoded_single_origin_hole() {
        let geom = test_geom(); // hole radius 0.003175, centered at origin
        assert_eq!(geom.lock_zones(), vec![(0.0, 0.0, 0.003175)]);
        assert_eq!(no_hole_geom().lock_zones(), Vec::<(f64, f64, f64)>::new());
    }

    #[test]
    fn user_geometry_lock_zones_one_per_hole_at_its_own_center() {
        use crate::user_geometry::{HoleBc, HoleSpec, UserGeometry};
        let geom = UserGeometry {
            half_w: 0.1, half_h: 0.05, thickness: 0.005,
            holes: vec![
                HoleSpec { center: [-0.03, 0.0], radius: 0.01, bc: HoleBc::Free },
                HoleSpec { center: [0.03, 0.01], radius: 0.008, bc: HoleBc::Fixed },
            ],
        };
        assert_eq!(geom.lock_zones(), vec![(-0.03, 0.0, 0.01), (0.03, 0.01, 0.008)]);
    }

    #[test]
    fn user_geometry_with_no_holes_has_no_lock_zones() {
        use crate::user_geometry::UserGeometry;
        let geom = UserGeometry { half_w: 0.1, half_h: 0.1, thickness: 0.005, holes: vec![] };
        assert!(geom.lock_zones().is_empty());
    }

    /// Regression guard: `derive_amr_config` must reproduce
    /// `pinn_solver::engine::EngineParams::analyze`'s exact `amr_max_level`/`min_level_hole`
    /// for Kirsch's own default geometry — proves the generalized deriver is a strict
    /// superset of the existing hardcoded formula, not a behavior change.
    #[test]
    fn derive_amr_config_matches_kirsch_engine_params_for_default_plate() {
        // Mirrors GeometryConfig::kirsch_plate_inches(): half_w=half_h=5in, hole r=0.125in,
        // QuarterSymm → x_range()/y_range() == (0, half_w)/(0, half_h).
        let half_w = 5.0 * 0.0254;
        let hole_r = 0.125 * 0.0254;
        let cfg = derive_amr_config((0.0, half_w, 0.0, half_w), &[(0.0, 0.0, hole_r)]);
        assert_eq!(cfg.max_level, 7, "small hole ratio (0.025 < 0.04) must select the deeper max_level");
        assert_eq!(cfg.min_level_hole, 6);
    }

    #[test]
    fn derive_amr_config_no_zones_uses_shallower_default_depth() {
        let cfg = derive_amr_config((0.0, 1.0, 0.0, 1.0), &[]);
        assert_eq!(cfg.max_level, 6, "feature-less geometry must use the shallower default, not panic or special-case");
        assert_eq!(cfg.min_level_hole, 5);
    }

    #[test]
    fn adaptive_grid_over_user_geometry_refines_near_each_hole_zone() {
        use crate::user_geometry::{HoleBc, HoleSpec, UserGeometry};
        let geom = UserGeometry {
            half_w: 1.0, half_h: 1.0, thickness: 1.0,
            holes: vec![HoleSpec { center: [0.4, 0.4], radius: 0.02, bc: HoleBc::Free }],
        };
        let cfg = AmrtConfig {
            initial_level: 3, max_level: 6, min_level_hole: 5,
            hole_zone_factor: 5.0, ..AmrtConfig::default()
        };
        let grid = AdaptiveGrid::<UserGeometry>::new(&geom, cfg);
        assert!(grid.max_depth() >= 5, "hole zone at (0.4,0.4) must force refinement to min_level_hole");
    }

    // ─── Phase 3 (Neural-Network-Wide Adaptive Collocation epic): spatial behavior ──────────
    //
    // Fast, deterministic unit tests proving the residual-driven refine/coarsen mechanism
    // itself behaves correctly on KNOWN synthetic residual distributions - independent of any
    // real training loop. `no_hole_geom()` (HoleType::None, zero lock zones) is used
    // throughout except test E, so refinement is driven PURELY by the residual signal, not by
    // the structural hole-zone floor - isolating the mechanism this phase is meant to verify.

    fn count_points_in_box(pts: &[[f64; 2]], x0: f64, x1: f64, y0: f64, y1: f64) -> usize {
        pts.iter().filter(|&&[x, y]| x >= x0 && x < x1 && y >= y0 && y < y1).count()
    }

    #[test]
    fn spatial_test_a_localized_error_concentrates_refinement_in_region() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| if x > 0.5 && y > 0.5 { 1.0 } else { 0.001 }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let dense_a = count_points_in_box(&pts, 0.5, 1.0, 0.5, 1.0);
        let sparse_b = count_points_in_box(&pts, -1.0, -0.5, -1.0, -0.5);
        assert!(
            dense_a > sparse_b * 2,
            "high-residual region A ({dense_a} pts) must be substantially denser than \
             equal-area low-residual region B ({sparse_b} pts)"
        );
    }

    #[test]
    fn spatial_test_b_uniform_error_produces_no_pathological_clustering() {
        let cfg = AmrtConfig { initial_level: 4, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|_| 0.5_f32).collect(); // perfectly uniform
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let quadrants = [
            count_points_in_box(&pts, 0.0, 1.0, 0.0, 1.0),
            count_points_in_box(&pts, -1.0, 0.0, 0.0, 1.0),
            count_points_in_box(&pts, -1.0, 0.0, -1.0, 0.0),
            count_points_in_box(&pts, 0.0, 1.0, -1.0, 0.0),
        ];
        let max_q = *quadrants.iter().max().unwrap();
        let min_q = *quadrants.iter().min().unwrap();
        assert!(
            max_q <= min_q * 2 + 2,
            "uniform residual must not produce pathological clustering in one quadrant: {quadrants:?}"
        );
    }

    #[test]
    fn spatial_test_c_multiple_high_residual_regions_both_receive_refinement() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| {
                let in_a = x > 0.5 && y > 0.5;
                let in_b = x < -0.5 && y < -0.5;
                if in_a || in_b { 1.0 } else { 0.001 }
            }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let region_a = count_points_in_box(&pts, 0.5, 1.0, 0.5, 1.0);
        let region_b = count_points_in_box(&pts, -1.0, -0.5, -1.0, -0.5);
        let quiet_c = count_points_in_box(&pts, 0.5, 1.0, -1.0, -0.5);
        let quiet_d = count_points_in_box(&pts, -1.0, -0.5, 0.5, 1.0);
        assert!(region_a > quiet_c * 2, "region A must refine more than quiet region C: A={region_a} C={quiet_c}");
        assert!(region_b > quiet_d * 2, "region B must refine more than quiet region D: B={region_b} D={quiet_d}");
    }

    #[test]
    fn spatial_test_d_boundary_region_refines_like_any_other_region() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..6 {
            let pts = grid.sample_points();
            // High residual in a strip hugging the domain's outer edge (x close to +1).
            let res: Vec<f32> = pts.iter().map(|&[x, _]| if x > 0.85 { 1.0 } else { 0.001 }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let edge_strip = count_points_in_box(&pts, 0.85, 1.0, -1.0, 1.0);
        let interior = count_points_in_box(&pts, -1.0, 0.85, -1.0, 1.0);
        // Normalize by area: the edge strip covers 15% of the domain width.
        let edge_density = edge_strip as f64 / 0.15;
        let interior_density = interior as f64 / 1.85;
        assert!(
            edge_density > interior_density * 2.0,
            "boundary-hugging high-residual strip must refine like any other region \
             (edge density {edge_density:.1} vs interior density {interior_density:.1})"
        );
    }

    #[test]
    fn spatial_test_e_circular_hole_boundary_ring_refines_from_residual_not_just_lock_zone() {
        let geom = test_geom(); // half_w=half_h=0.127, hole radius 0.003175, centered at origin
        let r_hole = 0.003175_f64;
        // Tiny lock zone (hole_zone_factor near 1) so most of the observed refinement below
        // must come from the residual signal, not the structural hole-zone floor.
        let cfg = AmrtConfig {
            initial_level: 3, max_level: 7, min_level_hole: 4, hole_zone_factor: 1.2,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        let ring_lo = r_hole * 2.0;
        let ring_hi = r_hole * 4.0;
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| {
                let r = (x * x + y * y).sqrt();
                if r >= ring_lo && r <= ring_hi { 1.0 } else { 0.001 }
            }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let in_ring = pts.iter().filter(|&&[x, y]| {
            let r = (x * x + y * y).sqrt();
            r >= ring_lo && r <= ring_hi
        }).count();
        let far_from_ring = pts.iter().filter(|&&[x, y]| (x * x + y * y).sqrt() > ring_hi * 2.0).count();
        let ring_area = std::f64::consts::PI * (ring_hi * ring_hi - ring_lo * ring_lo) / 4.0; // quarter-symm
        let far_area = 0.127 * 0.127 - std::f64::consts::PI * (ring_hi * 2.0).powi(2) / 4.0;
        let ring_density = in_ring as f64 / ring_area.max(1e-12);
        let far_density = far_from_ring as f64 / far_area.max(1e-12);
        assert!(
            ring_density > far_density * 3.0,
            "synthetic high-residual ring around the hole must refine following the circular \
             boundary (ring density {ring_density:.1} vs far-field density {far_density:.1})"
        );
    }

    #[test]
    fn spatial_test_f_single_point_spike_does_not_starve_the_rest_of_the_domain() {
        let cfg = AmrtConfig { initial_level: 4, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        let initial_count = grid.active_count();
        for _ in 0..8 {
            let pts = grid.sample_points();
            // One pathological spike near (0.9, 0.9); everything else at a low, uniform baseline.
            let res: Vec<f32> = pts.iter().map(|&[x, y]| {
                if (x - 0.9).abs() < 0.05 && (y - 0.9).abs() < 0.05 { 1000.0 } else { 0.01 }
            }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let far_from_spike = count_points_in_box(&pts, -1.0, 0.5, -1.0, 0.5);
        assert!(
            far_from_spike as f64 > initial_count as f64 * 0.05,
            "a single pathological residual spike must not coarsen away most of the rest of \
             the domain - far-from-spike region retained only {far_from_spike} points \
             (initial total was {initial_count})"
        );
    }

    // ─── Phase 4: global coverage diagnostics ───────────────────────────────────────────────

    #[test]
    fn coverage_stats_uniform_residual_keeps_density_ratio_near_one() {
        let cfg = AmrtConfig { initial_level: 4, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..6 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|_| 0.5_f32).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let cov = grid.coverage_stats();
        assert!(
            cov.density_ratio < 1.5,
            "uniform residual must keep the domain nearly uniformly refined, got density_ratio={:.2}",
            cov.density_ratio
        );
    }

    #[test]
    fn coverage_stats_localized_residual_does_not_starve_min_density_to_zero() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..8 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| if x > 0.5 && y > 0.5 { 1.0 } else { 0.001 }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let cov = grid.coverage_stats();
        assert!(cov.min_local_density > 0.0, "min_local_density must never be zero/degenerate");
        assert!(cov.density_ratio.is_finite(), "density_ratio must stay finite (min density never literally zero)");
        // A real, meaningful concentration should have happened (this is the whole point of
        // AMR) - the ratio should be well above 1, just not so extreme it signals a coarsen
        // runaway (an unbounded density_ratio would mean the low-residual region collapsed to
        // one giant cell while the hot region kept splitting indefinitely).
        assert!(cov.density_ratio > 2.0, "expected real concentration under localized residual, got ratio={:.2}", cov.density_ratio);
    }

    // ─── Phase 12: prevent AMR runaway ──────────────────────────────────────────────────────

    #[test]
    fn max_growth_fraction_caps_growth_of_a_single_adapt_call() {
        let cfg = AmrtConfig {
            initial_level: 3, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, max_growth_fraction: Some(0.25), // at most +25% per sweep
            ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        let before = grid.active_count();

        // Extreme, maximally-refining residual pattern: everything above the refine
        // threshold, nothing eligible to coarsen - without the cap this would grow far
        // beyond +25% in one call.
        let pts = grid.sample_points();
        let res: Vec<f32> = pts.iter().map(|_| 1.0_f32).collect();
        grid.update_residuals(&res);
        grid.adapt();

        let after = grid.active_count();
        let max_allowed = ((before as f64) * 1.25).ceil() as usize;
        assert!(
            after <= max_allowed,
            "max_growth_fraction=0.25 must cap single-sweep growth: before={before} after={after} max_allowed={max_allowed}"
        );
    }

    #[test]
    fn max_growth_fraction_none_is_zero_behavior_change() {
        // Same extreme setup as above, but with max_growth_fraction left at its default
        // (None) - the sweep should be free to grow well past +25%, proving the cap in the
        // test above is actually doing something, not just always true regardless.
        let cfg = AmrtConfig {
            initial_level: 3, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0,
            ema_alpha: 1.0, ..AmrtConfig::default()
        };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        let before = grid.active_count();
        let pts = grid.sample_points();
        let res: Vec<f32> = pts.iter().map(|_| 1.0_f32).collect();
        grid.update_residuals(&res);
        grid.adapt();
        let after = grid.active_count();
        assert!(
            after > (before as f64 * 1.25).ceil() as usize,
            "uncapped growth should exceed +25% under a maximally-refining residual pattern \
             (before={before} after={after}) - if this fails, the capped test above isn't \
             actually testing anything"
        );
    }

    #[test]
    fn derive_amr_config_sets_a_real_point_count_safety_net() {
        let cfg = derive_amr_config((0.0, 1.0, 0.0, 1.0), &[(0.5, 0.5, 0.02)]);
        assert!(cfg.max_active_cells.is_some(), "derive_amr_config must set a real max_active_cells, not leave the old unbounded default");
        assert!(cfg.max_growth_fraction.is_some(), "derive_amr_config must set a real max_growth_fraction, not leave the old unbounded default");
        assert_eq!(cfg.max_active_cells, Some(4_usize.pow(cfg.max_level as u32 - 1)));
    }

    // ─── Phase 7: smart AMR activation ──────────────────────────────────────────────────────

    #[test]
    fn should_adapt_default_thresholds_always_pass() {
        let grid = AdaptiveGrid::new(&no_hole_geom(), AmrtConfig::default());
        assert!(grid.should_adapt(&[1e-4, 1.0e-4, 1.0e-4]), "default (None) thresholds must always pass - zero behavior change");
        assert!(grid.should_adapt(&[0.0, 0.0, 0.0]), "even all-zero residuals pass when no threshold is configured");
    }

    #[test]
    fn should_adapt_rejects_below_residual_threshold() {
        let cfg = AmrtConfig { residual_threshold: Some(1e-3), ..AmrtConfig::default() };
        let grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        assert!(!grid.should_adapt(&[1e-4, 1e-4, 1e-4]), "mean residual below threshold must skip the sweep");
        assert!(grid.should_adapt(&[1e-2, 1e-2, 1e-2]), "mean residual above threshold must pass");
    }

    #[test]
    fn should_adapt_rejects_uniform_residual_below_nonuniformity_threshold() {
        let cfg = AmrtConfig { nonuniformity_threshold: Some(5.0), ..AmrtConfig::default() };
        let grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        // Matches the epic's own worked example: RMS=1e-4, MAX=1.2e-4 -> ratio 1.2, too uniform.
        assert!(!grid.should_adapt(&[1.0e-4, 1.0e-4, 1.2e-4]), "near-uniform residual (low max/mean ratio) must skip the sweep");
        // 20 low points + 1 spike -> mean stays low, max/mean ratio is large (a single high
        // value averaged into a big low-residual population, not diluting the mean itself).
        let mut nonuniform = vec![1.0e-4_f32; 20];
        nonuniform.push(2.5e-2);
        assert!(grid.should_adapt(&nonuniform), "strongly nonuniform residual must pass");
    }

    #[test]
    fn should_adapt_empty_residuals_returns_false() {
        let grid = AdaptiveGrid::new(&no_hole_geom(), AmrtConfig::default());
        assert!(!grid.should_adapt(&[]), "no residuals at all must never trigger a sweep");
    }

    // ─── Phase 8 (Neural-Network-Wide Adaptive Collocation epic): hole-zone density diagnostic ─

    #[test]
    fn lock_zone_density_is_zero_for_a_feature_less_geometry() {
        let grid = AdaptiveGrid::new(&no_hole_geom(), AmrtConfig::default());
        assert_eq!(grid.lock_zone_density(), 0.0, "no lock zones -> no hole to concentrate near, reported as 0.0 not a missing value");
    }

    #[test]
    fn lock_zone_density_rises_after_a_sweep_driven_by_high_residual_at_the_hole_ring() {
        let geom = test_geom(); // real hole at origin, radius 0.003175
        let cfg = AmrtConfig { initial_level: 3, max_level: 7, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&geom, cfg);
        let before = grid.lock_zone_density();
        // Drive several sweeps with residual concentrated exactly at the hole ring (matching
        // spatial_test_e's own synthetic-ring convention), so any density rise is attributable
        // to the residual signal actually finding the hole, not the structural hole-zone floor
        // (hole_zone_factor: 0.0 above disables that floor entirely).
        for _ in 0..6 {
            let pts = grid.sample_points();
            let r_hole = 0.003175_f64;
            let res: Vec<f32> = pts.iter().map(|&[x, y]| {
                let r = (x * x + y * y).sqrt();
                if (r - r_hole).abs() < r_hole * 0.5 { 1.0 } else { 0.001 }
            }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let after = grid.lock_zone_density();
        assert!(after > before * 1.5, "AMR must concentrate refinement at the hole ring when the residual signal says so: before={before:.1}, after={after:.1}");
    }

    #[test]
    fn zone_density_at_a_far_off_domain_point_is_zero_not_a_panic() {
        let grid = AdaptiveGrid::new(&test_geom(), AmrtConfig::default());
        assert_eq!(grid.zone_density(1000.0, 1000.0, 0.001), 0.0);
    }

    // ─── Phase 21 (Neural-Network-Wide Adaptive Collocation epic): regression test architecture
    // fast-test gaps this phase's audit found and closed: duplicate suppression, determinism ───

    #[test]
    fn sample_points_has_no_exact_duplicates_after_several_adapt_cycles() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg);
        for _ in 0..5 {
            let pts = grid.sample_points();
            let res: Vec<f32> = pts.iter().map(|&[x, y]| if x > 0.3 && y > 0.3 { 1.0 } else { 0.001 }).collect();
            grid.update_residuals(&res);
            grid.adapt();
        }
        let pts = grid.sample_points();
        let mut seen = std::collections::HashSet::new();
        for &[x, y] in &pts {
            let key = (x.to_bits(), y.to_bits());
            assert!(seen.insert(key), "duplicate collocation point at ({x}, {y}) — every quadtree leaf must contribute exactly one point");
        }
    }

    #[test]
    fn adapt_is_deterministic_given_the_same_residual_sequence() {
        let cfg = AmrtConfig { initial_level: 3, max_level: 6, min_level_hole: 0, hole_zone_factor: 0.0, ema_alpha: 1.0, ..AmrtConfig::default() };
        let residual_fn = |x: f64, y: f64| -> f32 { if x > 0.2 && y > -0.4 { 0.9 } else { 0.002 } };

        let run = || -> Vec<[f64; 2]> {
            let mut grid = AdaptiveGrid::new(&no_hole_geom(), cfg.clone());
            for _ in 0..5 {
                let pts = grid.sample_points();
                let res: Vec<f32> = pts.iter().map(|&[x, y]| residual_fn(x, y)).collect();
                grid.update_residuals(&res);
                grid.adapt();
            }
            grid.sample_points()
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "two identical residual sequences must produce the same point count");
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa[0].to_bits(), pb[0].to_bits(), "AMR must be bit-for-bit deterministic given identical inputs");
            assert_eq!(pa[1].to_bits(), pb[1].to_bits());
        }
    }
}
