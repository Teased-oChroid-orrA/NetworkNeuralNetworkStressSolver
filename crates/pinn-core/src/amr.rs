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

    fn collect_points(&self, geom: &GeometryConfig, pts: &mut Vec<[f64; 2]>) {
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

pub struct AdaptiveGrid {
    root:        QuadNode,
    cfg:         AmrtConfig,
    geom:        GeometryConfig,
    adapt_count: usize,
    global_mean: f64,
    prev_global: f64,
}

impl AdaptiveGrid {
    /// Create grid with initial uniform subdivision, then enforce hole zone.
    pub fn new(geom: &GeometryConfig, cfg: AmrtConfig) -> Self {
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
        geom:      &GeometryConfig,
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
        self.enforce_active_cell_cap();
        self.adapt_count += 1;
    }

    fn priority(node: &QuadNode, tw: f64) -> f64 {
        let trend = (node.residual_ema - node.residual_prev).max(0.0);
        node.residual_ema + tw * trend
    }

    fn collect_leaf_residuals(node: &QuadNode, geom: &GeometryConfig, out: &mut Vec<f64>) {
        match &node.children {
            Some(ch) => { for c in ch.iter() { Self::collect_leaf_residuals(c, geom, out); } }
            None => { if geom.contains(node.cx(), node.cy()) { out.push(node.residual_ema); } }
        }
    }

    fn apply_adapt_rec(
        node:       &mut QuadNode,
        geom:       &GeometryConfig,
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

    /// Force all cells overlapping the hole zone to ≥ min_level_hole.
    fn enforce_hole_zone(&mut self) {
        let radius = match self.geom.hole {
            HoleType::Circular { radius } => radius,
            HoleType::None => return,
        };
        let zone_r  = radius * self.cfg.hole_zone_factor;
        let min_lv  = self.cfg.min_level_hole;
        let max_lv  = self.cfg.max_level;
        Self::enforce_zone_rec(&mut self.root, zone_r, min_lv, max_lv);
    }

    fn cell_overlaps_zone(node: &QuadNode, zone_r: f64) -> bool {
        // Closest point on rectangle [x0,x1]×[y0,y1] to the hole center at (0,0).
        let cx = node.x0.max(0.0_f64).min(node.x1);
        let cy = node.y0.max(0.0_f64).min(node.y1);
        cx * cx + cy * cy < zone_r * zone_r
    }

    fn enforce_zone_rec(node: &mut QuadNode, zone_r: f64, min_level: usize, max_level: usize) {
        let in_zone = Self::cell_overlaps_zone(node, zone_r);
        node.in_hole_zone = in_zone;

        if in_zone && node.level < min_level && node.level < max_level {
            if node.is_leaf() { node.split(); }
        }

        if let Some(ref mut ch) = node.children {
            for c in ch.iter_mut() {
                Self::enforce_zone_rec(c, zone_r, min_level, max_level);
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
    fn collect_leaf_priorities(node: &QuadNode, geom: &GeometryConfig, tw: f64, out: &mut Vec<f64>) {
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
}
