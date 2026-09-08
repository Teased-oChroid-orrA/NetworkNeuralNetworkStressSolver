//! Smart adaptive architecture (v1): a training-progress-driven controller deciding when to
//! grow/shrink an `ElasticityNet`'s depth or width. Pure decision logic - this module never
//! touches burn tensors or the network itself; the caller (`runner.rs`/`parametric_problem.rs`)
//! applies the returned [`ArchAction`] to the live model + optimizer state, mirroring the
//! rebuild pattern `headless.rs` already uses for `grow_width`.
//!
//! Heuristic v1, not claimed optimal - see the "Smart adaptive architecture" plan's design
//! section 2 for the full rationale behind each policy below.

use crate::controllers::{ConvergenceTracker, MetricDirection};

/// One architecture decision [`ArchitectureController::observe`] can request. The caller
/// applies exactly one of these per call that returns `Some`.
#[derive(Clone, Debug, PartialEq)]
pub enum ArchAction {
    /// Grow every hidden layer's width to this new `hidden_dim` (via `ElasticityNet::grow_width`).
    GrowWidth(usize),
    /// Append one dormant hidden layer at the end (via `ElasticityNet::append_dormant_layer`).
    GrowDepth,
    /// Remove the (already-dormant) layer at this index (via `ElasticityNet::remove_layer`).
    /// Never watched/reverted - a dormant gated block is provably zero-effect to remove.
    ShrinkDepth(usize),
    /// Undo the most recent `GrowWidth`/`GrowDepth`/`PruneWidth` - restore the kept
    /// pre-action snapshot. Never returned for `ShrinkDepth`.
    RevertLastChange,
    /// Remove these global hidden-unit indices from EVERY layer uniformly (via
    /// `ElasticityNet::prune_width`, which is a global op - it must remove the same indices
    /// from every layer to keep the gated residual sum shape-consistent, mirroring
    /// `grow_width`'s own global, not per-layer, contract).
    PruneWidth { drop_indices: Vec<usize> },
}

/// Tunables for [`ArchitectureController`]. All heuristic v1 defaults - see [`Self::v1`].
#[derive(Clone, Debug)]
pub struct ArchitectureConfig {
    pub max_hidden_dim: usize,
    pub max_n_hidden: usize,
    pub min_n_hidden: usize,
    /// `|alpha| <= gate_epsilon` counts as dormant for the shrink-depth check.
    pub gate_epsilon: f32,
    /// Consecutive dormant observations before a gate's layer becomes a `ShrinkDepth` candidate.
    pub dormancy_threshold: usize,
    /// Width growth multiplier applied to the current `hidden_dim` (e.g. 1.5 = +50%).
    pub width_growth_factor: f64,
    /// Observations to watch a speculative grow/prune action before judging it.
    pub post_action_watch_window: usize,
    /// Required relative residual improvement over the watch window to keep the action;
    /// anything less triggers `RevertLastChange`.
    pub post_action_improve_rel_eps: f64,
    /// Passed straight through to `ConvergenceTracker::for_metric`'s `plateau_rel_eps`.
    pub plateau_rel_eps: f64,
    /// Observations between opportunistic "is the network coasting?" prune checks.
    pub prune_period: usize,
    /// Fraction of a layer's neurons to prune when coasting (floor 1, capped by
    /// `prune_min_layer_width`).
    pub prune_fraction: f64,
    /// Never prune a layer below this width.
    pub prune_min_layer_width: usize,
    /// Residual must drop to at most this fraction of its historical peak (the highest
    /// residual seen so far, i.e. roughly the early-training scale) to count as "coasting" -
    /// deep in a converged regime rather than merely mid-improvement. Small (e.g. 0.15), not a
    /// multiple - a value >= 1.0 would make ordinary monotonic improvement register as
    /// "coasting" on every tick, since current is always below its own prior best by
    /// construction.
    pub coasting_factor: f64,
}

impl ArchitectureConfig {
    pub fn v1(max_hidden_dim: usize, max_n_hidden: usize) -> Self {
        Self {
            max_hidden_dim,
            max_n_hidden,
            min_n_hidden: 1,
            gate_epsilon: 1e-3,
            dormancy_threshold: 10,
            width_growth_factor: 1.5,
            post_action_watch_window: 5,
            post_action_improve_rel_eps: 0.02,
            plateau_rel_eps: 0.05,
            prune_period: 40,
            prune_fraction: 0.1,
            prune_min_layer_width: 8,
            coasting_factor: 0.15,
        }
    }
}

struct PendingWatch {
    residual_at_action: f64,
    ticks_elapsed: usize,
}

/// Stateful controller - one instance per training run, fed one observation per existing
/// vis-cadence tick (no new cadence introduced). Not `Clone`/`Copy`: owns a running
/// [`ConvergenceTracker`] and per-gate dormancy counters that must persist across calls.
pub struct ArchitectureController {
    config: ArchitectureConfig,
    residual_tracker: ConvergenceTracker,
    dormancy_counts: Vec<usize>,
    watch: Option<PendingWatch>,
    ticks_since_prune_check: usize,
    /// Highest residual observed so far - the early-training scale a "coasting" comparison is
    /// measured against. `0.0` before any observation (no comparison is meaningful yet).
    residual_baseline_peak: f64,
}

impl ArchitectureController {
    pub fn new(config: ArchitectureConfig) -> Self {
        let plateau_rel_eps = config.plateau_rel_eps;
        Self {
            config,
            residual_tracker: ConvergenceTracker::for_metric(
                MetricDirection::SmallerIsBetter,
                plateau_rel_eps,
                f64::INFINITY, // crash detection unused by this controller
                0.0,           // no noise floor - residual RMS is always physically meaningful
            ),
            dormancy_counts: Vec::new(),
            watch: None,
            ticks_since_prune_check: 0,
            residual_baseline_peak: 0.0,
        }
    }

    /// Feed one observation. `awake_mask`/`layer_magnitudes` mirror the same live-model data
    /// already surfaced via `network_snapshot` (`awake_mask`, `layer_mean_abs_weight`) - no new
    /// physics probe needed. Returns at most one action; while a speculative grow/prune is
    /// being watched, only that watch's own resolution (keep silently, or `RevertLastChange`)
    /// is returned - dormancy/plateau/prune checks resume on the next call after it resolves.
    pub fn observe(
        &mut self,
        pde_rms: f64,
        hidden_dim: usize,
        n_hidden: usize,
        awake_mask: &[bool],
        layer_magnitudes: &[Vec<f32>],
    ) -> Option<ArchAction> {
        self.residual_tracker.push(pde_rms);
        let peak_before_this_reading = self.residual_baseline_peak;
        if pde_rms > self.residual_baseline_peak {
            self.residual_baseline_peak = pde_rms;
        }

        if self.dormancy_counts.len() != awake_mask.len() {
            self.dormancy_counts = vec![0; awake_mask.len()];
        }

        if let Some(mut watch) = self.watch.take() {
            watch.ticks_elapsed += 1;
            if watch.ticks_elapsed >= self.config.post_action_watch_window {
                let improved = watch.residual_at_action > 0.0
                    && (watch.residual_at_action - pde_rms) / watch.residual_at_action
                        >= self.config.post_action_improve_rel_eps;
                return if improved { None } else { Some(ArchAction::RevertLastChange) };
            }
            self.watch = Some(watch);
            return None;
        }

        // Dormancy -> shrink depth. Always allowed, provably zero-effect - see `ArchAction::
        // ShrinkDepth`'s doc comment - so it needs no watch/revert of its own.
        for (i, &awake) in awake_mask.iter().enumerate() {
            if awake {
                self.dormancy_counts[i] = 0;
            } else {
                self.dormancy_counts[i] += 1;
                if self.dormancy_counts[i] >= self.config.dormancy_threshold
                    && n_hidden > self.config.min_n_hidden
                {
                    self.dormancy_counts[i] = 0;
                    // gates[i] corresponds positionally to layers[i + 1] (see network.rs).
                    return Some(ArchAction::ShrinkDepth(i + 1));
                }
            }
        }

        if let Some(action) = self.check_plateau_growth(hidden_dim, n_hidden) {
            self.watch = Some(PendingWatch { residual_at_action: pde_rms, ticks_elapsed: 0 });
            return Some(action);
        }

        self.ticks_since_prune_check += 1;
        if self.ticks_since_prune_check >= self.config.prune_period {
            self.ticks_since_prune_check = 0;
            let coasting = peak_before_this_reading > 0.0
                && pde_rms <= peak_before_this_reading * self.config.coasting_factor;
            if coasting {
                if let Some(action) = self.plan_prune(layer_magnitudes) {
                    self.watch = Some(PendingWatch { residual_at_action: pde_rms, ticks_elapsed: 0 });
                    return Some(action);
                }
            }
        }

        None
    }

    fn check_plateau_growth(&mut self, hidden_dim: usize, n_hidden: usize) -> Option<ArchAction> {
        self.residual_tracker.check_plateau()?;
        if hidden_dim < self.config.max_hidden_dim {
            let grown = ((hidden_dim as f64) * self.config.width_growth_factor).ceil() as usize;
            Some(ArchAction::GrowWidth(grown.min(self.config.max_hidden_dim)))
        } else if n_hidden < self.config.max_n_hidden {
            Some(ArchAction::GrowDepth)
        } else {
            None
        }
    }

    /// Picks the globally lowest-average-magnitude hidden units and drops `prune_fraction` of
    /// them (floor 1), never pruning the network below `prune_min_layer_width` hidden units
    /// total. `ElasticityNet::prune_width` is a GLOBAL op (every layer shares one hidden_dim -
    /// see its doc comment), so there is no single "prunable layer" to pick; instead each
    /// hidden index's score is averaged across every entry in `layer_magnitudes` that has it
    /// (one entry per prunable layer, per `NetworkSnapshot::layer_mean_abs_weight`'s existing
    /// convention - all normally the same length, `hidden_dim`).
    fn plan_prune(&self, layer_magnitudes: &[Vec<f32>]) -> Option<ArchAction> {
        let hidden_dim = layer_magnitudes.iter().map(|m| m.len()).max().unwrap_or(0);
        if hidden_dim == 0 || hidden_dim <= self.config.prune_min_layer_width {
            return None;
        }
        let mut score_sum = vec![0.0f64; hidden_dim];
        let mut score_count = vec![0usize; hidden_dim];
        for mags in layer_magnitudes {
            for (i, &m) in mags.iter().enumerate() {
                score_sum[i] += m as f64;
                score_count[i] += 1;
            }
        }
        let mut indexed: Vec<(usize, f64)> = (0..hidden_dim)
            .filter(|&i| score_count[i] > 0)
            .map(|i| (i, score_sum[i] / score_count[i] as f64))
            .collect();
        if indexed.is_empty() {
            return None;
        }
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let max_droppable = hidden_dim.saturating_sub(self.config.prune_min_layer_width);
        let drop_count = ((hidden_dim as f64 * self.config.prune_fraction).floor() as usize)
            .max(1)
            .min(max_droppable)
            .min(indexed.len());
        if drop_count == 0 {
            return None;
        }
        let mut drop_indices: Vec<usize> = indexed.into_iter().take(drop_count).map(|(i, _)| i).collect();
        drop_indices.sort_unstable();
        Some(ArchAction::PruneWidth { drop_indices })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn awake_all(n: usize) -> Vec<bool> {
        vec![true; n]
    }

    #[test]
    fn plateaued_residual_triggers_grow_width() {
        let config = ArchitectureConfig::v1(128, 4);
        let mut ctl = ArchitectureController::new(config);
        // Feed a residual sequence that improves fast then flatlines - needs 2*PLATEAU_WINDOW
        // (40) readings total per controllers.rs's ConvergenceTracker::check_plateau.
        let mut action = None;
        for step in 0..60 {
            let rms = if step < 20 { 10.0 - step as f64 * 0.4 } else { 2.0 };
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &[]);
            if a.is_some() {
                action = a;
                break;
            }
        }
        assert_eq!(action, Some(ArchAction::GrowWidth(48)));
    }

    #[test]
    fn grow_then_no_improvement_triggers_revert() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.post_action_watch_window = 3;
        config.post_action_improve_rel_eps = 0.5; // demand a big improvement, easy to fail
        let mut ctl = ArchitectureController::new(config);
        for step in 0..60 {
            let rms = if step < 20 { 10.0 - step as f64 * 0.4 } else { 2.0 };
            if ctl.observe(rms, 32, 2, &awake_all(1), &[]).is_some() {
                break;
            }
        }
        // Watch window: residual stays flat at 2.0 - far short of the demanded 50% cut.
        let mut reverted = false;
        for _ in 0..3 {
            if ctl.observe(2.0, 48, 2, &awake_all(1), &[]) == Some(ArchAction::RevertLastChange) {
                reverted = true;
                break;
            }
        }
        assert!(reverted, "no-improvement watch window must resolve to RevertLastChange");
    }

    #[test]
    fn grow_then_real_improvement_is_kept_silently() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.post_action_watch_window = 3;
        config.post_action_improve_rel_eps = 0.1;
        let mut ctl = ArchitectureController::new(config);
        for step in 0..60 {
            let rms = if step < 20 { 10.0 - step as f64 * 0.4 } else { 2.0 };
            if ctl.observe(rms, 32, 2, &awake_all(1), &[]).is_some() {
                break;
            }
        }
        // Residual drops well past the 10% bar during the watch window.
        let mut saw_action = None;
        for _ in 0..3 {
            saw_action = ctl.observe(1.0, 48, 2, &awake_all(1), &[]);
        }
        assert_eq!(saw_action, None, "an improving watch window must not revert");
    }

    #[test]
    fn sustained_dormancy_triggers_shrink_depth() {
        let config = ArchitectureConfig::v1(128, 4);
        let mut ctl = ArchitectureController::new(config.clone());
        let mask = vec![true, false, true]; // gate index 1 is dormant
        let mut action = None;
        for _ in 0..config.dormancy_threshold {
            action = ctl.observe(5.0, 32, 4, &mask, &[]);
        }
        // gates[1] -> layers[2]
        assert_eq!(action, Some(ArchAction::ShrinkDepth(2)));
    }

    #[test]
    fn shrink_depth_never_fires_below_min_n_hidden() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.min_n_hidden = 2;
        let mut ctl = ArchitectureController::new(config.clone());
        let mask = vec![false]; // one gate, dormant
        let mut action = None;
        for _ in 0..(config.dormancy_threshold + 5) {
            action = ctl.observe(5.0, 32, 2, &mask, &[]); // n_hidden == min_n_hidden
        }
        assert_eq!(action, None, "must not shrink below min_n_hidden");
    }

    #[test]
    fn caps_prevent_growth_action_even_under_plateau() {
        let config = ArchitectureConfig::v1(32, 2); // hidden_dim and n_hidden both already at cap
        let mut ctl = ArchitectureController::new(config);
        let mut action = None;
        for step in 0..60 {
            let rms = if step < 20 { 10.0 - step as f64 * 0.4 } else { 2.0 };
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &[]);
            if a.is_some() {
                action = a;
            }
        }
        assert_eq!(action, None, "both caps already reached - no growth action should ever fire");
    }

    #[test]
    fn plateau_falls_back_to_grow_depth_once_width_is_capped() {
        let config = ArchitectureConfig::v1(32, 4); // hidden_dim capped, depth is not
        let mut ctl = ArchitectureController::new(config);
        let mut action = None;
        for step in 0..60 {
            let rms = if step < 20 { 10.0 - step as f64 * 0.4 } else { 2.0 };
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &[]);
            if a.is_some() {
                action = a;
                break;
            }
        }
        assert_eq!(action, Some(ArchAction::GrowDepth));
    }

    #[test]
    fn coasting_network_triggers_prune_of_lowest_magnitude_neurons() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.prune_period = 5;
        config.prune_fraction = 0.25;
        config.prune_min_layer_width = 2;
        let mut ctl = ArchitectureController::new(config.clone());
        // layer 0 has one clearly-lowest neuron (index 2); layer 1 has a higher floor.
        let magnitudes = vec![vec![1.0, 0.9, 0.01, 0.8], vec![2.0, 2.1, 2.2, 2.3]];
        let mut action = None;
        for step in 0..config.prune_period {
            // First reading sets a high peak; the rest drop to a small fraction of it,
            // satisfying the "coasting" check (residual well below its historical peak) on
            // every subsequent tick, including the triggering one.
            let rms = if step == 0 { 10.0 } else { 1.0 };
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &magnitudes);
            if step == config.prune_period - 1 {
                action = a;
            }
        }
        assert_eq!(action, Some(ArchAction::PruneWidth { drop_indices: vec![2] }));
    }

    #[test]
    fn prune_never_takes_a_layer_below_its_minimum_width() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.prune_period = 3;
        config.prune_fraction = 0.9; // would drop almost everything without the floor
        config.prune_min_layer_width = 3;
        let mut ctl = ArchitectureController::new(config.clone());
        let magnitudes = vec![vec![0.1, 0.2, 0.3, 0.4]]; // width 4, floor 3 -> at most 1 droppable
        let mut action = None;
        for step in 0..config.prune_period {
            let rms = if step == 0 { 10.0 } else { 1.0 }; // peak, then a coasting-low tail
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &magnitudes);
            if step == config.prune_period - 1 {
                action = a;
            }
        }
        assert_eq!(action, Some(ArchAction::PruneWidth { drop_indices: vec![0] }));
    }

    #[test]
    fn no_coasting_signal_means_no_prune() {
        let mut config = ArchitectureConfig::v1(128, 4);
        config.prune_period = 3;
        let mut ctl = ArchitectureController::new(config.clone());
        let magnitudes = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]];
        let mut action = None;
        // Residual keeps dropping - never "coasting" relative to its own historical best.
        for step in 0..config.prune_period {
            let rms = 10.0 - step as f64;
            let a = ctl.observe(rms, 32, 2, &awake_all(1), &magnitudes);
            if step == config.prune_period - 1 {
                action = a;
            }
        }
        assert_eq!(action, None);
    }
}
