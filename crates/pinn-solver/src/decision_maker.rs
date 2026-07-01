/// Three-tier meta-optimizer decision maker.
///
/// Drives dynamic switching between:
///   Tier 1 (Explore)  — AdamW weights         (Phase 1, low gradient conflict)
///   Tier 2 (Align)    — SOAP-Muon weights      (Phase 1 high conflict; Phase 2 default)
///   Tier 3 (Converge) — L-BFGS closure         (Phase 2, gradient near-stasis)
///
/// Architecture invariant: K_t is a post-hoc verification metric only. This module
/// never reads K_t, ConvergenceTracker history, or any analytical baseline — transitions
/// are driven entirely by gradient-signal metrics (cosine similarity + gradient norm) so
/// the decision maker is equally valid for problems without a closed-form solution.

use pinn_core::messages::DecisionMakerConfig;

/// Active optimizer tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerTier {
    Explore,
    Align,
    Converge,
}

impl OptimizerTier {
    pub fn as_u8(self) -> u8 {
        match self {
            OptimizerTier::Explore  => 0,
            OptimizerTier::Align    => 1,
            OptimizerTier::Converge => 2,
        }
    }
}

/// Gradient conflict measurement from dual backward passes.
///
/// Physics group:  e_loss + eq_loss + const_loss
/// BC group:       n_loss + h_loss + d_loss + w_neumann
#[derive(Clone, Copy, Debug)]
pub struct GradientConflict {
    pub cosine_sim:  f32,
    pub g_pde_norm:  f32,
    pub g_bc_norm:   f32,
}

/// Emitted by `PinnDecisionMaker::evaluate` when a tier change is warranted.
#[derive(Debug)]
pub struct TierTransition {
    pub new_tier:    OptimizerTier,
    /// Caller should recreate `WeightOptim` with the variant matching `new_tier`.
    pub reset_optim: bool,
    /// Caller should call `lr_sched.reset_for_phase2()`.
    pub reset_lr:    bool,
}

/// State machine that decides which optimizer tier is active.
pub struct PinnDecisionMaker {
    pub config:       DecisionMakerConfig,
    pub current_tier: OptimizerTier,
    steps_in_tier:    usize,
    step_counter:     usize,
}

impl PinnDecisionMaker {
    /// Create a new decision maker.
    ///
    /// `phase2_active`: if true, start in Align (Phase 2 default); otherwise start in Explore.
    pub fn new(config: DecisionMakerConfig, phase2_active: bool) -> Self {
        let current_tier = if phase2_active {
            OptimizerTier::Align
        } else {
            OptimizerTier::Explore
        };
        Self {
            config,
            current_tier,
            steps_in_tier: 0,
            step_counter:  0,
        }
    }

    /// Advance internal step counter. Returns `true` when the caller should compute
    /// the gradient conflict metric and call `evaluate()`.
    pub fn advance(&mut self) -> bool {
        self.step_counter  += 1;
        self.steps_in_tier += 1;
        self.config.enabled && (self.step_counter % self.config.check_interval == 0)
    }

    /// Evaluate whether a tier transition is warranted.
    ///
    /// `conflict`: exact dual-pass result, or `None` when `use_exact_cosine = false`.
    /// `proxy_ratio`: `(e_scalar + eq_scalar + const_scalar) / (n_scalar + h_scalar + d_scalar + w_scalar + 1e-8)`.
    /// `phase2_active`: whether Phase 2 has started.
    ///
    /// Returns `Some(TierTransition)` when the tier should change; internally updates
    /// `current_tier` and resets `steps_in_tier`.
    pub fn evaluate(
        &mut self,
        conflict: Option<GradientConflict>,
        proxy_ratio: f32,
        phase2_active: bool,
    ) -> Option<TierTransition> {
        if !self.config.enabled {
            return None;
        }

        // Minimum dwell before any transition is allowed.
        if self.steps_in_tier < self.config.min_dwell_steps {
            return None;
        }

        // Effective cosine: prefer exact; fall back to proxy mapping.
        let effective_cosine: f32 = if let Some(c) = conflict {
            c.cosine_sim
        } else {
            // Map proxy_ratio ∈ [0,∞) → [-1, 1] via 2/(1+r) - 1.
            // proxy_ratio ≈ 1 → 0 (balanced); >> 1 → -1 (physics dominates, high conflict).
            2.0 / (1.0 + proxy_ratio.max(1.0)) - 1.0
        };

        let g_total_norm: Option<f32> = conflict.map(|c| c.g_pde_norm + c.g_bc_norm);

        let transition = match (self.current_tier, phase2_active) {
            // Phase 1: Explore ↔ Align
            (OptimizerTier::Explore, false) => {
                if effective_cosine < self.config.conflict_threshold {
                    Some(TierTransition {
                        new_tier:    OptimizerTier::Align,
                        reset_optim: true,
                        reset_lr:    false,
                    })
                } else {
                    None
                }
            }
            (OptimizerTier::Align, false) => {
                if effective_cosine > self.config.alignment_threshold {
                    Some(TierTransition {
                        new_tier:    OptimizerTier::Explore,
                        reset_optim: true,
                        reset_lr:    false,
                    })
                } else {
                    None
                }
            }
            // Explore is locked out in Phase 2 — force to Align immediately.
            (OptimizerTier::Explore, true) => {
                Some(TierTransition {
                    new_tier:    OptimizerTier::Align,
                    reset_optim: true,
                    reset_lr:    false,
                })
            }
            // Phase 2: Align → Converge when gradients are well-aligned AND small.
            (OptimizerTier::Align, true) => {
                if let Some(gnorm) = g_total_norm {
                    if effective_cosine > self.config.converge_cosine_min
                        && gnorm < self.config.converge_grad_threshold
                    {
                        Some(TierTransition {
                            new_tier:    OptimizerTier::Converge,
                            reset_optim: false,
                            reset_lr:    true,
                        })
                    } else {
                        None
                    }
                } else {
                    // Without exact gradient norms, Converge entry is disallowed.
                    None
                }
            }
            // Phase 2: Converge → Align on conflict re-emergence or gradient spike.
            (OptimizerTier::Converge, true) => {
                let conflict_reemerged = effective_cosine < 0.50;
                let grad_spike = g_total_norm.map_or(false, |gnorm| {
                    gnorm > 3.0 * self.config.converge_grad_threshold
                });
                if conflict_reemerged || grad_spike {
                    Some(TierTransition {
                        new_tier:    OptimizerTier::Align,
                        reset_optim: false,
                        reset_lr:    false,
                    })
                } else {
                    None
                }
            }
            // Converge in Phase 1 shouldn't happen; treat same as Align fallback.
            (OptimizerTier::Converge, false) => {
                Some(TierTransition {
                    new_tier:    OptimizerTier::Align,
                    reset_optim: false,
                    reset_lr:    false,
                })
            }
        };

        if let Some(ref t) = transition {
            self.current_tier  = t.new_tier;
            self.steps_in_tier = 0;
        }

        transition
    }
}
