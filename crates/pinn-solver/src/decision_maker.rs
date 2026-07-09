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
    /// True for problems with no Kirsch-style Phase-1(BC)/Phase-2(physics) curriculum split
    /// (pin-in-lug). Unlocks Align→Converge entry and Converge's real exit logic WITHOUT
    /// touching `(Explore, true)`'s force-lock-to-Align semantics (only correct for a
    /// problem with a short BC-only Phase 1 to graduate out of). Every Kirsch call site
    /// passes `false` — `evaluate()`'s 6 original match arms stay byte-identical whenever
    /// this is `false`.
    allow_converge: bool,
}

impl PinnDecisionMaker {
    /// Create a new decision maker.
    ///
    /// `phase2_active`: if true, start in Align (Phase 2 default); otherwise start in Explore.
    /// `allow_converge`: see the field doc comment above — every Kirsch call site passes
    /// `false`; pin-in-lug passes `true`.
    pub fn new(config: DecisionMakerConfig, phase2_active: bool, allow_converge: bool) -> Self {
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
            allow_converge,
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
                // Pin-in-lug (allow_converge=true, no Phase-1/Phase-2 curriculum split):
                // Align -> Converge is checked FIRST, before the original hysteresis-exit
                // check, using the same gradient-alignment+magnitude gate Kirsch's own
                // (Align, true) arm uses. Falls through to the untouched hysteresis-exit
                // check below when this doesn't fire (or allow_converge is false).
                let converge_entry = self.allow_converge
                    && g_total_norm.is_some_and(|gnorm| {
                        effective_cosine > self.config.converge_cosine_min
                            && gnorm < self.config.converge_grad_threshold
                    });
                if converge_entry {
                    Some(TierTransition {
                        new_tier:    OptimizerTier::Converge,
                        reset_optim: false,
                        reset_lr:    true,
                    })
                } else if effective_cosine > self.config.alignment_threshold {
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
            // Converge without Kirsch's phase2_active flag: for pin-in-lug
            // (allow_converge=true), use the SAME real exit logic as Kirsch's
            // (Converge, true) arm — pin-in-lug has no phase2_active curriculum split, so
            // this is Converge's only reachable exit arm for it. For every Kirsch call site
            // (allow_converge=false), this "shouldn't happen" and unconditionally demotes,
            // exactly as before.
            (OptimizerTier::Converge, false) => {
                if self.allow_converge {
                    let conflict_reemerged = effective_cosine < 0.50;
                    let grad_spike = g_total_norm.is_some_and(|gnorm| {
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
                } else {
                    Some(TierTransition {
                        new_tier:    OptimizerTier::Align,
                        reset_optim: false,
                        reset_lr:    false,
                    })
                }
            }
        };

        if let Some(ref t) = transition {
            self.current_tier  = t.new_tier;
            self.steps_in_tier = 0;
        }

        transition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permissive_config() -> DecisionMakerConfig {
        let mut c = DecisionMakerConfig::default();
        c.enabled = true;
        c.min_dwell_steps = 0;
        c
    }

    #[test]
    fn evaluate_align_allow_converge_true_transitions_to_converge_on_favorable_gradients() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, true);
        dm.current_tier = OptimizerTier::Align;
        let conflict = GradientConflict { cosine_sim: 0.9, g_pde_norm: 1.0, g_bc_norm: 1.0 };
        let t = dm.evaluate(Some(conflict), 1.0, false).expect("must transition");
        assert_eq!(t.new_tier, OptimizerTier::Converge);
        assert!(!t.reset_optim);
        assert!(t.reset_lr);
        assert_eq!(dm.current_tier, OptimizerTier::Converge);
    }

    #[test]
    fn evaluate_align_allow_converge_false_falls_through_to_original_hysteresis_exit() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, false);
        dm.current_tier = OptimizerTier::Align;
        let conflict = GradientConflict { cosine_sim: 0.9, g_pde_norm: 1.0, g_bc_norm: 1.0 };
        let t = dm.evaluate(Some(conflict), 1.0, false).expect("must transition");
        assert_eq!(t.new_tier, OptimizerTier::Explore);
    }

    #[test]
    fn evaluate_converge_allow_converge_exits_on_conflict_reemergence() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, true);
        dm.current_tier = OptimizerTier::Converge;
        let conflict = GradientConflict { cosine_sim: 0.3, g_pde_norm: 0.1, g_bc_norm: 0.1 };
        let t = dm.evaluate(Some(conflict), 1.0, false).expect("must exit Converge");
        assert_eq!(t.new_tier, OptimizerTier::Align);
    }

    #[test]
    fn evaluate_converge_allow_converge_exits_on_grad_spike() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, true);
        dm.current_tier = OptimizerTier::Converge;
        let conflict = GradientConflict { cosine_sim: 0.9, g_pde_norm: 10.0, g_bc_norm: 10.0 }; // gnorm=20 > 3*5=15
        let t = dm.evaluate(Some(conflict), 1.0, false).expect("must exit Converge on grad spike");
        assert_eq!(t.new_tier, OptimizerTier::Align);
    }

    #[test]
    fn evaluate_converge_allow_converge_stays_when_stable() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, true);
        dm.current_tier = OptimizerTier::Converge;
        let conflict = GradientConflict { cosine_sim: 0.95, g_pde_norm: 0.5, g_bc_norm: 0.5 };
        assert!(dm.evaluate(Some(conflict), 1.0, false).is_none());
        assert_eq!(dm.current_tier, OptimizerTier::Converge);
    }

    #[test]
    fn evaluate_converge_allow_converge_false_unconditionally_demotes_regardless_of_gradient_state() {
        let mut dm = PinnDecisionMaker::new(permissive_config(), false, false);
        dm.current_tier = OptimizerTier::Converge;
        let conflict = GradientConflict { cosine_sim: 0.99, g_pde_norm: 0.01, g_bc_norm: 0.01 };
        let t = dm.evaluate(Some(conflict), 1.0, false).expect("Kirsch's (Converge,false) fallback always demotes");
        assert_eq!(t.new_tier, OptimizerTier::Align);
    }

    #[test]
    fn evaluate_kirsch_transition_table_unaffected_by_allow_converge_field_across_all_six_arms() {
        let cases: &[(OptimizerTier, bool, GradientConflict, Option<OptimizerTier>)] = &[
            (OptimizerTier::Explore,  false, GradientConflict{cosine_sim:0.1,g_pde_norm:1.0,g_bc_norm:1.0}, Some(OptimizerTier::Align)),
            (OptimizerTier::Explore,  false, GradientConflict{cosine_sim:0.9,g_pde_norm:1.0,g_bc_norm:1.0}, None),
            (OptimizerTier::Align,    false, GradientConflict{cosine_sim:0.1,g_pde_norm:1.0,g_bc_norm:1.0}, None),
            (OptimizerTier::Explore,  true,  GradientConflict{cosine_sim:0.9,g_pde_norm:1.0,g_bc_norm:1.0}, Some(OptimizerTier::Align)),
            (OptimizerTier::Align,    true,  GradientConflict{cosine_sim:0.9,g_pde_norm:1.0,g_bc_norm:1.0}, Some(OptimizerTier::Converge)),
            (OptimizerTier::Converge, true,  GradientConflict{cosine_sim:0.95,g_pde_norm:0.1,g_bc_norm:0.1}, None),
        ];
        for &(tier, phase2, conflict, expected) in cases {
            let mut dm = PinnDecisionMaker::new(permissive_config(), false, false);
            dm.current_tier = tier;
            let got = dm.evaluate(Some(conflict), 1.0, phase2).map(|t| t.new_tier);
            assert_eq!(got, expected, "arm (tier={tier:?}, phase2_active={phase2}) diverged from pre-existing behavior");
        }
    }
}
