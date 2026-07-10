/// Self-Adaptive Weights Based on Balanced Residual Decay Rate (SAW-BRDR)
///
/// Reference: arXiv:2407.01613 (2024)
///
/// Extension over the paper: maintains domain-informed BASE WEIGHTS (from the engine)
/// and adapts per-component MULTIPLIERS based on convergence rates.
///
/// Effective weight for term i = base_i × n × multiplier_i
///
/// When all terms converge at equal rate: multiplier_i = 1/n, effective = base_i. ✓
/// When term i converges slower: multiplier_i > 1/n, boosting that term. ✓

pub struct SawBrdr {
    /// Domain-informed base weights (from engine, never modified after init).
    pub base_weights: Vec<f32>,
    /// SAW-BRDR normalised multipliers (sum = 1, adapted each step).
    multipliers: Vec<f32>,
    /// Exponential moving average of inverse decay rates per component.
    decay_ema: Vec<f32>,
    /// EMA decay factor (β_w ≈ 0.95 → adapts over ~20 steps).
    beta_w: f32,
    prev_losses: Vec<Option<f32>>,
}

impl SawBrdr {
    /// Uniform initialisation with equal base weights (for backward compatibility).
    pub fn new(n_components: usize, beta_w: f32) -> Self {
        Self::with_base(vec![1.0; n_components], beta_w)
    }

    /// Initialise with domain-informed base weights (preferred — from engine).
    ///
    /// base_weights encode relative scale of each loss term; SAW-BRDR adapts
    /// multipliers on top to balance convergence rates automatically.
    pub fn with_base(base_weights: Vec<f32>, beta_w: f32) -> Self {
        let n = base_weights.len();
        Self {
            base_weights,
            multipliers:  vec![1.0 / n as f32; n],
            decay_ema:    vec![1.0; n],
            beta_w,
            prev_losses:  vec![None; n],
        }
    }

    /// Effective weights = base × (n × multiplier).
    ///
    /// Use these as the λ coefficients in the total loss.
    pub fn effective_weights(&self) -> Vec<f32> {
        let n = self.base_weights.len() as f32;
        self.base_weights.iter().zip(self.multipliers.iter())
            .map(|(&b, &m)| b * n * m)
            .collect()
    }

    /// Update multipliers given current scalar loss values per component.
    /// Returns effective weights (base × n × multiplier) for immediate use.
    pub fn update(&mut self, losses: &[f32]) -> Vec<f32> {
        let n = self.base_weights.len();
        assert_eq!(losses.len(), n, "loss count must match SAW-BRDR component count");

        for i in 0..n {
            let curr = losses[i].abs();
            // A non-finite reading (NaN/Inf) is skipped entirely, leaving `prev_losses[i]`/
            // `decay_ema[i]` at their last-known-finite values — self-healing: once the
            // reading returns to finite, adaptation resumes exactly as if the bad step never
            // happened. Storing a NaN unconditionally here (the old behavior) would poison
            // `decay_ema[i]` on the NEXT call via the `prev/curr` ratio, which poisons
            // `total_ema`, permanently freezing the ENTIRE multiplier-update block below
            // (not just this one term) since `total_ema > 1e-12` becomes false forever.
            if !curr.is_finite() {
                continue;
            }
            if let Some(prev) = self.prev_losses[i] {
                // Inverse decay rate: slow convergence → high irdr → high weight
                let rate = if curr > 1e-12 { (prev / curr).clamp(0.05, 20.0) } else { 1.0 };
                let irdr = 1.0 / rate;
                self.decay_ema[i] = self.beta_w * self.decay_ema[i]
                    + (1.0 - self.beta_w) * irdr;
            }
            self.prev_losses[i] = Some(curr);
        }

        // Normalise decay_ema so multipliers sum to 1
        let total_ema: f32 = self.decay_ema.iter().sum();
        if total_ema > 1e-12 {
            for i in 0..n {
                let m_ref = self.decay_ema[i] / total_ema;
                self.multipliers[i] = self.beta_w * self.multipliers[i]
                    + (1.0 - self.beta_w) * m_ref;
            }
            // Re-normalise multipliers to exactly sum = 1
            let m_sum: f32 = self.multipliers.iter().sum();
            if m_sum > 1e-12 {
                for m in self.multipliers.iter_mut() { *m /= m_sum; }
            }
        }

        self.effective_weights()
    }

    /// Reset adaptive state (for warm-start — base weights preserved).
    pub fn reset(&mut self) {
        let n = self.base_weights.len();
        self.multipliers  = vec![1.0 / n as f32; n];
        self.decay_ema    = vec![1.0; n];
        self.prev_losses  = vec![None; n];
    }

    /// Update base weights to a new set (for warm-start with new config).
    pub fn set_base_weights(&mut self, base: Vec<f32>) {
        assert_eq!(base.len(), self.base_weights.len());
        self.base_weights = base;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_with_nan_reading_on_first_call_does_not_panic_and_stays_uninitialized() {
        let mut saw = SawBrdr::with_base(vec![1.0, 1.0], 0.5);
        let out = saw.update(&[f32::NAN, 1.0]);
        assert!(out[0].is_finite() && out[1].is_finite(), "first-ever call being NaN must not corrupt effective_weights: {out:?}");
        let out2 = saw.update(&[2.0, 1.0]);
        assert!(out2.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn update_all_terms_nan_simultaneously_does_not_panic_or_poison_state() {
        let mut saw = SawBrdr::with_base(vec![1.0, 1.0], 0.5);
        saw.update(&[8.0, 8.0]);
        let out = saw.update(&[f32::NAN, f32::NAN]);
        assert!(out.iter().all(|w| w.is_finite()), "all-NaN reading must not poison effective_weights: {out:?}");
        let out2 = saw.update(&[4.0, 4.0]);
        assert!(out2.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn update_treats_infinite_reading_as_non_finite_and_skips_it_like_nan() {
        let mut saw = SawBrdr::with_base(vec![1.0, 1.0], 0.5);
        saw.update(&[8.0, 8.0]);
        let out = saw.update(&[f32::INFINITY, 8.0]);
        assert!(out.iter().all(|w| w.is_finite()), "+Infinity is non-finite and must be skipped exactly like NaN: {out:?}");
    }

    /// Mutation-testing-grade: hand-computed exact values (beta_w=0.5 for exact fractions)
    /// proving (1) a transient NaN doesn't freeze the OTHER term's adaptation, and (2) SawBrdr
    /// resumes adapting against its LAST FINITE reading, not a poisoned/reset one.
    #[test]
    fn update_resumes_adapting_against_last_finite_state_after_transient_nan() {
        let mut saw = SawBrdr::with_base(vec![1.0, 1.0], 0.5);

        saw.update(&[8.0, 8.0]);
        let w2 = saw.update(&[4.0, 8.0]);
        let expect2 = [0.9285714_f32, 1.0714286_f32];
        for (a, b) in w2.iter().zip(expect2) { assert!((a - b).abs() < 1e-4, "step2: {w2:?} vs {expect2:?}"); }

        let w3 = saw.update(&[f32::NAN, 8.0]);
        assert!(w3.iter().all(|w| w.is_finite()), "step3 (NaN injected): must stay finite: {w3:?}");
        let expect3 = [0.8928571_f32, 1.1071429_f32];
        for (a, b) in w3.iter().zip(expect3) { assert!((a - b).abs() < 1e-4, "step3: {w3:?} vs {expect3:?}"); }

        let w4 = saw.update(&[2.0, 8.0]);
        assert!(w4.iter().all(|w| w.is_finite()), "step4 (post-NaN recovery): must stay finite: {w4:?}");
        let expect4 = [0.831044_f32, 1.168956_f32];
        for (a, b) in w4.iter().zip(expect4) { assert!((a - b).abs() < 1e-4, "step4: {w4:?} vs {expect4:?}"); }

        // The assertion that actually distinguishes fixed-vs-buggy code: under the bug, decay_ema[0]
        // goes NaN at step 4 (poisoned by Some(NaN) stored at step 3), freezing the ENTIRE multiplier
        // update — so w4 would equal w3 EXACTLY. Assert genuine continued movement, not just finiteness.
        assert!((w4[0] - w3[0]).abs() > 1e-4, "weights must keep adapting after the NaN clears, not freeze: w3={w3:?} w4={w4:?}");
        assert!((w4[1] - w3[1]).abs() > 1e-4, "the OTHER (never-NaN) term must also keep adapting, not get dragged into a freeze: w3={w3:?} w4={w4:?}");
    }
}
