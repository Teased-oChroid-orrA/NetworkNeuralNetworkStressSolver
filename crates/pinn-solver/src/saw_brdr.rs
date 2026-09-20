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
    /// Domain-informed base weights (from engine). Not mutated by `SawBrdr` itself after
    /// init — `update()` only ever reads this field. Issue #77: a caller MAY still
    /// deliberately recalibrate specific entries between steps via [`SawBrdr::set_base_weights`]
    /// (e.g. [`grad_norm_damping_factors`]'s own per-term rescale) as a supported, opt-in use;
    /// this does not disturb `update()`'s own math, since `effective_weights()` re-reads
    /// `base_weights` fresh on every call rather than caching a snapshot from construction.
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

/// Issue #77 root-cause synthesis (see `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md` PH4-24
/// through PH4-35, and the investigation-branch plan for this fix): every one of twelve tested
/// remedies for the L5 Kt gap changed loss VALUE (representation, weight, sampling, LR,
/// residual choice) while `SawBrdr::update` above only ever reacts to loss-VALUE decay rate,
/// never to a term's actual backpropagated GRADIENT magnitude. `training_core::
/// gradient_share_report`/`term_grad_norms` already compute the real per-term gradient L2 norm
/// (opt-in, `MultiStepCtx::probe_term_gradients`) — but as a diagnostic only, never fed back
/// into `base_weights`. PH4-34b's own real A/B test proved the two quantities are only loosely
/// coupled: cutting `hole_free`'s nominal weight 10x moved its measured gradient SHARE the
/// WRONG direction (78.4% -> 82.4%), because gradient share is driven by an intrinsic
/// gradient-magnitude disparity between a pointwise boundary term and a domain-integrated
/// energy term, not by nominal weight.
///
/// This function closes that gap directly: given a step's real measured `term_grad_norms`,
/// compute a DAMPING-ONLY multiplicative factor for every term whose gradient exceeds the
/// `reference_terms`' own mean gradient norm — the reference terms are the pinned
/// physical-functional terms (`physical_potential`/`annulus_potential` for
/// `AnnularDecompositionProblem`) whose live SAW-BRDR coefficient `step_physics_multi`'s own
/// match arm already hardcodes to `1.0` regardless of `base_weights` (see that match's own
/// comment: "Keep its live coefficient fixed at the canonical unit scale"). Rescaling a
/// reference term's OWN base weight is therefore a structural no-op downstream — this function
/// does not need to special-case excluding them from the input, only from being assigned a
/// factor, since `factor_i` is only ever computed for names NOT in `reference_terms`.
///
/// Never amplifies (every factor is `<= 1.0`, clamped to `[floor, 1.0]`) — this only ever
/// pulls a structurally-privileged term's effective gradient contribution DOWN toward the
/// reference terms' own scale, never pushes another term up past its existing nominal weight.
/// Bounding below at `floor` (not `0.0`) keeps every term at least minimally active, avoiding
/// the "fully silence a term" failure mode a naive `factor=0` would risk.
///
/// A caller applies the returned factors via `saw.set_base_weights(...)`, multiplying each
/// non-exempt term's ORIGINAL base weight (not the previous step's already-rescaled one, to
/// avoid compounding shrink across repeated refreshes) by its factor here — see
/// `user_runner::run_annular_decomposition_training_inner`'s periodic refresh loop.
pub fn grad_norm_damping_factors(
    term_grad_norms: &std::collections::HashMap<&'static str, f32>,
    reference_terms: &[&str],
    floor: f32,
) -> std::collections::HashMap<&'static str, f32> {
    let refs: Vec<f32> = reference_terms.iter()
        .filter_map(|n| term_grad_norms.get(*n).copied())
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if refs.is_empty() {
        // No reference term measured this step (e.g. neither name present/nonzero) — nothing
        // to calibrate against; returning empty means every term keeps its current weight.
        return std::collections::HashMap::new();
    }
    let target: f32 = refs.iter().sum::<f32>() / refs.len() as f32;
    term_grad_norms.iter()
        .filter(|(name, _)| !reference_terms.contains(name))
        .filter_map(|(&name, &norm)| {
            if !norm.is_finite() || norm <= 0.0 {
                return None;
            }
            Some((name, (target / norm).clamp(floor, 1.0)))
        })
        .collect()
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

    /// Reproduces PH4-35's own real gradient-share numbers (a dominant `interface_traction_
    /// continuity` at 62.8% vs `annulus_potential` at 18.2%) as a synthetic fixture: the
    /// dominant term must be damped below 1.0, the reference term itself must never be
    /// assigned a factor at all (rescaling it is a structural no-op downstream, so it's
    /// intentionally absent from the map, not present-and-1.0), and an unrelated small term
    /// (already below the reference) must not be damped either.
    #[test]
    fn grad_norm_damping_factors_damps_the_dominant_term_and_exempts_the_reference() {
        let mut norms = std::collections::HashMap::new();
        norms.insert("annulus_potential", 0.20_f32); // reference
        norms.insert("interface_traction_continuity", 0.70_f32); // dominant, must be damped
        norms.insert("hole_free", 0.05_f32); // already small, must not be amplified

        let factors = grad_norm_damping_factors(&norms, &["annulus_potential", "physical_potential"], 0.1);

        assert!(!factors.contains_key("annulus_potential"),
            "reference term must never be assigned a rescale factor: {factors:?}");
        let dominant = factors["interface_traction_continuity"];
        assert!((dominant - (0.20 / 0.70)).abs() < 1e-6, "expected exact damping ratio, got {dominant}");
        assert!(dominant < 1.0, "dominant term must be damped below 1.0: {dominant}");
        let small = factors["hole_free"];
        assert!((small - 1.0).abs() < 1e-6, "a term already at/below the reference must not be amplified past 1.0: {small}");
    }

    /// Damping never amplifies and never fully silences a term — `floor` bounds how far a
    /// single refresh can shrink an extremely gradient-dominant term.
    #[test]
    fn grad_norm_damping_factors_are_bounded_below_by_floor_and_above_by_one() {
        let mut norms = std::collections::HashMap::new();
        norms.insert("physical_potential", 0.01_f32); // reference, tiny
        norms.insert("interface_traction_continuity", 1000.0_f32); // wildly dominant

        let factors = grad_norm_damping_factors(&norms, &["physical_potential"], 0.1);
        let f = factors["interface_traction_continuity"];
        assert!((f - 0.1).abs() < 1e-6, "factor must clamp at floor, not go arbitrarily small: {f}");
    }

    /// When no reference term is present in the reading (e.g. this step's active-term set
    /// didn't include it), the function must be a genuine no-op — empty map, not a panic or a
    /// spurious rescale derived from zero/missing data.
    #[test]
    fn grad_norm_damping_factors_is_noop_when_no_reference_term_present() {
        let mut norms = std::collections::HashMap::new();
        norms.insert("interface_traction_continuity", 0.70_f32);
        let factors = grad_norm_damping_factors(&norms, &["annulus_potential", "physical_potential"], 0.1);
        assert!(factors.is_empty(), "no reference term measured -> no factors computed: {factors:?}");
    }

    /// Non-finite/zero readings for a candidate term must be skipped (no factor entry), not
    /// produce Infinity/NaN/zero-division garbage.
    #[test]
    fn grad_norm_damping_factors_skips_non_finite_or_zero_candidate_readings() {
        let mut norms = std::collections::HashMap::new();
        norms.insert("annulus_potential", 0.20_f32);
        norms.insert("hole_free", f32::NAN);
        norms.insert("equilibrium", 0.0_f32);
        let factors = grad_norm_damping_factors(&norms, &["annulus_potential"], 0.1);
        assert!(!factors.contains_key("hole_free"), "NaN reading must not produce a factor: {factors:?}");
        assert!(!factors.contains_key("equilibrium"), "zero reading must not produce a factor: {factors:?}");
    }
}
