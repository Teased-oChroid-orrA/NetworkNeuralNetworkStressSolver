/// Learning-rate schedule: linear warmup → Adam + ReduceLROnPlateau → cosine annealing.

/// Warmup starts at this fraction of peak LR (not zero, so early steps still make progress).
const WARMUP_START_FRACTION: f64 = 0.01;
/// ReduceLROnPlateau multiplicative decay factor applied on each plateau trigger.
const PLATEAU_DECAY_FACTOR: f64 = 0.7;
/// LR floor — prevents cosine annealing from draining the LR to a useless range.
const MIN_LR: f64 = 4e-5;
/// Cosine annealing engages once LR drops to this fraction of peak.
const COSINE_TRIGGER_FRACTION: f64 = 0.1;
/// Cosine cycle length in steps — longer cycles give more time per LR level.
const COSINE_PERIOD_STEPS: usize = 1000;
/// SGDR warm-restart amplitude growth per cycle (re-amplifies LR so it doesn't stay trapped low).
const SGDR_AMPLITUDE_GROWTH: f64 = 1.5;
/// Cap on the SGDR-grown cycle amplitude, as a fraction of peak LR.
const SGDR_AMPLITUDE_CAP_FRACTION: f64 = 0.5;
/// Phase-2 LR reset target, as a fraction of peak — BCs are converged, kt_loss needs
/// immediate traction so it doesn't restart from the (much lower) warmup floor.
const PHASE2_RESET_LR_FRACTION: f64 = 0.5;

pub struct LrSchedule {
    // Warmup phase
    warmup_steps: usize,
    warmup_start: f64,
    peak_lr:      f64,

    // ReduceLROnPlateau
    current_lr:   f64,
    best_loss:    f64,
    patience:     usize,
    patience_cnt: usize,
    factor:       f64,
    min_lr:       f64,

    // Cosine annealing (activated when lr drops below cosine_trigger)
    cosine_trigger: f64,
    cosine_active:  bool,
    cosine_start_lr: f64,
    cosine_start_step: usize,
    cosine_period:  usize,

    step: usize,
}

impl LrSchedule {
    pub fn new(peak_lr: f64, warmup_steps: usize, patience: usize) -> Self {
        Self {
            warmup_steps,
            warmup_start:     peak_lr * WARMUP_START_FRACTION,
            peak_lr,
            current_lr:       peak_lr * WARMUP_START_FRACTION,
            best_loss:        f64::MAX,
            patience,
            patience_cnt:     0,
            factor:           PLATEAU_DECAY_FACTOR,
            min_lr:           MIN_LR,
            cosine_trigger:   peak_lr * COSINE_TRIGGER_FRACTION,
            cosine_active:    false,
            cosine_start_lr:  0.0,
            cosine_start_step: 0,
            cosine_period:    COSINE_PERIOD_STEPS,
            step: 0,
        }
    }

    /// Call once per training step with the current total loss.
    /// Returns the learning rate to use for this step.
    pub fn step(&mut self, loss: f64) -> f64 {
        let s = self.step;
        self.step += 1;

        // 1. Warmup phase
        if s < self.warmup_steps {
            let t = s as f64 / self.warmup_steps as f64;
            self.current_lr = self.warmup_start + t * (self.peak_lr - self.warmup_start);
            return self.current_lr;
        }

        // 2. Cosine annealing with SGDR restarts (re-amplify each cycle so LR doesn't trap low).
        if self.cosine_active {
            let elapsed = s - self.cosine_start_step;
            let cycle   = elapsed / self.cosine_period;
            let phase   = elapsed % self.cosine_period;

            // On each new cycle, raise the amplitude back toward peak (SGDR warm restart).
            let cycle_start_lr = (self.cosine_start_lr * SGDR_AMPLITUDE_GROWTH.powi(cycle as i32))
                .min(self.peak_lr * SGDR_AMPLITUDE_CAP_FRACTION);

            let t = std::f64::consts::PI * phase as f64 / self.cosine_period as f64;
            self.current_lr = self.min_lr
                + 0.5 * (cycle_start_lr - self.min_lr) * (1.0 + t.cos());
            return self.current_lr;
        }

        // 3. ReduceLROnPlateau
        if loss < self.best_loss {
            self.best_loss = loss;
            self.patience_cnt = 0;
        } else {
            self.patience_cnt += 1;
            if self.patience_cnt >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.patience_cnt = 0;

                // Trigger cosine annealing if LR drops below threshold
                if self.current_lr <= self.cosine_trigger {
                    self.cosine_active = true;
                    self.cosine_start_lr = self.current_lr;
                    self.cosine_start_step = s;
                }
            }
        }

        self.current_lr
    }

    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Reset the plateau-detection state shared by both reset paths below.
    fn reset_plateau_state(&mut self) {
        self.best_loss = f64::MAX;
        self.patience_cnt = 0;
        self.cosine_active = false;
    }

    /// Reset for warm-start (load change only)
    pub fn reset_for_warmstart(&mut self) {
        self.step = 0;
        self.current_lr = self.cosine_trigger;
        self.reset_plateau_state();
    }

    /// Reset for Phase 2 of two-phase curriculum.
    /// Starts at `PHASE2_RESET_LR_FRACTION` of peak LR — BCs are converged, kt_loss needs
    /// immediate traction. Skips warmup (step set past warmup boundary) and resets plateau
    /// detection.
    pub fn reset_for_phase2(&mut self) {
        self.step = self.warmup_steps + 1;
        self.current_lr = self.peak_lr * PHASE2_RESET_LR_FRACTION;
        self.reset_plateau_state();
    }
}
