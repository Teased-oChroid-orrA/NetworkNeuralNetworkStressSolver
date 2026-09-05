//! Hardware-adaptive execution, Phase 2: per-step profiling instrumentation
//! (wall time only in this phase - memory/numerical diagnostics are a later
//! phase's scope, not this one's).
//!
//! Gated by `SolverConfig::diagnostics.enabled` (default `false`), following
//! this codebase's established opt-in-subsystem convention
//! (`StiffnessConfig`/`DecisionMakerConfig`/`WidthGrowthConfig`). When
//! disabled, `training_core::step_physics` performs zero extra `Instant::
//! now()` calls and zero extra device syncs - genuinely zero-cost, not just
//! "cheap."
//!
//! **Why a device sync is unavoidable for honest numbers**: `burn`'s tensor
//! ops on a GPU backend are queued, not executed synchronously - naively
//! wrapping `Instant::now()` around `.backward()` without a sync in between
//! would measure "time to enqueue the op graph," not "time for the GPU to
//! actually finish computing it" (this is exactly what Phase 1's
//! `training_step` bench's suspiciously-flat tiny-vs-very_large timing
//! turned out to be evidence of - see `CLAUDE.md`). `Backend::sync(device)`
//! ("ensure all computation are finished," `burn-backend`'s own doc comment)
//! is called at each measurement boundary here for that reason. This makes
//! enabling diagnostics change *wall-clock timing* (a real, disclosed,
//! opt-in cost - the sync calls themselves take time) but never *computed
//! values* - inserting a wait for already-queued work to finish cannot alter
//! what that work computes.

use std::time::Instant;

use pinn_core::messages::DiagnosticsConfig;

use crate::training_core::BDevice;

/// Wall-clock cost of one `step_physics` call's three stages. `None` fields
/// don't exist - all three are always populated together when diagnostics
/// are enabled (there's no partial-instrumentation mode in Phase 2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepTiming {
    /// Forward pass(es) + SAW-BRDR loss assembly, up to (not including)
    /// `.backward()`.
    pub forward_us: u64,
    /// The single `.backward()` call.
    pub backward_us: u64,
    /// Gradient extraction + all three optimizer `.step()` calls (weights,
    /// biases, gates).
    pub optimizer_us: u64,
}

/// Internal helper `step_physics` uses to collect the three checkpoints
/// without branching on `enabled` at every single call site - construct
/// once at function entry, call `checkpoint` at each of the three stage
/// boundaries, call `finish` to get the `Option<StepTiming>` for
/// `StepOutput`. When `enabled=false`, `checkpoint`/`finish` never call
/// `Instant::now()` or `B::sync` at all.
pub struct StepTimer {
    enabled: bool,
    last: Option<Instant>,
    forward_us: u64,
    backward_us: u64,
}

impl StepTimer {
    pub fn start(config: &DiagnosticsConfig, device: &BDevice) -> Self {
        if !config.enabled {
            return Self { enabled: false, last: None, forward_us: 0, backward_us: 0 };
        }
        crate::training_core::sync_device(device);
        Self { enabled: true, last: Some(Instant::now()), forward_us: 0, backward_us: 0 }
    }

    /// Call after forward+loss-assembly is done (right before `.backward()`).
    pub fn mark_forward_done(&mut self, device: &BDevice) {
        if !self.enabled {
            return;
        }
        crate::training_core::sync_device(device);
        let now = Instant::now();
        self.forward_us = now.duration_since(self.last.expect("StepTimer::start must run first")).as_micros() as u64;
        self.last = Some(now);
    }

    /// Call after `.backward()` returns (before optimizer steps).
    pub fn mark_backward_done(&mut self, device: &BDevice) {
        if !self.enabled {
            return;
        }
        crate::training_core::sync_device(device);
        let now = Instant::now();
        self.backward_us = now.duration_since(self.last.expect("StepTimer::start must run first")).as_micros() as u64;
        self.last = Some(now);
    }

    /// Call after every optimizer `.step()` has returned. Returns `None` when
    /// diagnostics are disabled.
    pub fn finish(self, device: &BDevice) -> Option<StepTiming> {
        if !self.enabled {
            return None;
        }
        crate::training_core::sync_device(device);
        let now = Instant::now();
        let optimizer_us = now.duration_since(self.last.expect("StepTimer::start must run first")).as_micros() as u64;
        Some(StepTiming { forward_us: self.forward_us, backward_us: self.backward_us, optimizer_us })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_core::BDevice;

    #[test]
    fn disabled_timer_never_produces_timing() {
        let device = BDevice::default();
        let config = DiagnosticsConfig { enabled: false };
        let mut timer = StepTimer::start(&config, &device);
        timer.mark_forward_done(&device);
        timer.mark_backward_done(&device);
        assert_eq!(timer.finish(&device), None);
    }

    #[test]
    fn enabled_timer_produces_timing_with_all_three_stages_recorded() {
        let device = BDevice::default();
        let config = DiagnosticsConfig { enabled: true };
        let mut timer = StepTimer::start(&config, &device);
        timer.mark_forward_done(&device);
        timer.mark_backward_done(&device);
        // A `StepTiming` was produced at all - the real content-under-test is that this
        // struct exists and its three fields are internally consistent (each stage's
        // duration is independently non-negative by construction, u64), not any specific
        // magnitude - actual wall-clock values are inherently non-deterministic and belong
        // to `step_physics_diagnostics_enabled_matches_disabled_and_populates_timing`
        // (training_core.rs) for a real, non-trivial workload instead.
        assert!(timer.finish(&device).is_some());
    }
}
