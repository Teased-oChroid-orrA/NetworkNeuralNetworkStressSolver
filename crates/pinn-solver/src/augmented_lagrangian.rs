//! Augmented-Lagrangian dual-variable primitive - General-PINN architecture recommendations
//! §40 ("constraint/augmented-Lagrangian framework").
//!
//! This codebase's own two real inequality constraints (pin-lug's Signorini KKT contact
//! conditions, `pinlug_problem::InterfacePenetrationTerm`/`InterfaceNonTensionTerm` - see
//! `crate::problem::ConstraintKind::PenaltyInequality`) are currently enforced via a pure
//! quadratic penalty (squared-hinge, e.g. `gap.neg().clamp_min(0.0).powf_scalar(2.0)`) with NO
//! dual variable. A pure penalty method's solution is only exact in the ρ→∞ limit; any
//! practically-trainable finite ρ leaves a real, uncorrected constraint-violation bias. The
//! standard fix is the augmented Lagrangian (Hestenes-Powell-Rockafellar) method: track a dual
//! variable λ per constraint and update it additively based on the observed violation, which
//! corrects exactly that bias without needing ρ→∞.
//!
//! [`AugmentedLagrangianState`] is that primitive - real, tested arithmetic, deliberately NOT
//! wired into `InterfacePenetrationTerm`/`InterfaceNonTensionTerm`'s live `compute()` bodies.
//! Swapping a working, extensively-verified penalty term's live numerics for a new mechanism is
//! a genuine behavior change (would need its own dedicated verification pass, the same
//! discipline this whole session's Kt investigation already established for any live-numerics
//! change) - out of scope for a classification/capability-building pass. This module makes the
//! capability real and available, not speculative.

/// Augmented-Lagrangian dual state for a SINGLE inequality constraint of the form `g(x) <= 0`
/// (satisfied when `g(x) <= 0`, violated when `g(x) > 0`) - the standard Hestenes-Powell-
/// Rockafellar convention. `lambda` starts at `0.0` (no penalty pressure until a violation is
/// observed).
#[derive(Debug, Clone, Copy)]
pub struct AugmentedLagrangianState {
    pub lambda: f64,
    pub rho: f64,
}

impl AugmentedLagrangianState {
    /// `rho` (the quadratic-penalty weight) must be strictly positive - a non-positive `rho`
    /// makes the penalty term meaningless (division by zero or a concave, non-penalizing
    /// "penalty").
    pub fn new(rho: f64) -> Self {
        assert!(rho > 0.0, "AugmentedLagrangianState::new: rho must be positive, got {rho}");
        Self { lambda: 0.0, rho }
    }

    /// The augmented-Lagrangian penalty TERM to add to the loss for constraint value `g` -
    /// `(max(0, λ+ρg)² - λ²) / (2ρ)`. Convex and C¹-continuous in `g`: matches a plain quadratic
    /// penalty exactly once the bracket is active (`λ+ρg >= 0`), and is exactly the constant
    /// `-λ²/(2ρ)` (independent of `g`) while the constraint stays comfortably satisfied - the
    /// textbook Hestenes-Powell-Rockafellar penalty, not a hand-approximated one.
    pub fn penalty(&self, g: f64) -> f64 {
        let z = (self.lambda + self.rho * g).max(0.0);
        (z * z - self.lambda * self.lambda) / (2.0 * self.rho)
    }

    /// Dual ascent step: `λ <- max(0, λ + ρg)`. Call once per OUTER iteration, not every
    /// training step - the standard AL schedule updates the multiplier on a slower cadence than
    /// the inner unconstrained minimization (the same "adapt on a coarser cadence than the raw
    /// training loop" convention `SawBrdr`'s own weight adaptation and this codebase's AMR sweep
    /// interval already use elsewhere). `lambda` only ever increases while `g` stays positive
    /// (constraint violated) and is projected back to `0.0`, never negative, once `g` is
    /// sufficiently negative (constraint comfortably satisfied) - it does not "carry over" debt
    /// from a past violation once the constraint is well inside the feasible region.
    pub fn update(&mut self, g: f64) {
        self.lambda = (self.lambda + self.rho * g).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "rho must be positive")]
    fn new_panics_on_non_positive_rho() {
        AugmentedLagrangianState::new(0.0);
    }

    #[test]
    fn penalty_is_zero_when_constraint_is_comfortably_satisfied_and_lambda_is_zero() {
        let state = AugmentedLagrangianState::new(2.0);
        // g=-1: lambda + rho*g = 0 + 2*(-1) = -2 -> max(0,-2)=0 -> penalty = (0-0)/(2*2) = 0.
        assert_eq!(state.penalty(-1.0), 0.0);
    }

    #[test]
    fn penalty_matches_hand_computed_value_when_constraint_is_violated() {
        // rho=2.0, lambda=1.0, g=3.0 -> z=max(0, 1+2*3)=7 -> penalty=(49-1)/4=12.0.
        let state = AugmentedLagrangianState { lambda: 1.0, rho: 2.0 };
        assert!((state.penalty(3.0) - 12.0).abs() < 1e-12, "{}", state.penalty(3.0));
    }

    #[test]
    fn update_matches_hand_computed_dual_ascent_when_violated() {
        let mut state = AugmentedLagrangianState::new(2.0);
        state.update(3.0); // lambda <- max(0, 0 + 2*3) = 6
        assert!((state.lambda - 6.0).abs() < 1e-12, "{}", state.lambda);
        state.update(1.0); // lambda <- max(0, 6 + 2*1) = 8
        assert!((state.lambda - 8.0).abs() < 1e-12, "{}", state.lambda);
    }

    #[test]
    fn update_projects_lambda_to_zero_not_negative_once_comfortably_feasible() {
        let mut state = AugmentedLagrangianState { lambda: 0.5, rho: 1.0 };
        state.update(-10.0); // lambda <- max(0, 0.5 + 1*(-10)) = max(0, -9.5) = 0
        assert_eq!(state.lambda, 0.0);
    }
}
