//! Signorini contact-mechanics penalty terms (KKT complementarity conditions for
//! frictionless unilateral contact) and stress-tensor radial decomposition, shared
//! infrastructure for the pin-in-lug contact problem.
//!
//! RED phase (TDD): production bodies are `todo!()` stubs. See the architect's
//! `BoundaryValueProblem` trait-family spec; this module will be wired into
//! `PinLugProblem`'s loss terms once that trait exists (tracked separately —
//! sections B/C/D of the test-authoring plan).

/// Penalty for interpenetration (gap < 0 means bodies overlap). Zero for gap >= 0
/// (non-penetration satisfied), quadratic in the overlap depth otherwise.
#[allow(dead_code)]
pub fn penetration_penalty(gap: f64) -> f64 {
    todo!()
}

/// Penalty for tensile (pulling) contact pressure, which is non-physical for
/// frictionless unilateral contact (contact pressure must be <= 0, i.e. compressive
/// only, under the sign convention where negative = compression). Zero for
/// contact_pressure <= 0, quadratic in the tensile magnitude otherwise.
#[allow(dead_code)]
pub fn non_tension_penalty(contact_pressure: f64) -> f64 {
    todo!()
}

/// Decompose Cartesian stress components (sxx, syy, sxy) into polar/radial components
/// (s_rr, s_tt, s_rt) at angle `theta` (radians), via standard 2D stress-tensor rotation.
#[allow(dead_code)]
pub fn decompose_radial(sxx: f64, syy: f64, sxy: f64, theta: f64) -> (f64, f64, f64) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // === penetration_penalty ===

    #[test]
    fn penetration_penalty_zero_gap_is_exactly_zero() {
        assert_eq!(penetration_penalty(0.0), 0.0);
    }

    #[test]
    fn penetration_penalty_positive_gap_no_penalty() {
        assert_eq!(penetration_penalty(1.0), 0.0);
        assert_eq!(penetration_penalty(1.0e-3), 0.0);
        assert_eq!(penetration_penalty(1.0e6), 0.0);
    }

    #[test]
    fn penetration_penalty_negative_gap_quadratic() {
        // gap = -0.01 (1cm overlap) -> penalty = gap^2 = 1e-4
        let p = penetration_penalty(-0.01);
        assert!((p - 1.0e-4).abs() < 1e-10, "expected 1e-4, got {p}");

        // gap = -2.0 -> penalty = 4.0
        let p2 = penetration_penalty(-2.0);
        assert!((p2 - 4.0).abs() < 1e-10, "expected 4.0, got {p2}");
    }

    #[test]
    fn penetration_penalty_is_monotonic_in_overlap_depth() {
        let shallow = penetration_penalty(-0.1);
        let deep = penetration_penalty(-1.0);
        let deeper = penetration_penalty(-10.0);
        assert!(shallow < deep, "shallow={shallow} should be < deep={deep}");
        assert!(deep < deeper, "deep={deep} should be < deeper={deeper}");
        // Boundary: penalty strictly increases in overlap depth (magnitude of negative gap).
        assert!(penetration_penalty(-1e-9) > 0.0);
    }

    // === non_tension_penalty ===

    #[test]
    fn non_tension_penalty_zero_pressure_is_exactly_zero() {
        assert_eq!(non_tension_penalty(0.0), 0.0);
    }

    #[test]
    fn non_tension_penalty_compressive_pressure_no_penalty() {
        // Compressive contact pressure (<= 0 under this sign convention) is physical.
        assert_eq!(non_tension_penalty(-1.0), 0.0);
        assert_eq!(non_tension_penalty(-1.0e-3), 0.0);
        assert_eq!(non_tension_penalty(-1.0e6), 0.0);
    }

    #[test]
    fn non_tension_penalty_tensile_pressure_quadratic() {
        // contact_pressure = 0.01 -> penalty = 1e-4
        let p = non_tension_penalty(0.01);
        assert!((p - 1.0e-4).abs() < 1e-10, "expected 1e-4, got {p}");

        // contact_pressure = 3.0 -> penalty = 9.0
        let p2 = non_tension_penalty(3.0);
        assert!((p2 - 9.0).abs() < 1e-10, "expected 9.0, got {p2}");
    }

    #[test]
    fn non_tension_penalty_is_monotonic_in_tension_magnitude() {
        let small = non_tension_penalty(0.1);
        let mid = non_tension_penalty(1.0);
        let large = non_tension_penalty(10.0);
        assert!(small < mid, "small={small} should be < mid={mid}");
        assert!(mid < large, "mid={mid} should be < large={large}");
        assert!(non_tension_penalty(1e-9) > 0.0);
    }

    // === decompose_radial ===

    #[test]
    fn radial_decomposition_matches_hand_computed_case() {
        // sxx=100, syy=50, sxy=25, theta=PI/4.
        // Standard 2D stress rotation to polar/radial axes at angle theta:
        //   s_rr = sxx*c^2 + syy*s^2 + 2*sxy*s*c
        //   s_tt = sxx*s^2 + syy*c^2 - 2*sxy*s*c
        //   s_rt = (syy - sxx)*s*c + sxy*(c^2 - s^2)
        // At theta = PI/4: c = s = sqrt(2)/2, c^2 = s^2 = 0.5, c^2 - s^2 = 0.
        //   s_rr = 100*0.5 + 50*0.5 + 2*25*0.5 = 50 + 25 + 25 = 100
        //   s_tt = 100*0.5 + 50*0.5 - 2*25*0.5 = 50 + 25 - 25 = 50
        //   s_rt = (50 - 100)*0.5 + 25*0 = -25
        let (sxx, syy, sxy, theta) = (100.0_f64, 50.0_f64, 25.0_f64, PI / 4.0);
        let (s_rr, s_tt, s_rt) = decompose_radial(sxx, syy, sxy, theta);

        assert!(
            (s_rr - 100.0).abs() < 1e-6,
            "s_rr: expected 100.0, got {s_rr}"
        );
        assert!(
            (s_tt - 50.0).abs() < 1e-6,
            "s_tt: expected 50.0, got {s_tt}"
        );
        assert!(
            (s_rt - (-25.0)).abs() < 1e-6,
            "s_rt: expected -25.0, got {s_rt}"
        );

        // Invariant check: trace is preserved under rotation (sanity, not a substitute
        // for the hand-computed values above).
        assert!(((s_rr + s_tt) - (sxx + syy)).abs() < 1e-6);
    }

    #[test]
    fn radial_decomposition_round_trips_with_kirsch_probe_rotation() {
        // Forward rotation formula, pinned verbatim from
        // `compute_kirsch_probes` in training_core.rs (radial -> Cartesian):
        //   sxx = s_rr*c*c + s_tt*s*s - 2*s_rt*s*c
        //   syy = s_rr*s*s + s_tt*c*c + 2*s_rt*s*c
        //   sxy = (s_rr - s_tt)*s*c + s_rt*(c*c - s*s)
        // decompose_radial must be the exact inverse of this transform.
        let cases: [(f64, f64, f64, f64); 4] = [
            (100.0, -20.0, 5.0, 0.0),
            (100.0, -20.0, 5.0, PI / 6.0),
            (50.0, 30.0, -15.0, PI / 3.0),
            (0.0, 200.0, 10.0, PI / 2.0),
        ];

        for (s_rr, s_tt, s_rt, theta) in cases {
            let c = theta.cos();
            let s = theta.sin();
            let sxx = s_rr * c * c + s_tt * s * s - 2.0 * s_rt * s * c;
            let syy = s_rr * s * s + s_tt * c * c + 2.0 * s_rt * s * c;
            let sxy = (s_rr - s_tt) * s * c + s_rt * (c * c - s * s);

            let (s_rr_rt, s_tt_rt, s_rt_rt) = decompose_radial(sxx, syy, sxy, theta);

            assert!(
                (s_rr_rt - s_rr).abs() < 1e-6,
                "theta={theta}: s_rr round-trip: expected {s_rr}, got {s_rr_rt}"
            );
            assert!(
                (s_tt_rt - s_tt).abs() < 1e-6,
                "theta={theta}: s_tt round-trip: expected {s_tt}, got {s_tt_rt}"
            );
            assert!(
                (s_rt_rt - s_rt).abs() < 1e-6,
                "theta={theta}: s_rt round-trip: expected {s_rt}, got {s_rt_rt}"
            );
        }
    }
}
