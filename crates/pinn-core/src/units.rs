//! Unit conversion constants — single source of truth for SI ↔ US Customary conversion.
//!
//! Internal storage is always SI (`Pa` for stress/modulus, `m` for length); the UI and
//! console output display US Customary units (psi/ksi/Msi, inches) via these constants.

/// 1 inch in meters.
pub const IN_TO_M: f64 = 0.0254;

/// 1 psi in pascals.
pub const PSI_TO_PA: f64 = 6_894.757;

/// 1 ksi (1000 psi) in pascals.
pub const KSI_TO_PA: f64 = PSI_TO_PA * 1_000.0;

/// 1 Msi (1,000,000 psi) in pascals — the customary unit for elastic modulus.
pub const MSI_TO_PA: f64 = PSI_TO_PA * 1_000_000.0;

/// 1 pound-force in newtons — used to convert a resultant contact FORCE (e.g. pin-in-lug's
/// 20,000 lbf driving load) into SI before deriving an equivalent traction. `LoadConfig`'s
/// `px`/`py` are far-field *stress* [Pa], not force, so any force-based BC must go through
/// this constant then be divided by an appropriate area (see
/// `pinn_solver::pinlug_problem::PinLugProblem::new`'s force-to-traction derivation).
pub const LBF_TO_N: f64 = 4.4482216153;

/// 1 foot in meters.
pub const FT_TO_M: f64 = IN_TO_M * 12.0;

/// 1 kip (1000 lbf) in newtons.
pub const KIP_TO_N: f64 = LBF_TO_N * 1_000.0;

/// 1 inch-pound-force (in·lbf) in joules — the USCS engineering energy unit.
pub const IN_LBF_TO_J: f64 = IN_TO_M * LBF_TO_N;

// ---------------------------------------------------------------------------------------
// Display-layer unit system — added so the UI can offer a genuine USCS/SI toggle rather
// than only ever showing hardcoded SI literals. This is deliberately a DISPLAY/INPUT-layer
// concern only: `ProblemSpec`/`ParametricProblemSpec` (and everything downstream of them —
// `training_core::compute_reference_scales`, `energy.rs`, `user_problem.rs`,
// `parametric_problem.rs`) continue to store and consume raw SI (Pa/m/kg·m⁻³) exactly as
// before. Nothing below this line changes what a spec's raw f64 fields mean; it only adds a
// way to convert an already-SI value to/from a chosen display system and to format it with a
// sensibly-scaled engineering unit. See `enhancement.md` Phases 43-65.
// ---------------------------------------------------------------------------------------

/// Which unit system the UI currently displays/accepts values in.
///
/// USCS is the default per `enhancement.md`'s explicit requirement (Phase 45). This has no
/// effect on any spec's raw SI storage — see the module note above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum UnitSystem {
    #[default]
    Uscs,
    Si,
}

/// The physical dimension a value carries, used to pick a sensibly-scaled unit for display.
///
/// `Strain` is dimensionless like `Dimensionless`, but is conventionally displayed in
/// microstrain (µε) rather than a bare ratio — kept as its own variant for that reason
/// (`enhancement.md` Phase 53).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalQuantity {
    Length,
    Stress,
    Force,
    Energy,
    Strain,
    Dimensionless,
}

/// Picks the sensibly-scaled engineering unit for a value's magnitude (`enhancement.md`
/// Phase 45's explicit requirement — e.g. stress prefers ksi over an awkward
/// `0.000002 ksi`-style psi value once it's large enough). Separated from [`to_unit`] so a
/// caller displaying several related values together (e.g. a slider's current value and its
/// min/max range) can pick ONE unit from a representative magnitude and convert every value
/// through that same unit — never mixing units within one row.
pub fn pick_unit(value_si: f64, qty: PhysicalQuantity, system: UnitSystem) -> &'static str {
    match (qty, system) {
        (PhysicalQuantity::Dimensionless, _) => "",
        (PhysicalQuantity::Strain, _) => "\u{b5}\u{3b5}",

        (PhysicalQuantity::Length, UnitSystem::Si) => if (value_si * 1_000.0).abs() >= 1_000.0 { "m" } else { "mm" },
        (PhysicalQuantity::Length, UnitSystem::Uscs) => if (value_si / IN_TO_M).abs() >= 36.0 { "ft" } else { "in" },

        (PhysicalQuantity::Stress, UnitSystem::Si) => {
            let mpa = (value_si / 1.0e6).abs();
            if mpa >= 1_000.0 { "GPa" } else if mpa >= 1.0 { "MPa" } else { "kPa" }
        }
        (PhysicalQuantity::Stress, UnitSystem::Uscs) => if (value_si / KSI_TO_PA).abs() >= 1.0 { "ksi" } else { "psi" },

        (PhysicalQuantity::Force, UnitSystem::Si) => if (value_si / 1_000.0).abs() >= 1.0 { "kN" } else { "N" },
        (PhysicalQuantity::Force, UnitSystem::Uscs) => if (value_si / KIP_TO_N).abs() >= 1.0 { "kip" } else { "lbf" },

        (PhysicalQuantity::Energy, UnitSystem::Si) => "J",
        (PhysicalQuantity::Energy, UnitSystem::Uscs) => "in\u{b7}lbf",
    }
}

/// Converts an SI value to a given (already-known) display unit's raw number — the inverse of
/// [`convert_from_display`]. Use [`convert_for_display`] instead when the unit itself should
/// also be chosen by magnitude; use this directly when several values must share one
/// already-chosen unit (see [`pick_unit`]'s doc comment).
pub fn to_unit(value_si: f64, qty: PhysicalQuantity, unit: &str) -> f64 {
    match (qty, unit) {
        (PhysicalQuantity::Dimensionless, _) => value_si,
        (PhysicalQuantity::Strain, _) => value_si * 1.0e6,

        (PhysicalQuantity::Length, "mm") => value_si * 1_000.0,
        (PhysicalQuantity::Length, "m") => value_si,
        (PhysicalQuantity::Length, "in") => value_si / IN_TO_M,
        (PhysicalQuantity::Length, "ft") => value_si / FT_TO_M,

        (PhysicalQuantity::Stress, "kPa") => value_si / 1.0e3,
        (PhysicalQuantity::Stress, "MPa") => value_si / 1.0e6,
        (PhysicalQuantity::Stress, "GPa") => value_si / 1.0e9,
        (PhysicalQuantity::Stress, "psi") => value_si / PSI_TO_PA,
        (PhysicalQuantity::Stress, "ksi") => value_si / KSI_TO_PA,

        (PhysicalQuantity::Force, "N") => value_si,
        (PhysicalQuantity::Force, "kN") => value_si / 1_000.0,
        (PhysicalQuantity::Force, "lbf") => value_si / LBF_TO_N,
        (PhysicalQuantity::Force, "kip") => value_si / KIP_TO_N,

        (PhysicalQuantity::Energy, "J") => value_si,
        (PhysicalQuantity::Energy, _) => value_si / IN_LBF_TO_J,

        _ => value_si,
    }
}

/// Converts an SI value to the given display system's most natural raw number + unit suffix.
/// Does not round or format the number — callers needing display text should use
/// [`format_value`], which calls this and formats the result.
pub fn convert_for_display(value_si: f64, qty: PhysicalQuantity, system: UnitSystem) -> (f64, &'static str) {
    let unit = pick_unit(value_si, qty, system);
    (to_unit(value_si, qty, unit), unit)
}

/// Converts a value the user entered/dragged in the given display system + unit suffix (as
/// returned by [`convert_for_display`] for the same `qty`/`system`) back to canonical SI —
/// the inverse of `convert_for_display`. Used by unit-aware input widgets (Stage G) so an
/// edit made in whatever unit is currently displayed round-trips correctly into the spec's
/// SI-only storage.
pub fn convert_from_display(value_display: f64, qty: PhysicalQuantity, unit: &str) -> f64 {
    match (qty, unit) {
        (PhysicalQuantity::Dimensionless, _) => value_display,
        (PhysicalQuantity::Strain, _) => value_display / 1.0e6,

        (PhysicalQuantity::Length, "mm") => value_display / 1_000.0,
        (PhysicalQuantity::Length, "m") => value_display,
        (PhysicalQuantity::Length, "in") => value_display * IN_TO_M,
        (PhysicalQuantity::Length, "ft") => value_display * FT_TO_M,

        (PhysicalQuantity::Stress, "kPa") => value_display * 1.0e3,
        (PhysicalQuantity::Stress, "MPa") => value_display * 1.0e6,
        (PhysicalQuantity::Stress, "GPa") => value_display * 1.0e9,
        (PhysicalQuantity::Stress, "psi") => value_display * PSI_TO_PA,
        (PhysicalQuantity::Stress, "ksi") => value_display * KSI_TO_PA,

        (PhysicalQuantity::Force, "N") => value_display,
        (PhysicalQuantity::Force, "kN") => value_display * 1_000.0,
        (PhysicalQuantity::Force, "lbf") => value_display * LBF_TO_N,
        (PhysicalQuantity::Force, "kip") => value_display * KIP_TO_N,

        (PhysicalQuantity::Energy, "J") => value_display,
        (PhysicalQuantity::Energy, _) => value_display * IN_LBF_TO_J,

        // Unknown unit string: no conversion is safer than a silent wrong guess — callers
        // control the unit strings they pass (always sourced from `convert_for_display`'s
        // own output), so this arm should be unreachable in practice.
        _ => value_display,
    }
}

/// Formats an SI value as display-ready text: sensibly-scaled number + unit suffix.
/// `Dimensionless` values are formatted bare (no trailing space, no unit).
pub fn format_value(value_si: f64, qty: PhysicalQuantity, system: UnitSystem) -> String {
    let (v, unit) = convert_for_display(value_si, qty, system);
    if unit.is_empty() {
        format!("{v:.4}")
    } else if qty == PhysicalQuantity::Strain {
        format!("{v:.1} {unit}")
    } else {
        format!("{v:.4e} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(value_si: f64, qty: PhysicalQuantity, system: UnitSystem) {
        let (v, unit) = convert_for_display(value_si, qty, system);
        let back = convert_from_display(v, qty, unit);
        let rel_err = if value_si.abs() > 1e-12 { (back - value_si).abs() / value_si.abs() } else { (back - value_si).abs() };
        assert!(rel_err < 1e-9, "roundtrip failed for {qty:?}/{system:?}: {value_si} -> ({v}, {unit}) -> {back}, rel_err={rel_err}");
    }

    #[test]
    fn length_roundtrips_both_systems_small_and_large() {
        for &v in &[0.001, 0.0254, 0.3048, 12.0, -0.5] {
            roundtrip(v, PhysicalQuantity::Length, UnitSystem::Si);
            roundtrip(v, PhysicalQuantity::Length, UnitSystem::Uscs);
        }
    }

    #[test]
    fn stress_roundtrips_both_systems_small_and_large() {
        for &v in &[1.0e3, 1.0e6, 2.93e8, 4.25e8, 6.894757e8, -1.0e7] {
            roundtrip(v, PhysicalQuantity::Stress, UnitSystem::Si);
            roundtrip(v, PhysicalQuantity::Stress, UnitSystem::Uscs);
        }
    }

    #[test]
    fn force_roundtrips_both_systems_small_and_large() {
        for &v in &[10.0, 2224.11, 22_241.1, -5000.0] {
            roundtrip(v, PhysicalQuantity::Force, UnitSystem::Si);
            roundtrip(v, PhysicalQuantity::Force, UnitSystem::Uscs);
        }
    }

    #[test]
    fn energy_roundtrips_both_systems() {
        for &v in &[0.1, 100.0, 1.0e5] {
            roundtrip(v, PhysicalQuantity::Energy, UnitSystem::Si);
            roundtrip(v, PhysicalQuantity::Energy, UnitSystem::Uscs);
        }
    }

    #[test]
    fn dimensionless_and_strain_roundtrip() {
        roundtrip(0.33, PhysicalQuantity::Dimensionless, UnitSystem::Uscs);
        roundtrip(0.002, PhysicalQuantity::Strain, UnitSystem::Si);
    }

    /// `enhancement.md` Phase 46's own worked example: a USCS problem (12 in plate width,
    /// 5000 lbf load, 42.5 ksi stress) displayed in SI should read ~304.8 mm / 22.24 kN /
    /// 293.0 MPa. This is the doc's own explicit "physical invariance" acceptance case, not
    /// an arbitrary tolerance we invented (Phase 57).
    #[test]
    fn enhancement_md_worked_example_uscs_to_si() {
        let width_si = 12.0 * IN_TO_M;
        let (mm, unit) = convert_for_display(width_si, PhysicalQuantity::Length, UnitSystem::Si);
        assert_eq!(unit, "mm");
        assert!((mm - 304.8).abs() < 0.1, "expected ~304.8 mm, got {mm}");

        let load_si = 5000.0 * LBF_TO_N;
        let (kn, unit) = convert_for_display(load_si, PhysicalQuantity::Force, UnitSystem::Si);
        assert_eq!(unit, "kN");
        assert!((kn - 22.24).abs() < 0.01, "expected ~22.24 kN, got {kn}");

        let stress_si = 42.5 * KSI_TO_PA;
        let (mpa, unit) = convert_for_display(stress_si, PhysicalQuantity::Stress, UnitSystem::Si);
        assert_eq!(unit, "MPa");
        assert!((mpa - 293.0).abs() < 0.5, "expected ~293.0 MPa, got {mpa}");
    }

    #[test]
    fn unit_system_default_is_uscs() {
        assert_eq!(UnitSystem::default(), UnitSystem::Uscs);
    }
}
