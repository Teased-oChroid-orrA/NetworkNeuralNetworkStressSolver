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
