use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialProps {
    /// Young's modulus [Pa]
    pub e: f64,
    /// Poisson's ratio (dimensionless)
    pub nu: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Ultimate tensile strength [Pa] — used ONLY as an opt-in normalization reference
    /// (`SolverConfig::use_ultimate_strength_scaling`, `compute_reference_scales`); never
    /// read by `lame()`/`plane_stress_c()`/the constitutive law. See per-constructor doc
    /// comments for the exact citation and, where applicable, the temper caveat.
    pub ultimate_strength_pa: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LameConsts {
    /// First Lamé parameter λ [Pa]
    pub lambda: f64,
    /// Second Lamé parameter (shear modulus) μ [Pa]
    pub mu: f64,
}

impl MaterialProps {
    /// Al 7075-T6 defaults: E=10.4 Msi (71.7 GPa), ν=0.33, ρ=2710 kg/m³
    pub fn al7075_t6() -> Self {
        use crate::units::PSI_TO_PA;
        Self {
            e: 71.7e9,
            nu: 0.33,
            density: 2710.0,
            // ASTM B209 minimum specified UTS for 7075-T6 sheet/plate: 83,000 psi (572 MPa).
            // Standard handbook value for the same alloy/temper as the existing E=10.4 Msi
            // figure.
            ultimate_strength_pa: 83_000.0 * PSI_TO_PA,
        }
    }

    /// 4340 steel (heat-treated) defaults: E=30 Msi, ν=0.29, ρ=7850 kg/m³. Used for the
    /// pin-in-lug contact problem (`pinn_solver::pinlug_problem::PinLugProblem`).
    pub fn steel_4340() -> Self {
        use crate::units::{MSI_TO_PA, KSI_TO_PA};
        Self {
            e: 30.0 * MSI_TO_PA,
            nu: 0.29,
            density: 7850.0,
            // Representative quenched-and-tempered condition, ~200 ksi UTS class; actual
            // 4340 UTS ranges 125-287 ksi by temper. Unlike E/nu (treatment-invariant for
            // this alloy), this value is temper-specific — do not treat it as a fixed
            // material constant the way E/nu are.
            ultimate_strength_pa: 200.0 * KSI_TO_PA,
        }
    }

    pub fn lame(&self) -> LameConsts {
        let e = self.e;
        let nu = self.nu;
        LameConsts {
            lambda: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            mu: e / (2.0 * (1.0 + nu)),
        }
    }

    /// 3×3 plane-stress constitutive matrix C in Voigt notation:
    /// {σ_xx, σ_yy, σ_xy} = C · {ε_xx, ε_yy, 2·ε_xy}
    pub fn plane_stress_c(&self) -> [[f64; 3]; 3] {
        let e = self.e;
        let nu = self.nu;
        let s = e / (1.0 - nu * nu);
        [
            [s,        s * nu,    0.0              ],
            [s * nu,   s,         0.0              ],
            [0.0,      0.0,       s * (1.0 - nu) / 2.0],
        ]
    }

    /// Von Mises stress from principal stresses (plane stress)
    pub fn von_mises_plane_stress(sxx: f64, syy: f64, sxy: f64) -> f64 {
        (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).sqrt()
    }

    /// E / F_c — dimensionless modulus ratio for diagnostic/logging use only. NOT wired
    /// into any constitutive computation (`lame()`, `plane_stress_c()`, `energy.rs`'s
    /// stress/strain chain are all unaffected). `f_c` is typically
    /// `self.ultimate_strength_pa` but is passed explicitly so callers can also probe
    /// against an arbitrary reference stress.
    pub fn dimensionless_modulus(&self, f_c: f64) -> f64 {
        self.e / f_c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{KSI_TO_PA, PSI_TO_PA};

    #[test]
    fn al7075_t6_ultimate_strength_matches_astm_b209_minimum() {
        let mat = MaterialProps::al7075_t6();
        let expected = 83_000.0 * PSI_TO_PA;
        assert!(
            (mat.ultimate_strength_pa - expected).abs() < 1.0,
            "expected {expected} Pa, got {}",
            mat.ultimate_strength_pa
        );
    }

    #[test]
    fn steel_4340_ultimate_strength_matches_representative_qt_condition() {
        let mat = MaterialProps::steel_4340();
        let expected = 200.0 * KSI_TO_PA;
        assert!(
            (mat.ultimate_strength_pa - expected).abs() < 1.0,
            "expected {expected} Pa, got {}",
            mat.ultimate_strength_pa
        );
    }

    #[test]
    fn dimensionless_modulus_is_e_over_f_c() {
        let mat = MaterialProps::al7075_t6();
        let f_c = 1.0e8;
        let result = mat.dimensionless_modulus(f_c);
        let expected = mat.e / f_c;
        assert!(
            ((result - expected) / expected).abs() < 1e-12,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn dimensionless_modulus_with_own_ultimate_strength_is_e_over_uts() {
        let mat = MaterialProps::steel_4340();
        let result = mat.dimensionless_modulus(mat.ultimate_strength_pa);
        let expected = mat.e / mat.ultimate_strength_pa;
        assert!(
            ((result - expected) / expected).abs() < 1e-12,
            "expected {expected}, got {result}"
        );
    }
}
