use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialProps {
    /// Young's modulus [Pa]
    pub e: f64,
    /// Poisson's ratio (dimensionless)
    pub nu: f64,
    /// Density [kg/m³]
    pub density: f64,
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
        Self {
            e: 71.7e9,
            nu: 0.33,
            density: 2710.0,
        }
    }

    /// 4340 steel (heat-treated) defaults: E=30 Msi, ν=0.29, ρ=7850 kg/m³. Used for the
    /// pin-in-lug contact problem (`pinn_solver::pinlug_problem::PinLugProblem`).
    pub fn steel_4340() -> Self {
        use crate::units::MSI_TO_PA;
        Self {
            e: 30.0 * MSI_TO_PA,
            nu: 0.29,
            density: 7850.0,
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
}
