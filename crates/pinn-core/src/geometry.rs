use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SymmetryMode {
    /// Full plate domain
    Full,
    /// Exploit two-fold symmetry — solve quarter plate only
    QuarterSymm,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HoleType {
    /// Circular hole with given radius [m]
    Circular { radius: f64 },
    /// No hole
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryConfig {
    /// Plate half-width in x [m] — full width = 2*half_w
    pub half_w: f64,
    /// Plate half-height in y [m] — full height = 2*half_h
    pub half_h: f64,
    /// Thickness [m] (for plane-stress scaling)
    pub thickness: f64,
    pub hole: HoleType,
    pub symmetry: SymmetryMode,
}

impl GeometryConfig {
    /// Standard 10×10×0.1 inch plate with 0.25-in diameter hole
    pub fn kirsch_plate_inches() -> Self {
        use crate::units::IN_TO_M;
        Self {
            half_w:    5.0 * IN_TO_M,
            half_h:    5.0 * IN_TO_M,
            thickness: 0.1 * IN_TO_M,
            hole:      HoleType::Circular { radius: 0.125 * IN_TO_M },
            symmetry:  SymmetryMode::QuarterSymm,
        }
    }

    /// Pin-in-lug problem: LUG domain. W=1.5in (half_w=0.75in), t=0.4in, central hole
    /// R=0.5in, full (not quarter-symmetric — contact loading breaks the Kirsch-style
    /// symmetry) plate geometry.
    pub fn pinlug_lug_inches() -> Self {
        use crate::units::IN_TO_M;
        Self {
            half_w:    0.75 * IN_TO_M,
            half_h:    0.75 * IN_TO_M,
            thickness: 0.4 * IN_TO_M,
            hole:      HoleType::Circular { radius: 0.5 * IN_TO_M },
            symmetry:  SymmetryMode::Full,
        }
    }

    /// Pin-in-lug problem: PIN domain — a solid disk of radius 0.5in. `GeometryConfig` is
    /// fundamentally plate-with-(optional)-hole shaped and cannot represent a solid disk
    /// directly; the simplest correct representation within the existing type is a square
    /// bounding box (`half_w=half_h=radius`) with `HoleType::None` (no interior exclusion)
    /// — `contains()` then reduces to the bounding-box check alone. The pin's sampling
    /// strategy (`PinLugSamplingStrategy` in `pinn_solver::pinlug_problem`) is responsible
    /// for rejecting points outside the disk (`x²+y² <= radius²`) itself, the same way
    /// `HoleType::Circular` sampling rejects points *inside* a hole today — this is a
    /// documented scope choice, not an oversight (flagged per the design brief).
    pub fn pinlug_pin_inches() -> Self {
        use crate::units::IN_TO_M;
        let radius = 0.5 * IN_TO_M;
        Self {
            half_w:    radius,
            half_h:    radius,
            thickness: 0.4 * IN_TO_M,
            hole:      HoleType::None,
            symmetry:  SymmetryMode::Full,
        }
    }

    /// Domain bounds in physical coords based on symmetry mode
    pub fn x_range(&self) -> (f64, f64) {
        match self.symmetry {
            SymmetryMode::Full      => (-self.half_w, self.half_w),
            SymmetryMode::QuarterSymm => (0.0, self.half_w),
        }
    }

    pub fn y_range(&self) -> (f64, f64) {
        match self.symmetry {
            SymmetryMode::Full      => (-self.half_h, self.half_h),
            SymmetryMode::QuarterSymm => (0.0, self.half_h),
        }
    }

    /// Returns true if (x,y) is inside the plate domain (outside hole)
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (x0, x1) = self.x_range();
        let (y0, y1) = self.y_range();
        if x < x0 || x > x1 || y < y0 || y > y1 {
            return false;
        }
        match self.hole {
            HoleType::Circular { radius } => (x * x + y * y) >= radius * radius,
            HoleType::None => true,
        }
    }

    /// Smooth distance to hole boundary (used for BC ansatz). Returns 0 at hole edge.
    pub fn hole_distance(&self, x: f64, y: f64) -> f64 {
        match self.hole {
            HoleType::Circular { radius } => {
                let r = (x * x + y * y).sqrt();
                (r - radius).max(0.0)
            }
            HoleType::None => f64::MAX,
        }
    }

    /// Normalized coords: maps (x,y) in [x0,x1]×[y0,y1] → [-1,1]²
    pub fn normalize(&self, x: f64, y: f64) -> (f64, f64) {
        let (x0, x1) = self.x_range();
        let (y0, y1) = self.y_range();
        (
            2.0 * (x - x0) / (x1 - x0) - 1.0,
            2.0 * (y - y0) / (y1 - y0) - 1.0,
        )
    }

    /// Denormalize: [-1,1]² → physical domain
    pub fn denormalize(&self, xn: f64, yn: f64) -> (f64, f64) {
        let (x0, x1) = self.x_range();
        let (y0, y1) = self.y_range();
        (
            x0 + (xn + 1.0) * 0.5 * (x1 - x0),
            y0 + (yn + 1.0) * 0.5 * (y1 - y0),
        )
    }

    pub fn area(&self) -> f64 {
        let (x0, x1) = self.x_range();
        let (y0, y1) = self.y_range();
        let plate_area = (x1 - x0) * (y1 - y0);
        let hole_area = match self.hole {
            HoleType::Circular { radius } => {
                let full_area = std::f64::consts::PI * radius * radius;
                match self.symmetry {
                    SymmetryMode::QuarterSymm => full_area / 4.0,
                    SymmetryMode::Full => full_area,
                }
            }
            HoleType::None => 0.0,
        };
        plate_area - hole_area
    }

    /// FNV-1a hash for detecting geometry changes (warm-start logic)
    pub fn geometry_hash(&self) -> u64 {
        let mut h: u64 = 14695981039346656037;
        let bytes = [
            self.half_w.to_bits(),
            self.half_h.to_bits(),
            self.thickness.to_bits(),
            match self.hole {
                HoleType::Circular { radius } => radius.to_bits(),
                HoleType::None => 0,
            },
        ];
        for b in bytes.iter() {
            h ^= b;
            h = h.wrapping_mul(1099511628211);
        }
        h ^= self.symmetry as u64;
        h = h.wrapping_mul(1099511628211);
        h
    }
}
