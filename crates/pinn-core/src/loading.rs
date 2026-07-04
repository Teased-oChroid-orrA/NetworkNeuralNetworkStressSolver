use serde::{Deserialize, Serialize};

use crate::problem::DomainId;

/// Far-field applied loads [Pa]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoadConfig {
    /// Far-field traction in x-direction applied at x = ±half_w
    pub px: f64,
    /// Far-field traction in y-direction applied at y = ±half_h
    pub py: f64,
}

impl LoadConfig {
    pub fn uniaxial_x(px: f64) -> Self {
        Self { px, py: 0.0 }
    }
    pub fn biaxial(px: f64, py: f64) -> Self {
        Self { px, py }
    }
    pub fn default_10ksi() -> Self {
        Self { px: 10.0 * crate::units::KSI_TO_PA, py: 0.0 }
    }
}

/// A single boundary point with its associated traction vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryPoint {
    /// Physical coordinates [m]
    pub x: f64,
    pub y: f64,
    /// Outward unit normal
    pub nx: f64,
    pub ny: f64,
    /// Prescribed traction [Pa]
    pub tx: f64,
    pub ty: f64,
    pub kind: BoundaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryKind {
    /// Symmetry edge: normal displacement = 0, tangential traction-free
    Symmetry,
    /// Applied far-field traction
    NeumannLoad,
    /// Stress-free (hole edge, free edges)
    NeumannFree,
    /// Contact/interface boundary shared with another domain (e.g. pin-in-lug contact
    /// surface) — traction here is not prescribed in closed form but resolved against the
    /// partner domain's state by a cross-domain loss term.
    Interface { partner_domain: DomainId },
}
