use crate::{
    geometry::{GeometryConfig, HoleType, SymmetryMode},
    loading::{BoundaryKind, BoundaryPoint, LoadConfig},
};

// Deterministic LCG seeds — distinct per call site so the four point sets (interior fill,
// near-hole ring, boundary edges, equilibrium ring) don't share a random stream.
const SEED_INTERIOR_FILL: u64 = 42;
const SEED_NEAR_HOLE_RING: u64 = 777;
const SEED_BOUNDARY: u64 = 99;
const SEED_EQ_RING: u64 = 42_424_242;

/// Rejection-sampling attempt budget, as a multiple of the target point count, before
/// falling back to a deterministic grid sweep.
const REJECTION_SAMPLE_ATTEMPTS_FACTOR: usize = 20;
/// Minimum near-hole interior points guaranteed regardless of random draw outcome — see
/// the comment at its use site for why uniform sampling alone isn't enough.
const MIN_NEAR_HOLE_INTERIOR_POINTS: usize = 200;
/// Near-hole interior ring: inner radius just outside the hole boundary (avoids sampling
/// exactly on it) through 3x the hole radius, where the stress concentration is localized.
const NEAR_HOLE_RING_INNER_FACTOR: f64 = 1.001;
const NEAR_HOLE_RING_OUTER_FACTOR: f64 = 3.0;
/// Equilibrium-residual ring: starts further out (2x hole radius) than the near-hole
/// interior ring so ±fd_h meta-shifts never clip inside the hole.
const EQ_RING_INNER_FACTOR: f64 = 2.0;
const EQ_RING_OUTER_FACTOR: f64 = 3.0;

/// A set of collocation points for the physics solver
pub struct CollocationSet {
    /// Interior domain points [x, y] in physical coords [m]
    pub interior: Vec<[f64; 2]>,
    /// Boundary points with normals and tractions
    pub boundary: Vec<BoundaryPoint>,
}

impl CollocationSet {
    /// Generate initial collocation set using simple random sampling
    pub fn new(geom: &GeometryConfig, load: &LoadConfig, n_int: usize, n_bnd: usize) -> Self {
        let interior = sample_interior(geom, n_int);
        let boundary = sample_boundary(geom, load, n_bnd);
        Self { interior, boundary }
    }

    /// Regenerate interior points (used by FI-PINN resampling)
    pub fn resample_interior(&mut self, _geom: &GeometryConfig, new_pts: Vec<[f64; 2]>) {
        self.interior = new_pts;
    }
}

/// Generate N interior collocation points via Latin Hypercube Sampling (stratified random)
pub fn sample_interior(geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
    let (x0, x1) = geom.x_range();
    let (y0, y1) = geom.y_range();

    let mut pts = Vec::with_capacity(n);
    let mut attempts = 0_usize;
    let max_attempts = n * REJECTION_SAMPLE_ATTEMPTS_FACTOR;

    // Simple stratified sampling with rejection
    let grid = (n as f64).sqrt().ceil() as usize + 2;
    let dx = (x1 - x0) / grid as f64;
    let dy = (y1 - y0) / grid as f64;

    // LCG pseudo-random (deterministic, no external rand dependency)
    let mut rng = LcgRng::new(SEED_INTERIOR_FILL);

    while pts.len() < n && attempts < max_attempts {
        let x = x0 + rng.next_f64() * (x1 - x0);
        let y = y0 + rng.next_f64() * (y1 - y0);
        if geom.contains(x, y) {
            pts.push([x, y]);
        }
        attempts += 1;
    }

    // If not enough, fill with grid sweep
    if pts.len() < n {
        'outer: for i in 0..grid {
            for j in 0..grid {
                let x = x0 + (i as f64 + 0.5) * dx;
                let y = y0 + (j as f64 + 0.5) * dy;
                if geom.contains(x, y) {
                    pts.push([x, y]);
                    if pts.len() >= n {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Guarantee ≥200 near-hole interior points regardless of random draw outcome.
    // The stress concentration is localised to r < 3·r_hole (0.8% of domain area),
    // so uniform sampling puts only ~32 points there — not enough for K_t convergence.
    if let HoleType::Circular { radius } = geom.hole {
        let n_ring = MIN_NEAR_HOLE_INTERIOR_POINTS.min(n / 4);
        let r_inner = radius * NEAR_HOLE_RING_INNER_FACTOR;
        let r_outer = radius * NEAR_HOLE_RING_OUTER_FACTOR;
        let mut rng2 = LcgRng::new(SEED_NEAR_HOLE_RING);
        let ring_pts: Vec<[f64; 2]> = (0..n_ring).filter_map(|k| {
            let angle = std::f64::consts::FRAC_PI_2 * k as f64 / n_ring as f64;
            let r = r_inner + (r_outer - r_inner) * rng2.next_f64();
            let p = [r * angle.cos(), r * angle.sin()];
            if geom.contains(p[0], p[1]) { Some(p) } else { None }
        }).collect();
        // Prepend ring points so they survive the truncate; far-field pts fill the rest
        let mut combined = ring_pts;
        combined.extend_from_slice(&pts);
        combined.truncate(n);
        return combined;
    }
    pts.truncate(n);
    pts
}

/// Generate boundary collocation points on all domain edges
pub fn sample_boundary(geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<BoundaryPoint> {
    let mut pts = Vec::new();
    let n_per_edge = (n / 4).max(1);

    let (x0, x1) = geom.x_range();
    let (y0, y1) = geom.y_range();
    let mut rng = LcgRng::new(SEED_BOUNDARY);

    // Right edge: x = x1, normal = (+1, 0), traction = (Px, 0)
    for _ in 0..n_per_edge {
        let y = y0 + rng.next_f64() * (y1 - y0);
        pts.push(BoundaryPoint {
            x: x1, y,
            nx: 1.0, ny: 0.0,
            tx: load.px, ty: 0.0,
            kind: BoundaryKind::NeumannLoad,
        });
    }

    // Top edge: y = y1, normal = (0, +1), traction = (0, Py)
    for _ in 0..n_per_edge {
        let x = x0 + rng.next_f64() * (x1 - x0);
        pts.push(BoundaryPoint {
            x, y: y1,
            nx: 0.0, ny: 1.0,
            tx: 0.0, ty: load.py,
            kind: BoundaryKind::NeumannLoad,
        });
    }

    // Symmetry edges (quarter-model) or free edges (full model)
    match geom.symmetry {
        SymmetryMode::QuarterSymm => {
            // Left edge (x=0): symmetry → u_x = 0 (hard), traction in y = 0
            for _ in 0..n_per_edge {
                let y = y0 + rng.next_f64() * (y1 - y0);
                pts.push(BoundaryPoint {
                    x: x0, y,
                    nx: -1.0, ny: 0.0,
                    tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::Symmetry,
                });
            }
            // Bottom edge (y=0): symmetry → u_y = 0 (hard), traction in x = 0
            for _ in 0..n_per_edge {
                let x = x0 + rng.next_f64() * (x1 - x0);
                pts.push(BoundaryPoint {
                    x, y: y0,
                    nx: 0.0, ny: -1.0,
                    tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::Symmetry,
                });
            }
        }
        SymmetryMode::Full => {
            // Left edge: traction = (-Px, 0)
            for _ in 0..n_per_edge {
                let y = y0 + rng.next_f64() * (y1 - y0);
                pts.push(BoundaryPoint {
                    x: x0, y,
                    nx: -1.0, ny: 0.0,
                    tx: -load.px, ty: 0.0,
                    kind: BoundaryKind::NeumannLoad,
                });
            }
            // Bottom edge: traction = (0, -Py)
            for _ in 0..n_per_edge {
                let x = x0 + rng.next_f64() * (x1 - x0);
                pts.push(BoundaryPoint {
                    x, y: y0,
                    nx: 0.0, ny: -1.0,
                    tx: 0.0, ty: -load.py,
                    kind: BoundaryKind::NeumannLoad,
                });
            }
        }
    }

    // Hole boundary: stress-free (σ·n = 0)
    if let HoleType::Circular { radius } = geom.hole {
        let n_hole = n / 2;
        for k in 0..n_hole {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n_hole as f64;
            // For quarter model, only sample first quadrant (angle 0..pi/2)
            let angle = match geom.symmetry {
                SymmetryMode::QuarterSymm => angle * 0.25,
                SymmetryMode::Full => angle,
            };
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            // Outward normal of the *solid plate material*: the plate occupies r > radius,
            // so its boundary normal points away from the material — radially inward,
            // toward the hole center (i.e. into the void, not into the plate).
            let nx = -angle.cos();
            let ny = -angle.sin();
            pts.push(BoundaryPoint {
                x, y, nx, ny,
                tx: 0.0, ty: 0.0,
                kind: BoundaryKind::NeumannFree,
            });
        }
    }

    pts
}

/// Sample N equilibrium check points in the ring r ∈ [2·r_hole, 3·r_hole] (first quadrant).
/// These are far enough from the hole surface that ±fd_h meta-shifts never clip inside the hole.
pub fn sample_eq_ring(geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]> {
    let HoleType::Circular { radius } = geom.hole else { return Vec::new() };
    if n == 0 { return Vec::new(); }
    let r_inner = radius * EQ_RING_INNER_FACTOR;
    let r_outer = radius * EQ_RING_OUTER_FACTOR;
    let mut rng = LcgRng::new(SEED_EQ_RING);
    (0..n).filter_map(|i| {
        let angle = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
        let r = r_inner + (r_outer - r_inner) * rng.next_f64();
        let p = [r * angle.cos(), r * angle.sin()];
        if geom.contains(p[0], p[1]) { Some(p) } else { None }
    }).collect()
}

/// Simple LCG pseudo-random number generator (no external dependency)
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }
}
