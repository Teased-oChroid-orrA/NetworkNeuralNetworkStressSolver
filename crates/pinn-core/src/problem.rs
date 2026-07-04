/// Generic boundary-value-problem trait family (geometry/material/sampling layer, no ML
/// deps). Lets `pinn-solver` drive an arbitrary number of physical domains and loss terms
/// through a common interface instead of the training loop being hardwired to the
/// single-domain Kirsch problem.
///
/// This crate stays burn-free — `DirichletAnsatz::eval` and `DomainSamplingStrategy`'s
/// methods work on scalars / `Vec<[f64;2]>`, not tensors. `pinn-solver::problem` is where
/// tensor-broadcasting and the `BoundaryValueProblem`/`LossTerm` (tensor-graph) traits live.
use crate::{geometry::GeometryConfig, loading::LoadConfig, material::MaterialProps};

/// Identifies one physical domain within a (possibly multi-domain) boundary-value problem.
/// E.g. Kirsch has exactly one domain; a pin-in-lug contact problem has two (pin + lug).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainId(pub u32);

/// Static specification of one domain: its geometry, material, and network output width.
#[derive(Debug, Clone)]
pub struct DomainSpec {
    pub id: DomainId,
    pub geometry: GeometryConfig,
    pub material: MaterialProps,
    /// Network output dimension for this domain (3 = plain DEM u,v,w; 5 = mDEM
    /// u,v,sigma_xx,sigma_yy,sigma_xy).
    pub output_dim: usize,
}

/// A named collection of boundary points — e.g. "hole", "load_edge", "interface" — used
/// wherever a sampling strategy needs to hand back more than one flavor of boundary point.
pub struct NamedPointSet {
    pub name: &'static str,
    pub points: Vec<crate::loading::BoundaryPoint>,
}

/// Per-domain collocation-point sampling. Implementations own all seeding/RNG so that
/// migrating an existing problem (e.g. Kirsch) onto this trait can reproduce bit-identical
/// output by copying the existing sampling bodies verbatim.
pub trait DomainSamplingStrategy: Send + Sync {
    /// Interior collocation points (physical coords, [m]) for this domain.
    fn sample_interior(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]>;

    /// Boundary collocation points (with normals/tractions) for this domain.
    fn sample_boundary(&self, geom: &GeometryConfig, load: &LoadConfig, n: usize) -> Vec<crate::loading::BoundaryPoint>;

    /// True if the given collocation-grid cell center must be locked at the AMR grid's
    /// minimum refinement level (e.g. near a hole or contact interface) regardless of its
    /// residual history.
    fn amr_lock_zone(&self, geom: &GeometryConfig, cell_center: [f64; 2]) -> bool;

    /// Extra ring of points used for equilibrium-residual (or other auxiliary) checks —
    /// generalizes Kirsch's near-hole equilibrium ring to any domain.
    fn sample_extra_ring(&self, geom: &GeometryConfig, n: usize) -> Vec<[f64; 2]>;
}

/// Per-domain Dirichlet (hard) displacement-BC ansatz, evaluated pointwise so this crate
/// stays burn-free. `pinn-solver` broadcasts this into tensors column-by-column.
///
/// `(xn, yn)`: normalized coordinates in [-1,1]^2. `k`: ansatz saturation factor.
/// Returns the `(u, v)` scale factors the raw network output should be multiplied by.
pub trait DirichletAnsatz: Send + Sync {
    fn eval(&self, xn: f32, yn: f32, k: f32) -> (f32, f32);
}
