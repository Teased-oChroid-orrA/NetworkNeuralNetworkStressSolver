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

/// Shared angle parametrization for a multi-domain contact interface (e.g. pin-in-lug).
/// Burn-free — just the angles [radians]. Generated once by the owning problem's
/// constructor (e.g. `PinLugProblem::new`) and handed to BOTH sides' sampling strategies
/// so their `"interface"` named point-sets are angle-index-aligned: index `i` denotes the
/// same physical angle on both domains. This alignment is the load-bearing correctness
/// constraint for any cross-domain interface loss term (e.g. contact-gap penalties) that
/// zips the two domains' interface point-sets index-by-index — a per-domain-independent
/// sampler could produce different angle sets of possibly different lengths, silently
/// mismatching which points are compared as "the same physical location".
#[derive(Debug, Clone)]
pub struct InterfaceParametrization {
    pub thetas: Vec<f64>,
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

    /// Named collections of boundary points beyond the single flat list `sample_boundary`
    /// returns — e.g. a pin-in-lug domain's `"interface"` point-set (angle-aligned with its
    /// partner domain via a shared [`InterfaceParametrization`]). Defaulted to empty so
    /// existing single-domain strategies (e.g. Kirsch) keep compiling unchanged; only
    /// multi-domain problems whose loss terms need more than one flavor of boundary point
    /// need to override this.
    fn named_point_sets(&self, bnd_pts: &[crate::loading::BoundaryPoint]) -> Vec<NamedPointSet> {
        let _ = bnd_pts;
        Vec::new()
    }
}

/// Per-domain Dirichlet (hard) displacement-BC ansatz, evaluated pointwise so this crate
/// stays burn-free. `pinn-solver` broadcasts this into tensors column-by-column.
///
/// `(xn, yn)`: normalized coordinates in [-1,1]^2. `k`: ansatz saturation factor.
/// Returns the `(u, v)` scale factors the raw network output should be multiplied by.
pub trait DirichletAnsatz: Send + Sync {
    fn eval(&self, xn: f32, yn: f32, k: f32) -> (f32, f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loading::BoundaryKind;

    /// Two fake sampling strategies (standing in for pin/lug) fed the SAME
    /// `InterfaceParametrization` — proves index `i` denotes the same physical angle on
    /// both domains' `"interface"` point-set, rather than each domain independently
    /// sampling its own (possibly differently-sized/ordered) angle set.
    struct FakeInterfaceSampling {
        radius: f64,
        params: InterfaceParametrization,
    }
    impl DomainSamplingStrategy for FakeInterfaceSampling {
        fn sample_interior(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> { Vec::new() }
        fn sample_boundary(&self, _geom: &GeometryConfig, _load: &LoadConfig, _n: usize) -> Vec<crate::loading::BoundaryPoint> { Vec::new() }
        fn amr_lock_zone(&self, _geom: &GeometryConfig, _cell_center: [f64; 2]) -> bool { false }
        fn sample_extra_ring(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> { Vec::new() }
        fn named_point_sets(&self, _bnd_pts: &[crate::loading::BoundaryPoint]) -> Vec<NamedPointSet> {
            let points = self.params.thetas.iter().map(|&theta| {
                let x = self.radius * theta.cos();
                let y = self.radius * theta.sin();
                crate::loading::BoundaryPoint {
                    x, y, nx: theta.cos(), ny: theta.sin(), tx: 0.0, ty: 0.0,
                    kind: BoundaryKind::Interface { partner_domain: DomainId(999) },
                }
            }).collect();
            vec![NamedPointSet { name: "interface", points }]
        }
    }

    #[test]
    fn interface_point_sets_are_index_aligned_by_shared_theta_not_independent_sampler_size() {
        let thetas: Vec<f64> = (0..8).map(|i| i as f64 * std::f64::consts::PI / 8.0).collect();
        let params = InterfaceParametrization { thetas: thetas.clone() };

        let pin = FakeInterfaceSampling { radius: 0.5, params: params.clone() };
        let lug = FakeInterfaceSampling { radius: 0.6, params: params.clone() };

        let pin_sets = pin.named_point_sets(&[]);
        let lug_sets = lug.named_point_sets(&[]);

        let pin_iface = &pin_sets.iter().find(|s| s.name == "interface").unwrap().points;
        let lug_iface = &lug_sets.iter().find(|s| s.name == "interface").unwrap().points;

        assert_eq!(pin_iface.len(), 8);
        assert_eq!(lug_iface.len(), 8);

        for i in 0..8 {
            let pin_theta = pin_iface[i].y.atan2(pin_iface[i].x);
            let lug_theta = lug_iface[i].y.atan2(lug_iface[i].x);
            assert!(
                (pin_theta - lug_theta).abs() < 1e-9,
                "index {i}: pin_theta={pin_theta} lug_theta={lug_theta} must match (shared parametrization)"
            );
            assert!((pin_theta - thetas[i]).abs() < 1e-9);
        }
    }
}
