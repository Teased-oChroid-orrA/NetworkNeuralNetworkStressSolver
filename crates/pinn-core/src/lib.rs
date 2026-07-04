pub mod amr;
pub mod geometry;
pub mod kirsch;
pub mod loading;
pub mod material;
pub mod messages;
pub mod problem;
pub mod sampling;
pub mod state;
pub mod units;

pub use geometry::{GeometryConfig, HoleType, SymmetryMode};
pub use kirsch::{kirsch_stress, stress_concentration_factor};
pub use loading::{BoundaryKind, BoundaryPoint, LoadConfig};
pub use material::{LameConsts, MaterialProps};
pub use messages::{
    ControlMsg, DecisionMakerConfig, PinLugTrainingUpdate, PinLugVisFields, ProblemKind,
    SolverConfig, StiffnessConfig, TrainingMsg, TrainingUpdate, VisFields,
};
pub use problem::{
    DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec, InterfaceParametrization,
    NamedPointSet,
};
pub use sampling::{CollocationSet, LcgRng, sample_eq_ring};
pub use state::{FieldType, SolverStatus, TrainingState};
