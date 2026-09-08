pub mod amr;
pub mod beam_spec;
pub mod geometry;
pub mod inference_envelope;
pub mod kirsch;
pub mod loading;
pub mod material;
pub mod messages;
pub mod param_distance;
pub mod parametric_spec;
pub mod problem;
pub mod problem_spec;
pub mod sampling;
pub mod state;
pub mod units;
pub mod user_geometry;

pub use geometry::{GeometryConfig, HoleType, SymmetryMode};
pub use inference_envelope::{
    classify_inference, user_problem_parameter_envelope, InferenceClass, ParameterClass,
    ParameterEnvelope,
};
pub use parametric_spec::{ParamRange, ParametricProblemSpec};
pub use kirsch::{kirsch_stress, stress_concentration_factor};
pub use loading::{BoundaryKind, BoundaryPoint, LoadConfig};
pub use material::{LameConsts, MaterialProps};
pub use messages::{
    AmrSweepReport, BeamTrainingUpdate, ControlMsg, DecisionMakerConfig, DiagnosticsConfig,
    ExecutionConfig, ExecutionMode, HoleAnalysis, HoleBoundaryPoint, ParametricInferenceResult,
    ParametricTrainingUpdate, PerformanceProfile, PinLugTrainingUpdate, PinLugVisFields,
    ProblemKind, SolverConfig, StiffnessConfig, StressConcentration, TrainingMsg, TrainingUpdate,
    VisFields,
};
pub use problem::{
    DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec, InterfaceParametrization,
    NamedPointSet,
};
pub use sampling::{CollocationSet, LcgRng, sample_eq_ring};
pub use state::{FieldType, SolverStatus, TrainingState};
