pub mod bc;
pub mod controllers; // ConvergenceTracker is `pub` (see controllers.rs) for cross-crate curriculum reuse
pub mod decision_maker;
pub mod engine;
pub mod energy;
pub mod fd_stencil;
pub mod headless;
pub mod kirsch_problem;
pub mod lr_schedule;
pub mod network;
pub mod optim;
pub mod problem;
pub mod runner;
pub mod saw_brdr;
pub mod signorini;
pub mod stiffness;
pub mod training_core;

pub use runner::run_training;
pub use headless::run_headless;
