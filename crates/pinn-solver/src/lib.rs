pub mod bc;
pub mod contact_export;
pub mod controllers; // ConvergenceTracker is `pub` (see controllers.rs) for cross-crate curriculum reuse
pub mod decision_maker;
pub mod engine;
pub mod energy;
pub mod execution;
pub mod fd_stencil;
pub mod headless;
pub mod kirsch_problem;
pub mod lr_schedule;
pub mod network;
pub mod optim;
pub mod pinlug_problem;
pub mod problem;
pub mod runner;
pub mod saw_brdr;
pub mod signorini;
pub mod stiffness;
pub mod training_core;

pub use runner::{run_training, run_training_pinlug};
pub use headless::{run_headless, run_headless_pinlug};
