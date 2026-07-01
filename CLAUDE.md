# PINN Structural Stress Solver

A physics-informed neural network (PINN) solver for the Kirsch problem — a plate with a
circular hole under remote tension — built on `burn` (ML framework) + `egui`/`wgpu` (GUI).
Validation target: stress concentration factor K_t = 3.0 at the hole boundary.

Workspace crates: `pinn-core` (geometry/material/sampling, no ML deps), `pinn-solver`
(training loop, optimizer, losses), `pinn-gui` (egui panels), `pinn-app` (binary; `--headless`
for terminal-only training, no GUI).

## Units

Internal storage is always SI: stress/modulus in **Pa**, length in **m**. The UI and console
output display **US Customary** units instead — psi/ksi/Msi for stress and modulus, inches
for length. `pinn_core::units` (`IN_TO_M`, `PSI_TO_PA`, `KSI_TO_PA`, `MSI_TO_PA`) is the single
conversion source of truth; every display-layer conversion should go through it rather than
hand-rolled literals. `Px`/`Py` (`pinn_core::loading::LoadConfig`) are far-field *stress*
(traction) boundary conditions — not forces — which is why they're in Pa/ksi like any other
stress quantity, not N/lbf.

## Optimizer

Weight matrices (2D) in `ElasticityNet`'s `Linear` layers are trained with a custom
**SOAP-Muon hybrid** optimizer (`pinn_solver::optim::SoapMuon`); biases (1D) go through plain
`AdamW`. `pinn_solver::optim::WeightOptim` is the runtime-selectable wrapper — set
`SolverConfig::use_soap_muon = false` to fall back to AdamW-only training for every parameter
if the hybrid proves unstable on a given configuration.

The algorithm (ported from `github.com/nikhilvyas/SOAP` and `github.com/nikhilvyas/SOAP_MUON`,
per "Improving SOAP Using Iterative Whitening and Muon", Vyas et al.) is **not** a per-dimension
split between the two optimizers. Each step: SOAP maintains Shampoo-style per-dimension
preconditioners, projects the gradient into their eigenbasis, runs a standard Adam update in
that rotated space, and projects back — then Muon's Newton-Schulz orthogonalization is applied
to the resulting update as a refinement pass. The eigenbasis is refreshed via a full
`nalgebra` eigendecomposition every `precondition_frequency` steps (simplified from the
original's power-iteration+QR approximation, which exists there to amortize cost on
LLM-scale matrices — this network's weights are at most `hidden_dim`×`hidden_dim`, where a
full eigh is microseconds).

`burn`'s native `Muon` optimizer (added in burn 0.21) is not used directly — it's a
self-contained optimizer, not a composable building block — but its Newton-Schulz defaults
(`ns_coefficients`, `ns_steps`) are mirrored for consistency.

## Convergence cascade

`pinn_solver::controllers::ConvergenceTracker` drives Phase-2 warm restarts when K_t plateaus
or crashes. The plateau-comparison window (`PLATEAU_WINDOW = 20`) is coupled to the training
schedule (AMR sweep interval) and must not be changed independently — see the comment at its
definition. Plateau and crash restarts draw from separate 4-restart budgets
(`MAX_PLATEAU_RESTARTS`, `MAX_CRASH_RESTARTS`); both feed the same `lam_h_cap`/`lam_d_cap`
decay cascade (50 → 30 → 18 → 15) regardless of which budget fired.
