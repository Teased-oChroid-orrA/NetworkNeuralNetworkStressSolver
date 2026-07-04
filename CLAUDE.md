# PINN Structural Stress Solver

A physics-informed neural network (PINN) solver for structural boundary-value problems, built
on `burn` (ML framework) + `egui`/`wgpu` (GUI). Ships two problems: **Kirsch** (a plate with a
circular hole under remote tension; validation target K_t = 3.0 at the hole boundary) and
**pin-in-lug** (a two-domain pin/lug contact-mechanics problem with a Signorini contact
interface; headless-only, `--problem pinlug`).

Workspace crates: `pinn-core` (geometry/material/sampling, no ML deps), `pinn-solver`
(training loop, optimizer, losses), `pinn-gui` (egui panels), `pinn-app` (binary; `--headless`
for terminal-only training, no GUI).

## Pluggable boundary-value problems

`pinn_solver::problem::BoundaryValueProblem` is the trait a new problem implements to reuse the
generic training/optimizer/convergence machinery without touching it: `domains()` (one
`DomainSpec` per physical domain — Kirsch has 1, pin-lug has 2), `sampling_strategy()`/
`ansatz()` per domain, `loss_terms()` (a `Vec<Box<dyn LossTerm>>` — the stable order is the
SAW-BRDR component vector), `convergence_metric()` (a plain `Option<f64>`; Kirsch returns K_t,
pin-lug returns interface-gap RMS — `ConvergenceTracker` neither knows nor cares which). A
`LossTerm` declares which domain(s) and named point-set(s) it needs (`domains()`/
`point_sets()`); cross-domain terms (e.g. pin-lug's Signorini penetration/non-tension
penalties, `pinn_solver::signorini`) declare more than one domain and receive both domains'
forward-pass outputs in `compute()`.

**Two step-driver functions, not one, by design.** `training_core::step_physics` is the frozen,
byte-proven single-domain path Kirsch runs through — it predates the trait, is regression-tested
bit-for-bit against the pre-trait hardcoded implementation, and must never be refactored into a
thin wrapper around the newer path (that would remove the independent code path the regression
test relies on to catch a future mistake). `training_core::step_physics_multi` is the additive,
N-domain generalization pin-in-lug uses; a `step_physics_multi_single_domain_matches_
step_physics_kirsch` test proves the two agree for N=1. Both preserve the same invariant: every
loss term is summed into one SAW-BRDR-weighted scalar, `.backward()` is called exactly once, and
gradients are split back to each domain's own optimizer via `GradientsParams::from_params` keyed
by that domain's own `ParamId`s (globally unique per `burn` `Param`, so domain iteration order
during the split is irrelevant — see `gradient_split_attributes_domain_b_step_only_to_domain_b_
params` for the adversarial proof this can't cross-contaminate).

Every `LossTerm`/reference-scale field must be normalized to O(1) before SAW-BRDR weighting
(divide by a computed `ref_energy`/`ref_stress2`/similar physical scale) — a term left at raw
Pa/Pa²/m² magnitude will dominate the SAW-BRDR total by many orders of magnitude and silently
starve every correctly-normalized term of gradient signal. `pinn.env`'s generic
`MATERIAL_*`/`LOAD_*`/`GEOM_*` overrides are Kirsch-tuned and must not be applied to problems
whose material/geometry/load are fixed as part of the problem definition (see `apply_env`'s
`skip_problem_specific` parameter in `pinn-app/src/main.rs`).

## Units

Internal storage is always SI: stress/modulus in **Pa**, length in **m**. The UI and console
output display **US Customary** units instead — psi/ksi/Msi for stress and modulus, inches
for length. `pinn_core::units` (`IN_TO_M`, `PSI_TO_PA`, `KSI_TO_PA`, `MSI_TO_PA`) is the single
conversion source of truth; every display-layer conversion should go through it rather than
hand-rolled literals. `Px`/`Py` (`pinn_core::loading::LoadConfig`) are far-field *stress*
(traction) boundary conditions — not forces — which is why they're in Pa/ksi like any other
stress quantity, not N/lbf.

## Reference-scale normalization

Every loss term is normalized to O(1) before SAW-BRDR weighting (see `compute_reference_scales`
in `pinn-solver::training_core`) by dividing by a physical reference stress `P` squared
(`ref_stress2 = P²`) and its energy-scale derivative (`ref_energy = 0.5*P²/E`); the length
reference is always `config.geometry.half_w` — that domain's own `GeometryConfig`, never a
shared/hardcoded constant, since a different problem's domain has a different characteristic
length. By default `P` is the applied far-field load (`config.load.px` for Kirsch,
`equivalent_traction_pa` for pin-lug) — the convention this codebase's K_t=3.0 validation and
pin-lug's tuned SAW-BRDR/LR/`ConvergenceTracker` thresholds were established against.

Setting `SolverConfig::use_ultimate_strength_scaling = true` (default `false`, opt-in) switches
`P` to `config.material.ultimate_strength_pa` (`PinLugProblem` takes the equivalent
`PinLugScalingMode::UltimateStrength` instead, since it has no `SolverConfig` of its own) — the
material's ultimate tensile strength (`MaterialProps::al7075_t6()`: 83,000 psi, ASTM B209
minimum spec; `MaterialProps::steel_4340()`: 200,000 psi, a representative quenched-and-tempered
condition — actual 4340 UTS spans 125,000–287,000 psi by temper, unlike `e`/`nu` which are
treatment-invariant). This makes the reference stress **itself** normalize to exactly 1.0 by
construction (it's the denominator: `P²/P² = 1`) — it is **not** a claim that the solved stress
field will reach or approach the material's ultimate strength. Changing this flag changes every
normalized loss term's magnitude by roughly `(P_load/F_c)²` and has not been validated against
the existing training hyperparameters — treat it as an experimental alternate scaling, not a
drop-in improvement.

`MaterialProps::dimensionless_modulus(f_c) -> f64` (`E/f_c`) is a pure diagnostic helper only —
it is never wired into `energy.rs`'s constitutive law (`compute_stress`/`compute_strains`).
Nondimensionalization lives only at the reference-scale/loss-normalization layer described
above; the constitutive law itself always operates on physical SI values.

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

`pinn_solver::controllers::ConvergenceTracker` drives Phase-2 warm restarts when a problem's
`convergence_metric()` (a plain `f64` — K_t for Kirsch, interface-gap RMS for pin-lug; the
tracker itself is problem-agnostic and was already generic before the trait existed) plateaus
or crashes. The plateau-comparison window (`PLATEAU_WINDOW = 20`) is coupled to the training
schedule (AMR sweep interval) and must not be changed independently — see the comment at its
definition. Plateau and crash restarts draw from separate 4-restart budgets
(`MAX_PLATEAU_RESTARTS`, `MAX_CRASH_RESTARTS`); both feed the same `lam_h_cap`/`lam_d_cap`
decay cascade (50 → 30 → 18 → 15) regardless of which budget fired.
