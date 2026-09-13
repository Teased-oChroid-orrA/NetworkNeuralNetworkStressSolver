# Phase 4 Mathematical Objective Audit

Audit date: 2026-09-12. Source examined: `user_problem.rs`, `training_core.rs`,
`measure_integral.rs`, `problem_spec.rs`.

For plane stress elasticity with thickness `t`, applied traction `tbar`, and displacement `u`,

`Pi[u] = U[u] - W_ext[u]`

`U = integral_Omega (1/2 sigma:epsilon) t dA`

`W_ext = integral_Gamma_t (tbar dot u) t ds`.

| Source expression | Mathematical expression | Units before normalization | Physical coefficient | Optimization coefficient | Expected magnitude |
| --- | --- | --- | --- | --- | --- |
| `dem_energy_per_point` | `1/2 sigma:epsilon` | Pa = J/m3 | `+1` | formerly independent `lam(interior_energy)` | O(`P²/E`) |
| `domain_integral*_tensor(area,t,...)` | `U` | J | `+1` | one `lam(physical_potential)` | O(`P² A t/E`) |
| boundary `px*nx*u + py*ny*v` | `tbar dot u` | J/m2 | `+1` inside `W_ext` | one `lam(physical_potential)` | O(`P² A/E`) |
| `boundary_integral*_tensor(...,t)` | `W_ext` | J | `-1` in `Pi` | one `lam(physical_potential)` | O(`P² A t/E`) |
| `PhysicalPotentialEnergyTerm::compute` | `(U-W_ext)/E_ref` | dimensionless | `+1:-1` | one scalar | O(1) |
| `TranslationGaugeTerm`, fixed-hole penalty | admissibility constraint | m2 before own scaling | not part of Pi | independent constraint lambda | configuration dependent |

## Historical defect

Before PH4-03, `loss_terms()` returned `interior_energy` and `external_work` separately for
`FormulationSelection::Variational`. `step_physics_multi()` applied `raw * lam_by_name[name]`
for every returned term. Therefore live objective was

`lambda_U U_norm + lambda_W (-W_norm) + constraints`,

with base seeds `lambda_U=1` and `lambda_W=20`, and later independent SAW adaptation. This is
not a uniform rescaling of Pi and changes its stationary point. PH4-03 replaces this only for
the Variational formulation with one atomic tensor before SAW is applied.

The old mean-integral path has a second, independent defect for a Variational functional:
`mean(energy density)` is Pa while `mean(traction dot displacement)` is Pa·m. It neither
contains the matching domain/boundary measures nor has compatible dimensions. Consequently
`Variational` now fails loudly unless `training.measure_aware_training=true`; Hybrid legacy
compatibility remains available but is not a valid generalized Variational mode.

## Affine reduction

For no-hole `u=a x`, `v=-nu a y` under uniaxial `sigma0`, plane stress gives
`sigma_xx=E a`, `sigma_yy=sigma_xy=0`. For rectangular area `A`,

`U(a) = 1/2 E A t a²`,

`W_ext(a) = sigma0 A t a`,

so `Pi(a) = C a² - D a`, where `C=1/2 E A t` and `D=sigma0 A t`.
Thus `dPi/da=2Ca-D=0` and `a*=sigma0/E`.

`verification_ladder::run_affine_amplitude_test` independently implements this reduction using
the same differentiable measure-aware integral primitives. PH4-03 additionally executes
`PhysicalPotentialEnergyTerm` through live multi-point-set term machinery. PH4-06 must execute
the corrected live production term and persist a ladder result before this audit can be VERIFIED.

## Coordinate and normalization audit

Physical geometry is SI metres. Sampling is normalized only for network input; live output is
scaled before strain/stress evaluation in `compute_domain_forwards`. `ref_energy` is a density
scale (`0.5 P²/E`); measure-aware terms divide absolute joules by
`ref_energy * domain_area * thickness`. Boundary `ds` follows actual edge length and includes
thickness. AMR compensation weights apply only to interior density estimates.

## Diagnostic-versus-functional work convention

`EnergyBalance.external_work` is deliberately `W_ext/2`: in linear elastic loading the
work-energy diagnostic checks `U = W_ext/2`. It cannot itself be inserted into `Pi`. PH4-05
therefore persists `MathematicalObjectiveSnapshot.physical_w_ext = 2 *
EnergyBalance.external_work` and `physical_pi = physical_u - physical_w_ext`, alongside the
normalized values and the last live SAW-BRDR term weights. `probe_energy_balance` now uses
prescribed boundary traction, never model-derived traction, for its work integral.

## Nullspace finding

For pure Neumann elasticity the rigid-body nullspace has two translations and one rotation.
The live `TranslationGaugeTerm` constrains only mean `u` and mean `v`; it does not constrain
rigid rotation. Its input contains symmetric strain, which intentionally discards the
antisymmetric derivative needed to measure rotation. PH4-07 therefore remains open rather than
adding a displacement-moment penalty that could incorrectly constrain physical affine strain on
asymmetric topology.

## PH4-06 hidden-constraint finding

`step_physics_multi` previously injected `constitutive_consistency` for every mDEM network,
including pure Variational runs whose declared terms never consume the network's direct stress
channels. In corrected Variational this was an undeclared auxiliary-stress constraint with a
fixed weight of 50, outside SAW-BRDR and outside `UserDefinedProblem::loss_terms()`. It did not
change the formal continuum stationary displacement when exactly satisfiable, but it changed the
finite-network optimization problem and violated the declared atomic-objective contract.

The live driver now injects this constraint only when an active term actually has
`StressSource::Direct` or `StressSource::Both`. Thus it remains present for explicit mixed
formulations that need direct/constitutive reconciliation, but is absent from pure Variational
Pi plus gauge constraints. The PH4-06 controlled rerun remains pending.
