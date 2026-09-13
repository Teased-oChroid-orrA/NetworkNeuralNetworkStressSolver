# Phase 4 Implementation Manifest

Issue #63 is authoritative. Updated as work proceeds; `VERIFIED` requires the issue's
mathematical, source, test, runtime-artifact, and benchmark evidence.

## PH4-01 — Freeze and classify solver modes

Status: BLOCKED

Hypothesis: legacy Hybrid success and corrected Variational support are distinct modes.

Evidence before change: Phase 3 recorded Hybrid + legacy mean as operational for its narrow
no-hole configuration, and Variational + measure-aware as not operational. Source showed
Variational also had independently weighted `interior_energy` and `external_work` terms.

## PH4-02 — Live mathematical objective audit

Status: IMPLEMENTED

Hypothesis: live SAW-BRDR/base weighting changed the physical `U:W_ext` ratio.

Evidence before change: `UserDefinedProblem::loss_terms()` returned separate terms and
`step_physics_multi()` independently multiplied every term by its SAW/base weight. The base
weights were `LAM_INTERIOR_ENERGY=1.0` and `LAM_EXTERNAL_WORK=20.0`.

Mathematical derivation and source mapping: `PHASE_4_MATHEMATICAL_OBJECTIVE_AUDIT.md`.

Change made: none in this audit item; PH4-03 owns remediation.

Tests/runtime artifact/benchmark/regression: pending PH4-03 through PH4-06.

Known limitation: historical Phase 3 Variational evidence used the invalid independent ratio;
it cannot establish corrected-Variational behavior.

Mode artifact directories are present. A/B/C notes explicitly retain blocked historical or
unexecuted states; no missing runtime evidence is silently synthesized.

## PH4-03 — Create atomic physical U-W functional

Status: IMPLEMENTED

Hypothesis: one `LossTerm` containing `U-W_ext` prevents adaptive/base weighting from changing
the physical coefficient ratio.

Change made: `PhysicalPotentialEnergyTerm` requests both live point sets and returns one tensor;
`FormulationSelection::Variational` activates it, while legacy Hybrid remains explicit and
unchanged for compatibility.

Tests: `cargo check -p pinn-solver --features ndarray-backend` passed. Focused production-path
tests passed: `physical_potential_is_one_live_atomic_u_minus_w_term` and all three
`variational_formulation_*` tests; the latter executes `step_physics_multi` with the live
Variational term set.

PH4-03 affine acceptance: `physical_potential_live_term_has_the_correct_affine_minimizer`
constructs actual interior/boundary `DomainForwardOutputs` in production point-set order and
proves the atomic term has its minimum and zero numerical derivative at `a=sigma0/E`.

Runtime artifact/benchmark: real corrected Variational execution and persisted Phase 4 artifact
remain pending PH4-06. Not VERIFIED.

## PH4-04 — Live integral unbiasedness

Status: INVESTIGATING

Evidence before change: PH3 proved helper and separate `InteriorEnergyTerm`/`ExternalWorkTerm`
branches, but did not exercise one atomic production Pi term. PH4-03's focused live-term test
now proves the measure-aware `PhysicalPotentialEnergyTerm` uses both real domain and boundary
integral primitives. `physical_potential_live_measure_aware_weights_remove_nonuniform_interior_bias`
executes the actual atomic term with a deliberately biased AMR compensation shape and verifies
its weighted result against the differentiable physical integral. Uniform/nonuniform/AMR real
training artifacts remain required before VERIFIED.

## PH4-05 — Physical versus optimization values

Status: IMPLEMENTED

Evidence before change: `EnergyBalance` persisted internal energy and half prescribed work for
the linear energy-balance diagnostic, but no report exposed full `W_ext`, `Pi`, normalization,
active terms, or live SAW-BRDR weights. Its pre-PH4 probe also used model-derived traction,
which is not the prescribed-load work in the variational functional.

Mathematical derivation: linear energy balance uses `U = W_ext/2`; potential minimization uses
`Pi = U-W_ext`. The report therefore reconstructs full work as twice the diagnostic half-work,
not by changing the energy-balance convention.

Change made: `probe_energy_balance` now integrates prescribed `tbar dot u`. New serialized
`MathematicalObjectiveSnapshot` distinguishes physical/normalized `U`, full `W_ext`, and `Pi`;
records reference energy, geometry measures, estimator, active terms, base weights, constraints,
last live adaptive weights, and optimizer policy. Plate checkpoint and GUI export use the same
`build_plate_authoritative_report` producer.

Files changed: `user_problem.rs`, `provenance.rs`, `runner.rs`, downstream
`app-egui/src/stress_solver.rs`.

Tests: focused prescribed-work test and snapshot reconstruction/JSON round-trip pass. Solver
`cargo check` passes. Downstream `cargo check` and `cargo test` pass.

Integration evidence (2026-09-12): solver commit
`cacdcd86b8dc01a1f582bde87e63370d4521e0ce`, dirty; powershell_tool commit
`0500a6ad62c0923378769fbeb9fea4e8c1583919`, dirty only at required
`app-egui/src/stress_solver.rs` compatibility call-site. `app-egui/Cargo.lock` SHA-256:
`3aba2a2dd087c976d5817c4cfd50cb04651555bc6180317fda927a6a0e3f3cdf`.
Commands passed: `cargo check`, `cargo test`, and `cargo build --release` in
`../powerShell/powershell_tool/app-egui`.

Runtime artifact/benchmark/regression: checkpoint-producing corrected-Variational run remains
pending PH4-06. Not VERIFIED.

## PH4-06 — Variational divergence root cause

Status: INVESTIGATING

Runtime evidence: `Debug_run/phase4/D_variational_atomic_pi/` records the controlled full-NN,
uniform, AMR-off, 15-step run after atomic Pi and rotational gauge changes. It is finite and
loss decreases, but L4 fails decisively (`load_transfer_ratio=0.1996`; all five hard metrics
fail). This rules out treating finite loss or a short decreasing trajectory as convergence.
Its recorded energy-balance value predates PH4-05's prescribed-traction correction and must not
be used as physical-Pi evidence. Next ladder comparison must change one variable only.

Post-PH4-05 rerun: same 15-step D command completed. Correct prescribed-work energy-balance
error is `1.7482`; L4 still fails all five hard metrics. Runtime evidence is diagnostic only.

New source finding: pure Variational was still receiving synthetic, fixed-weight
`constitutive_consistency` although it declared no direct-stress consumer. This hidden
constraint is now suppressed unless active terms explicitly consume direct stress. Focused
live regression asserts no `constitutive_consistency` gradient entry in pure Variational.

Additional source trace: `compute_domain_forwards` applies `IdentityAnsatz` (unit factors),
scales displacement by `u_ref`, scales direct stress by `px`, and converts normalized FD
derivatives with `FdConfig.sx/sy`; `PhysicalPotentialEnergyTerm` consumes the same first `n`
rows and shared point-set measures. No scaling or sample-order mismatch was found in this
static trace. Runtime optimization failure therefore remains unresolved rather than being
silently attributed to quadrature or sign error.

Controlled runtime controls: 1,000 steps with `fd_h=1e-2` and fixed unit
`physical_potential` weight reached normalized `Pi=-1.02345`, L4 health pass
(`energy_balance_error=1.61e-3`), but load transfer `0.8336` and stress/traction hard metrics
failed. Increasing only outer-boundary sampling to 1024 points reached normalized
`Pi=-1.02045`, L4 health pass, load transfer `0.8781`, and still failed hard metrics. A
3,000-step 512/256 run reached `Pi=-1.04625` and load transfer `0.6053`. These controls rule
out simple optimizer duration, SAW multiplier drift, and boundary sample count as sufficient
fixes; L4 remains blocked.

Capacity control (`hidden_dim=32`, `n_hidden=3`, otherwise identical to the 1024-boundary
control) reached normalized `Pi=-1.02487`, L4 health pass (`1.93e-3`), but load transfer
`0.8044` and stress/traction hard metrics failed. Larger capacity alone does not recover the
boundary field.

Follow-up tracking: GitHub sub-issue [#64](https://github.com/Teased-oChroid-orrA/NetworkNeuralNetworkStressSolver/issues/64)
tracks field-recovery failure and candidate general neural residual/linear representation work.

Issue #64 architecture update: `ElasticityNet` now has a zero-initialized, trainable
`[x_norm, y_norm, z] -> [u, v]` coordinate-residual projection. Production `fwd` and
`fwd_masked` retain raw normalized coordinates while the MLP continues to receive its optional
Fourier embedding. The residual is added only to displacement columns; direct stress columns
remain MLP-produced. This is geometry-independent and leaves the embedded-input `forward` API
unchanged. The residual weight and bias participate in ordinary optimizer gradient collection;
width/depth growth and pruning retain them unchanged. New checkpoints persist this branch; a
legacy-record fallback loads old checkpoints with an exactly zero residual. Focused checkpoint
round-trip and legacy-load tests pass, and `cargo check -p pinn-app --features ndarray-backend`
passes. The WGPU-only network unit suite cannot execute on this host because no Metal adapter is
available. No L4 result is claimed: controlled runtime ladder and field acceptance remain open.

Issue #64 controlled H result: coordinate residual was tested for 1,000 steps against F's fixed
seed, corrected objective, `fd_h=1e-2`, 512 interior, 256 boundary, AMR-off configuration.
It improved health (`energy_balance_error=2.7854e-4`) but failed every hard L4 metric:
`sigma_xx_relative_error=0.1525`, `sigma_yy_over_ref=0.0410`,
`sigma_xy_over_ref=0.0239`, `traction_rms_over_ref=0.1231`, and load transfer `0.8049`.
F MLP-only measured load transfer `0.8838` and sigma-x error `0.1196`; H is not a winning
configuration and is not extended. Evidence: `Debug_run/phase4/H_coordinate_residual/`.

Production change: `step_physics_multi` now fixes the atomic Variational term multiplier at
unit scale; constraint terms remain adaptive. Focused objective tests and the full user-problem
focused suite pass. Runtime controls show this removes multiplier drift but does not satisfy
hard L4, so item remains INVESTIGATING.

Controlled rerun changing only that condition is persisted in D notes: total loss drops from
`14.08062` to `0.361697` at step zero, proving the hidden constraint dominated optimizer loss;
however corrected physical metrics remain effectively unchanged after 15 steps
(`load_transfer=0.1990`, energy balance `1.7463`, L4 five-of-five fail). Root cause remains
unresolved; no optimizer tuning has been applied.

E ladder evidence (same corrected path, only budget 15 to 200): Pi falls to `-1.898899e-1 J`,
but load transfer falls to `0.0699` and all hard L4 metrics fail. Final physical-Pi gradient
norm is `2.100251`; translation/rotation gauge norms are `1.46e-8`/`7.49e-9`, ruling out gauge
dominance in this run. Full stdout, config, and notes are persisted under
`Debug_run/phase4/E_variational_stabilized/`. Do not interpret its loose L4 health pass as a
hard benchmark pass.

The same E configuration was rerun after the dimensional correction to the translation gauge
only. At step 199, loss is `1.771334e-1`, physical-Pi gradient `2.181287`, translation-gauge
gradient `6.169782e-2`, and rotation-gauge gradient `2.548643e-9`; neither gauge dominates.
Physical results remain invalid: `U=2.799427e-1 J`, `W_ext=-7.325458e-1 J`,
`Pi=1.012488e0 J`, load transfer `0.0780`, health fails, and all five L4 hard metrics fail.
This is persisted in `E_variational_stabilized/solver.log` and `RUN_NOTES.md`. Next required
controlled ladder stage is restricted affine representation, not optimizer tuning.

Restricted-neural ladder stage is now executing from
`D_variational_atomic_pi/restricted_affine_neural.toml`: existing MLP only, two hidden units
and one hidden layer, with every other E setting identical. It is intentionally not an
analytical substitution or a benchmark-specific ansatz. It completed with
`Pi=-3.575255e-1 J`, physical gradient `6.821348e-1`, translation gradient `3.410605e-1`,
load transfer `0.0960`, L4 health failure, and five-of-five hard L4 failure. Persisted notes
and full output are `D_variational_atomic_pi/RESTRICTED_AFFINE_NEURAL_NOTES.md` and
`restricted_affine_neural.log`. Capacity alone is not root cause; optimizer contract is now
the next controlled candidate, without changing physical objective or benchmarks.

## PH4-07 — Gauge/nullspace compatibility

Status: IMPLEMENTED

Evidence before change: pure-Neumann user plates register `TranslationGaugeTerm`, which removes
two translations only. Source inspection found no rotational gauge, nullspace projection, or
point-constraint alternative.

Change made: `RotationGaugeTerm` uses outer-edge circulation to constrain mean infinitesimal
rotation. It is only added to corrected Variational pure-Neumann runs, preserving frozen legacy
Hybrid/Strong trajectories. `rotation_gauge_removes_only_rigid_rotation_not_affine_symmetric_
strain` proves rigid rotation is penalized and affine extension is not.

Runtime/benchmark proof and constraint-gradient dominance evidence remain required before
VERIFIED.

## PH4-08 — Variational optimizer contract

Status: INVESTIGATING

Controlled probe changes only peak learning rate from `1e-3` to `3e-3` at fixed full NN,
atomic Pi, gauges, uniform sampling, AMR-off, seed, and 200-step budget. It decreases sampled
normalized Pi from `+3.616969e-1` to `-1.767713e-1`; translation-gauge raw value falls from
`1.623270e-1` to `4.686160e-6`, and its final gradient (`6.311712e-3`) is far below physical
Pi (`2.010310`). Yet load transfer is `0.0868` and all L4 metrics fail.

Probe normalized Pi is `-1.804754e-1`; exact affine continuum minimum is `-1` under common
reference scaling, so this is non-stationary rather than a discrete lower-than-continuum
minimum. Next test changes uniform collocation resolution only. Full evidence:
`E_variational_stabilized/OPTIMIZER_PEAK_LR_3E-3_NOTES.md`.

Fourfold uniform-resolution control (`512/256` points, all other peak-LR settings fixed) ends
at normalized Pi `-1.538412e-1`, physical gradient `2.127576`, translation gradient
`3.486059e-3`, and load transfer `0.0836`; L4 health and all hard metrics still fail. Thus low
quadrature count alone is not root cause. AMR remains off. Evidence:
`E_variational_stabilized/UNIFORM_RESOLUTION_512_NOTES.md`.

`F_no_hole_final` now runs justified convergence gate: 1,000 steps at same corrected 512/256
uniform configuration and `3e-3` peak LR. One sequential solver process. Final log decides
whether Pi, physical gradients, gauges, energy balance, field errors, and load transfer
stabilize; no acceptance status assigned before evidence.

Run ended before final ledger at step 100; `F_no_hole_final/solver.log` is retained and marked
incomplete in `F_no_hole_final/RUN_NOTES.md`. Partial trend data cannot establish convergence.

## PH4-09 — AMR evidence

Status: BLOCKED

Blocking condition: Issue #63 requires AMR comparison only after mathematically correct uniform
Variational baseline. All corrected uniform controls still fail L4; AMR remains disabled by
policy. No AMR result may be used to rescue this baseline.

## PH4-10 — Formulation-aware convergence

Status: IMPLEMENTED

Change made: headless Variational runs now emit persisted cadence records for normalized Pi,
whole physical-block SAW weight, and translation-gauge raw value, in addition to existing final
physical/constraint gradient ledger, U/W/Pi, energy balance, field metrics, and load transfer.
The optimizer and 512-point controls exercise this live path in their solver logs.

Limitation: no run meets all convergence conditions; these records prevent a falling total loss
from being called convergence but do not establish it. Real passing L4 and regression evidence
remain required before VERIFIED.

## PH4-11 — DifferentialOperator production integration

Status: IMPLEMENTED

Evidence before change: declared FD-only production policy existed, but generic live forward
path called `fd_stencil::compute_strains` directly. Policy was descriptive, not a choke point.

Change made: `differential_operator::PRODUCTION_POLICY` declares FD as sole
weight-autodiff-safe backend. `production_strain` now drives
`training_core::compute_domain_forwards`, plus user diagnostics, legacy runner/headless, and
parametric production paths. AD stays an explicit
first-derivative diagnostic because Burn 0.21 returns detached input-gradient tensors and does
not support nested differentiation; Hessians remain FD-only.

Tests: six focused differential-operator tests pass under NdArray. Runtime proof for every
production formulation remains required before VERIFIED. Direct FD calls left in manufactured
fields and unit-test oracles are intentionally non-production.

## PH4-12 — Executable FieldKind

Status: IMPLEMENTED

Evidence before change: `FieldKind` supplied dependency-chain reporting only. Consumers could
slice raw output ad hoc, so constitutive field was not structurally distinct from auxiliary
direct stress.

Change made: `field_graph::resolve_field` resolves typed network output, displacement, strain,
constitutive stress, or direct mDEM stress and rejects missing prerequisites. Live atomic
potential requests displacement through resolver; free-hole traction requests direct stress.
Negative tests prove two-channel network cannot resolve direct stress and constitutive request
does not silently return direct stress.

Source correction: `evaluate_user_vis_grid` was classified and documented as constitutive but
published direct mDEM stress in engineering fields. It now publishes constitutive stress;
direct-minus-constitutive remains only explicit residual diagnostic. This removes a real
source-policy contradiction, not a benchmark correction.

Tests: 11 focused FieldKind tests pass under NdArray. Remaining production field consumers
need migration and real hole runtime evidence before VERIFIED.

Downstream compatibility after public resolver addition (2026-09-13): in
`../powerShell/powershell_tool/app-egui`, `cargo check` and `cargo test` passed; release binary
`target/release/app-egui` was rebuilt at `2026-09-13 05:24:14`. Final SHA/dirty/Cargo.lock
evidence remains consolidated under PH4-21.

## PH4-13 — Hole-boundary stress source

Status: BLOCKED

Source policy is implemented: boundary-limit Kt uses derived constitutive stress at an explicit
radial offset; direct mDEM stress remains a separate hole-BC diagnostic. Acceptance is blocked
by missing corrected no-hole L4 companion, so no Kt result is promoted.

## PH4-14 — Real L5

Status: BLOCKED

Blocking condition: Issue #63 requires a verified corrected-Variational L4 companion before
L5 acceptance. No such companion exists yet.

## PH4-15 — Independent displacement and strain validation

Status: BLOCKED

Implementation: added `validate_no_hole_fields`, an independent grid validator over published
displacement, strain, and constitutive stress fields. It reports L2/L∞ field errors plus rigid
translation and antisymmetric-gradient rotation residuals. Test
`no_hole_field_validation_accepts_affine_and_detects_translation` passes with an exact affine
field and rejects a unit rigid translation. This test is measure/source independent and does
not consume training loss.

Blocking condition: independent field errors require a verified corrected no-hole Variational
L4 companion. Current F run ended before final ledger; D/E controls fail L4 and cannot support
field acceptance.

## PH4-16 — Non-square geometry

Status: BLOCKED

Blocking condition: generalized geometry acceptance requires corrected no-hole field baseline;
only affine measure-level tests exist, so no non-square trained result is accepted.

## PH4-17 — Arbitrary topology/multiple holes

Status: BLOCKED

Blocking condition: hole and topology acceptance is gated by corrected no-hole L4; no accepted
companion exists. Existing source supports N holes but runtime proof is absent.

## PH4-18 — Formulation support matrix

Status: IMPLEMENTED

`docs/FORMULATION_SUPPORT_MATRIX.md` records only `VERIFIED`, `SUPPORTED_WITH_LIMITATION`, and
`EXPLICITLY_UNSUPPORTED`; it does not promote legacy Hybrid evidence to generalized support.
Runtime evidence remains insufficient for VERIFIED capability rows.

## PH4-19 — Mathematical objective snapshot

Status: IMPLEMENTED

Shared plate checkpoint and GUI-report builder now saves `MathematicalObjectiveSnapshot`; see
PH4-05. Backward JSON compatibility uses `#[serde(default)]`. A real checkpoint artifact from
corrected Variational training remains required before VERIFIED.

Regression: `serve_loaded_plate_checkpoint_saves_a_full_authoritative_report_for_a_no_hole_geometry`
now performs actual save/load JSON round-trip and asserts physical `U`, full `W_ext`, `Pi`,
active terms, and positive reference energy are present in persisted report.

## PH4-20 — General physical regressions

Status: INVESTIGATING

Focused atomic-objective, gauge, FieldKind, differential, provenance, and topology sampling
regressions pass. Full physical matrix remains pending corrected L4 and independent field data;
no benchmark-specific correction is permitted.

## PH4-21 — Final operational status

Status: BLOCKED

Blocking conditions remain: corrected Variational L4 fails; controlled ladder has not isolated
the remaining divergence; PH4-04/07-20 lack required runtime and independent benchmark proof.
Issue #63 forbids declaring OPERATIONAL or tuning around these failures.

Validation limitation: focused NdArray solver tests and compilation pass. A hard-coded WGPU
unit-test alias prevented CPU verification on this machine; it now uses the selected backend and
the formerly blocked energy tests pass. Full-suite completion has not yet been captured. Clippy
with `-D warnings` still fails pre-existing `pinn-core` warnings. Neither is treated as Phase 4
proof.
