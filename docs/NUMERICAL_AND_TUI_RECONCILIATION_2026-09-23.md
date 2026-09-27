# Numerical and TUI reconciliation — 2026-09-23

## Scope and repository state

Implementation branch: `feature/pinn-numerics-tui-audit-20260923`, based on
`e3a55a0f5f1be131d093027fe9222d6c05af8f87`. At branch creation, local `main`, cached
`origin/main`, and live GitHub `main` all matched that commit. Existing deleted and untracked
`Debug_runs` files were preserved and remain outside this work's commits.

Downstream `powershell_tool/app-egui` work is isolated on
`feature/pinn-numerics-tui-integration-20260923`, based on
`0500a6ad62c0923378769fbeb9fea4e8c1583919`. Its pre-existing uncommitted
`app-egui/src/stress_solver.rs` changes were recorded before this work and preserved.

GitHub reports no repository rulesets and returns `404 Branch not protected` for `main`.
Changing repository administration requires separate approval. This implementation never
targets `main`.

## Negative loss and gradient finding

The user observed negative `total_loss`, `energy_loss`, and `neumann` plus high `grad_norm` on
fresh TUI and GUI runs, including `notched_plate.toml` and no-hole plates. Both interfaces call
the same `runner::run_training_user_problem` path, so matching readings are expected.

The old scalar labels do not have one stable mathematical meaning across all paths:

- `total_loss` is the signed weighted objective. It may legitimately be negative when it
  contains potential energy `Pi = U - W_ext`.
- `energy_loss` is a raw normalized optimizer term. Under the atomic Variational path this can
  itself be the signed `physical_potential`; under legacy Hybrid it is normally nonnegative
  strain energy.
- On the generic plate path, `neumann_loss` was assigned as
  `total_loss - energy_loss`. It can therefore include external work, constraints,
  constitutive consistency, and every other active term. It is not a Neumann traction-residual
  invariant and may be negative.
- A true squared Neumann traction residual is nonnegative apart from negligible roundoff.
- `grad_norm` is the L2 norm of model-parameter gradients before the optimizer update. Its
  magnitude is scale- and objective-dependent. A high finite value alone is not a defect.

Authoritative telemetry now transports the solver's existing loss ledger: named normalized raw
values, effective weights, weighted contributions, total, reconstruction difference, optimizer
tier, and pre-optimizer gradient norm. The TUI persists this data per run and shows signed
values without clamping. Physical probes separately show U, full load-potential work, Pi,
force balance, boundary residuals, and per-hole Kt when available.

`EnergyBalance.external_work` historically stores proportional-loading work
`0.5 * integral(t dot u)`, whose equilibrium value matches U. The optimizer's potential uses
the full load work. `EnergyBalance::load_potential_work()` and `physical_potential()` make that
factor explicit without changing serialized data or training mathematics.

## Preserved mathematical baseline

- Variational training retains one atomic `physical_potential = U - W_ext` term.
- Measure-aware integration and per-step resampling remain unchanged.
- SAW-BRDR uses absolute magnitudes only for decay-rate adaptation; it does not make the signed
  physical functional positive.
- Documented no-hole Variational L4 evidence and single-hole hard-constraint Kt evidence around
  2.43–2.44 remain the regression standard.
- Multi-hole machinery remains operational, but trained Free-hole Kt remains roughly 10–20%
  below corrected FEM evidence, with a larger Fixed-hole gap. Existing controlled experiments
  reject blind density, width, learning-rate, embedding, and sampling changes as solutions.
  This capability remains experimental until formulation and local stress reconstruction pass
  an independent acceptance gate.

## Entry-point reconciliation

| Entry point | Training path | Status |
|---|---|---|
| User plate GUI | `runner::run_training_user_problem` | Authoritative streaming path |
| User plate TUI | same as GUI | Objective/configuration parity by construction |
| User plate diagnostic example | same as GUI/TUI | Persists bounded JSONL evidence |
| User plate plain headless | `user_runner::run_headless_user_problem` | Independent console path; shares core step machinery but has distinct dispatch/reporting |
| Built-in Kirsch GUI/TUI | `runner::run_training` | Frozen single-domain path |
| Built-in Kirsch headless | `headless::run_headless` | Independent regression path by design |
| Pin-lug GUI/TUI | `runner::run_training_pinlug` | Streaming multi-domain path |
| Pin-lug headless | `headless::run_headless_pinlug` | Decision-maker behavior differs and remains documented |

Independent headless paths are intentional regression or console paths. UI parity claims apply
to TUI versus GUI for the same problem and effective configuration.

## Correctness and lifecycle changes

- Missing or invalid active L-BFGS weights fail closed in debug and release for single- and
  multi-domain objectives. Explicit zero remains distinct from a missing key.
- CLI and `pinn.env` parsing reject unknown arguments/problems/keys, missing option values,
  duplicate keys, malformed/nonfinite values, invalid ranges, unreadable explicitly selected
  files, and invalid post-conversion geometry before training starts.
- TUI provides Overview, Loss/gradients, Physics/holes, Sampling/AMR, Run/backend, and Logs
  tabs; signed charts; bounded history; small-terminal fallback; scrolling/filtering;
  pause/resume requests; cancellation; worker-panic propagation; persisted diagnostics; and
  bounded data/control channels.
- Terminal state uses RAII cleanup after every setup stage. Unix stdout and stderr are sent to
  the run's `solver.log` during the alternate screen, then restored. Worker disconnection is a
  visible error.
- CI labels short training as runtime smoke only. Deterministic L0/L3 and affine-field fixtures
  are explicit physics gates. A scheduled/manual workflow runs expensive trained no-hole
  Variational L4 and single-hole hard-constraint L5 tests with numerical acceptance assertions.
- Burn remains on 0.21 because the confirmed concurrent-backward fix is not in a compatible
  stable release. Solver tests remain serialized; concurrent autodiff runs are unsupported.

## Cross-repository reproducibility

`app-egui` consumes the solver by relative path and sees uncommitted solver changes. Its
independent lockfile controls resolution. Companion GUI charts now preserve signed losses and
label the legacy aggregate accurately. CI still needs a reviewed solver revision contract: its
sibling checkout reproduces directory layout but does not pin a solver `ref`, so identical GUI
commits can resolve different solver revisions. This is a release/CI policy decision.

## Operational-status rule

Compilation, test success, process completion, finite loss, and a falling objective are not
physical acceptance. A run is operational for an engineering case only when applicable
analytical/reference thresholds, field checks, force/energy balance, boundary residuals,
convergence evidence, and provenance are present and pass. Missing evidence means not evaluated.

## Follow-up convergence investigation (2026-09-24)

The shared-runner diagnostic completed 3000 fresh updates for the shipped mixed-hole
`notched_plate.toml`. Its legacy Hybrid objective fell to −0.4433, but the physical probes
still reported 20.48% energy-balance error and Free/Fixed hole von-Mises Kt of 0.734/1.060.
The 1000-update shipped no-hole Hybrid run likewise ended with 19.65% energy-balance error and
9.60 MPa outer-boundary residual RMS. These are nonconverged results, not merely a negative
loss-label problem. No negative `energy_loss` or nonfinite gradient appeared in either run.

The mechanism for the shipped notched objective is explicit in `UserDefinedProblem::loss_terms()`:
its default legacy Hybrid registers `interior_energy` and `external_work` as separate terms.
Their base weights are 1 and 20, and SAW-BRDR adapts them separately. Therefore their sum is
not the stationary physical potential `U − W_ext`; minimizing it does not imply energy balance.
This default remains unchanged for compatibility. Corrected Variational uses one atomic
`physical_potential` term. A controlled 1000-update mixed-hole Variational run improved
energy-balance error to 0.260%, but its Free-hole Kt was still 1.554. The off-center Free hole
gets no explicit traction term or hard constraint under this configuration; the Fixed-hole
displacement term is the only hole-specific objective term. Falling potential and energy
balance alone therefore do not certify the local Free-hole stress field.

An initial full-plate FEM comparison matched hole geometry, Free/Fixed labels, material,
load, FD step, and radial probe offset, but still imposed three outer rigid-body pins. The
Fixed hole already removes those modes, so extra pins changed the boundary-value problem.
That initial Free/Fixed von-Mises Kt of 3.016/1.654 is **not** the matched reference.
The FEM tool now applies outer gauge pins only to all-Free cases. With just the Fixed-hole
Dirichlet support, three refined meshes (4180/6542/9483 nodes) pass its <3% mesh-change gate
on the last two refinements. Final Kt is 3.186/1.347 at 360 angles, or 3.164/1.338 at the
PINN's same 72 angles. Both use the same `fd_h=1e-3` physical steps (0.0001 m in x,
0.00005 m in y) and the same 0.0004 m offset from each hole boundary. Mesh-change stability
is evidence, not a rigorous error bound. FEM enforces the Fixed hole exactly; PINN's soft
penalty must separately show a small displacement residual before local Kt can pass.

The aggregate `probe_boundary_residuals()` is not a uniform physical traction metric on this
mixed geometry. It combines outer-boundary derived-stress residuals in Pa, Free-hole residuals
from direct network stress columns (which Variational does not train), and Fixed-hole
displacement residuals in meters. Its RMS must not be used as a standalone acceptance gate
until these components are separated and the Free-hole stress source matches the physical
derived field. The independently reconstructed Kt/FEM mismatch remains decisive without it.

A controlled Variational run with the existing Free-hole hard-constraint ansatz raised
Free-hole Kt to 2.694 by update 300, where it stayed through update 1000.
The Fixed-hole Kt moved from 1.245 to 1.810 over those updates; it did not settle. This
establishes sensitivity to boundary representation, not convergence. Its reported energy
balance (90.77% at update 300, 73.38% at update 1000) came from a probe that read the raw
network output without the training ansatz or affine field. That probe result is invalid for
this run and must be recomputed after the field-reconstruction fix. The `hole_fixed` term is
also physically under-scaled: it squares displacement in meters without dividing by the
reference-displacement square, leaving weighted contribution around 5.6e-9 versus physical
potential around 0.95 at update 1000. This is a separate, evidence-backed constraint defect.
The follow-up patch makes the Fixed-hole constraint dimensionless and includes affine
background displacement in its target. A fresh 300-update comparison produced a weighted
Fixed-hole contribution of 0.0368 (rather than ~1e-9), corrected-field energy-balance error
10.30%, and Free/Fixed Kt of 2.693/0.929. This is a meaningful constraint signal but still
fails physical acceptance. At update 1000, energy-balance error was 4.61%, Fixed-hole
normalized raw penalty 3.31e-5, and Free/Fixed Kt 2.695/1.569. Against matched 72-angle FEM,
Free Kt is 14.8% low and Fixed Kt is 17.2% high. No acceptance gate passes on that evidence.
For affine-completed hard-constraint fields, `PhysicalPotentialEnergyTerm` formerly added
affine strain to internal energy but omitted affine displacement from external work. The
missing work is constant with respect to trainable parameters, so it shifted reported
objective values rather than gradients. The term now includes that analytic work constant;
an affine no-hole regression verifies external work equals twice exact strain energy.

A controlled 300-update trial applied the existing multi-Free-hole envelope saturation rule
to this mixed plate's lone Free hole. That made the network correction visible at the Kt
probe radius, but Free-hole Kt exploded to 12.034 against matched FEM 3.164 while the energy
balance error fell to 6.79%. The now-corrected signed objective was −0.996, showing directly
that negative/falling loss and better global energy balance can coexist with a grossly wrong
local stress field. The saturation trial was reverted; canonical single-hole behavior and
the prior mixed-hole scale remain unchanged. This is a rejected hypothesis, not an accepted
numerical fix.

A second controlled 300-update trial registered the existing FD-derived Free-hole traction
term for the noncanonical mixed Variational geometry, with zero target and no affine
decomposition. It produced Free-hole Kt 0.318 and 70.4% energy-balance error; the Free-hole
term had only 1.65% of total gradient share at the final step. The trial was reverted.
Merely adding that boundary penalty does not recover the local field under the current
representation and objective balance.

## Matched-field follow-up (2026-09-24)

The user requested implementation of the evidence-gated investigation without treating a
negative loss or a completed run as convergence. A unit-separated boundary diagnostic now
reports outer derived traction in Pa, Free-hole derived traction at the FD-safe offset ring
in Pa, and Fixed-hole total displacement at the exact boundary in m. The legacy mixed-unit
aggregate remains for compatibility, but holed-plate auto-stop no longer consumes it. The TUI
physics panel shows the separated values with units; the aggregate cannot promote a holed
run to `plausibly_converged` in its final or persisted convergence evidence.

A fresh Variational + hard-Free-hole 1000-step trace with AMR off gave energy-balance error
4.73%, Free/Fixed Kt 2.694/1.412, outer traction RMS 2.64 MPa, Free-hole offset traction
RMS 9.08 MPa, and Fixed-hole displacement RMS 0.556 micrometers (0.58% of reference
displacement). A separate three-level corrected FEM check using 4216/6189/9484 nodes gave
72-angle Kt 3.086/1.334, 3.126/1.332, and 3.187/1.334. This independently reproduces
the prior corrected-reference range, though its finest mesh is not the identical 9483-node
mesh recorded above. The finest FEM Free-hole offset traction RMS is 9.51 MPa: the nonzero
offset-ring traction is expected and agrees reasonably with PINN. Zero traction applies at
the actual Free boundary, not at the 0.4 mm probe offset.

Matched 72-angle von-Mises profiles expose errors hidden by a peak-only comparison. The
AMR-off PINN Free profile has correlation 0.987 and RMS error 10.9% of FEM profile RMS;
its peak remains 15.5% low. The Fixed profile has correlation -0.082 and RMS error 59.2%.
An otherwise identical AMR-on 1000-step trial shifted Fixed peak Kt to 1.333, almost the
FEM peak, but Fixed profile correlation remained -0.130 with 59.7% RMS error. Free Kt stayed
2.694. Therefore AMR alone is rejected as a local-stress solution, and Fixed peak agreement
alone is a false pass. The FEM Fixed profile itself had correlation 0.994 between its last
two meshes, so its angular shape is not a coarse-mesh artifact.

The Fixed displacement residual is small relative to the plate-scale displacement, but not
necessarily small for stress 0.4 mm from the support. The dimensional estimate
`E * residual / offset` is about 100 MPa at the AMR-off endpoint, comparable with nominal
69 MPa stress. This is a mechanism hypothesis, not a proven error bound. A proposed
1%-of-plate-displacement Fixed gate is therefore too weak; acceptance must compare the
whole matched stress profile and set a local, stress-sensitive displacement tolerance.

The hard-ansatz hole profile also reported affine displacement relative to each hole center,
while training and the Fixed-hole penalty use the global affine origin. A focused regression
now checks off-center total displacement. Stress and Kt were unaffected because affine strain
was already added correctly.

A one-variable Fixed-hole enforcement trial first raised the base weight 50 to 500, but
telemetry showed effective weight stayed at 50 because the shared plate step caps the
displacement constraint there. That first trial did not exercise the intended change. A
second 300-step trial raised both base and cap to 500; telemetry confirmed an effective
weight near 500. Fixed displacement RMS fell from 2.710 to 1.679 micrometers, but energy
error rose from 10.7% to 17.1%, outer traction RMS from 3.63 to 6.19 MPa, and Fixed Kt fell
from 0.953 to 0.836. This rejects stronger Fixed penalty alone as a solution. Both temporary
constants were reverted. No 3000-step extension is warranted for this candidate.

The remaining Free-hole peak mismatch has a stronger structural explanation than optimizer
duration. At the actual Free boundary, the hard-ansatz envelope and its first derivative
both vanish, so the learned correction contributes no boundary strain. That freezes hoop
stress as well as normal/shear traction to the isolated-hole closed form, although only the
traction components must vanish for this finite mixed-hole problem. A focused calculation
on the exact notched geometry gives pure-ansatz Kt 2.690804 at the matched offset, versus
trained Kt 2.694340 after 1000 steps and corrected FEM about 3.16-3.19. The network envelope
there is 0.001599. This directly shows the learned field barely changes near-hole Kt.

A controlled intermediate saturation-scale trial (1 to 8, AMR off) was also rejected at
300 steps. Free Kt stayed 2.699; Free-hole offset traction RMS rose from 8.27 to 35.1 MPa
versus FEM 9.51 MPa; radial Kt change reached 40.8%. The earlier larger-scale trial had
overshot Kt to 12.034. Changing the scalar recovery rate cannot remove the exact-boundary
overconstraint, and these trials show it also damages the nearby physical field. The scale
change was reverted.

A fresh plain Variational control used the now-correct Fixed displacement normalization and
the same AMR-on configuration as the earlier plain run. At 300 steps it had 59.7% energy
error and Free/Fixed Kt 0.917/0.471. By 1000 steps energy error fell to 0.956%, but force
error was 10.7%, Free-hole offset traction RMS 58.2 MPa versus FEM 9.51 MPa, Fixed
displacement RMS 1.024 micrometers, and Free/Fixed Kt only 1.533/1.188. Thus global energy
balance can appear good while the local field and force balance still fail. This control
does not supply a replacement for the overconstrained hard ansatz.

## Decision and remaining gate

The mixed Free/Fixed notched plate is **not physically accepted**. Its strongest hard-ansatz
trial misses the matched FEM Free-hole peak by about 15%, and the Fixed-hole angular profile
by about 59% RMS. The plain Variational control misses both local and force-balance checks.
Neither a negative objective nor a near-matching Fixed peak can override these failures.
No 3000-step extension is justified for the rejected configurations.

The next representation must allow learned tangential boundary strain at a Free hole while
preserving zero normal and shear traction there. The current scalar envelope sets both its
value and first radial derivative to zero at the boundary, fixing all learned boundary strain.
The current `DirichletAnsatz` interface only multiplies raw displacement pointwise and adds a
fixed field. It cannot impose the traction relation on freely learned boundary displacement
derivatives without a richer field operator. This is a structural limitation, not evidence
for another scalar weight or saturation tune.
Validate any replacement first on a manufactured or matched finite-domain field, then use
short training comparisons. Extend to 3000 steps only if full angular profiles, boundary
conditions, force and energy balance, and refinement trend improve together. Set numerical
tolerances before judging a candidate against the mesh-refinement spread and local stress
sensitivity; do not use plate-scale displacement percentage as the Fixed-hole gate.

Loaded-checkpoint evaluation now reconstructs the ansatz and affine field from the saved
`ProblemSpec.architecture`, as live training does. Its reported term metadata uses that same
problem instance. A focused regression distinguishes the hard-constraint profile from the
old identity reconstruction and verifies the reported boundary terms. This fixes checkpoint
reporting, not the mixed-hole convergence failure.

## Independent reference repair and boundary-lift trial (2026-09-25)

The earlier Free-hole FEM Kt range near 3.16 was not a reliable acceptance target. Its
piecewise-linear CST displacement has elementwise-constant, discontinuous stress. As the
graded radial mesh refines, the first ring crosses the fixed 0.4 mm probe offset. Stress
read at that fixed radius therefore jumps between elements. The old peak-only mesh gate
missed this full-profile error. The FEM tool now exposes full angle-aligned displacement,
stress, and traction fields, plus area-weighted recovered stress; the separate
`tools/compare_plate_trace.py` checks exact geometry/load/BC/probe alignment and records
all three stress estimators. Raw element stress remains diagnostic because it is
discontinuous. Physical reference quality requires three meshes, under 2% recovered
full-profile change, and under 3% FD-versus-recovered full-profile disagreement.

On 19,408, 25,326, and 32,664-node mixed-BC meshes, the last recovered Free profile
changes 0.91% RMS and FD/recovered disagree 2.35% RMS. Fixed values are 0.61% and
0.71%. Finest FEM internal energy is 3.44218 J for the 5 mm plate. The external
acceptance gate now checks run internal energy against this value within 2%, in
addition to its own energy balance. This gate passes, but does not claim a rigorous FEM error bound or an independent
higher-order element check. Against the 32,664-node recovered profile, the prior 1000-step
hard-ansatz run has Free peak error 7.8%, Free full-profile RMS error 7.7%, and Fixed
full-profile RMS error 59.9%. Thus the earlier quoted 15% Free peak gap was inflated by
the unstable CST probe. Fixed profile failure persists.

An opt-in circular Free-boundary lift now lets the network learn boundary displacement
and hoop strain, while deriving its radial displacement derivative from the plane-stress
zero-traction equations. It is a solver field operator, not a coordinate- or material-
specific patch. Existing ansatz behavior remains the default. Current implementation is
limited to one circular Free hole, single-domain Variational training, and no trainable
saturation scale; unsupported combinations fail closed. A focused network-field probe
verifies under 3% normalized boundary traction with a changed boundary hoop field.
The lifted field still uses the existing soft Fixed-hole constraint.

A completed 300-step lifted trial (`notched_plate_boundary_lift_trial.toml`) has Free
peak error 3.1% but full Free profile RMS error 9.0%; Fixed full-profile RMS error
55.3%; energy error 5.91%; force error 7.05%; outer traction RMS 3.09 MPa; Fixed
displacement RMS 2.73 micrometers. It is not accepted. This is enough Free-peak movement
to justify a 1000-step extension, not a 3000-step run. The trial also confirms the
generic Variational telemetry correction: `energy_loss` now shows the signed atomic
`physical_potential` instead of a false zero; objective sign itself is not a gate.

The external comparison verdict is fail-closed on reference quality, a matched passing
no-hole L0/L4 companion, full profiles, energy, force, separated boundary units, and
local-stress-scaled Fixed displacement. Historical traces without coordinates can be
compared but cannot pass. No mixed-hole operational claim is made.

The completed 1000-step lifted trial improved Free von-Mises peak/full-profile error to
1.4%/3.6%. Energy error is 0.56% and force error 1.35%. Yet Free `syy`/`sxy` profile
errors are 8.5%/5.3%, outer traction RMS is 2.29 MPa, and Fixed full-profile RMS error
is 49.9%. Fixed displacement RMS is 0.649 micrometers: scaled by `E / 0.4 mm`, this
could produce stress larger than the applied load. Thus longer training alone is not
justified. A change in Kt between two physical radii is not itself numerical
nonconvergence; the external gate instead checks whole-profile change over a 100-step
training window at the same matched radius.

An optional exact Fixed-hole mask has been added as a separate intervention. It acts
on total displacement, including the affine background, and reaches identity before
any Free or outer boundary. Its first derivative at the Fixed boundary remains nonzero,
so it enforces only displacement, not zero strain. Geometric clearances must resolve
the FD stencil or construction fails. Tests cover one Free plus Fixed and two Fixed
holes without a Free hole. The next controlled run changes only this enforcement mode.

The first exact-Fixed 300-step trial used a transition width of half the nearest
boundary clearance (21 mm here). It drove Fixed displacement RMS to 1.4e-12 m, but
Fixed peak Kt collapsed from 1.40 initially to 0.403 against FEM about 1.33;
full-profile RMS error became 77.6%. The mask value at the 0.4 mm stress probe was
only 0.0194. PINN displacement RMS on that ring was 0.103 micrometers, versus FEM
0.551 micrometers. Exact displacement alone thus did not produce correct support
stress; this broad mask repeated the near-boundary suppression mechanism. A
controlled follow-up caps transition width at five FD-safe probe margins (2 mm here),
while retaining the clearance bound and stencil-resolution check. The resulting
mask value at the first probe is above 0.2; a focused regression protects that
signal. No 1000-step extension is justified for the broad-mask trial.

The five-probe-margin (2 mm) 300-step trial reached Fixed Kt 3.739 versus FEM about
1.33, a 182% peak error and 133.6% full-profile RMS error. Its near-support field
was amplified instead. This brackets a strong, unwanted dependence on mask width,
not a solved boundary condition. A final geometry-scaled check uses the Fixed hole's
own radius (8 mm here), capped by half the nearest boundary clearance. Radius is a
physical length scale independent of FD resolution; this is not fitted to FEM Kt.
If it fails short-run field gates, stop mask-width trials and investigate a boundary
field representation whose local strain is independently trainable.

The radius-scaled 300-step trial completed and failed that decision gate. Against the
same 32,664-node recovered FEM reference, its Fixed peak/full-profile errors were
30.8%/48.7%; the Fixed full profile still changed 14.3% over the last 100 steps.
Free full-profile error was 6.3%, and internal energy differed 2.51% from FEM.
The 0.915% run energy-balance error and 1.94% force-balance error therefore do not
establish physical convergence. No further width tuning or 1000-step extension is
justified for the multiplicative Fixed mask.

Its failure mechanism is algebraic. For a masked field `u=q(d)f(d)` with `q(0)=0`,
the support-normal derivative is `q'(0)f(0)`. The chosen transition width fixes
`q'(0)`, tying support strain to the unmasked boundary value. Width trials moved
the Fixed peak from 0.403 through 0.917 to 3.739 without a stable full profile.
The opt-in Fixed operator now subtracts the pre-Fixed field's angular boundary
trace with a compact weight whose value is one and whose radial slope is zero at
the support. Thus total displacement is zero there, while its normal derivative
remains the pre-Fixed field's learnable derivative. The subtraction includes the
affine background and Free-hole lift, and vanishes before any other boundary.
The 300-step matched-field trial of this operator completed and failed. The
Fixed full-profile error fell to 36.0% at
step 150, then rose to 41.9% at step 299; its peak error was 19.7% and its
last-100-step profile change was 10.6%. The Free full-profile error was 6.01%,
with `syy` error 16.3%; force-balance error was 5.33%, outer traction RMS was
3.27% of applied stress, and internal energy was 1.25% from FEM. Fixed
displacement RMS was 1.37 nm. Thus freeing radial strain improved the Fixed
field over the multiplicative mask but did not solve its angular shape or
convergence. The Fixed error already worsened before AMR adaptation at step
200, so the AMR switch alone cannot explain it. At this checkpoint the
original whole-field decision rule did not justify a 1000-step extension.
The Fixed ring's recovered FEM von-Mises mean is 1.002 times applied stress;
the projected PINN ring's mean is only 0.621. FEM's fourth angular Fourier
amplitude is 0.114, versus PINN 0.019. FEM is mirror-symmetric to 0.55% RMS;
the PINN profile has 15.3% mirror asymmetry. These are broad field-shape
defects, not a single misplaced peak. The opt-in projection removes a
boundary-strain restriction but does not supply enough local field accuracy
under this training configuration.
An FD-crossing hypothesis was falsified against the production AMR wiring:
the runner builds its adaptive grid on a collocation-only geometry with hole
radii inflated by the FD-safe margin. A focused two-hole regression confirms
the resulting 2048 sampled interior points all clear that margin. Constructing
the grid on the uninflated physical geometry would be unsafe, but is not what
the runner does.

A first 1000-step no-hole companion using the mixed trial's 64-by-3 network and
2048/512 collocation budget failed L4 stress, traction, and load-transfer gates;
its final energy-balance error was 8.34%. This is not a passing prerequisite.
The companion specification now uses the previously verified no-hole L4 training
budget (64-by-8, 4096/2048, AMR off, 3000 steps) while keeping the same plate,
material, load, and formulation. It tests solver health, not matched optimizer
capacity; the mixed-hole FEM comparison remains exactly geometry/load/BC/probe
matched.
That 3000-step companion completed and passed L0/L4. The first L4 pass was at
step 2460. At step 2999 its `sigma_xx` error was 0.110%, traction RMS was
0.182% of applied stress, and load transfer was 0.99817; energy-balance error
was 0.0177%. The 1000-step, smaller-network companion had failed, and even
this stronger no-hole run was far outside L4 limits at step 1000. Thus a
300-step nonstationary mixed-hole trace cannot rule out later convergence on
duration alone. A 1000-step projection extension is now being run to test
whether its whole Fixed profile recovers; acceptance still requires all
physical and reference-quality gates.

The FEM reference-quality gate itself needed one correction. It had checked
refinement and FD-versus-recovered agreement only for von Mises, while later
judging individual stress components at a 5% PINN error threshold. On the
finest current mesh, Free-hole `syy` FD/recovered profiles differ 5.3% of the
recovered `syy` RMS; Fixed-hole `syy` changes 2.93% from the preceding mesh.
Those components cannot support a 5% acceptance claim yet. The comparator
now requires the same 2% mesh-change and 3% estimator-agreement gates for
`sxx`, `syy`, and `sxy`, so this reference fails closed until refined or
independently checked. Von-Mises-only comparisons remain useful diagnostics.
Refinement to 57,226 and 71,698 nodes kept internal energy stable near 3.4423 J
and the Fixed von-Mises profile far from PINN, but did not clear component
reference gates: finest Free `syy` changed 2.30% from the preceding mesh and
FD/recovered disagreed 3.49%; Fixed `syy` disagreed 3.53%. More CST refinement
alone is not a justified acceptance shortcut; an independent higher-order or
better stress-recovery reference is the next reference-quality check.

## Projection extension and controlled falsifications (2026-09-25/26)

The 1000-step boundary-lift/Fixed-projection extension completed with finite
telemetry. The previously completed 3000-step no-hole companion passed L0/L4,
so that prerequisite is present. Against the same 71,698-node mixed-BC FEM
reference, the Free-hole recovered von-Mises full-profile error was 3.38% and
peak error 0.62%. Its `sxx`/`syy`/`sxy` full-profile errors were
2.47%/8.86%/5.25%. The Fixed-hole recovered von-Mises full-profile error was
48.31%, peak error 17.96%, and component errors 46.52%/62.26%/62.28%.
The last-100-step Fixed profile changed 7.98%. Force-balance error was 2.34%,
outer-traction RMS was 3.13% of applied stress, run energy-balance error was
0.31%, and internal energy differed 0.89% from FEM. Thus global energy and
Free-hole von-Mises improvement do not imply Fixed-hole convergence. This
candidate does not justify 3000 steps. The FEM component reference-quality
gate still fails on `syy`; these component errors are diagnostic, not a
five-percent certification.

A controlled affine-gauge experiment centered the affine background on the
Fixed support before local trace projection. An affine-only calculation reduced
its Fixed von-Mises profile error from 69% to 29% against the FEM ring, but a
matched 300-step training run moved the Fixed error only from 41.96% to 41.71%.
Its last-100-step Fixed change was 7.89% versus 10.61% control. Force error
fell from 5.33% to 0.43%; outer traction still missed its gate. The gauge
change was reverted: it did not solve local stress accuracy.

The training AMR grid excludes a 0.4 mm annulus around every hole to keep
central-FD stencils outside the void; the matched stress probe is at that same
offset. On the 32,664-node FEM solution, the excluded bands contain about
0.38% of total strain energy around the Free hole and 0.12% around the Fixed
hole. This is a plausible local-resolution weakness despite its small global
energy share, not a proven sole cause. A controlled Fixed-projection trial
reduced only the AMR collocation margin to 1.25 FD steps (0.125 mm), retaining
the original 0.4 mm stress probe and all other settings. At 300 steps its
Fixed full-profile error was 41.03% versus 41.96% control; Fixed last-window
change was 8.01%, force error 2.31%, and outer traction 2.88% of load. The
margin change was reverted because all relevant local and boundary gates still
failed. No short-run evidence supports margin tuning as the solution.

The 1000-step sampled objective is −1.064169, nearly the FEM minimum
−1.064229 after the same 3.23455 J normalization. That apparent agreement is
not convergence evidence. The post-update independently probed field has
`Pi = internal_energy - 2 * external_work = −3.43311 J`, while the sampled
pre-update training objective represents −3.44211 J at that step: a 0.00900 J gap. Its
last ten probe checkpoints averaged a −0.0120 J objective-minus-probe gap;
individual gaps reached several hundredths of a joule. The passing no-hole
companion's last-ten mean gap was only about 0.000011 J. The objective is
evaluated before the optimizer update on weighted AMR points; the probe is
evaluated after the update with a reinitialized plain sampler. Thus the gap
mixes model change, quadrature variance, and possible quadrature bias. It is
evidence that matching a
single sampled objective value to FEM's minimum cannot certify the mixed-hole
field. The exact contribution of each source remains open.
