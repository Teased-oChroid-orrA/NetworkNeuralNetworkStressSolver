# PH4-01 mode C — Variational + MeasureAware (pre-issue-#64, broken)

Status: VERIFIED HISTORICAL FAILURE (issue #65, PH4-01 mode classification matrix)

Historical Variational behavior used independently weighted U/W terms (fixed by PH4-03's atomic
`PhysicalPotentialEnergyTerm`) and, even after that fix, failed all five hard L4 metrics at
every controlled-ladder stage recorded under `Debug_run/phase4/{D_variational_atomic_pi,
E_variational_stabilized}/` — root-caused in issue #64 to `UserSamplingStrategy` returning a
static point cloud on every training step for no-hole geometries (see
`docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-06 entry for the full mechanism). Not valid
corrected-objective evidence — kept here as the documented "this is what was broken and why"
record, distinct from Mode D's real, independently-earned PASS.
