# PH4-01 mode B — Hybrid + MeasureAware

Status: NOT INDEPENDENTLY RUN — for THIS configuration (square plate, `half_w==half_h`, uniform
sampling, no AMR), this is a mathematically forced identity with Mode A, not a missing-evidence
gap (issue #63 rule #16: unsupported/uncovered behavior must be marked explicitly, never
silently approximated — and equally, a covered-by-construction case should not be padded with a
redundant run either).

`InteriorEnergyTerm`'s `ref_energy_absolute = ref_energy * domain_area * thickness` is
constructed so `domain_integral_tensor`'s `mean(f)*area*thickness / ref_energy_absolute`
reduces to exactly `mean(f) / ref_energy` — the legacy `.mean()` path's own formula, with the
`area*thickness` factor cancelling exactly. `ExternalWorkTerm`'s measure-aware path is likewise
byte-identical to its legacy path for a square plate (every edge's `ds` is identical) per that
term's own doc comment. Both unweighted-equivalence claims are independently covered by the
existing, passing `interior_energy_term_measure_aware_with_no_weights_matches_domain_integral_tensor_directly`
and `external_work_term_measure_aware_matches_boundary_integral_tensor_directly` tests
(`user_problem.rs`). So Mode A's real, already-verified result (`Debug_run/baseline_legacy_no_hole/`,
PH3-09's resume) IS Mode B's real result for this cell — an independent training run here would
reproduce the same numbers, not add evidence.

This equivalence does **not** extend to a non-square plate (different `ds` per edge — see
issue #63 sub-issue #68) or an AMR-nonuniform sampling regime (see sub-issue #67). Those are the
configurations where Mode B genuinely needs its own independent run, and are correctly scoped
there, not here.
