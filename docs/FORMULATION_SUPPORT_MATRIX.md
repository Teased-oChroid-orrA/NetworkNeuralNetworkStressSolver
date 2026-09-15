# Formulation Support Matrix

This matrix uses only Issue #63 statuses. `VERIFIED` means independent physical evidence exists;
source availability or a unit test alone is not enough.

| Capability | Strong | Weak | Variational | Hybrid |
| --- | --- | --- | --- | --- |
| FD live derivatives | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION | VERIFIED | SUPPORTED_WITH_LIMITATION |
| AD diagnostic derivatives | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION |
| Natural BCs | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION |
| Measure-aware integration | EXPLICITLY_UNSUPPORTED | EXPLICITLY_UNSUPPORTED | VERIFIED | SUPPORTED_WITH_LIMITATION |
| Translation gauge | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED | VERIFIED | SUPPORTED_WITH_LIMITATION |
| Rotation gauge | EXPLICITLY_UNSUPPORTED | EXPLICITLY_UNSUPPORTED | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED |
| AMR | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION | VERIFIED_DISABLED (issue #67/#74 - crashes past ~1200-2200 steps, kept off) | SUPPORTED_WITH_LIMITATION |
| L4 no-hole (square) | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED | VERIFIED (issue #64/#66) | VERIFIED only for legacy Hybrid artifact |
| L4 no-hole (non-square) | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED | VERIFIED (issue #68) | SUPPORTED_WITH_LIMITATION |
| L5 hole/Kt | EXPLICITLY_UNSUPPORTED | EXPLICITLY_UNSUPPORTED | SUPPORTED_WITH_LIMITATION (issue #70 - mechanism works; real attempts without AND with AMR both give kt~1.01 vs theoretical 3.0, ~66% error either way - #74 fixed but did not close the accuracy gap) | EXPLICITLY_UNSUPPORTED |
| Multi-hole topology (machinery only) | EXPLICITLY_UNSUPPORTED | EXPLICITLY_UNSUPPORTED | SUPPORTED_WITH_LIMITATION (issue #69 - runs cleanly, Kt accuracy not claimed) | EXPLICITLY_UNSUPPORTED |
| FieldKind enforcement | SUPPORTED_WITH_LIMITATION | SUPPORTED_WITH_LIMITATION | VERIFIED (issue #69 - real hole runtime evidence) | SUPPORTED_WITH_LIMITATION |
| QoI stress source | SUPPORTED_WITH_LIMITATION | EXPLICITLY_UNSUPPORTED | VERIFIED (issue #69) | SUPPORTED_WITH_LIMITATION |

`Weak` has no independently implemented production formulation in this repository. The legacy
Hybrid L4 artifact does not establish generalized Hybrid support; its captured L4 result also
fails current hard thresholds. `Strong`/`Hybrid` `FormulationSelection` variants are wired for
user-defined problems (term-selection match in `user_problem.rs`) but have never been exercised
in a real training run this epic — only unit-tested. Every `VERIFIED` cell above traces to a
specific real run or a fast fixture cited in `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`
(PH4-06/08/09/10/11/12/13/15/16/17); this matrix is not updated speculatively.
