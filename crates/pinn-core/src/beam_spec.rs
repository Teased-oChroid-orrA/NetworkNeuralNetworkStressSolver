//! Spec format for the 1D Euler-Bernoulli beam sanity-check problem
//! (`pinn_solver::toy_beam`), loadable the same way `problem_spec::ProblemSpec` is — a
//! deliberately SEPARATE format, not folded into `ProblemSpec`: a 1D beam and a 2D N-hole
//! plate are structurally different problems (different network input width, different
//! physics, no field/heatmap concept for the beam). A loader tries each schema in turn
//! (`BeamSpec` requires `bc`, `ProblemSpec` requires `geometry`/`material`/`load` — neither
//! has anything the other would accidentally satisfy, so this is unambiguous without a
//! separate discriminator tag).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamBcSpec {
    /// Clamped-free.
    Cantilever,
    /// Pinned-pinned.
    SimplySupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BeamNetworkSpec {
    pub hidden_dim: usize,
    pub n_hidden: usize,
}

impl Default for BeamNetworkSpec {
    fn default() -> Self {
        Self { hidden_dim: 48, n_hidden: 3 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BeamTrainingSpec {
    pub steps: usize,
    pub n_points: usize,
}

impl Default for BeamTrainingSpec {
    fn default() -> Self {
        Self { steps: 3000, n_points: 32 }
    }
}

/// The complete user-facing beam spec. `bc` is the only required field — everything else
/// defaults to the same values `toy_beam`'s own regression tests use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BeamSpec {
    pub bc: BeamBcSpec,
    #[serde(default)]
    pub network: BeamNetworkSpec,
    #[serde(default)]
    pub training: BeamTrainingSpec,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_spec_round_trips_through_toml() {
        let spec = BeamSpec { bc: BeamBcSpec::Cantilever, network: BeamNetworkSpec::default(), training: BeamTrainingSpec::default() };
        let toml_str = toml::to_string(&spec).expect("serialize");
        let parsed: BeamSpec = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed, spec);
    }

    #[test]
    fn network_and_training_default_when_omitted() {
        let parsed: BeamSpec = toml::from_str("bc = \"SimplySupported\"").expect("deserialize");
        assert_eq!(parsed.bc, BeamBcSpec::SimplySupported);
        assert_eq!(parsed.network, BeamNetworkSpec::default());
        assert_eq!(parsed.training, BeamTrainingSpec::default());
    }

    #[test]
    fn a_plate_shaped_toml_fails_to_parse_as_a_beam_spec() {
        let plate_like = "[geometry]\nhalf_w = 0.1\n";
        assert!(toml::from_str::<BeamSpec>(plate_like).is_err());
    }

    /// Regression guard: every shipped `examples/problems/beam_*.toml` spec must stay
    /// loadable by this exact struct — mirrors `problem_spec::tests::
    /// shipped_plate_example_specs_parse`'s same rationale.
    #[test]
    fn shipped_beam_example_specs_parse() {
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/problems");
        for name in ["beam_cantilever.toml", "beam_simply_supported.toml"] {
            let path = examples_dir.join(name);
            let contents = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
            toml::from_str::<BeamSpec>(&contents).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        }
    }
}
