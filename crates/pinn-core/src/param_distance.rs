//! `enhancement.md` Phase 21 ("Do Not Rely Only on Min/Max") — a real distance-to-training-
//! distribution metric for the parametric PINN path, replacing pure min/max rectangle
//! containment (`ParametricProblemSpec::in_range`) as the ONLY signal. A rectangular envelope
//! doesn't prove the model actually saw every combination inside it (the doc's own explicit
//! example: dense coverage in one corner, sparse elsewhere, still passes a min/max check).
//!
//! Pure logic, no `burn`/tensor dependency — fits alongside `sampling.rs`'s existing
//! plain-math role. Operates on already-normalized `[-1, 1]^3` triples (the same space
//! `ParamRange::normalize` produces for `(E, nu, Px)`), so callers own converting real
//! physical values into that space first.
//!
//! The threshold this feeds into is ratio-based (query distance vs. the training samples'
//! OWN median nearest-neighbor spacing), not an arbitrary absolute constant — the same
//! "compare against the run's own observed baseline" convention already established for the
//! GREEN/YELLOW/RED physics-residual check (`app-egui`'s `classify_infer_result`), per this
//! project's own explicit rule against unearned precision in thresholds.

/// Euclidean distance between two normalized `(e_n, nu_n, p_n)` triples.
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance from `query` to the closest point in `samples`. `f32::INFINITY` for an empty
/// `samples` (no coverage information at all — callers should treat this as "unknown", not
/// "close").
pub fn nearest_neighbor_distance(query: [f32; 3], samples: &[[f32; 3]]) -> f32 {
    samples.iter().map(|&s| dist3(query, s)).fold(f32::INFINITY, f32::min)
}

/// Median pairwise nearest-neighbor spacing WITHIN `samples` itself — how densely training
/// actually sampled this region, used as the self-baseline `nearest_neighbor_distance` is
/// compared against. `0.0` for fewer than 2 samples (no spacing to measure). `O(n^2)`,
/// acceptable at the bounded reservoir sizes (~hundreds) this is called with, and only ever
/// called on-demand at inference-query time, never in a per-step hot loop.
pub fn median_nn_spacing(samples: &[[f32; 3]]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut nn_dists: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            samples
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &o)| dist3(s, o))
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nn_dists[nn_dists.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_neighbor_distance_finds_the_closest_point_not_just_the_first() {
        let samples = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let d = nearest_neighbor_distance([0.1, 0.0, 0.0], &samples);
        assert!((d - 0.1).abs() < 1e-6, "expected distance to [0,0,0] (0.1), got {d}");
    }

    #[test]
    fn nearest_neighbor_distance_is_zero_for_an_exact_match() {
        let samples = [[0.3, -0.2, 0.7]];
        let d = nearest_neighbor_distance([0.3, -0.2, 0.7], &samples);
        assert!(d < 1e-6, "expected ~0.0 for an exact match, got {d}");
    }

    #[test]
    fn nearest_neighbor_distance_is_infinite_for_empty_samples() {
        let d = nearest_neighbor_distance([0.0, 0.0, 0.0], &[]);
        assert_eq!(d, f32::INFINITY);
    }

    #[test]
    fn median_nn_spacing_matches_hand_computed_value_on_a_regular_grid() {
        // A 1D line of evenly-spaced points along the first axis: every point's nearest
        // neighbor is exactly `step` away except the two endpoints (still `step` away, just
        // to their one neighbor), so the median must equal `step` exactly.
        let step = 0.25f32;
        let samples: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * step, 0.0, 0.0]).collect();
        let m = median_nn_spacing(&samples);
        assert!((m - step).abs() < 1e-5, "expected median spacing {step}, got {m}");
    }

    #[test]
    fn median_nn_spacing_is_zero_for_fewer_than_two_samples() {
        assert_eq!(median_nn_spacing(&[]), 0.0);
        assert_eq!(median_nn_spacing(&[[0.0, 0.0, 0.0]]), 0.0);
    }

    #[test]
    fn median_nn_spacing_reflects_a_sparse_region_not_just_a_dense_one() {
        // Nine points clustered tightly, one point far away - the median (5th of 10 sorted
        // nearest-neighbor distances) must still be dominated by the dense cluster, not
        // averaged toward the outlier - this is the whole point of using median, not mean.
        let mut samples: Vec<[f32; 3]> = (0..9).map(|i| [i as f32 * 0.01, 0.0, 0.0]).collect();
        samples.push([10.0, 0.0, 0.0]);
        let m = median_nn_spacing(&samples);
        assert!(m < 0.1, "median should reflect the dense cluster's tight spacing, got {m}");
    }
}
