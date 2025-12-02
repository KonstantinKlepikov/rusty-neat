// ES-HyperNEAT smoke tests (placeholders)
//
// These tests validate ES-specific parameters used by the ES-HyperNEAT
// phenotype builder: `DivisionThreshold`, `IterationLevel`, `MaxDepth`.
//
// The Rust port currently does not implement ES-HyperNEAT. These tests are
// intentionally marked `#[ignore]` and contain guidance and example code
// to run once `BuildESHyperNEATPhenotype` is implemented in the Rust API.
// Keep the tests in-tree so they can be enabled later (remove `#[ignore]`).

// Example usage (once implemented):
//
// let mut params = Parameters::default();
// params.DivisionThreshold = 0.5; // how deep the quadtree splits
// params.MaxDepth = 3;            // maximum subdivision depth
// params.IterationLevel = 1;      // number of hidden iterations/layers
//
// let g = common::make_fixed_cppn(cppn_inputs, cppn_outputs, &out_weights);
// let mut net = NeuralNetwork::new();
// g.build_es_hyperneat_phenotype(&mut net, &substrate, &params);

// The real smoke tests should assert that:
// - with a high DivisionThreshold the subdivision halts early and fewer
//   phenotype nodes are produced;
// - increasing MaxDepth increases the potential number of nodes (bounded by
//   variance checks) and that IterationLevel controls additional hidden
//   subdivisions/layers;
// - the resulting phenotype contains expected connections and neuron
//   parameters mapped from the CPPN outputs.

mod common;
// The following imports are intentionally kept as examples for the eventual
// ES-HyperNEAT implementation. They are unused in the placeholder tests and
// therefore suppressed to avoid warnings until the tests are enabled.
#[allow(unused_imports)]
use crate::common::make_fixed_cppn;
#[allow(unused_imports)]
use rusty_neat::{NeuralNetwork, Substrate};

// Placeholder test for DivisionThreshold behaviour. Ignored until ES is
// implemented. Left as a minimal compile-time checked test that is skipped
// by default.
#[test]
#[ignore]
fn es_division_threshold_smoke() {
    // Once implemented, build a CPPN and a substrate with leaky=false and
    // tune params.DivisionThreshold to observe subdivision behaviour.
    // See commented example above.
    assert!(true);
}

// Placeholder test for IterationLevel behaviour. Ignored by default.
#[test]
#[ignore]
fn es_iteration_level_smoke() {
    // Once implemented, run BuildESHyperNEATPhenotype with different
    // IterationLevel values and assert expected layer counts / connections.
    assert!(true);
}

// Placeholder test for MaxDepth behaviour. Ignored by default.
#[test]
#[ignore]
fn es_max_depth_smoke() {
    // Once implemented, vary MaxDepth and assert limits on generated nodes.
    assert!(true);
}
