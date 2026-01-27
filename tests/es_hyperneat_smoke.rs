use rusty_neat::Substrate;
use rusty_neat::NeuralNetwork;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod common;
use common::make_fixed_cppn;

/// Smoke test for HyperNEAT phenotype build.
/// Marked `#[ignore]` because it exercises phenotype construction and
/// can be relatively heavier than unit tests.
#[test]
fn es_hyperneat_smoke_test() {
    // deterministic RNG for reproducibility (may be useful later)
    let _rng = StdRng::seed_from_u64(20260127);

    // build a simple substrate with one input and one output coordinate
    let input_coords = vec![vec![0.0_f64, 0.0_f64]];
    let hidden_coords: Vec<Vec<f64>> = Vec::new();
    let output_coords = vec![vec![1.0_f64, 0.0_f64]];
    let mut subst = Substrate::with_coords(input_coords, hidden_coords, output_coords);

    // configure some substrate options to exercise different code paths
    subst.with_distance = false;
    subst.query_weights_only = false;

    // Determine required CPPN IO sizes and create a trivial CPPN genome
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();

    // create a CPPN that returns fixed weights for connections
    let out_weights: Vec<f64> = vec![0.5_f64; cppn_outputs];
    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &out_weights);

    // phenotype network to fill
    let mut net = NeuralNetwork::new();

    // build phenotype using the genome as CPPN
    // This should not panic and should produce zero-or-more connections
    g.build_hyperneat_phenotype(&mut net, &subst);

    // Basic sanity checks: network input/output dimensions should match substrate
    assert_eq!(net.num_inputs(), subst.input_coords.len(), "num_inputs mismatch");
    assert_eq!(net.num_outputs(), subst.output_coords.len(), "num_outputs mismatch");
}
