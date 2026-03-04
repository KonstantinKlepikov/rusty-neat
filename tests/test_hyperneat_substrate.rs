use rusty_neat::Genome;
use rusty_neat::Substrate;
use rusty_neat::genes::{ActivationFunction, LinkGene, NeuronGene, NeuronType};

// Test that Substrate reports the correct maximum dimensionality and that
// the computed minimal CPPN input/output sizes follow the formulae used in
// the C++ implementation (2*max_dims + optional distance + bias, and
// outputs depend on query_weights_only and leaky flags).
#[test]
fn substrate_dims_and_cppn_io_sizes() {
    // 2D input and 2D output -> max_dims = 2
    let input = vec![vec![-1.0, -1.0], vec![0.0, 0.0]];
    let hidden: Vec<Vec<f64>> = Vec::new();
    let output = vec![vec![1.0, 1.0]];

    let s = Substrate::with_coords(input.clone(), hidden.clone(), output.clone());

    assert_eq!(s.get_max_dims(), 2);

    // with_distance = false by default in with_coords
    assert_eq!(s.get_min_cppn_inputs(), 2 * 2 + 1); // 2*max_dims + bias

    // outputs: query_weights_only=false and leaky=false => 2 outputs
    assert_eq!(s.get_min_cppn_outputs(), 2);
}

// Test behavior when `with_distance` and `leaky` are enabled: ensure the
// min CPPN input/output counts reflect the additional distance input and
// the two extra outputs required for leaky substrates.
#[test]
fn substrate_with_distance_and_leaky() {
    let input = vec![vec![0.0_f64]]; // 1D coords
    let hidden = vec![vec![0.0_f64]];
    let output = vec![vec![0.0_f64]];

    let mut s = Substrate::with_coords(input, hidden, output);
    s.with_distance = true; // enable distance input
    s.leaky = true; // enable leaky substrate

    // max_dims = 1
    assert_eq!(s.get_max_dims(), 1);

    // min cppn inputs = 2*1 + 1(distance) + 1(bias) = 4
    assert_eq!(s.get_min_cppn_inputs(), 4);

    // min cppn outputs: query_weights_only=false => 2, leaky=true => +2
    assert_eq!(s.get_min_cppn_outputs(), 4);
}

// Smoke test for `set_neurons` and `print_info` to ensure coordinates are
// stored and accessible; `print_info` is exercised to ensure it doesn't panic
// (no assertions on printed output here).
#[test]
fn set_neurons_and_print_info_smoke() {
    let input = vec![vec![0.0_f64]; 3];
    let hidden: Vec<Vec<f64>> = Vec::new();
    let output = vec![vec![0.0_f64]; 1];

    let mut s = Substrate::new();
    s.set_neurons(input.clone(), hidden.clone(), output.clone());

    // Basic checks
    assert_eq!(s.input_coords.len(), 3);
    assert_eq!(s.output_coords.len(), 1);

    // print_info is just a debug helper; call to ensure no panic
    s.print_info();
}

// Test that `query_weights_only = true` changes the minimal CPPN outputs
// count (one output for weights) and interacts with `leaky` flag.
#[test]
fn query_weights_only_min_outputs() {
    let mut s = Substrate::new();
    s.query_weights_only = true;
    // without leaky -> 1 output
    assert_eq!(s.get_min_cppn_outputs(), 1);

    s.leaky = true;
    // leaky adds +2 outputs
    assert_eq!(s.get_min_cppn_outputs(), 3);
}

// Test that `with_coords` constructor sets `custom_conn_obeys_flags = false`.
#[test]
fn with_coords_sets_custom_conn_obeys_flags_false() {
    let s = Substrate::with_coords(vec![vec![0.0_f64]], vec![], vec![vec![0.0_f64]]);
    assert_eq!(s.custom_conn_obeys_flags, false);
}

// Build a minimal CPPN-like genome that always turns links on by wiring the
// bias input to each CPPN output with a positive linear weight. Use this
// genome to exercise link-generation filtering in `BuildHyperNEATPhenotype`.
fn make_constant_cppn(cppn_inputs: usize, cppn_outputs: usize) -> Genome {
    let mut g = Genome::default();
    g.set_num_inputs(cppn_inputs);
    g.set_num_outputs(cppn_outputs);

    // Assign neuron ids: inputs 1..cppn_inputs, outputs next
    let mut next_id: u64 = 1;
    for i in 0..cppn_inputs {
        let ntype = if i + 1 == cppn_inputs {
            NeuronType::Bias
        } else {
            NeuronType::Input
        };
        let ng = NeuronGene::new(
            next_id,
            ntype,
            0,
            0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            ActivationFunction::UnsignedSigmoid,
        );
        g.push_neuron(ng);
        next_id += 1;
    }
    // outputs
    for _ in 0..cppn_outputs {
        let ng = NeuronGene::new(
            next_id,
            NeuronType::Output,
            0,
            0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            ActivationFunction::Linear,
        );
        g.push_neuron(ng);
        next_id += 1;
    }

    // Connect bias (last input id = cppn_inputs) to every output with weight 1.0
    let bias_id = cppn_inputs as u64; // ids started at 1
    let mut innov = 1u64;
    for out_idx in 0..cppn_outputs {
        let out_id = (cppn_inputs + out_idx) as u64 + 1; // since next id started after inputs
        let lg = LinkGene::new(bias_id, out_id, innov, 1.0, false);
        g.push_link(lg);
        innov += 1;
    }

    g
}

// Test filtering of candidate pairs by substrate flags (allow_input_output_links)
// and custom_connectivity obeying flags.
#[test]
fn filtering_allow_flags_and_custom_connectivity() {
    // Simple substrate: 1 input, 1 output
    let mut subst = Substrate::with_coords(vec![vec![-1.0, 0.0]], vec![], vec![vec![1.0, 0.0]]);

    // Build a CPPN genome with appropriate IO sizes
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    let g = make_constant_cppn(cppn_inputs, cppn_outputs);

    // Case 1: allow input->output links
    subst.allow_input_output_links = true;
    let mut net1 = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net1, &subst);
    // When allowed, there should be exactly one connection from input->output
    assert_eq!(net1.connections.len(), 1);

    // Case 2: disallow input->output links
    subst.allow_input_output_links = false;
    let mut net2 = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net2, &subst);
    // No connections should be generated
    assert_eq!(net2.connections.len(), 0);

    // Now test custom_connectivity obeying flags. Add a custom mapping for the
    // same input->output pair and set custom_conn_obeys_flags = true. With
    // allow_input_output_links = false the custom entry should be skipped.
    subst.custom_connectivity.clear();
    subst.custom_connectivity.push(vec![0, 0, 3, 0]); // INPUT(0,idx0) -> OUTPUT(3,idx0)
    subst.custom_conn_obeys_flags = true;
    let mut net3 = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net3, &subst);
    assert_eq!(net3.connections.len(), 0);

    // If custom_conn_obeys_flags = false the custom mapping should be used
    subst.custom_conn_obeys_flags = false;
    let mut net4 = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net4, &subst);
    assert_eq!(net4.connections.len(), 1);
}

// Test substrate with 3D coordinates affecting max_dims
#[test]
fn substrate_three_dimensional_coords() {
    let input = vec![vec![0.0_f64, 0.0, 0.0]]; // 3D
    let hidden = vec![vec![0.0_f64, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
    let output = vec![vec![0.0_f64, 0.0, 0.0]];

    let s = Substrate::with_coords(input, hidden, output);
    assert_eq!(s.get_max_dims(), 3);
    // min inputs = 2*3 + bias = 7
    assert_eq!(s.get_min_cppn_inputs(), 7);
}
