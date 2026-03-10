// Tests for BuildHyperNEATPhenotype and related HyperNEAT helpers.
// Each test constructs a small deterministic CPPN genome (via helpers)
// and a `Substrate`, then builds the hyperneat phenotype and asserts
// expected connectivity and parameter mapping (weights, bias, timeconstants).
use rusty_neat::Substrate;
mod common;
use common::{assert_weight_approx, make_fixed_cppn};

// Test: scaling of CPPN outputs to phenotype weights and handling of
// `query_weights_only` vs full-output CPPNs (link_on + weight).
#[test]
fn build_hyperneat_phenotype_scaling_and_query_outputs() {
    // Simple substrate: one input coord, one output coord
    let mut subst = Substrate::with_coords(vec![vec![-1.0_f64]], vec![], vec![vec![1.0_f64]]);
    // Explicitly set max weight for clarity
    subst.max_weight_and_bias = 5.0_f64;
    // allow input->output connectivity for this explicit substrate (with_coords sets it false by default)
    subst.allow_input_output_links = true;

    // Case A: query_weights_only = false -> CPPN must have 2 outputs: link_on, weight
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    // two outputs: link_on, weight
    assert_eq!(cppn_outputs, 2);

    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &[1.0_f64, 0.25_f64]);
    let mut net = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net, &subst);

    // expected one input->output connection
    assert_eq!(net.connections.len(), 1);
    for c in &net.connections {
        let expected = 0.25_f64 * subst.max_weight_and_bias;
        assert_weight_approx(c.weight, expected, 1e-6);
    }
}

// Test: when `leaky=true` the CPPN requires additional outputs (timeconst, bias)
// and these are mapped into neuron parameters on the produced phenotype.
#[test]
fn build_hyperneat_leaky_requires_additional_outputs_and_sets_neuron_params() {
    // Simple substrate: 1 input, 1 output
    let mut subst = Substrate::with_coords(vec![vec![-1.0_f64]], vec![], vec![vec![1.0_f64]]);
    subst.leaky = true;
    subst.allow_input_output_links = true;
    subst.max_weight_and_bias = 6.0;
    subst.min_time_const = 0.1;
    subst.max_time_const = 1.0;

    // Now get CPPN IO sizes
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    // For leaky + query_weights_only=false we expect 4 outputs: link, weight, tc, bias
    assert_eq!(cppn_outputs, 4);

    // Construct CPPN outputs: link_on=1.0, weight=1.0, tc=-1.0 (maps to min_time_const), bias=0.5
    let g = make_fixed_cppn(
        cppn_inputs,
        cppn_outputs,
        &[1.0_f64, 1.0_f64, -1.0_f64, 0.5_f64],
    );
    let mut net = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net, &subst);

    // one input->output connection expected
    assert_eq!(net.connections.len(), 1);
    let conn = &net.connections[0];
    let expected_w = 1.0_f64 * subst.max_weight_and_bias;
    assert_weight_approx(conn.weight, expected_w, 1e-6);

    // Neuron params for the output neuron should be set by CPPN outputs: index of first non-input neuron is 1
    // timeconst mapped from -1.0 -> min_time_const, bias mapped via [-1,1] -> [-max_weight,max_weight]
    let out_neuron = &net.neurons[1];
    assert!((out_neuron.timeconst - subst.min_time_const).abs() < 1e-12);
    let expected_bias = 0.5_f64 * subst.max_weight_and_bias;
    assert_weight_approx(out_neuron.bias, expected_bias, 1e-6);
}

// Test: leaky + query_weights_only combination reduces CPPN outputs while
// still producing correct weight/timeconst/bias mappings for the phenotype.
#[test]
fn build_hyperneat_leaky_query_weights_only_case() {
    // Substrate with leaky and query_weights_only = true
    let mut subst = Substrate::with_coords(vec![vec![-1.0_f64]], vec![], vec![vec![1.0_f64]]);
    subst.leaky = true;
    subst.query_weights_only = true;
    subst.allow_input_output_links = true;
    subst.max_weight_and_bias = 8.0;
    subst.min_time_const = 0.2;
    subst.max_time_const = 2.0;

    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    // query_weights_only + leaky -> outputs = 1 + 2 = 3
    assert_eq!(cppn_outputs, 3);

    // Construct CPPN outputs: weight=0.4, tc=0.0 (maps to mid), bias=-0.5
    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &[0.4_f64, 0.0_f64, -0.5_f64]);
    let mut net = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net, &subst);

    // Connection exists
    assert_eq!(net.connections.len(), 1);
    let conn = &net.connections[0];
    let expected_w = 0.4_f64 * subst.max_weight_and_bias;
    assert_weight_approx(conn.weight, expected_w, 1e-6);

    // Check neuron params: timeconst should map from 0.0 -> mid between min/max
    let out_neuron = &net.neurons[1];
    let expected_tc = (subst.min_time_const + subst.max_time_const) / 2.0;
    assert!((out_neuron.timeconst - expected_tc).abs() < 1e-12);
    let expected_bias = subst.max_weight_and_bias * -0.5_f64;
    assert_weight_approx(out_neuron.bias, expected_bias, 1e-6);
}
