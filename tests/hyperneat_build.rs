use rusty_neat::Genome;
use rusty_neat::Substrate;
use rusty_neat::genes::{ActivationFunction, LinkGene, NeuronGene, NeuronType};

// Helper: approximate weight comparator with configurable tolerance
fn assert_weight_approx(actual: f64, expected: f64, tol: f64) {
    let d = (actual - expected).abs();
    assert!(
        d <= tol,
        "weight differs by {} which is > tol {}: got {} expected {}",
        d,
        tol,
        actual,
        expected
    );
}

// Helper: create a minimal CPPN-like Genome that outputs fixed values on its outputs
// It builds a small feed-forward CPPN where the bias input is connected to every
// CPPN output with the provided weights. Outputs use Linear activation so that
// output == bias * weight (bias input set to 1.0 in BuildHyperNEATPhenotype).
fn make_fixed_cppn(cppn_inputs: usize, cppn_outputs: usize, out_weights: &[f64]) -> Genome {
    assert!(cppn_outputs == out_weights.len());
    let mut g = Genome::default();
    g.num_inputs = cppn_inputs;
    g.num_outputs = cppn_outputs;

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
        g.neuron_genes.push(ng);
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
        g.neuron_genes.push(ng);
        next_id += 1;
    }

    // Connect bias (last input id = cppn_inputs) to every output with given weight
    let bias_id = cppn_inputs as u64; // ids started at 1
    let mut innov = 1u64;
    for (out_idx, &w) in out_weights.iter().enumerate() {
        let out_id = (cppn_inputs + out_idx) as u64 + 1; // since outputs come after inputs
        let lg = LinkGene::new(bias_id, out_id, innov, w, false);
        g.link_genes.push(lg);
        innov += 1;
    }

    g
}

#[test]
fn build_hyperneat_phenotype_scaling_and_query_outputs() {
    // Simple substrate: one input coord, one output coord
    let mut subst = Substrate::with_coords(vec![vec![-1.0_f64]], vec![], vec![vec![1.0_f64]]);
    // Explicitly set max weight for clarity
    subst.max_weight_and_bias = 5.0_f64;

    // Case A: query_weights_only = false -> CPPN must have 2 outputs: link_on, weight
    subst.query_weights_only = false;
    // `with_coords` constructor disables input->output links by default (C++ semantics),
    // enable them for this test so candidate pairs are considered.
    subst.allow_input_output_links = true;
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    // we want link_on = 1.0, weight = 0.5
    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &[1.0_f64, 0.5_f64]);

    let mut net = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net, &subst);

    // One candidate pair -> one connection (assuming flags allow)
    assert_eq!(net.connections.len(), 1);

    let conn = &net.connections[0];
    let expected_weight = 0.5_f64 * subst.max_weight_and_bias;
    assert_weight_approx(conn.weight, expected_weight, 1e-6);

    // Case B: query_weights_only = true -> CPPN must have 1 output (weight only)
    let mut subst2 = subst.clone();
    subst2.query_weights_only = true;
    // ensure input->output connectivity is allowed for the second case as well
    subst2.allow_input_output_links = true;
    let cppn_inputs2 = subst2.get_min_cppn_inputs();
    let cppn_outputs2 = subst2.get_min_cppn_outputs();
    assert_eq!(cppn_outputs2, 1);

    // CPPN now should have single output which we set to 0.25
    let g2 = make_fixed_cppn(cppn_inputs2, cppn_outputs2, &[0.25_f64]);
    let mut net2 = rusty_neat::NeuralNetwork::new();
    g2.build_hyperneat_phenotype(&mut net2, &subst2);

    assert_eq!(net2.connections.len(), 1);
    let conn2 = &net2.connections[0];
    let expected2 = 0.25_f64 * subst2.max_weight_and_bias;
    assert_weight_approx(conn2.weight, expected2, 1e-6);
}

#[test]
fn build_hyperneat_phenotype_respects_flags_allow_input_output() {
    // Substrate with 1 input and 1 output
    let mut subst = Substrate::with_coords(vec![vec![-1.0_f64]], vec![], vec![vec![1.0_f64]]);
    subst.max_weight_and_bias = 3.0;
    // CPPN that always turns link on and weight 1.0
    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &[1.0_f64, 1.0_f64]);

    // allow input->output = true
    subst.allow_input_output_links = true;
    let mut net_allow = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net_allow, &subst);
    assert_eq!(net_allow.connections.len(), 1);

    // disallow input->output
    subst.allow_input_output_links = false;
    let mut net_disallow = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net_disallow, &subst);
    assert_eq!(net_disallow.connections.len(), 0);
}

#[test]
fn build_hyperneat_multiple_inputs_outputs() {
    // Substrate: 2 inputs (2D), 3 outputs (2D)
    let mut subst = Substrate::with_coords(
        vec![vec![-1.0_f64, -1.0], vec![0.0_f64, 0.0]],
        vec![],
        vec![vec![1.0_f64, 1.0], vec![0.5_f64, 0.5], vec![0.25_f64, 0.25]],
    );
    subst.max_weight_and_bias = 4.0;
    subst.allow_input_output_links = true;

    let cppn_inputs = subst.get_min_cppn_inputs();
    let cppn_outputs = subst.get_min_cppn_outputs();
    // two outputs: link_on, weight
    assert_eq!(cppn_outputs, 2);

    let g = make_fixed_cppn(cppn_inputs, cppn_outputs, &[1.0_f64, 0.25_f64]);
    let mut net = rusty_neat::NeuralNetwork::new();
    g.build_hyperneat_phenotype(&mut net, &subst);

    // expected connections = num_inputs * num_outputs = 2 * 3 = 6
    assert_eq!(net.connections.len(), 6);
    for c in &net.connections {
        let expected = 0.25_f64 * subst.max_weight_and_bias;
        assert_weight_approx(c.weight, expected, 1e-6);
    }
}

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
