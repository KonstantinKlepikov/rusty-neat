// Test that `activate()` ignores internal neuron bias while
// `activate_use_internal_bias()` applies it to the activation input.
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

/// `activate()` should ignore neuron `bias`, while
/// `activate_use_internal_bias()` should include it in the activation input.
#[test]
fn activate_ignores_internal_bias_but_activate_use_internal_bias_applies_it() {
    let mut net = NeuralNetwork::new();

    let in_n = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let mut out_n = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
    // set a non-zero internal bias on the output neuron
    out_n.bias = 0.5;

    net.add_neuron(in_n);
    net.add_neuron(out_n);
    net.set_input_output_dimensions(1, 1);

    let conn = Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 1.0,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    net.add_connection(conn);

    net.input(vec![1.0]);

    // activate() should NOT include bias: x = activesum = 1.0
    net.activate();
    let out = net.output();
    let expected_no_bias = 1.0 / (1.0 + (-1.0f64).exp());
    assert!((out[0] - expected_no_bias).abs() < 1e-12);

    // reset and set input again
    net.input(vec![1.0]);
    // use method that applies internal bias
    net.activate_use_internal_bias();
    let out2 = net.output();
    let expected_with_bias = 1.0 / (1.0 + (-(1.0f64 + 0.5f64)).exp());
    assert!((out2[0] - expected_with_bias).abs() < 1e-12);
}
