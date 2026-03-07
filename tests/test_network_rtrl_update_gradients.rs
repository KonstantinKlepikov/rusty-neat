// Tests for RTRL gradient computation for different activation functions
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

// Unsigned sigmoid: gradient entries should be y*(1-y)*pre_activation
#[test]
fn rtrl_update_gradients_unsigned_sigmoid_basic() {
    let mut net = NeuralNetwork::new();

    // create input (0) and output (1)
    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Output,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.set_input_output_dimensions(1, 1);

    // add a connection 0 -> 1
    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 1.0,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });

    // initialize RTRL matrices (zeros)
    net.init_rtrl_matrix();

    // set activations: input and output
    net.neurons[0].activation = 0.6; // input
    net.neurons[1].activation = 0.4; // output (used for derivative)

    // run gradients update
    net.rtrl_update_gradients();

    // For i=1 (to), j=0 (from) there is a connection, i==k==1 so t_sum = neuron[j].activation
    // derivative for unsigned sigmoid = y*(1-y) where y is output activation (0.4)
    let deriv = 0.4 * (1.0 - 0.4); // 0.24
    let expected = deriv * 0.6; // 0.144

    let val = net.neurons[1].sensitivity_matrix[1][0];
    assert!(
        approx_eq(val, expected, 1e-12),
        "expected {} got {}",
        expected,
        val
    );

    // entry for j=1 should be zero (no connection 1->1)
    assert!(approx_eq(
        net.neurons[1].sensitivity_matrix[1][1],
        0.0,
        1e-12
    ));
}

// Tanh activation uses derivative 1 - y^2
#[test]
fn rtrl_update_gradients_tanh_derivative() {
    let mut net = NeuralNetwork::new();

    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    // use TANH for output
    let out = Neuron::new(NeuronType::Output, ActivationFunction::Tanh);
    net.add_neuron(out);
    net.set_input_output_dimensions(1, 1);

    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 1.0,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });

    net.init_rtrl_matrix();

    net.neurons[0].activation = 0.2;
    net.neurons[1].activation = 0.5; // for tanh derivative -> 1 - x^2 = 0.75

    net.rtrl_update_gradients();

    let deriv = 1.0 - 0.5 * 0.5; // 0.75
    let expected = deriv * 0.2; // 0.15

    let val = net.neurons[1].sensitivity_matrix[1][0];
    assert!(
        approx_eq(val, expected, 1e-12),
        "expected {} got {}",
        expected,
        val
    );
}
