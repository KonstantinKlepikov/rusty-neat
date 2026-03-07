// Tests for RTRL weight application: accumulated changes are applied and cleared
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

// Single connection: the accumulated change is added to weight and cleared
#[test]
fn rtrl_update_weights_single_connection_applies_and_clears() {
    let mut net = NeuralNetwork::new();

    // input + output
    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Output,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.set_input_output_dimensions(1, 1);

    // single connection 0 -> 1 with initial weight
    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.5,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });

    net.init_rtrl_matrix();

    // set an accumulated weight change for the only connection
    net.total_weight_changes_mut()[0] = 0.123;

    net.rtrl_update_weights();

    // weight must be incremented and accumulator cleared
    assert!(approx_eq(net.connections[0].weight, 0.623, 1e-15));
    assert!(approx_eq(net.total_weight_changes()[0], 0.0, 1e-15));
}

// Multiple connections: all accumulators are applied to their connections
#[test]
fn rtrl_update_weights_multiple_connections_applies_all_changes() {
    let mut net = NeuralNetwork::new();

    // create three neurons (1 input + 2 outputs treated as neurons in network)
    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Hidden,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Output,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.set_input_output_dimensions(1, 1);

    // add multiple connections
    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.1,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });
    net.add_connection(Connection {
        source_neuron_idx: 1,
        target_neuron_idx: 2,
        weight: -0.2,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });
    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 2,
        weight: 0.0,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });

    net.init_rtrl_matrix();

    // set different accumulated changes
    net.total_weight_changes_mut()[0] = 0.01;
    net.total_weight_changes_mut()[1] = -0.05;
    net.total_weight_changes_mut()[2] = 0.5;

    net.rtrl_update_weights();

    // verify weights updated
    assert!(approx_eq(net.connections[0].weight, 0.11, 1e-15));
    assert!(approx_eq(net.connections[1].weight, -0.25, 1e-15));
    assert!(approx_eq(net.connections[2].weight, 0.5, 1e-15));

    // and all accumulators cleared
    for &v in net.total_weight_changes() {
        assert!(approx_eq(v, 0.0, 1e-15));
    }
}
