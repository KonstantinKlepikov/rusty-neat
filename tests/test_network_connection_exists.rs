// Tests for `connection_exists` API: ensures correct Some(index)/None results
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

// Existing connection should return its index
#[test]
fn connection_exists_returns_index_for_existing_connection() {
    // Build network with inputs and outputs similar to with_minimal
    let mut net = NeuralNetwork::new();

    // two neurons: input(0) and output(1)
    let in_n = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let out_n = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
    net.add_neuron(in_n);
    net.add_neuron(out_n);
    net.set_input_output_dimensions(1, 1);

    let conn = Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.25,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    net.add_connection(conn);

    // existing connection should return Some(0)
    assert_eq!(net.connection_exists(1, 0), Some(0));
}

// Missing connection queries should return None
#[test]
fn connection_exists_returns_none_for_missing_connection() {
    let mut net = NeuralNetwork::new();
    // three neurons
    for _ in 0..3 {
        net.add_neuron(Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid));
    }
    net.set_input_output_dimensions(3, 0);

    // no connections added -> any query returns None
    assert_eq!(net.connection_exists(2, 1), None);
}

// Verify `with_minimal` topology contains the expected connections
#[test]
fn connection_exists_matches_with_minimal_topology() {
    // Use the convenience constructor to create the example topology
    let net = NeuralNetwork::with_minimal(false);

    // Per implementation, first connection is from 0 -> 3
    assert_eq!(net.connection_exists(3, 0), Some(0));
    // last connection (index 6) is from 4 -> 3
    assert_eq!(net.connection_exists(3, 4), Some(6));
    // a non-existing pair should be None
    assert_eq!(net.connection_exists(0, 4), None);
}
