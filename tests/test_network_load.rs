// Tests for `save_writer` + `load_reader` roundtrip
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};
use std::io::Cursor;

// Ensure saving to a buffer and loading back reconstructs the network
#[test]
fn save_then_load_roundtrip() {
    let mut net = NeuralNetwork::new();

    net.add_neuron(Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid));
    net.add_neuron(Neuron::new(NeuronType::Output, ActivationFunction::Tanh));
    net.set_input_output_dimensions(1, 1);

    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.777,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.11,
        hebb_pre_rate: 0.22,
    });

    // write to buffer
    let mut buf: Vec<u8> = Vec::new();
    net.save_writer(&mut buf).expect("save failed");

    // load into a fresh network
    let mut net2 = NeuralNetwork::new();
    let mut cursor = Cursor::new(buf);
    let ok = net2.load_reader(&mut cursor);
    assert!(ok, "load_reader returned false");

    assert_eq!(net2.num_inputs, net.num_inputs);
    assert_eq!(net2.num_outputs, net.num_outputs);
    assert_eq!(net2.neurons.len(), net.neurons.len());
    assert_eq!(net2.connections.len(), net.connections.len());
    assert!((net2.connections[0].weight - 0.777).abs() < 1e-12);
}
