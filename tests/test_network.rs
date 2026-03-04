use rusty_neat::network::{Connection, Neuron, NeuralNetwork};
use rusty_neat::genes::{ActivationFunction, NeuronType};

#[test]
fn activate_sets_signal_and_propagates() {
    // Build a minimal network with 2 neurons: input (0) and output (1)
    let mut net = NeuralNetwork::new();

    let in_neuron = Neuron {
        activesum: 0.0,
        activation: 0.0,
        a: 1.0,
        b: 0.0,
        timeconst: 1.0,
        bias: 0.0,
        membrane_potential: 0.0,
        activation_function_type: ActivationFunction::UnsignedSigmoid,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        sx: 0.0,
        sy: 0.0,
        sz: 0.0,
        substrate_coords: Vec::new(),
        split_y: 0.0,
        neuron_type: NeuronType::Input,
        sensitivity_matrix: Vec::new(),
    };

    let out_neuron = Neuron { activation: 0.0, activesum: 0.0, a:1.0, b:0.0, timeconst:1.0, bias:0.0, membrane_potential:0.0, activation_function_type: ActivationFunction::UnsignedSigmoid, x:0.0, y:0.0, z:0.0, sx:0.0, sy:0.0, sz:0.0, substrate_coords: Vec::new(), split_y:0.0, neuron_type: NeuronType::Output, sensitivity_matrix: Vec::new() };

    net.add_neuron(in_neuron);
    net.add_neuron(out_neuron);
    net.set_input_output_dimensions(1, 1);

    // connection 0 -> 1 weight 0.5
    let conn = Connection { source_neuron_idx: 0, target_neuron_idx: 1, weight: 0.5, signal: 0.0, recur_flag:false, hebb_rate:0.0, hebb_pre_rate:0.0 };
    net.add_connection(conn);

    // set input activation
    net.input(vec![1.0]);

    // activate
    net.activate();

    // check that connection.signal was set to activation*weight
    assert!((net.connections[0].signal - 0.5).abs() < 1e-12);

    // output neuron activation should be non-zero after activation
    let out = net.output();
    assert_eq!(out.len(), 1);
    assert!(out[0] >= 0.0 && out[0] <= 1.0);
}

#[test]
#[should_panic]
fn activate_panics_on_invalid_indices_source() {
    let mut net = NeuralNetwork::new();
    // no neurons
    let bad_conn = Connection { source_neuron_idx: 5, target_neuron_idx: 0, weight: 1.0, signal:0.0, recur_flag:false, hebb_rate:0.0, hebb_pre_rate:0.0 };
    net.add_connection(bad_conn);
    net.activate(); // should panic due to invalid source index
}

#[test]
#[should_panic]
fn activate_panics_on_invalid_indices_target() {
    let mut net = NeuralNetwork::new();
    let n = Neuron { activesum:0.0, activation:0.0, a:1.0, b:0.0, timeconst:1.0, bias:0.0, membrane_potential:0.0, activation_function_type: ActivationFunction::UnsignedSigmoid, x:0.0, y:0.0, z:0.0, sx:0.0, sy:0.0, sz:0.0, substrate_coords: Vec::new(), split_y:0.0, neuron_type: NeuronType::Hidden, sensitivity_matrix: Vec::new() };
    net.add_neuron(n);
    let bad_conn = Connection { source_neuron_idx: 0, target_neuron_idx: 5, weight: 1.0, signal:0.0, recur_flag:false, hebb_rate:0.0, hebb_pre_rate:0.0 };
    net.add_connection(bad_conn);
    net.activate(); // should panic due to invalid target index
}
