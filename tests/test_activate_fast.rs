use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

#[test]
fn activate_fast_sets_signal_and_propagates() {
    // Build a minimal network with 2 neurons: input (0) and output (1)
    let mut net = NeuralNetwork::new();

    let in_neuron = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let mut out_neuron = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
    // ensure default params
    out_neuron.a = 1.0;
    out_neuron.b = 0.0;

    net.add_neuron(in_neuron);
    net.add_neuron(out_neuron);
    net.set_input_output_dimensions(1, 1);

    // connection 0 -> 1 weight 0.5
    let conn = Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.5,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    net.add_connection(conn);

    // set input activation
    net.input(vec![1.0]);

    // activate_fast
    net.activate_fast();

    // check that connection.signal was set to activation*weight
    assert!((net.connections[0].signal - 0.5).abs() < 1e-12);

    // output neuron activation should be unsigned sigmoid(0.5)
    let expected = 1.0 / (1.0 + (-0.5f64).exp());
    let out = net.output();
    assert_eq!(out.len(), 1);
    assert!((out[0] - expected).abs() < 1e-12);
}

#[test]
fn activate_fast_clears_activesum_and_keeps_inputs() {
    let mut net = NeuralNetwork::new();
    let in_n = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let out_n = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
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

    net.input(vec![0.7]);
    // precondition: out neuron activesum is zero
    assert_eq!(net.neurons[1].activesum, 0.0);

    net.activate_fast();

    // after activation activesum should be cleared
    assert_eq!(net.neurons[1].activesum, 0.0);
    // input activation should remain
    assert!((net.neurons[0].activation - 0.7).abs() < 1e-12);
}
