// Tests for `activate_leaky`: membrane potential integration and outputs
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

#[test]
fn activate_leaky_updates_membrane_and_activation_from_zero() {
    let mut net = NeuralNetwork::new();

    let in_n = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let mut out_n = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
    // timeconst = 2.0 => t_const = a_dtime / timeconst = 1/2
    out_n.timeconst = 2.0;
    out_n.membrane_potential = 0.0;

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

    // perform leaky step with a_dtime = 1.0
    net.activate_leaky(1.0);

    // After step: membrane_potential = 0.5*0 + 0.5*activesum (activesum == 1.0)
    let mp = net.neurons[1].membrane_potential;
    assert!((mp - 0.5).abs() < 1e-12);

    // activation should be unsigned sigmoid(mp)
    let expected = 1.0 / (1.0 + (-0.5f64).exp());
    let out = net.output();
    assert_eq!(out.len(), 1);
    assert!((out[0] - expected).abs() < 1e-12);

    // activesum must be cleared
    assert_eq!(net.neurons[1].activesum, 0.0);
}

// Leaky activation updates from a previous membrane potential
#[test]
fn activate_leaky_updates_from_previous_membrane() {
    let mut net = NeuralNetwork::new();

    let in_n = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let mut out_n = Neuron::new(NeuronType::Output, ActivationFunction::UnsignedSigmoid);
    out_n.timeconst = 2.0;
    out_n.membrane_potential = 0.2; // previous state

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

    // step: t_const = 0.5 => new_mp = 0.5*0.2 + 0.5*1.0 = 0.6
    net.activate_leaky(1.0);
    let mp = net.neurons[1].membrane_potential;
    assert!((mp - 0.6).abs() < 1e-12);

    // activation is sigmoid(0.6)
    let expected = 1.0 / (1.0 + (-0.6f64).exp());
    let out = net.output();
    assert!((out[0] - expected).abs() < 1e-12);
}
