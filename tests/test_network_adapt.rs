use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};
use rusty_neat::parameters::Parameters;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn adapt_positive_weight_updates_towards_max() {
    let mut net = NeuralNetwork::new();

    // three neurons: src0, src1, tgt
    let n0 = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let n1 = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let nt = Neuron::new(NeuronType::Hidden, ActivationFunction::UnsignedSigmoid);
    net.add_neuron(n0);
    net.add_neuron(n1);
    net.add_neuron(nt);

    net.set_input_output_dimensions(2, 1);

    // connection 0 -> 2 with weight 0.8 (this will be t_max_weight)
    let c0 = Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 2,
        weight: 0.8,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    // connection 1 -> 2 with weight 0.5 (this one will be adapted)
    let c1 = Connection {
        source_neuron_idx: 1,
        target_neuron_idx: 2,
        weight: 0.5,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.1,
        hebb_pre_rate: 0.0,
    };

    net.add_connection(c0);
    net.add_connection(c1.clone());

    // set activations (incoming and outgoing)
    net.neurons[1].activation = 1.0; // source for c1
    net.neurons[2].activation = 1.0; // target

    let mut params = Parameters::default();
    params.max_weight = 10.0;

    // expected delta = hebb_rate * (t_max - weight) * in * out
    let t_max = 0.8_f64;
    let expected_delta = c1.hebb_rate * (t_max - c1.weight) * 1.0 * 1.0;
    let expected = c1.weight + expected_delta;

    net.adapt(&params);

    let got = net.connections[1].weight;
    assert!(approx_eq(got, expected), "got {} expected {}", got, expected);
}

#[test]
fn adapt_negative_weight_case_flips_sign_as_in_cpp() {
    let mut net = NeuralNetwork::new();

    let n0 = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let n1 = Neuron::new(NeuronType::Input, ActivationFunction::UnsignedSigmoid);
    let nt = Neuron::new(NeuronType::Hidden, ActivationFunction::UnsignedSigmoid);
    net.add_neuron(n0);
    net.add_neuron(n1);
    net.add_neuron(nt);

    net.set_input_output_dimensions(2, 1);

    // connection 0 -> 2 weight 0.8 (t_max)
    let c0 = Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 2,
        weight: 0.8,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    // connection 1 -> 2 weight -0.5 (negative case)
    let c1 = Connection {
        source_neuron_idx: 1,
        target_neuron_idx: 2,
        weight: -0.5,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.2,
    };

    net.add_connection(c0);
    net.add_connection(c1.clone());

    // activations: incoming=1, outgoing=0 to hit the (1 - out) branch
    net.neurons[1].activation = 1.0;
    net.neurons[2].activation = 0.0;

    let mut params = Parameters::default();
    params.max_weight = 10.0;

    // compute expected for negative case per C++:
    // t_delta = hebb_pre_rate * (t_max - weight) * in * (1 - out) - hebb_rate * t_max * in * out
    // new weight = -(weight + t_delta)
    let t_max = 0.8_f64;
    let t_delta = c1.hebb_pre_rate * (t_max - c1.weight) * 1.0 * (1.0 - 0.0)
        - c1.hebb_rate * t_max * 1.0 * 0.0;
    let expected = -(c1.weight + t_delta);

    net.adapt(&params);
    let got = net.connections[1].weight;
    assert!(approx_eq(got, expected), "got {} expected {}", got, expected);
}
