// Tests for RTRL error propagation: total_weight_changes accumulator updated
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

// Ensure `rtrl_update_error` multiplies sensitivity by error and learning rate
#[test]
fn rtrl_update_error_updates_total_weight_change() {
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

    // initialize RTRL matrices (zeros) and total_weight_change
    net.init_rtrl_matrix();

    // set activations: input and output
    net.neurons[0].activation = 0.3; // input
    net.neurons[1].activation = 0.25; // output

    // set sensitivity for the first output neuron at [i=1][j=0]
    // find connection indices for convenience
    let t_idx = 0usize; // only one connection was added, index 0
    // set sensitivity in the output neuron's sensitivity matrix
    net.neurons[net.num_inputs].sensitivity_matrix[1][0] = 2.0;

    // choose a target and run error update
    let target = 1.0;
    let out0 = net.output().get(0).cloned().unwrap_or(0.0);
    let total_error = target - out0; // expected _total_error inside method

    net.rtrl_update_error(target);

    // expected change = LEARNING_RATE * total_error * sens
    let sens = 2.0;
    let expected = 0.0001 * total_error * sens;

    let val = net.total_weight_changes()[t_idx];
    assert!(
        approx_eq(val, expected, 1e-15),
        "expected {} got {}",
        expected,
        val
    );
}
