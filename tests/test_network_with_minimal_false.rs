use rusty_neat::network::NeuralNetwork;

#[test]
fn with_minimal_false_creates_xor_example() {
    let nn = NeuralNetwork::with_minimal(false);

    // structural expectations matching the C++ constructor
    assert_eq!(nn.num_inputs, 3);
    assert_eq!(nn.num_outputs, 1);
    assert_eq!(nn.neurons.len(), 5);
    assert_eq!(nn.connections.len(), 7);

    // weights randomized into [-0.5, 0.5]
    for c in &nn.connections {
        assert!(
            c.weight >= -0.5 && c.weight <= 0.5,
            "weight out of range: {}",
            c.weight
        );
    }

    // init_rtrl_matrix should have allocated sensitivity matrices sized n x n and zeroed them
    let n = nn.neurons.len();
    for neuron in &nn.neurons {
        assert_eq!(neuron.sensitivity_matrix.len(), n);
        for row in &neuron.sensitivity_matrix {
            assert_eq!(row.len(), n);
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }

        // neuron params set by constructor cleanup
        assert_eq!(neuron.a, 1.0);
        assert_eq!(neuron.b, 0.0);
        assert_eq!(neuron.timeconst, 0.0);
        assert_eq!(neuron.bias, 0.0);
        assert_eq!(neuron.membrane_potential, 0.0);
    }
}
