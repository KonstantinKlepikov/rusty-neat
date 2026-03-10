use rusty_neat::network::NeuralNetwork;

#[test]
fn with_minimal_true_creates_empty_network() {
    // with_minimal(true) should yield a minimal/empty network matching C++ behaviour
    let nn = NeuralNetwork::with_minimal(true);

    // expect no inputs/outputs and no neurons/connections allocated
    assert_eq!(nn.num_inputs, 0);
    assert_eq!(nn.num_outputs, 0);
    assert_eq!(nn.neurons.len(), 0);
    assert_eq!(nn.connections.len(), 0);
    // total_weight_change is internal; ensure public state is empty instead
}
