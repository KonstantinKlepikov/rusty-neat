from rusty_neat_py import PyNeuralNetwork


def test_with_dimensions_and_getters_and_reset():
    nn = PyNeuralNetwork.with_dimensions(3, 2)
    assert nn.num_inputs() == 3
    assert nn.num_outputs() == 2

    # reset should clear dimensions
    nn.reset()
    assert nn.num_inputs() == 0
    assert nn.num_outputs() == 0


def test_default_new_has_zero_dims():
    nn = PyNeuralNetwork()
    assert nn.num_inputs() == 0
    assert nn.num_outputs() == 0
