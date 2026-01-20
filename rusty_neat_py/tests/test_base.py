import rusty_neat_py


def test_base() -> None:
    """smoke test: import extension and call basic API"""

    # Parameters
    params = rusty_neat_py.PyParameters()
    mw = params.get_min_weight()
    assert isinstance(mw, float)

    # Substrate
    substrate = rusty_neat_py.PySubstrate()
    mins = substrate.get_min_cppn_inputs()
    assert isinstance(mins, int)

    # RNG
    rng = rusty_neat_py.PyRNG()
    rf = rng.rand_float()
    assert isinstance(rf, float)

    # Neural network basic flow
    nn = rusty_neat_py.PyNeuralNetwork()
    nn.input([0.1, 0.2, 0.3])
    nn.activate()
    out = nn.output()
    # output may be a list-like Python object
    assert out is not None

    # Genome
    g = rusty_neat_py.PyGenome(1, 3, 0, 1)
    assert g.get_id() == 1
