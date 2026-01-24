import rusty_neat_py as rnp


def test_base() -> None:
    """smoke test: import extension and call basic API"""

    # Parameters
    params = rnp.PyParameters()
    mw = params.get_min_weight()
    assert isinstance(mw, float)

    # Substrate
    substrate = rnp.PySubstrate()
    mins = substrate.get_min_cppn_inputs()
    assert isinstance(mins, int)

    # RNG
    rng = rnp.PyRNG()
    rf = rng.rand_float()
    assert isinstance(rf, float)

    # Neural network basic flow
    nn = rnp.PyNeuralNetwork()
    nn.input([0.1, 0.2, 0.3])
    nn.activate()
    out = nn.output()
    # output may be a list-like Python object
    assert out is not None

    # Genome
    g = rnp.PyGenome(1, 3, 0, 1)
    assert g.get_id() == 1
