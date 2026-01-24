import numpy as np

import rusty_neat_py as rnp


def test_input_activate_output_list_and_ndarray(tmp_path):
    # Create a genome with 2 inputs and 1 output, build phenotype into a network
    genome = rnp.PyGenome(1, 2, 0, 1)
    net = rnp.PyNeuralNetwork()
    genome.build_phenotype(net)

    # Provide inputs as a Python list
    net.input([0.5, -0.5])
    net.activate()
    out1 = net.output()
    assert isinstance(out1, np.ndarray)
    assert out1.ndim == 1

    # Provide inputs as a NumPy ndarray
    net.input(np.array([0.1, 0.2], dtype=float))
    net.activate(steps=2)
    out2 = net.output()
    assert isinstance(out2, np.ndarray)

    # Save and load (JSON) round-trip
    p = tmp_path / 'net.json'
    net.save(str(p))
    net2 = rnp.PyNeuralNetwork.load(str(p))
    assert isinstance(net2, rnp.PyNeuralNetwork)

    # Loaded network should be callable (output returns ndarray)
    out3 = net2.output()
    assert isinstance(out3, np.ndarray)


def test_build_hyperneat_phenotype(tmp_path):
    # Create a genome that will act as a CPPN:
    # number of inputs must be >= substrate.get_min_cppn_inputs()
    # For 2D coords min_cppn_inputs = 2*2 + 1 = 5,
    # and min outputs = 2 (link output + bias/other)
    genome = rnp.PyGenome(2, 5, 0, 2)
    net = rnp.PyNeuralNetwork()

    # Create a simple substrate: two input coords, no hidden, one output coord
    substrate = rnp.PySubstrate.with_coords([[0.0, 0.0], [1.0, 0.0]], [], [[0.5, 0.5]])

    # Build HyperNEAT phenotype
    genome.build_hyperneat_phenotype(net, substrate)

    # Use the network (smoke) — should not raise and return numpy arrays
    net.input([0.1, 0.2])
    net.activate()
    out = net.output()
    assert isinstance(out, np.ndarray)
