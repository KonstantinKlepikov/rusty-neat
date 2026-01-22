import pickle

import rusty_neat_py as rnp


def test_substrate_pickle_roundtrip():
    # create a substrate with explicit coords and non-default flags
    inputs = [[0.0, 0.0], [0.1, 0.2]]
    hidden = [[1.0, 1.0]]
    outputs = [[2.0, 2.0]]

    s = rnp.PySubstrate.with_coords(inputs, hidden, outputs)
    s.set_leaky(True)
    s.set_with_distance(True)
    s.set_query_weights_only(True)
    s.set_output_nodes_activation("Tanh")
    s.set_hidden_nodes_activation("Relu")
    s.set_custom_connectivity([[0, 1], [1, 2]])

    data = pickle.dumps(s)
    s2 = pickle.loads(data)

    # flags and activations should survive roundtrip
    assert s2.get_leaky() is True
    assert s2.get_with_distance() is True
    assert s2.get_query_weights_only() is True
    assert s2.get_output_nodes_activation() == "Tanh"
    assert s2.get_hidden_nodes_activation() == "Relu"

