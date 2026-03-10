import rusty_neat_py as rnp


def test_substrate_flags_and_min_cppn():
    s = rnp.PySubstrate()

    # default leaky = false
    assert s.get_leaky() is False
    s.set_leaky(True)
    assert s.get_leaky() is True

    # query_weights_only default false => base outs = 2; leaky true => +2 => 4
    assert s.get_min_cppn_outputs() == 4

    # change query_weights_only to true: base outs = 1; leaky true => +2 => 3
    s.set_query_weights_only(True)
    assert s.get_query_weights_only() is True
    assert s.get_min_cppn_outputs() == 3

    # toggling with_distance should affect min cppn inputs
    base_inputs = s.get_min_cppn_inputs()
    s.set_with_distance(True)
    assert s.get_with_distance() is True
    assert s.get_min_cppn_inputs() == base_inputs + 1

    # connectivity flag roundtrip
    assert s.get_allow_input_output_links() in (True, False)
    s.set_allow_input_output_links(False)
    assert s.get_allow_input_output_links() is False
    s.set_allow_input_output_links(True)
    assert s.get_allow_input_output_links() is True

    # activation function set/get
    s.set_output_nodes_activation('Tanh')
    assert s.get_output_nodes_activation() == 'Tanh'
    s.set_hidden_nodes_activation('Relu')
    assert s.get_hidden_nodes_activation() == 'Relu'
