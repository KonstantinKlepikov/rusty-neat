import pickle

import rusty_neat_py as rnp


def test_parameters_getstate_setstate_direct():
    p = rnp.PyParameters()

    # get state, mutate it and apply to a fresh object
    state = p.__getstate__()
    assert isinstance(state, dict)

    # change a numeric field and ensure __setstate__ applies it
    state['min_weight'] = 3.14159
    p2 = rnp.PyParameters()
    p2.__setstate__(state)
    assert abs(p2.get_min_weight() - 3.14159) < 1e-12


def test_parameters_pickle_roundtrip():
    p = rnp.PyParameters()
    # use setter API to change a value so pickle will capture non-default
    p.set_min_weight(0.424242)

    data = pickle.dumps(p)
    p2 = pickle.loads(data)

    assert abs(p2.get_min_weight() - 0.424242) < 1e-12
