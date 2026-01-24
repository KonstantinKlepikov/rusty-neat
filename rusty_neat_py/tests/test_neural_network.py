import os
import tempfile

import numpy as np
import rusty_neat_py as rnp


def test_input_activate_output_and_save_load():
    net = rnp.PyNeuralNetwork()

    # input as python list — should not raise
    net.input([0.1, 0.2, 0.3])

    # input as numpy ndarray — should not raise
    arr = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    net.input(arr)

    # activate (should release GIL internally) — no exception
    net.activate(steps=1)

    # output should be a numpy array (possibly empty for an uninitialized net)
    out = net.output()
    assert isinstance(out, np.ndarray)

    # Save to a temporary file and load back
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        net.save(path)
        loaded = net.load(path)
        # basic sanity: loaded object exists and has output() method
        out2 = loaded.output()
        assert isinstance(out2, np.ndarray)
    finally:
        os.remove(path)
