import importlib
import tempfile
import os
import pytest
import numpy as np


def _import_module_or_skip():
    try:
        return importlib.import_module("rusty_neat_py")
    except Exception as e:
        pytest.skip(f"rusty_neat_py extension not available: {e}")


def test_input_activate_output_and_save_load():
    m = _import_module_or_skip()

    NN = m.PyNeuralNetwork

    net = NN()

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
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        net.save(path)
        loaded = NN.load(path)
        # basic sanity: loaded object exists and has output() method
        out2 = loaded.output()
        assert isinstance(out2, np.ndarray)
    finally:
        os.remove(path)
