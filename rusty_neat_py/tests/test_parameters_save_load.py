from rusty_neat_py import PyParameters


def test_parameters_save_load(tmp_path):
    p = PyParameters()
    p.set_min_weight(-3.14)
    path = tmp_path / 'params.json'
    p.save(str(path))
    assert path.exists()

    # change and reload
    p.set_min_weight(0.0)
    p.load(str(path))
    assert abs(p.get_min_weight() - -3.14) < 1e-12
