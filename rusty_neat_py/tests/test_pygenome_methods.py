import pytest

import rusty_neat_py


def test_pygenome_get_set_save(tmp_path):
    # construct a simple genome
    g = rusty_neat_py.PyGenome(1, 1, 0, 1)

    # default fitness should be 0.0
    assert pytest.approx(g.get_fitness()) == 0.0

    # set and get fitness
    g.set_fitness(1.2345)
    assert pytest.approx(g.get_fitness(), rel=1e-12) == 1.2345

    # save should create a file (debug representation)
    p = tmp_path / "genome_dump.txt"
    g.save(str(p))
    assert p.exists()
    data = p.read_text()
    assert "Genome" in data or "id" in data


def test_pygenome_mutate_and_randomize_no_links():
    # empty genome (no links): mutate_link_weights should return False and not crash
    g = rusty_neat_py.PyGenome(2, 1, 0, 1)
    res = g.mutate_link_weights()
    assert res is False

    # randomize_link_weights should be a no-op and not raise
    g.randomize_link_weights()


# End of file
