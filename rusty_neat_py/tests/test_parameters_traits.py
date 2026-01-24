import rusty_neat_py as rnp


def test_neuron_trait_parameters_roundtrip():
    p = rnp.PyParameters()

    tp = {
        'importance_coeff': 2.0,
        'mutation_prob': 0.5,
        'detail': {
            'type': 'Int',
            'min': -5,
            'max': 5,
            'mut_power': 2,
            'mut_replace_prob': 0.1,
        },
        'dep_key': None,
        'dep_values': [],
    }

    # set
    p.set_neuron_trait_parameters('mytrait', tp)

    names = p.list_neuron_trait_parameters()
    assert 'mytrait' in names

    got = p.get_neuron_trait_parameters('mytrait')
    assert got is not None
    # got should be a dict-like mapping
    assert abs(got['importance_coeff'] - 2.0) < 1e-12
    assert abs(got['mutation_prob'] - 0.5) < 1e-12
    assert got['detail']['type'] == 'Int'
    assert got['detail']['min'] == -5
    assert got['detail']['max'] == 5


def test_link_and_genome_trait_setters():
    p = rnp.PyParameters()

    tp_f = {
        'importance_coeff': 1.0,
        'mutation_prob': 0.2,
        'detail': {
            'type': 'Float',
            'min': 0.0,
            'max': 1.0,
            'mut_power': 0.1,
            'mut_replace_prob': 0.05,
        },
        'dep_key': None,
        'dep_values': [],
    }

    p.set_link_trait_parameters('ltrait', tp_f)
    assert 'ltrait' in p.list_link_trait_parameters()
    got = p.get_link_trait_parameters('ltrait')
    assert got is not None
    assert got['detail']['type'] == 'Float'

    p.set_genome_trait_parameters('gtrait', tp_f)
    assert 'gtrait' in p.list_genome_trait_parameters()
    gotg = p.get_genome_trait_parameters('gtrait')
    assert gotg is not None
    assert gotg['detail']['type'] == 'Float'
