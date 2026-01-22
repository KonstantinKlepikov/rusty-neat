import rusty_neat_py


def test_link_gene_simple():
    lg = rusty_neat_py.PyLinkGene(1, 2, 42, 0.5, False)
    assert lg.from_neuron_id == 1
    assert lg.to_neuron_id == 2
    assert lg.innovation_id == 42
    assert abs(lg.weight - 0.5) < 1e-12
    lg.weight = 0.25
    assert abs(lg.weight - 0.25) < 1e-12


def test_neuron_gene_simple():
    ng = rusty_neat_py.PyNeuronGene(7, x=3, y=4, split_y=0.2, a=1.1, b=0.0, timeconstant=1.0, bias=0.0)
    assert ng.id == 7
    assert ng.x == 3 and ng.y == 4
    assert abs(ng.split_y - 0.2) < 1e-12
    ng.x = 10
    assert ng.x == 10


def test_genome_init_struct_defaults():
    gis = rusty_neat_py.PyGenomeInitStruct(2, 3, 1)
    assert gis.num_inputs == 2
    assert gis.num_hidden == 3
    assert gis.num_outputs == 1
    assert gis.fs_neat is False
    assert isinstance(gis.output_act_type, str)
