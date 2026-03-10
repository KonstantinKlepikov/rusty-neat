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


def test_build_hyperneat_phenotype_smoke():
    # Smoke test: create a simple substrate and a CPPN genome via Population,
    # then build HyperNEAT phenotype into a network to ensure bindings work.
    substrate = rnp.PySubstrate.with_coords(
        inputs=[[-1, -1], [-1, 0]],
        hidden=[[0, 0]],
        outputs=[[1, 0]],
    )

    substrate.set_allow_input_hidden_links(True)
    substrate.set_allow_hidden_output_links(True)

    params = rnp.PyParameters()
    min_inputs = substrate.get_min_cppn_inputs()
    min_outputs = substrate.get_min_cppn_outputs()

    init_struct = rnp.PyGenomeInitStruct(
        num_inputs=min_inputs,
        num_hidden=0,
        num_outputs=min_outputs,
        fs_neat=False,
        output_act_type='tanh',
        hidden_act_type='tanh',
        seed_type='minimal',
        num_layers=1,
        fs_neat_links=None,
    )

    pop = rnp.PyPopulation(params)
    pop.add_genome_from_initstruct(init_struct)
    genomes = pop.list_genomes()
    assert len(genomes) > 0
    g = genomes[0]

    net = rnp.PyNeuralNetwork.with_dimensions(min_inputs, min_outputs)
    # Should not raise
    g.build_hyperneat_phenotype(net, substrate)
    out = net.output()
    assert isinstance(out, np.ndarray)
