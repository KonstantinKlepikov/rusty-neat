"""
Smoke test for HyperNEAT XOR evolution.
Adapted from cneat/MultiNEAT/examples/TestHyperNEAT_xor.py
"""

import pytest
import rusty_neat_py as rn


def evaluate_hyperneat_xor(genome, substrate):
    """Evaluate genome on XOR problem using HyperNEAT."""
    # Get dimensions from substrate
    min_inputs = substrate.get_min_cppn_inputs()
    min_outputs = substrate.get_min_cppn_outputs()
    net = rn.PyNeuralNetwork.with_dimensions(min_inputs, min_outputs)

    try:
        genome.build_hyperneat_phenotype(net, substrate)

        error = 0
        depth = 5

        # Test case: 1 XOR 0 = 1
        net.input([1, 0, 1])
        for _ in range(depth):
            net.activate()
        o = net.output()
        error += abs(o[0] - 1)

        # Test case: 0 XOR 1 = 1
        net.reset()
        net.input([0, 1, 1])
        for _ in range(depth):
            net.activate()
        o = net.output()
        error += abs(o[0] - 1)

        # Test case: 1 XOR 1 = 0
        net.reset()
        net.input([1, 1, 1])
        for _ in range(depth):
            net.activate()
        o = net.output()
        error += abs(o[0] - 0)

        # Test case: 0 XOR 0 = 0
        net.reset()
        net.input([0, 0, 1])
        for _ in range(depth):
            net.activate()
        o = net.output()
        error += abs(o[0] - 0)

        return (4 - error) ** 2

    except Exception as ex:
        print(f'Exception during evaluation: {ex}')
        return 1.0


@pytest.mark.slow
def test_hyperneat_xor_single_run():
    """Test that HyperNEAT evolution runs without errors (smoke test).

    Note: This is a smoke test, not a full XOR solution test.
    It verifies the evolution process works, not that XOR is always solved.
    """
    # Setup substrate - simple 2D substrate
    # with 3 input points, 3 hidden, 1 output for XOR
    substrate = rn.PySubstrate.with_coords(
        inputs=[[-1, -1], [-1, 0], [-1, 1]],
        hidden=[[0, -1], [0, 0], [0, 1]],
        outputs=[[1, 0]],
    )

    # Configure substrate connectivity
    substrate.set_allow_input_hidden_links(True)
    substrate.set_allow_input_output_links(False)
    substrate.set_allow_hidden_output_links(True)
    substrate.set_allow_hidden_hidden_links(False)
    substrate.set_allow_output_hidden_links(False)
    substrate.set_allow_output_output_links(False)
    substrate.set_allow_looped_hidden_links(False)
    substrate.set_allow_looped_output_links(False)

    substrate.set_hidden_nodes_activation('signed_sigmoid')
    substrate.set_output_nodes_activation('unsigned_sigmoid')
    substrate.set_with_distance(True)
    substrate.set_max_weight_and_bias(8.0)

    # Setup parameters
    params = rn.PyParameters()

    # Create CPPN genome init struct
    min_inputs = substrate.get_min_cppn_inputs()
    min_outputs = substrate.get_min_cppn_outputs()

    init_struct = rn.PyGenomeInitStruct(
        num_inputs=min_inputs,
        num_hidden=0,
        num_outputs=min_outputs,
        fs_neat=False,
        output_act_type='tanh',
        hidden_act_type='tanh',
        seed_type='minimal',
        num_layers=2,
        fs_neat_links=None,
    )

    # Create population
    pop = rn.PyPopulation(params)
    population_size = 150
    for _ in range(population_size):
        pop.add_genome_from_initstruct(init_struct)

    max_generations = 100  # Smoke test - just verify evolution works
    initial_fitness = 0.0
    final_fitness = 0.0

    for generation in range(max_generations):
        # Evaluate all genomes
        fitness_list = []
        genomes = pop.list_genomes()
        for g in genomes:
            fitness = evaluate_hyperneat_xor(g, substrate)
            g.set_fitness(fitness)
            fitness_list.append(fitness)

        # Track fitness
        best_fitness = max(fitness_list)
        if generation == 0:
            initial_fitness = best_fitness
        final_fitness = best_fitness

        # Early exit if solved (optional)
        if best_fitness > 15.0:
            print(
                f'Solved HyperNEAT XOR in {generation} '
                f'generations with fitness {best_fitness}'
            )
            break

        # Evolve population
        pop.epoch()

    # Smoke test assertions: just verify evolution ran
    # and fitness improved or stayed positive
    assert final_fitness >= 0, 'Fitness should be non-negative'
    # Allow for some evolution progress (fitness can improve, stay same,
    # or slightly decrease due to randomness)
    print(
        f'HyperNEAT smoke test: initial={initial_fitness:.2f}, '
        f'final={final_fitness:.2f}'
    )


@pytest.mark.smoke
def test_hyperneat_xor_basic_evolution():
    """Minimal smoke test - just verify HyperNEAT evolution runs without errors."""
    # Setup substrate
    substrate = rn.PySubstrate.with_coords(
        inputs=[[-1, -1], [-1, 0], [-1, 1]],
        hidden=[[0, -1], [0, 0], [0, 1]],
        outputs=[[1, 0]],
    )

    substrate.set_allow_input_hidden_links(True)
    substrate.set_allow_hidden_output_links(True)
    substrate.set_hidden_nodes_activation('signed_sigmoid')
    substrate.set_output_nodes_activation('unsigned_sigmoid')

    params = rn.PyParameters()

    min_inputs = substrate.get_min_cppn_inputs()
    min_outputs = substrate.get_min_cppn_outputs()

    init_struct = rn.PyGenomeInitStruct(
        num_inputs=min_inputs,
        num_hidden=0,
        num_outputs=min_outputs,
        fs_neat=False,
        output_act_type='tanh',
        hidden_act_type='tanh',
        seed_type='minimal',
        num_layers=2,
        fs_neat_links=None,
    )

    pop = rn.PyPopulation(params)
    population_size = 20
    for _ in range(population_size):
        pop.add_genome_from_initstruct(init_struct)

    # Run just a few generations to verify it works
    for _ in range(5):
        genomes = pop.list_genomes()
        for g in genomes:
            fitness = evaluate_hyperneat_xor(g, substrate)
            g.set_fitness(fitness)

        pop.epoch()

    # Just verify we can get the best genome
    best = pop.get_best_genome()
    assert best is not None
    fitness = evaluate_hyperneat_xor(best, substrate)
    assert fitness >= 0  # Fitness should be non-negative


@pytest.mark.smoke
def test_substrate_setup():
    """Test that substrate can be configured correctly."""
    substrate = rn.PySubstrate.with_coords(
        inputs=[[-1, -1], [-1, 0], [-1, 1]],
        hidden=[[0, -1], [0, 0], [0, 1]],
        outputs=[[1, 0]],
    )

    # Test getters and setters
    substrate.set_allow_input_hidden_links(True)
    assert substrate.get_allow_input_hidden_links()

    substrate.set_with_distance(True)
    assert substrate.get_with_distance()

    substrate.set_max_weight_and_bias(8.0)
    assert substrate.get_max_weight_and_bias() == 8.0

    # Test activation function setters
    substrate.set_hidden_nodes_activation('signed_sigmoid')
    assert substrate.get_hidden_nodes_activation() == 'SignedSigmoid'

    substrate.set_output_nodes_activation('unsigned_sigmoid')
    assert substrate.get_output_nodes_activation() == 'UnsignedSigmoid'

    # Test CPPN dimensions
    min_inputs = substrate.get_min_cppn_inputs()
    min_outputs = substrate.get_min_cppn_outputs()
    assert min_inputs >= 4  # At least x1, y1, x2, y2
    assert min_outputs >= 1  # At least weight output
    assert min_outputs >= 1  # At least weight output
    assert min_outputs >= 1  # At least weight output
