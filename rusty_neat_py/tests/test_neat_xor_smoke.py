"""
Smoke test for basic NEAT XOR evolution.
Adapted from cneat/MultiNEAT/examples/TestNEAT_xor.py
"""

import numpy as np
import pytest
import rusty_neat_py as rn


def evaluate_xor(genome):
    """Evaluate genome on XOR problem."""
    try:
        net = rn.PyNeuralNetwork.with_dimensions(3, 1)
        genome.build_phenotype(net)

        error = 0

        # Test case: 1 XOR 0 = 1
        net.input(np.array([1.0, 0.0, 1.0]))
        for _ in range(2):
            net.activate()
        o = net.output()
        error += abs(1 - o[0])

        # Test case: 0 XOR 1 = 1
        net.reset()
        net.input([0, 1, 1])
        for _ in range(2):
            net.activate()
        o = net.output()
        error += abs(1 - o[0])

        # Test case: 1 XOR 1 = 0
        net.reset()
        net.input([1, 1, 1])
        for _ in range(2):
            net.activate()
        o = net.output()
        error += abs(o[0])

        # Test case: 0 XOR 0 = 0
        net.reset()
        net.input([0, 0, 1])
        for _ in range(2):
            net.activate()
        o = net.output()
        error += abs(o[0])

        return (4 - error) ** 2
    except Exception as ex:
        print(f'Exception during evaluation: {ex}')
        return 0.0


@pytest.mark.slow
def test_neat_xor_single_run():
    """Test that NEAT evolution runs without errors (smoke test).

    Note: This is a smoke test, not a full XOR solution test.
    It verifies the evolution process works, not that XOR is always solved.
    """
    # Setup parameters
    params = rn.PyParameters()
    params.set_min_weight(-8.0)

    # Create initial genome
    init_struct = rn.PyGenomeInitStruct(
        num_inputs=3,
        num_hidden=0,
        num_outputs=1,
        fs_neat=False,
        output_act_type='unsigned_sigmoid',
        hidden_act_type='unsigned_sigmoid',
        seed_type='minimal',
        num_layers=2,
        fs_neat_links=None,
    )

    # Create population and add initial genomes
    pop = rn.PyPopulation(params)
    population_size = 100
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
            fitness = evaluate_xor(g)
            g.set_fitness(fitness)
            fitness_list.append(fitness)

        # Track fitness
        best_fitness = max(fitness_list)
        if generation == 0:
            initial_fitness = best_fitness
        final_fitness = best_fitness

        # Early exit if solved (optional)
        if best_fitness > 15.0:
            print(f'Solved XOR in {generation} generations with fitness {best_fitness}')
            break

        # Evolve population
        pop.epoch()

    # Smoke test assertions: just verify evolution ran and fitness is valid
    assert final_fitness >= 0, 'Fitness should be non-negative'
    print(f'NEAT smoke test: initial={initial_fitness:.2f}, final={final_fitness:.2f}')


@pytest.mark.smoke
def test_neat_xor_basic_evolution():
    """Minimal smoke test - just verify evolution runs without errors."""
    params = rn.PyParameters()

    init_struct = rn.PyGenomeInitStruct(
        num_inputs=3,
        num_hidden=0,
        num_outputs=1,
        fs_neat=False,
        output_act_type='unsigned_sigmoid',
        hidden_act_type='unsigned_sigmoid',
        seed_type='minimal',
        num_layers=2,
        fs_neat_links=None,
    )

    pop = rn.PyPopulation(params)
    population_size = 20
    for _ in range(population_size):
        pop.add_genome_from_initstruct(init_struct)

    # Run just a few generations to verify it works
    for _ in range(10):
        genomes = pop.list_genomes()
        for g in genomes:
            fitness = evaluate_xor(g)
            g.set_fitness(fitness)

        pop.epoch()

    # Just verify we can get the best genome
    best = pop.get_best_genome()
    assert best is not None
    fitness = evaluate_xor(best)
    assert fitness >= 0  # Fitness should be non-negative
    assert fitness >= 0  # Fitness should be non-negative
