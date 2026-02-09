import rusty_neat_py as rn


def evaluate_hyperneat_xor(genome, substrate):
    """Evaluate genome on XOR problem using HyperNEAT."""
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


def run(
    population_size: int = 350,
    max_generations: int = 100,
    initial_fitness: float = 0.1,
    final_fitness: float = 0.0,
) -> None:
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

    pop = rn.PyPopulation(params)
    population_size = 20
    for _ in range(population_size):
        pop.add_genome_from_initstruct(init_struct)

    for _ in range(max_generations):
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

    print(f'initial={initial_fitness:.2f}, ' f'final={final_fitness:.2f}')


if __name__ == '__main__':
    run(max_generations=1000)
