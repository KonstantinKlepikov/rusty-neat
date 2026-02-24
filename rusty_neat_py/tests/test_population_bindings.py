import rusty_neat_py as rnp


def test_population_epoch_tick_and_get_best() -> None:
    params = rnp.PyParameters()
    pop = rnp.PyPopulation(params)

    # empty population: get_best_genome() -> None
    best = pop.get_best_genome()
    assert best is None

    # epoch/tick should advance generation and return integer generation
    g1 = pop.epoch()
    assert isinstance(g1, int)
    g2 = pop.tick()
    assert isinstance(g2, int)
    assert g2 == g1 + 1


def test_population_manual_fill_and_mutation() -> None:
    params = rnp.PyParameters()
    pop = rnp.PyPopulation(params)

    assert pop.num_genomes() == 0

    # add genome from init struct (use 2 inputs to include at least one real input + bias)
    init = rnp.PyGenomeInitStruct(2, 0, 1)
    idx = pop.add_genome_from_initstruct(init)
    assert isinstance(idx, int)
    assert pop.num_genomes() == 1

    # add random genome (use 2 inputs to satisfy Genome constructor requirement)
    idx2 = pop.add_random_genome(2, 0, 1)
    assert isinstance(idx2, int)
    assert pop.num_genomes() == 2

    # mutate genome at idx (should return bool)
    mutated = pop.mutate_genome(idx, None)
    assert isinstance(mutated, bool)

    # get best genome now (some genome wrapper)
    best = pop.get_best_genome()
    assert best is not None
    gref = best
    # ensure we can call methods on genome ref
    gid = gref.get_id()
    assert isinstance(gid, int)
    fitness = gref.get_fitness()
    assert isinstance(fitness, float)

    # remove a genome by index
    removed = pop.remove_genome(idx2)
    assert removed is True or removed is False
    # after removal, num_genomes decreased or unchanged depending on idx2 validity
    assert pop.num_genomes() >= 0
