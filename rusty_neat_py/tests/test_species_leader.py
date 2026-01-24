import rusty_neat_py as rnt


def test_species_get_leader():
    params = rnt.PyParameters()
    pop = rnt.PyPopulation(params)

    # add three genomes and set fitnesses
    _ = pop.add_random_genome(1, 0, 1)
    _ = pop.add_random_genome(1, 0, 1)
    _ = pop.add_random_genome(1, 0, 1)

    genomes = pop.list_genomes()
    g0 = genomes[0]
    g1 = genomes[1]
    g2 = genomes[2]
    g0.set_fitness(1.0)
    g1.set_fitness(5.0)
    g2.set_fitness(3.0)

    # create a species containing these genomes (by indices)
    sp_idx = pop.add_species([0, 1, 2])
    assert isinstance(sp_idx, int)

    # fetch species wrapper and leader
    sref = pop.get_species(sp_idx)
    assert sref is not None
    leader = sref.get_leader()
    assert leader is not None

    # leader should be the genome with fitness 5.0
    assert abs(leader.get_fitness() - 5.0) < 1e-12
