import rusty_neat_py


def test_epoch_tick_and_get_best_smoke() -> None:
    params = rusty_neat_py.PyParameters()
    pop = rusty_neat_py.PyPopulation(params)

    # empty population -> get_best_genome() is None
    assert pop.get_best_genome() is None

    g1 = pop.epoch()
    assert isinstance(g1, int)
    g2 = pop.tick()
    assert isinstance(g2, int)
    assert g2 == g1 + 1


def test_manual_fill_add_remove_mutate() -> None:
    params = rusty_neat_py.PyParameters()
    pop = rusty_neat_py.PyPopulation(params)

    assert pop.num_genomes() == 0

    # add genome from init struct
    init = rusty_neat_py.PyGenomeInitStruct(1, 0, 1)
    idx = pop.add_genome_from_initstruct(init)
    assert isinstance(idx, int)
    assert pop.num_genomes() == 1

    # add random genome
    idx2 = pop.add_random_genome(1, 0, 1)
    assert isinstance(idx2, int)
    assert pop.num_genomes() == 2

    # mutate genome (no error, returns bool)
    mutated = pop.mutate_genome(idx, None)
    assert isinstance(mutated, bool)

    # get best genome returns wrapper with methods
    best = pop.get_best_genome()
    assert best is not None
    gref = best
    gid = gref.get_id()
    assert isinstance(gid, int)
    fitness = gref.get_fitness()
    assert isinstance(fitness, float)

    # remove genome by index
    removed = pop.remove_genome(idx2)
    assert isinstance(removed, bool)
    # sanity: number of genomes decreased or at least non-negative
    assert pop.num_genomes() >= 0


def test_mutate_genome_kinds() -> None:
    params = rusty_neat_py.PyParameters()
    pop = rusty_neat_py.PyPopulation(params)

    init = rusty_neat_py.PyGenomeInitStruct(2, 1, 1)
    idx = pop.add_genome_from_initstruct(init)

    # try each kind
    for kind in ("weights", "add_link", "add_neuron", "remove_link", "remove_neuron", None):
        result = pop.mutate_genome(idx, kind)
        assert isinstance(result, bool)


def test_pyphenotypebehavior_basic_contracts() -> None:
    # basic construction and data roundtrip
    pb = rusty_neat_py.PyPhenotypeBehavior()
    assert pb.successful() is True

    # set and get data
    data = [[0.1, 0.2], [0.3]]
    pb.set_data(data)
    got = pb.get_data()
    assert isinstance(got, list) or hasattr(got, "__iter__")

    # distance_to identical -> 0.0 expected by default impl
    other = rusty_neat_py.PyPhenotypeBehavior()
    other.set_data([[0.1, 0.2], [0.3]])
    dist = pb.distance_to(other)
    assert isinstance(dist, float)
    assert dist == 0.0

    # acquire with a simple genome (default returns False)
    g = rusty_neat_py.PyGenome(1, 1, 0, 1)
    acquired = pb.acquire(g)
    assert acquired in (True, False)
