use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_remove_worst_basic() {
    let params = Parameters::default();
    let inits = GenomeInitStruct {
        num_inputs: 2,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
        hidden_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
        seed_type: GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);
    let mut g3 = Genome::new(3, &params, &inits);

    // set adjusted fitnesses; genome 2 is worst
    g1.set_adj_fitness(5.0);
    g2.set_adj_fitness(2.0);
    g3.set_adj_fitness(7.0);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone(), g3.clone()];
    // put each genome in its own species
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1],
        age: 0,
        best_fitness: 0.0,
    });
    pop.species.push(rusty_neat::species::Species {
        id: 2,
        members: vec![2],
        age: 0,
        best_fitness: 0.0,
    });
    pop.species.push(rusty_neat::species::Species {
        id: 3,
        members: vec![3],
        age: 0,
        best_fitness: 0.0,
    });

    let before_len = pop.genomes.len();
    let removed = pop.remove_worst_individual();
    assert!(removed.is_some());
    let rem = removed.unwrap();
    assert_eq!(rem.get_id(), 2);
    assert_eq!(pop.genomes.len(), before_len - 1);
    // ensure species with id 2 was removed (its single member was deleted)
    assert!(pop.species.iter().all(|s| !s.members.contains(&2)));
}

#[test]
fn test_remove_worst_cleans_empty_species() {
    let params = Parameters::default();
    let inits = GenomeInitStruct {
        num_inputs: 2,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
        hidden_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
        seed_type: GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);

    g1.set_adj_fitness(1.0);
    g2.set_adj_fitness(0.5);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone()];
    // species 1 contains both genomes, species 2 contains only genome 2 (worst)
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1, 2],
        age: 0,
        best_fitness: 0.0,
    });
    pop.species.push(rusty_neat::species::Species {
        id: 2,
        members: vec![2],
        age: 0,
        best_fitness: 0.0,
    });

    let removed = pop.remove_worst_individual();
    assert!(removed.is_some());
    let rem = removed.unwrap();
    assert_eq!(rem.get_id(), 2);
    // no species should contain id 2
    assert!(pop.species.iter().all(|s| !s.members.contains(&2)));
}

#[test]
fn test_remove_worst_none_on_empty_population() {
    let params = Parameters::default();
    let mut pop = Population::new(params);
    let removed = pop.remove_worst_individual();
    assert!(removed.is_none());
}
