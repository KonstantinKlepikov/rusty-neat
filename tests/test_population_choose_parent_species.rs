use rand::SeedableRng;
use rand::rngs::SmallRng;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_choose_parent_species_weighted_single_positive() {
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

    // Set adjusted fitness directly; only species 0 has positive weight
    g1.set_adj_fitness(10.0);
    g2.set_adj_fitness(0.0);
    g3.set_adj_fitness(0.0);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1, g2, g3];
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

    let mut rng = SmallRng::seed_from_u64(42);

    // Repeated calls should consistently pick species 0 when weights are [10,0,0]
    for _ in 0..20 {
        let picked = pop.choose_parent_species(&mut rng);
        assert_eq!(picked, Some(0));
    }
}

#[test]
fn test_choose_parent_species_fallback_uniform_on_nonpositive_total() {
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

    // All non-positive adjusted fitness -> should fall back to uniform random
    g1.set_adj_fitness(0.0);
    g2.set_adj_fitness(0.0);
    g3.set_adj_fitness(0.0);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1, g2, g3];
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

    let mut rng = SmallRng::seed_from_u64(12345);

    for _ in 0..50 {
        let picked = pop.choose_parent_species(&mut rng);
        assert!(picked.is_some());
        let idx = picked.unwrap();
        assert!(idx < pop.species.len());
    }
}
