use rand::SeedableRng;
use rand::rngs::SmallRng;
use rusty_neat::genes::ActivationFunction;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_tick_replaces_worst() {
    let params = Parameters::default();
    let inits = GenomeInitStruct {
        num_inputs: 2,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: ActivationFunction::SignedSigmoid,
        hidden_act_type: ActivationFunction::SignedSigmoid,
        seed_type: GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);
    let mut g3 = Genome::new(3, &params, &inits);

    g1.set_fitness(5.0);
    g2.set_fitness(4.0);
    g3.set_fitness(1.0);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1, g2, g3];
    // All in one species
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1, 2, 3],
        age: 0,
        best_fitness: 0.0,
    });

    let mut rng = SmallRng::seed_from_u64(42);

    let before_len = pop.genomes.len();
    let baby = pop.tick(&mut rng);
    assert!(baby.is_some());
    assert_eq!(pop.genomes.len(), before_len);
}
