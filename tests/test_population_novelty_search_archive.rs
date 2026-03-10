use rand::SeedableRng;
use rand::rngs::SmallRng;
use rusty_neat::genes::ActivationFunction;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType, PhenotypeBehavior};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_novelty_search_tick_adds_to_archive() {
    let mut params = Parameters::default();
    // make Pmin small so sparseness easily exceeds it
    params.novelty_search_p_min = 0.1;

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

    // Create three genomes
    let mut g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);
    let mut g3 = Genome::new(3, &params, &inits);

    // Assign phenotype behaviors: g1 and g2 similar, g3 unique
    let mut pb_common = PhenotypeBehavior::new();
    pb_common.m_data = vec![vec![0.0]];
    g1.set_phenotype_behavior(Some(pb_common.clone()));
    g2.set_phenotype_behavior(Some(pb_common.clone()));

    let mut pb_unique = PhenotypeBehavior::new();
    pb_unique.m_data = vec![vec![10.0]];
    g3.set_phenotype_behavior(Some(pb_unique));

    // Fitness such that g3 is worst
    g1.set_fitness(5.0);
    g2.set_fitness(4.0);
    g3.set_fitness(1.0);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1, g2, g3];
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1, 2, 3],
        age: 0,
        best_fitness: 0.0,
    });

    // ensure adjusted fitness is set so remove_worst_individual finds g3
    pop.adjust_fitness();

    let mut rng = SmallRng::seed_from_u64(777);

    let before = pop.behavior_archive_len();
    let baby = pop.novelty_search_tick(&mut rng);
    assert!(baby.is_some());
    assert_eq!(pop.behavior_archive_len(), before + 1);
}
