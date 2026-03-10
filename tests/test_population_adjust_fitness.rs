use rusty_neat::genes::ActivationFunction;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;
use rusty_neat::species::Species;

#[test]
fn test_adjust_fitness_sharing_basic() {
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

    // Create three genomes with known fitness values
    let mut g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);
    let mut g3 = Genome::new(3, &params, &inits);

    g1.set_fitness(1.0);
    g2.set_fitness(2.0);
    g3.set_fitness(-5.0); // should be floored

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1, g2, g3];

    // make a single species containing all genomes
    let s = Species {
        id: 1,
        members: vec![1, 2, 3],
        age: 0,
        best_fitness: 0.0,
    };
    pop.species = vec![s];

    pop.adjust_fitness();

    // After sharing: adj = floored_fitness / 3
    let adj1 = pop.genomes[0].get_adj_fitness();
    let adj2 = pop.genomes[1].get_adj_fitness();
    let adj3 = pop.genomes[2].get_adj_fitness();

    let eps = 1e-12;
    assert!((adj1 - (1.0 / 3.0)).abs() < eps);
    assert!((adj2 - (2.0 / 3.0)).abs() < eps);
    // floored value is 1e-10
    assert!((adj3 - (0.0000000001_f64 / 3.0)).abs() < 1e-20);
}
