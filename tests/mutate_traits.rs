use rusty_neat::genome::Genome;
use rusty_neat::genes::{NeuronGene, NeuronType, LinkGene, TraitValue, Gene, ActivationFunction};
use rusty_neat::parameters::Parameters;
use rusty_neat::genes::{TraitParameters, TraitDetail};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn test_mutate_and_randomize_traits() {
    let mut params = Parameters::default();
    // define TraitParameters with mutation_prob == 1.0 for a test trait
    params.neuron_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Float { min: 0.0, max: 1.0, mut_power: 0.1, mut_replace_prob: 0.5 },
            dep_key: None,
            dep_values: Vec::new(),
        }
    );
    params.link_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int { min: 0, max: 5, mut_power: 1, mut_replace_prob: 0.5 },
            dep_key: None,
            dep_values: Vec::new(),
        }
    );
    params.genome_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int { min: 0, max: 1, mut_power: 1, mut_replace_prob: 0.5 },
            dep_key: None,
            dep_values: Vec::new(),
        }
    );

    let mut g = Genome::default();
    g.num_inputs = 1;
    g.num_outputs = 1;

    // add one hidden neuron with a float trait
    let mut ng = NeuronGene::new(2, NeuronType::Hidden, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, ActivationFunction::SignedSigmoid);
    ng.traits.insert("t1".to_string(), TraitValue::Float(0.5));
    g.neuron_genes.push(ng);

    // add a link with trait
    let mut lg = LinkGene::new(1, 2, 1, 0.1, false);
    lg.traits.insert("t1".to_string(), TraitValue::Int(1));
    g.link_genes.push(lg);

    // add genome-level gene (int trait)
    let mut gg = Gene::new();
    gg.traits.insert("t1".to_string(), TraitValue::Int(0));
    g.genome_gene = Some(gg);

    let mut rng = StdRng::seed_from_u64(123);

    // Mutation should flip or change traits because prob=1.0
    assert!(g.mutate_neuron_traits(&params, &mut rng));
    assert!(g.mutate_link_traits(&params, &mut rng));
    assert!(g.mutate_genome_traits(&params, &mut rng));

    // Randomize traits should not panic and should change values types appropriately
    g.randomize_traits(&mut rng);
}
