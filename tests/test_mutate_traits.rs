mod common;
use common::seeded_rng;
use rusty_neat::genes::{ActivationFunction, Gene, LinkGene, NeuronGene, NeuronType, TraitValue};
use rusty_neat::genes::{TraitDetail, TraitParameters};
use rusty_neat::genome::Genome;
use rusty_neat::parameters::Parameters;

// Test for trait mutation and randomization.
//
// This test creates a minimal genome with a neuron, a link and a genome-level
// gene, each carrying a trait `t1`. It then configures `Parameters` to force
// mutation (mutation_prob = 1.0) and verifies that:
//  - `mutate_neuron_traits`, `mutate_link_traits` and `mutate_genome_traits`
//    report changes when run with a seeded RNG.
//  - `randomize_traits` executes without panic and assigns proper types/ranges.
#[test]
fn test_mutate_and_randomize_traits() {
    let mut params = Parameters::default();
    // define TraitParameters with mutation_prob == 1.0 for a test trait
    params.neuron_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Float {
                min: 0.0,
                max: 1.0,
                mut_power: 0.1,
                mut_replace_prob: 0.5,
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );
    params.link_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 5,
                mut_power: 1,
                mut_replace_prob: 0.5,
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );
    params.genome_trait_parameters.insert(
        "t1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 1,
                mut_power: 1,
                mut_replace_prob: 0.5,
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    let mut g = Genome::default();
    g.set_num_inputs(1);
    g.set_num_outputs(1);

    // add one hidden neuron with a float trait
    let mut ng = NeuronGene::new(
        2,
        NeuronType::Hidden,
        0,
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        ActivationFunction::SignedSigmoid,
    );
    ng.traits.insert("t1".to_string(), TraitValue::Float(0.5));
    g.push_neuron(ng);

    // add a link with trait
    let mut lg = LinkGene::new(1, 2, 1, 0.1, false);
    lg.traits.insert("t1".to_string(), TraitValue::Int(1));
    g.push_link(lg);

    // add genome-level gene (int trait)
    let mut gg = Gene::new();
    gg.traits.insert("t1".to_string(), TraitValue::Int(0));
    g.set_genome_gene(Some(gg));

    let mut rng = seeded_rng(123);

    // Mutation should flip or change traits because prob=1.0
    // Save old values and verify they changed after mutation
    let old_neuron_t1 = g.neuron_gene_at(0).traits.get("t1").cloned();
    let old_link_t1 = g.link_gene_at(0).traits.get("t1").cloned();
    let old_genome_t1 = g.genome_gene().and_then(|gg| gg.traits.get("t1").cloned());

    assert!(g.mutate_neuron_traits(&params, &mut rng));
    assert!(g.mutate_link_traits(&params, &mut rng));
    assert!(g.mutate_genome_traits(&params, &mut rng));

    // check that values actually changed
    let new_neuron_t1 = g.neuron_gene_at(0).traits.get("t1").cloned();
    let new_link_t1 = g.link_gene_at(0).traits.get("t1").cloned();
    let new_genome_t1 = g.genome_gene().and_then(|gg| gg.traits.get("t1").cloned());

    assert_ne!(
        old_neuron_t1, new_neuron_t1,
        "neuron trait t1 did not change"
    );
    assert_ne!(old_link_t1, new_link_t1, "link trait t1 did not change");
    assert_ne!(
        old_genome_t1, new_genome_t1,
        "genome trait t1 did not change"
    );

    // Randomize traits should not panic and should change values types appropriately
    // Save values before randomize and ensure they change and have expected types
    let pre_rand_neuron_t1 = g.neuron_gene_at(0).traits.get("t1").cloned();
    let pre_rand_link_t1 = g.link_gene_at(0).traits.get("t1").cloned();
    let _pre_rand_genome_t1 = g.genome_gene().and_then(|gg| gg.traits.get("t1").cloned());

    g.randomize_traits(&params, &mut rng);

    let post_rand_neuron_t1 = g.neuron_gene_at(0).traits.get("t1").cloned();
    let post_rand_link_t1 = g.link_gene_at(0).traits.get("t1").cloned();
    let post_rand_genome_t1 = g.genome_gene().and_then(|gg| gg.traits.get("t1").cloned());

    assert_ne!(
        pre_rand_neuron_t1, post_rand_neuron_t1,
        "neuron trait t1 did not change on randomize"
    );
    assert_ne!(
        pre_rand_link_t1, post_rand_link_t1,
        "link trait t1 did not change on randomize"
    );
    // genome-level trait may be randomized to the same value depending on
    // TraitParameters; don't require it to change deterministically here.

    // Types should remain appropriate (neuron: Float, link: Int, genome: Int)
    assert!(
        matches!(post_rand_neuron_t1, Some(TraitValue::Float(_))),
        "neuron trait type changed"
    );
    assert!(
        matches!(post_rand_link_t1, Some(TraitValue::Int(_))),
        "link trait type changed"
    );
    assert!(
        matches!(post_rand_genome_t1, Some(TraitValue::Int(_))),
        "genome trait type changed"
    );
}
