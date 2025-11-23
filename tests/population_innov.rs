use rand::rngs::ThreadRng;
use rusty_neat::*;

#[test]
fn innovations_are_shared_between_genomes() {
    let params = Parameters::default();

    // create minimal genomes: input id=1, output id=2, one link 1->2 with innov 1
    let mut g1 = Genome::default();
    g1.neuron_genes.push(crate::genes::NeuronGene::new(
        1,
        crate::genes::NeuronType::Input,
        0,
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        crate::genes::ActivationFunction::Linear,
    ));
    g1.neuron_genes.push(crate::genes::NeuronGene::new(
        2,
        crate::genes::NeuronType::Output,
        0,
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        crate::genes::ActivationFunction::Linear,
    ));
    g1.link_genes
        .push(crate::genes::LinkGene::new(1, 2, 1, 1.0, false));

    let g2 = g1.clone();

    // Create population from genomes so innovation DB starts after existing ids
    let mut pop = Population::from_genomes(vec![g1, g2], params);

    let mut rng: ThreadRng = rand::rngs::ThreadRng::default();

    // Mutate both genomes by splitting the single link; since the DB is shared,
    // the second mutation should reuse the neuron innovation created by the first.
    let ok1 = pop.mutate_add_neuron_for(0, &mut rng);
    assert!(ok1, "first mutation failed");
    let ok2 = pop.mutate_add_neuron_for(1, &mut rng);
    assert!(ok2, "second mutation failed");

    // find the new neuron ids in each genome (they should share the same id)
    let g0 = &pop.genomes[0];
    let g1 = &pop.genomes[1];

    let ids0: Vec<u64> = g0.neuron_genes.iter().map(|n| n.id).collect();
    let ids1: Vec<u64> = g1.neuron_genes.iter().map(|n| n.id).collect();

    // find neuron ids that are not 1 or 2
    let new0: Vec<u64> = ids0.into_iter().filter(|&id| id != 1 && id != 2).collect();
    let new1: Vec<u64> = ids1.into_iter().filter(|&id| id != 1 && id != 2).collect();

    assert!(!new0.is_empty(), "no new neuron in genome 0");
    assert!(!new1.is_empty(), "no new neuron in genome 1");
    assert_eq!(
        new0[0], new1[0],
        "innovations were not shared between genomes"
    );
}
