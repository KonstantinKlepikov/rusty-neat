use rusty_neat::*;
use rand::rngs::ThreadRng;

#[test]
fn randomize_and_mutate_weights() {
    let mut g = Genome::default();
    // add two neurons and one link
    g.neuron_genes.push(crate::genes::NeuronGene::new(1, crate::genes::NeuronType::Input, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, crate::genes::ActivationFunction::Linear));
    g.neuron_genes.push(crate::genes::NeuronGene::new(2, crate::genes::NeuronType::Output, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, crate::genes::ActivationFunction::Linear));
    g.link_genes.push(crate::genes::LinkGene::new(1, 2, 1, 0.5, false));

    let mut params = Parameters::default();
    params.min_weight = -2.0;
    params.max_weight = 2.0;
    let mut rng: ThreadRng = rand::rngs::ThreadRng::default();

    // randomize
    g.randomize_link_weights(&params, &mut rng);
    let w = g.link_genes[0].weight;
    assert!(w >= params.min_weight && w <= params.max_weight, "randomize produced out-of-range weight");

    // set deterministic mutation settings
    params.mutate_weights_severe_prob = 0.0; // non-severe
    params.weight_mutation_rate = 1.0; // mutate every link
    params.weight_replacement_rate = 0.0; // force perturbation

    let prev_w = g.link_genes[0].weight;
    let changed = g.mutate_link_weights(&params, &mut rng);
    assert!(changed, "mutate_link_weights reported no change");
    let w2 = g.link_genes[0].weight;
    assert!(w2 != prev_w, "weight did not change on mutation");
}
