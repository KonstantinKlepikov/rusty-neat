use rusty_neat::*;

// Two identical genomes should be placed in the same species with default parameters
#[test]
fn speciate_combines_identical_genomes() {
    let params = Parameters::default();

    // create minimal genome: input id=1, output id=2, one link 1->2
    let mut g1 = Genome::default();
    g1.push_neuron(crate::genes::NeuronGene::new(
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
    g1.push_neuron(crate::genes::NeuronGene::new(
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
    g1.push_link(crate::genes::LinkGene::new(1, 2, 1, 1.0, false));

    let g2 = g1.clone();

    let mut pop = Population::from_genomes(vec![g1, g2], params);
    pop.speciate();

    // identical genomes -> single species
    assert_eq!(pop.species.len(), 1);
    let s = &pop.species[0];
    assert!(s.members.len() >= 2, "species should contain both genomes");
}

// When compatibility threshold is low, small structural differences should split species
#[test]
fn speciate_separates_when_threshold_low() {
    let mut params = Parameters::default();
    params.compat_treshold = 0.5; // make threshold strict

    // base genome
    let mut g1 = Genome::default();
    g1.push_neuron(crate::genes::NeuronGene::new(
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
    g1.push_neuron(crate::genes::NeuronGene::new(
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
    g1.push_link(crate::genes::LinkGene::new(1, 2, 1, 1.0, false));

    // genome with one extra hidden neuron (structural difference = 1)
    let mut g2 = g1.clone();
    g2.push_neuron(crate::genes::NeuronGene::new(
        3,
        crate::genes::NeuronType::Hidden,
        0,
        0,
        0.5,
        1.0,
        0.0,
        1.0,
        0.0,
        crate::genes::ActivationFunction::Linear,
    ));

    let mut pop = Population::from_genomes(vec![g1, g2], params);
    pop.speciate();

    // with strict threshold, they should be in separate species
    assert!(
        pop.species.len() >= 2,
        "genomes should be separated into different species"
    );
}
