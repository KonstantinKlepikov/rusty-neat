use rusty_neat::genes::{ActivationFunction, NeuronGene, NeuronType};
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_reassign_species_adds_to_existing() {
    let params = Parameters::default();
    // keep default compat_treshold (3.0) so identical genomes match
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

    let g1 = Genome::new(1, &params, &inits);
    let g2 = Genome::new(2, &params, &inits);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone()];

    // initial species contains only genome 1
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1],
        age: 0,
        best_fitness: 0.0,
    });

    // assign genome at index 1 (id 2) to matching species
    pop.reassign_species(1);

    // species[0] should now contain id 2 as well
    assert!(pop.species[0].members.contains(&2));
}

#[test]
fn test_reassign_species_moves_and_cleans_old_species() {
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

    let g1 = Genome::new(1, &params, &inits);
    let g2 = Genome::new(2, &params, &inits);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone()];

    // genome 1 in species A, genome 2 in its own species B
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

    // reassign genome index 1 (id 2) to match species 1
    pop.reassign_species(1);

    // species containing id 2 should no longer exist
    assert!(
        pop.species
            .iter()
            .all(|s| !s.members.contains(&2) || s.id == 1)
    );
    // species 1 should contain both ids
    assert!(
        pop.species
            .iter()
            .any(|s| s.id == 1 && s.members.contains(&2))
    );
}

#[test]
fn test_reassign_species_creates_new_when_no_match() {
    // Use tight compat threshold so differing genome sizes won't match
    let mut params = Parameters::default();
    params.compat_treshold = 0.0; // only exact equal sizes will match

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

    let g1 = Genome::new(1, &params, &inits);
    let mut g2 = Genome::new(2, &params, &inits);

    // make g2 differ by adding an extra neuron
    let extra = NeuronGene::new(
        999,
        NeuronType::Hidden,
        0,
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        ActivationFunction::Linear,
    );
    g2.push_neuron(extra);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone()];

    // only initial species for genome 1
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1],
        age: 0,
        best_fitness: 0.0,
    });

    // reassign genome index 1 (id 2) - should create a new species
    pop.reassign_species(1);

    // expect two species: original and new one for genome 2
    assert!(pop.species.len() >= 2);
    assert!(pop.species.iter().any(|s| s.members.contains(&2)));
}
