use rusty_neat::genes::ActivationFunction;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

use std::env;
use std::fs;

#[test]
fn test_population_save_load_roundtrip() {
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

    // make slight difference so equality is meaningful
    g1.set_fitness(3.14);
    g2.set_fitness(2.71);

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![g1.clone(), g2.clone()];

    // temp file path
    let mut path = env::temp_dir();
    path.push(format!("rusty_neat_pop_rt_{}.txt", std::process::id()));
    let p = path.to_str().unwrap();

    // save and load
    pop.save(p).expect("save should succeed");
    let loaded = Population::from_file(p).expect("load should succeed");

    // cleanup file
    let _ = fs::remove_file(p);

    // basic checks: number of genomes and key fields
    assert_eq!(loaded.genomes.len(), pop.genomes.len());
    for (orig, ld) in pop.genomes.iter().zip(loaded.genomes.iter()) {
        assert_eq!(orig.get_id(), ld.get_id());
        assert!((orig.get_fitness() - ld.get_fitness()).abs() < 1e-12);
        assert_eq!(orig.num_inputs(), ld.num_inputs());
        assert_eq!(orig.num_outputs(), ld.num_outputs());
        assert_eq!(orig.neuron_genes().len(), ld.neuron_genes().len());
        assert_eq!(orig.link_genes().len(), ld.link_genes().len());
        // compare link endpoints and innovation ids
        for (l1, l2) in orig.link_genes().iter().zip(ld.link_genes().iter()) {
            assert_eq!(l1.from_neuron_id, l2.from_neuron_id);
            assert_eq!(l1.to_neuron_id, l2.to_neuron_id);
            assert_eq!(l1.innovation_id, l2.innovation_id);
            assert!((l1.weight - l2.weight).abs() < 1e-12);
        }
    }

    // basic round-trip equality of genomes verified above; innovation DB is private
}
