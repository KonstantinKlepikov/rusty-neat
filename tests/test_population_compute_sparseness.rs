use rusty_neat::genes::ActivationFunction;
use rusty_neat::genome::{Genome, GenomeInitStruct, GenomeSeedType, PhenotypeBehavior};
use rusty_neat::parameters::Parameters;
use rusty_neat::population::Population;

#[test]
fn test_compute_sparseness_basic() {
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

    // Genome A with phenotype [[1.0]]
    let mut a = Genome::new(1, &params, &inits);
    let mut pb_a = PhenotypeBehavior::new();
    pb_a.m_data = vec![vec![1.0]];
    a.set_phenotype_behavior(Some(pb_a));

    // Genome B with phenotype [[0.0]]
    let mut b = Genome::new(2, &params, &inits);
    let mut pb_b = PhenotypeBehavior::new();
    pb_b.m_data = vec![vec![0.0]];
    b.set_phenotype_behavior(Some(pb_b));

    let mut pop = Population::new(params.clone());
    pop.genomes = vec![a.clone(), b.clone()];
    pop.species.push(rusty_neat::species::Species {
        id: 1,
        members: vec![1, 2],
        age: 0,
        best_fitness: 0.0,
    });

    // Distance between A and B should be 1.0; sparseness for A should be 1.0
    let spar_a = pop.compute_sparseness(&pop.genomes[0]);
    assert!((spar_a - 1.0).abs() < 1e-9);

    // Now add an archive entry [[4.0]] and verify sparseness considers it
    let mut arch_pb = PhenotypeBehavior::new();
    arch_pb.m_data = vec![vec![4.0]];
    pop.add_behavior_archive(arch_pb);

    // If A has value 2.0 and B is 0.0 and archive is 4.0, sparseness should average neighbors (2.0)
    let mut a2 = Genome::new(3, &params, &inits);
    let mut pb_a2 = PhenotypeBehavior::new();
    pb_a2.m_data = vec![vec![2.0]];
    a2.set_phenotype_behavior(Some(pb_a2));
    pop.genomes.push(a2.clone());
    pop.species[0].members.push(3);

    let spar_a2 = pop.compute_sparseness(&pop.genomes.last().unwrap());
    // distances: to 1.0 ->1.0, to 0.0 ->2.0, archive 4.0 ->2.0 -> sorted [0,1,2,2]
    // k = min(15, 3) = 3 -> average of [1,2,2] = 5/3
    assert!((spar_a2 - (5.0 / 3.0)).abs() < 1e-9);
}
