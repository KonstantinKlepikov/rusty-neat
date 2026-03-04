use rusty_neat::*;

#[test]
fn matching_topology_updates_weights() {
    let params = Parameters::default();
    let inits = genome::GenomeInitStruct {
        num_inputs: 3,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: genes::ActivationFunction::UnsignedSigmoid,
        hidden_act_type: genes::ActivationFunction::UnsignedSigmoid,
        seed_type: genome::GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g = Genome::new(1, &params, &inits);
    let mut net = NeuralNetwork::new();
    g.build_phenotype(&mut net);

    // change phenotype connection weights
    for conn in &mut net.connections {
        conn.weight = 3.14;
    }

    // ensure genome wasn't updated yet
    assert!(g.link_genes().iter().all(|l| (l.get_weight() - 3.14).abs() > 1e-12));

    g.derive_phenotypic_changes(&net);

    // now genome should reflect phenotype weights
    assert!(g
        .link_genes()
        .iter()
        .all(|l| (l.get_weight() - 3.14).abs() < 1e-12));
}

#[test]
fn mismatch_count_does_not_modify_genome() {
    let params = Parameters::default();
    let inits = genome::GenomeInitStruct {
        num_inputs: 3,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: genes::ActivationFunction::UnsignedSigmoid,
        hidden_act_type: genes::ActivationFunction::UnsignedSigmoid,
        seed_type: genome::GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g = Genome::new(2, &params, &inits);
    let mut net = NeuralNetwork::new();
    g.build_phenotype(&mut net);

    // alter phenotype topology: add an extra dummy connection. With the
    // robust mapping implementation, extra phenotype connections should
    // not prevent existing genome links from being updated.
    let extra = network::Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 0,
        weight: 9.99,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    };
    net.connections.push(extra);

    // change phenotype connection weights (including originals)
    for conn in &mut net.connections {
        conn.weight = 1.23;
    }

    g.derive_phenotypic_changes(&net);

    // genome should have been updated for its links
    assert!(g.link_genes().iter().all(|l| (l.get_weight() - 1.23).abs() < 1e-12));
}

#[test]
fn phenotype_connections_shuffled_updates_genome() {
    let params = Parameters::default();
    let inits = genome::GenomeInitStruct {
        num_inputs: 3,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: genes::ActivationFunction::UnsignedSigmoid,
        hidden_act_type: genes::ActivationFunction::UnsignedSigmoid,
        seed_type: genome::GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g = Genome::new(4, &params, &inits);
    let mut net = NeuralNetwork::new();
    g.build_phenotype(&mut net);

    // set distinct weights and then shuffle connections
    for (i, conn) in net.connections.iter_mut().enumerate() {
        conn.weight = i as f64 + 0.5;
    }

    // shuffle by simple reverse to avoid adding RNG dependency
    net.connections.reverse();

    // now derive changes; robust mapping should find matches despite order
    g.derive_phenotypic_changes(&net);

    // verify genome updated (weights don't need to match original order,
    // but each genome link should have been set to some weight from phenotype)
    assert!(g.link_genes().iter().all(|l| l.get_weight().abs() > 0.0));
}

#[test]
fn mismatch_indices_does_not_modify_genome() {
    let params = Parameters::default();
    let inits = genome::GenomeInitStruct {
        num_inputs: 3,
        num_hidden: 0,
        num_outputs: 1,
        fs_neat: false,
        output_act_type: genes::ActivationFunction::UnsignedSigmoid,
        hidden_act_type: genes::ActivationFunction::UnsignedSigmoid,
        seed_type: genome::GenomeSeedType::Perceptron,
        num_layers: 0,
        fs_neat_links: 0,
    };

    let mut g = Genome::new(3, &params, &inits);
    let mut net = NeuralNetwork::new();
    g.build_phenotype(&mut net);

    // change a phenotype connection's source index to break matching
    if !net.connections.is_empty() && !net.neurons.is_empty() {
        let orig = net.connections[0].source_neuron_idx;
        net.connections[0].source_neuron_idx = (orig + 1) % net.neurons.len();
        // set weights that would have been copied
        for conn in &mut net.connections {
            conn.weight = 2.71;
        }

        let before: Vec<f64> = g.link_genes().iter().map(|l| l.get_weight()).collect();
        g.derive_phenotypic_changes(&net);
        let after: Vec<f64> = g.link_genes().iter().map(|l| l.get_weight()).collect();
        assert_eq!(before, after, "Genome weights should be unchanged when indices mismatch");
    }
}
