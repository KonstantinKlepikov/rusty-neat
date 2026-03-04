use rusty_neat::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn mutate_add_link_prefers_bias_when_configured() {
    let mut g = Genome::default();
    // two inputs: one regular and one bias (bias is last input), one output
    g.set_num_inputs(2);
    g.set_num_outputs(1);
    // create neurons: id 1 input, id 2 bias, id 3 output
    let n1 = genes::NeuronGene::new(1, genes::NeuronType::Input, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, genes::ActivationFunction::UnsignedSigmoid);
    let n2 = genes::NeuronGene::new(2, genes::NeuronType::Bias, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, genes::ActivationFunction::UnsignedSigmoid);
    let n3 = genes::NeuronGene::new(3, genes::NeuronType::Output, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, genes::ActivationFunction::Linear);
    g.push_neuron(n1);
    g.push_neuron(n2);
    g.push_neuron(n3);

    // clear any links to start fresh
    // (default genome has none)

    let mut params = Parameters::default();
    params.mutate_add_link_from_bias_prob = 1.0; // always pick bias as source

    let mut innov_db = innovation::InnovationDatabase::new(100, 1000);
    let mut rng = StdRng::seed_from_u64(42);

    let ok = g.mutate_add_link(&mut innov_db, &params, &mut rng);
    assert!(ok, "mutate_add_link should succeed when bias-first and space available");

    // find newly added link and assert its source is the bias neuron id (2)
    let added = g.link_genes().last().expect("link added");
    assert_eq!(added.from_neuron_id, 2);
}

#[test]
fn mutate_add_link_sets_recurrent_when_requested() {
    let mut g = Genome::default();
    g.set_num_inputs(1);
    g.set_num_outputs(1);
    // neurons: id1 input, id2 output
    g.push_neuron(genes::NeuronGene::new(1, genes::NeuronType::Input, 0, 0, 0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::UnsignedSigmoid));
    g.push_neuron(genes::NeuronGene::new(2, genes::NeuronType::Output, 0, 0, 0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::Linear));

    let mut params = Parameters::default();
    params.recurrent_prob = 1.0; // always recurrent
    params.recurrent_loop_prob = 0.0; // avoid looped recurrent

    let mut innov_db = innovation::InnovationDatabase::new(200, 2000);
    let mut rng = StdRng::seed_from_u64(7);

    let ok = g.mutate_add_link(&mut innov_db, &params, &mut rng);
    assert!(ok, "mutate_add_link should succeed when recurrent requested");

    let added = g.link_genes().last().expect("link added");
    assert!(added.is_recurrent(), "new link should be marked recurrent");
}

#[test]
fn mutate_add_neuron_respects_split_looped_recurrent_flag() {
    let mut g = Genome::default();
    // single hidden neuron with a looped recurrent link
    g.set_num_inputs(0);
    g.set_num_outputs(0);
    g.push_neuron(genes::NeuronGene::new(1, genes::NeuronType::Hidden, 0,0,0.5,1.0,0.0,1.0,0.0, genes::ActivationFunction::SignedSigmoid));
    // add a looped recurrent link (from 1 to 1)
    g.push_link(genes::LinkGene::new(1, 1, 1, 0.5, true));

    let mut innov_db = innovation::InnovationDatabase::new(10, 10);
    let mut params = Parameters::default();
    params.split_looped_recurrent = false;
    let mut rng = StdRng::seed_from_u64(13);

    // when split_looped_recurrent is false, splitting should fail
    let ok = g.mutate_add_neuron(&mut innov_db, &params, &mut rng);
    assert!(!ok, "mutate_add_neuron should not split looped recurrent links when disabled");

    // enable splitting of looped recurrent links and try again
    let mut g2 = g.clone();
    let mut innov_db2 = innovation::InnovationDatabase::new(10, 20);
    params.split_looped_recurrent = true;
    let mut rng2 = StdRng::seed_from_u64(17);
    let ok2 = g2.mutate_add_neuron(&mut innov_db2, &params, &mut rng2);
    assert!(ok2, "mutate_add_neuron should split looped recurrent links when enabled");

    // after successful split, there should be an extra neuron and net +1 link
    assert!(g2.neuron_genes().len() > g.neuron_genes().len());
    assert!(g2.link_genes().len() >= 2);
}

    #[test]
    fn mutate_add_link_fails_when_link_tries_zero() {
        let mut g = Genome::default();
        g.set_num_inputs(1);
        g.set_num_outputs(1);
        g.push_neuron(genes::NeuronGene::new(1, genes::NeuronType::Input, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::UnsignedSigmoid));
        g.push_neuron(genes::NeuronGene::new(2, genes::NeuronType::Output, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::Linear));

        let mut params = Parameters::default();
        params.link_tries = 0; // immediate exhaustion

        let mut innov_db = innovation::InnovationDatabase::new(1, 1);
        let mut rng = StdRng::seed_from_u64(123);

        let ok = g.mutate_add_link(&mut innov_db, &params, &mut rng);
        assert!(!ok, "mutate_add_link should fail immediately when link_tries is zero");
    }

    #[test]
    fn mutate_add_link_reuses_existing_innovation_id() {
        let mut g = Genome::default();
        // Create only bias and output neurons so the bias->output pair is the obvious choice
        g.push_neuron(genes::NeuronGene::new(2, genes::NeuronType::Bias, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::UnsignedSigmoid));
        g.push_neuron(genes::NeuronGene::new(3, genes::NeuronType::Output, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::Linear));

        let mut innov_db = innovation::InnovationDatabase::new(50, 500);
        // pre-add innovation for bias(2)->output(3)
        let pre_innov = innov_db.add_link_innovation(2, 3);

        let mut params = Parameters::default();
        params.mutate_add_link_from_bias_prob = 1.0; // force bias source
        params.allow_loops = false; // avoid creating looped link (from==to)

        let mut rng = StdRng::seed_from_u64(42);
        let ok = g.mutate_add_link(&mut innov_db, &params, &mut rng);
        assert!(ok, "mutate_add_link should succeed");

        let added = g.link_genes().last().expect("link added");
        assert_eq!(added.innovation_id, pre_innov, "mutate_add_link should reuse existing innovation id");
    }

    #[test]
    fn mutate_add_link_exhausts_tries_when_no_valid_targets() {
        // Create genome with two input neurons only — no valid target (targets cannot be Input)
        let mut g = Genome::default();
        g.push_neuron(genes::NeuronGene::new(1, genes::NeuronType::Input, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::UnsignedSigmoid));
        g.push_neuron(genes::NeuronGene::new(2, genes::NeuronType::Input, 0,0,0.0,1.0,0.0,1.0,0.0, genes::ActivationFunction::UnsignedSigmoid));

        let mut params = Parameters::default();
        params.link_tries = 4; // a few attempts

        let mut innov_db = innovation::InnovationDatabase::new(5, 5);
        let mut rng = StdRng::seed_from_u64(2026);

        // All possible picks lead to target being an Input, so mutate_add_link must exhaust tries and fail
        let ok = g.mutate_add_link(&mut innov_db, &params, &mut rng);
        assert!(!ok, "mutate_add_link should fail after exhausting attempts when no valid target exists");
    }
