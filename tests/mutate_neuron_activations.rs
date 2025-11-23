use rand::SeedableRng;
use rand::rngs::StdRng;
use rusty_neat::genes::{ActivationFunction, NeuronGene, NeuronType};
use rusty_neat::genome::Genome;
use rusty_neat::parameters::Parameters;

#[test]
fn test_mutate_neuron_activations_a_and_b_and_type() {
    let mut params = Parameters::default();
    // make activation function selection deterministic and favor UnsignedSigmoid
    params.activation_function_probs = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    params.activation_a_mutation_max_power = 0.5;
    params.activation_b_mutation_max_power = 0.5;
    params.min_activation_a = 0.1;
    params.max_activation_a = 10.0;
    params.min_activation_b = -10.0;
    params.max_activation_b = 10.0;

    // Construct a genome with 1 input and 2 hidden neurons
    let mut g = Genome::default();
    g.num_inputs = 1;
    g.num_outputs = 1;

    // push one input neuron
    let input = NeuronGene::new(
        1,
        NeuronType::Input,
        0,
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        ActivationFunction::Linear,
    );
    g.neuron_genes.push(input);

    // two hidden neurons
    let h1 = NeuronGene::new(
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
    let h2 = NeuronGene::new(
        3,
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

    let orig_a_h1 = h1.a;
    let orig_b_h1 = h1.b;

    g.neuron_genes.push(h1);
    g.neuron_genes.push(h2);

    let mut rng = StdRng::seed_from_u64(42);

    let changed_a = g.mutate_neuron_activations_a(&params, &mut rng);
    assert!(changed_a, "mutate_neuron_activations_a should return true");

    // ensure at least one hidden neuron's a changed from original
    let h1_after = g.neuron_genes.iter().find(|n| n.id == 2).unwrap();
    assert!(
        (h1_after.a - orig_a_h1).abs() > 0.0,
        "A parameter should have been perturbed"
    );

    let changed_b = g.mutate_neuron_activations_b(&params, &mut rng);
    assert!(changed_b, "mutate_neuron_activations_b should return true");

    let h1_after_b = g.neuron_genes.iter().find(|n| n.id == 2).unwrap();
    assert!(
        (h1_after_b.b - orig_b_h1).abs() > 0.0,
        "B parameter should have been perturbed"
    );

    // change activation type for a random hidden neuron
    let changed_type = g.mutate_neuron_activation_type(&params, &mut rng);
    assert!(
        changed_type,
        "mutate_neuron_activation_type should indicate a change when selecting a new type"
    );
}
