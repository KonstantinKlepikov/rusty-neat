use rusty_neat::Genome;
use rusty_neat::genes::TraitValue;
use rusty_neat::genes::{ActivationFunction, LinkGene, NeuronGene, NeuronType};
// small helper module for tests

use rand::SeedableRng;
use rand::rngs::StdRng;

// Helper: approximate weight comparator with configurable tolerance
#[allow(dead_code)]
pub fn assert_weight_approx(actual: f64, expected: f64, tol: f64) {
    let d = (actual - expected).abs();
    assert!(
        d <= tol,
        "weight differs by {} which is > tol {}: got {} expected {}",
        d,
        tol,
        actual,
        expected
    );
}

// Helper: create a minimal CPPN-like Genome that outputs fixed values on its outputs
// It builds a small feed-forward CPPN where the bias input is connected to every
// CPPN output with the provided weights. Outputs use Linear activation so that
// output == bias * weight (bias input set to 1.0 in BuildHyperNEATPhenotype).
/// Create a seeded StdRng for deterministic tests
#[allow(dead_code)]
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[allow(dead_code)]
pub fn make_fixed_cppn(cppn_inputs: usize, cppn_outputs: usize, out_weights: &[f64]) -> Genome {
    assert!(cppn_outputs == out_weights.len());
    let mut g = Genome::default();
    g.set_num_inputs(cppn_inputs);
    g.set_num_outputs(cppn_outputs);

    // Assign neuron ids: inputs 1..cppn_inputs, outputs next
    let mut next_id: u64 = 1;
    for i in 0..cppn_inputs {
        let ntype = if i + 1 == cppn_inputs {
            NeuronType::Bias
        } else {
            NeuronType::Input
        };
        let ng = NeuronGene::new(
            next_id,
            ntype,
            0,
            0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            ActivationFunction::UnsignedSigmoid,
        );
        g.push_neuron(ng);
        next_id += 1;
    }
    // outputs
    for _ in 0..cppn_outputs {
        let ng = NeuronGene::new(
            next_id,
            NeuronType::Output,
            0,
            0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            ActivationFunction::Linear,
        );
        g.push_neuron(ng);
        next_id += 1;
    }

    // Connect bias (last input id = cppn_inputs) to every output with given weight
    let bias_id = cppn_inputs as u64; // ids started at 1
    let mut innov = 1u64;
    for (out_idx, &w) in out_weights.iter().enumerate() {
        let out_id = (cppn_inputs + out_idx) as u64 + 1; // since outputs come after inputs
        let lg = LinkGene::new(bias_id, out_id, innov, w, false);
        g.push_link(lg);
        innov += 1;
    }

    g
}

/// Create a small Genome populated with one hidden neuron, one link and
/// an optional genome-level gene. The caller can supply lists of trait
/// (name, value) pairs to set on neuron, link and genome_gene respectively.
#[allow(dead_code)]
pub fn make_simple_genome_with_traits(
    neuron_traits: &[(&str, TraitValue)],
    link_traits: &[(&str, TraitValue)],
    genome_traits: &[(&str, TraitValue)],
) -> Genome {
    let mut g = Genome::default();
    g.set_num_inputs(1);
    g.set_num_outputs(1);

    // neuron
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
    for (k, v) in neuron_traits {
        ng.traits.insert(k.to_string(), v.clone());
    }
    g.push_neuron(ng);

    // link
    let mut lg = LinkGene::new(1, 2, 1, 0.1, false);
    for (k, v) in link_traits {
        lg.traits.insert(k.to_string(), v.clone());
    }
    g.push_link(lg);

    if !genome_traits.is_empty() {
        let mut gg = rusty_neat::genes::Gene::new();
        for (k, v) in genome_traits {
            gg.traits.insert(k.to_string(), v.clone());
        }
        g.set_genome_gene(Some(gg));
    }

    g
}
