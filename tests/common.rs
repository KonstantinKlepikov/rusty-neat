#![allow(dead_code)]

use rusty_neat::Genome;
use rusty_neat::genes::{ActivationFunction, LinkGene, NeuronGene, NeuronType};

use rand::SeedableRng;
use rand::rngs::StdRng;

// Helper: approximate weight comparator with configurable tolerance
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
/// Default seed constant for tests
pub const DEFAULT_SEED: u64 = 42;

/// Create a seeded StdRng for deterministic tests
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Create a StdRng with the default seed
pub fn default_rng() -> StdRng {
    seeded_rng(DEFAULT_SEED)
}

pub fn make_fixed_cppn(cppn_inputs: usize, cppn_outputs: usize, out_weights: &[f64]) -> Genome {
    assert!(cppn_outputs == out_weights.len());
    let mut g = Genome::default();
    g.num_inputs = cppn_inputs;
    g.num_outputs = cppn_outputs;

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
        g.neuron_genes.push(ng);
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
        g.neuron_genes.push(ng);
        next_id += 1;
    }

    // Connect bias (last input id = cppn_inputs) to every output with given weight
    let bias_id = cppn_inputs as u64; // ids started at 1
    let mut innov = 1u64;
    for (out_idx, &w) in out_weights.iter().enumerate() {
        let out_id = (cppn_inputs + out_idx) as u64 + 1; // since outputs come after inputs
        let lg = LinkGene::new(bias_id, out_id, innov, w, false);
        g.link_genes.push(lg);
        innov += 1;
    }

    g
}
