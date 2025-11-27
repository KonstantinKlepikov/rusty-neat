use crate::genes::TraitParameters;
use std::collections::HashMap as StdHashMap;

/// Simplified Parameters struct capturing only fields needed for mutations.
#[derive(Debug, Clone)]
pub struct Parameters {
    pub min_weight: f64,
    pub max_weight: f64,
    pub split_recurrent: bool,
    pub dont_use_bias_neuron: bool,

    // probabilities
    pub mutate_add_link_from_bias_prob: f64,
    pub recurrent_prob: f64,
    pub recurrent_loop_prob: f64,
    // global trait mutation probabilities (from C++ defaults)
    pub mutate_neuron_traits_prob: f64,
    pub mutate_link_traits_prob: f64,
    pub mutate_genome_traits_prob: f64,

    // weight mutation
    pub mutate_weights_severe_prob: f64,
    pub weight_mutation_rate: f64,
    pub weight_replacement_rate: f64,
    pub weight_mutation_max_power: f64,
    pub weight_replacement_max_power: f64,

    // neuron activation mutations
    pub activation_a_mutation_max_power: f64,
    pub min_activation_a: f64,
    pub max_activation_a: f64,
    pub activation_b_mutation_max_power: f64,
    pub min_activation_b: f64,
    pub max_activation_b: f64,

    // neuron time-constant and bias mutation parameters
    pub timeconstant_mutation_max_power: f64,
    pub min_neuron_time_constant: f64,
    pub max_neuron_time_constant: f64,
    pub bias_mutation_max_power: f64,
    pub min_neuron_bias: f64,
    pub max_neuron_bias: f64,

    /// Probability vector for choosing activation function types.
    /// Length should match number of ActivationFunction variants.
    pub activation_function_probs: Vec<f64>,

    // trait mutation parameters maps (full TraitParameters used for C++-like semantics)
    pub neuron_trait_parameters: StdHashMap<String, TraitParameters>,
    pub link_trait_parameters: StdHashMap<String, TraitParameters>,
    pub genome_trait_parameters: StdHashMap<String, TraitParameters>,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            min_weight: -8.0,
            max_weight: 8.0,
            split_recurrent: false,
            dont_use_bias_neuron: false,
            mutate_add_link_from_bias_prob: 0.0,
            recurrent_prob: 0.25,
            recurrent_loop_prob: 0.25,
            // global trait mutation probabilities (C++ defaults are 0.0)
            mutate_neuron_traits_prob: 0.0,
            mutate_link_traits_prob: 0.0,
            mutate_genome_traits_prob: 0.0,
            mutate_weights_severe_prob: 0.25,
            weight_mutation_rate: 1.0,
            weight_replacement_rate: 0.2,
            weight_mutation_max_power: 1.0,
            weight_replacement_max_power: 1.0,
            activation_a_mutation_max_power: 0.0,
            min_activation_a: 1.0,
            max_activation_a: 1.0,
            activation_b_mutation_max_power: 0.0,
            min_activation_b: 0.0,
            max_activation_b: 0.0,
            timeconstant_mutation_max_power: 0.0,
            min_neuron_time_constant: 0.0,
            max_neuron_time_constant: 0.0,
            bias_mutation_max_power: 1.0,
            min_neuron_bias: 0.0,
            max_neuron_bias: 0.0,
            // C++ defaults: UnsignedSigmoid probability = 1.0 (second entry)
            activation_function_probs: vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            neuron_trait_parameters: StdHashMap::new(),
            link_trait_parameters: StdHashMap::new(),
            genome_trait_parameters: StdHashMap::new(),
        }
    }
}
