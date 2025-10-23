//! Gene definitions moved out of genome.rs
use std::collections::HashMap as StdHashMap;
use rand::prelude::*;

/// Trait value container (simplified)
#[derive(Debug, Clone, PartialEq)]
pub enum TraitValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// Activation function type (copied from original C++ types)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    SignedSigmoid,
    UnsignedSigmoid,
    Tanh,
    TanhCubic,
    SignedStep,
    UnsignedStep,
    SignedGauss,
    UnsignedGauss,
    Abs,
    SignedSine,
    UnsignedSine,
    Linear,
    Relu,
    Softplus,
}

/// Neuron type (copied from original C++ types)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    Input,
    Output,
    Hidden,
    Bias,
}

/// Simplified Gene struct mirroring core behavior from the C++ Gene class.
///
/// Assumptions / simplifications vs original C++ implementation:
/// - The full Trait / TraitParameters system from C++ is not ported here.
///   Instead `TraitValue` holds basic types (Int/Float/Str/Bool).
/// - `InitTraits` in C++ uses detailed TraitParameters and RNG-based
///   initialisation; here we provide helpers to mate/mutate traits using
///   a simplified probabilistic behaviour.
/// - This is intentionally lightweight: it provides compatibility points
///   for genome-level operations (mate/mutate/distance) while leaving
///   full parameter parsing to another module if needed.
#[derive(Debug, Clone, PartialEq)]
pub struct Gene {
    /// Map of trait name -> value
    pub traits: StdHashMap<String, TraitValue>,
}

impl Gene {
    /// Create empty gene
    pub fn new() -> Self {
        Gene {
            traits: StdHashMap::new(),
        }
    }

    /// Get trait by name
    pub fn get_trait(&self, name: &str) -> Option<&TraitValue> {
        self.traits.get(name)
    }

    /// Set trait by name
    pub fn set_trait(&mut self, name: &str, value: TraitValue) {
        self.traits.insert(name.to_string(), value);
    }

    /// Mate traits with another parent. Simplified logic:
    /// - If types mismatch, panic (mirrors C++ behaviour choice to error).
    /// - For numeric types we either pick one parent's value at random or
    ///   average them (50% chance each).
    /// - For strings/bools pick either parent at random.
    pub fn mate_traits(&mut self, other: &Gene, rng: &mut impl Rng) {
        for (k, v_other) in &other.traits {
            match (self.traits.get(k), v_other) {
                (Some(v_mine), v_other) => {
                    // types must match
                    match (v_mine, v_other) {
                        (TraitValue::Int(mi), TraitValue::Int(oi)) => {
                            if rng.random::<f64>() < 0.5 {
                                        self.traits.insert(k.clone(), TraitValue::Int(*mi));
                                    } else if rng.random::<f64>() < 0.5 {
                                        self.traits.insert(k.clone(), TraitValue::Int(*oi));
                                    } else {
                                        self.traits.insert(k.clone(), TraitValue::Int((mi + oi) / 2));
                                    }
                        }
                        (TraitValue::Float(mf), TraitValue::Float(of)) => {
                            if rng.random::<f64>() < 0.5 {
                                self.traits.insert(k.clone(), TraitValue::Float(*mf));
                            } else if rng.random::<f64>() < 0.5 {
                                self.traits.insert(k.clone(), TraitValue::Float(*of));
                            } else {
                                self.traits.insert(k.clone(), TraitValue::Float((*mf + *of) / 2.0));
                            }
                        }
                        (TraitValue::Str(_), TraitValue::Str(_)) => {
                            if rng.random::<f64>() < 0.5 {
                                // keep mine
                            } else {
                                self.traits.insert(k.clone(), v_other.clone());
                            }
                        }
                        (TraitValue::Bool(_), TraitValue::Bool(_)) => {
                            if rng.random::<f64>() < 0.5 {
                                // keep mine
                            } else {
                                self.traits.insert(k.clone(), v_other.clone());
                            }
                        }
                        _ => {
                            // Types mismatch: follow C++ choice and panic; caller
                            // should ensure trait types match before mating.
                            panic!("Trait types mismatch during mating for key {}", k);
                        }
                    }
                }
                (None, v_other) => {
                    // other parent has trait we don't; copy it
                    self.traits.insert(k.clone(), v_other.clone());
                }
            }
        }
    }

    /// Mutate traits according to a per-trait probability map.
    /// `mutation_probs` maps trait name -> probability in [0,1].
    /// Returns true if any trait mutated.
    pub fn mutate_traits(&mut self, mutation_probs: &StdHashMap<String, f64>, rng: &mut impl Rng) -> bool {
        let mut did_mutate = false;
        for (name, prob) in mutation_probs {
                    if rng.random::<f64>() < *prob {
                if let Some(tv) = self.traits.get_mut(name) {
                    match tv {
                        TraitValue::Int(v) => {
                            // small integer perturbation
                            let delta = (rng.random_range(-2i64..=2i64)) as i64;
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Float(v) => {
                            // Gaussian-like perturbation
                            let delta = (rng.random::<f64>() - 0.5) * 0.2 * ((*v).abs().max(1.0));
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Str(s) => {
                            // no sensible mutation defined for strings; flip to empty
                            *s = String::new();
                            did_mutate = true;
                        }
                        TraitValue::Bool(b) => {
                            *b = !*b;
                            did_mutate = true;
                        }
                    }
                } else {
                    // trait not present but mutation probability set — ignore
                }
            }
        }

        did_mutate
    }

    /// Compute simple trait distances between this gene and `other`.
    /// Numeric traits return absolute difference; string traits return 0/1.
    pub fn get_trait_distances(&self, other: &Gene) -> StdHashMap<String, f64> {
        let mut dist: StdHashMap<String, f64> = StdHashMap::new();
        for (k, v_other) in &other.traits {
            if let Some(v_my) = self.traits.get(k) {
                match (v_my, v_other) {
                    (TraitValue::Int(a), TraitValue::Int(b)) => {
                        dist.insert(k.clone(), ((*a - *b).abs()) as f64);
                    }
                    (TraitValue::Float(a), TraitValue::Float(b)) => {
                        dist.insert(k.clone(), (a - b).abs());
                    }
                    (TraitValue::Str(a), TraitValue::Str(b)) => {
                        dist.insert(k.clone(), if a == b { 0.0 } else { 1.0 });
                    }
                    (TraitValue::Bool(a), TraitValue::Bool(b)) => {
                        dist.insert(k.clone(), if a == b { 0.0 } else { 1.0 });
                    }
                    _ => {
                        // Type mismatch - skip or set large distance
                        dist.insert(k.clone(), f64::INFINITY);
                    }
                }
            }
        }

        dist
    }
}

impl Default for Gene {
    fn default() -> Self {
        Gene::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuronGene {
    // Unique identification number
    pub id: u64,
    // Its type and role in the network
    pub neuron_type: NeuronType,

    // Display coordinates (as in C++: int x, y)
    pub x: i32,
    pub y: i32,

    // Position (depth) within the network
    pub split_y: f64,

    // Activation function parameters
    pub a: f64,
    pub b: f64,
    pub timeconstant: f64,
    pub bias: f64,
    pub activation_function: ActivationFunction,

    /// Optional trait map for extra parameters (keeps arbitrary traits)
    pub traits: StdHashMap<String, TraitValue>,
}

impl NeuronGene {
    /// Create a new NeuronGene with explicit fields
    pub fn new(id: u64, neuron_type: NeuronType, x: i32, y: i32, split_y: f64, a: f64, b: f64, timeconstant: f64, bias: f64, activation_function: ActivationFunction) -> Self {
        NeuronGene {
            id,
            neuron_type,
            x,
            y,
            split_y,
            a,
            b,
            timeconstant,
            bias,
            activation_function,
            traits: StdHashMap::new(),
        }
    }

    /// Empty/default neuron gene
    pub fn empty() -> Self {
        NeuronGene::new(0, NeuronType::Hidden, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, ActivationFunction::SignedSigmoid)
    }

    /// Get id
    pub fn get_id(&self) -> u64 {
        self.id
    }

    /// Set id
    pub fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    /// Get coordinates
    pub fn coords(&self) -> (i32, i32) {
        (self.x, self.y)
    }

    /// Set coordinates
    pub fn set_coords(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }

    /// Get neuron type
    pub fn get_neuron_type(&self) -> NeuronType {
        self.neuron_type
    }

    /// Set neuron type
    pub fn set_neuron_type(&mut self, nt: NeuronType) {
        self.neuron_type = nt;
    }

    /// Get split_y
    pub fn get_split_y(&self) -> f64 {
        self.split_y
    }

    /// Set split_y
    pub fn set_split_y(&mut self, s: f64) {
        self.split_y = s;
    }

    /// Get activation params (a,b,timeconst,bias)
    pub fn activation_params(&self) -> (f64, f64, f64, f64) {
        (self.a, self.b, self.timeconstant, self.bias)
    }

    /// Set activation params
    pub fn set_activation_params(&mut self, a: f64, b: f64, timeconst: f64, bias: f64) {
        self.a = a;
        self.b = b;
        self.timeconstant = timeconst;
        self.bias = bias;
    }

    /// Helper: get a float trait by name
    pub fn get_trait_float(&self, name: &str) -> Option<f64> {
        match self.traits.get(name) {
            Some(TraitValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Helper: set a float trait
    pub fn set_trait_float(&mut self, name: &str, value: f64) {
        self.traits.insert(name.to_string(), TraitValue::Float(value));
    }
}

impl Default for NeuronGene {
    fn default() -> Self {
        NeuronGene::empty()
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct LinkGene {
    pub from_neuron_id: u64,
    pub to_neuron_id: u64,
    pub innovation_id: u64,
    pub weight: f64,
    pub is_recurrent: bool,
    /// Optional trait map for extra parameters (hebb_rate, hebb_pre_rate, ...)
    pub traits: StdHashMap<String, TraitValue>,
}

impl LinkGene {
    /// Create a new LinkGene with explicit fields
    pub fn new(from_neuron_id: u64, to_neuron_id: u64, innovation_id: u64, weight: f64, is_recurrent: bool) -> Self {
        LinkGene {
            from_neuron_id,
            to_neuron_id,
            innovation_id,
            weight,
            is_recurrent,
            traits: StdHashMap::new(),
        }
    }

    /// Default empty link
    pub fn empty() -> Self {
        LinkGene::new(0, 0, 0, 0.0, false)
    }

    /// Get the link weight
    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    /// Set the link weight
    pub fn set_weight(&mut self, w: f64) {
        self.weight = w;
    }

    /// Get innovation id
    pub fn get_innovation_id(&self) -> u64 {
        self.innovation_id
    }

    /// Set innovation id
    pub fn set_innovation_id(&mut self, id: u64) {
        self.innovation_id = id;
    }

    /// Get endpoints (from,to)
    pub fn endpoints(&self) -> (u64, u64) {
        (self.from_neuron_id, self.to_neuron_id)
    }

    /// Set endpoints
    pub fn set_endpoints(&mut self, from: u64, to: u64) {
        self.from_neuron_id = from;
        self.to_neuron_id = to;
    }

    /// Check whether link is recurrent
    pub fn is_recurrent(&self) -> bool {
        self.is_recurrent
    }

    /// Set recurrent flag
    pub fn set_recurrent(&mut self, r: bool) {
        self.is_recurrent = r;
    }

    /// Helper: get a float trait by name
    pub fn get_trait_float(&self, name: &str) -> Option<f64> {
        match self.traits.get(name) {
            Some(TraitValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Helper: set a float trait
    pub fn set_trait_float(&mut self, name: &str, value: f64) {
        self.traits.insert(name.to_string(), TraitValue::Float(value));
    }
}

impl Default for LinkGene {
    fn default() -> Self {
        LinkGene::empty()
    }
}
