//! Gene definitions moved out of genome.rs
use std::collections::HashMap as StdHashMap;

/// Trait value container (simplified)
#[derive(Debug, Clone, PartialEq)]
pub enum TraitValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

use crate::network::{ActivationFunction, NeuronType};

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
