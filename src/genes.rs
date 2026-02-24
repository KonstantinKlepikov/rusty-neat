use crate::utils::{clamp_f64, clamp_i64};
use rand::prelude::*;
use std::collections::HashMap as StdHashMap;
use serde::{Deserialize, Serialize};

/// Trait parameter details (simplified)
#[derive(Debug, Clone)]
pub enum TraitDetail {
    Int {
        min: i64,
        max: i64,
        mut_power: i64,
        mut_replace_prob: f64,
    },
    Float {
        min: f64,
        max: f64,
        mut_power: f64,
        mut_replace_prob: f64,
    },
    Str {
        set: Vec<String>,
        probs: Vec<f64>,
    },
    IntSet {
        set: Vec<i64>,
        probs: Vec<f64>,
    },
    FloatSet {
        set: Vec<f64>,
        probs: Vec<f64>,
    },
}

/// Trait value container (simplified)
#[derive(Debug, Clone, PartialEq)]
pub enum TraitValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// Trait parameters
#[derive(Debug, Clone)]
pub struct TraitParameters {
    pub importance_coeff: f64,
    pub mutation_prob: f64,
    pub detail: TraitDetail,
    pub dep_key: Option<String>,
    pub dep_values: Vec<TraitValue>,
}

/// Activation function type
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

/// Neuron type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NeuronType {
    Input,
    Output,
    Hidden,
    Bias,
}

/// Simplified Gene struct.
///
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
    ///
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
                                self.traits
                                    .insert(k.clone(), TraitValue::Int((mi + oi) / 2));
                            }
                        }
                        (TraitValue::Float(mf), TraitValue::Float(of)) => {
                            if rng.random::<f64>() < 0.5 {
                                self.traits.insert(k.clone(), TraitValue::Float(*mf));
                            } else if rng.random::<f64>() < 0.5 {
                                self.traits.insert(k.clone(), TraitValue::Float(*of));
                            } else {
                                self.traits
                                    .insert(k.clone(), TraitValue::Float((*mf + *of) / 2.0));
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
    /// Mutate traits according to a map of `TraitParameters` (matches C++ behavior).
    /// Returns true if any trait mutated.
    pub fn mutate_traits(
        &mut self,
        tp: &StdHashMap<String, TraitParameters>,
        rng: &mut impl Rng,
    ) -> bool {
        // delegate to helper that operates on a trait map
        mutate_trait_map(&mut self.traits, tp, rng)
    }

    /// Mutate traits using a simple name->probability map (fallback API used by Genome)
    pub fn mutate_traits_map(
        &mut self,
        mutation_probs: &StdHashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> bool {
        let mut did_mutate = false;
        for (name, prob) in mutation_probs {
            if rng.random::<f64>() < *prob {
                if let Some(tv) = self.traits.get_mut(name) {
                    match tv {
                        TraitValue::Int(v) => {
                            let delta = rng.random_range(-2i64..=2i64);
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Float(v) => {
                            let delta = (rng.random::<f64>() - 0.5) * 0.2 * ((*v).abs().max(1.0));
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Str(s) => {
                            *s = String::new();
                            did_mutate = true;
                        }
                        TraitValue::Bool(b) => {
                            *b = !*b;
                            did_mutate = true;
                        }
                    }
                }
            }
        }
        did_mutate
    }

    /// Initialize traits according to `TraitParameters` (mirrors C++ InitTraits)
    pub fn init_traits(&mut self, tp: &StdHashMap<String, TraitParameters>, rng: &mut impl Rng) {
        init_trait_map(&mut self.traits, tp, rng);
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

/// Neuron gene
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
    pub fn new(
        id: u64,
        neuron_type: NeuronType,
        x: i32,
        y: i32,
        split_y: f64,
        a: f64,
        b: f64,
        timeconstant: f64,
        bias: f64,
        activation_function: ActivationFunction,
    ) -> Self {
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
        NeuronGene::new(
            0,
            NeuronType::Hidden,
            0,
            0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            ActivationFunction::SignedSigmoid,
        )
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
        self.traits
            .insert(name.to_string(), TraitValue::Float(value));
    }

    /// Mutate traits stored on this neuron using the provided probability map.
    pub fn mutate_traits_map(
        &mut self,
        mutation_probs: &StdHashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> bool {
        let mut did_mutate = false;
        for (name, prob) in mutation_probs {
            if rng.random::<f64>() < *prob {
                if let Some(tv) = self.traits.get_mut(name) {
                    match tv {
                        TraitValue::Int(v) => {
                            let delta = rng.random_range(-2i64..=2i64);
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Float(v) => {
                            let delta = (rng.random::<f64>() - 0.5) * 0.2 * ((*v).abs().max(1.0));
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Str(s) => {
                            *s = String::new();
                            did_mutate = true;
                        }
                        TraitValue::Bool(b) => {
                            *b = !*b;
                            did_mutate = true;
                        }
                    }
                }
            }
        }
        did_mutate
    }

    /// Randomize traits for this neuron (simple heuristics per type).
    pub fn randomize_traits_map(&mut self, rng: &mut impl Rng) {
        // Fallback simple randomization (used when no TraitParameters are present)
        for (_k, v) in self.traits.iter_mut() {
            match v {
                TraitValue::Int(iv) => {
                    *iv = rng.random_range(-5i64..=5i64);
                }
                TraitValue::Float(fv) => {
                    *fv = rng.random_range(-1.0..1.0);
                }
                TraitValue::Str(s) => {
                    *s = String::new();
                }
                TraitValue::Bool(b) => {
                    *b = rng.random::<bool>();
                }
            }
        }
    }

    /// Initialize traits according to `TraitParameters` (mirrors C++ InitTraits)
    pub fn init_traits(&mut self, tp: &StdHashMap<String, TraitParameters>, rng: &mut impl Rng) {
        init_trait_map(&mut self.traits, tp, rng);
    }

    /// Mutate traits using full `TraitParameters` map (C++-like semantics)
    pub fn mutate_traits(
        &mut self,
        tp: &StdHashMap<String, TraitParameters>,
        rng: &mut impl Rng,
    ) -> bool {
        mutate_trait_map(&mut self.traits, tp, rng)
    }
}

impl Default for NeuronGene {
    fn default() -> Self {
        NeuronGene::empty()
    }
}

/// Link gene
#[derive(Debug, Clone)]
pub struct LinkGene {
    /// These variables are initialized once and cannot be changed anymore
    /// The IDs of the neurons that this link connects
    pub from_neuron_id: u64,
    pub to_neuron_id: u64,
    /// The link's innovation ID
    pub innovation_id: u64,

    /// This variable is modified during evolution
    /// The weight of the connection
    pub weight: f64,
    /// Is it recurrent?
    pub is_recurrent: bool,

    /// Optional trait map for extra parameters (hebb_rate, hebb_pre_rate, ...)
    pub traits: StdHashMap<String, TraitValue>,
}

impl LinkGene {
    /// Create a new LinkGene with explicit fields
    pub fn new(
        from_neuron_id: u64,
        to_neuron_id: u64,
        innovation_id: u64,
        weight: f64,
        is_recurrent: bool,
    ) -> Self {
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

    /// Check whether the link is looped recurrent (from == to)
    pub fn is_looped_recurrent(&self) -> bool {
        self.from_neuron_id == self.to_neuron_id
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
        self.traits
            .insert(name.to_string(), TraitValue::Float(value));
    }

    /// Mutate traits on this link using the provided probability map.
    pub fn mutate_traits_map(
        &mut self,
        mutation_probs: &StdHashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> bool {
        let mut did_mutate = false;
        for (name, prob) in mutation_probs {
            if rng.random::<f64>() < *prob {
                if let Some(tv) = self.traits.get_mut(name) {
                    match tv {
                        TraitValue::Int(v) => {
                            let delta = rng.random_range(-2i64..=2i64);
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Float(v) => {
                            let delta = (rng.random::<f64>() - 0.5) * 0.2 * ((*v).abs().max(1.0));
                            *v = *v + delta;
                            did_mutate = true;
                        }
                        TraitValue::Str(s) => {
                            *s = String::new();
                            did_mutate = true;
                        }
                        TraitValue::Bool(b) => {
                            *b = !*b;
                            did_mutate = true;
                        }
                    }
                }
            }
        }
        did_mutate
    }

    /// Mutate traits using full `TraitParameters` map (C++-like semantics)
    pub fn mutate_traits(
        &mut self,
        tp: &StdHashMap<String, TraitParameters>,
        rng: &mut impl Rng,
    ) -> bool {
        // delegate to generic helper operating on trait HashMap
        mutate_trait_map(&mut self.traits, tp, rng)
    }

    /// Randomize traits on this link.
    pub fn randomize_traits_map(&mut self, rng: &mut impl Rng) {
        for (_k, v) in self.traits.iter_mut() {
            match v {
                TraitValue::Int(iv) => {
                    *iv = rng.random_range(-5i64..=5i64);
                }
                TraitValue::Float(fv) => {
                    *fv = rng.random_range(-1.0..1.0);
                }
                TraitValue::Str(s) => {
                    *s = String::new();
                }
                TraitValue::Bool(b) => {
                    *b = rng.random::<bool>();
                }
            }
        }
    }

    /// Initialize traits according to `TraitParameters` (mirrors C++ InitTraits)
    pub fn init_traits(&mut self, tp: &StdHashMap<String, TraitParameters>, rng: &mut impl Rng) {
        init_trait_map(&mut self.traits, tp, rng);
    }
}

impl Default for LinkGene {
    fn default() -> Self {
        LinkGene::empty()
    }
}

// Implement equality and ordering based only on `innovation_id`, matching C++ semantics
impl PartialEq for LinkGene {
    fn eq(&self, other: &Self) -> bool {
        self.innovation_id == other.innovation_id
    }
}

impl Eq for LinkGene {}

impl PartialOrd for LinkGene {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LinkGene {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.innovation_id.cmp(&other.innovation_id)
    }
}

// ----------------------
// Helper functions: init/mutate trait maps using TraitParameters (C++-like)
// ----------------------

/// Roulette index
fn roulette_index(probs: &Vec<f64>, rng: &mut impl Rng) -> usize {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let mut pick = rng.random::<f64>() * total;
    let mut idx = 0usize;
    for p in probs {
        if pick <= *p {
            break;
        }
        pick -= *p;
        idx += 1;
    }
    if idx >= probs.len() {
        probs.len() - 1
    } else {
        idx
    }
}

/// Init trait map
fn init_trait_map(
    map: &mut StdHashMap<String, TraitValue>,
    tp: &StdHashMap<String, TraitParameters>,
    rng: &mut impl Rng,
) {
    for (name, params) in tp.iter() {
        match &params.detail {
            TraitDetail::Int { min, max, .. } => {
                let v = rng.random_range(*min..=*max);
                map.insert(name.clone(), TraitValue::Int(v));
            }
            TraitDetail::Float { min, max, .. } => {
                let x = rng.random::<f64>();
                // scale to [min,max]
                let val = x * (max - min) + min;
                map.insert(name.clone(), TraitValue::Float(val));
            }
            TraitDetail::Str { set, probs } => {
                if set.is_empty() {
                    continue;
                }
                let idx = roulette_index(probs, rng);
                map.insert(name.clone(), TraitValue::Str(set[idx].clone()));
            }
            TraitDetail::IntSet { set, probs } => {
                if set.is_empty() {
                    continue;
                }
                let idx = roulette_index(probs, rng);
                map.insert(name.clone(), TraitValue::Int(set[idx]));
            }
            TraitDetail::FloatSet { set, probs } => {
                if set.is_empty() {
                    continue;
                }
                let idx = roulette_index(probs, rng);
                map.insert(name.clone(), TraitValue::Float(set[idx]));
            }
        }
    }
}

/// Mutate trait map
fn mutate_trait_map(
    map: &mut StdHashMap<String, TraitValue>,
    tp: &StdHashMap<String, TraitParameters>,
    rng: &mut impl Rng,
) -> bool {
    let mut did_mutate = false;

    for (name, params) in tp.iter() {
        // dependency check
        let mut doit = true;
        if let Some(dep_key) = &params.dep_key {
            if let Some(dep_val) = map.get(dep_key) {
                // require dep_values contains dep_val
                let mut ok = false;
                for dv in &params.dep_values {
                    if dv == dep_val {
                        ok = true;
                        break;
                    }
                }
                doit = ok;
            } else {
                doit = false;
            }
        }

        if !doit {
            continue;
        }

        if rng.random::<f64>() < params.mutation_prob {
            match &params.detail {
                TraitDetail::Int {
                    min,
                    max,
                    mut_power,
                    mut_replace_prob,
                } => {
                    if let Some(TraitValue::Int(cur)) = map.get(name).cloned() {
                        if rng.random::<f64>() < *mut_replace_prob {
                            // replace until different
                            let mut val = cur;
                            while val == cur {
                                val = rng.random_range(*min..=*max);
                            }
                            map.insert(name.clone(), TraitValue::Int(val));
                            did_mutate = true;
                        } else {
                            let mut val = cur;
                            while val == cur {
                                let delta = rng.random_range(-(*mut_power)..=*mut_power);
                                val = clamp_i64(val + delta, *min, *max);
                            }
                            map.insert(name.clone(), TraitValue::Int(val));
                            did_mutate = true;
                        }
                    }
                }
                TraitDetail::Float {
                    min,
                    max,
                    mut_power,
                    mut_replace_prob,
                } => {
                    if let Some(TraitValue::Float(cur)) = map.get(name).cloned() {
                        if rng.random::<f64>() < *mut_replace_prob {
                            let mut val = cur;
                            while (val - cur).abs() < std::f64::EPSILON {
                                let x = rng.random::<f64>();
                                val = x * (max - min) + min;
                            }
                            map.insert(name.clone(), TraitValue::Float(val));
                            did_mutate = true;
                        } else {
                            let mut val = cur;
                            while (val - cur).abs() < std::f64::EPSILON {
                                val += (rng.random::<f64>() * 2.0 - 1.0) * (*mut_power);
                                val = clamp_f64(val, *min, *max);
                            }
                            map.insert(name.clone(), TraitValue::Float(val));
                            did_mutate = true;
                        }
                    }
                }
                TraitDetail::Str { set, probs } => {
                    if let Some(TraitValue::Str(cur)) = map.get(name).cloned() {
                        if set.is_empty() {
                            continue;
                        }
                        let mut idx = roulette_index(probs, rng);
                        while set[idx] == cur {
                            idx = roulette_index(probs, rng);
                        }
                        map.insert(name.clone(), TraitValue::Str(set[idx].clone()));
                        did_mutate = true;
                    }
                }
                TraitDetail::IntSet { set, probs } => {
                    if let Some(TraitValue::Int(cur)) = map.get(name).cloned() {
                        if set.is_empty() {
                            continue;
                        }
                        let mut idx = roulette_index(probs, rng);
                        while set[idx] == cur as i64 {
                            idx = roulette_index(probs, rng);
                        }
                        map.insert(name.clone(), TraitValue::Int(set[idx]));
                        did_mutate = true;
                    }
                }
                TraitDetail::FloatSet { set, probs } => {
                    if let Some(TraitValue::Float(cur)) = map.get(name).cloned() {
                        if set.is_empty() {
                            continue;
                        }
                        let mut idx = roulette_index(probs, rng);
                        while (set[idx] - cur).abs() < std::f64::EPSILON {
                            idx = roulette_index(probs, rng);
                        }
                        map.insert(name.clone(), TraitValue::Float(set[idx]));
                        did_mutate = true;
                    }
                }
            }
        }
    }

    did_mutate
}
