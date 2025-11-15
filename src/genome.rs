//! Genome and gene module (neurons, links)

// === Imports analogous to C++ standard and Boost libraries ===
use std::fs::File;
use std::io::{Read, BufReader, Result as IoResult};
use std::cmp::{PartialEq, PartialOrd, Ordering};
use crate::network::{NeuralNetwork, Neuron as PhNeuron, Connection as PhConnection};
use crate::hyperneat::Substrate;
use crate::genes::{TraitValue, NeuronGene, LinkGene, ActivationFunction};
use rand::Rng;
use crate::innovation::InnovationDatabase;
use crate::parameters::Parameters;



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomeSeedType {
    Perceptron = 0,
    Layered = 1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenomeInitStruct {
    pub num_inputs: usize,
    pub num_hidden: usize, // ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
    pub num_outputs: usize,
    pub fs_neat: bool,
    pub output_act_type: ActivationFunction,
    pub hidden_act_type: ActivationFunction,
    pub seed_type: GenomeSeedType,
    pub num_layers: usize,
    pub fs_neat_links: usize,
}

/// Use Gene defined in `genes.rs` for trait storage and manipulation
use crate::genes::Gene as GenomeGene;

/// Placeholder for PhenotypeBehavior struct (to be implemented)
#[derive(Debug, Clone, PartialEq)]
pub struct PhenotypeBehavior;

/// Central NEAT genome structure (analog of C++ class Genome)
#[derive(Debug, Clone, PartialEq)]
pub struct Genome {

    /// Genome ID
    pub id: u64,
    /// Number of input neurons
    pub num_inputs: usize,
    /// Number of output neurons
    pub num_outputs: usize,
    /// Genome's fitness score
    pub fitness: f64,
    /// Genome's adjusted fitness score
    pub adjusted_fitness: f64,
    /// Depth of the network
    pub depth: usize,
    /// How many individuals this genome should spawn
    pub offspring_amount: f64,
    /// List of neuron genes
    pub neuron_genes: Vec<NeuronGene>,
    /// List of link genes
    pub link_genes: Vec<LinkGene>,
    /// Traits that belong to the genome itself
    pub genome_gene: Option<GenomeGene>,
    /// Whether this genome was already evaluated (used in steady state evolution)
    pub evaluated: bool,
    /// Initial genome complexity: number of neurons
    pub initial_num_neurons: usize,
    /// Initial genome complexity: number of links
    pub initial_num_links: usize,
    /// Pointer to phenotype behavior (for novelty search, placeholder for now)
    pub phenotype_behavior: Option<PhenotypeBehavior>,
}

impl Genome {
    /// Create a new Genome from parameters and initialization struct
    pub fn new(id: u64, params: &GenomeInitStruct) -> Self {
        // This is a stub. Real logic should initialize neuron_genes, link_genes, etc.
        Genome {
            id,
            num_inputs: params.num_inputs,
            num_outputs: params.num_outputs,
            fitness: 0.0,
            adjusted_fitness: 0.0,
            depth: 0,
            offspring_amount: 0.0,
            neuron_genes: Vec::new(),
            link_genes: Vec::new(),
            genome_gene: None,
            evaluated: false,
            initial_num_neurons: params.num_inputs + params.num_hidden + params.num_outputs,
            initial_num_links: 0,
            phenotype_behavior: None,
        }
    }

    /// Copy constructor (clone)
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Load Genome from file (stub, needs real deserialization logic)
    pub fn from_file(path: &str) -> IoResult<Self> {
        let file = File::open(path)?;
        let mut reader: BufReader<File> = BufReader::new(file);
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        // TODO: parse contents into Genome
        // For now, return a default Genome
        Ok(Genome::default())
    }
}

impl Genome {
    /// Count links inputting from a given neuron ID
    pub fn links_inputting_from(&self, id: u64) -> usize {
        self.link_genes.iter().filter(|l| l.from_neuron_id == id).count()
    }

    /// Count links outputting to a given neuron ID
    pub fn links_outputting_to(&self, id: u64) -> usize {
        self.link_genes.iter().filter(|l| l.to_neuron_id == id).count()
    }

    /// Mutate: add a neuron by splitting an existing link.
    /// Uses shared `InnovationDatabase` to allocate innovation and neuron ids and `Parameters` for rules.
    pub fn mutate_add_neuron(&mut self, innov_db: &mut InnovationDatabase, params: &Parameters, rng: &mut impl Rng) -> bool {
        if self.link_genes.is_empty() { return false; }

        // try to find a valid link to split
        let mut tries = 256usize;
        let mut chosen_idx: Option<usize> = None;
        while tries > 0 {
            let idx = rng.random_range(0..self.link_genes.len());
            let lg = &self.link_genes[idx];
            // skip recurrent links if not allowed
            if lg.is_recurrent && !params.split_recurrent {
                tries -= 1;
                continue;
            }
            // skip bias source splitting if policy forbids
            if !params.dont_use_bias_neuron {
                // if from neuron is bias (assume last input is bias)
                if let Some(from_idx) = self.get_neuron_index(lg.from_neuron_id) {
                    if from_idx + 1 == self.num_inputs { // bias is last input index
                        tries -= 1;
                        continue;
                    }
                }
            }

            chosen_idx = Some(idx);
            break;
        }

        let idx = match chosen_idx { Some(i) => i, None => return false };
        let chosen = self.link_genes[idx].clone();

        // remove the chosen link
        self.link_genes.remove(idx);

        // check for existing neuron innovation for this split
        let from = chosen.from_neuron_id;
        let to = chosen.to_neuron_id;
        let nid = if let Some(existing) = innov_db.check_neuron_innovation(from, to) {
            existing
        } else {
            innov_db.add_neuron_innovation(from, to)
        };

        // compute split_y
        let split_y = if let (Some(fi), Some(ti)) = (self.get_neuron_index(from), self.get_neuron_index(to)) {
            (self.neuron_genes[fi].split_y + self.neuron_genes[ti].split_y) / 2.0
        } else { 0.5 };

        // add neuron gene if not present
        if self.get_neuron_index(nid).is_none() {
            let new_ng = NeuronGene::new(nid, crate::genes::NeuronType::Hidden, 0, 0, split_y, 1.0, 0.0, 1.0, 0.0, ActivationFunction::SignedSigmoid);
            self.neuron_genes.push(new_ng);
        }

        // ensure link innovations exist
        let l1_innov = innov_db.add_link_innovation(from, nid);
        let l2_innov = innov_db.add_link_innovation(nid, to);

        // create links
        let l1 = LinkGene::new(from, nid, l1_innov, 1.0, chosen.is_recurrent);
        let l2 = LinkGene::new(nid, to, l2_innov, chosen.weight, chosen.is_recurrent);
        self.link_genes.push(l1);
        self.link_genes.push(l2);

        true
    }

    /// Mutate: add a link between two existing neurons.
    /// Simplified: pick random pair (from != to) that doesn't already exist.
    pub fn mutate_add_link(&mut self, innov_db: &mut InnovationDatabase, params: &Parameters, rng: &mut impl Rng) -> bool {
        if self.neuron_genes.len() < 2 { return false; }
        let n = self.neuron_genes.len();
        // try some attempts
        for _ in 0..32 {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            if i == j { continue; }
            let from = self.neuron_genes[i].id;
            let to = self.neuron_genes[j].id;
            // skip if link exists
            if self.link_genes.iter().any(|l| l.from_neuron_id == from && l.to_neuron_id == to) {
                continue;
            }
            // create new innovation id
            // create or reuse innovation id
            let innov = innov_db.check_link_innovation(from, to).unwrap_or_else(|| innov_db.add_link_innovation(from, to));
            // random weight in range
            let w: f64 = rng.random_range(params.min_weight..params.max_weight);
            // decide recurrence
            let mut recur = false;
            if rng.random::<f64>() < params.recurrent_prob {
                recur = true;
                // looped recurrent?
                if rng.random::<f64>() < params.recurrent_loop_prob {
                    // make it looped by setting from==to
                }
            }
            let lg = LinkGene::new(from, to, innov, w, recur);
            self.link_genes.push(lg);
            return true;
        }
        false
    }

    /// Mutate: remove a random link (if more than one exists)
    pub fn mutate_remove_link(&mut self, _params: &Parameters, rng: &mut impl Rng) -> bool {
    if self.link_genes.len() < 2 { return false; }
    let idx = rng.random_range(0..self.link_genes.len());
        self.link_genes.remove(idx);
        true
    }

    /// Mutate: remove a simple hidden neuron (one input, one output) and replace with a direct link.
    pub fn mutate_remove_simple_neuron(&mut self, innov_db: &mut InnovationDatabase, _params: &Parameters, rng: &mut impl Rng) -> bool {
        // find candidate hidden neurons
        let mut candidates: Vec<usize> = Vec::new();
        for (i, ng) in self.neuron_genes.iter().enumerate() {
            if ng.neuron_type == crate::genes::NeuronType::Hidden {
                let in_count = self.links_inputting_from(ng.id);
                let out_count = self.links_outputting_to(ng.id);
                if in_count == 1 && out_count == 1 {
                    candidates.push(i);
                }
            }
        }

    if candidates.is_empty() { return false; }
    let choice = candidates[rng.random_range(0..candidates.len())];
        let nid = self.neuron_genes[choice].id;

        // find the single incoming and outgoing links
        let mut in_idx: Option<usize> = None;
        let mut out_idx: Option<usize> = None;
        for (i, lg) in self.link_genes.iter().enumerate() {
            if lg.to_neuron_id == nid { in_idx = Some(i); }
            if lg.from_neuron_id == nid { out_idx = Some(i); }
        }

    if in_idx.is_none() || out_idx.is_none() { return false; }
        let in_l = self.link_genes[in_idx.unwrap()].clone();
        let out_l = self.link_genes[out_idx.unwrap()].clone();

        // if link from->to already exists, just remove neuron and links
        let from = in_l.from_neuron_id;
        let to = out_l.to_neuron_id;
        let exists = self.link_genes.iter().any(|l| l.from_neuron_id == from && l.to_neuron_id == to);


        // remove links connected to nid (filter)
        self.link_genes.retain(|l| l.from_neuron_id != nid && l.to_neuron_id != nid);

        // remove neuron
        self.neuron_genes.retain(|n| n.id != nid);

        if !exists {
            let innov = innov_db.check_link_innovation(from, to).unwrap_or_else(|| innov_db.add_link_innovation(from, to));
            let lg = LinkGene::new(from, to, innov, in_l.weight, false);
            self.link_genes.push(lg);
        }

        true
    }

    /// Mutate link weights (perturb or replace weights according to Parameters)
    pub fn mutate_link_weights(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        // determine genome tail as in C++ (unused for now but kept for compatibility)
        let mut t_genometail: usize = 0;
        if self.link_genes.len() > self.initial_num_links {
            t_genometail = ((self.link_genes.len() as f64) * 0.8) as usize;
        }
        if t_genometail < self.initial_num_links { t_genometail = self.initial_num_links; }

        let mut did_mutate = false;
        let t_severe_mutation = rng.random::<f64>() < params.mutate_weights_severe_prob;

        for (i, lg) in self.link_genes.iter_mut().enumerate() {
            if !t_severe_mutation && (rng.random::<f64>() < params.weight_mutation_rate) {
                // non-severe mutation: either replace or perturb
                // determine whether this gene is in the genome tail
                let ontail = i >= t_genometail;
                let mut w = lg.weight;

                if ontail || (rng.random::<f64>() < params.weight_replacement_rate) {
                    w = rng.random_range(params.min_weight..params.max_weight);
                } else {
                    // small perturbation proportional to weight range (10%)
                    let range = params.max_weight - params.min_weight;
                    let delta = (rng.random::<f64>() * 2.0 - 1.0) * range * 0.1;
                    w += delta;
                }

                // clamp
                if w < params.min_weight { w = params.min_weight; }
                if w > params.max_weight { w = params.max_weight; }
                lg.set_weight(w);
                did_mutate = true;
            } else if t_severe_mutation {
                if rng.random::<f64>() < params.weight_mutation_rate {
                    let mut w = rng.random_range(params.min_weight..params.max_weight);
                    if w < params.min_weight { w = params.min_weight; }
                    if w > params.max_weight { w = params.max_weight; }
                    lg.set_weight(w);
                    did_mutate = true;
                }
            }
        }

        did_mutate
    }

    /// Randomize all link weights to uniform random in [min_weight, max_weight]
    pub fn randomize_link_weights(&mut self, params: &Parameters, rng: &mut impl Rng) {
        for lg in self.link_genes.iter_mut() {
            let w = rng.random_range(params.min_weight..params.max_weight);
            lg.set_weight(w);
        }
    }

}

/// Pick a random ActivationFunction according to probabilities in Parameters
fn get_random_activation(params: &Parameters, rng: &mut impl Rng) -> ActivationFunction {
    // Ensure probabilities length matches number of ActivationFunction variants
    let probs = &params.activation_function_probs;
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return ActivationFunction::SignedSigmoid;
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
    // clamp idx into range
    let variants = vec![
        ActivationFunction::SignedSigmoid,
        ActivationFunction::UnsignedSigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::TanhCubic,
        ActivationFunction::SignedStep,
        ActivationFunction::UnsignedStep,
        ActivationFunction::SignedGauss,
        ActivationFunction::UnsignedGauss,
        ActivationFunction::Abs,
        ActivationFunction::SignedSine,
        ActivationFunction::UnsignedSine,
        ActivationFunction::Linear,
        ActivationFunction::Relu,
        ActivationFunction::Softplus,
    ];
    let i = if idx >= variants.len() { variants.len() - 1 } else { idx };
    variants[i]
}

impl Genome {
    /// Perturbs the A parameters of the neuron activation functions
    pub fn mutate_neuron_activations_a(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        for ng in self.neuron_genes.iter_mut() {
            if ng.neuron_type != crate::genes::NeuronType::Input && ng.neuron_type != crate::genes::NeuronType::Bias {
                let delta = (rng.random::<f64>() * 2.0 - 1.0) * params.activation_a_mutation_max_power;
                ng.a += delta;
                if ng.a < params.min_activation_a { ng.a = params.min_activation_a; }
                if ng.a > params.max_activation_a { ng.a = params.max_activation_a; }
            }
        }
        true
    }

    /// Perturbs the B parameters of the neuron activation functions
    pub fn mutate_neuron_activations_b(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        for ng in self.neuron_genes.iter_mut() {
            if ng.neuron_type != crate::genes::NeuronType::Input && ng.neuron_type != crate::genes::NeuronType::Bias {
                let delta = (rng.random::<f64>() * 2.0 - 1.0) * params.activation_b_mutation_max_power;
                ng.b += delta;
                if ng.b < params.min_activation_b { ng.b = params.min_activation_b; }
                if ng.b > params.max_activation_b { ng.b = params.max_activation_b; }
            }
        }
        true
    }

    /// Changes the activation function type for a random neuron
    pub fn mutate_neuron_activation_type(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        if self.neuron_genes.len() <= self.num_inputs { return false; }
        let start = self.num_inputs;
        let end = self.neuron_genes.len();
        let idx = rng.random_range(start..end);
        let cur = self.neuron_genes[idx].activation_function;
        let new_act = get_random_activation(params, rng);
        self.neuron_genes[idx].activation_function = new_act;
        new_act != cur
    }

    /// Perturbs the neuron time constants
    pub fn mutate_neuron_timeconstants(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        for ng in self.neuron_genes.iter_mut() {
            if ng.neuron_type != crate::genes::NeuronType::Input && ng.neuron_type != crate::genes::NeuronType::Bias {
                let delta = (rng.random::<f64>() * 2.0 - 1.0) * params.timeconstant_mutation_max_power;
                ng.timeconstant += delta;
                if ng.timeconstant < params.min_neuron_time_constant { ng.timeconstant = params.min_neuron_time_constant; }
                if ng.timeconstant > params.max_neuron_time_constant { ng.timeconstant = params.max_neuron_time_constant; }
            }
        }
        true
    }

    /// Perturbs the neuron biases
    pub fn mutate_neuron_biases(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        for ng in self.neuron_genes.iter_mut() {
            if ng.neuron_type != crate::genes::NeuronType::Input && ng.neuron_type != crate::genes::NeuronType::Bias {
                let delta = (rng.random::<f64>() * 2.0 - 1.0) * params.bias_mutation_max_power;
                ng.bias += delta;
                if ng.bias < params.min_neuron_bias { ng.bias = params.min_neuron_bias; }
                if ng.bias > params.max_neuron_bias { ng.bias = params.max_neuron_bias; }
            }
        }
        true
    }

    /// Mutate neuron traits using parameters' mutation probability map
    pub fn mutate_neuron_traits(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        let mut did = false;
        for ng in self.neuron_genes.iter_mut() {
            if ng.mutate_traits(&params.neuron_trait_parameters, rng) { did = true; }
        }
        did
    }

    /// Mutate link traits
    pub fn mutate_link_traits(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        let mut did = false;
        for lg in self.link_genes.iter_mut() {
            if lg.mutate_traits(&params.link_trait_parameters, rng) { did = true; }
        }
        did
    }

    /// Mutate genome-level traits (the optional `genome_gene` container)
    pub fn mutate_genome_traits(&mut self, params: &Parameters, rng: &mut impl Rng) -> bool {
        if let Some(gene) = self.genome_gene.as_mut() {
            return gene.mutate_traits(&params.genome_trait_parameters, rng);
        }
        false
    }

    /// Randomize traits for neurons, links and genome-level gene
    pub fn randomize_traits(&mut self, rng: &mut impl Rng) {
        for ng in self.neuron_genes.iter_mut() {
            ng.randomize_traits_map(rng);
        }
        for lg in self.link_genes.iter_mut() {
            lg.randomize_traits_map(rng);
        }
        if let Some(g) = self.genome_gene.as_mut() {
            // implement a simple randomization for Gene's trait map
            for (_k, v) in g.traits.iter_mut() {
                match v {
                    TraitValue::Int(iv) => { *iv = rng.random_range(-5i64..=5i64); }
                    TraitValue::Float(fv) => { *fv = rng.random_range(-1.0..1.0); }
                    TraitValue::Str(s) => { *s = String::new(); }
                    TraitValue::Bool(b) => { *b = rng.random::<bool>(); }
                }
            }
        }
    }
}

/// Implement Default for Genome (empty genome)
impl Default for Genome {
    fn default() -> Self {
        Genome {
            id: 0,
            num_inputs: 0,
            num_outputs: 0,
            fitness: 0.0,
            adjusted_fitness: 0.0,
            depth: 0,
            offspring_amount: 0.0,
            neuron_genes: Vec::new(),
            link_genes: Vec::new(),
            genome_gene: None,
            evaluated: false,
            initial_num_neurons: 0,
            initial_num_links: 0,
            phenotype_behavior: None,
        }
    }
}

/// Implement Eq and Ord for Genome for comparison operators
impl Eq for Genome {}

impl PartialOrd for Genome {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Genome {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by fitness, then by id as tiebreaker
        self.fitness
            .partial_cmp(&other.fitness)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Genome {
    /// Get neuron by its ID
    pub fn get_neuron_by_id(&self, id: u64) -> Option<&NeuronGene> {
        self.neuron_genes.iter().find(|n| n.id == id)
    }

    /// Get neuron by its index
    pub fn get_neuron_by_index(&self, idx: usize) -> Option<&NeuronGene> {
        self.neuron_genes.get(idx)
    }

    /// Get link by its innovation ID
    pub fn get_link_by_innov_id(&self, innov_id: u64) -> Option<&LinkGene> {
        self.link_genes.iter().find(|l| l.innovation_id == innov_id)
    }

    /// Get link by its index
    pub fn get_link_by_index(&self, idx: usize) -> Option<&LinkGene> {
        self.link_genes.get(idx)
    }

    /// Get neuron index by ID
    pub fn get_neuron_index(&self, id: u64) -> Option<usize> {
        self.neuron_genes.iter().position(|n| n.id == id)
    }

    /// Get link index by innovation ID
    pub fn get_link_index(&self, innov_id: u64) -> Option<usize> {
        self.link_genes.iter().position(|l| l.innovation_id == innov_id)
    }

    /// Number of neurons
    pub fn num_neurons(&self) -> usize {
        self.neuron_genes.len()
    }

    /// Number of links
    pub fn num_links(&self) -> usize {
        self.link_genes.len()
    }

    /// Number of inputs
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Number of outputs
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Set neuron X and Y coordinates (integers, as in C++ original)
    pub fn set_neuron_xy(&mut self, idx: usize, x: i32, y: i32) {
        assert!(idx < self.neuron_genes.len(), "Index out of bounds in set_neuron_xy");
        self.neuron_genes[idx].x = x;
        self.neuron_genes[idx].y = y;
    }

    /// Set neuron X coordinate
    pub fn set_neuron_x(&mut self, idx: usize, x: i32) {
        assert!(idx < self.neuron_genes.len(), "Index out of bounds in set_neuron_x");
        self.neuron_genes[idx].x = x;
    }

    /// Set neuron Y coordinate
    pub fn set_neuron_y(&mut self, idx: usize, y: i32) {
        assert!(idx < self.neuron_genes.len(), "Index out of bounds in set_neuron_y");
        self.neuron_genes[idx].y = y;
    }

    /// Get fitness
    pub fn get_fitness(&self) -> f64 {
        self.fitness
    }

    /// Set fitness
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    /// Get adjusted fitness
    pub fn get_adj_fitness(&self) -> f64 {
        self.adjusted_fitness
    }

    /// Set adjusted fitness
    pub fn set_adj_fitness(&mut self, adj_fitness: f64) {
        self.adjusted_fitness = adj_fitness;
    }

    /// Get genome ID
    pub fn get_id(&self) -> u64 {
        self.id
    }

    /// Set genome ID
    pub fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    /// Get depth
    pub fn get_depth(&self) -> usize {
        self.depth
    }

    /// Set depth
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    /// Get offspring amount
    pub fn get_offspring_amount(&self) -> f64 {
        self.offspring_amount
    }

    /// Set offspring amount
    pub fn set_offspring_amount(&mut self, amount: f64) {
        self.offspring_amount = amount;
    }

    /// Is genome evaluated
    pub fn is_evaluated(&self) -> bool {
        self.evaluated
    }

    /// Set evaluated flag
    pub fn set_evaluated(&mut self) {
        self.evaluated = true;
    }

    /// Reset evaluated flag
    pub fn reset_evaluated(&mut self) {
        self.evaluated = false;
    }
}

impl Genome {
    /// Build phenotype neural network from genome
    pub fn build_phenotype(&self, net: &mut NeuralNetwork) {
        // Clear the network
        net.clear();
        net.set_input_output_dimensions(self.num_inputs, self.num_outputs);

        // Fill the net with the neurons
        for ng in &self.neuron_genes {
            // Map Genome's NeuronGene -> phenotype Neuron
            let n = PhNeuron {
                activesum: 0.0,
                activation: 0.0,
                a: ng.a,
                b: ng.b,
                timeconst: ng.timeconstant,
                bias: ng.bias,
                membrane_potential: 0.0,
                activation_function_type: ng.activation_function,
                x: ng.x as f64,
                y: ng.y as f64,
                z: 0.0,
                sx: 0.0,
                sy: 0.0,
                sz: 0.0,
                substrate_coords: Vec::new(),
                split_y: ng.split_y,
                neuron_type: ng.neuron_type,
                sensitivity_matrix: Vec::new(),
            };

            net.add_neuron(n);
        }

        // Fill the net with the connections
        for lg in &self.link_genes {
            // find source/target neuron indices in the phenotype (by neuron id)
            if let (Some(src_idx), Some(dst_idx)) = (self.get_neuron_index(lg.from_neuron_id), self.get_neuron_index(lg.to_neuron_id)) {
                // extract hebbian params from traits if present
                let mut hebb_rate = 0.3_f64;
                let mut hebb_pre_rate = 0.1_f64;
                if let Some(TraitValue::Float(v)) = lg.traits.get("hebb_rate") {
                    hebb_rate = *v;
                }
                if let Some(TraitValue::Float(v)) = lg.traits.get("hebb_pre_rate") {
                    hebb_pre_rate = *v;
                }

                let c = PhConnection {
                    source_neuron_idx: src_idx,
                    target_neuron_idx: dst_idx,
                    weight: lg.weight,
                    signal: 0.0,
                    recur_flag: lg.is_recurrent,
                    hebb_rate,
                    hebb_pre_rate,
                };

                net.add_connection(c);
            }
            else {
                // neuron id not found in phenotype; skip this connection
            }
        }
        // flush network state (matches C++ a_Net.Flush())
        net.flush();

        // Note: RTRL variables and advanced features are not implemented in this prototype
    }

    /// Build a HyperNEAT phenotype based on a substrate and this genome acting as the CPPN.
    /// This is a pragmatic port of the C++ Genome::BuildHyperNEATPhenotype implementation.
    pub fn build_hyperneat_phenotype(&self, net: &mut NeuralNetwork, subst: &Substrate) {
        // minimal validations
        assert!(!subst.input_coords.is_empty(), "substrate must have input coords");
        assert!(!subst.output_coords.is_empty(), "substrate must have output coords");

        let max_dims = subst.get_max_dims();

        // ensure CPPN I/O sizes are compatible
        assert!(subst.get_min_cppn_inputs() > 0);
        assert!(self.num_inputs >= subst.get_min_cppn_inputs(), "genome (CPPN) does not have enough inputs");
        assert!(self.num_outputs >= subst.get_min_cppn_outputs(), "genome (CPPN) does not have enough outputs");

        // create target substrate network
        net.clear();
        net.set_input_output_dimensions(subst.input_coords.len(), subst.output_coords.len());

        // Inputs
        for coord in &subst.input_coords {
            let n = PhNeuron {
                activesum: 0.0,
                activation: 0.0,
                a: 1.0,
                b: 0.0,
                timeconst: 1.0,
                bias: 0.0,
                membrane_potential: 0.0,
                activation_function_type: ActivationFunction::Linear,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                sx: 0.0,
                sy: 0.0,
                sz: 0.0,
                substrate_coords: coord.clone(),
                split_y: 0.0,
                neuron_type: crate::genes::NeuronType::Input,
                sensitivity_matrix: Vec::new(),
            };
            net.add_neuron(n);
        }

        // Outputs
        for coord in &subst.output_coords {
            let n = PhNeuron {
                activesum: 0.0,
                activation: 0.0,
                a: 1.0,
                b: 0.0,
                timeconst: 1.0,
                bias: 0.0,
                membrane_potential: 0.0,
                activation_function_type: subst.output_nodes_activation,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                sx: 0.0,
                sy: 0.0,
                sz: 0.0,
                substrate_coords: coord.clone(),
                split_y: 0.0,
                neuron_type: crate::genes::NeuronType::Output,
                sensitivity_matrix: Vec::new(),
            };
            net.add_neuron(n);
        }

        // Hidden
        for coord in &subst.hidden_coords {
            let n = PhNeuron {
                activesum: 0.0,
                activation: 0.0,
                a: 1.0,
                b: 0.0,
                timeconst: 1.0,
                bias: 0.0,
                membrane_potential: 0.0,
                activation_function_type: subst.hidden_nodes_activation,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                sx: 0.0,
                sy: 0.0,
                sz: 0.0,
                substrate_coords: coord.clone(),
                split_y: 0.0,
                neuron_type: crate::genes::NeuronType::Hidden,
                sensitivity_matrix: Vec::new(),
            };
            net.add_neuron(n);
        }

        // Build the CPPN phenotype (this genome acts as CPPN)
        let mut t_temp_phenotype = NeuralNetwork::new();
        self.build_phenotype(&mut t_temp_phenotype);
        t_temp_phenotype.flush();

        // relaxation depth
        let dp = 8usize;

        // If substrate is leaky, first set neuron-specific properties
        if subst.leaky {
            for i in net.num_inputs()..net.neurons.len() {
                t_temp_phenotype.flush();

                // prepare inputs for CPPN: fill with neuron's substrate coords
                let mut t_inputs = vec![0.0_f64; self.num_inputs];
                let sc = &net.neurons[i].substrate_coords;
                for n in 0..sc.len() {
                    if n < t_inputs.len() { t_inputs[n] = sc[n]; }
                }

                if subst.with_distance {
                    // Euclidean distance to origin
                    let mut sum = 0.0f64;
                    for n in 0..max_dims {
                        let v = if n < sc.len() { sc[n] } else { 0.0 };
                        sum += v * v;
                    }
                    sum = sum.sqrt();
                    if self.num_inputs >= 2 { t_inputs[self.num_inputs - 2] = sum; }
                }
                if self.num_inputs > 0 { t_inputs[self.num_inputs - 1] = 1.0; }

                t_temp_phenotype.input(t_inputs);
                for _ in 0..dp { t_temp_phenotype.activate(); }

                let out = t_temp_phenotype.output();
                if out.len() >= 2 {
                    let mut t_tc = out[self.num_outputs.saturating_sub(2)];
                    let mut t_bias = out[self.num_outputs.saturating_sub(1)];
                    // clamp
                    if t_tc < -1.0 { t_tc = -1.0; } else if t_tc > 1.0 { t_tc = 1.0; }
                    if t_bias < -1.0 { t_bias = -1.0; } else if t_bias > 1.0 { t_bias = 1.0; }

                    // scale
                    let scaled_tc = (t_tc + 1.0) * 0.5 * (subst.max_time_const - subst.min_time_const) + subst.min_time_const;
                    let scaled_bias = t_bias * subst.max_weight_and_bias * -1.0; // match sign scaling in C++

                    net.neurons[i].timeconst = scaled_tc;
                    net.neurons[i].bias = scaled_bias;
                }
            }
        }

        // Collect candidate connections
        let mut t_to_query: Vec<(usize, usize)> = Vec::new();

        if subst.custom_connectivity.is_empty() {
            for i in net.num_inputs()..net.neurons.len() {
                for j in 0..net.neurons.len() {
                    // apply substrate flags (mirror C++ conditions)
                    let src_t = net.neurons[j].neuron_type;
                    let dst_t = net.neurons[i].neuron_type;

                    if (!subst.allow_input_hidden_links && src_t == crate::genes::NeuronType::Input && dst_t == crate::genes::NeuronType::Hidden) ||
                       (!subst.allow_input_output_links && src_t == crate::genes::NeuronType::Input && dst_t == crate::genes::NeuronType::Output) ||
                       (!subst.allow_hidden_hidden_links && src_t == crate::genes::NeuronType::Hidden && dst_t == crate::genes::NeuronType::Hidden && i != j) ||
                       (!subst.allow_hidden_output_links && src_t == crate::genes::NeuronType::Hidden && dst_t == crate::genes::NeuronType::Output) ||
                       (!subst.allow_output_hidden_links && src_t == crate::genes::NeuronType::Output && dst_t == crate::genes::NeuronType::Hidden) ||
                       (!subst.allow_output_output_links && src_t == crate::genes::NeuronType::Output && dst_t == crate::genes::NeuronType::Output && i != j) ||
                       (!subst.allow_looped_hidden_links && src_t == crate::genes::NeuronType::Hidden && dst_t == crate::genes::NeuronType::Hidden && i == j) ||
                       (!subst.allow_looped_output_links && src_t == crate::genes::NeuronType::Output && dst_t == crate::genes::NeuronType::Output && i == j)
                    {
                        continue;
                    }

                    t_to_query.push((j, i));
                }
            }
        } else {
            // custom connectivity - map indices
            for entry in &subst.custom_connectivity {
                if entry.len() < 4 { continue; }
                let src_type = entry[0];
                let src_idx = entry[1] as usize;
                let dst_type = entry[2];
                let dst_idx = entry[3] as usize;

                let j = match src_type {
                    0 => src_idx, // INPUT
                    1 => src_idx, // BIAS treated as input index
                    2 => subst.input_coords.len() + subst.output_coords.len() + src_idx, // HIDDEN
                    3 => subst.input_coords.len() + src_idx, // OUTPUT
                    _ => continue,
                };

                let i = match dst_type {
                    0 => dst_idx,
                    1 => dst_idx,
                    2 => subst.input_coords.len() + subst.output_coords.len() + dst_idx,
                    3 => subst.input_coords.len() + dst_idx,
                    _ => continue,
                };

                // obey flags if requested
                if subst.custom_conn_obeys_flags {
                    let src_t = net.neurons[j].neuron_type;
                    let dst_t = net.neurons[i].neuron_type;
                    if (!subst.allow_input_hidden_links && src_t == crate::genes::NeuronType::Input && dst_t == crate::genes::NeuronType::Hidden) ||
                       (!subst.allow_input_output_links && src_t == crate::genes::NeuronType::Input && dst_t == crate::genes::NeuronType::Output)
                    {
                        continue;
                    }
                }

                t_to_query.push((j, i));
            }
        }

        // Query CPPN for each candidate connection
        for (j, i) in t_to_query {
            let mut t_inputs = vec![0.0_f64; self.num_inputs];

            let from_dims = net.neurons[j].substrate_coords.len();
            let to_dims = net.neurons[i].substrate_coords.len();

            for n in 0..from_dims {
                if n < t_inputs.len() { t_inputs[n] = net.neurons[j].substrate_coords[n]; }
            }
            for n in 0..to_dims {
                let idx = max_dims + n;
                if idx < t_inputs.len() { t_inputs[idx] = net.neurons[i].substrate_coords[n]; }
            }

            if subst.with_distance {
                let mut sum = 0.0f64;
                for n in 0..max_dims {
                    let a = if n < net.neurons[j].substrate_coords.len() { net.neurons[j].substrate_coords[n] } else { 0.0 };
                    let b = if n < net.neurons[i].substrate_coords.len() { net.neurons[i].substrate_coords[n] } else { 0.0 };
                    sum += (a - b) * (a - b);
                }
                sum = sum.sqrt();
                if self.num_inputs >= 2 { t_inputs[self.num_inputs - 2] = sum; }
            }

            if self.num_inputs > 0 { t_inputs[self.num_inputs - 1] = 1.0; }

            t_temp_phenotype.flush();
            t_temp_phenotype.input(t_inputs);
            for _ in 0..dp { t_temp_phenotype.activate(); }

            let out = t_temp_phenotype.output();
            let mut t_link = 0.0;
            let mut t_weight = 0.0;

            if subst.query_weights_only {
                if !out.is_empty() { t_weight = out[0]; }
            } else {
                if !out.is_empty() { t_link = out[0]; }
                if out.len() > 1 { t_weight = out[1]; }
            }

            if (t_link > 0.0 && !subst.query_weights_only) || subst.query_weights_only {
                t_weight *= subst.max_weight_and_bias;
                let c = PhConnection {
                    source_neuron_idx: j,
                    target_neuron_idx: i,
                    weight: t_weight,
                    signal: 0.0,
                    recur_flag: false,
                    hebb_rate: 0.0,
                    hebb_pre_rate: 0.0,
                };
                net.add_connection(c);
            }
        }
    }

    /// Project weight changes from a phenotype `NeuralNetwork` back to this genome.
    /// Behavior mirrors C++ Genome::DerivePhenotypicChanges: it assumes the
    /// phenotype and genome have identical topology and copies connection
    /// weights back into the genome's `link_genes` array.
    pub fn derive_phenotypic_changes(&mut self, net: &NeuralNetwork) {
        // If topology differs (different number of connections) do nothing.
        if net.connections.len() < self.link_genes.len() {
            // topology mismatch: abort
            return;
        }

        for i in 0..self.link_genes.len() {
            self.link_genes[i].set_weight(net.connections[i].weight);
        }
    }

    /// Sort genes: neurons by `id`, links by `innovation_id`.
    /// Mirrors C++ Genome::SortGenes.
    pub fn sort_genes(&mut self) {
        self.neuron_genes.sort_by_key(|n| n.id);
        self.link_genes.sort_by_key(|l| l.innovation_id);
    }

    /// Recursive helper to compute depth of a neuron (number of steps
    /// to reach an input/bias neuron). Mirrors C++ Genome::NeuronDepth.
    pub fn neuron_depth(&self, neuron_id: u64, depth: usize) -> usize {
        const MAX_DEPTH: usize = 16384;

        if depth > MAX_DEPTH {
            // safeguard against infinite recursion in cyclic graphs
            return MAX_DEPTH;
        }

        // If neuron not found, return current depth
        let neuron = match self.get_neuron_by_id(neuron_id) {
            Some(n) => n,
            None => return depth,
        };

        // Base case: inputs and bias have depth = current depth
        if neuron.neuron_type == crate::genes::NeuronType::Input || neuron.neuron_type == crate::genes::NeuronType::Bias {
            return depth;
        }

        // Find all links that output to this neuron
        let mut inputting_links_idx: Vec<usize> = Vec::new();
        for (i, lg) in self.link_genes.iter().enumerate() {
            if lg.to_neuron_id == neuron_id {
                inputting_links_idx.push(i);
            }
        }

        // For each incoming link, recurse and take maximum depth
        let mut max_depth = depth;
        for idx in inputting_links_idx {
            let link = &self.link_genes[idx];
            let cur = self.neuron_depth(link.from_neuron_id, depth + 1);
            if cur > max_depth {
                max_depth = cur;
            }
        }

        max_depth
    }

    /// Calculate overall genome depth (maximum neuron depth across outputs).
    /// Mirrors C++ Genome::CalculateDepth.
    pub fn calculate_depth(&mut self) {
        // Quick case: no hidden neurons
        if self.neuron_genes.len() == (self.num_inputs + self.num_outputs) {
            self.depth = 1;
            return;
        }

        // Collect output neuron IDs
        let mut output_ids: Vec<u64> = Vec::new();
        for ng in &self.neuron_genes {
            if ng.neuron_type == crate::genes::NeuronType::Output {
                output_ids.push(ng.id);
            }
        }

        let mut max_depth: usize = 0;
        for oid in output_ids {
            let cur = self.neuron_depth(oid, 0);
            if cur > max_depth {
                max_depth = cur;
            }
        }

        self.depth = max_depth;
    }

    /// Returns true if the genome contains directed cycles (loops).
    /// Builds a temporary phenotype and checks for cycles using Kahn's algorithm
    /// (topological sort). Mirrors C++ Genome::HasLoops which uses boost::topological_sort.
    pub fn has_loops(&self) -> bool {
        // Build phenotype
        let mut net = NeuralNetwork::new();
        self.build_phenotype(&mut net);

        let n = net.neurons.len();
        if n == 0 { return false; }

        // compute in-degree
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for conn in &net.connections {
            let s = conn.source_neuron_idx;
            let t = conn.target_neuron_idx;
            if s < n && t < n {
                adj[s].push(t);
                indeg[t] += 1;
            }
        }

        // Kahn's algorithm
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for i in 0..n { if indeg[i] == 0 { q.push_back(i); } }

        let mut visited = 0usize;
        while let Some(v) = q.pop_front() {
            visited += 1;
            for &w in &adj[v] {
                indeg[w] -= 1;
                if indeg[w] == 0 { q.push_back(w); }
            }
        }

        // if visited != n then there is a cycle
        visited != n
    }

    /// Return true if the specified neuron ID is a dead end or isolated
    pub fn is_dead_end_neuron(&self, id: u64) -> bool {
        let mut no_incoming = true;
        let mut no_outgoing = true;

        for lg in &self.link_genes {
            // incoming
            if lg.to_neuron_id == id {
                // ignore looped recurrent links and links coming from bias
                let is_looped_recurrent = (lg.from_neuron_id == lg.to_neuron_id) && lg.is_recurrent;
                if !is_looped_recurrent {
                    if let Some(from_neuron) = self.get_neuron_by_id(lg.from_neuron_id) {
                        if from_neuron.neuron_type != crate::genes::NeuronType::Bias {
                            no_incoming = false;
                        }
                    }
                }
            }

            // outgoing
            if lg.from_neuron_id == id {
                let is_looped_recurrent = (lg.from_neuron_id == lg.to_neuron_id) && lg.is_recurrent;
                if !is_looped_recurrent {
                    if let Some(from_neuron) = self.get_neuron_by_id(lg.from_neuron_id) {
                        if from_neuron.neuron_type != crate::genes::NeuronType::Bias {
                            no_outgoing = false;
                        }
                    }
                }
            }
        }

        if no_incoming || no_outgoing {
            return true;
        }

        false
    }

    /// Returns true if the genome has any dead-end hidden neurons or isolated outputs.
    /// Mirrors C++ Genome::HasDeadEnds.
    pub fn has_dead_ends(&self) -> bool {
        // Any dead-end hidden neurons?
        for ng in &self.neuron_genes {
            if ng.neuron_type == crate::genes::NeuronType::Hidden {
                if self.is_dead_end_neuron(ng.id) {
                    return true;
                }
            }
        }

        // Special case: isolated outputs - outputs having one and only one looped recurrent connection or no connections at all
        for ng in &self.neuron_genes {
            if ng.neuron_type == crate::genes::NeuronType::Output {
                // count links connected to this output
                let mut conn_count = 0usize;
                let mut looped_recurrent_count = 0usize;
                for lg in &self.link_genes {
                    if lg.from_neuron_id == ng.id || lg.to_neuron_id == ng.id {
                        conn_count += 1;
                        if lg.from_neuron_id == lg.to_neuron_id && lg.is_recurrent {
                            looped_recurrent_count += 1;
                        }
                    }
                }

                if conn_count == 0 || (conn_count == 1 && looped_recurrent_count == 1) {
                    return true;
                }
            }
        }

        false
    }
}
