//! Genome and gene module (neurons, links)

// === Imports analogous to C++ standard and Boost libraries ===
use std::collections::{VecDeque, HashMap};
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufWriter, Result as IoResult};
use std::cmp::{PartialEq, PartialOrd, Ordering};
use std::f64;
use std::ops::{Add, Sub};
use std::fmt;

// For random number generation (analogue to RNG/Random.h)
use rand::prelude::*;

// For graph operations (analogue to boost::graph)
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::Topo;
use petgraph::Direction;

// For smart pointers (analogue to boost::shared_ptr)
use std::rc::Rc;
use std::cell::RefCell;

// For statistics/accumulators (analogue to boost::accumulators)
// (In Rust, use itertools or custom code for variance, mean, etc.)

use itertools::Itertools;
use crate::network::{ActivationFunction, NeuralNetwork, Neuron as PhNeuron, Connection as PhConnection};
use crate::genes::{TraitValue, NeuronGene, LinkGene};



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

/// Placeholder for Gene struct (to be implemented)
#[derive(Debug, Clone, PartialEq)]
pub struct Gene;

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
    /// Traits that belong to the genome itself (placeholder for now)
    pub genome_gene: Option<Gene>,
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
        // Note: RTRL variables and advanced features are not implemented in this prototype
    }
}



