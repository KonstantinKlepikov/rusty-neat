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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomeSeedType {
    Perceptron = 0,
    Layered = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    Linear,
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

#[derive(Debug, Clone, PartialEq)]
pub struct NeuronGene {
    pub id: u64,
    pub neuron_type: NeuronType,
    pub activation: ActivationFunction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkGene {
    pub from: u64,
    pub to: u64,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    Input,
    Hidden,
    Output,
    Bias,
}
