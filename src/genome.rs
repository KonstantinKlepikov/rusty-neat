//! Genome and gene module (neurons, links)

// === Imports analogous to C++ standard and Boost libraries ===
use std::collections::{VecDeque, HashMap};
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufWriter};
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

#[derive(Debug, Clone, PartialEq)]
pub struct Genome {
    pub id: u64,
    pub neuron_genes: Vec<NeuronGene>,
    pub link_genes: Vec<LinkGene>,
    pub fitness: f64,
    pub evaluated: bool,
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
