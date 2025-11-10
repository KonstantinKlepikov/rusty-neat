//! Genome and gene module (neurons, links)

// === Imports analogous to C++ standard and Boost libraries ===
use std::fs::File;
use std::io::{Read, BufReader, Result as IoResult};
use std::cmp::{PartialEq, PartialOrd, Ordering};
use crate::network::{NeuralNetwork, Neuron as PhNeuron, Connection as PhConnection};
use crate::hyperneat::Substrate;
use crate::genes::{TraitValue, NeuronGene, LinkGene, ActivationFunction};



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
}



