//! Phenotype module (neural network)

use crate::genes::{ActivationFunction, NeuronType};

/// Connection between neurons (phenotype)
#[derive(Debug, Clone)]
pub struct Connection {
    pub source_neuron_idx: usize, // index of source neuron
    pub target_neuron_idx: usize, // index of target neuron
    pub weight: f64,              // weight of the connection
    pub signal: f64,              // weight * input signal
    pub recur_flag: bool,         // recurrence flag (for display)
    // Hebbian learning parameters
    // Ignored in case there is no lifetime learning
    pub hebb_rate: f64,
    pub hebb_pre_rate: f64,
}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        // comparison operator (nessesary for boost::python)
        self.source_neuron_idx == other.source_neuron_idx
            && self.target_neuron_idx == other.target_neuron_idx
    }
}

/// Neuron in the phenotype network
#[derive(Debug, Clone)]
pub struct Neuron {
    pub activesum: f64,  // synaptic input
    pub activation: f64, // output after activation function
    // misc parameters
    pub a: f64,
    pub b: f64,
    pub timeconst: f64,
    pub bias: f64,
    pub membrane_potential: f64, // for leaky integrator mode
    pub activation_function_type: ActivationFunction,
    // Display/geometry
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sx: f64,
    pub sy: f64,
    pub sz: f64,
    pub substrate_coords: Vec<f64>,
    pub split_y: f64,
    pub neuron_type: NeuronType,
    // Sensitivity matrix for RTRL learning
    pub sensitivity_matrix: Vec<Vec<f64>>,
}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.neuron_type == other.neuron_type
            && self.split_y == other.split_y
            && self.activation_function_type == other.activation_function_type
    }
}

/// Phenotype neural network
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub neurons: Vec<Neuron>,
    pub connections: Vec<Connection>, // array size - number of connections
    // RTRL variables
    _total_error: f64,
    total_weight_change: Vec<f64>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            neurons: Vec::new(),
            connections: Vec::new(),
            _total_error: 0.0,
            total_weight_change: Vec::new(),
        }
    }

    pub fn add_neuron(&mut self, neuron: Neuron) {
        self.neurons.push(neuron);
    }

    pub fn add_connection(&mut self, conn: Connection) {
        self.connections.push(conn);
    }

    pub fn set_input_output_dimensions(&mut self, num_inputs: usize, num_outputs: usize) {
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn clear(&mut self) {
        self.neurons.clear();
        self.connections.clear();
        self.total_weight_change.clear();
        self.num_inputs = 0;
        self.num_outputs = 0;
    }

    /// Reset neuron activations / activesums (similar to Flush in C++ impl)
    pub fn flush(&mut self) {
        for n in &mut self.neurons {
            n.activesum = 0.0;
            n.activation = 0.0;
            n.membrane_potential = 0.0;
        }
    }

    /// Provide inputs to the network (fills first `num_inputs` neurons' activation)
    pub fn input(&mut self, inputs: Vec<f64>) {
        let take = usize::min(inputs.len(), self.num_inputs);
        for i in 0..take {
            if i < self.neurons.len() {
                self.neurons[i].activation = inputs[i];
            }
        }
    }

    /// Single activation step: propagate signals through connections and apply activation functions
    pub fn activate(&mut self) {
        // accumulate inputs
        for conn in &self.connections {
            if conn.source_neuron_idx < self.neurons.len()
                && conn.target_neuron_idx < self.neurons.len()
            {
                let src = self.neurons[conn.source_neuron_idx].activation;
                self.neurons[conn.target_neuron_idx].activesum += src * conn.weight;
            }
        }

        // apply activation for non-input neurons
        for idx in self.num_inputs..self.neurons.len() {
            let n = &mut self.neurons[idx];
            let x = n.activesum + n.bias;
            n.activation = apply_activation(n.activation_function_type, x, n.a, n.b);
            // reset activesum for next cycle
            n.activesum = 0.0;
        }
    }

    /// Return outputs (last `num_outputs` neurons are treated as outputs)
    pub fn output(&self) -> Vec<f64> {
        let mut out = Vec::new();
        if self.num_outputs == 0 {
            return out;
        }
        // assume outputs are placed after inputs; original ordering depends on genome
        // Here we pick the last `num_outputs` neurons
        let start = self.neurons.len().saturating_sub(self.num_outputs);
        for i in start..self.neurons.len() {
            out.push(self.neurons[i].activation);
        }
        out
    }
}

fn apply_activation(ftype: crate::genes::ActivationFunction, x: f64, _a: f64, _b: f64) -> f64 {
    match ftype {
        crate::genes::ActivationFunction::SignedSigmoid => {
            // classic NEAT signed sigmoid approximation
            let v = 1.0 / (1.0 + (-4.924273 * x).exp());
            2.0 * v - 1.0
        }
        crate::genes::ActivationFunction::UnsignedSigmoid => 1.0 / (1.0 + (-4.924273 * x).exp()),
        crate::genes::ActivationFunction::Tanh => x.tanh(),
        crate::genes::ActivationFunction::Linear => x,
        crate::genes::ActivationFunction::Relu => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        crate::genes::ActivationFunction::Softplus => (1.0 + x.exp()).ln(),
        // fallback to tanh for other types for now
        _ => x.tanh(),
    }
}
