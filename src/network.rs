//! Phenotype module (neural network)

use crate::genes::{ActivationFunction, NeuronType};
use rand::{Rng, rng};

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

impl Neuron {
    /// Create a new Neuron with provided type and activation function and reasonable defaults.
    pub fn new(neuron_type: NeuronType, activation_function_type: ActivationFunction) -> Self {
        Self {
            activesum: 0.0,
            activation: 0.0,
            a: 1.0,
            b: 0.0,
            timeconst: 0.0,
            bias: 0.0,
            membrane_potential: 0.0,
            activation_function_type,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            sx: 0.0,
            sy: 0.0,
            sz: 0.0,
            substrate_coords: Vec::new(),
            split_y: 0.0,
            neuron_type,
            sensitivity_matrix: Vec::new(),
        }
    }
}

impl Default for Neuron {
    fn default() -> Self {
        Neuron::new(NeuronType::Hidden, ActivationFunction::SignedSigmoid)
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

    /// Construct network optionally populated with a default XOR example (matches C++ behaviour when a_Minimal=false)
    pub fn with_minimal(a_minimal: bool) -> Self {
        let mut nn = Self::new();
        if !a_minimal {
            // build an XOR-like example network
            // inputs 0,1,2; output 3; hidden 4
            let i1 = Neuron::new(NeuronType::Input, ActivationFunction::SignedSigmoid);
            let i2 = Neuron::new(NeuronType::Input, ActivationFunction::SignedSigmoid);
            let i3 = Neuron::new(NeuronType::Input, ActivationFunction::SignedSigmoid);
            let o1 = Neuron::new(NeuronType::Output, ActivationFunction::SignedSigmoid);
            let h1 = Neuron::new(NeuronType::Hidden, ActivationFunction::SignedSigmoid);

            nn.neurons.push(i1);
            nn.neurons.push(i2);
            nn.neurons.push(i3);
            nn.neurons.push(o1);
            nn.neurons.push(h1);

            // connections: inputs -> output (3), inputs -> hidden (3), hidden -> output (1)
            let add_conn = |src: usize, tgt: usize, conns: &mut Vec<Connection>| {
                conns.push(Connection {
                    source_neuron_idx: src,
                    target_neuron_idx: tgt,
                    weight: 0.0,
                    signal: 0.0,
                    recur_flag: false,
                    hebb_rate: 0.0,
                    hebb_pre_rate: 0.0,
                });
            };

            add_conn(0, 3, &mut nn.connections);
            add_conn(1, 3, &mut nn.connections);
            add_conn(2, 3, &mut nn.connections);
            add_conn(0, 4, &mut nn.connections);
            add_conn(1, 4, &mut nn.connections);
            add_conn(2, 4, &mut nn.connections);
            add_conn(4, 3, &mut nn.connections);

            nn.num_inputs = 3;
            nn.num_outputs = 1;

            // randomize weights in [-0.5, 0.5]
            let mut rng = rng();
            for c in &mut nn.connections {
                c.weight = rng.random_range(-0.5..0.5);
            }

            // align neuron params with C++ clean-up defaults
            for n in &mut nn.neurons {
                n.a = 1.0;
                n.b = 0.0;
                n.timeconst = 0.0;
                n.bias = 0.0;
                n.membrane_potential = 0.0;
            }

            nn.init_rtrl_matrix();
        }

        nn
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

    /// Initialize the RTRL sensitivity matrices (cube) and related state.
    pub fn init_rtrl_matrix(&mut self) {
        let n = self.neurons.len();
        for i in 0..n {
            self.neurons[i].sensitivity_matrix = vec![vec![0.0; n]; n];
        }
        self.flush_cube();
        self._total_error = 0.0;
        self.total_weight_change = vec![0.0; self.connections.len()];
    }

    /// Zero out the sensitivity matrices (cube)
    pub fn flush_cube(&mut self) {
        for neuron in &mut self.neurons {
            for row in &mut neuron.sensitivity_matrix {
                for v in row.iter_mut() {
                    *v = 0.0;
                }
            }
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
        // compute each connection's signal first (like C++: m_signal = activation*weight)
        let neurons_len = self.neurons.len();
        for conn in &mut self.connections {
            if conn.source_neuron_idx < neurons_len && conn.target_neuron_idx < neurons_len {
                let src = self.neurons[conn.source_neuron_idx].activation;
                conn.signal = src * conn.weight;
            } else {
                panic!(
                    "Invalid connection indices: source={} target={} neurons_len={}",
                    conn.source_neuron_idx, conn.target_neuron_idx, neurons_len
                );
            }
        }

        // then add signals to target neurons' activesum
        // panic on invalid target indices (prefer explicit failure over silent skipping)
        for i in 0..self.connections.len() {
            let conn = &self.connections[i];
            if conn.target_neuron_idx >= neurons_len {
                panic!(
                    "Invalid connection target index: target={} neurons_len={}",
                    conn.target_neuron_idx, neurons_len
                );
            }
            self.neurons[conn.target_neuron_idx].activesum += conn.signal;
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

    /// Fast activation assuming unsigned sigmoid everywhere (no bias).
    pub fn activate_fast(&mut self) {
        // compute each connection's signal first
        let neurons_len = self.neurons.len();
        for conn in &mut self.connections {
            if conn.source_neuron_idx < neurons_len && conn.target_neuron_idx < neurons_len {
                let src = self.neurons[conn.source_neuron_idx].activation;
                conn.signal = src * conn.weight;
            } else {
                panic!(
                    "Invalid connection indices: source={} target={} neurons_len={}",
                    conn.source_neuron_idx, conn.target_neuron_idx, neurons_len
                );
            }
        }

        // add signals to target neurons' activesum
        for i in 0..self.connections.len() {
            let conn = &self.connections[i];
            if conn.target_neuron_idx >= neurons_len {
                panic!(
                    "Invalid connection target index: target={} neurons_len={}",
                    conn.target_neuron_idx, neurons_len
                );
            }
            self.neurons[conn.target_neuron_idx].activesum += conn.signal;
        }

        // apply unsigned sigmoid activation for non-input neurons (no bias)
        for idx in self.num_inputs..self.neurons.len() {
            let n = &mut self.neurons[idx];
            let x = n.activesum;
            n.activesum = 0.0;
            n.activation = apply_activation(ActivationFunction::UnsignedSigmoid, x, n.a, n.b);
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

fn apply_activation(ftype: ActivationFunction, x: f64, a: f64, b: f64) -> f64 {
    match ftype {
        ActivationFunction::SignedSigmoid => {
            let v = 1.0 / (1.0 + (-a * x - b).exp());
            2.0 * v - 1.0
        }
        ActivationFunction::UnsignedSigmoid => 1.0 / (1.0 + (-a * x - b).exp()),
        ActivationFunction::Tanh => (a * x).tanh(),
        ActivationFunction::TanhCubic => ((x * x * x) * a).tanh(),
        ActivationFunction::SignedStep => {
            if x > b {
                1.0
            } else {
                -1.0
            }
        }
        ActivationFunction::UnsignedStep => {
            if x > (0.5 + b) {
                1.0
            } else {
                0.0
            }
        }
        ActivationFunction::SignedGauss => {
            let t = (-a * x * x + b).exp();
            (t - 0.5) * 2.0
        }
        ActivationFunction::UnsignedGauss => (-a * x * x + b).exp(),
        ActivationFunction::Abs => (x + b).abs(),
        ActivationFunction::SignedSine => (x * a + b).sin(),
        ActivationFunction::UnsignedSine => {
            let t = (x * a + b).sin();
            (t + 1.0) / 2.0
        }
        ActivationFunction::Linear => x + b,
        ActivationFunction::Relu => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        ActivationFunction::Softplus => (1.0 + x.exp()).ln(),
    }
}
