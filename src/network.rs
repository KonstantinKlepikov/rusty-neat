//! Phenotype module (neural network)

use crate::genes::{ActivationFunction, NeuronType};
use crate::parameters::Parameters;
use crate::utils::clamp_f64;
use rand::{Rng, rng};

const LEARNING_RATE: f64 = 0.0001;

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

    /// Save network to a file path. Port of C++ `Save(const char*)`.
    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        let f = File::create(filename)?;
        let mut bw = BufWriter::new(f);
        self.save_writer(&mut bw)
    }

    /// Save network to any writer (FILE* equivalent). Port of C++ `Save(FILE*)`.
    pub fn save_writer(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {

        // header and dims
        write!(w, "NNstart\n")?;
        write!(w, "{} {}\n", self.num_inputs, self.num_outputs)?;

        // neurons
        for n in &self.neurons {
            // neuron <type> <a> <b> <timeconst> <bias> <activation_function> <split_y>
            write!(
                w,
                "neuron {} {:#.18} {:#.18} {:#.18} {:#.18} {} {:#.18}\n",
                n.neuron_type as i32,
                n.a,
                n.b,
                n.timeconst,
                n.bias,
                n.activation_function_type as i32,
                n.split_y
            )?;
        }

        // connections
        for c in &self.connections {
            // connection <from> <to> <weight> <isrecur> <hebb_rate> <hebb_pre_rate>
            write!(
                w,
                "connection {} {} {:#.18} {} {:#.18} {:#.18}\n",
                c.source_neuron_idx,
                c.target_neuron_idx,
                c.weight,
                if c.recur_flag { 1 } else { 0 },
                c.hebb_rate,
                c.hebb_pre_rate
            )?;
        }

        // footer
        write!(w, "NNend\n\n")?;
        Ok(())
    }

    /// Load network from a file path. Returns `true` on success, `false` otherwise.
    pub fn load(&mut self, filename: &str) -> bool {
        use std::fs::File;

        let mut f = match File::open(filename) {
            Ok(f) => f,
            Err(_) => return false,
        };
        self.load_reader(&mut f)
    }

    /// Load network from any reader (token stream). Returns `true` on success.
    pub fn load_reader(&mut self, r: &mut impl std::io::Read) -> bool {

        let mut s = String::new();
        if r.read_to_string(&mut s).is_err() {
            return false;
        }

        let mut toks = s.split_whitespace();

        // search for NNstart
        loop {
            if let Some(t) = toks.next() {
                if t == "NNstart" {
                    break;
                }
            } else {
                return false; // not found
            }
        }

        // read dims
        let num_inputs = toks.next().and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
        let num_outputs = toks.next().and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);

        // clear current network
        self.clear();
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;

        while let Some(tok) = toks.next() {
            if tok == "neuron" {
                // parse: type a b timeconst bias activation_function split_y
                let ntype_i = toks.next().and_then(|t| t.parse::<i32>().ok()).unwrap_or(0);
                let a = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(1.0);
                let b = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);
                let timeconst = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);
                let bias = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);
                let aftype_i = toks.next().and_then(|t| t.parse::<i32>().ok()).unwrap_or(0);
                let split_y = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);

                let neuron_type = match ntype_i {
                    0 => NeuronType::Input,
                    1 => NeuronType::Output,
                    2 => NeuronType::Hidden,
                    3 => NeuronType::Bias,
                    _ => NeuronType::Hidden,
                };

                let af = match aftype_i {
                    0 => ActivationFunction::SignedSigmoid,
                    1 => ActivationFunction::UnsignedSigmoid,
                    2 => ActivationFunction::Tanh,
                    3 => ActivationFunction::TanhCubic,
                    4 => ActivationFunction::SignedStep,
                    5 => ActivationFunction::UnsignedStep,
                    6 => ActivationFunction::SignedGauss,
                    7 => ActivationFunction::UnsignedGauss,
                    8 => ActivationFunction::Abs,
                    9 => ActivationFunction::SignedSine,
                    10 => ActivationFunction::UnsignedSine,
                    11 => ActivationFunction::Linear,
                    12 => ActivationFunction::Relu,
                    13 => ActivationFunction::Softplus,
                    _ => ActivationFunction::SignedSigmoid,
                };

                let mut n = Neuron::new(neuron_type, af);
                n.a = a;
                n.b = b;
                n.timeconst = timeconst;
                n.bias = bias;
                n.split_y = split_y;
                self.neurons.push(n);
            } else if tok == "connection" {
                // parse: from to weight isrecur hebb_rate hebb_pre_rate
                let from = toks.next().and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
                let to = toks.next().and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
                let weight = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);
                let isrecur = toks.next().and_then(|t| t.parse::<i32>().ok()).unwrap_or(0) != 0;
                let hebb_rate = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);
                let hebb_pre_rate = toks.next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(0.0);

                self.connections.push(Connection {
                    source_neuron_idx: from,
                    target_neuron_idx: to,
                    weight,
                    signal: 0.0,
                    recur_flag: isrecur,
                    hebb_rate,
                    hebb_pre_rate,
                });
            } else if tok == "NNend" {
                break;
            } else {
                // ignore unknown tokens
            }
        }

        true
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
            // C++ `Activate()` uses only the activesum (bias is applied in
            // ActivateUseInternalBias()). Keep `activate()` faithful to C++ `Activate()`.
            let x = n.activesum;
            n.activation = apply_activation(n.activation_function_type, x, n.a, n.b);
            // reset activesum for next cycle
            n.activesum = 0.0;
        }
    }

    /// Activate using each neuron's internal `bias` (matches C++ `ActivateUseInternalBias`).
    pub fn activate_use_internal_bias(&mut self) {
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

        // apply activation for non-input neurons, including internal bias
        for idx in self.num_inputs..self.neurons.len() {
            let n = &mut self.neurons[idx];
            let x = n.activesum + n.bias;
            n.activation = apply_activation(n.activation_function_type, x, n.a, n.b);
            // reset activesum for next cycle
            n.activesum = 0.0;
        }
    }

    /// Leaky integrator activation step. Matches C++ `ActivateLeaky(double)`.
    pub fn activate_leaky(&mut self, a_dtime: f64) {
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

        // Leaky integrator update of membrane potential
        for idx in self.num_inputs..self.neurons.len() {
            let n = &mut self.neurons[idx];
            let t_const = a_dtime / n.timeconst;
            n.membrane_potential = (1.0 - t_const) * n.membrane_potential + t_const * n.activesum;
        }

        // apply activation based on membrane potential + bias
        for idx in self.num_inputs..self.neurons.len() {
            let n = &mut self.neurons[idx];
            let x = n.membrane_potential + n.bias;
            n.activesum = 0.0;
            n.activation = apply_activation(n.activation_function_type, x, n.a, n.b);
        }
    }

    /// Hebbian learning adaptation. Port of C++ `NeuralNetwork::Adapt(Parameters&)`.
    pub fn adapt(&mut self, a_parameters: &Parameters) {
        // find max absolute magnitude of the weight
        let mut t_max_weight: f64 = 0.0;
        for c in &self.connections {
            let aw = c.weight.abs();
            if aw > t_max_weight {
                t_max_weight = aw;
            }
        }

        for c in &mut self.connections {
            let t_incoming_neuron_activation = self.neurons[c.source_neuron_idx].activation;
            let t_outgoing_neuron_activation = self.neurons[c.target_neuron_idx].activation;
            if c.weight > 0.0 {
                let t_delta = c.hebb_rate
                    * (t_max_weight - c.weight)
                    * t_incoming_neuron_activation
                    * t_outgoing_neuron_activation
                    + c.hebb_pre_rate * t_max_weight
                        * t_incoming_neuron_activation
                        * (t_outgoing_neuron_activation - 1.0);
                c.weight = c.weight + t_delta;
            } else if c.weight < 0.0 {
                let t_delta = c.hebb_pre_rate
                    * (t_max_weight - c.weight)
                    * t_incoming_neuron_activation
                    * (1.0 - t_outgoing_neuron_activation)
                    - c.hebb_rate * t_max_weight
                        * t_incoming_neuron_activation
                        * t_outgoing_neuron_activation;
                c.weight = -(c.weight + t_delta);
            }

            c.weight = clamp_f64(c.weight, -a_parameters.max_weight, a_parameters.max_weight);
        }
    }

    /// Check whether a connection from `a_from` to `a_to` exists.
    /// Returns `Some(index)` if found, otherwise `None`.
    pub fn connection_exists(&self, a_to: usize, a_from: usize) -> Option<usize> {
        for (i, c) in self.connections.iter().enumerate() {
            if c.source_neuron_idx == a_from && c.target_neuron_idx == a_to {
                return Some(i);
            }
        }
        None
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
        // Match C++: outputs are expected to be placed immediately after inputs.
        // Start at `num_inputs` and collect `num_outputs` activations. Do safe
        // bounds checks to avoid panics if the network is malformed.
        let start = self.num_inputs;
        let len = self.neurons.len();
        if start >= len {
            return out;
        }
        for i in 0..self.num_outputs {
            let idx = start + i;
            if idx < len {
                out.push(self.neurons[idx].activation);
            } else {
                // If the requested output index is out of bounds, push 0.0 as a
                // safe default to preserve vector length and avoid panics.
                out.push(0.0);
            }
        }
        out
    }

    /// Update RTRL sensitivity matrices (gradients) for every neuron/connection.
    /// Port of C++ `NeuralNetwork::RTRL_update_gradients()`.
    pub fn rtrl_update_gradients(&mut self) {
        let n = self.neurons.len();

        // helper derivatives expect activation value as input
        let unsigned_sigmoid_derivative = |x: f64| -> f64 { x * (1.0 - x) };
        let tanh_derivative = |x: f64| -> f64 { 1.0 - x * x };

        for k in self.num_inputs..n {
            for i in self.num_inputs..n {
                for j in 0..n {
                    if let Some(_conn_idx) = self.connection_exists(i, j) {
                        // determine derivative based on neuron's activation function
                        let t_derivative = match self.neurons[k].activation_function_type {
                            ActivationFunction::UnsignedSigmoid => {
                                unsigned_sigmoid_derivative(self.neurons[k].activation)
                            }
                            ActivationFunction::Tanh => {
                                tanh_derivative(self.neurons[k].activation)
                            }
                            _ => 0.0,
                        };

                        // compute the summation over l
                        let mut t_sum = 0.0;
                        for l in 0..n {
                            if let Some(t_l_idx) = self.connection_exists(k, l) {
                                t_sum += self.connections[t_l_idx].weight
                                    * self.neurons[l].sensitivity_matrix[i][j];
                            }
                        }

                        if i == k {
                            t_sum += self.neurons[j].activation;
                        }

                        self.neurons[k].sensitivity_matrix[i][j] = t_derivative * t_sum;
                    } else {
                        self.neurons[k].sensitivity_matrix[i][j] = 0.0;
                    }
                }
            }
        }
    }

    /// Use RTRL sensitivities to compute weight changes for the first output.
    /// Port of C++ `NeuralNetwork::RTRL_update_error(double a_target)`.
    pub fn rtrl_update_error(&mut self, a_target: f64) {
        // add to total error (first output assumed at index `num_inputs`)
        let out0 = self.output().get(0).cloned().unwrap_or(0.0);
        self._total_error = a_target - out0;

        let n = self.neurons.len();
        for i in 0..n {
            for j in 0..n {
                if let Some(t_idx) = self.connection_exists(i, j) {
                    let sens = self.neurons[self.num_inputs].sensitivity_matrix[i][j];
                    let t_delta = self._total_error * sens;
                    if t_idx < self.total_weight_change.len() {
                        self.total_weight_change[t_idx] += t_delta * LEARNING_RATE;
                    }
                }
            }
        }
    }

    /// Apply accumulated RTRL weight changes to connections and clear accumulators.
    /// Port of C++ `NeuralNetwork::RTRL_update_weights()`.
    pub fn rtrl_update_weights(&mut self) {
        for i in 0..self.connections.len() {
            // guard in case vectors are out of sync
            if i < self.total_weight_change.len() {
                self.connections[i].weight += self.total_weight_change[i];
                self.total_weight_change[i] = 0.0;
            }
        }
        self._total_error = 0.0;
    }

    /// Return a reference to the accumulated total weight changes (RTRL).
    /// Exposed for testing and inspection.
    pub fn total_weight_changes(&self) -> &[f64] {
        &self.total_weight_change
    }

    /// Mutable access to accumulated total weight changes for testing/inspection.
    pub fn total_weight_changes_mut(&mut self) -> &mut [f64] {
        &mut self.total_weight_change
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
