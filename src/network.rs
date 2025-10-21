//! Phenotype module (neural network)

/// Activation function type (stub, expand as needed)
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

/// Neuron type (stub, expand as needed)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    Input,
    Output,
    Hidden,
    Bias,
}


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
        self.source_neuron_idx == other.source_neuron_idx &&
        self.target_neuron_idx == other.target_neuron_idx
    }
}

/// Neuron in the phenotype network
#[derive(Debug, Clone)]
pub struct Neuron {
    pub activesum: f64, // synaptic input
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
        self.neuron_type == other.neuron_type &&
        self.split_y == other.split_y &&
        self.activation_function_type == other.activation_function_type
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
    total_error: f64,
    total_weight_change: Vec<f64>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            neurons: Vec::new(),
            connections: Vec::new(),
            total_error: 0.0,
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
}
