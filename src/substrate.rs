//! Substrate module: lightweight Substrate used by HyperNEAT pipeline
use crate::genes::ActivationFunction;

/// Minimal Substrate structure carrying the fields required by the
/// BuildHyperNEATPhenotype port. This is intentionally a pragmatic,
/// lightweight representation to let the Genome -> HyperNEAT pipeline
/// compile and run basic examples. It mirrors the fields used in the
/// original C++ implementation.
#[derive(Debug, Clone)]
pub struct Substrate {
    pub input_coords: Vec<Vec<f64>>,
    pub output_coords: Vec<Vec<f64>>,
    pub hidden_coords: Vec<Vec<f64>>,

    pub leaky: bool,
    pub with_distance: bool,

    pub min_time_const: f64,
    pub max_time_const: f64,

    pub max_weight_and_bias: f64,

    pub output_nodes_activation: ActivationFunction,
    pub hidden_nodes_activation: ActivationFunction,

    /// Custom connectivity: each entry is [src_type, src_idx, dst_type, dst_idx]
    pub custom_connectivity: Vec<Vec<i32>>,
    pub custom_conn_obeys_flags: bool,

    /// Connectivity flags (allow/disallow certain link types)
    pub allow_input_hidden_links: bool,
    pub allow_input_output_links: bool,
    pub allow_hidden_hidden_links: bool,
    pub allow_hidden_output_links: bool,
    pub allow_output_hidden_links: bool,
    pub allow_output_output_links: bool,
    pub allow_looped_hidden_links: bool,
    pub allow_looped_output_links: bool,

    /// If true, query only weights (one output) otherwise first output is link
    pub query_weights_only: bool,
}

impl Substrate {
    pub fn new() -> Self {
        Self {
            input_coords: Vec::new(),
            output_coords: Vec::new(),
            hidden_coords: Vec::new(),
            leaky: false,
            with_distance: false,
            min_time_const: 0.1,
            max_time_const: 1.0,
            max_weight_and_bias: 5.0,
            output_nodes_activation: ActivationFunction::UnsignedSigmoid,
            hidden_nodes_activation: ActivationFunction::UnsignedSigmoid,
            custom_connectivity: Vec::new(),
            custom_conn_obeys_flags: true,
            allow_input_hidden_links: true,
            allow_input_output_links: true,
            allow_hidden_hidden_links: false,
            allow_hidden_output_links: true,
            allow_output_hidden_links: false,
            allow_output_output_links: false,
            allow_looped_hidden_links: false,
            allow_looped_output_links: false,
            query_weights_only: false,
        }
    }

    /// Create a Substrate with explicit input/hidden/output coordinates.
    /// Matches the C++ Substrate(inputs, hidden, outputs) constructor semantics
    /// (sets several flags suitable for explicit substrates).
    pub fn with_coords(
        input_coords: Vec<Vec<f64>>,
        hidden_coords: Vec<Vec<f64>>,
        output_coords: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            input_coords,
            output_coords,
            hidden_coords,
            leaky: false,
            with_distance: false,
            min_time_const: 0.1,
            max_time_const: 1.0,
            max_weight_and_bias: 5.0,
            output_nodes_activation: ActivationFunction::UnsignedSigmoid,
            hidden_nodes_activation: ActivationFunction::UnsignedSigmoid,
            custom_connectivity: Vec::new(),
            custom_conn_obeys_flags: false,
            allow_input_hidden_links: true,
            allow_input_output_links: false,
            allow_hidden_hidden_links: false,
            allow_hidden_output_links: true,
            allow_output_hidden_links: false,
            allow_output_output_links: false,
            allow_looped_hidden_links: false,
            allow_looped_output_links: false,
            query_weights_only: false,
        }
    }

    /// Replace neurons coordinates (inputs, hidden, outputs). Mirrors C++ SetNeurons.
    pub fn set_neurons(
        &mut self,
        input_coords: Vec<Vec<f64>>,
        hidden_coords: Vec<Vec<f64>>,
        output_coords: Vec<Vec<f64>>,
    ) {
        self.input_coords = input_coords;
        self.hidden_coords = hidden_coords;
        self.output_coords = output_coords;
    }

    /// Print substrate info to stderr (useful for debugging)
    pub fn print_info(&self) {
        eprintln!("Inputs: {}", self.input_coords.len());
        eprintln!("Hidden: {}", self.hidden_coords.len());
        eprintln!("Outputs: {}", self.output_coords.len());
        eprintln!("Dimensions: {}", self.get_min_cppn_inputs());
    }

    /// Maximum dimensionality across input/output/hidden coordinates
    pub fn get_max_dims(&self) -> usize {
        let mut md = 0usize;
        for p in &self.input_coords {
            md = md.max(p.len());
        }
        for p in &self.output_coords {
            md = md.max(p.len());
        }
        for p in &self.hidden_coords {
            md = md.max(p.len());
        }
        md
    }

    /// Minimal number of CPPN inputs expected (heuristic: 2*max_dims + optional distance + bias)
    pub fn get_min_cppn_inputs(&self) -> usize {
        // 2 * max_dims + (with_distance ? 1 : 0) + 1 (bias)
        let md = self.get_max_dims();
        2 * md + (if self.with_distance { 1 } else { 0 }) + 1
    }

    /// Minimal number of CPPN outputs (if leaky substrate we need two extra outputs)
    pub fn get_min_cppn_outputs(&self) -> usize {
        let mut outs = if self.query_weights_only { 1 } else { 2 };
        if self.leaky {
            outs += 2;
        }
        outs
    }

    /// Set custom connectivity from a list of [src_type, src_idx, dst_type, dst_idx]
    pub fn set_custom_connectivity(&mut self, conns: Vec<Vec<i32>>) {
        self.custom_connectivity = conns;
    }

    pub fn clear_custom_connectivity(&mut self) {
        self.custom_connectivity.clear();
    }
}
