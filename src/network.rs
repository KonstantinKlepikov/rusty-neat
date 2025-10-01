//! Phenotype module (neural network)

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub neurons: Vec<Neuron>,
    pub links: Vec<Link>,
    pub input_indices: Vec<usize>,
    pub output_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: u64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct Link {
    pub from: u64,
    pub to: u64,
    pub weight: f64,
    pub enabled: bool,
}
