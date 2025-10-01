//! Genome and gene module (neurons, links)

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    Linear,
}
