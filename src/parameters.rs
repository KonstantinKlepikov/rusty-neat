//! Evolution parameters module

#[derive(Debug, Clone)]
pub struct Parameters {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    // ... other parameters
}
