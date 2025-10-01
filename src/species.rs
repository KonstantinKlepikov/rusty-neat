//! Species module

#[derive(Debug, Clone)]
pub struct Species {
    pub id: u64,
    pub members: Vec<u64>, // genome ids
    pub age: u64,
    pub best_fitness: f64,
}
