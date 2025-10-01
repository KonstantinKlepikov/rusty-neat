//! Population and evolution module

use crate::genome::Genome;
use crate::species::Species;
use crate::parameters::Parameters;

#[derive(Debug, Clone)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub generation: u64,
    pub parameters: Parameters,
}
