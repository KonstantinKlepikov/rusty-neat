//! Population and evolution module

use crate::genome::Genome;
use crate::parameters::Parameters;
use crate::species::Species;
use crate::innovation::InnovationDatabase;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub generation: u64,
    pub parameters: Parameters,
    /// Shared innovation database used by the whole population
    pub innovation_db: InnovationDatabase,
}

impl Population {
    /// Create a new empty population with given parameters
    pub fn new(parameters: Parameters) -> Self {
        // start ids from 1
        let innov_db = InnovationDatabase::new(1, 1);
        Population {
            genomes: Vec::new(),
            species: Vec::new(),
            generation: 0,
            parameters,
            innovation_db: innov_db,
        }
    }

    /// Create a population from existing genomes and parameters.
    /// The innovation DB will be initialised to continue after existing max ids.
    pub fn from_genomes(genomes: Vec<Genome>, parameters: Parameters) -> Self {
        // compute next ids from existing genomes
        let mut max_neuron_id: u64 = 0;
        let mut max_innov_id: u64 = 0;
        for g in &genomes {
            for n in &g.neuron_genes {
                if n.id > max_neuron_id { max_neuron_id = n.id; }
            }
            for l in &g.link_genes {
                if l.innovation_id > max_innov_id { max_innov_id = l.innovation_id; }
            }
        }
        // next ids should be one greater than max found (or 1 if none)
        let next_neuron = if max_neuron_id == 0 { 1 } else { max_neuron_id + 1 };
        let next_innov = if max_innov_id == 0 { 1 } else { max_innov_id + 1 };
        let innov_db = InnovationDatabase::new(next_neuron, next_innov);

        Population {
            genomes,
            species: Vec::new(),
            generation: 0,
            parameters,
            innovation_db: innov_db,
        }
    }

    /// Helper: apply mutate_add_neuron for genome at index `gidx` using the shared InnovationDatabase
    pub fn mutate_add_neuron_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() { return false; }
        self.genomes[gidx].mutate_add_neuron(&mut self.innovation_db, &self.parameters, rng)
    }

    /// Helper: apply mutate_add_link for genome at index `gidx` using the shared InnovationDatabase
    pub fn mutate_add_link_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() { return false; }
        self.genomes[gidx].mutate_add_link(&mut self.innovation_db, &self.parameters, rng)
    }

    /// Helper: apply mutate_remove_link for genome at index `gidx` using population parameters
    pub fn mutate_remove_link_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() { return false; }
        self.genomes[gidx].mutate_remove_link(&self.parameters, rng)
    }

    /// Helper: apply mutate_remove_simple_neuron for genome at index `gidx` using shared InnovationDatabase
    pub fn mutate_remove_simple_neuron_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() { return false; }
        self.genomes[gidx].mutate_remove_simple_neuron(&mut self.innovation_db, &self.parameters, rng)
    }
}
