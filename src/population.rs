//! Population and evolution module

use crate::genome::{Genome, PhenotypeBehavior};
use crate::innovation::InnovationDatabase;
use crate::parameters::Parameters;
use crate::species::Species;
use rand::Rng;
use std::fs::File;
use std::io::{Read, Result as IoResult};

/// Search mode for phased searching (parity with C++ SearchMode)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Complexifying,
    Simplifying,
    Blended,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub generation: u64,
    pub parameters: Parameters,
    /// Shared innovation database used by the whole population
    pub innovation_db: InnovationDatabase,
    // --- additional state fields ported from C++ ---
    next_genome_id: u64,
    next_species_id: u64,
    /// Search mode for phased search. Kept for parity with the C++ codebase.
    /// Currently unused in Rust implementation but preserved for future features.
    #[allow(dead_code)]
    search_mode: SearchMode,
    /// Current maximum phenotypic complexity used by phased search (C++ parity).
    #[allow(dead_code)]
    current_mpc: f64,
    /// Previous MPC value (for tracking transitions). Preserved from C++.
    #[allow(dead_code)]
    old_mpc: f64,
    /// Base MPC value used as a reference. Preserved for compatibility.
    #[allow(dead_code)]
    base_mpc: f64,
    /// Best fitness seen in the population so far.
    best_fitness_ever: f64,
    /// Best genome in the current generation (may be used by reporters).
    best_genome: Option<Genome>,
    /// Historical best genome across generations. Kept for compatibility.
    #[allow(dead_code)]
    best_genome_ever: Option<Genome>,
    /// Counters tracking when best fitness last changed. Present for diagnostics.
    #[allow(dead_code)]
    gens_since_best_fitness_last_changed: u64,
    #[allow(dead_code)]
    evals_since_best_fitness_last_changed: u64,
    /// Counter tracking when MPC last changed. Kept for compatibility/diagnostics.
    #[allow(dead_code)]
    gens_since_mpc_last_changed: u64,
    /// Total number of evaluations performed by the population (used by novelty search path).
    num_evaluations: u64,
    // Novelty search archive and counters
    behavior_archive: Vec<PhenotypeBehavior>,
    gens_since_last_archiving: u32,
    quick_add_counter: u32,
    /// Optional population identifier (unused in current code).
    #[allow(dead_code)]
    id: i32,
    /// Archive of genomes (preserved for C++ parity, currently unused).
    #[allow(dead_code)]
    genome_archive: Vec<Genome>,
    /// Temporary species storage used by some C++ code paths; kept for compatibility.
    #[allow(dead_code)]
    temp_species: Vec<Species>,
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
            next_genome_id: 1,
            next_species_id: 1,
            search_mode: SearchMode::Blended,
            current_mpc: 0.0,
            old_mpc: 0.0,
            base_mpc: 0.0,
            best_fitness_ever: 0.0,
            best_genome: None,
            best_genome_ever: None,
            gens_since_best_fitness_last_changed: 0,
            evals_since_best_fitness_last_changed: 0,
            gens_since_mpc_last_changed: 0,
            num_evaluations: 0,
            behavior_archive: Vec::new(),
            gens_since_last_archiving: 0,
            quick_add_counter: 0,
            id: 0,
            genome_archive: Vec::new(),
            temp_species: Vec::new(),
        }
    }

    /// Create a population from existing genomes and parameters.
    /// The innovation DB will be initialised to continue after existing max ids.
    pub fn from_genomes(genomes: Vec<Genome>, parameters: Parameters) -> Self {
        // compute next ids from existing genomes
        let mut max_neuron_id: u64 = 0;
        let mut max_innov_id: u64 = 0;
        for g in &genomes {
            for n in g.neuron_genes() {
                if n.id > max_neuron_id {
                    max_neuron_id = n.id;
                }
            }
            for l in g.link_genes() {
                if l.innovation_id > max_innov_id {
                    max_innov_id = l.innovation_id;
                }
            }
        }
        // next ids should be one greater than max found (or 1 if none)
        let next_neuron = if max_neuron_id == 0 {
            1
        } else {
            max_neuron_id + 1
        };
        let next_innov = if max_innov_id == 0 {
            1
        } else {
            max_innov_id + 1
        };

        // Create innovation DB starting after existing ids and populate it
        // with all link innovations present in the provided genomes. This
        // mirrors C++ behavior where the database is initialized from a
        // genome so that subsequent mutations reuse existing innovations.
        let mut innov_db = InnovationDatabase::new(next_neuron, next_innov);
        // collect all link genes from genomes to seed the innovation DB
        let mut all_links: Vec<crate::genes::LinkGene> = Vec::new();
        for g in &genomes {
            all_links.extend(g.link_genes().clone());
        }
        innov_db.init_from_genome(&all_links, next_neuron, next_innov);

        // default ids: next genome id = number of genomes + 1
        let next_genome_id = (genomes.len() as u64).saturating_add(1);

        Population {
            genomes,
            species: Vec::new(),
            generation: 0,
            parameters,
            innovation_db: innov_db,
            next_genome_id,
            next_species_id: 1,
            search_mode: SearchMode::Blended,
            current_mpc: 0.0,
            old_mpc: 0.0,
            base_mpc: 0.0,
            best_fitness_ever: 0.0,
            best_genome: None,
            best_genome_ever: None,
            gens_since_best_fitness_last_changed: 0,
            evals_since_best_fitness_last_changed: 0,
            gens_since_mpc_last_changed: 0,
            num_evaluations: 0,
            behavior_archive: Vec::new(),
            gens_since_last_archiving: 0,
            quick_add_counter: 0,
            id: 0,
            genome_archive: Vec::new(),
            temp_species: Vec::new(),
        }
    }

    /// Helper: apply mutate_add_neuron for genome at index `gidx` using the shared InnovationDatabase
    pub fn mutate_add_neuron_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() {
            return false;
        }
        self.genomes[gidx].mutate_add_neuron(&mut self.innovation_db, &self.parameters, rng)
    }

    /// Helper: apply mutate_add_link for genome at index `gidx` using the shared InnovationDatabase
    pub fn mutate_add_link_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() {
            return false;
        }
        self.genomes[gidx].mutate_add_link(&mut self.innovation_db, &self.parameters, rng)
    }

    /// Helper: apply mutate_remove_link for genome at index `gidx` using population parameters
    pub fn mutate_remove_link_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() {
            return false;
        }
        self.genomes[gidx].mutate_remove_link(&self.parameters, rng)
    }

    /// Helper: apply mutate_remove_simple_neuron for genome at index `gidx` using shared InnovationDatabase
    pub fn mutate_remove_simple_neuron_for(&mut self, gidx: usize, rng: &mut impl Rng) -> bool {
        if gidx >= self.genomes.len() {
            return false;
        }
        self.genomes[gidx].mutate_remove_simple_neuron(
            &mut self.innovation_db,
            &self.parameters,
            rng,
        )
    }

    /// Separate the population into species based on a simple compatibility metric.
    /// This is a lightweight port of the C++ `Speciate()` behaviour: it ensures
    /// each genome has a unique id, then groups genomes by comparing a simple
    /// distance (difference in neuron count + link count) against
    /// `Parameters.compat_treshold`.
    pub fn speciate(&mut self) {
        self.species.clear();

        if self.genomes.is_empty() {
            return;
        }

        // ensure all genomes have ids
        for g in self.genomes.iter_mut() {
            if g.get_id() == 0 {
                g.set_id(self.next_genome_id);
                self.next_genome_id = self.next_genome_id.saturating_add(1);
            }
        }

        // helper: we'll look up genomes directly when needed

        // iterate genomes and assign to species
        for g in &self.genomes {
            // build a snapshot of current representatives to avoid borrow conflicts
            let reps: Vec<Genome> = self
                .species
                .iter()
                .filter_map(|s| {
                    s.members
                        .first()
                        .and_then(|rid| self.genomes.iter().find(|gg| gg.get_id() == *rid).cloned())
                })
                .collect();

            let mut added = false;
            for (idx, rep) in reps.iter().enumerate() {
                // compute simple compatibility distance
                let neurons_a = g.neuron_genes().len() as f64;
                let neurons_b = rep.neuron_genes().len() as f64;
                let links_a = g.link_genes().len() as f64;
                let links_b = rep.link_genes().len() as f64;

                let mut dist = (neurons_a - neurons_b).abs() + (links_a - links_b).abs();

                // optional normalization
                if self.parameters.normalize_genome_size {
                    let norm = (neurons_a.max(neurons_b) + links_a.max(links_b)).max(1.0);
                    dist /= norm;
                }

                if dist <= self.parameters.compat_treshold {
                    // push into actual species slot (idx corresponds)
                    if let Some(s) = self.species.get_mut(idx) {
                        s.members.push(g.get_id());
                    }
                    added = true;
                    break;
                }
            }

            if !added {
                // create new species
                let sid = self.next_species_id;
                self.next_species_id = self.next_species_id.saturating_add(1);
                let mut s = Species {
                    id: sid,
                    members: Vec::new(),
                    age: 0,
                    best_fitness: g.get_fitness(),
                };
                s.members.push(g.get_id());
                self.species.push(s);
            }
        }
    }

    /// Adjust fitness of all genomes using fitness sharing per-species.
    /// Basic port of C++/r1 logic: floor non-positive fitness to small
    /// positive value and divide by species size (sharing).
    pub fn adjust_fitness(&mut self) {
        if self.genomes.is_empty() || self.species.is_empty() {
            return;
        }

        // iterate species and apply sharing
        for s in self.species.iter_mut() {
            // collect indices of genomes that belong to this species
            let mut indices: Vec<usize> = Vec::new();
            for gid in &s.members {
                if let Some(pos) = self.genomes.iter().position(|g| g.get_id() == *gid) {
                    indices.push(pos);
                }
            }

            let size = indices.len() as f64;
            if size == 0.0 {
                continue;
            }

            // sort indices by fitness descending so we can find species best
            indices.sort_by(|&a, &b| {
                self.genomes[b]
                    .get_fitness()
                    .partial_cmp(&self.genomes[a].get_fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // update species best and population-wide best if applicable
            if let Some(&best_idx) = indices.first() {
                let best_fit = self.genomes[best_idx].get_fitness();
                if best_fit > s.best_fitness {
                    s.best_fitness = best_fit;
                }
                if best_fit > self.best_fitness_ever {
                    self.best_fitness_ever = best_fit;
                    self.best_genome = Some(self.genomes[best_idx].clone());
                }
            }

            // apply floor + sharing
            for &idx in &indices {
                let mut t_fitness = self.genomes[idx].get_fitness();
                if t_fitness <= 0.0 || t_fitness.is_nan() || t_fitness.is_infinite() {
                    t_fitness = 0.0000000001_f64;
                }
                let adj = t_fitness / size;
                self.genomes[idx].set_adj_fitness(adj);
            }
        }
    }

    /// Run one generation: count offspring, reproduce, and create new population.
    /// Uses external RNG passed as `rng` (design choice: RNG is external to Population).
    pub fn epoch(&mut self, rng: &mut impl Rng) {
        if self.genomes.is_empty() {
            // Even if population is empty, advance generation counter so
            // bindings/tests that expect epoch/tick to progress still behave
            // sensibly.
            self.generation = self.generation.saturating_add(1);
            return;
        }

        // Ensure species exist (speciate if needed)
        if self.species.is_empty() {
            self.speciate();
        }

        // Adjust fitness (sharing + basic floors)
        self.adjust_fitness();

        // Update global best genome tracking
        if let Some(best) = self.genomes.iter().max_by(|a, b| {
            a.get_fitness()
                .partial_cmp(&b.get_fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if best.get_fitness() > self.best_fitness_ever {
                self.best_fitness_ever = best.get_fitness();
                self.best_genome = Some(best.clone());
                self.gens_since_best_fitness_last_changed = 0;
            } else {
                self.gens_since_best_fitness_last_changed =
                    self.gens_since_best_fitness_last_changed.saturating_add(1);
            }
        }

        // helper: rounding to nearest integer (0.5 up)
        fn round_nearest(v: f64) -> i64 {
            let i = v as i64;
            if (v - i as f64) < 0.5 { i } else { i + 1 }
        }

        let target_pop = self.parameters.population_size as usize;

        // total adjusted fitness
        let total_adjusted: f64 = self.genomes.iter().map(|g| g.get_adj_fitness()).sum();
        let avg_adjusted = if total_adjusted > 0.0 {
            total_adjusted / target_pop as f64
        } else {
            1.0
        };

        // compute offspring counts per species
        let mut offspring_counts: Vec<usize> = Vec::with_capacity(self.species.len());
        let mut total_offspring = 0usize;

        for s in &self.species {
            let mut species_sum = 0.0f64;
            for gid in &s.members {
                if let Some(g) = self.genomes.iter().find(|gg| gg.get_id() == *gid) {
                    species_sum += g.get_adj_fitness();
                }
            }
            let share = if avg_adjusted > 0.0 {
                species_sum / avg_adjusted
            } else {
                0.0
            };
            let cnt = round_nearest(share).max(0) as usize;
            offspring_counts.push(cnt);
            total_offspring += cnt;
        }

        // adjust counts down if too many
        while total_offspring > target_pop && offspring_counts.len() > 1 {
            if let Some((max_idx, _)) = offspring_counts.iter().enumerate().max_by_key(|(_, c)| *c)
            {
                if offspring_counts[max_idx] > 1 {
                    offspring_counts[max_idx] -= 1;
                    total_offspring -= 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // increase if too few: favor species with highest best_fitness
        while total_offspring < target_pop && !self.species.is_empty() {
            if let Some(best_idx) = self
                .species
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.best_fitness
                        .partial_cmp(&b.best_fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                offspring_counts[best_idx] += 1;
                total_offspring += 1;
            } else {
                break;
            }
        }

        // Collect offspring
        let mut all_offspring: Vec<Genome> = Vec::new();

        for (sidx, s) in self.species.iter().enumerate() {
            let count = *offspring_counts.get(sidx).unwrap_or(&0);
            if count == 0 {
                continue;
            }

            // collect individuals for this species
            let mut individuals: Vec<Genome> = s
                .members
                .iter()
                .filter_map(|gid| self.genomes.iter().find(|gg| gg.get_id() == *gid).cloned())
                .collect();

            if individuals.is_empty() {
                continue;
            }

            // sort by fitness desc
            individuals.sort_by(|a, b| {
                b.get_fitness()
                    .partial_cmp(&a.get_fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // elitism: copy best as first child if count>0
            if count > 0 {
                let elite = individuals[0].clone();
                all_offspring.push(elite);
            }

            // determine truncated parent pool
            let num_parents = ((self.parameters.survival_rate * individuals.len() as f64) as usize)
                .max(1)
                .min(individuals.len());

            // generate exactly count-1 (since elite already added) or count if elite not used
            let to_create = if count > 0 {
                count.saturating_sub(1)
            } else {
                0
            };
            for _ in 0..to_create {
                // select a parent from top `num_parents`
                let pidx = if num_parents <= 1 {
                    0
                } else {
                    rng.random_range(0..num_parents)
                };
                let mut child = individuals[pidx].clone();

                // Mutations: attempt a set of mutations based on params
                if rng.random::<f64>() < self.parameters.mutate_add_neuron_prob {
                    let _ = child.mutate_add_neuron(&mut self.innovation_db, &self.parameters, rng);
                }
                if rng.random::<f64>() < self.parameters.mutate_add_link_prob {
                    let _ = child.mutate_add_link(&mut self.innovation_db, &self.parameters, rng);
                }
                if rng.random::<f64>() < self.parameters.mutate_rem_link_prob {
                    let _ = child.mutate_remove_link(&self.parameters, rng);
                }
                if rng.random::<f64>() < self.parameters.mutate_rem_simple_neuron_prob {
                    let _ = child.mutate_remove_simple_neuron(
                        &mut self.innovation_db,
                        &self.parameters,
                        rng,
                    );
                }
                if rng.random::<f64>() < self.parameters.mutate_weights_prob {
                    let _ = child.mutate_link_weights(&self.parameters, rng);
                }

                all_offspring.push(child);
            }
        }

        // If offspring count mismatched (due to earlier generation logic), trim or fill with clones of best
        if all_offspring.len() > target_pop {
            all_offspring.truncate(target_pop);
        }
        while all_offspring.len() < target_pop {
            // clone global best or last genome
            if let Some(ref best) = self.best_genome {
                all_offspring.push(best.clone());
            } else if let Some(g) = self.genomes.last() {
                all_offspring.push(g.clone());
            } else {
                break;
            }
        }

        // assign new ids
        for child in &mut all_offspring {
            child.set_id(self.next_genome_id);
            self.next_genome_id = self.next_genome_id.saturating_add(1);
        }

        // replace genomes and re-speciate
        self.genomes = all_offspring;
        self.species.clear();
        self.speciate();

        // flush innovation DB (keep counters)
        self.innovation_db.flush();

        // advance generation
        self.generation = self.generation.saturating_add(1);
    }

    /// Compute sparseness of `genome` relative to current population + archive.
    pub fn compute_sparseness(&self, genome: &Genome) -> f64 {
        use std::cmp::Ordering;

        // need a phenotype behavior on the genome
        let pb_opt = genome.get_phenotype_behavior();
        if pb_opt.is_none() {
            return 0.0;
        }
        let pb = pb_opt.unwrap();

        let mut distances: Vec<f64> = Vec::new();

        // distances to all individuals in population
        for s in &self.species {
            for gid in &s.members {
                if let Some(g) = self.genomes.iter().find(|gg| gg.get_id() == *gid) {
                    if let Some(ref other_pb) = g.get_phenotype_behavior() {
                        distances.push(pb.distance_to(other_pb));
                    }
                }
            }
        }

        // distances to archive
        for arch in &self.behavior_archive {
            distances.push(pb.distance_to(arch));
        }

        if distances.is_empty() {
            return 0.0;
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let k = (self.parameters.novelty_search_k as usize).min(distances.len().saturating_sub(1));
        if k == 0 {
            // no neighbors other than possibly itself
            return distances.iter().copied().sum::<f64>() / (distances.len() as f64);
        }

        // C++ sums distances[1..k] (skipping index 0 which may be self-distance)
        let mut sum = 0.0;
        // start from 1, but ensure we don't overflow
        let start = 1.min(distances.len() - 1);
        let end = (start + k).min(distances.len());
        for v in &distances[start..end] {
            sum += *v;
        }
        sum / (k as f64)
    }

    /// Novelty search tick: perform one realtime reproduction step and
    /// evaluate novelty. Returns the new baby genome if created.
    pub fn novelty_search_tick(&mut self, rng: &mut impl Rng) -> Option<Genome> {
        // Recompute sparseness/fitness for all individuals periodically
        if self.num_evaluations % (self.parameters.novelty_search_recompute_sparseness_each as u64)
            == 0
        {
            for s in &self.species {
                for gid in &s.members {
                    if let Some(pos) = self.genomes.iter().position(|g| g.get_id() == *gid) {
                        let spar = self.compute_sparseness(&self.genomes[pos].clone());
                        if let Some(g) = self.genomes.get_mut(pos) {
                            g.set_fitness(spar);
                        }
                    }
                }
            }
        }

        if self.genomes.is_empty() {
            return None;
        }

        // pick parent species and parent, create child (similar to `tick` logic)
        let sidx = self.choose_parent_species(rng)?;
        if self.species.is_empty() || sidx >= self.species.len() {
            return None;
        }

        // collect individuals in species
        let mut individuals: Vec<Genome> = self.species[sidx]
            .members
            .iter()
            .filter_map(|gid| self.genomes.iter().find(|gg| gg.get_id() == *gid).cloned())
            .collect();
        if individuals.is_empty() {
            return None;
        }

        // sort and pick parent via truncated survival_rate
        individuals.sort_by(|a, b| {
            b.get_fitness()
                .partial_cmp(&a.get_fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let num_parents = ((self.parameters.survival_rate * individuals.len() as f64) as usize)
            .max(1)
            .min(individuals.len());
        let pidx = if num_parents <= 1 {
            0
        } else {
            rng.random_range(0..num_parents)
        };
        let mut child = individuals[pidx].clone();

        // perform mutations
        if rng.random::<f64>() < self.parameters.overall_mutation_rate {
            let _ = child.mutate_link_weights(&self.parameters, rng);
        }
        if rng.random::<f64>() < self.parameters.mutate_add_neuron_prob {
            let _ = child.mutate_add_neuron(&mut self.innovation_db, &self.parameters, rng);
        }

        // remove worst from population
        let removed = self.remove_worst_individual()?;

        // transfer phenotype behavior from removed to child
        child.set_id(removed.get_id());
        let mut removed = removed;
        let taken_pb = removed.take_phenotype_behavior();
        child.set_phenotype_behavior(taken_pb);

        // insert child and reassign species
        self.genomes.push(child.clone());
        let new_idx = self.genomes.len() - 1;
        self.reassign_species(new_idx);

        // Acquire behavior from the child (if it has a phenotype_behavior)
        let mut acquired = false;
        // safely take the phenotype behavior out, call acquire, then put it back
        let pb_opt = {
            let g = &mut self.genomes[new_idx];
            g.take_phenotype_behavior()
        };
        if let Some(mut pb) = pb_opt {
            let g_clone = self.genomes[new_idx].clone();
            acquired = pb.acquire(&g_clone);
            // restore behavior into genome
            self.genomes[new_idx].set_phenotype_behavior(Some(pb));
        }

        // if successful behavior found, return it early
        if acquired {
            return Some(self.genomes[new_idx].clone());
        }

        // compute sparseness for the new baby
        let sparseness = self.compute_sparseness(&self.genomes[new_idx].clone());

        // decide whether to add to archive
        self.gens_since_last_archiving = self.gens_since_last_archiving.saturating_add(1);
        if sparseness > self.parameters.novelty_search_p_min {
            // simple uniqueness check (exact data match) optionally expensive
            let mut present = false;
            if let Some(pb) = self.genomes[new_idx].get_phenotype_behavior() {
                for a in &self.behavior_archive {
                    if a.m_data == pb.m_data {
                        present = true;
                        break;
                    }
                }
            }
            if !present {
                if let Some(pb) = self.genomes[new_idx].get_phenotype_behavior() {
                    self.behavior_archive.push(pb.clone());
                    self.gens_since_last_archiving = 0;
                    self.quick_add_counter = self.quick_add_counter.saturating_add(1);
                }
            }
        } else {
            self.quick_add_counter = 0;
        }

        // dynamic Pmin adjustment
        if self.parameters.novelty_search_dynamic_pmin {
            if self.gens_since_last_archiving
                > self
                    .parameters
                    .novelty_search_no_archiving_stagnation_treshold
            {
                // lower Pmin
                // mutate parameters in-place (mirror C++ behaviour)
                // NOTE: altering Parameters here mutates the working copy in Population
                self.parameters.novelty_search_p_min *=
                    self.parameters.novelty_search_pmin_lowering_multiplier;
                if self.parameters.novelty_search_p_min < self.parameters.novelty_search_pmin_min {
                    self.parameters.novelty_search_p_min = self.parameters.novelty_search_pmin_min;
                }
            }
            if self.quick_add_counter
                > self
                    .parameters
                    .novelty_search_quick_archiving_min_evaluations
            {
                self.parameters.novelty_search_p_min *=
                    self.parameters.novelty_search_pmin_raising_multiplier;
            }
        }

        // set fitness = sparseness for new baby
        if let Some(g) = self.genomes.get_mut(new_idx) {
            g.set_fitness(sparseness);
        }

        Some(self.genomes[new_idx].clone())
    }

    /// Add a behavior to the novelty search archive (test / API helper)
    pub fn add_behavior_archive(&mut self, pb: PhenotypeBehavior) {
        self.behavior_archive.push(pb);
    }

    /// Return number of behaviors stored in archive (test helper)
    pub fn behavior_archive_len(&self) -> usize {
        self.behavior_archive.len()
    }

    /// Save population to a single file. The format contains Parameters,
    /// InnovationDatabase and then repeated Genome blocks. This mirrors the
    /// original C++ `Population::Save` behaviour (text-based sections).
    pub fn save(&self, filename: &str) -> IoResult<()> {
        let mut f = File::create(filename)?;
        // write parameters
        self.parameters.save_to_writer(&mut f)?;
        // write innovation DB
        self.innovation_db.save_to_writer(&mut f)?;
        // write all genomes
        for g in &self.genomes {
            g.save_to_writer(&mut f)?;
        }
        Ok(())
    }

    /// Load a population from a file previously written by `save`.
    /// Returns a `Population` with parameters/innovation DB populated and
    /// genomes reconstructed; species will be rebuilt with `speciate()`.
    pub fn from_file(path: &str) -> IoResult<Self> {
        // read whole file
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // load parameters from text
        let mut params = Parameters::default();
        params.load_from_str(&contents)?;

        // create population skeleton
        let mut pop = Population::new(params.clone());

        // load innovation DB (it scans the same file for its tokens)
        pop.innovation_db.load(path)?;

        // find all GenomeStart..GenomeEnd blocks and parse them
        let mut start = 0usize;
        loop {
            if let Some(spos) = contents[start..].find("GenomeStart") {
                let sidx = start + spos;
                if let Some(epos) = contents[sidx..].find("GenomeEnd") {
                    let eidx = sidx + epos + "GenomeEnd".len();
                    let block = &contents[sidx..eidx];
                    if let Ok(g) = crate::genome::Genome::load_from_str(block) {
                        pop.genomes.push(g);
                    }
                    start = eidx;
                    continue;
                }
            }
            break;
        }

        // compute next ids
        let mut max_id: u64 = 0;
        for g in &pop.genomes {
            if g.get_id() > max_id {
                max_id = g.get_id();
            }
        }
        pop.next_genome_id = max_id.saturating_add(1);

        // re-speciate population
        pop.species.clear();
        pop.speciate();

        Ok(pop)
    }

    /// Choose a parent species index probabilistically weighted by total adjusted fitness.
    pub fn choose_parent_species(&self, rng: &mut impl Rng) -> Option<usize> {
        if self.species.is_empty() {
            return None;
        }
        // compute weight per species
        let mut weights: Vec<f64> = Vec::with_capacity(self.species.len());
        let mut total = 0.0;
        for s in &self.species {
            let mut sum = 0.0;
            for gid in &s.members {
                if let Some(g) = self.genomes.iter().find(|gg| gg.get_id() == *gid) {
                    sum += g.get_adj_fitness();
                }
            }
            weights.push(sum);
            total += sum;
        }

        if total <= 0.0 {
            // fall back to uniform random species
            let idx = rng.random_range(0..self.species.len());
            return Some(idx);
        }

        let mut pick = rng.random::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            if pick <= w {
                return Some(i);
            }
            pick -= w;
        }
        // rounding fallback
        Some(self.species.len().saturating_sub(1))
    }

    /// Remove worst individual based on adjusted fitness. Returns removed genome if any.
    pub fn remove_worst_individual(&mut self) -> Option<Genome> {
        if self.genomes.is_empty() {
            return None;
        }
        // find index of min adjusted fitness
        let mut worst_idx: Option<usize> = None;
        let mut worst_val = f64::INFINITY;
        for (i, g) in self.genomes.iter().enumerate() {
            let v = g.get_adj_fitness();
            if v < worst_val {
                worst_val = v;
                worst_idx = Some(i);
            }
        }
        let idx = worst_idx?;
        let removed = self.genomes.remove(idx);

        // remove from species membership lists
        for s in &mut self.species {
            s.members.retain(|&mid| mid != removed.get_id());
        }

        // cleanup empty species
        self.species.retain(|s| !s.members.is_empty());

        Some(removed)
    }

    /// Reassign a genome (by index in self.genomes) to the correct species based on compatibility.
    pub fn reassign_species(&mut self, genome_idx: usize) {
        if genome_idx >= self.genomes.len() {
            return;
        }
        let g = &self.genomes[genome_idx];
        // find best matching species
        let mut found_idx: Option<usize> = None;
        for (i, s) in self.species.iter().enumerate() {
            if let Some(rep_id) = s.members.first() {
                if let Some(rep) = self.genomes.iter().find(|gg| gg.get_id() == *rep_id) {
                    let neurons_a = g.neuron_genes().len() as f64;
                    let neurons_b = rep.neuron_genes().len() as f64;
                    let links_a = g.link_genes().len() as f64;
                    let links_b = rep.link_genes().len() as f64;
                    let mut dist = (neurons_a - neurons_b).abs() + (links_a - links_b).abs();
                    if self.parameters.normalize_genome_size {
                        let norm = (neurons_a.max(neurons_b) + links_a.max(links_b)).max(1.0);
                        dist /= norm;
                    }
                    if dist <= self.parameters.compat_treshold {
                        found_idx = Some(i);
                        break;
                    }
                }
            }
        }

        // remove genome id from any species it currently belongs to
        let gid = g.get_id();
        for s in &mut self.species {
            s.members.retain(|&mid| mid != gid);
        }

        if let Some(si) = found_idx {
            self.species[si].members.push(gid);
        } else {
            // create new species
            let sid = self.next_species_id;
            self.next_species_id = self.next_species_id.saturating_add(1);
            let mut s = Species {
                id: sid,
                members: Vec::new(),
                age: 0,
                best_fitness: g.get_fitness(),
            };
            s.members.push(gid);
            self.species.push(s);
        }
    }

    /// Realtime tick: replace worst individual with a baby, return the baby genome.
    /// This mirrors simplified C++ Population::Tick behaviour.
    pub fn tick(&mut self, rng: &mut impl Rng) -> Option<Genome> {
        if self.genomes.is_empty() {
            return None;
        }

        // ensure species and fitness adjusted
        if self.species.is_empty() {
            self.speciate();
        }
        self.adjust_fitness();

        // choose parent species
        let sidx = self.choose_parent_species(rng)?;
        if self.species.is_empty() || sidx >= self.species.len() {
            return None;
        }

        // collect individuals in species
        let mut individuals: Vec<Genome> = self.species[sidx]
            .members
            .iter()
            .filter_map(|gid| self.genomes.iter().find(|gg| gg.get_id() == *gid).cloned())
            .collect();
        if individuals.is_empty() {
            return None;
        }

        // sort and pick parent via truncated survival_rate
        individuals.sort_by(|a, b| {
            b.get_fitness()
                .partial_cmp(&a.get_fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let num_parents = ((self.parameters.survival_rate * individuals.len() as f64) as usize)
            .max(1)
            .min(individuals.len());
        let pidx = if num_parents <= 1 {
            0
        } else {
            rng.random_range(0..num_parents)
        };
        let mut child = individuals[pidx].clone();

        // mutate child
        if rng.random::<f64>() < self.parameters.overall_mutation_rate {
            // perform some mutations (reuse existing helpers)
            let _ = child.mutate_link_weights(&self.parameters, rng);
        }
        if rng.random::<f64>() < self.parameters.mutate_add_neuron_prob {
            let _ = child.mutate_add_neuron(&mut self.innovation_db, &self.parameters, rng);
        }

        // remove worst individual and replace it with child (preserve id)
        if let Some(removed) = self.remove_worst_individual() {
            child.set_id(removed.get_id());
            self.genomes.push(child.clone());
            // reassign species for new child
            let new_idx = self.genomes.len() - 1;
            self.reassign_species(new_idx);
            return Some(child);
        }

        None
    }
}
