use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult, Write};

use crate::genes::NeuronType;
use serde::{Deserialize, Serialize};

/// Innovation types (NEW_NEURON / NEW_LINK)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnovationType {
    NewNeuron,
    NewLink,
}

/// Single innovation record (port of C++ `Innovation`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Innovation {
    id: u64,
    innov_type: InnovationType,
    /// Neuron/Link specific data
    from_neuron_id: u64,
    to_neuron_id: u64,
    neuron_id: u64,
    neuron_type: NeuronType,
}

impl Innovation {
    pub fn new(
        id: u64,
        innov_type: InnovationType,
        from: u64,
        to: u64,
        n_type: NeuronType,
        n_id: u64,
    ) -> Self {
        Self {
            id,
            innov_type,
            from_neuron_id: from,
            to_neuron_id: to,
            neuron_id: n_id,
            neuron_type: n_type,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn innov_type(&self) -> InnovationType {
        self.innov_type
    }
    pub fn from_neuron_id(&self) -> u64 {
        self.from_neuron_id
    }
    pub fn to_neuron_id(&self) -> u64 {
        self.to_neuron_id
    }
    pub fn neuron_id(&self) -> u64 {
        self.neuron_id
    }
    pub fn neuron_type(&self) -> NeuronType {
        self.neuron_type
    }
}

impl Default for Innovation {
    fn default() -> Self {
        Self {
            id: 0,
            innov_type: InnovationType::NewLink,
            from_neuron_id: 0,
            to_neuron_id: 0,
            neuron_id: 0,
            neuron_type: NeuronType::Hidden,
        }
    }
}

/// Innovation database used by genomes.
#[derive(Debug, Clone)]
pub struct InnovationDatabase {
    pub innovations: Vec<Innovation>,
    next_neuron_id: u64,
    next_innov_id: u64,
    // map (from,to) -> innovation id for links
    link_map: HashMap<(u64, u64), u64>,
    // map (from,to) -> neuron id for split-neuron innovations
    neuron_map: HashMap<(u64, u64), u64>,
}

impl InnovationDatabase {
    /// Create an empty database with starting IDs
    pub fn new(start_neuron_id: u64, start_innov_id: u64) -> Self {
        Self {
            innovations: Vec::new(),
            next_neuron_id: start_neuron_id,
            next_innov_id: start_innov_id,
            link_map: HashMap::new(),
            neuron_map: HashMap::new(),
        }
    }

    /// Initialize the innovation database from an existing genome's links.
    ///
    /// This mirrors the C++ `Init(const Genome& a_Genome)` behavior by
    /// populating the database with link innovations taken from the genome's
    /// `LinkGene` list and setting the next neuron/innovation IDs accordingly.
    ///
    /// `links` - slice of `LinkGene` from the genome
    /// `last_neuron_id` - starting value for `next_neuron_id`
    /// `last_innovation_id` - starting value for `next_innov_id`
    pub fn init_from_genome(
        &mut self,
        links: &[crate::genes::LinkGene],
        last_neuron_id: u64,
        last_innovation_id: u64,
    ) {
        self.innovations.clear();
        self.link_map.clear();
        self.neuron_map.clear();
        for link in links {
            let innov_id = link.innovation_id;
            let from = link.from_neuron_id;
            let to = link.to_neuron_id;
            self.innovations.push(Innovation::new(
                innov_id,
                InnovationType::NewLink,
                from,
                to,
                NeuronType::Hidden,
                0,
            ));
            self.link_map.insert((from, to), innov_id);
        }
        self.next_neuron_id = last_neuron_id;
        self.next_innov_id = last_innovation_id;
    }

    /// Check if a link innovation exists.
    ///
    /// Use `check_innovation(from, to, InnovationType::NewLink)` which is the
    /// preferred idiomatic API.

    /// Add a link innovation and return its id
    pub fn add_link_innovation(&mut self, from: u64, to: u64) -> u64 {
        if let Some(&id) = self.link_map.get(&(from, to)) {
            return id;
        }
        let id = self.next_innov_id;
        self.next_innov_id += 1;
        self.link_map.insert((from, to), id);
        self.innovations
            .push(Innovation::new(id, InnovationType::NewLink, from, to, NeuronType::Hidden, 0));
        id
    }

    /// Check neuron innovation (split of link from->to) via direct lookup.
    ///
    /// Use `check_innovation(from, to, InnovationType::NewNeuron)` which is the
    /// preferred idiomatic API.

    /// Add neuron innovation with default `NeuronType::Hidden`.
    ///
    /// Prefer the explicit `add_neuron_innovation_with_type(from, to, neuron_type)` API.

    /// Check if an innovation exists. Returns `Some(id)` if found, otherwise `None`.
    /// For `NewLink` returns the link innovation id, for `NewNeuron` returns the neuron id.
    pub fn check_innovation(
        &self,
        from: u64,
        to: u64,
        typ: InnovationType,
    ) -> Option<u64> {
        match typ {
            InnovationType::NewLink => self.link_map.get(&(from, to)).copied(),
            InnovationType::NewNeuron => self.neuron_map.get(&(from, to)).copied(),
        }
    }

    /// Check for the last matching innovation. Returns the most-recent matching
    /// innovation id (for links) or neuron id (for neuron innovations) if any.
    pub fn check_last_innovation(&self, from: u64, to: u64, typ: InnovationType) -> Option<u64> {
        let mut result: Option<u64> = None;
        for innov in &self.innovations {
            if innov.from_neuron_id() == from && innov.to_neuron_id() == to && innov.innov_type() == typ {
                let val = match typ {
                    InnovationType::NewLink => innov.id(),
                    InnovationType::NewNeuron => innov.neuron_id(),
                };
                result = Some(val);
            }
        }
        result
    }

    /// Returns a list of indexes in the database of identical innovations.
    pub fn check_all_innovations(&self, from: u64, to: u64, typ: InnovationType) -> Vec<usize> {
        self.innovations
            .iter()
            .enumerate()
            .filter(|(_, innov)| {
                innov.from_neuron_id() == from && innov.to_neuron_id() == to && innov.innov_type() == typ
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the neuron ID given the in and out neurons, or `None` if not found.
    pub fn find_neuron_id(&self, from: u64, to: u64) -> Option<u64> {
        self.neuron_map.get(&(from, to)).copied()
    }

    /// Returns the last neuron ID for the given connection, or `None` if not found.
    pub fn find_last_neuron_id(&self, from: u64, to: u64) -> Option<u64> {
        let mut result: Option<u64> = None;
        for innov in &self.innovations {
            if innov.from_neuron_id() == from && innov.to_neuron_id() == to && innov.innov_type() == InnovationType::NewNeuron {
                result = Some(innov.neuron_id());
            }
        }
        result
    }

    /// Add a new neuron innovation with an explicit `NeuronType` and return the new neuron id.
    /// This is an idiomatic alternative to the previous default-typed helper.
    pub fn add_neuron_innovation_with_type(&mut self, from: u64, to: u64, n_type: NeuronType) -> u64 {
        if let Some(&nid) = self.neuron_map.get(&(from, to)) {
            return nid;
        }
        let nid = self.next_neuron_id;
        let innov_id = self.next_innov_id;
        self.next_neuron_id += 1;
        self.next_innov_id += 1;
        self.neuron_map.insert((from, to), nid);
        self.innovations.push(Innovation::new(
            innov_id,
            InnovationType::NewNeuron,
            from,
            to,
            n_type,
            nid,
        ));
        nid
    }

    /// Clears all innovations in the database (does not change next id counters).
    pub fn flush(&mut self) {
        self.innovations.clear();
        self.link_map.clear();
        self.neuron_map.clear();
    }

    /// Get innovation by index
    pub fn get_innovation_by_idx(&self, idx: usize) -> Option<&Innovation> {
        self.innovations.get(idx)
    }

    /// Save database to a text file. Format is human-readable and compatible
    /// with the cneat r1 port format.
    pub fn save(&self, filename: &str) -> IoResult<()> {
        let mut f = File::create(filename)?;
        writeln!(f, "InnovationDatabaseStart")?;
        writeln!(f, "NextInnovNum: {}", self.next_innov_id)?;
        writeln!(f, "NextNeuronID: {}", self.next_neuron_id)?;
        for innov in &self.innovations {
            writeln!(
                f,
                "Innovation {} {} {} {} {} {}",
                innov.id(),
                match innov.innov_type() {
                    InnovationType::NewNeuron => 0,
                    InnovationType::NewLink => 1,
                },
                innov.from_neuron_id(),
                innov.to_neuron_id(),
                neuron_type_to_u8(innov.neuron_type()),
                innov.neuron_id(),
            )?;
        }
        writeln!(f, "InnovationDatabaseEnd")?;
        Ok(())
    }

    /// Load database from a file previously written by `save`.
    pub fn load(&mut self, filename: &str) -> IoResult<()> {
        let f = File::open(filename)?;
        let reader = BufReader::new(f);
        self.innovations.clear();
        self.link_map.clear();
        self.neuron_map.clear();
        for line in reader.lines() {
            let line = line?;
            let s = line.trim();
            if s.is_empty() {
                continue;
            }
            if s.starts_with("NextInnovNum:") {
                if let Some(v) = s.split(':').nth(1) {
                    self.next_innov_id = v.trim().parse().unwrap_or(self.next_innov_id);
                }
            } else if s.starts_with("NextNeuronID:") {
                if let Some(v) = s.split(':').nth(1) {
                    self.next_neuron_id = v.trim().parse().unwrap_or(self.next_neuron_id);
                }
            } else if s.starts_with("Innovation ") {
                // Innovation id type from to neuron_type neuron_id
                let parts: Vec<&str> = s.split_whitespace().collect();
                if parts.len() >= 7 {
                    let id: u64 = parts[1].parse().unwrap_or(0);
                    let typ: u8 = parts[2].parse().unwrap_or(1);
                    let from: u64 = parts[3].parse().unwrap_or(0);
                    let to: u64 = parts[4].parse().unwrap_or(0);
                    let ntype: u8 = parts[5].parse().unwrap_or(2);
                    let nid: u64 = parts[6].parse().unwrap_or(0);
                    let innov_type = if typ == 0 {
                        InnovationType::NewNeuron
                    } else {
                        InnovationType::NewLink
                    };
                    let neuron_type = u8_to_neuron_type(ntype);
                    self.innovations.push(Innovation::new(
                        id, innov_type, from, to, neuron_type, nid,
                    ));
                    match innov_type {
                        InnovationType::NewLink => {
                            self.link_map.insert((from, to), id);
                        }
                        InnovationType::NewNeuron => {
                            self.neuron_map.insert((from, to), nid);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn neuron_type_to_u8(nt: NeuronType) -> u8 {
    match nt {
        NeuronType::Input => 0,
        NeuronType::Output => 1,
        NeuronType::Hidden => 2,
        NeuronType::Bias => 3,
    }
}

fn u8_to_neuron_type(v: u8) -> NeuronType {
    match v {
        0 => NeuronType::Input,
        1 => NeuronType::Output,
        3 => NeuronType::Bias,
        _ => NeuronType::Hidden,
    }
}
