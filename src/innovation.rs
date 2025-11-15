use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct InnovationDatabase {
    next_neuron_id: u64,
    next_innov_id: u64,
    // map (from,to) -> innovation id for links
    link_map: HashMap<(u64,u64), u64>,
    // map (from,to) -> neuron id for split-neuron innovations
    neuron_map: HashMap<(u64,u64), u64>,
}

impl InnovationDatabase {
    pub fn new(start_neuron_id: u64, start_innov_id: u64) -> Self {
        Self {
            next_neuron_id: start_neuron_id,
            next_innov_id: start_innov_id,
            link_map: HashMap::new(),
            neuron_map: HashMap::new(),
        }
    }

    /// Check if a link innovation exists
    pub fn check_link_innovation(&self, from: u64, to: u64) -> Option<u64> {
        self.link_map.get(&(from,to)).copied()
    }

    /// Add a link innovation and return its id
    pub fn add_link_innovation(&mut self, from: u64, to: u64) -> u64 {
        if let Some(&id) = self.link_map.get(&(from,to)) {
            return id;
        }
        let id = self.next_innov_id;
        self.next_innov_id += 1;
        self.link_map.insert((from,to), id);
        id
    }

    /// Check neuron innovation (split of link from->to)
    pub fn check_neuron_innovation(&self, from: u64, to: u64) -> Option<u64> {
        self.neuron_map.get(&(from,to)).copied()
    }

    /// Add neuron innovation (splitting link from->to) and return new neuron id
    /// Also does NOT automatically create link innovations — caller should add them.
    pub fn add_neuron_innovation(&mut self, from: u64, to: u64) -> u64 {
        if let Some(&nid) = self.neuron_map.get(&(from,to)) {
            return nid;
        }
        let nid = self.next_neuron_id;
        self.next_neuron_id += 1;
        self.neuron_map.insert((from,to), nid);
        nid
    }
}
