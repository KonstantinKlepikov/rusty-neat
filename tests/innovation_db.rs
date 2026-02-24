// This test verifies basic behavior of `InnovationDatabase`:
// - correct assignment of identifiers when adding link and neuron innovations
//   (we initialize `next_neuron_id` and `next_innov_id` and expect specific values);
// - correctness of saving to a file and loading (round-trip) — the structure
//   should be restored with the same records;
// - after loading, the mappings must contain the expected values.
use rusty_neat::innovation::InnovationDatabase;
use tempfile::NamedTempFile;

#[test]
fn test_add_and_save_load() {
    let mut db = InnovationDatabase::new(10, 20);
    let id1 = db.add_link_innovation(1, 2);
    assert_eq!(id1, 20);
    let nid = db.add_neuron_innovation_with_type(2, 3, NeuronType::Hidden);
    assert_eq!(nid, 10);
    // Use a NamedTempFile so the file is removed automatically when dropped.
    let tmpfile = NamedTempFile::new().expect("create temp file");
    let tmp_path = tmpfile.path().to_str().expect("path to str");
    db.save(tmp_path).expect("save failed");

    let mut db2 = InnovationDatabase::new(1, 1);
    db2.load(tmp_path).expect("load failed");
    // confirm maps using the idiomatic `check_innovation`
    use rusty_neat::innovation::InnovationType;
    assert_eq!(db2.check_innovation(1, 2, InnovationType::NewLink), Some(20));
    assert_eq!(db2.check_innovation(2, 3, InnovationType::NewNeuron), Some(10));
}

// English comments required by project rules.
// Tests for the newly added idiomatic InnovationDatabase methods.

use rusty_neat::innovation::{InnovationType};
use rusty_neat::genes::{NeuronType, LinkGene};

#[test]
fn test_check_and_add_link_innovation() {
    let mut db = InnovationDatabase::new(1, 1);
    // initially no innovation
    assert_eq!(db.check_innovation(1, 2, InnovationType::NewLink), None);

    // add link and verify
    let id = db.add_link_innovation(1, 2);
    assert!(id >= 1);
    assert_eq!(db.check_innovation(1, 2, InnovationType::NewLink), Some(id));
    // check_all finds at least one index
    let matches = db.check_all_innovations(1, 2, InnovationType::NewLink);
    assert!(!matches.is_empty());
}

#[test]
fn test_add_neuron_innovation_and_find() {
    let mut db = InnovationDatabase::new(10, 100);
    // add neuron innovation with explicit type; next neuron id should start at the
    // `start_neuron_id` provided (10)
    let nid = db.add_neuron_innovation_with_type(3, 4, NeuronType::Hidden);
    assert_eq!(nid, 10);

    // find neuron id
    assert_eq!(db.find_neuron_id(3, 4), Some(nid));
    // last neuron id should also be equal
    assert_eq!(db.find_last_neuron_id(3, 4), Some(nid));
}

#[test]
fn test_check_last_innovation_behavior() {
    let mut db = InnovationDatabase::new(1, 1);
    // add two innovations for same connection
    let _ = db.add_link_innovation(5, 6);
    let second = db.add_link_innovation(5, 6);
    // check_last_innovation should return the most recent
    assert_eq!(db.check_last_innovation(5, 6, InnovationType::NewLink), Some(second));
}

#[test]
fn test_flush_clears_data() {
    let mut db = InnovationDatabase::new(1, 1);
    db.add_link_innovation(7, 8);
    assert!(db.check_innovation(7, 8, InnovationType::NewLink).is_some());
    db.flush();
    assert_eq!(db.check_innovation(7, 8, InnovationType::NewLink), None);
}

// Test for `InnovationDatabase::init_from_genome`
//
// This test ensures that:
// - innovations present in a genome's `LinkGene` list are copied into the
//   database (check via `check_innovation`).
// - `next_neuron_id` and `next_innov_id` are set from the provided values
//   (validated indirectly by adding new link/neuron and checking returned ids).

#[test]
fn test_init_from_genome_populates_db_and_ids() {
    // Prepare mock genome links
    let links = vec![
        LinkGene::new(1, 2, 10, 0.5, false),
        LinkGene::new(3, 4, 11, -1.2, true),
    ];

    let mut db = InnovationDatabase::new(100, 200);

    // Initialize from genome links; last ids passed should become the "next" ids
    db.init_from_genome(&links, 100, 200);

    // Check that genome link innovations were imported
    use rusty_neat::innovation::InnovationType;
    assert_eq!(db.check_innovation(1, 2, InnovationType::NewLink), Some(10));
    assert_eq!(db.check_innovation(3, 4, InnovationType::NewLink), Some(11));

    // Adding a new link (not present) should return the starting innovation id (200)
    let new_link_id = db.add_link_innovation(5, 6);
    assert_eq!(new_link_id, 200);

    // Adding a new neuron (not present) should return the starting neuron id (100)
    let new_neuron_id = db.add_neuron_innovation_with_type(7, 8, NeuronType::Hidden);
    assert_eq!(new_neuron_id, 100);
}
