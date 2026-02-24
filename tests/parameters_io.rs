use rusty_neat::Parameters;
use tempfile::NamedTempFile;

#[test]
fn test_reset_restores_defaults() {
    // Verify that `reset()` restores parameters to their default values.
    // Change several fields, call `reset()`, and compare with `Parameters::default()`.
    let mut p = Parameters::default();
    // change several fields
    p.population_size = 1234;
    p.speciation = !p.speciation;
    p.constraint_trials = 42;

    // reset
    p.reset();

    let d = Parameters::default();
    assert_eq!(p.population_size, d.population_size);
    assert_eq!(p.speciation, d.speciation);
    assert_eq!(p.constraint_trials, d.constraint_trials);
}

#[test]
fn test_load_from_str_parses_values() {
    // Verify parsing of a string representation via `load_from_str()`.
    // Construct a minimal section with a few keys and ensure values are applied correctly.
    let mut p = Parameters::default();
    let data = "NEAT_ParametersStart\nPopulationSize 42\nSpeciation false\nConstraintTrials 7\nNEAT_ParametersEnd\n";

    p.load_from_str(data).expect("load_from_str failed");

    assert_eq!(p.population_size, 42);
    assert_eq!(p.speciation, false);
    assert_eq!(p.constraint_trials, 7);
}

#[test]
fn test_save_and_load_file_roundtrip() {
    // Round-trip save/load test: write parameters to a file via `save()`,
    // then read them back via `load()` and verify field equality.
    let mut p = Parameters::default();
    p.population_size = 9001;
    p.speciation = false;
    p.constraint_trials = 11;

    // Use a NamedTempFile so the file is cleaned up automatically.
    let tmpfile = NamedTempFile::new().expect("create temp file");
    let tmp_path = tmpfile.path().to_str().expect("path to str");

    // save to file
    p.save(tmp_path).expect("save failed");

    // load into new object
    let mut q = Parameters::default();
    q.load(tmp_path).expect("load failed");

    assert_eq!(q.population_size, 9001);
    assert_eq!(q.speciation, false);
    assert_eq!(q.constraint_trials, 11);
}
