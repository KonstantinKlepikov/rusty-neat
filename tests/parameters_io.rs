use rusty_neat::Parameters;
use std::fs;

#[test]
fn test_reset_restores_defaults() {
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
    let mut p = Parameters::default();
    let data = "NEAT_ParametersStart\nPopulationSize 42\nSpeciation false\nConstraintTrials 7\nNEAT_ParametersEnd\n";

    p.load_from_str(data).expect("load_from_str failed");

    assert_eq!(p.population_size, 42);
    assert_eq!(p.speciation, false);
    assert_eq!(p.constraint_trials, 7);
}

#[test]
fn test_save_and_load_file_roundtrip() {
    let mut p = Parameters::default();
    p.population_size = 9001;
    p.speciation = false;
    p.constraint_trials = 11;

    let tmp_name = format!("rusty_neat_params_test_{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos());
    let tmp_path = std::env::temp_dir().join(&tmp_name);

    // save to file
    p.save(tmp_path.to_str().unwrap()).expect("save failed");

    // load into new object
    let mut q = Parameters::default();
    q.load(tmp_path.to_str().unwrap()).expect("load failed");

    assert_eq!(q.population_size, 9001);
    assert_eq!(q.speciation, false);
    assert_eq!(q.constraint_trials, 11);

    // cleanup
    let _ = fs::remove_file(tmp_path);
}
