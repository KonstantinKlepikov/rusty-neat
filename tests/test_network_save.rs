// Tests for `save` and `save_writer`: output format and file writing
use rusty_neat::genes::{ActivationFunction, NeuronType};
use rusty_neat::network::{Connection, NeuralNetwork, Neuron};
use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn save_writer_emits_expected_sections() {
    let mut net = NeuralNetwork::new();

    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Output,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.set_input_output_dimensions(1, 1);

    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: 0.42,
        signal: 0.0,
        recur_flag: false,
        hebb_rate: 0.1,
        hebb_pre_rate: 0.2,
    });

    let mut buf: Vec<u8> = Vec::new();
    net.save_writer(&mut buf).expect("save_writer failed");
    let s = String::from_utf8(buf).expect("invalid utf8");

    assert!(s.starts_with("NNstart\n"));
    assert!(s.contains("1 1\n"), "dims line missing");
    assert!(s.contains("neuron"), "neuron line missing");
    assert!(s.contains("connection"), "connection line missing");
    assert!(s.contains("NNend\n\n"), "footer missing");
}

// `save` should create a file and write the network representation
#[test]
fn save_creates_file_and_writes_contents() {
    let mut net = NeuralNetwork::new();

    net.add_neuron(Neuron::new(
        NeuronType::Input,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.add_neuron(Neuron::new(
        NeuronType::Output,
        ActivationFunction::UnsignedSigmoid,
    ));
    net.set_input_output_dimensions(1, 1);

    net.add_connection(Connection {
        source_neuron_idx: 0,
        target_neuron_idx: 1,
        weight: -0.33,
        signal: 0.0,
        recur_flag: true,
        hebb_rate: 0.0,
        hebb_pre_rate: 0.0,
    });

    // unique tempfile path in temp dir
    let mut path = env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    path.push(format!("rusty_neat_test_save_{}.nn", ts));
    let path_str = path.to_string_lossy().into_owned();

    // save to file
    net.save(&path_str).expect("save failed");

    // read back and check
    let content = fs::read_to_string(&path_str).expect("read failed");
    assert!(content.starts_with("NNstart\n"));
    assert!(content.contains("connection"));

    // cleanup
    let _ = fs::remove_file(&path_str);
}
