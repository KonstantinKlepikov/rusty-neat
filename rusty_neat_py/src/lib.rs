use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};
use numpy::{PyArray1, PyReadonlyArray1};
use serde_json::json;
use std::fs;
use std::path::Path;

/// Thin PyO3 wrapper for `rusty_neat::Genome`
#[pyclass]
pub struct PyGenome {
    inner: Arc<RwLock<rusty_neat::Genome>>,
}

#[pymethods]
impl PyGenome {
    /// Construct a new Genome with basic dimensions. Other parameters use sensible defaults.
    #[new]
    fn new(id: u64, num_inputs: usize, num_hidden: usize, num_outputs: usize) -> Self {
        let params = rusty_neat::GenomeInitStruct {
            num_inputs,
            num_hidden,
            num_outputs,
            fs_neat: false,
            output_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
            hidden_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
            seed_type: rusty_neat::genome::GenomeSeedType::Perceptron,
            num_layers: 1,
            fs_neat_links: 0,
        };

        let g = rusty_neat::Genome::new(id, &params);
        PyGenome {
            inner: Arc::new(RwLock::new(g)),
        }
    }

    /// Load genome from a file (delegates to rusty_neat::Genome::from_file)
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        match rusty_neat::Genome::from_file(path) {
            Ok(g) => Ok(PyGenome {
                inner: Arc::new(RwLock::new(g)),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("IO error: {}", e))),
        }
    }

    fn get_id(&self) -> PyResult<u64> {
        let g = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(g.id)
    }

    fn set_id(&mut self, id: u64) -> PyResult<()> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        g.id = id;
        Ok(())
    }
}

/// Thin PyO3 wrapper for `rusty_neat::NeuralNetwork`
#[pyclass]
pub struct PyNeuralNetwork {
    inner: Arc<RwLock<rusty_neat::NeuralNetwork>>,
}

#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new() -> Self {
        PyNeuralNetwork {
            inner: Arc::new(RwLock::new(rusty_neat::NeuralNetwork::new())),
        }
    }

    /// Provide inputs (accepts any Python sequence convertible to Vec<f64>)
    fn input(&self, inputs: &PyAny) -> PyResult<()> {
        // Try NumPy first
        if let Ok(arr) = inputs.extract::<PyReadonlyArray1<f64>>() {
            let slice = arr.as_array();
            let vec: Vec<f64> = slice.to_vec();
            let mut nn = self
                .inner
                .write()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            nn.input(vec);
            return Ok(());
        }

        // Fallback: any sequence convertible to Vec<f64>
        let seq = inputs
            .extract::<Vec<f64>>()
            .map_err(|e| PyRuntimeError::new_err(format!("expected sequence of floats: {}", e)))?;
        let mut nn = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        nn.input(seq);
        Ok(())
    }

    /// Run activation steps. `steps` defaults to 1. This releases the GIL for the duration of the work.
    fn activate(&self, py: Python, steps: Option<usize>) -> PyResult<()> {
        let steps = steps.unwrap_or(1);
        let inner = self.inner.clone();
        py.allow_threads(move || {
            let mut nn = inner.write().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            for _ in 0..steps {
                nn.activate();
            }
            Ok(())
        })
    }

    /// Get outputs as a NumPy 1-D array (f64)
    fn output(&self, py: Python) -> PyResult<PyObject> {
        let nn = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let out = nn.output();
        Ok(PyArray1::from_vec(py, out).to_object(py))
    }

    /// Save network state to a JSON file (simple, portable format).
    fn save(&self, path: &str) -> PyResult<()> {
        let nn = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;

        // Build JSON structure
        let neurons: Vec<_> = nn
            .neurons
            .iter()
            .map(|n| {
                json!({
                    "activation": n.activation,
                    "bias": n.bias,
                    "a": n.a,
                    "b": n.b,
                    "timeconst": n.timeconst,
                    "membrane_potential": n.membrane_potential,
                    "activation_function": format!("{:?}", n.activation_function_type),
                    "neuron_type": format!("{:?}", n.neuron_type),
                })
            })
            .collect();

        let conns: Vec<_> = nn
            .connections
            .iter()
            .map(|c| {
                json!({
                    "source": c.source_neuron_idx,
                    "target": c.target_neuron_idx,
                    "weight": c.weight,
                    "recur_flag": c.recur_flag,
                })
            })
            .collect();

        let root = json!({
            "schema_version": "1.0",
            "num_inputs": nn.num_inputs,
            "num_outputs": nn.num_outputs,
            "neurons": neurons,
            "connections": conns,
        });

        fs::write(path, serde_json::to_string_pretty(&root).map_err(|e| PyRuntimeError::new_err(format!("serialize error: {}", e)))?)
            .map_err(|e| PyRuntimeError::new_err(format!("IO error: {}", e)))?;
        Ok(())
    }

    /// Load network state from a JSON file (returns new PyNeuralNetwork)
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        if !Path::new(path).exists() {
            return Err(PyRuntimeError::new_err("file not found"));
        }
        let data = fs::read_to_string(path).map_err(|e| PyRuntimeError::new_err(format!("IO error: {}", e)))?;
        let v: serde_json::Value = serde_json::from_str(&data).map_err(|e| PyRuntimeError::new_err(format!("parse error: {}", e)))?;

        let mut net = rusty_neat::NeuralNetwork::new();
        if let Some(n_inputs) = v.get("num_inputs").and_then(|x| x.as_u64()) {
            net.set_input_output_dimensions(n_inputs as usize, v.get("num_outputs").and_then(|x| x.as_u64()).unwrap_or(0) as usize);
        }

        if let Some(arr) = v.get("neurons").and_then(|x| x.as_array()) {
            for nn in arr {
                let activation = nn.get("activation").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let bias = nn.get("bias").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let a = nn.get("a").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let b = nn.get("b").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let timeconst = nn.get("timeconst").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let membrane_potential = nn.get("membrane_potential").and_then(|x| x.as_f64()).unwrap_or(0.0);
                // map activation function by name fallback to SignedSigmoid
                let act_name = nn.get("activation_function").and_then(|x| x.as_str()).unwrap_or("SignedSigmoid");
                let act = match act_name {
                    "UnsignedSigmoid" => rusty_neat::genes::ActivationFunction::UnsignedSigmoid,
                    "Tanh" => rusty_neat::genes::ActivationFunction::Tanh,
                    "Linear" => rusty_neat::genes::ActivationFunction::Linear,
                    "Relu" => rusty_neat::genes::ActivationFunction::Relu,
                    "Softplus" => rusty_neat::genes::ActivationFunction::Softplus,
                    _ => rusty_neat::genes::ActivationFunction::SignedSigmoid,
                };

                let neuron = rusty_neat::network::Neuron {
                    activesum: 0.0,
                    activation,
                    a,
                    b,
                    timeconst,
                    bias,
                    membrane_potential,
                    activation_function_type: act,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    sx: 0.0,
                    sy: 0.0,
                    sz: 0.0,
                    substrate_coords: Vec::new(),
                    split_y: 0.0,
                    neuron_type: rusty_neat::genes::NeuronType::Hidden,
                    sensitivity_matrix: Vec::new(),
                };
                net.add_neuron(neuron);
            }
        }

        if let Some(arr) = v.get("connections").and_then(|x| x.as_array()) {
            for c in arr {
                let source = c.get("source").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let target = c.get("target").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let weight = c.get("weight").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let recur = c.get("recur_flag").and_then(|x| x.as_bool()).unwrap_or(false);
                let conn = rusty_neat::network::Connection {
                    source_neuron_idx: source,
                    target_neuron_idx: target,
                    weight,
                    signal: 0.0,
                    recur_flag: recur,
                    hebb_rate: 0.0,
                    hebb_pre_rate: 0.0,
                };
                net.add_connection(conn);
            }
        }

        Ok(PyNeuralNetwork {
            inner: Arc::new(RwLock::new(net)),
        })
    }
}

/// Thin PyO3 wrapper for `rusty_neat::Parameters`
#[pyclass]
pub struct PyParameters {
    inner: Arc<RwLock<rusty_neat::Parameters>>,
}

#[pymethods]
impl PyParameters {
    #[new]
    fn new() -> Self {
        PyParameters {
            inner: Arc::new(RwLock::new(rusty_neat::Parameters::default())),
        }
    }

    fn get_min_weight(&self) -> PyResult<f64> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(p.min_weight)
    }

    fn set_min_weight(&mut self, v: f64) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        p.min_weight = v;
        Ok(())
    }

    fn reset(&mut self) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        *p = rusty_neat::Parameters::default();
        Ok(())
    }
}

/// Thin PyO3 wrapper for `rusty_neat::Substrate`
#[pyclass]
pub struct PySubstrate {
    inner: Arc<RwLock<rusty_neat::Substrate>>,
}

#[pymethods]
impl PySubstrate {
    #[new]
    fn new() -> Self {
        PySubstrate {
            inner: Arc::new(RwLock::new(rusty_neat::Substrate::new())),
        }
    }

    /// Construct with explicit coords: accepts sequences of sequences of floats
    #[staticmethod]
    fn with_coords(inputs: &PyAny, hidden: &PyAny, outputs: &PyAny) -> PyResult<Self> {
        let in_coords = inputs.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for inputs: {}",
                e
            ))
        })?;
        let hidden_coords = hidden.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for hidden: {}",
                e
            ))
        })?;
        let out_coords = outputs.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for outputs: {}",
                e
            ))
        })?;

        Ok(PySubstrate {
            inner: Arc::new(RwLock::new(rusty_neat::Substrate::with_coords(
                in_coords,
                hidden_coords,
                out_coords,
            ))),
        })
    }

    fn get_min_cppn_inputs(&self) -> PyResult<usize> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.get_min_cppn_inputs())
    }

    fn get_min_cppn_outputs(&self) -> PyResult<usize> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.get_min_cppn_outputs())
    }

    fn print_info(&self) -> PyResult<()> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.print_info();
        Ok(())
    }

    fn set_neurons(
        &mut self,
        inputs: Vec<Vec<f64>>,
        hidden: Vec<Vec<f64>>,
        outputs: Vec<Vec<f64>>,
    ) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.set_neurons(inputs, hidden, outputs);
        Ok(())
    }

    fn clear_custom_connectivity(&mut self) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.clear_custom_connectivity();
        Ok(())
    }

    fn set_custom_connectivity(&mut self, conns: Vec<Vec<i32>>) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.set_custom_connectivity(conns);
        Ok(())
    }
}

/// Thin PyO3 wrapper for random utilities (rusty_neat::random::Random)
#[pyclass]
pub struct PyRNG;

#[pymethods]
impl PyRNG {
    #[new]
    fn new() -> Self {
        PyRNG {}
    }

    fn rand_float(&self) -> PyResult<f64> {
        Ok(rusty_neat::random::Random::rand_float())
    }

    fn rand_int(&self, max: u64) -> PyResult<u64> {
        Ok(rusty_neat::random::Random::rand_int(max))
    }
}

#[pymodule]
fn rusty_neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGenome>()?;
    m.add_class::<PyNeuralNetwork>()?;
    m.add_class::<PyParameters>()?;
    m.add_class::<PySubstrate>()?;
    m.add_class::<PyRNG>()?;
    Ok(())
}
