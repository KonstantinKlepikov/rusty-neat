use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

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

    /// Return a lightweight iterator over neuron genes (does not copy the whole vec)
    fn iter_neurons(&self) -> PyResult<PyNeuronIterator> {
        Ok(PyNeuronIterator {
            genome: self.inner.clone(),
            idx: 0,
        })
    }

    /// Return a lightweight iterator over link genes
    fn iter_links(&self) -> PyResult<PyLinkIterator> {
        Ok(PyLinkIterator {
            genome: self.inner.clone(),
            idx: 0,
        })
    }

    /// Return neurons as a Python list (copies elements)
    fn neurons_list(&self, py: Python) -> PyResult<PyObject> {
        let g = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let list = PyList::empty(py);
        for ng in &g.neuron_genes {
            let pyng = PyNeuronGene {
                id: ng.id,
                neuron_type: format!("{:?}", ng.neuron_type),
                x: ng.x,
                y: ng.y,
                split_y: ng.split_y,
                a: ng.a,
                b: ng.b,
                timeconstant: ng.timeconstant,
                bias: ng.bias,
            };
            list.append(Py::new(py, pyng)?)?;
        }
        Ok(list.to_object(py))
    }

    /// Return links as a Python list (copies elements)
    fn links_list(&self, py: Python) -> PyResult<PyObject> {
        let g = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let list = PyList::empty(py);
        for lg in &g.link_genes {
            let pylg = PyLinkGene {
                from_neuron_id: lg.from_neuron_id,
                to_neuron_id: lg.to_neuron_id,
                innovation_id: lg.innovation_id,
                weight: lg.weight,
                is_recurrent: lg.is_recurrent,
            };
            list.append(Py::new(py, pylg)?)?;
        }
        Ok(list.to_object(py))
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

    /// Build phenotype into a provided `PyNeuralNetwork`.
    /// This may be a heavy operation; it releases the GIL while running.
    fn build_phenotype(&self, py: Python, py_net: PyRef<PyNeuralNetwork>) -> PyResult<()> {
        let genome_arc = self.inner.clone();
        let net_arc = py_net.inner.clone();
        py.allow_threads(move || {
            let g = genome_arc
                .read()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            let mut net = net_arc
                .write()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            g.build_phenotype(&mut *net);
            Ok(())
        })
    }

    /// Build HyperNEAT phenotype using a Substrate. Accepts `PyNeuralNetwork` and `PySubstrate`.
    fn build_hyperneat_phenotype(
        &self,
        py: Python,
        py_net: PyRef<PyNeuralNetwork>,
        py_subst: PyRef<PySubstrate>,
    ) -> PyResult<()> {
        let genome_arc = self.inner.clone();
        let net_arc = py_net.inner.clone();
        let subst_arc = py_subst.inner.clone();
        py.allow_threads(move || {
            let g = genome_arc
                .read()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            let mut net = net_arc
                .write()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            let subst = subst_arc
                .read()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            g.build_hyperneat_phenotype(&mut *net, &*subst);
            Ok(())
        })
    }
}

/// Iterator over neuron_genes in a Genome (lightweight, reads one-by-one)
#[pyclass]
pub struct PyNeuronIterator {
    genome: Arc<RwLock<rusty_neat::Genome>>,
    idx: usize,
}

#[pymethods]
impl PyNeuronIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        // try to read one neuron; return None at end
        let maybe_ng = {
            let g = self
                .genome
                .read()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            if self.idx >= g.neuron_genes.len() {
                None
            } else {
                Some(g.neuron_genes[self.idx].clone())
            }
        };

        match maybe_ng {
            None => Ok(None),
            Some(ng) => {
                self.idx += 1;
                // convert to PyNeuronGene and return as PyObject
                Python::with_gil(|py| {
                    let pyng = PyNeuronGene {
                        id: ng.id,
                        neuron_type: format!("{:?}", ng.neuron_type),
                        x: ng.x,
                        y: ng.y,
                        split_y: ng.split_y,
                        a: ng.a,
                        b: ng.b,
                        timeconstant: ng.timeconstant,
                        bias: ng.bias,
                    };
                    let obj = Py::new(py, pyng)?.into_py(py);
                    Ok(Some(obj))
                })
            }
        }
    }
}

/// Iterator over link_genes in a Genome
#[pyclass]
pub struct PyLinkIterator {
    genome: Arc<RwLock<rusty_neat::Genome>>,
    idx: usize,
}

#[pymethods]
impl PyLinkIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        let maybe_lg = {
            let g = self
                .genome
                .read()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
            if self.idx >= g.link_genes.len() {
                None
            } else {
                Some(g.link_genes[self.idx].clone())
            }
        };

        match maybe_lg {
            None => Ok(None),
            Some(lg) => {
                self.idx += 1;
                Python::with_gil(|py| {
                    let pylg = PyLinkGene {
                        from_neuron_id: lg.from_neuron_id,
                        to_neuron_id: lg.to_neuron_id,
                        innovation_id: lg.innovation_id,
                        weight: lg.weight,
                        is_recurrent: lg.is_recurrent,
                    };
                    let obj = Py::new(py, pylg)?.into_py(py);
                    Ok(Some(obj))
                })
            }
        }
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
            let mut nn = inner
                .write()
                .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
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

        fs::write(
            path,
            serde_json::to_string_pretty(&root)
                .map_err(|e| PyRuntimeError::new_err(format!("serialize error: {}", e)))?,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("IO error: {}", e)))?;
        Ok(())
    }

    /// Load network state from a JSON file (returns new PyNeuralNetwork)
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        if !Path::new(path).exists() {
            return Err(PyRuntimeError::new_err("file not found"));
        }
        let data = fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("IO error: {}", e)))?;
        let v: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("parse error: {}", e)))?;

        let mut net = rusty_neat::NeuralNetwork::new();
        if let Some(n_inputs) = v.get("num_inputs").and_then(|x| x.as_u64()) {
            net.set_input_output_dimensions(
                n_inputs as usize,
                v.get("num_outputs").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            );
        }

        if let Some(arr) = v.get("neurons").and_then(|x| x.as_array()) {
            for nn in arr {
                let activation = nn.get("activation").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let bias = nn.get("bias").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let a = nn.get("a").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let b = nn.get("b").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let timeconst = nn.get("timeconst").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let membrane_potential = nn
                    .get("membrane_potential")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0);
                // map activation function by name fallback to SignedSigmoid
                let act_name = nn
                    .get("activation_function")
                    .and_then(|x| x.as_str())
                    .unwrap_or("SignedSigmoid");
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
                let recur = c
                    .get("recur_flag")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false);
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

/// Lightweight export of LinkGene for Python (namedtuple-like)
#[pyclass]
pub struct PyLinkGene {
    #[pyo3(get, set)]
    pub from_neuron_id: u64,
    #[pyo3(get, set)]
    pub to_neuron_id: u64,
    #[pyo3(get, set)]
    pub innovation_id: u64,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub is_recurrent: bool,
}

#[pymethods]
impl PyLinkGene {
    #[new]
    fn new(
        from_neuron_id: u64,
        to_neuron_id: u64,
        innovation_id: u64,
        weight: f64,
        is_recurrent: bool,
    ) -> Self {
        PyLinkGene {
            from_neuron_id,
            to_neuron_id,
            innovation_id,
            weight,
            is_recurrent,
        }
    }

    #[staticmethod]
    fn empty() -> Self {
        PyLinkGene::new(0, 0, 0, 0.0, false)
    }
}

/// Lightweight export of NeuronGene for Python (namedtuple-like)
#[pyclass]
pub struct PyNeuronGene {
    #[pyo3(get, set)]
    pub id: u64,
    #[pyo3(get, set)]
    pub neuron_type: String,
    #[pyo3(get, set)]
    pub x: i32,
    #[pyo3(get, set)]
    pub y: i32,
    #[pyo3(get, set)]
    pub split_y: f64,
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub timeconstant: f64,
    #[pyo3(get, set)]
    pub bias: f64,
}

#[pymethods]
impl PyNeuronGene {
    #[new]
    #[pyo3(signature = (id, x=0, y=0, split_y=0.0, a=1.0, b=0.0, timeconstant=1.0, bias=0.0, neuron_type=None))]
    fn new(
        id: u64,
        x: i32,
        y: i32,
        split_y: f64,
        a: f64,
        b: f64,
        timeconstant: f64,
        bias: f64,
        neuron_type: Option<String>,
    ) -> Self {
        PyNeuronGene {
            id,
            neuron_type: neuron_type.unwrap_or_else(|| "Hidden".to_string()),
            x,
            y,
            split_y,
            a,
            b,
            timeconstant,
            bias,
        }
    }

    #[staticmethod]
    fn empty() -> Self {
        PyNeuronGene::new(0, 0, 0, 0.0, 1.0, 0.0, 1.0, 0.0, Some("Hidden".to_string()))
    }
}

/// Lightweight export of GenomeInitStruct for Python
#[pyclass]
pub struct PyGenomeInitStruct {
    #[pyo3(get, set)]
    pub num_inputs: usize,
    #[pyo3(get, set)]
    pub num_hidden: usize,
    #[pyo3(get, set)]
    pub num_outputs: usize,
    #[pyo3(get, set)]
    pub fs_neat: bool,
    #[pyo3(get, set)]
    pub output_act_type: String,
    #[pyo3(get, set)]
    pub hidden_act_type: String,
    #[pyo3(get, set)]
    pub seed_type: String,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub fs_neat_links: usize,
}

#[pymethods]
impl PyGenomeInitStruct {
    #[new]
    fn new(
        num_inputs: usize,
        num_hidden: usize,
        num_outputs: usize,
        fs_neat: Option<bool>,
        output_act_type: Option<String>,
        hidden_act_type: Option<String>,
        seed_type: Option<String>,
        num_layers: Option<usize>,
        fs_neat_links: Option<usize>,
    ) -> Self {
        PyGenomeInitStruct {
            num_inputs,
            num_hidden,
            num_outputs,
            fs_neat: fs_neat.unwrap_or(false),
            output_act_type: output_act_type.unwrap_or_else(|| "SignedSigmoid".to_string()),
            hidden_act_type: hidden_act_type.unwrap_or_else(|| "SignedSigmoid".to_string()),
            seed_type: seed_type.unwrap_or_else(|| "Perceptron".to_string()),
            num_layers: num_layers.unwrap_or(1),
            fs_neat_links: fs_neat_links.unwrap_or(0),
        }
    }

    // TODO: add a conversion helper to produce a Rust `GenomeInitStruct` when needed.
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
    m.add_class::<PyLinkGene>()?;
    m.add_class::<PyNeuronGene>()?;
    m.add_class::<PyGenomeInitStruct>()?;
    m.add_class::<PyNeuronIterator>()?;
    m.add_class::<PyLinkIterator>()?;
    Ok(())
}
