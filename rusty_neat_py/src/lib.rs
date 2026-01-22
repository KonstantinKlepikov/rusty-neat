use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Thin PyO3 wrapper for `rusty_neat::Genome`
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
pub struct PyParameters {
    inner: Arc<RwLock<rusty_neat::Parameters>>,
}

/// Lightweight export of LinkGene for Python (namedtuple-like)
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
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
#[pyclass(module = "rusty_neat_py")]
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

    /// List neuron trait parameter names
    fn list_neuron_trait_parameters(&self) -> PyResult<Vec<String>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(p.neuron_trait_parameters.keys().cloned().collect())
    }

    /// Get neuron trait parameters by name. Returns None if not present.
    fn get_neuron_trait_parameters(&self, py: Python, name: &str) -> PyResult<Option<PyObject>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if let Some(tp) = p.neuron_trait_parameters.get(name) {
            Ok(Some(trait_parameters_to_pydict(py, tp)))
        } else {
            Ok(None)
        }
    }

    /// Set neuron trait parameters from a Python dict following the documented schema.
    fn set_neuron_trait_parameters(&mut self, name: &str, params: &PyAny) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let tp = pydict_to_trait_parameters(params)?;
        p.neuron_trait_parameters.insert(name.to_string(), tp);
        Ok(())
    }

    /// List link trait parameter names
    fn list_link_trait_parameters(&self) -> PyResult<Vec<String>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(p.link_trait_parameters.keys().cloned().collect())
    }

    fn get_link_trait_parameters(&self, py: Python, name: &str) -> PyResult<Option<PyObject>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if let Some(tp) = p.link_trait_parameters.get(name) {
            Ok(Some(trait_parameters_to_pydict(py, tp)))
        } else {
            Ok(None)
        }
    }

    fn set_link_trait_parameters(&mut self, name: &str, params: &PyAny) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let tp = pydict_to_trait_parameters(params)?;
        p.link_trait_parameters.insert(name.to_string(), tp);
        Ok(())
    }

    /// List genome trait parameter names
    fn list_genome_trait_parameters(&self) -> PyResult<Vec<String>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(p.genome_trait_parameters.keys().cloned().collect())
    }

    fn get_genome_trait_parameters(&self, py: Python, name: &str) -> PyResult<Option<PyObject>> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        if let Some(tp) = p.genome_trait_parameters.get(name) {
            Ok(Some(trait_parameters_to_pydict(py, tp)))
        } else {
            Ok(None)
        }
    }

    fn set_genome_trait_parameters(&mut self, name: &str, params: &PyAny) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let tp = pydict_to_trait_parameters(params)?;
        p.genome_trait_parameters.insert(name.to_string(), tp);
        Ok(())
    }

    /// Support pickling: return a Python dict with all parameters
    fn __getstate__(&self, py: Python) -> PyObject {
        let p = match self.inner.read() {
            Ok(v) => v,
            Err(_) => panic!("lock poisoned"),
        };
        let d = PyDict::new(py);
        // primitives
        let _ = d.set_item("min_weight", p.min_weight);
        let _ = d.set_item("max_weight", p.max_weight);
        let _ = d.set_item("split_recurrent", p.split_recurrent);
        let _ = d.set_item("dont_use_bias_neuron", p.dont_use_bias_neuron);

        // probabilities
        let _ = d.set_item(
            "mutate_add_link_from_bias_prob",
            p.mutate_add_link_from_bias_prob,
        );
        let _ = d.set_item("recurrent_prob", p.recurrent_prob);
        let _ = d.set_item("recurrent_loop_prob", p.recurrent_loop_prob);
        let _ = d.set_item("mutate_neuron_traits_prob", p.mutate_neuron_traits_prob);
        let _ = d.set_item("mutate_link_traits_prob", p.mutate_link_traits_prob);
        let _ = d.set_item("mutate_genome_traits_prob", p.mutate_genome_traits_prob);

        // weight mutation
        let _ = d.set_item("mutate_weights_severe_prob", p.mutate_weights_severe_prob);
        let _ = d.set_item("weight_mutation_rate", p.weight_mutation_rate);
        let _ = d.set_item("weight_replacement_rate", p.weight_replacement_rate);
        let _ = d.set_item("weight_mutation_max_power", p.weight_mutation_max_power);
        let _ = d.set_item(
            "weight_replacement_max_power",
            p.weight_replacement_max_power,
        );

        // neuron activation mutations
        let _ = d.set_item(
            "activation_a_mutation_max_power",
            p.activation_a_mutation_max_power,
        );
        let _ = d.set_item("min_activation_a", p.min_activation_a);
        let _ = d.set_item("max_activation_a", p.max_activation_a);
        let _ = d.set_item(
            "activation_b_mutation_max_power",
            p.activation_b_mutation_max_power,
        );
        let _ = d.set_item("min_activation_b", p.min_activation_b);
        let _ = d.set_item("max_activation_b", p.max_activation_b);

        // time-constant and bias
        let _ = d.set_item(
            "timeconstant_mutation_max_power",
            p.timeconstant_mutation_max_power,
        );
        let _ = d.set_item("min_neuron_time_constant", p.min_neuron_time_constant);
        let _ = d.set_item("max_neuron_time_constant", p.max_neuron_time_constant);
        let _ = d.set_item("bias_mutation_max_power", p.bias_mutation_max_power);
        let _ = d.set_item("min_neuron_bias", p.min_neuron_bias);
        let _ = d.set_item("max_neuron_bias", p.max_neuron_bias);

        // activation function probs
        let _ = d.set_item(
            "activation_function_probs",
            p.activation_function_probs.clone(),
        );

        // trait maps: convert each entry to a dict via helper
        let neuron_tp = PyDict::new(py);
        for (k, v) in &p.neuron_trait_parameters {
            let _ = neuron_tp.set_item(k, trait_parameters_to_pydict(py, v));
        }
        let _ = d.set_item("neuron_trait_parameters", neuron_tp);

        let link_tp = PyDict::new(py);
        for (k, v) in &p.link_trait_parameters {
            let _ = link_tp.set_item(k, trait_parameters_to_pydict(py, v));
        }
        let _ = d.set_item("link_trait_parameters", link_tp);

        let genome_tp = PyDict::new(py);
        for (k, v) in &p.genome_trait_parameters {
            let _ = genome_tp.set_item(k, trait_parameters_to_pydict(py, v));
        }
        let _ = d.set_item("genome_trait_parameters", genome_tp);

        d.to_object(py)
    }

    /// Support unpickling from dict produced by __getstate__
    fn __setstate__(&mut self, state: &PyAny) -> PyResult<()> {
        let d = state
            .downcast::<PyDict>()
            .map_err(|_| PyRuntimeError::new_err("__setstate__ expects a dict"))?;
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;

        // Manually assign known fields (explicit to avoid mistakes)
        if let Some(v) = d.get_item("min_weight") {
            p.min_weight = v
                .extract::<f64>()
                .map_err(|e| PyRuntimeError::new_err(format!("min_weight must be float: {}", e)))?;
        }
        if let Some(v) = d.get_item("max_weight") {
            p.max_weight = v
                .extract::<f64>()
                .map_err(|e| PyRuntimeError::new_err(format!("max_weight must be float: {}", e)))?;
        }
        if let Some(v) = d.get_item("split_recurrent") {
            p.split_recurrent = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("split_recurrent must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("dont_use_bias_neuron") {
            p.dont_use_bias_neuron = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("dont_use_bias_neuron must be bool: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("mutate_add_link_from_bias_prob") {
            p.mutate_add_link_from_bias_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "mutate_add_link_from_bias_prob must be float: {}",
                    e
                ))
            })?;
        }
        if let Some(v) = d.get_item("recurrent_prob") {
            p.recurrent_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("recurrent_prob must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("recurrent_loop_prob") {
            p.recurrent_loop_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("recurrent_loop_prob must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("mutate_neuron_traits_prob") {
            p.mutate_neuron_traits_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("mutate_neuron_traits_prob must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("mutate_link_traits_prob") {
            p.mutate_link_traits_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("mutate_link_traits_prob must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("mutate_genome_traits_prob") {
            p.mutate_genome_traits_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("mutate_genome_traits_prob must be float: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("mutate_weights_severe_prob") {
            p.mutate_weights_severe_prob = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("mutate_weights_severe_prob must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("weight_mutation_rate") {
            p.weight_mutation_rate = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("weight_mutation_rate must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("weight_replacement_rate") {
            p.weight_replacement_rate = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("weight_replacement_rate must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("weight_mutation_max_power") {
            p.weight_mutation_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("weight_mutation_max_power must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("weight_replacement_max_power") {
            p.weight_replacement_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "weight_replacement_max_power must be float: {}",
                    e
                ))
            })?;
        }

        if let Some(v) = d.get_item("activation_a_mutation_max_power") {
            p.activation_a_mutation_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "activation_a_mutation_max_power must be float: {}",
                    e
                ))
            })?;
        }
        if let Some(v) = d.get_item("min_activation_a") {
            p.min_activation_a = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("min_activation_a must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_activation_a") {
            p.max_activation_a = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_activation_a must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("activation_b_mutation_max_power") {
            p.activation_b_mutation_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "activation_b_mutation_max_power must be float: {}",
                    e
                ))
            })?;
        }
        if let Some(v) = d.get_item("min_activation_b") {
            p.min_activation_b = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("min_activation_b must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_activation_b") {
            p.max_activation_b = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_activation_b must be float: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("timeconstant_mutation_max_power") {
            p.timeconstant_mutation_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "timeconstant_mutation_max_power must be float: {}",
                    e
                ))
            })?;
        }
        if let Some(v) = d.get_item("min_neuron_time_constant") {
            p.min_neuron_time_constant = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("min_neuron_time_constant must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_neuron_time_constant") {
            p.max_neuron_time_constant = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_neuron_time_constant must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("bias_mutation_max_power") {
            p.bias_mutation_max_power = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("bias_mutation_max_power must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("min_neuron_bias") {
            p.min_neuron_bias = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("min_neuron_bias must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_neuron_bias") {
            p.max_neuron_bias = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_neuron_bias must be float: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("activation_function_probs") {
            p.activation_function_probs = v.extract::<Vec<f64>>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "activation_function_probs must be list of floats: {}",
                    e
                ))
            })?;
        }

        // trait maps
        p.neuron_trait_parameters.clear();
        if let Some(v) = d.get_item("neuron_trait_parameters") {
            if !v.is_none() {
                let tp_dict = v.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err("neuron_trait_parameters must be a dict")
                })?;
                for (k, val) in tp_dict.iter() {
                    let key = k
                        .extract::<String>()
                        .map_err(|_| PyRuntimeError::new_err("trait map keys must be strings"))?;
                    let tp = pydict_to_trait_parameters(val)?;
                    p.neuron_trait_parameters.insert(key, tp);
                }
            }
        }

        p.link_trait_parameters.clear();
        if let Some(v) = d.get_item("link_trait_parameters") {
            if !v.is_none() {
                let tp_dict = v
                    .downcast::<PyDict>()
                    .map_err(|_| PyRuntimeError::new_err("link_trait_parameters must be a dict"))?;
                for (k, val) in tp_dict.iter() {
                    let key = k
                        .extract::<String>()
                        .map_err(|_| PyRuntimeError::new_err("trait map keys must be strings"))?;
                    let tp = pydict_to_trait_parameters(val)?;
                    p.link_trait_parameters.insert(key, tp);
                }
            }
        }

        p.genome_trait_parameters.clear();
        if let Some(v) = d.get_item("genome_trait_parameters") {
            if !v.is_none() {
                let tp_dict = v.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err("genome_trait_parameters must be a dict")
                })?;
                for (k, val) in tp_dict.iter() {
                    let key = k
                        .extract::<String>()
                        .map_err(|_| PyRuntimeError::new_err("trait map keys must be strings"))?;
                    let tp = pydict_to_trait_parameters(val)?;
                    p.genome_trait_parameters.insert(key, tp);
                }
            }
        }

        Ok(())
    }
}

/// Thin PyO3 wrapper for `rusty_neat::Substrate`
#[pyclass(module = "rusty_neat_py")]
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

    // -- Flags and properties accessors
    fn get_leaky(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.leaky)
    }

    fn set_leaky(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.leaky = v;
        Ok(())
    }

    fn get_with_distance(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.with_distance)
    }

    fn set_with_distance(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.with_distance = v;
        Ok(())
    }

    fn get_min_time_const(&self) -> PyResult<f64> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.min_time_const)
    }

    fn set_min_time_const(&mut self, v: f64) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.min_time_const = v;
        Ok(())
    }

    fn get_max_time_const(&self) -> PyResult<f64> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.max_time_const)
    }

    fn set_max_time_const(&mut self, v: f64) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.max_time_const = v;
        Ok(())
    }

    fn get_max_weight_and_bias(&self) -> PyResult<f64> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.max_weight_and_bias)
    }

    fn set_max_weight_and_bias(&mut self, v: f64) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.max_weight_and_bias = v;
        Ok(())
    }

    // Activation function accessors as string names
    fn get_output_nodes_activation(&self) -> PyResult<String> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(format!("{:?}", s.output_nodes_activation))
    }

    fn set_output_nodes_activation(&mut self, name: &str) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.output_nodes_activation = match name {
            "UnsignedSigmoid" => rusty_neat::genes::ActivationFunction::UnsignedSigmoid,
            "Tanh" => rusty_neat::genes::ActivationFunction::Tanh,
            "Linear" => rusty_neat::genes::ActivationFunction::Linear,
            "Relu" => rusty_neat::genes::ActivationFunction::Relu,
            "Softplus" => rusty_neat::genes::ActivationFunction::Softplus,
            _ => rusty_neat::genes::ActivationFunction::SignedSigmoid,
        };
        Ok(())
    }

    fn get_hidden_nodes_activation(&self) -> PyResult<String> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(format!("{:?}", s.hidden_nodes_activation))
    }

    fn set_hidden_nodes_activation(&mut self, name: &str) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.hidden_nodes_activation = match name {
            "UnsignedSigmoid" => rusty_neat::genes::ActivationFunction::UnsignedSigmoid,
            "Tanh" => rusty_neat::genes::ActivationFunction::Tanh,
            "Linear" => rusty_neat::genes::ActivationFunction::Linear,
            "Relu" => rusty_neat::genes::ActivationFunction::Relu,
            "Softplus" => rusty_neat::genes::ActivationFunction::Softplus,
            _ => rusty_neat::genes::ActivationFunction::SignedSigmoid,
        };
        Ok(())
    }

    fn get_custom_conn_obeys_flags(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.custom_conn_obeys_flags)
    }

    fn set_custom_conn_obeys_flags(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.custom_conn_obeys_flags = v;
        Ok(())
    }

    // Connectivity flags
    fn get_allow_input_hidden_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_input_hidden_links)
    }

    fn set_allow_input_hidden_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_input_hidden_links = v;
        Ok(())
    }

    fn get_allow_input_output_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_input_output_links)
    }

    fn set_allow_input_output_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_input_output_links = v;
        Ok(())
    }

    fn get_allow_hidden_hidden_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_hidden_hidden_links)
    }

    fn set_allow_hidden_hidden_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_hidden_hidden_links = v;
        Ok(())
    }

    fn get_allow_hidden_output_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_hidden_output_links)
    }

    fn set_allow_hidden_output_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_hidden_output_links = v;
        Ok(())
    }

    fn get_allow_output_hidden_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_output_hidden_links)
    }

    fn set_allow_output_hidden_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_output_hidden_links = v;
        Ok(())
    }

    fn get_allow_output_output_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_output_output_links)
    }

    fn set_allow_output_output_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_output_output_links = v;
        Ok(())
    }

    fn get_allow_looped_hidden_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_looped_hidden_links)
    }

    fn set_allow_looped_hidden_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_looped_hidden_links = v;
        Ok(())
    }

    fn get_allow_looped_output_links(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.allow_looped_output_links)
    }

    fn set_allow_looped_output_links(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.allow_looped_output_links = v;
        Ok(())
    }

    fn get_query_weights_only(&self) -> PyResult<bool> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.query_weights_only)
    }

    fn set_query_weights_only(&mut self, v: bool) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.query_weights_only = v;
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

    /// Support pickling for Substrate: produce a dict with all relevant fields
    fn __getstate__(&self, py: Python) -> PyObject {
        let s = match self.inner.read() {
            Ok(v) => v,
            Err(_) => panic!("lock poisoned"),
        };
        let d = PyDict::new(py);
        let _ = d.set_item("input_coords", s.input_coords.clone());
        let _ = d.set_item("hidden_coords", s.hidden_coords.clone());
        let _ = d.set_item("output_coords", s.output_coords.clone());

        let _ = d.set_item("leaky", s.leaky);
        let _ = d.set_item("with_distance", s.with_distance);

        let _ = d.set_item("min_time_const", s.min_time_const);
        let _ = d.set_item("max_time_const", s.max_time_const);
        let _ = d.set_item("max_weight_and_bias", s.max_weight_and_bias);

        let _ = d.set_item(
            "output_nodes_activation",
            format!("{:?}", s.output_nodes_activation),
        );
        let _ = d.set_item(
            "hidden_nodes_activation",
            format!("{:?}", s.hidden_nodes_activation),
        );

        let _ = d.set_item("custom_connectivity", s.custom_connectivity.clone());
        let _ = d.set_item("custom_conn_obeys_flags", s.custom_conn_obeys_flags);

        // connectivity flags
        let _ = d.set_item("allow_input_hidden_links", s.allow_input_hidden_links);
        let _ = d.set_item("allow_input_output_links", s.allow_input_output_links);
        let _ = d.set_item("allow_hidden_hidden_links", s.allow_hidden_hidden_links);
        let _ = d.set_item("allow_hidden_output_links", s.allow_hidden_output_links);
        let _ = d.set_item("allow_output_hidden_links", s.allow_output_hidden_links);
        let _ = d.set_item("allow_output_output_links", s.allow_output_output_links);
        let _ = d.set_item("allow_looped_hidden_links", s.allow_looped_hidden_links);
        let _ = d.set_item("allow_looped_output_links", s.allow_looped_output_links);

        let _ = d.set_item("query_weights_only", s.query_weights_only);

        d.to_object(py)
    }

    /// Restore Substrate from dict produced by __getstate__
    fn __setstate__(&mut self, state: &PyAny) -> PyResult<()> {
        let d = state
            .downcast::<PyDict>()
            .map_err(|_| PyRuntimeError::new_err("__setstate__ expects a dict"))?;
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;

        if let Some(v) = d.get_item("input_coords") {
            s.input_coords = v.extract::<Vec<Vec<f64>>>().map_err(|e| {
                PyRuntimeError::new_err(format!("input_coords must be nested float lists: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("hidden_coords") {
            s.hidden_coords = v.extract::<Vec<Vec<f64>>>().map_err(|e| {
                PyRuntimeError::new_err(format!("hidden_coords must be nested float lists: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("output_coords") {
            s.output_coords = v.extract::<Vec<Vec<f64>>>().map_err(|e| {
                PyRuntimeError::new_err(format!("output_coords must be nested float lists: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("leaky") {
            s.leaky = v
                .extract::<bool>()
                .map_err(|e| PyRuntimeError::new_err(format!("leaky must be bool: {}", e)))?;
        }
        if let Some(v) = d.get_item("with_distance") {
            s.with_distance = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("with_distance must be bool: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("min_time_const") {
            s.min_time_const = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("min_time_const must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_time_const") {
            s.max_time_const = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_time_const must be float: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("max_weight_and_bias") {
            s.max_weight_and_bias = v.extract::<f64>().map_err(|e| {
                PyRuntimeError::new_err(format!("max_weight_and_bias must be float: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("output_nodes_activation") {
            let name = v.extract::<String>().map_err(|e| {
                PyRuntimeError::new_err(format!("output_nodes_activation must be string: {}", e))
            })?;
            s.output_nodes_activation = match name.as_str() {
                "UnsignedSigmoid" => rusty_neat::genes::ActivationFunction::UnsignedSigmoid,
                "Tanh" => rusty_neat::genes::ActivationFunction::Tanh,
                "Linear" => rusty_neat::genes::ActivationFunction::Linear,
                "Relu" => rusty_neat::genes::ActivationFunction::Relu,
                "Softplus" => rusty_neat::genes::ActivationFunction::Softplus,
                _ => rusty_neat::genes::ActivationFunction::SignedSigmoid,
            };
        }

        if let Some(v) = d.get_item("hidden_nodes_activation") {
            let name = v.extract::<String>().map_err(|e| {
                PyRuntimeError::new_err(format!("hidden_nodes_activation must be string: {}", e))
            })?;
            s.hidden_nodes_activation = match name.as_str() {
                "UnsignedSigmoid" => rusty_neat::genes::ActivationFunction::UnsignedSigmoid,
                "Tanh" => rusty_neat::genes::ActivationFunction::Tanh,
                "Linear" => rusty_neat::genes::ActivationFunction::Linear,
                "Relu" => rusty_neat::genes::ActivationFunction::Relu,
                "Softplus" => rusty_neat::genes::ActivationFunction::Softplus,
                _ => rusty_neat::genes::ActivationFunction::SignedSigmoid,
            };
        }

        if let Some(v) = d.get_item("custom_connectivity") {
            s.custom_connectivity = v.extract::<Vec<Vec<i32>>>().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "custom_connectivity must be nested int lists: {}",
                    e
                ))
            })?;
        }
        if let Some(v) = d.get_item("custom_conn_obeys_flags") {
            s.custom_conn_obeys_flags = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("custom_conn_obeys_flags must be bool: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("allow_input_hidden_links") {
            s.allow_input_hidden_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_input_hidden_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_input_output_links") {
            s.allow_input_output_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_input_output_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_hidden_hidden_links") {
            s.allow_hidden_hidden_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_hidden_hidden_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_hidden_output_links") {
            s.allow_hidden_output_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_hidden_output_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_output_hidden_links") {
            s.allow_output_hidden_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_output_hidden_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_output_output_links") {
            s.allow_output_output_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_output_output_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_looped_hidden_links") {
            s.allow_looped_hidden_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_looped_hidden_links must be bool: {}", e))
            })?;
        }
        if let Some(v) = d.get_item("allow_looped_output_links") {
            s.allow_looped_output_links = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("allow_looped_output_links must be bool: {}", e))
            })?;
        }

        if let Some(v) = d.get_item("query_weights_only") {
            s.query_weights_only = v.extract::<bool>().map_err(|e| {
                PyRuntimeError::new_err(format!("query_weights_only must be bool: {}", e))
            })?;
        }

        Ok(())
    }
}

/// Thin PyO3 wrapper for random utilities (rusty_neat::random::Random)
#[pyclass(module = "rusty_neat_py")]
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

// Helper: convert Rust TraitParameters -> Python dict
fn trait_parameters_to_pydict(py: Python, tp: &rusty_neat::genes::TraitParameters) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item("importance_coeff", tp.importance_coeff);
    let _ = dict.set_item("mutation_prob", tp.mutation_prob);

    // dep_key (optional)
    if let Some(k) = &tp.dep_key {
        let _ = dict.set_item("dep_key", k);
    } else {
        let _ = dict.set_item("dep_key", py.None());
    }

    // dep_values: list
    let dv_list = PyList::empty(py);
    for dv in &tp.dep_values {
        match dv {
            rusty_neat::genes::TraitValue::Int(i) => {
                dv_list.append(i).ok();
            }
            rusty_neat::genes::TraitValue::Float(f) => {
                dv_list.append(f).ok();
            }
            rusty_neat::genes::TraitValue::Str(s) => {
                dv_list.append(s).ok();
            }
            rusty_neat::genes::TraitValue::Bool(b) => {
                dv_list.append(b).ok();
            }
        }
    }
    let _ = dict.set_item("dep_values", dv_list);

    // detail
    let detail = PyDict::new(py);
    match &tp.detail {
        rusty_neat::genes::TraitDetail::Int {
            min,
            max,
            mut_power,
            mut_replace_prob,
        } => {
            let _ = detail.set_item("type", "Int");
            let _ = detail.set_item("min", *min);
            let _ = detail.set_item("max", *max);
            let _ = detail.set_item("mut_power", *mut_power);
            let _ = detail.set_item("mut_replace_prob", *mut_replace_prob);
        }
        rusty_neat::genes::TraitDetail::Float {
            min,
            max,
            mut_power,
            mut_replace_prob,
        } => {
            let _ = detail.set_item("type", "Float");
            let _ = detail.set_item("min", *min);
            let _ = detail.set_item("max", *max);
            let _ = detail.set_item("mut_power", *mut_power);
            let _ = detail.set_item("mut_replace_prob", *mut_replace_prob);
        }
        rusty_neat::genes::TraitDetail::Str { set, probs } => {
            let _ = detail.set_item("type", "Str");
            let _ = detail.set_item("set", set.clone());
            let _ = detail.set_item("probs", probs.clone());
        }
        rusty_neat::genes::TraitDetail::IntSet { set, probs } => {
            let _ = detail.set_item("type", "IntSet");
            let _ = detail.set_item("set", set.clone());
            let _ = detail.set_item("probs", probs.clone());
        }
        rusty_neat::genes::TraitDetail::FloatSet { set, probs } => {
            let _ = detail.set_item("type", "FloatSet");
            let _ = detail.set_item("set", set.clone());
            let _ = detail.set_item("probs", probs.clone());
        }
    }
    let _ = dict.set_item("detail", detail);

    dict.to_object(py)
}

// Helper: convert Python dict -> Rust TraitParameters
fn pydict_to_trait_parameters(params: &PyAny) -> PyResult<rusty_neat::genes::TraitParameters> {
    let d = params
        .downcast::<PyDict>()
        .map_err(|_| PyRuntimeError::new_err("expected dict for TraitParameters"))?;

    let importance_coeff: f64 = match d.get_item("importance_coeff") {
        Some(v) => v.extract::<f64>().map_err(|e| {
            PyRuntimeError::new_err(format!("importance_coeff must be float: {}", e))
        })?,
        None => 1.0,
    };
    let mutation_prob: f64 = match d.get_item("mutation_prob") {
        Some(v) => v
            .extract::<f64>()
            .map_err(|e| PyRuntimeError::new_err(format!("mutation_prob must be float: {}", e)))?,
        None => 0.0,
    };

    // dep_key
    let dep_key: Option<String> = match d.get_item("dep_key") {
        Some(v) => {
            if v.is_none() {
                None
            } else {
                Some(v.extract::<String>().map_err(|e| {
                    PyRuntimeError::new_err(format!("dep_key must be string: {}", e))
                })?)
            }
        }
        None => None,
    };

    // dep_values
    let mut dep_values: Vec<rusty_neat::genes::TraitValue> = Vec::new();
    if let Some(v) = d.get_item("dep_values") {
        if !v.is_none() {
            let seq = v
                .downcast::<PyList>()
                .map_err(|_| PyRuntimeError::new_err("dep_values must be a list"))?;
            for item in seq.iter() {
                // try types in order: int, float, str, bool
                if let Ok(i) = item.extract::<i64>() {
                    dep_values.push(rusty_neat::genes::TraitValue::Int(i));
                    continue;
                }
                if let Ok(f) = item.extract::<f64>() {
                    dep_values.push(rusty_neat::genes::TraitValue::Float(f));
                    continue;
                }
                if let Ok(s) = item.extract::<String>() {
                    dep_values.push(rusty_neat::genes::TraitValue::Str(s));
                    continue;
                }
                if let Ok(b) = item.extract::<bool>() {
                    dep_values.push(rusty_neat::genes::TraitValue::Bool(b));
                    continue;
                }
                return Err(PyRuntimeError::new_err("unsupported type in dep_values"));
            }
        }
    }

    // detail
    let detail_any = d
        .get_item("detail")
        .ok_or_else(|| PyRuntimeError::new_err("detail key required in TraitParameters"))?;
    let detail_dict = detail_any
        .downcast::<PyDict>()
        .map_err(|_| PyRuntimeError::new_err("detail must be a dict"))?;
    let ttype = detail_dict
        .get_item("type")
        .and_then(|v| v.extract::<String>().ok())
        .ok_or_else(|| PyRuntimeError::new_err("detail.type must be a string"))?;

    let detail =
        match ttype.as_str() {
            "Int" => {
                let min = detail_dict
                    .get_item("min")
                    .and_then(|v| v.extract::<i64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Int.detail.min required and must be int")
                    })?;
                let max = detail_dict
                    .get_item("max")
                    .and_then(|v| v.extract::<i64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Int.detail.max required and must be int")
                    })?;
                let mut_power = detail_dict
                    .get_item("mut_power")
                    .and_then(|v| v.extract::<i64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Int.detail.mut_power required and must be int")
                    })?;
                let mut_replace_prob = detail_dict
                    .get_item("mut_replace_prob")
                    .and_then(|v| v.extract::<f64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "Int.detail.mut_replace_prob required and must be float",
                        )
                    })?;
                rusty_neat::genes::TraitDetail::Int {
                    min,
                    max,
                    mut_power,
                    mut_replace_prob,
                }
            }
            "Float" => {
                let min = detail_dict
                    .get_item("min")
                    .and_then(|v| v.extract::<f64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Float.detail.min required and must be float")
                    })?;
                let max = detail_dict
                    .get_item("max")
                    .and_then(|v| v.extract::<f64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Float.detail.max required and must be float")
                    })?;
                let mut_power = detail_dict
                    .get_item("mut_power")
                    .and_then(|v| v.extract::<f64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("Float.detail.mut_power required and must be float")
                    })?;
                let mut_replace_prob = detail_dict
                    .get_item("mut_replace_prob")
                    .and_then(|v| v.extract::<f64>().ok())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "Float.detail.mut_replace_prob required and must be float",
                        )
                    })?;
                rusty_neat::genes::TraitDetail::Float {
                    min,
                    max,
                    mut_power,
                    mut_replace_prob,
                }
            }
            "Str" => {
                let set_any = detail_dict
                    .get_item("set")
                    .ok_or_else(|| PyRuntimeError::new_err("Str.detail.set required"))?;
                let set_list = set_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("Str.detail.set must be list of strings")
                })?;
                let mut set: Vec<String> = Vec::new();
                for it in set_list.iter() {
                    set.push(it.extract::<String>().map_err(|_| {
                        PyRuntimeError::new_err("Str.detail.set must contain strings")
                    })?);
                }
                let probs_any = detail_dict
                    .get_item("probs")
                    .ok_or_else(|| PyRuntimeError::new_err("Str.detail.probs required"))?;
                let probs_list = probs_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("Str.detail.probs must be list of floats")
                })?;
                let mut probs: Vec<f64> = Vec::new();
                for it in probs_list.iter() {
                    probs.push(it.extract::<f64>().map_err(|_| {
                        PyRuntimeError::new_err("Str.detail.probs must contain floats")
                    })?);
                }
                rusty_neat::genes::TraitDetail::Str { set, probs }
            }
            "IntSet" => {
                let set_any = detail_dict
                    .get_item("set")
                    .ok_or_else(|| PyRuntimeError::new_err("IntSet.detail.set required"))?;
                let set_list = set_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("IntSet.detail.set must be list of ints")
                })?;
                let mut set: Vec<i64> = Vec::new();
                for it in set_list.iter() {
                    set.push(it.extract::<i64>().map_err(|_| {
                        PyRuntimeError::new_err("IntSet.detail.set must contain ints")
                    })?);
                }
                let probs_any = detail_dict
                    .get_item("probs")
                    .ok_or_else(|| PyRuntimeError::new_err("IntSet.detail.probs required"))?;
                let probs_list = probs_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("IntSet.detail.probs must be list of floats")
                })?;
                let mut probs: Vec<f64> = Vec::new();
                for it in probs_list.iter() {
                    probs.push(it.extract::<f64>().map_err(|_| {
                        PyRuntimeError::new_err("IntSet.detail.probs must contain floats")
                    })?);
                }
                rusty_neat::genes::TraitDetail::IntSet { set, probs }
            }
            "FloatSet" => {
                let set_any = detail_dict
                    .get_item("set")
                    .ok_or_else(|| PyRuntimeError::new_err("FloatSet.detail.set required"))?;
                let set_list = set_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("FloatSet.detail.set must be list of floats")
                })?;
                let mut set: Vec<f64> = Vec::new();
                for it in set_list.iter() {
                    set.push(it.extract::<f64>().map_err(|_| {
                        PyRuntimeError::new_err("FloatSet.detail.set must contain floats")
                    })?);
                }
                let probs_any = detail_dict
                    .get_item("probs")
                    .ok_or_else(|| PyRuntimeError::new_err("FloatSet.detail.probs required"))?;
                let probs_list = probs_any.downcast::<PyList>().map_err(|_| {
                    PyRuntimeError::new_err("FloatSet.detail.probs must be list of floats")
                })?;
                let mut probs: Vec<f64> = Vec::new();
                for it in probs_list.iter() {
                    probs.push(it.extract::<f64>().map_err(|_| {
                        PyRuntimeError::new_err("FloatSet.detail.probs must contain floats")
                    })?);
                }
                rusty_neat::genes::TraitDetail::FloatSet { set, probs }
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown detail.type: {}",
                    other
                )))
            }
        };

    Ok(rusty_neat::genes::TraitParameters {
        importance_coeff,
        mutation_prob,
        detail,
        dep_key,
        dep_values,
    })
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
