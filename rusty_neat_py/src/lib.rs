use pyo3::prelude::*;

/// Minimal PyGenome skeleton
#[pyclass]
#[derive(Clone)]
pub struct PyGenome {
    id: usize,
}

#[pymethods]
impl PyGenome {
    #[new]
    fn new() -> Self {
        PyGenome { id: 0 }
    }

    fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    fn get_id(&self) -> usize {
        self.id
    }
}

/// Minimal PyNeuralNetwork skeleton
#[pyclass]
pub struct PyNeuralNetwork {
    // placeholder
}

#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new() -> Self {
        PyNeuralNetwork {}
    }

    /// Activate with a Python sequence of floats and return a Vec<f64>
    fn activate(&self, py: Python, inputs: &PyAny) -> PyResult<PyObject> {
        let seq = inputs.extract::<Vec<f64>>()?;
        Ok(seq.to_object(py))
    }
}

/// Module definition
#[pymodule]
fn rusty_neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGenome>()?;
    m.add_class::<PyNeuralNetwork>()?;
    Ok(())
}
