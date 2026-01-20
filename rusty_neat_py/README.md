# Rusty-NEAT-py

Python binding to Rusty-NEAT.

## Development

Build and install the Python extension into the current virtualenv (recommended via Poetry):

```bash
# from the `rusty_neat_py` crate directory
poetry run maturin develop --release
```

Basic smoke test (Python)

After a successful `maturin develop`, you can run a small smoke test to verify the module imports and basic API surface:

```python
# smoke_test.py
import rusty_neat_py

def smoke():

    # create basic objects
    params = rusty_neat_py.PyParameters()
    print('min_weight:', params.get_min_weight())

    substrate = rusty_neat_py.PySubstrate()
    print('min cppn inputs:', substrate.get_min_cppn_inputs())

    rng = rusty_neat_py.PyRNG()
    print('rand float:', rng.rand_float())

    nn = rusty_neat_py.PyNeuralNetwork()
    nn.input([0.1, 0.2, 0.3])
    nn.activate()
    out = nn.output()
    print('nn output:', out)

    g = rusty_neat_py.PyGenome(1, 3, 0, 1)
    print('genome id:', g.get_id())

if __name__ == '__main__':
    smoke()
```

Run it with the same Python interpreter that has the extension installed (usually the Poetry environment):

```bash
# inside the `rusty_neat_py` directory (same env as used for maturin develop)
poetry run python smoke_test.py
```

Notes

- If you prefer not to use Poetry you can run `maturin develop --release` in any active virtualenv. The `poetry run` wrapper ensures the wheel is built against the environment Poetry manages.
- For development cycles use `maturin develop` (without --release) to speed up builds.
