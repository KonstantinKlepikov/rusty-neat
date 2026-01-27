# Rusty-NEAT

Multy-NEAT rust implementation.

Warning: Pickle files produced by the original C++ MultiNEAT are not guaranteed to be compatible with Rusty-NEAT.

Pickle files produced by the original C++ implementation of MultiNEAT are not guaranteed to be compatible with `rusty_neat` serialization. Directly loading C++ pickles with the Python bindings (`rusty_neat_py`) may result in deserialization errors. We recommend exporting data from the C++ implementation to JSON (or another neutral format) or using the provided migration utilities when transferring data between implementations.

Detailed recommendation (what to document and how):

- Always include explicit `format` and `schema_version` fields in `__getstate__/__setstate__` payloads to distinguish C++ pickles from Rust pickles.
- Add detection logic and a friendly error when a C++ pickle is encountered, for example:

```py
state = pickle.loads(raw)
fmt = state.get('format')
ver = state.get('schema_version')
if fmt == 'custom_cxx' or (fmt is None and looks_like_cxx(raw)):
    raise RuntimeError('C++ pickle detected: not supported. Use migration tools.')
```
