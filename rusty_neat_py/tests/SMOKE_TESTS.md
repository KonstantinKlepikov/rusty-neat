# Smoke Tests для rusty_neat_py

Smoke тесты адаптированы из примеров `cneat/MultiNEAT/examples` для проверки работоспособности Rust биндингов.

## Структура тестов

### Smoke тесты (быстрые)

Помечены маркером `@pytest.mark.smoke` - выполняются быстро (секунды):

- `test_neat_xor_basic_evolution` - базовая проверка эволюции NEAT на XOR (10 поколений)
- `test_hyperneat_xor_basic_evolution` - базовая проверка эволюции HyperNEAT на XOR (5 поколений)
- `test_substrate_setup` - проверка настройки Substrate

### Медленные тесты

Помечены маркером `@pytest.mark.slow` - могут занимать минуты:

- `test_neat_xor_single_run` - полное решение XOR с помощью NEAT (до 500 поколений)
- `test_hyperneat_xor_single_run` - полное решение XOR с помощью HyperNEAT (до 500 поколений)

## Запуск тестов

### Все тесты

```bash
pytest tests/test_*_smoke.py -v
```

### Только быстрые smoke тесты

```bash
pytest tests/test_*_smoke.py -v -m smoke
```

### Только медленные тесты

```bash
pytest tests/test_*_smoke.py -v -m slow
```

### Исключить медленные тесты

```bash
pytest tests/test_*_smoke.py -v -m "not slow"
```

## Адаптация из MultiNEAT

Основные изменения при адаптации:

1. **Импорты**: `MultiNEAT` → `rusty_neat_py`
2. **Классы**: `NEAT.NeuralNetwork()` → `rn.PyNeuralNetwork()`
3. **Методы**: `genome.BuildPhenotype()` → `genome.build_phenotype()`
4. **Population API**: Используется `pop.access_genome_by_index()` вместо `GetGenomeList/ZipFitness`
5. **Substrate**: `NEAT.Substrate(...)` → `rn.PySubstrate.with_coords(...)`
6. **Активация**: `net.Flush()` → `net.reset()` для очистки состояния сети

## Соответствие примерам

- `test_neat_xor_smoke.py` ← `TestNEAT_xor.py`
- `test_hyperneat_xor_smoke.py` ← `TestHyperNEAT_xor.py`

## CI Integration

Рекомендуется:

- В PR проверках запускать только smoke тесты (`-m smoke`)
- В nightly builds запускать все тесты включая slow
- Использовать pytest-timeout для ограничения времени выполнения

Пример в GitHub Actions:

```yaml
- name: Run smoke tests
  run: |
    source .venv/bin/activate
    pytest tests/test_*_smoke.py -v -m smoke

- name: Run slow tests (nightly)
  if: github.event_name == 'schedule'
  run: |
    source .venv/bin/activate
    pytest tests/test_*_smoke.py -v -m slow --timeout=600
```
