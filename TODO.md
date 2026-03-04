# Этапы переписывания MultiNEAT с C++ на Rust

- [x] Анализ исходного кода и структуры проекта
  - [x] Изучите содержимое папки `cneat/MultiNEAT`, включая папки `src`, `MultiNEAT/`, а также файлы `CMakeLists.txt`, `setup.py`, `README.md`.
  - [x] Определите основные модули, классы и функции, реализующие ключевую логику NEAT/HyperNEAT.

- [x] Определение публичного API
  - [x] Ознакомьтесь с Python-обёрткой (`_MultiNEAT.pyx`, `setup.py`) и публичными интерфейсами, используемыми пользователями библиотеки.
  - [x] Зафиксируйте, какие функции и классы должны быть доступны в Rust-версии.

- [x] Планирование архитектуры на Rust
  - [x] Решите, будет ли библиотека только на Rust или потребуется поддержка FFI (например, для Python через PyO3/maturin).
  - [x] Определите основные модули и структуры данных в стиле Rust (структуры, трейты, enum).

- [x] Постепенное переписывание модулей
  - [x] Начните с базовых структур данных (`Genome`, `Network`, `Population` и т.д.).
    - [x] Genome
      - [x] GenomeSeedType
      - [x] GenomeInitStruct
      - [x] Genome
      - [x] Конструкторы и базовые методы
      - [x] Доступ к генам и параметрам
      - [x] Построение фенотипа
        - [x] `BuildPhenotype(NeuralNetwork&)`
        - [x] `BuildHyperNEATPhenotype(NeuralNetwork&, Substrate&)`
        - [x] `DerivePhenotypicChanges(NeuralNetwork&)`
  - [x] Перенесите алгоритмы эволюции, мутаций, кроссовера, расчёта фитнеса.
    - [x] `Mutate_AddNeuron`, `Mutate_AddLink`, `Mutate_RemoveLink`, `Mutate_RemoveSimpleNeuron`
    - [x] `Mutate_LinkWeights`, `Randomize_LinkWeights`
    - [x] `Mutate_NeuronActivations_A/B`, `Mutate_NeuronActivation_Type`
    - [x] `Mutate_NeuronTimeConstants`, `Mutate_NeurонBiases`
    - [x] `Mutate_NeuronTraits`, `Mutate_LinkTraits`, `Mutate_GenomeTraits`
    - [x] `Randomize_Traits`
  - [x] Реализуйте вспомогательные функции (генераторы случайных чисел, сериализацию).

- [x] Покрытие тестами
  - [x] Для каждого переписанного модуля пишите модульные тесты на Rust, используя примеры из `cneat/MultiNEAT/examples`.
  - [x] Сравнивайте результаты работы Rust-версии с оригинальной C++.
  - [x] Рекомендуемые подпункты по приоритету:
    - [x] Substrate / HyperNEAT helpers — unit-тесты для `get_max_dims`, `get_min_cppn_inputs`, `get_min_cppn_outputs`, `with_coords`, `set_neurons`, `print_info`.
    - [x] BuildHyperNEATPhenotype — быстрый smoke-test: детерминированный CPPN/phenotype, проверка количества связей, фильтрации по флагам и масштабирования весов (`max_weight_and_bias`).
    - [x] Traits behaviour — unit-тесты для `init_trait_map` и `mutate_trait_map`: replace vs perturb, clamping, dep_key/dep_values, roulette для set/string.
    - [x] PhenotypeBehavior — тесты контрактов: `acquire()` (false по-умолчанию), `distance_to()` (0 для идентичных `m_data`), `successful()` (true по-умолчанию), сравнение `m_data`.
    - [x] Parameters ↔ Traits integration — небольшие интеграционные тесты: `SetNeuronTraitParameters`/`SetLinkTraitParameters` + `Genome::randomize_traits`.
    - [x] ES-HyperNEAT (опционально) — smoke-тесты для ES-параметров (DivisionThreshold, IterationLevel, MaxDepth); эти тесты можно помечать `#[ignore]` или запускать отдельно.
    - [x] ~~Serialization / pickling (опционально) — round-trip тесты `Substrate` и простых `Genome`; pickling-совместимость с Python — отдельная интеграция~~.
    - [x] ~~CI policy — определить бюджет времени и пометить тяжёлые эволюционные примеры как `#[ignore]` или вынести в nightly/integration workflow~~.

- [ ] Интеграция и сборка
  - [x] Используйте `Cargo.toml` для управления зависимостями и сборкой.
  - [x] Для поддержки Python добавьте PyO3 и настройте сборку через `maturin` или `wasm-pack`.

    Этапы реализации биндингов на Python (подпункты)

    - Фаза 0 — Инвентаризация и согласование API

      - [x] Собрать полный список экспортируемых типов и методов (см. `research/python_integration.md`).
      - [x] Решить совместимость pickle/серриализации и ожидания NumPy I/O. Сохранено в `research/pickle.md`.
      - [x] Составить таблицу маппинга: C++ сигнатура → Rust type → PyO3 signature (включая типы NumPy). Сохранено в `research/bind_map.md`.

    - Фаза 1 — Дизайн API

      - [x] Решить контракты владения/потоков: `Arc<Mutex<T>>` vs `PyRef` для каждого экспортируемого класса.
      - [x] Спроектировать Python‑уровень: какие поля делать свойствами, какие методы — напрямую, какие пакеты — упаковывать в маленькие Py‑классы (LinkGene/NeuronGene). Сохранено в `research/python_integration.md`.
      - [x] Документировать соглашения по Pickle/Сериализации (совместимость C++ → Rust — опционально). Сохранено в `research/pickle.md`.

    - Фаза 2 — Создание skeleton crate `rusty_neat_py`

      - [x] Создать `rusty_neat_py` с `Cargo.toml` и `src/lib.rs` (#[pymodule]).
      - [x] Реализовать минимальные `PyGenome` и `PyNeuralNetwork` как скелетоны для контрактов и smoke-tests.
      - [x] Создать crate в workspace с `crate-type = ["cdylib"]`, добавить зависимости `pyo3`, `pyo3-ndarray`, `ndarray`, и локальную зависимость `rusty_neat`.
      - [x] Добавить `#[pymodule]` и регистрации базовых классов (`PyGenome`, `PyNeuralNetwork`, `PyParameters`, `PySubstrate`, `PyRNG`) с минимальными конструкторами/методами.
      - [x] Добавить пример `maturin develop` запуск в README и basic smoke example на Python.

    - Фаза 3 — Ядро биндингов

      - [x] Реализовать `PyNeuralNetwork` с поддержкой `Input` из Python list и `PyArray` (через `pyo3-ndarray`), `Activate`, `Output` и сохранение/загрузку.
      - [x] Реализовать `PyGenome` с `BuildPhenotype` и `BuildHyperNEATPhenotype` (принятие `PySubstrate`).
      - [x] Экспортировать ключевые структуры: `LinkGene`, `NeuronGene`, `GenomeInitStruct` (как простые `pyclass`/namedtuple).
      - [x] Реализовать минимальные маппинги контейнеров (vec ↔ list) и provide light iterators for large lists.

    - Фаза 4 — Traits / Parameters / Substrate

      - [x] Экспортировать `Parameters` с API trait management: `ListNeuronTraitParameters`, `SetNeuronTraitParameters`, `GetNeuronTraitParameters` и т.д.
      - [x] Реализовать `PySubstrate` с флагами (allow_input_output_links ...) и методами `GetMinCPPNInputs/GetMinCPPNOutputs`.
      - [x] Pickling для `Parameters`/`Substrate` (implement `__getstate__/__setstate__`).

    - Фаза 5 — Population / Species / PhenotypeBehavior

      - [x] Реализовать `PyPopulation` с методами `Epoch`, `Tick`, `GetBestGenome` и доступом к `Species`/`Genome` через ссылочные обёртки.
      - [x] Экспорт `PhenotypeBehavior` и его контрактных методов (Acquire, Distance_To, Successful).
      - [x] Подумать про освобождение GIL в тяжёлых операциях (use `py.allow_threads`) - сохранено в `research/gil_recomendation.md`.

    - Фаза 6 — Тесты и CI
      - [x] Добавить pytest smoke tests, взяв примеры из `cneat/MultiNEAT/examples` (TestNEAT_xor, TestHyperNEAT_xor и др.) и адаптировать их для Rust биндингов.
      - [x] Добавить GitHub Actions job: `maturin build` → `pip install target/wheels/*.whl` → `pytest`.
      - [x] Маркировать тяжёлые/долгие тесты как `@pytest.mark.slow` или запускать в nightly workflow.

    - Фаза 7 — Release и совместимость
      - [x] Выстроить политику по версии API и декларацию несовместимости pickles с C++ при наличии.
      - [x] Подготовить `maturin` сборки для целевых платформ и инструкцию публикации на PyPI. сохранено в `research/pypi.md`.

    - Риски и замечания
      - [x] ~~Copy overhead: для больших векторов отдавать предпочтение `PyArray`/views~~.
      - [x] ~~Pickle‑совместимость с C++ редко достижима «бесплатно» — документировать и предоставить миграционные утилиты при необходимости~~.

    - [ ] Добавить стабы для типизации python апи

      Приоритетный минимальный набор API для релиза 1.0 (порядок по приоритету):

      1) Genome (core)
          - Конструкторы: `Genome()` и `Genome(const char* filename)` или конструктор по `Parameters`+`GenomeInitStruct`.
          - Ключевые методы: `BuildPhenotype(NeuralNetwork&)`, `BuildHyperNEATPhenotype(NeuralNetwork&, Substrate&)`, `GetFitness`/`SetFitness`, `Save`/`Load`, `Randomize_LinkWeights`, `Randomize_Traits`, основные `Mutate_*` (минимум: `Mutate_NeuronActivations_A/B`, `Mutate_NeuronActivation_Type`, `Mutate_NeuronTimeConstants`, `Mutate_NeuronBiases`).
          - Поля/доступ: `NeuronGenes`, `LinkGenes` (как коллекции/итерируемые объекты).

      2) NeuralNetwork (I/O + execution)
          - Конструкторы: `NeuralNetwork()` и `NeuralNetwork(bool)`.
          - Ключевые методы: `Input` (поддержка Python list и NumPy), `Activate` / `ActivateFast` / `ActivateLeaky`, `Output`, `Clear`, `Save`/`Load`.
          - Helpers: `SetInputOutputDimentions`, `NumInputs`, `NumOutputs`.
          - Поля: `m_neurons`, `m_connections` (как опциональные для инспекции).

      3) Parameters (configuration + traits API)
          - Методы: `Load`, `Save`, `Reset`.
          - Критичные поля: базовые параметры генерации/мутаций (PopulationSize, MutateAddLinkProb, MutateWeightsProb и т.д.) и HyperNEAT/ES параметры (DivisionThreshold, MaxDepth, IterationLevel, CPPN_Bias, Width/Height, Leo*).
          - Traits API: `ListNeuronTraitParameters`, `SetNeuronTraitParameters`, `GetNeuronTraitParameters` (и аналогично для Link/Genome) — экспортовать как удобные методы.

      4) Substrate (HyperNEAT substrate)
          - Конструктор: `Substrate(inputs, hidden, outputs)` (принимает координаты).
          - Методы: `GetMinCPPNInputs`, `GetMinCPPNOutputs`, `PrintInfo`, `SetCustomConnectivity`, `ClearCustomConnectivity`.
          - Параметры/флаги: разрешения связей (`m_allow_*`), `m_max_weight_and_bias`, координаты узлов.

      5) RNG (utility)
          - Методы: `Seed`, `TimeSeed`, `RandInt`, `RandFloat`, `RandFloatSigned`, `RandGaussSigned`, `Roulette`.
          - NOTE: Отсутствует (не реализовано): `Seed`, `TimeSeed`, `RandFloatSigned`, `RandGaussSigned`, `Roulette`. Причина: в `rusty_neat::random` сейчас реализованы только `rand_float()` и `rand_int()`, поэтому в PyRNG экспортированы только они. Можно добавить недостающие методы в `rusty_neat::random` (например, `seed(u64)`, `time_seed()`, `rand_float_signed()`, `rand_gauss_signed()`, `roulette(weights: &[f64])`) и экспортировать их в PyRNG с соответствующими тестами и документацией.

      6) Дополнительно (низкий приоритет для 1.0, но полезно скоро):
          - Population: `Epoch`, `Tick`, `GetBestGenome`, `AccessGenomeByIndex` (можно отложить в 1.1 при необходимости).
          - Species: минимальный доступ к `Individuals` и `GetLeader`.

- [ ] Документация и примеры
  - [ ] Перенесите и адаптируйте документацию из оригинального `README.md` и примеры.
  - [ ] Обновите `README.md` для Rust-версии.

- [ ] Автоматизация и CI
  - [x] Настройте тестирование и сборку через GitHub Actions (см. примеры в `.github/workflows`).
  - [x] ~~Добавьте сборку для разных платформ и, при необходимости, публикацию бинарников/библиотек~~.
  - [ ] добавить релиз со сборкой и публикацией на pypi

**Рекомендуемый порядок переписывания:**

1. Базовые структуры данных (геномы, нейроны, связи)
2. Алгоритмы эволюции (мутации, кроссовер, селекция)
3. Симуляция сети (активация, расчёт выхода)
4. Вспомогательные утилиты (рандомизация, сериализация)
5. Публичный API и интеграция с внешними языками (Python, WASM)
6. Документация и примеры

## Чеклист ручной проверки проекта `rusty-neat`

Ниже — упорядоченный чеклист для ручной проверки модулей на Rust, Rust‑тестов, Python‑биндингов и Python‑тестов. Порядок проверок даёт минимальную зависимую последовательность (сначала core crates, затем биндинги и тесты).

### 1. Crates / сборка

- [x] `cargo build --release` — билд успешен для всех crate'ов в workspace
- [x] `cargo test` — базовые unit‑тесты Rust проходят (см. раздел Rust‑тесты)
- [x] `poetry install` и `poetry run maturin build --release` в `rusty_neat_py` — сборка wheel проходит

### 2. Порядок проверки модулей (Rust)

1) `genes`
   - [x] структуры (проверить поля)
     - [x] `NeuronGene`
     - [x] `LinkGene`
     - [x] `Gene`
     - [x] `traits` map
       - [x] NOTE: реализован упрощенный трейт в виде маппинга int, float, str, bool
   - [x] конструкторы/методы
     - [x] `new`
     - [x] `randomize_traits_map`
     - [x] `set_weight` и т.п.
   - [x] тесты (см. tests)
     - [x] `mutate_traits`
     - [x] `traits_behavior`

2) `parameters`
   - [x] структура `Parameters` — проверить присутствие критичных полей (PopulationSize, min/max weights, HyperNEAT/ES поля: `DivisionThreshold`, `MaxDepth`, `IterationLevel`, `CPPN_Bias`, и т.д.)
   - [x] `Default` значения соответствуют ожиданиям
   - [x] чтение/запись параметров (если реализовано)

3) `genome`
   - [x] структура `Genome` — ключевые поля (`neuron_genes`, `link_genes`, `genome_gene`, `num_inputs`, `num_outputs`)
   - [x] методы: NOTE: не проверял реализацию, просто проверил что методы есть и делают туже работу
     - [x] `build_phenotype`
     - [x] `build_hyperneat_phenotype`
     - [x] `derive_phenotypic_changes`
     - [x] `mutate_*`
     - [x] `randomize_link_weights`
     - [x] `randomize_traits`
   - [x] проверка `Default`/конструкторов и корректного поведения при пустых данных

4) `network` (Phenotype)
   - [x] `Connection`
   - [ ] `NeuralNetwork.activate()`
   - [x] `activate_fast()`
   - [x] `Neuron`
   - [x] функции активации
   - [x] `NeuralNetwork::new`, `add_neuron`, `add_connection`, `input`, `output`, `flush`, `set_input_output_dimensions`
   - [ ] консистентность `num_inputs/num_outputs` и позиционирования output нейронов

5) `hyperneat` / `substrate`
   - [ ] `Substrate::new`, `with_coords`, `get_min_cppn_inputs`, `get_min_cppn_outputs`, `get_max_dims`, `set_neurons`, `print_info`
   - [ ] флаги: `allow_*`, `with_distance`, `leaky`, `query_weights_only`, `custom_connectivity`
   - [ ] интеграция: `Genome::build_hyperneat_phenotype(net, subst)` — проверить на простых CPPN (см. тест `es_hyperneat_smoke.rs`)

6) `population`, `species`, `innovation` (если используются)
   - [ ] базовые методы: создание популяции, `epoch`/`tick`/`get_best_genome`, распределение в species
   - [ ] интеграционные smoke-прогоны небольших population примеров

7) `random` и утилиты
   - [ ] RNG: `rand_float`, `rand_int`, (опционально `seed`, `roulette`) — проверить детерминированность при семени
   - [ ] вспомогательные `utils::clamp`, `scale` и т.п.

8) `serialization` / pickling
   - [ ] функции `to_bincode/from_bincode`, `to_json/from_json` (если реализованы)
   - [ ] `__getstate__/__setstate__` реализация в биндингах — проверить round‑trip
   - [ ] убедиться, что в payload присутствуют `format` и `schema_version`

### 3. Rust‑тесты (ручная проверка)

- [x] Запустить все тесты: `cargo test` — убедиться, что проходят unit‑тесты
- [ ] Запустить специфичные smoke‑тесты: `cargo test --test es_hyperneat_smoke` (тесты, помеченные `#[ignore]` должны запускаться отдельно только при необходимости)
- [ ] Проверить тесты из `tests/` папки: `parameters_traits_integration`, `traits_behavior`, `phenotype_behavior`, `mutate_*` — убедиться, что логика совпадает с ожиданиями

### 4. Python‑биндинги (`rusty_neat_py`) — ручная проверка

1) Сборка и установка
   - [ ] `poetry install` и `poetry run maturin develop` — установка в dev‑окружение
   - [ ] проверить `import rusty_neat_py` и базовый `help()`

2) Экспортированные классы / API
   - [ ] `PyGenome` / `PyNeuralNetwork` / `PyParameters` / `PySubstrate` / `PyPopulation` / `PyRNG` присутствуют
   - [ ] конструкторы и базовые методы работают (создание, `BuildPhenotype`, `Input/Activate/Output`)
   - [ ] методы, принимающие NumPy/`PyArray` и list, работают корректно

3) Pickle / сериализация
   - [ ] `__getstate__` возвращает dict с `format` и `schema_version` и `payload`
   - [ ] `pickle.dumps` / `pickle.loads` round‑trip для `Parameters`/`Substrate` и простых `Genome` (smoke)
   - [ ] при попытке загрузить C++‑pickle показывается дружелюбная ошибка и ссылка на `research/pickle.md`

4) Поведение при GIL/параллелизме
   - [ ] тяжёлые операции (например, `Population::epoch`) освобождают GIL при необходимости (`py.allow_threads`)

### 5. Python‑тесты (ручная проверка)

- [x] В `rusty_neat_py/tests` запустить `pytest` (в virtualenv/poetry env)
- [x] smoke‑tests: `test_neat_xor_smoke.py`, `test_hyperneat_xor_smoke.py` — убедиться, что проходят
- [x] тесты pickling (если есть) — `test_parameters_pickling.py`, `test_substrate_pickling.py`

### 6. Релизные шаги проверки

- [ ] Обновить `CHANGELOG.md` с чётким указанием breaking changes
- [ ] Убедиться, что `pyproject.toml` / `Cargo.toml` имеют согласованную версию и метаданные
- [ ] Собрать wheels через `poetry run maturin build --release` и локально установить для smoke‑проверки

### 7. Примечания и советы при ручной проверке

- При проблемах с десериализацией добавляйте вывод `schema_version` и первые несколько байт `payload` для диагностики.
- Для ES‑HyperNEAT используйте небольшой детерминированный CPPN (см. `tests/common.rs::make_fixed_cppn`) чтобы быстро проверять поведение `build_hyperneat_phenotype`.
- Тяжёлые эволюционные тесты помечайте в CI как `#[ignore]` или `@pytest.mark.slow` и запускайте только в nightly/workflow с большим бюджетом времени.
