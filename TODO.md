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

- [ ] Постепенное переписывание модулей
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

- [ ] Покрытие тестами
  - [x] Для каждого переписанного модуля пишите модульные тесты на Rust, используя примеры из `cneat/MultiNEAT/examples`.
  - [x] Сравнивайте результаты работы Rust-версии с оригинальной C++.
  - [ ] Рекомендуемые подпункты по приоритету:
    - [x] Substrate / HyperNEAT helpers — unit-тесты для `get_max_dims`, `get_min_cppn_inputs`, `get_min_cppn_outputs`, `with_coords`, `set_neurons`, `print_info`.
    - [x] BuildHyperNEATPhenotype — быстрый smoke-test: детерминированный CPPN/phenotype, проверка количества связей, фильтрации по флагам и масштабирования весов (`max_weight_and_bias`).
    - [x] Traits behaviour — unit-тесты для `init_trait_map` и `mutate_trait_map`: replace vs perturb, clamping, dep_key/dep_values, roulette для set/string.
    - [x] PhenotypeBehavior — тесты контрактов: `acquire()` (false по-умолчанию), `distance_to()` (0 для идентичных `m_data`), `successful()` (true по-умолчанию), сравнение `m_data`.
    - [x] Parameters ↔ Traits integration — небольшие интеграционные тесты: `SetNeuronTraitParameters`/`SetLinkTraitParameters` + `Genome::randomize_traits`.
    - [ ] ES-HyperNEAT (опционально) — smoke-тесты для ES-параметров (DivisionThreshold, IterationLevel, MaxDepth); эти тесты можно помечать `#[ignore]` или запускать отдельно.
    - [ ] Serialization / pickling (опционально) — round-trip тесты `Substrate` и простых `Genome`; pickling-совместимость с Python — отдельная интеграция.
    - [ ] CI policy — определить бюджет времени и пометить тяжёлые эволюционные примеры как `#[ignore]` или вынести в nightly/integration workflow.

- [ ] Интеграция и сборка
  - [x] Используйте `Cargo.toml` для управления зависимостями и сборкой.
  - [x] Для поддержки Python добавьте PyO3 и настройте сборку через `maturin` или `wasm-pack`.

    Этапы реализации биндингов на Python (подпункты)

    - Фаза 0 — Инвентаризация и согласование API

      - [x] Собрать полный список экспортируемых типов и методов (см. `research/python_integration.md`).
      - [x] Решить совместимость pickle/серриализации и ожидания NumPy I/O.
      - [x] Составить таблицу маппинга: C++ сигнатура → Rust type → PyO3 signature (включая типы NumPy).

    - Фаза 1 — Дизайн API

      - [x] Решить контракты владения/потоков: `Arc<Mutex<T>>` vs `PyRef` для каждого экспортируемого класса.
      - [x] Спроектировать Python‑уровень: какие поля делать свойствами, какие методы — напрямую, какие пакеты — упаковывать в маленькие Py‑классы (LinkGene/NeuronGene).
      - [x] Документировать соглашения по Pickle/Сериализации (совместимость C++ → Rust — опционально).

    - Фаза 2 — Создание skeleton crate `rusty_neat_py`

      - [x] Создать `rusty_neat_py` с `Cargo.toml` и `src/lib.rs` (#[pymodule]).
      - [x] Реализовать минимальные `PyGenome` и `PyNeuralNetwork` как скелетоны для контрактов и smoke-tests.
      - [x] Создать crate в workspace с `crate-type = ["cdylib"]`, добавить зависимости `pyo3`, `pyo3-ndarray`, `ndarray`, и локальную зависимость `rusty_neat`.
      - [x] Добавить `#[pymodule]` и регистрации базовых классов (`PyGenome`, `PyNeuralNetwork`, `PyParameters`, `PySubstrate`, `PyRNG`) с минимальными конструкторами/методами.
      - [x] Добавить пример `maturin develop` запуск в README и basic smoke example на Python.

    - Фаза 3 — Ядро биндингов

      - [x] Реализовать `PyNeuralNetwork` с поддержкой `Input` из Python list и `PyArray` (через `pyo3-ndarray`), `Activate`, `Output` и сохранение/загрузку.
      - [ ] Реализовать `PyGenome` с `BuildPhenotype` и `BuildHyperNEATPhenotype` (принятие `PySubstrate`).
      - [ ] Экспортировать ключевые структуры: `LinkGene`, `NeuronGene`, `GenomeInitStruct` (как простые `pyclass`/namedtuple).
      - [ ] Реализовать минимальные маппинги контейнеров (vec ↔ list) и provide light iterators for large lists.

    - Фаза 4 — Traits / Parameters / Substrate

      - [ ] Экспортировать `Parameters` с API trait management: `ListNeuronTraitParameters`, `SetNeuronTraitParameters`, `GetNeuronTraitParameters` и т.д.
      - [ ] Реализовать `PySubstrate` с флагами (allow_input_output_links ...) и методами `GetMinCPPNInputs/GetMinCPPNOutputs`.
      - [ ] Pickling для `Parameters`/`Substrate` (implement `__getstate__/__setstate__`).

    - Фаза 5 — Population / Species / PhenotypeBehavior

      - [ ] Реализовать `PyPopulation` с методами `Epoch`, `Tick`, `GetBestGenome` и доступом к `Species`/`Genome` через ссылочные обёртки.
      - [ ] Экспорт `PhenotypeBehavior` и его контрактных методов (Acquire, Distance_To, Successful).
      - [ ] Подумать про освобождение GIL в тяжёлых операциях (use `py.allow_threads`).

    - Фаза 6 — Тесты и CI
      - [ ] Добавить pytest smoke tests, взяв примеры из `cneat/MultiNEAT/examples` (TestNEAT_xor, TestHyperNEAT_xor и др.) и адаптировать их для Rust биндингов.
      - [ ] Добавить GitHub Actions job: `maturin build` → `pip install target/wheels/*.whl` → `pytest`.
      - [ ] Маркировать тяжёлые/долгие тесты как `@pytest.mark.slow` или запускать в nightly workflow.

    - Фаза 7 — Release и совместимость
      - [ ] Выстроить политику по версии API и декларацию несовместимости pickles с C++ при наличии.
      - [ ] Подготовить `maturin` сборки для целевых платформ и инструкцию публикации на PyPI.

    - Риски и замечания
      - [ ] Copy overhead: для больших векторов отдавать предпочтение `PyArray`/views.
      - [ ] Pickle‑совместимость с C++ редко достижима «бесплатно» — документировать и предоставить миграционные утилиты при необходимости.

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

      6) Дополнительно (низкий приоритет для 1.0, но полезно скоро):
          - Population: `Epoch`, `Tick`, `GetBestGenome`, `AccessGenomeByIndex` (можно отложить в 1.1 при необходимости).
          - Species: минимальный доступ к `Individuals` и `GetLeader`.

- [ ] Документация и примеры
  - [ ] Перенесите и адаптируйте документацию из оригинального `README.md` и примеры.
  - [ ] Обновите `README.md` и `README_WEB.md` для Rust-версии.

- [ ] Автоматизация и CI
  - [x] Настройте тестирование и сборку через GitHub Actions (см. примеры в `.github/workflows`).
  - [ ] Добавьте сборку для разных платформ и, при необходимости, публикацию бинарников/библиотек.

**Рекомендуемый порядок переписывания:**

1. Базовые структуры данных (геномы, нейроны, связи)
2. Алгоритмы эволюции (мутации, кроссовер, селекция)
3. Симуляция сети (активация, расчёт выхода)
4. Вспомогательные утилиты (рандомизация, сериализация)
5. Публичный API и интеграция с внешними языками (Python, WASM)
6. Документация и примеры
