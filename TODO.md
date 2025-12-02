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
    - [x] `Mutate_NeuronTimeConstants`, `Mutate_NeuronBiases`
    - [x] `Mutate_NeuronTraits`, `Mutate_LinkTraits`, `Mutate_GenomeTraits`
    - [x] `Randomize_Traits`
  - [x] Реализуйте вспомогательные функции (генераторы случайных чисел, сериализацию).

- [ ] Покрытие тестами
  - [x] Для каждого переписанного модуля пишите модульные тесты на Rust, используя примеры из `cneat/MultiNEAT/examples`.
  - [ ] Сравнивайте результаты работы Rust-версии с оригинальной C++.
  - [ ] Рекомендуемые подпункты по приоритету:
    - [x] Substrate / HyperNEAT helpers — unit-тесты для `get_max_dims`, `get_min_cppn_inputs`, `get_min_cppn_outputs`, `with_coords`, `set_neurons`, `print_info`.
    - [x] BuildHyperNEATPhenotype — быстрый smoke-test: детерминированный CPPN/phenotype, проверка количества связей, фильтрации по флагам и масштабирования весов (`max_weight_and_bias`).
    - [x] Traits behaviour — unit-тесты для `init_trait_map` и `mutate_trait_map`: replace vs perturb, clamping, dep_key/dep_values, roulette для set/string.
    - [x] PhenotypeBehavior — тесты контрактов: `acquire()` (false по-умолчанию), `distance_to()` (0 для идентичных `m_data`), `successful()` (true по-умолчанию), сравнение `m_data`.
    - [x] Parameters ↔ Traits integration — небольшие интеграционные тесты: `SetNeuronTraitParameters`/`SetLinkTraitParameters` + `Genome::randomize_traits`.
    - [ ] ES-HyperNEAT (опционально) — smoke-тесты для ES-параметров (DivisionThreshold, IterationLevel, MaxDepth); эти тесты можно помечать `#[ignore]` или запускать отдельно. NOTE: в настоящий момент не реализована логика самого ES-HyperNEAT
    - [ ] Serialization / pickling (опционально) — round-trip тесты `Substrate` и простых `Genome`; pickling-совместимость с Python — отдельная интеграция.
    - [ ] CI policy — определить бюджет времени и пометить тяжёлые эволюционные примеры как `#[ignore]` или вынести в nightly/integration workflow.

- [ ] Интеграция и сборка
  - [ ] Используйте `Cargo.toml` для управления зависимостями и сборкой.
  - [ ] Для поддержки Python добавьте PyO3 и настройте сборку через `maturin` или `wasm-pack`.
    - [x] Инвентаризация публичного API C++: просмотреть `PythonBindings.cpp`, `_MultiNEAT.pyx`, `cMultiNeat.pxd` и зафиксировать критичные классы/методы для экспорта.
    - [ ] Спроектировать Python‑уровень API: решить, какие интерфейсы экспортировать 1:1, а какие представить в более идиоматичном Python‑виде.
    - [ ] Создать биндинговый crate `rusty_neat_py` в workspace (crate‑type = `cdylib`) в котором будут `#[pyclass]` обёртки (`PyGenome`, `PyNeuralNetwork`, `PyParameters`, `PySubstrate`).
    - [ ] Реализовать конвертацию NumPy ↔ `ndarray` через `pyo3-ndarray` для эффективного обмена входами/выходами нейросетей.
    - [ ] Добавить сериализацию/pickling: реализовать `__getstate__/__setstate__` (serde + bincode или внутренний `to_bytes`/`from_bytes`) и удобный `__repr__`.
    - [ ] Покрыть smoke‑tests и примеры: адаптировать примеры из `cneat/MultiNEAT/examples` как pytest‑smoke, добавить job в CI для сборки wheel и запуска тестов.
    - [ ] Совместимость и развёртывание: собирать и публиковать wheels через `maturin` для целевых платформ; при необходимости добавить `wasm-pack` workflow для WASM-артефактов.

- [ ] Документация и примеры
  - [ ] Перенесите и адаптируйте документацию из оригинального `README.md` и примеры.
  - [ ] Обновите `README.md` и `README_WEB.md` для Rust-версии.

- [ ] Автоматизация и CI
  - [x] Настройте тестирование и сборку через GitHub Actions (см. примеры в `.github/workflows`).
  - [ ] Добавьте сборку для разных платформ и, при необходимости, публикацию бинарников/библиотек.

- [ ] Миграционный план — Python биндинги (детализировано)
  - Цель: поэтапно реализовать удобные и производительные Python биндинги для `rusty_neat`, совместимые с NumPy и удобные для пользователей MultiNEAT.
  - Фаза 0 — Подготовка и инвентаризация (готово / reference)
    - [x] Использовать `research/python_integration.md` — раздел "Инвентаризация публичного API (C++)" как исходную карту экспорта.
    - [x] Уточнить список приоритетных API (минимальный поверхностный набор для 1.0) — детализация ниже.

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

   Причины выбора: этот набор позволяет сконструировать/сгенерировать Genome, собрать фенотип (включая HyperNEAT с Substrate), выполнить прогон через NeuralNetwork (I/O + Activate), управлять конфигурацией через Parameters, и обеспечить детерминизм/рандомность через RNG. Population/Species можно добавить сразу после базовой рабочей цепочки.

  - Фаза 1 — Дизайн API
    - [ ] Решить контракты владения/потоков: `Arc<Mutex<T>>` vs `PyRef` для каждого экспортируемого класса.
    - [ ] Спроектировать Python‑уровень: какие поля делать свойствами, какие методы — напрямую, какие пакеты — упаковывать в маленькие Py‑классы (LinkGene/NeuronGene).
    - [ ] Документировать соглашения поPickle/Сериализации (совместимость C++ → Rust — опционально).

  - Фаза 2 — Создание skeleton crate `rusty_neat_py`
    - [ ] Создать crate в workspace с `crate-type = ["cdylib"]`, добавить зависимости `pyo3`, `pyo3-ndarray`, `ndarray`, и локальную зависимость `rusty_neat`.
    - [ ] Добавить `#[pymodule]` и регистрации базовых классов (`PyGenome`, `PyNeuralNetwork`, `PyParameters`, `PySubstrate`, `PyRNG`) с минимальными конструкторами/методами.
    - [ ] Добавить пример `maturin develop` запуск в README и basic smoke example на Python.

  - Фаза 3 — Ядро биндингов
    - [ ] Реализовать `PyNeuralNetwork` с поддержкой `Input` из Python list и `PyArray` (через `pyo3-ndarray`), `Activate`, `Output` и сохранение/загрузку.
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

---

**Рекомендуемый порядок переписывания:**

1. Базовые структуры данных (геномы, нейроны, связи)
2. Алгоритмы эволюции (мутации, кроссовер, селекция)
3. Симуляция сети (активация, расчёт выхода)
4. Вспомогательные утилиты (рандомизация, сериализация)
5. Публичный API и интеграция с внешними языками (Python, WASM)
6. Документация и примеры

**Советы:**

- Используйте идиоматичный Rust: владение памятью, `Option`/`Result`, обработка ошибок.
- Для сложных алгоритмов пишите интеграционные тесты, сверяясь с результатами C++.
- Не пытайтесь сразу переписать всё — двигайтесь поэтапно, начиная с самого простого.
