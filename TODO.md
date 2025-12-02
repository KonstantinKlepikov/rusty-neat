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
  - [ ] Для каждого переписанного модуля пишите модульные тесты на Rust, используя примеры из `cneat/MultiNEAT/examples`.
  - [ ] Сравнивайте результаты работы Rust-версии с оригинальной C++.
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
  - [ ] Используйте `Cargo.toml` для управления зависимостями и сборкой.
  - [ ] Для поддержки Python добавьте PyO3 и настройте сборку через `maturin` или `wasm-pack`.

- [ ] Документация и примеры
  - [ ] Перенесите и адаптируйте документацию из оригинального `README.md` и примеры.
  - [ ] Обновите `README.md` и `README_WEB.md` для Rust-версии.

- [ ] Автоматизация и CI
  - [x] Настройте тестирование и сборку через GitHub Actions (см. примеры в `.github/workflows`).
  - [ ] Добавьте сборку для разных платформ и, при необходимости, публикацию бинарников/библиотек.

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
