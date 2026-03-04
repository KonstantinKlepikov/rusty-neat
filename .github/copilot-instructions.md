# Инструкции для Copilot / AI-агентов

Краткие и практические рекомендации для работы с этим репозиторием.

## Project Overview

- Основная логика реализована на Rust в корне репозитория (`Cargo.toml`) и в `src/`.
- Python-интеграция и биндинги находятся в `rusty_neat_py/` (см. `rusty_neat_py/src/lib.rs`).
- Старые/внешние реализации и вспомогательные скрипты расположены в `cneat/MultiNEAT/`.

## Code Style

- Rust: используйте `rustfmt` и `clippy` (встроенные правила). Примеры стиля см. в `src/` (например `src/genome.rs`).
- Python: соблюдайте настройки в `pyproject.toml` / `poetry.toml` (используйте `black`/`isort` по конфигурации проекта).

## Architecture

- Ядро — библиотека на Rust (нейроэволюция, геномы, популяции). Публичный API экспонируется в `src/lib.rs` и модулях под `src/`.
- Python-слой (`rusty_neat_py/`) обеспечивает биндинги и пример использования из Python-тестов и `examples/`.
- `cneat/MultiNEAT` содержит оригинальную реализацию (C/C++/Python) — используйте их как справочные примеры для портирования в rust.
- `cneat/r1` всегда игнорировать.
- `rusty-neat/research` содерижит справочные материалы о портирвоании из c++ в rust
- портируемый код на c++ находится в папке `/home/kklepikov/_code_/rusty-neat/cneat/MultiNEAT/src`
- код портируется на rust в папку `/home/kklepikov/_code_/rusty-neat/src`
- тесты портированного кода находятся в папке `/home/kklepikov/_code_/rusty-neat/tests`.
- биндинги на python портированного кода находятся в папке `/home/kklepikov/_code_/rusty-neat/rusty_neat_py/src`
- тесты биндингов на python находятся в папке `/home/kklepikov/_code_/rusty-neat/rusty_neat_py/tests`

## Build and Test

- Rust (корень):

  - `cargo build --workspace`
  - `cargo test --workspace`
  - `cargo fmt --all` и `cargo clippy --all-targets -- -D warnings`

- Python bindings (в `rusty_neat_py/`):

  - Сборка: `cd rusty_neat_py && poetry run maturin develop --release`
  - Тесты: `cd rusty_neat_py && poetry run pytest -q tests/`

- Если изменение затрагивает FFI/биндинги, прогоняйте интеграционные тесты для Python и Rust.

## Project Conventions

- Все функции и структуры должны быть снабжены подробными комментариями. Включая и тесты.
- Комментарии выполняйте только на английском
- Тесты добавляйте только в папку `tests`.
- Не добавлять тесты в файлы в папке `src`
- Малые изменения API: пишите backward-compatible патчи и добавляйте тесты в `tests/`.
- Для новых экспортив/биндингов добавляйте примеры в `rusty_neat_py/examples/` и соответствующие тесты.

## Integration Points

- Python ↔ Rust: `rusty_neat_py/` (maturin/poetry/Cargo). Проверяйте, что сборка биндингов работает на CI перед изменением API.
- C++ legacy: `cneat/MultiNEAT/` содержит `setup.py` и CMake-конфиг — полезно для сравнения алгоритмов и тестов.

## Security & Sensitive Data

- Репозиторий не должен содержать секреты, ключи или локальные конфигурации. Игнорируйте и не коммитьте файлы окружения.

## When You Edit This File

- Если вы меняете организацию репозитория (перенос/удаление модулей), обновите этот файл и добавьте короткое описание причин в PR.
