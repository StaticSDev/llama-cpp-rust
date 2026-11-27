# llama.cpp → Rust

Русский · [English](README.md)

Рабочий порт llama.cpp на Rust: CPU-режим, опциональный CUDA-бэкенд для видеокарт
Turing и новее, а также локальный чат в пиксельном стиле.

Исходник-ориентир зафиксирован на коммите `15fcfc7357f8de73671d39e7d8c594e1beb3052d`.
Дерево лежит в `reference/llama.cpp` и не попадает в репозиторий.

---

## Что работает

- Плотные и гибридные слои Qwen3.5: gated delta net, полное внимание, RoPE, FFN.
- Загрузка GGUF с типами Q4_K, Q5_K, Q6_K, Q8_0, F32, F16 и BF16.
- Жадная генерация и сэмплинг temperature + top-p.
- Matvec на 12 CPU-потоках; CUDA-matvec с весами в VRAM.
- HTTP-сервер без внешних зависимостей, потоковая передача NDJSON и живая телеметрия.

## Быстрый старт

Сборка по умолчанию, только CPU:

```sh
cargo build --release --bin llama-rs
target/release/llama-rs \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "Hi, who are you?" -n 8 --temp 0
```

Сборка с CUDA, Turing и новее (`sm_75`):

```sh
cargo build --release --features cuda --bin llama-rs
target/release/llama-rs \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "Hi, who are you?" -n 8 --temp 0 --gpu
```

`build.rs` компилирует `cuda/lcr_kernels.cu` через `nvcc -arch=sm_75`.
Без `--features cuda` флаг `--gpu` отдаёт понятную ошибку.

## Веб-чат

```sh
cargo build --release --features cuda --bin llama-web
target/release/llama-web \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8080 --gpu
```

Открой <http://127.0.0.1:8080/>. Интерфейс хранит чаты в `localStorage`,
показывает прогресс префилла десятью пиксельными сегментами и выводит скорость
в токенах в секунду. За один раз обрабатывается один запрос.

API:

| Маршрут | Назначение |
| --- | --- |
| `GET /api/status` | Путь модели, флаг готовности, имя GPU |
| `POST /api/chat/stream` | NDJSON-события: `prompt`, `token`, `done`, `error` |

---

## Замер скорости

Замер через `bench/speed_bench.py` на одном и том же промпте. Движки запускаются
по очереди, чтобы в VRAM одновременно была только одна копия модели.

**Железо:** GTX 1660 SUPER (TU116, 6 ГБ), CUDA 13.4, драйвер 615.78, 12 потоков CPU,
15 ГБ ОЗУ. **Модель:** Qwen3.5-4B-Q4_K_M.gguf.

| Метрика | Оригинальный llama.cpp | Наш порт | Порт / оригинал |
| --- | ---: | ---: | ---: |
| Холодный префилл, 203 / 202 токена | 154.98 tok/s | 6.51 tok/s | 0.04× |
| Генерация, 6 токенов | 46.14 tok/s | 6.53 tok/s | 0.14× |
| Генерация, 200 токенов, 3 прогона | 54.57 tok/s | 6.34 tok/s | 0.12× |

Флаги оригинала: `-ngl 99 -c 32768 -t 6 -tb 12 -b 2048 -ub 512 -np 1 -fa on
-ctk q4_0 -ctv q4_0 --context-shift --jinja`. Для холодного префилла кэш промпта
был отключён.

Прогон на шести токенах шумный; стабильная цифра — прогон на 200 токенах.
Наш порт примерно в 8 раз медленнее оригинала. Ядра CUDA деквантуют скалярно
по элементам, запускаются по разу на каждую матрицу и синхронизируются после
каждого запуска; внимание и рекуррентное состояние считаются на CPU. Оригинал
упирается в пропускную способность памяти этой карты, наш порт — нет.

Воспроизвести:

```sh
python3 bench/speed_bench.py --runs 1 --max-tokens 6 --out bench/results.json
```

Сырые данные: [bench/results.json](bench/results.json).

---

## Структура

```text
src/
  runtime/     загрузка, деквант, forward Qwen3.5, сэмплинг, CUDA FFI
  gguf/        контейнер GGUF и таблицы типов
  ggml/        блоки квантов, квантайзеры, каркас бэкендов
  vocab/       токенизатор и метаданные чата
  web/         HTTP-сервер и статический чат
cuda/          ядра sm_75 для декванта и matvec, собираются build.rs
bench/         сравнение скорости с оригинальным llama.cpp
reference/     зафиксированный исходник llama.cpp, не в git
models/        локальные GGUF, не в git
```

## Тесты

```sh
cargo test                 # 250 прошли
cargo test --features cuda # 250 прошли
```

Ноль предупреждений в обеих конфигурациях.

## Лицензия

Части кода, заимствованные из llama.cpp, остаются под оригинальной лицензией MIT,
Copyright (c) 2023–2026 ggml authors. См. `LICENSE`.
