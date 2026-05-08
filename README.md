# medical-chat-bot-

Medical Chat Bot — это RAG-приложение для ответов на вопросы по документации медицинской информационной системы (МИС). Пользователь задает вопрос в веб-интерфейсе, приложение ищет релевантные фрагменты документации, передает их в LLM и возвращает ответ со ссылками на источники.

Проект состоит из трех основных частей:

- `rag_app/` — FastAPI-приложение, веб-интерфейс, RAG-поиск, история диалогов и API.
- `pipeline/` — пайплайн подготовки чанков из исходных текстов, изображений и глоссария.
- Docker Compose-стек — приложение, Qdrant для векторного индекса, PostgreSQL для истории и отдельная задача оценки метрик.

## Что умеет приложение

- Отвечает на вопросы по русскоязычной документации МИС.
- Использует гибридный поиск: dense embeddings через Qdrant и sparse BM25 по тексту.
- Переписывает исходный вопрос в несколько поисковых формулировок через LLM, если включен `ENABLE_QUERY_REWRITE`.
- Объединяет результаты dense/sparse поиска через Reciprocal Rank Fusion.
- Может выполнять LLM rerank найденных кандидатов, если включен `ENABLE_LLM_RERANK`.
- Формирует ответ только по найденному контексту и добавляет ссылки на источники.
- Показывает найденные источники в интерфейсе: файл, `chunk_id`, фрагмент текста и скор.
- Хранит историю диалогов в PostgreSQL.
- Позволяет переиндексировать чанки из интерфейса или через API.
- Считает метрики качества поиска на размеченном наборе вопросов.

## Как устроен RAG

При старте приложение читает файл чанков `pipeline/output/chunks.jsonl`. Каждая строка JSONL превращается в `ChunkDocument` с текстом чанка, контекстом, заголовками, исходным файлом и идентификатором.

Дальше создаются две поисковые структуры:

- dense-индекс в Qdrant: текст чанков кодируется моделью `sentence-transformers`, по умолчанию `jinaai/jina-embeddings-v3`;
- sparse-индекс BM25: строится в памяти по тем же чанкам.

Для пользовательского вопроса выполняется такой сценарий:

1. Если включен query rewrite, LLM генерирует дополнительные поисковые запросы с синонимами, терминами интерфейса и аббревиатурами.
2. Для каждого запроса приложение делает dense-поиск в Qdrant и sparse-поиск по BM25.
3. Результаты объединяются RRF-скорингом. Исходный вопрос получает больший вес, переписанные запросы — чуть меньший.
4. Из объединенного списка берется `candidate_limit` кандидатов.
5. Если включен LLM rerank, LLM сортирует кандидатов по полезности для ответа.
6. В финальный контекст ответа попадает до `answer_context_limit` лучших фрагментов.
7. LLM пишет ответ на языке пользователя, используя только переданный контекст.
8. Ответ, источники, модель и latency сохраняются в PostgreSQL.

## Технологии

- Python 3.11
- FastAPI и Uvicorn
- Qdrant
- PostgreSQL 16
- `sentence-transformers`
- `rank-bm25`
- OpenAI-compatible Chat Completions API, например LM Studio
- Веб-интерфейс на статических HTML/CSS/JS

## Структура проекта

```text
.
├── rag_app/
│   ├── main.py              # FastAPI, API endpoints, жизненный цикл приложения
│   ├── retrieval.py         # RAGService: hybrid retrieval, RRF, rerank, answer
│   ├── embeddings.py        # загрузка и вызов sentence-transformers
│   ├── qdrant_store.py      # коллекция Qdrant, upsert, dense search
│   ├── bm25.py              # sparse BM25 индекс
│   ├── llm.py               # OpenAI-compatible LLM клиент
│   ├── history.py           # PostgreSQL-схема и история диалогов
│   ├── documents.py         # чтение JSONL-чанков
│   ├── metrics.py           # оценка поиска (профили dense/sparse/hybrid/pipeline/full)
│   └── static/              # веб-интерфейс
├── pipeline/
│   ├── cli.py               # CLI пайплайна подготовки данных
│   ├── sources/             # источники данных для сборки чанков
│   ├── output/              # артефакты пайплайна и chunks.jsonl
│   ├── questions.csv        # вопросы для метрик
│   ├── texts.csv            # страницы/источники для метрик
│   └── README.md            # подробная документация пайплайна
├── scripts/
│   ├── docker-entrypoint.sh
│   └── preload_hf_model.py
├── docker-compose.yml
├── docker-compose.gpu.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Быстрый запуск через Docker Compose

### 1. Подготовьте `.env`

Скопируйте пример настроек:

```bash
cp .env.example .env
```

По умолчанию приложение ожидает OpenAI-compatible API на хосте:

```env
LLM_BASE_URL=http://host.docker.internal:1488/v1
LLM_MODEL=lmstudio-community/gemma-4-e4b-it
LLM_API_KEY=lm-studio
```

Такая конфигурация подходит для LM Studio, запущенной на машине разработчика. В LM Studio нужно включить локальный сервер, совместимый с OpenAI API, и загрузить модель, указанную в `LLM_MODEL`.

### 2. Запустите стек

```bash
docker compose up --build app -d
```

Будут подняты:

- `app` — FastAPI-приложение на порту `8000`;
- `qdrant` — Qdrant на портах `6333` и `6334`;
- `postgres` — PostgreSQL на порту `5432`.

При первом запуске приложение:

1. скачает модель эмбеддингов в Docker volume `hf_cache`;
2. прочитает `pipeline/output/chunks.jsonl`;
3. создаст или проверит коллекцию Qdrant `medical_chunks`;
4. при `AUTO_INDEX=true` построит dense-индекс, если он еще не готов;
5. создаст таблицы истории в PostgreSQL.

После старта откройте:

```text
http://localhost:8000
```

Проверить состояние API можно так:

```bash
curl http://localhost:8000/api/health
```

## Запуск с GPU

Для NVIDIA GPU нужен установленный NVIDIA Container Toolkit. Запуск:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build app
```

Файл `docker-compose.gpu.yml` включает:

- сборку с CUDA-версией PyTorch;
- `gpus: all`;
- `EMBEDDING_DEVICE=cuda`.

## Локальный запуск без контейнера приложения

Этот режим удобен для разработки кода. Qdrant и PostgreSQL все равно проще поднять через Docker:

```bash
docker compose up qdrant postgres
```

Затем установите зависимости в Python 3.11 окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

Создайте `.env` из `.env.example` и для локального запуска используйте адреса без Docker-сетей:

```env
QDRANT_URL=http://localhost:6333
POSTGRES_DSN=postgresql://rag:rag@localhost:5432/rag
LLM_BASE_URL=http://127.0.0.1:1488/v1
```

Запуск приложения:

```bash
uvicorn rag_app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Основные переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|------------|------------------------|------------|
| `CHUNKS_PATH` | `pipeline/output/chunks.jsonl` | Файл с финальными чанками для RAG. |
| `AUTO_INDEX` | `true` | Индексировать чанки при старте, если коллекция Qdrant не готова. |
| `FORCE_REINDEX_ON_START` | `false` | Принудительно пересоздавать индекс при каждом старте. |
| `QDRANT_URL` | `http://localhost:6333` локально, `http://qdrant:6333` в Docker | Адрес Qdrant. |
| `QDRANT_API_KEY` | пусто | API key для Qdrant, если используется защищенный инстанс. |
| `QDRANT_COLLECTION` | `medical_chunks` | Имя коллекции с dense-векторами. |
| `POSTGRES_DSN` | `postgresql://rag:rag@localhost:5432/rag` | DSN PostgreSQL для истории диалогов. |
| `POSTGRES_PASSWORD` | `rag` | Пароль PostgreSQL в Docker Compose. |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v3` | Модель эмбеддингов. |
| `EMBEDDING_DEVICE` | авто | Устройство для эмбеддингов: `cpu`, `cuda`, `mps` или пусто. |
| `EMBEDDING_BATCH_SIZE` | `16` | Размер батча при кодировании текстов. |
| `EMBEDDING_QUERY_TASK` | `retrieval.query` | Task для query embedding у Jina v3. |
| `EMBEDDING_PASSAGE_TASK` | `retrieval.passage` | Task для passage embedding у Jina v3. |
| `LLM_BASE_URL` | `http://127.0.0.1:1488/v1` | OpenAI-compatible endpoint. |
| `LLM_MODEL` | `lmstudio-community/gemma-4-e4b-it` | Модель для rewrite, rerank и ответа. |
| `LLM_API_KEY` | `lm-studio` | API key для LLM endpoint. |
| `LLM_TEMPERATURE` | `0.1` | Температура финального ответа. |
| `DENSE_LIMIT` | `24` | Сколько dense-кандидатов брать из Qdrant. |
| `SPARSE_LIMIT` | `24` | Сколько sparse-кандидатов брать из BM25. |
| `CANDIDATE_LIMIT` | `24` | Размер общего пула после fusion. |
| `ANSWER_CONTEXT_LIMIT` | `6` | Сколько лучших фрагментов передавать в ответ. |
| `RRF_K` | `60` | Константа Reciprocal Rank Fusion. |
| `QUERY_REWRITE_COUNT` | `3` | Сколько дополнительных запросов генерировать. |
| `ENABLE_QUERY_REWRITE` | `true` | Включить LLM-переписывание запроса. |
| `ENABLE_LLM_RERANK` | `true` | Включить LLM rerank кандидатов. |

## API

### `GET /`

Возвращает веб-интерфейс чата.

### `GET /api/health`

Проверка состояния приложения.

Пример ответа:

```json
{
  "status": "ok",
  "chunks_loaded": 1234,
  "collection": "medical_chunks",
  "model": "lmstudio-community/gemma-4-e4b-it"
}
```

### `GET /api/config`

Возвращает часть активной конфигурации: LLM-модель, embedding-модель, коллекцию Qdrant, путь к чанкам и флаги RAG.

### `POST /api/index`

Запускает индексацию чанков в Qdrant.

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"force": false}'
```

Если `force=true`, коллекция может быть пересоздана.

### `POST /api/chat`

Отправляет сообщение в чат.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Как настроить роли для диспансеризации?"}'
```

Для продолжения существующего диалога передайте `session_id`:

```json
{
  "message": "А где это находится в интерфейсе?",
  "session_id": "00000000-0000-0000-0000-000000000000"
}
```

Ответ содержит:

- `answer` — текст ответа;
- `sources` — найденные источники;
- `session_id` — идентификатор диалога;
- `call_id` — идентификатор обращения;
- `model` — использованная LLM;
- `latency_ms` — время обработки.

### `GET /api/sessions`

Возвращает список последних диалогов.

### `GET /api/sessions/{session_id}/calls`

Возвращает все сообщения и ответы выбранного диалога.

### `DELETE /api/sessions/{session_id}`

Удаляет диалог и связанные с ним вызовы.

## Веб-интерфейс

Интерфейс находится в `rag_app/static/` и отдается самим FastAPI-приложением.

В нем есть:

- список диалогов;
- создание нового чата;
- удаление диалога;
- поле вопроса;
- markdown-рендеринг ответа;
- блок источников под ответом;
- системная панель с состоянием модели, коллекции и количеством чанков;
- кнопка переиндексации.

Для иконок, Markdown и KaTeX используются CDN-скрипты в `index.html`, поэтому при открытии интерфейса браузеру нужен доступ к CDN.

## Подготовка данных

Финальный RAG использует файл:

```text
pipeline/output/chunks.jsonl
```

Формат — JSON Lines, одна строка на чанк. Важные поля:

- `chunk_id` — числовой идентификатор;
- `chunk_text` — основной текст фрагмента;
- `chunk_context` — дополнительный контекст;
- `source_file` — исходный файл или страница;
- `doc_title` — название документа;
- `h1` ... `h6` — путь заголовков;
- `source` — идентификатор источника данных.

Полный пайплайн описан в `pipeline/README.md`. Коротко:

```bash
python -m pipeline --help
python -m pipeline captions
python -m pipeline build
python -m pipeline expand
python -m pipeline context
```

Объединенный запуск без LLM-контекста:

```bash
python -m pipeline run --steps captions,build,expand
```

С генерацией `chunk_context` через LLM:

```bash
python -m pipeline run --steps captions,build,expand --with-context
```

Если после подготовки данных изменился `chunks.jsonl`, нужно переиндексировать Qdrant:

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"force": true}'
```

Или нажать кнопку «Переиндексировать» в интерфейсе.

## Метрики RAG

Запуск метрик выполняется как отдельная задача Docker Compose. Он использует тот же файл чанков, модель эмбеддингов, индекс BM25 и коллекцию Qdrant, что и приложение, а затем оценивает `pipeline/questions.csv` на основе меток страниц из `pipeline/texts.csv`.

Сначала запустите основной стек, чтобы в Qdrant уже были проиндексированные векторы:

```bash
docker compose up app
```

Затем запустите оценку:

```bash
docker compose run --rm metrics
```

По умолчанию выводятся метрики профилей **dense**, **sparse**, **hybrid** и **full**:

- **dense** — только векторный поиск в Qdrant;
- **sparse** — только BM25;
- **hybrid** — RRF-слияние dense + sparse **без** query rewrite и LLM rerank (чистый retrieval, как при отключённых флагах в оценке);
- **full** — тот же `retrieve()`, что и у приложения, с флагами из окружения (`ENABLE_QUERY_REWRITE`, `ENABLE_LLM_RERANK`); при включённых флагах в `.env` в прогон попадают query rewrite и LLM rerank.

Для каждого выбранного профиля считаются общие метрики:

- `hit@k`;
- `precision@k`;
- `recall@k`;
- `nDCG@k`;
- MRR;
- MAP;
- ранг первого релевантного результата;
- latency;
- score margin;
- слабые случаи.

Также создаются файлы:

```text
pipeline/output/metrics/rag_metrics_details.csv
pipeline/output/metrics/rag_metrics_report.json
```

Чтобы **принудительно** включить query rewrite и LLM rerank на время прогона (независимо от `.env`) и сравнить с «только RRF» (`hybrid`), добавьте профиль **`pipeline`**:

```bash
docker compose run --rm metrics --profiles dense,sparse,hybrid,full,pipeline
```

Полезные параметры:

```bash
docker compose run --rm metrics --top-k 1,3,5,10,20,50
docker compose run --rm metrics --profiles hybrid --candidate-limit 50
docker compose run --rm metrics --qdrant-url http://qdrant:6333 --collection medical_chunks
```

## Типовые сценарии разработки

### Пересобрать приложение

```bash
docker compose up --build app
```

### Пересоздать индекс Qdrant

Через API:

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"force": true}'
```

Или перезапустите приложение с:

```env
FORCE_REINDEX_ON_START=true
```

### Отключить LLM rewrite и rerank

Это полезно для проверки чистого hybrid retrieval:

```env
ENABLE_QUERY_REWRITE=false
ENABLE_LLM_RERANK=false
```

### Сбросить данные контейнеров

Остановить стек с удалением volumes:

```bash
docker compose down -v
```

После этого Qdrant, PostgreSQL и Hugging Face cache будут очищены, а при следующем запуске модель эмбеддингов скачается заново.

## Возможные проблемы

### Приложение не отвечает на вопросы

Проверьте, что LLM endpoint доступен:

```bash
curl http://localhost:1488/v1/models
```

В Docker контейнере адрес хоста должен быть `host.docker.internal`, поэтому в `.env` обычно нужен:

```env
LLM_BASE_URL=http://host.docker.internal:1488/v1
```

### Qdrant пустой

Запустите индексацию:

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"force": true}'
```

### Ошибка с Hugging Face cache

Если загрузка модели эмбеддингов оборвалась и cache поврежден, сбросьте volume:

```bash
docker compose down -v
docker compose up --build app
```

### PostgreSQL еще не готов

`HistoryStore` сам делает несколько попыток подключения. Если приложение не стартовало, проверьте контейнер:

```bash
docker compose ps
docker compose logs postgres
```

## Безопасность и ограничения

Это ассистент по документации, а не медицинская экспертная система. Он отвечает только по найденному контексту и должен сообщать, если в документации недостаточно информации. Для клинических решений, диагностики и назначения лечения ответы нельзя использовать без проверки ответственным специалистом.
