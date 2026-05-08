# Пайплайн подготовки чанков для RAG

Единая точка входа для шагов, которые раньше были разрозненными скриптами: подписи к изображениям, сборка чанков из текстов, расшифровка аббревиатур и (опционально) генерация `chunk_context` через LLM.

## Расположение данных

Все пути по умолчанию относительно каталога `pipeline/`:

| Что | Где |
|-----|-----|
| Исходные изображения | `pipeline/images/` |
| Тексты с плейсхолдерами `{{имя_файла}}` | `pipeline/texts/*.txt` |
| Артефакты (jsonl, логи, чекпойнты) | `pipeline/output/` |

Старые пути `dataset/images` и `dataset/texts` пайплайн не использует.

## Зависимости

Нужен Python 3.10+ и пакеты:

```text
openai
tqdm
pillow
```

Пример установки:

```bash
pip install openai tqdm pillow
```

## Конфигурация

Файл `pipeline/config.json` (можно скопировать из `config.example.json`):

- **`paths`** — каталоги и файлы: `images`, `texts`, `output_dir`, `image_descriptions`, `chunks`, `chunks_expanded`, `terms` (CSV с колонками `name`, `meaning`).
- **`sources`** — список модулей-источников чанков. У каждой записи:
  - `id` — короткое имя для CLI (`--sources`);
  - `class` — импорт в формате `модуль:Класс` (см. ниже);
  - `enabled` — участвует ли источник в обычной сборке;
  - `options` — опции, специфичные для источника.

По умолчанию `terms` указывает на `../dataset/terms.csv`. При желании скопируйте `terms.csv` в `pipeline/` и поменяйте путь на `"terms": "terms.csv"`.

## Запуск

Рабочая директория — **корень репозитория** (где лежит пакет `pipeline`):

```bash
python -m pipeline --help
python -m pipeline captions
python -m pipeline build
python -m pipeline expand
python -m pipeline context
```

Объединённый прогон (без LLM-контекста по умолчанию):

```bash
python -m pipeline run --steps captions,build,expand
```

С шагом контекста (нужен запущенный OpenAI-совместимый API, например LM Studio):

```bash
python -m pipeline run --steps captions,build,expand --with-context
```

Другой файл конфигурации:

```bash
python -m pipeline --config path/to/config.json build
```

### Подписи к изображениям (`captions`)

Читает файлы из `pipeline/images/`, дописывает строки в `output/image_descriptions.jsonl`, ведёт лог и прогресс в `output/`. Параметры API как в старом `generate_image_descriptions.py`: `--base-url`, `--model`, `--limit` и т.д.

### Сборка чанков (`build`)

Обходит все источники с `"enabled": true` в порядке списка, объединяет записи и перезаписывает `output/chunks.jsonl`, выставляя `chunk_id` подряд с 1.

У каждой строки JSONL поля: `chunk_id`, `chunk_text`, `source_file`, `chunk_context`, `doc_title`, `h1`–`h6`, `source` (идентификатор модуля).

### Добавление новых данных без пересборки всего корпуса (`build --append`)

Чтобы **дописать** чанки в конец уже существующего `chunks.jsonl`:

1. Укажите флаг `--append`.
2. Обязательно укажите `--sources id1,id2` — какие источники прогнать (так не получится случайно продублировать весь `txt_docs`).

Пример: в конфиге заведён источник `markdown_notes`, в каталоге `pipeline/markdown_import/` лежат `.md` файлы:

```bash
python -m pipeline build --append --sources markdown_notes
```

Явный список `--sources` игнорирует флаг `enabled` в конфиге: можно один раз собрать основной корпус с `markdown_notes` выключенным, потом догнать только markdown.

### Расшифровка аббревиатур (`expand`)

Читает `output/chunks.jsonl`, пишет `output/chunks_expanded.jsonl` по глоссарию из `terms.csv`.

### Контекст чанков (`context`)

По умолчанию берёт `chunks_expanded.jsonl`, если файла нет — `chunks.jsonl`. Результат пишет в тот же файл, что и вход, если не задан `--output`. Чекпойнты: `output/checkpoints/`.

## Как добавить свой источник данных

1. Создайте класс с атрибутом `source_id` и методом `collect(self, ctx, options) -> list[dict]`, где `ctx` — `PipelineContext` (`root`, `texts_dir`, `images_dir`, `output_dir`, `image_descriptions`).

2. Каждая запись в возвращаемом списке — словарь с полями как у текущего пайплайна (как минимум `chunk_text`, `source_file`, `chunk_context`, `doc_title`, `h1`–`h6`, `source`). Поле `chunk_id` задавать не нужно — его проставит `build`.

3. Зарегистрируйте класс в `config.json` в массиве `sources`:

```json
{
  "id": "my_wiki",
  "class": "my_package.my_module:MyWikiSource",
  "enabled": false,
  "options": { "glob": "wiki/**/*.md" }
}
```

4. Убедитесь, что модуль доступен в `PYTHONPATH` (при разработке внутри репозитория положите код рядом и импортируйте от корня, либо установите пакет в editable-режиме).

### Встроенные источники

| ID | Класс | Назначение |
|----|--------|------------|
| `txt_docs` | `pipeline.sources.txt_docs:TxtDocsSource` | Логика бывшего `build_smart_chunks.py`: `.txt`, заголовки `h1.`–`h6.`, подстановка описаний картинок. |
| `markdown_notes` | `pipeline.sources.markdown_files:MarkdownFilesSource` | Пример: один чанк на `.md` файл, шаблон пути в `options.glob`. |