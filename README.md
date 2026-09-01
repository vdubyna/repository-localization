# Repository localization experiment

Мінімальний конфіг-керований пайплайн для академічного експерименту з пошуку релевантних файлів
через Codex CLI:

```text
repository-localization prepare
repository-localization run
repository-localization features
repository-localization analyze
repository-localization report
```

- `prepare` перевіряє конфіг, задачі та Git commit-и й створює читабельний `plan.json`;
- `run` запускає клітинки `task × profile × condition × repeat`;
- `features` виділяє відкриті фічі prompt, ресурсів і читання документації;
- `analyze` відкриває gold, обчислює якість, тип задачі та trajectory-ознаки;
- `report` формує дані для спільного notebook і, з `--figures`, вісім рисунків.

## Джерело задач

Задачі походять з офіційного набору
[`Contextbench/ContextBench`](https://huggingface.co/datasets/Contextbench/ContextBench), конфігурація
`default`, розділ `train`, ревізія
`c2855792b006af41c67202d33883fb9d46362853`. Відбір задач, legal review і балансування виконуються
до цього пайплайна. Пайплайн не завантажує набір і не змінює вибірку.

Публічна частина задачі міститься в `tasks.jsonl`. Еталонні файли зберігаються окремо в
`protected/gold.jsonl` і читаються лише командою `analyze`.

## Репозиторії

Один локальний Git mirror може обслуговувати всі задачі того самого репозиторію, навіть якщо вони
посилаються на різні commit-и. Наприклад:

```bash
mkdir -p .local/repositories
git clone --mirror https://github.com/sympy/sympy.git \
  .local/repositories/sympy.git
```

Перевірте кожен потрібний commit:

```bash
git -C .local/repositories/sympy.git cat-file -e \
  b4777fdcef467b7132c055f8ac2c9a5059e6a145^{commit}
```

Стан вихідного коду задається самим Git commit. Окремих експортованих `source_root` і checksum-ів
дерева немає. Під час запуску потрібний commit матеріалізується через `git archive` у тимчасовий
read-only каталог без `.git`.

## Конфігурація

Встановіть Python 3.13, `uv` і Codex CLI:

```bash
uv sync --group dev
uv run repository-localization --help
cp experiment.example.toml experiment.toml
```

Єдиний контракт міститься в [`experiment.example.toml`](experiment.example.toml). Репозиторії
задаються один раз:

```toml
schema_version = 1
experiment_id = "sympy-10-tasks"
experiment_version = "2026-09-01-v1"
artifact_dir = "results"

[inputs]
tasks = "tasks.jsonl"
gold = "protected/gold.jsonl"
dataset_revision = "c2855792b006af41c67202d33883fb9d46362853"

[[repositories]]
name = "sympy/sympy"
path = ".local/repositories/sympy.git"

[design]
repeats = 1

[runner]
binary = "/absolute/path/to/native/codex"
parallelism = 8
timeout_seconds = 900

[[runner.profiles]]
model = "gpt-5.6-terra"
reasoning_effort = "medium"
```

`runner.profiles` приймає від одного до восьми унікальних профілів. Повний приклад містить усі
вісім профілів дослідження. `parallelism` задає від одного до восьми одночасних клітинок і
записується у frozen plan; не змінюйте його в межах однієї версії експерименту. Паралельний запуск
скорочує wall-clock час, але `elapsed_seconds` включає конкуренцію за локальні ресурси.
`experiment_version` задається оператором і записується в plan, claims, observations, проміжні
дані, обидві CSV та report.

Codex executable має бути прямим native executable, а не shim менеджера версій. Для Volta це
зазвичай файл `vendor/aarch64-apple-darwin/bin/codex` усередині встановленого пакета.

## Задачі та gold

Один рядок `tasks.jsonl`:

```json
{"task_id":"stable-task-id","repository":"sympy/sympy","base_commit":"b4777fdcef467b7132c055f8ac2c9a5059e6a145","prompt":"Locate files relevant to this issue.","documentation_entry":"doc/src/index.rst"}
```

- `repository` має відповідати `repositories[].name` у конфігу;
- `base_commit` — повний 40-символьний Git SHA;
- `documentation_entry` — непорожній файл у цьому commit;
- один репозиторій може мати багато задач і багато різних commit-ів.

Відповідний рядок `protected/gold.jsonl`:

```json
{"task_id":"stable-task-id","files":["sympy/printing/str.py"]}
```

## Умови

- `NO-DOC` — без додаткової інструкції про документацію;
- `OPTIONAL` — Codex отримує шлях до стартового документа і може його прочитати;
- `DOC-FIRST` — перший tool call Codex має прочитати тільки стартовий документ. `run` перевіряє
  порядок нативних events і одразу зупиняє експеримент при порушенні.

Для кожної пари `task × condition` створюється один read-only workspace і повторно використовується
профілями. Кожна клітинка має окремий тимчасовий `HOME`, `CODEX_HOME`, output schema та ephemeral
Codex session, тому задачі й профілі не ділять стан агента.
`run` примусово задає Codex CLI `web_search = "disabled"`; read-only sandbox блокує мережу для
shell-команд, а raw events додатково перевіряються на web search і MCP. Виявлення зовнішнього
інструмента одразу зупиняє запуск, щоб клітинки залишалися локальними й порівнюваними.

## Git-процес і запуск

Конфіг, задачі, gold, локальні mirrors і результати ігноруються в `main`. Для експерименту створіть
окрему гілку та спочатку закомітьте входи:

```bash
git switch -c experiment/sympy-10-tasks
git add -f experiment.toml tasks.jsonl protected/gold.jsonl
git commit -m "experiment: freeze sympy 10-task inputs"
```

Запустіть етапи послідовно:

```bash
uv run repository-localization prepare experiment.toml
uv run repository-localization run experiment.toml
uv run repository-localization features experiment.toml
uv run repository-localization analyze experiment.toml
uv run repository-localization report experiment.toml
uv run repository-localization report experiment.toml --figures
```

Після переривання дозволене лише явне продовження:

```bash
uv run repository-localization run experiment.toml --resume
```

Автоматичних retry немає. Завершені та terminal клітинки не запускаються повторно. Claim без
durable outcome також не повторюється, бо невідомо, чи provider уже виконав запит.

Після завершення закомітьте результати окремо:

```bash
git add -f results/sympy-10-tasks/2026-09-01-v1
git commit -m "experiment: record sympy 10-task results"
```

Git commit входів фіксує дизайн, а наступний Git commit фіксує всі отримані байти. Додаткових
checksum-manifest-ів немає.

## Результати

```text
results/<experiment-id>/<experiment-version>/
  plan.json
  claims/<cell-id>.json
  runs/<cell-id>/
    observation.json
    events.jsonl
    stderr.log
    final-output.json
  features/
    data.jsonl
    cell_features.csv
    task_features.csv
  analysis/data.json
  report/
    data.json
    figures/*.png
    figures/*.pdf
```

`cell_features.csv` має один рядок на клітинку. `task_features.csv` спочатку усереднює успішні
profile/repeat спостереження в межах `task × condition`, а потім записує парні task-level різниці.

Спільний notebook читає ці дві таблиці через той самий `experiment.toml`:

```bash
EXPERIMENT_CONFIG=experiment.toml jupyter lab analysis/eda.ipynb
```

## Перевірка коду

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```
