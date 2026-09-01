# Repository localization experiment

Цей репозиторій містить локальний академічний експеримент із пошуку релевантних файлів через
Codex CLI. Інтерфейс має рівно п'ять команд:

```text
repository-localization prepare
repository-localization run
repository-localization features
repository-localization analyze
repository-localization report
```

- `prepare` перевіряє та заморожує всі відкриті входи;
- `run` запускає Codex у трьох режимах роботи з документацією;
- `features` виділяє відкриті фічі prompt і перетворює сирі події на таблицю спостережень;
- `analyze` вперше відкриває gold, додає gold-aware фічу та обчислює метрики;
- `report` формує стабільний `report/data.json` для спільного EDA notebook.

## Структура коду

Код пайплайна розділений за відповідальністю:

- `cli.py` — аргументи п'яти команд і операторські повідомлення;
- `pipeline.py` — малий стабільний API, який з'єднує команди з реалізацією;
- `core.py` — TOML/JSON-контракти, валідація, checksum-и та безпечний запис артефактів;
- `planning.py` — перевірка repository snapshots, побудова й заморожування плану;
- `execution.py` — ізольований запуск клітинок і перевірка сирих Codex events;
- `analysis.py` — `features`, відкриття gold на етапі `analyze` і підготовка EDA-даних;
- `analysis/eda.ipynb` — єдиний notebook для всіх експериментів.

Для розуміння операторського шляху достатньо читати `cli.py`, а потім модуль потрібного етапу.

## Звідки беруться репозиторії та задачі

Основне джерело задач — офіційний набір
[`Contextbench/ContextBench`](https://huggingface.co/datasets/Contextbench/ContextBench), конфігурація
`default`, розділ `train`, зафіксована ревізія
`c2855792b006af41c67202d33883fb9d46362853`. Для запланованого основного запуску використовується вже
відібраний до отримання результатів набір із 82 задач у шести публічних Python-репозиторіях:

- `ansible/ansible` — 8;
- `django/django` — 20;
- `huggingface/transformers` — 14;
- `matplotlib/matplotlib` — 8;
- `sphinx-doc/sphinx` — 12;
- `sympy/sympy` — 20.

Пайплайн не завантажує ContextBench чи Git-репозиторії, не відбирає задачі та не виконує юридичну
перевірку або балансування. Ці рішення й підготовка виконуються окремо до експерименту. На вхід
передаються тільки готові локальні файли.

Поля ContextBench переносяться так:

| ContextBench | Вхід експерименту | Призначення |
|---|---|---|
| `instance_id` | `task_id` | незмінний ідентифікатор задачі |
| `repo` | `repository` | публічний репозиторій `owner/name` |
| `base_commit` | `base_commit` | точний стан репозиторію до виправлення |
| `problem_statement` | `prompt` | однаковий текст задачі для всіх режимів |
| підготовлене дерево Git | `source_root` | локальний знімок коду разом зі штатною документацією |
| зовнішній опис документації | `documentation_entry` | загальна для цього знімка стартова сторінка документації |

Для кожного відібраного рядка зовнішня підготовка повинна:

1. отримати публічний репозиторій і перевірити наявність точного `base_commit`;
2. експортувати саме цей commit, наприклад через `git archive`, у локальний каталог без `.git`;
3. залишити в експорті штатну документацію репозиторію;
4. вибрати `documentation_entry` за загальним правилом для репозиторію, а не за підказками конкретної
   задачі;
5. записати відкриту частину в `tasks.jsonl`, а правильні шляхи — окремо в захищений `gold.jsonl`.

Один знімок `repository + base_commit` можна використовувати для кількох задач. Пайплайн не доводить
відповідність експортованих байтів Git commit без `.git`; це обов'язок зовнішньої підготовки. Під час
`prepare` він фіксує одночасно заявлений `base_commit` і контрольну суму всіх фактичних байтів
знімка, тому після заморожування непомітно підмінити дерево не можна.

Рекомендована структура входів:

```text
experiment.toml
tasks.jsonl
protected/
  gold.jsonl
sources/
  <repository>/
    <base-commit>/
      ...код і штатна документація...
```

## Конфігурація

Потрібні Python 3.13, `uv` та самодостатній executable Codex CLI.

```bash
uv sync --group dev
uv run repository-localization --help
```

Скопіюйте tracked-приклад у локальний конфіг і відредагуйте значення:

```bash
cp experiment.example.toml experiment.toml
```

Повний контракт запуску наведений один раз у
[`experiment.example.toml`](experiment.example.toml). Той самий `experiment.toml` використовують
команди пайплайна і спільний notebook; окремої EDA-конфігурації немає.

`dataset_revision` — повний Git commit набору ContextBench, з якого зовні підготовлено вибірку.
`experiment_version` задає оператор. Вона записується в усі артефакти запуску. `prepare` також
обчислює `plan_id` із ревізії набору, точного конфігу, `tasks.jsonl`, `base_commit`, байтів усіх
знімків, `documentation_entry` і Codex executable. Якщо щось із цього змінено, наявну версію
продовжити не можна — потрібно вказати нову `experiment_version`.

`runner.binary` має вказувати безпосередньо на executable Codex CLI, а не на shim менеджера версій.

## Формат задач

`tasks.jsonl` містить по одному відкритому рядку на задачу:

```json
{"task_id":"SWE-Bench-Verified__python__maintenance__bugfix__example","repository":"django/django","base_commit":"0123456789abcdef0123456789abcdef01234567","prompt":"Locate the files relevant to this issue.","source_root":"sources/django/0123456789abcdef0123456789abcdef01234567","documentation_entry":"docs/index.txt"}
```

`base_commit` повинен бути повним 40-символьним Git SHA нижнього регістру. `source_root` може бути
відносним до `experiment.toml` або абсолютним. Це знімок без службових даних Git: `prepare` відхиляє
`.git`, `.codex`, `.agents`, `.experiment`, символічні й жорсткі посилання та сторонні
`AGENTS*.md`.

`documentation_entry` — наявний непорожній файл усередині того самого `source_root`. Його не можна
вибирати під конкретну задачу або за близькістю до правильної відповіді.

`gold.jsonl` зберігається окремо від відкритих задач і source roots:

```json
{"task_id":"SWE-Bench-Verified__python__maintenance__bugfix__example","files":["django/core/handlers/base.py"]}
```

`prepare`, `run`, `features` і `report` не відкривають gold. Його читає лише `analyze` після
завершення запусків.

## Фічі тексту задачі

`features` детерміновано виділяє з публічного `prompt` три булеві поля без доступу до gold:

- `prompt_has_path` — є path-like значення з `/`;
- `prompt_has_filename` — є ім'я файла з підтримуваним source/documentation розширенням;
- `prompt_has_symbol` — є backtick/call, `snake_case` або `CamelCase` ідентифікатор.

На етапі `analyze` додається `gold_locator_mentioned`. Воно дорівнює `true`, якщо знайдений у
`prompt` шлях або filename відповідає gold-файлу, або явний symbol відповідає class/function/method,
визначеному в Python gold-файлі. Ця фіча не потрапляє у `features/data.jsonl`, бо до `analyze`
правильні файли залишаються закритими. Це описова фіча задачі, а не оцінка її складності.

## Режими роботи з документацією

Код, штатна документація та текст задачі однакові в усіх трьох режимах:

- `NO-DOC` — додаткової інструкції про документацію немає;
- `OPTIONAL` — кореневий `AGENTS.md` повідомляє стартовий файл документації та дозволяє
  скористатися ним за потреби;
- `DOC-FIRST` — `AGENTS.md` вимагає прочитати стартовий файл документації
  перед першим переглядом вихідного коду.

Пайплайн створює нову checksum-verified копію того самого `source_root` для кожного режиму. У двох
режимах з інструкцією він додає лише короткий заморожений кореневий `AGENTS.md`. Інша документація
не генерується і не копіюється.

Codex повертає від одного до п'яти унікальних шляхів у порядку ймовірності, без штучного доповнення
списку. `analyze` обчислює основні `Recall@3`, `nDCG@3`, `returned_set_f1` і додатковий `Recall@5`.

## Запуск

### Git-процес експерименту

`experiment.toml`, `tasks.jsonl`, `protected/`, `sources/`, `experiments/` і `results/` ігноруються за
замовчуванням. Завдяки цьому конфігурація конкретного запуску, gold-відповіді, локальні repository
snapshots і результати випадково не потрапляють у release коду.

Кожен фактичний експеримент виконуйте в окремій гілці, яку не потрібно merge-ити в `main`:

```bash
git switch -c experiment/<experiment-id>-<experiment-version>
cp experiment.example.toml experiment.toml
# Підготуйте experiment.toml, tasks.jsonl, protected/gold.jsonl і локальні source_root.

git add -f experiment.toml tasks.jsonl
git commit -m "experiment: freeze <experiment-id> <experiment-version> config"
```

Якщо experiment-гілка приватна і має зберігати gold, додайте `protected/gold.jsonl` до першого
коміту через `git add -f`. У публічну гілку gold не додавайте. Repository snapshots зазвичай не
комітяться через їхній розмір; `prepare` запише заявлений commit і checksum фактичного дерева в
`plan.json`.

Після першого коміту не змінюйте конфіг, задачі або source snapshots для цієї
`experiment_version`. Запустіть увесь пайплайн:

```bash
uv run repository-localization prepare experiment.toml
uv run repository-localization run experiment.toml
uv run repository-localization features experiment.toml
uv run repository-localization analyze experiment.toml
uv run repository-localization report experiment.toml
```

`report` не копіює notebook у каталог результатів. Після його виконання відкрийте один спільний
[`analysis/eda.ipynb`](analysis/eda.ipynb). Notebook читає той самий `experiment.toml` зі змінної
`EXPERIMENT_CONFIG` або використовує `experiment.toml` у поточному каталозі:

```bash
EXPERIMENT_CONFIG=experiment.toml jupyter lab analysis/eda.ipynb
```

Notebook обчислює шлях до `report/data.json` із `artifact_dir`, `experiment_id` та
`experiment_version`. Тому конфігурація запуску й джерело EDA завжди описують одну версію
експерименту.

Після завершення додайте створені артефакти другим окремим комітом:

```bash
git add -f results/<experiment-id>/<experiment-version>
git commit -m "experiment: record <experiment-id> <experiment-version> results"
```

Так конфіг зафіксований до появи outcomes, а результати мають окремий перевірюваний commit. Кілька
задач одного репозиторію задаються окремими рядками `tasks.jsonl` і не заважають одна одній.

Після перерваного запуску дозволене тільки явне продовження:

```bash
uv run repository-localization run experiment.toml --resume
```

Автоматичних retry немає. Завершені, terminal або claimed-without-outcome cells повторно не
запускаються. Terminal однієї клітинки не зупиняє незалежні untouched клітинки; після збереження
всіх outcomes `run` повертає помилку із загальним переліком terminal. `features`, `analyze` і
`report` зберігають такі outcomes окремо, а метрики рахують лише для успішних спостережень. Якщо
Codex CLI тричі поспіль повідомляє про очікування мережі, клітинка стає `network_unavailable`, не
чекаючи загального timeout.

## Результати

```text
<artifact_dir>/<experiment_id>/<experiment_version>/
  plan.json
  claims/<cell-id>.json
  runs/<cell-id>/
    manifest.json
    observation.json
    events.jsonl
    stderr.log
    final-output.json
  features/{manifest.json,data.jsonl}
  analysis/{manifest.json,data.json}
  report/{manifest.json,data.json}
```

`experiment_id`, `experiment_version` і `plan_id` записані в plan, claims, observations, похідні
дані та manifests. Сирі Codex events, stderr і final output зберігаються без втрати та прив'язані
checksums через `observation.json`.

## Перевірка коду

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

## Окремий release-репозиторій

Не додавайте новий remote безпосередньо до цього worktree: його Git-історія містить попередній
проєкт і вже виконані експерименти. Для публікації без цієї історії експортуйте лише tracked tree
підготовленого release-коміту в новий каталог, створіть там новий Git-репозиторій і зробіть один
початковий коміт. Перед першим release перевірте, що `git ls-files` не містить `results/`, фактичних
`experiment.toml`, `tasks.jsonl` або `protected/`, а потім виконайте тести й `uv build`.

У новому репозиторії `main` містить тільки код, документацію, тести та
`experiment.example.toml`. Реальні запуски зберігаються в окремих приватних experiment-гілках або в
окремому архівному remote. Для першої версії достатньо annotated tag `v0.1.0` на чистому release-
коміті.
