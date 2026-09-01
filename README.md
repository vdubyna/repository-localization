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
- `features` виділяє відкриті фічі prompt і документаційні читання із сирих подій;
- `analyze` вперше відкриває gold, додає тип задачі, trajectory-ознаки, метрики й дві канонічні CSV;
- `report` перевіряє похідні артефакти та формує стабільний машинозчитуваний підсумок.

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

### Як завантажити й розмістити репозиторій

Git checkout використовується лише як локальний кеш. На вхід експерименту передається чистий
знімок точного `base_commit` без `.git`. Наприклад, для `sympy/sympy` і commit
`b4777fdcef467b7132c055f8ac2c9a5059e6a145`:

```bash
mkdir -p .local/repositories/sympy
git clone --filter=blob:none --no-checkout \
  https://github.com/sympy/sympy.git \
  .local/repositories/sympy/sympy

git -C .local/repositories/sympy/sympy fetch --depth=1 origin \
  b4777fdcef467b7132c055f8ac2c9a5059e6a145
git -C .local/repositories/sympy/sympy rev-parse \
  'b4777fdcef467b7132c055f8ac2c9a5059e6a145^{commit}'

mkdir -p sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145
git -C .local/repositories/sympy/sympy archive \
  b4777fdcef467b7132c055f8ac2c9a5059e6a145 \
  | tar -xf - -C sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145
```

`rev-parse` має надрукувати той самий повний 40-символьний commit. Після експорту рядок задачі
посилається саме на каталог знімка:

```json
{"task_id":"<stable-task-id>","repository":"sympy/sympy","base_commit":"b4777fdcef467b7132c055f8ac2c9a5059e6a145","prompt":"<task text>","source_root":"sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145","documentation_entry":"doc/src/index.rst"}
```

`source_root` обчислюється від каталогу, де лежить `experiment.toml`. `documentation_entry`
обчислюється від `source_root`, тому тут файл має існувати як
`sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145/doc/src/index.rst`.
Перевірте знімок до `prepare`:

```bash
test ! -d sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145/.git
test -s sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145/doc/src/index.rst
find sources/sympy/sympy/b4777fdcef467b7132c055f8ac2c9a5059e6a145 -type l -print
```

Остання команда не повинна нічого вивести. `prepare` додатково відхиляє жорсткі посилання,
службові каталоги `.git`, `.codex`, `.agents`, `.experiment` і наявні `AGENTS*.md`. Якщо точний
commit містить такі файли, його не можна мовчки змінювати: виключіть задачу до формування вибірки
або заздалегідь визначте й задокументуйте єдине правило матеріалізації для всіх задач.

Для іншого commit того самого репозиторію повторіть `fetch`, `rev-parse` й `archive`; повторний
`clone` не потрібен.

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
`runner.profiles` містить до восьми унікальних пар `model`/`reasoning_effort`. Основний контракт у
`experiment.example.toml` фіксує всі вісім профілів дослідження. Для кожної задачі `prepare` створює
окрему клітинку для кожної комбінації profile × condition × repeat; модель і reasoning-рівень
записуються в plan, claim, observation та канонічну таблицю.

## Формат задач

`tasks.jsonl` містить по одному відкритому рядку на задачу:

```json
{"task_id":"SWE-Bench-Verified__python__maintenance__bugfix__example","repository":"django/django","base_commit":"0123456789abcdef0123456789abcdef01234567","prompt":"Locate the files relevant to this issue.","source_root":"sources/django/django/0123456789abcdef0123456789abcdef01234567","documentation_entry":"docs/index.txt"}
```

`base_commit` повинен бути повним 40-символьним Git SHA нижнього регістру. `source_root` може бути
відносним до `experiment.toml` або абсолютним. Це знімок без службових даних Git: `prepare` відхиляє
`.git`, `.codex`, `.agents`, `.experiment`, символічні й жорсткі посилання та сторонні
`AGENTS*.md`.

`documentation_entry` — наявний непорожній файл усередині того самого `source_root`. Його не можна
вибирати під конкретну задачу або за близькістю до правильної відповіді. Під час `prepare` до plan
записується повний перелік файлів у каталозі цієї стартової сторінки. Для стартової сторінки в корені
репозиторію документаційним набором вважається лише вона. Саме заморожений перелік, а не розширення
файла чи назва каталогу, використовується для підрахунку читань.

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

На етапі `analyze` ці відкриті сигнали звіряються з gold. `task_type` дорівнює
`EXPLICIT_LOCATOR_CLUE`, якщо prompt містить точний gold-шлях, ім'я gold-файла або Python-символ,
визначений у gold-файлі; інакше значення — `NO_EXPLICIT_LOCATOR_CLUE`. До `analyze` ця класифікація
не зберігається.

Той самий етап додає три ознаки ходу пошуку:

- `gold_seen_any` — gold-шлях з'явився у виводі будь-якої дії з кодом;
- `gold_seen_by_3_source_actions` — це сталося не пізніше третьої дії з кодом;
- `gold_targeted_any` — команда безпосередньо звернулася до gold-шляху.

Чотири gold-free ознаки документації обчислюються раніше з `command_execution` events:
`wiki_read_count`, `wiki_tokens`, `unique_wiki_pages` і `beyond_entry_reads`. Документаційним читанням
є дія з непорожнім виводом, команда якої однозначно посилається лише на шляхи із замороженого
документаційного набору. `wiki_tokens` рахується за зафіксованим у plan
`tiktoken 0.13.0 / o200k_base`.

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
snapshots і результати випадково не потрапляють у `main`.

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

`report` не копіює notebook у каталог результатів. Після `analyze` відкрийте один спільний
[`analysis/eda.ipynb`](analysis/eda.ipynb). Notebook читає той самий `experiment.toml` зі змінної
`EXPERIMENT_CONFIG` або використовує `experiment.toml` у поточному каталозі:

```bash
EXPERIMENT_CONFIG=experiment.toml jupyter lab analysis/eda.ipynb
```

Notebook обчислює шляхи до `features/cell_features.csv` і `features/task_features.csv` із
`artifact_dir`, `experiment_id` та `experiment_version`, а потім перевіряє їхню identity. Окремого
EDA-конфігу, імпортованого генератора звіту або захардкодженого шляху немає.

### Рисунки розділу 4

Вісім окремих дослідницьких рисунків генеруються з таблиці
`features/cell_features.csv` без повторного запуску провайдера:

```bash
uv run repository-localization report experiments/82-tasks/experiment-record.toml --figures
```

Команда записує PNG і PDF у
`results/<experiment-id>/<experiment-version>/report/figures/`. `manifest.json` у цьому каталозі
містить версію експерименту, `plan_id`, checksum вхідної таблиці, точні значення на рисунках і
checksum кожного файла. Усі середні та різниці обчислюються з рядків запусків; у коді не
захардкоджено значення з тексту дослідження. Bootstrap і довірчі інтервали не будуються.

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
  features/
    manifest.json
    data.jsonl
    cell_features.csv
    task_features.csv
  analysis/{manifest.json,data.json}
  report/{manifest.json,data.json}
```

`experiment_id`, `experiment_version` і `plan_id` записані в plan, claims, observations, похідні
дані та manifests. Сирі Codex events, stderr і final output зберігаються без втрати та прив'язані
checksums через `observation.json`.

`cell_features.csv` містить один рядок на завершену або terminal клітинку. `task_features.csv`
спочатку усереднює успішні profile/repeat спостереження в межах task × condition, а потім записує
парні різниці `doc_first_minus_optional`, `doc_first_minus_no_doc` і `optional_minus_no_doc` для
якості, ресурсів, читання документації та trajectory-ознак. Незалежною одиницею такого порівняння є
задача, а не окремий модельний запуск.

## Перевірка коду

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```
