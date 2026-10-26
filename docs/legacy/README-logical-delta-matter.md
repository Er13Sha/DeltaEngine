# Logical Delta Matter Unfolder

Рабочий прототип для проекта развертки материи по логическим дельтам.

> **Примечание (27.10.2026).** Файл сохранён как справка о прежнем прототипе LogicalDeltaCorrect; текст ниже не менялся. **Код прототипа в DeltaEngine не перенесён** (решение заказчика: строим с нуля), поэтому команды из README здесь не работают; они работают в `C:\IT\ML\LogicalDeltaCorrect`.
> Для нового проекта сохранены только данные: `data/matter-deltas.json`, `data/rule-packs.json`, `data/confirmed-concepts.json`; PDF лежат в `data/sources/documents/`, отчёт — в `docs/research/`.
> Расчётные файлы прототипа (`computed-compound-space.json`, `exact-latest.json`) не перенесены. Контрольный перезапуск 27.10.2026 показал, что `computed-compound-space.json` из старого проекта не воспроизводится побайтно кодом прототипа (35 238 152 байта против 34 057 902), поэтому его нельзя считать проверенным эталоном; `exact-latest.json` совпал побайтно.

## Корневой домен

`VirtualMatterObject` - конечная запись об объекте внутри ограниченного виртуального объема. В прототипе запись содержит тотальные boolean-поля:

- `continuous`
- `grouped`
- `planetary`
- `preservesShape`
- `preservesVolume`
- `ionized`

Первая корневая дельта:

```text
delta_id: matter.continuous
domain: VirtualMatterObject
predicate: Continuous(x)
true: Field
false: Particles
```

Дельта допустима в прототипе, потому что решатель читает одно boolean-поле конечной записи: он тотален на валидном объекте, исключает одновременные ветви `true/false` и завершается за `O(1)`.

## Дерево материи

```mermaid
graph TD
  M[Материя]
  M -->|Continuous(x)| F[Поле]
  M -->|not Continuous(x)| P[Частицы]
  P -->|Grouped(x)| GP[Группы частиц]
  P -->|not Grouped(x)| FP[Свободные частицы]
  GP -->|Planetary(x)| SP[Субчастицы]
  GP -->|not Planetary(x)| S[Вещество]
  S -->|PreservesShape(x)| SOL[Твердое вещество]
  S -->|not PreservesShape(x)| FL[Жидкости и газы]
  FL -->|PreservesVolume(x)| LIQ[Жидкость]
  FL -->|not PreservesVolume(x)| GAS[Газ]
  GAS -->|Ionized(x)| PL[Плазма]
  GAS -->|not Ionized(x)| NG[Нейтральный газ]
```

Полный JSON-манифест лежит в [`data/matter-deltas.json`](data/matter-deltas.json).
Уточненная архитектурная спецификация лежит в [`docs/architecture.md`](docs/architecture.md), пакеты правил - в [`data/rule-packs.json`](data/rule-packs.json).

## Что делает прототип

Прототип является ограниченным символическим конвейером, а не физической симуляцией:

1. Принимает конечный виртуальный объем и целые запасы протонов, нейтронов, электронов.
2. Считает емкость выбранного диапазона элементов: `H, C, N, O, Na, Si, Cl, Fe`.
3. Считает возможные соединения по рецептам: `H2, O2, N2, H2O, CO2, CH4, NH3, NaCl, SiO2, C6H12O6, C2H5NO2`.
4. Выводит простые био- и техно-классы из рассчитанных соединений.
5. Классифицирует все материальные объекты по бинарному дереву дельт.
6. Принимает текстовый запрос пользователя и сопоставляет его с локальным словарем понятий.
7. Предрассчитывает пространство возможных/невозможных соединений для rule-pack `katyushchik_hypothesis` и показывает причины отсечения.

Расчет хранит только текущие агрегаты и не разворачивает все возможные миры в памяти.
Текущее состояние канонизируется и получает хэш `fnv1a32:*`; хэш включает входные счетчики, выбранный rule-pack, активные элементы, соединения, функциональные классы и листья классификации.

## Запросы пользователя

Пользователь может писать ограниченные предметные запросы, например:

- `хочу получить воду`
- `получить глюкозу`
- `нужен проводник`
- `техническое устройство`
- `сколько возможно кислорода`

Разбор запроса детерминированный: программа нормализует текст, ищет понятие в словаре и возвращает точную емкость, потребность в `p/n/e`, нехватки, rule-pack, хэш состояния и путь классификации. Если понятие не найдено или запрос неоднозначен, возвращается `UNKNOWN`, а не вероятностная догадка.

Словарь расширяется из [`data/confirmed-concepts.json`](data/confirmed-concepts.json). Текущий снимок содержит все 118 элементов из PubChem Periodic Table и расширенный seed фундаментальных соединений. Файл строится скриптом:

```bash
python3 scripts/build_confirmed_concepts.py
```

Список импортируемых соединений лежит в [`data/compound-seed.json`](data/compound-seed.json). Чтобы расширить научную базу, добавьте туда новые `query/formula/aliases_ru` и пересоберите `confirmed-concepts.json`.

Подтвержденные поля для элементов и соединений берутся из PubChem PUG-REST. Русские алиасы, агрегатное состояние для прототипа и расчетное число нейтронов помечены как локальный/model слой, а не как проверенный isotope claim. Полный PubChem целиком не вендорится в репозиторий, потому что это огромная внешняя база; вместо этого проект хранит проверяемый локальный снимок и воспроизводимый скрипт импорта.

## Вычисленное пространство соединений

Файл [`data/computed-compound-space.json`](data/computed-compound-space.json) строится из подтвержденных элементов/соединений и конечного конфига [`data/compound-space-config.json`](data/compound-space-config.json):

```bash
python3 -m logical_delta_matter.compound_cli \
  --concepts data/confirmed-concepts.json \
  --config data/compound-space-config.json \
  --out data/computed-compound-space.json
```

Текущий снимок: `12 890` кандидатов, `12 077` possible, `813` impossible. Это исследовательский слой `katyushchik_hypothesis`, а не подтвержденная химическая база. Для каждого кандидата хранится `decision_trace`: проверка известных элементов, минимального числа атомов, одноименного отталкивания для гомоядерных формул, разнообразия классов, инертных газов и ресурсной емкости `p/n/e`.

Веб-интерфейс читает этот файл во вкладке `Расчет`. Если пользователь вводит формулу из вычисленного пространства, например `Al2Ca`, программа возвращает вычисленный статус. Если выбран rule-pack `Гипотеза Катющика`, подтвержденные формулы дополнительно получают дельта-вердикт: например `H2` отсекается текущим правилом гомоядерного одноименного отталкивания, а `H2O` и `NaCl` проходят.

## Rule-pack режимы

- `reference_science` - опорный режим прототипа: атомы считаются через целочисленные счетчики `p/n/e`, соединения через локальные рецепты.
- `katyushchik_hypothesis` - исследовательский режим для альтернативных положений из материалов. В текущей версии он маркирует происхождение результата и входит в хэш, но не подменяет trusted-core физикой.

## Формальная логика, SMT и геометрия

Добавлен Python-слой для схем из нового ТЗ:

- [`logical_delta_matter/formal_domains.py`](logical_delta_matter/formal_domains.py) - классы и решатели для иерархии материи, 4 состояний дискретности и пар плоскостей.
- [`logical_delta_matter/exact_algebra.py`](logical_delta_matter/exact_algebra.py) - символьное форматирование пространства `R`, `n`, `L = R*n`, `T = R/n`; использует `sympy`, если он установлен, иначе сохраняет точные символические токены без `float`.
- [`logical_delta_matter/exact_polynomial.py`](logical_delta_matter/exact_polynomial.py) - точная алгебра `Q[n,n^-1]` на `Fraction`; `float` запрещен.
- [`logical_delta_matter/proof_engine.py`](logical_delta_matter/proof_engine.py) - strict `SOUND | UNKNOWN | REJECTED_CERTIFICATE`, proof obligations, запрет `Admitted/sorry/admit/unsafe` в proof artifacts и проверка отсутствия `float` в trusted path.
- [`logical_delta_matter/source_registry.py`](logical_delta_matter/source_registry.py) - SHA-256 реестр источников; текущий снимок лежит в [`data/source-registry.json`](data/source-registry.json).
- [`logical_delta_matter/smt_certificates.py`](logical_delta_matter/smt_certificates.py) - optional z3-сертификаты для `P / not P`; без `z3-solver` возвращает машинно-читаемое обязательство со статусом `unavailable`.
- [`logical_delta_matter/galaxy_shadow.py`](logical_delta_matter/galaxy_shadow.py) - direct `O(N^2)` pressure-shadow модуль: CPU reference, CUDA RawKernel source, optional CuPy RawKernel builder и optional Numba CUDA kernel builder.
- [`cuda/galaxy_shadow_direct.cu`](cuda/galaxy_shadow_direct.cu) - целевой CUDA C++ backend: SOA layout, tiled shared memory, direct `O(N^2)`, Velocity Verlet.
- [`logical_delta_matter/gui_app.py`](logical_delta_matter/gui_app.py) - минимальная DearPyGui-панель с ползунками давления и коэффициента тени; запускается только если установлен `dearpygui`.
- [`logical_delta_matter/vispy_viewer.py`](logical_delta_matter/vispy_viewer.py) - optional VisPy viewer: 3D частицы и 2D график радиальных скоростей.

Демо формального слоя:

```bash
python3 -m logical_delta_matter.formal_cli --demo all
```

Обновление реестра источников:

```bash
python3 scripts/build_source_registry.py
```

GUI-заготовка:

```bash
python3 -m logical_delta_matter.gui_app
```

VisPy-заготовка:

```bash
python3 -m logical_delta_matter.vispy_viewer
```

На машине без optional-зависимостей GUI/SMT/GPU-адаптеры не падают: ядро, тесты и CPU reference продолжают работать.

## Запуск

Веб-прототип:

```bash
python3 -m http.server 4173
```

Затем открыть:

```text
http://localhost:4173/
```

HTML можно открыть и напрямую, но через локальный сервер браузер корректно загрузит JSON-манифест.

Exact-дискретный Python слой:

```bash
python3 -m logical_delta_matter.cli \
  --config configs/exact_mvp.json \
  --out runs/exact_mvp \
  --export-ui data/exact-latest.json
```

После запуска CLI текущий веб-интерфейс показывает результат во вкладке `Exact`.

Предрасчет соединений для UI:

```bash
python3 -m logical_delta_matter.compound_cli
```

Тесты:

```bash
python3 -m unittest discover -s tests
```

Примерный запуск без аргументов:

```bash
python3 example_run.py
```

## Python exact-MVP

Поверх исходного веб-прототипа добавлен пакет [`logical_delta_matter`](logical_delta_matter):

- `delta_core.py` - компиляция, валидация и классификация бинарного дерева дельт.
- `world_exact.py` - точная 3D-решетка, целочисленные локальные потенциалы и детерминированный шаг мира.
- `aggregate.py` - поиск связных компонент и сигнатуры кластеров.
- `templates.py` - инженерное повышение кластеров в элементы-аналоги.
- `storage.py` - JSON/JSONL чекпоинты и event log.
- `viz.py` - SVG-срез мира и SVG-граф агрегатов.
- `runner.py` / `cli.py` - полный прогон, метрики и экспорт для UI.

Ядро не использует `float`: позиции, веса, потенциалы, сигнатуры и хэши считаются через целые числа и строки.

## Ограничения

- Это детерминированный прототип поля ограничений, а не модель квантовой хромодинамики, нуклеосинтеза или химической кинетики.
- Емкость целевого запроса считается по прямой потребности в `p/n/e`; общий обзор показывает максимумы для каждого типа отдельно.
- Для следующего уровня нужно добавить формальные решатели для совместной материализации состава, верифицируемые доказательства и расширяемый реестр источников.
- Exact-MVP формирует внутримодельные элементы-аналоги и молекулярные/сборочные уровни. Это исследовательская модель, а не утверждение о реальной химии.
