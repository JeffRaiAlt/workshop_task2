# Graph + Tree Edge Policy: подробный алгоритм

Файл с расчётом: `graph_tree_edge_policy_experiment.py`.

Цель: найти профили, которые относятся к одному клиенту, но не сравнивать каждый профиль с каждым. Поэтому процесс идёт в два слоя:

1. `blocking_index` быстро предлагает ограниченный набор пар-кандидатов.
2. Дерево решений и граф решают, какие из этих пар действительно склеивать.

Короткая схема:

```text
profile mart + blocking_index
-> recommended blocking rules
-> candidate pairs
-> pair evidence + similarity
-> decision tree edge policy
-> threshold + mutual top-K
-> NetworkX connected components
-> сравнение с entity_id
-> итоговые метрики
```

## Термины

`profile_id` — отдельный профиль/запись.

`entity_id` — настоящий клиент из разметки. Один `entity_id` может иметь один или несколько `profile_id`.

`blocking_index` — таблица вида:

```text
profile_id + block_family + block_rule + block_value
```

Она говорит: этот профиль попал в такой-то блок по такому-то правилу.

`block_rule` — конкретное правило построения блока, например:

```text
rule__coverage__np_subdivision__np_device__np_osfamily
```

`block_family` — семейство правила: `context`, `behavior`, `coverage_compound`, `identity_rescue` и т.д.

`block_value` — конкретное значение блока, например:

```text
RU-MOW|smartphone|android
```

`candidate pair` — пара профилей, которую мы решили сравнить. Она появляется, если два профиля попали в один и тот же блок.

`edge` / ребро — связь между двумя `profile_id` в графе. Если ребро поставлено, граф считает, что эти два профиля надо склеить.

`connected component` / компонента связности — группа профилей, соединённых рёбрами. В финальном решении одна компонента считается найденной группой клиента.

## 1. Загружаем входные артефакты

Скрипт читает:

```text
data/processed/er_profile_mart_multivalue/profile_core.parquet
data/processed/er_profile_mart_multivalue/profile_value_summary_long.parquet
data/processed/er_profile_mart_multivalue/blocking_index.parquet
data/processed/er_profile_mart_multivalue/recommended_blocking_rules.csv
data/processed/er_baseline_pair_model/baseline_assignment_model.pkl
```

Роль таблиц:

- `profile_core` — список профилей и настоящий `entity_id`; используется только для обучения/оценки, не как production-признак.
- `profile_value_summary_long` — все нормализованные значения признаков профиля.
- `blocking_index` — первый слой поиска кандидатов.
- `recommended_blocking_rules` — список правил, которые разрешено использовать в текущем эксперименте.
- `baseline_assignment_model.pkl` — старая pair-модель. Её `score` оставлен только для baseline-сравнения, дерево на нём не обучается.

## 2. Делаем split по `entity_id`

Split строится по `entity_id`, а не по `profile_id`.

Зачем: если профили одного настоящего клиента попадут одновременно в valid и test, получится утечка. Модель увидит почти тот же объект при подборе и проверке.

В скрипте:

- `valid` используется для обучения дерева и выбора threshold;
- `test` используется для финальной проверки.

## 3. Оставляем только recommended blocking rules

Из полного `blocking_index` берём только строки, где `block_rule` входит в `recommended_blocking_rules`.

Это означает, что граф не ищет пары по всем профилям. Он видит только пары, которые появились через выбранные blocking-правила.

Важное ограничение:

```text
если настоящая пара не попала в candidate pairs через blocking,
дерево и граф её уже не увидят
```

Поэтому качество всего подхода сильно зависит от blocking-правил.

## 4. Как строится composite blocking rule

Пример правила:

```python
rule__coverage__np_subdivision__np_device__np_osfamily
```

Оно собирает ключ из трёх компонентов:

```text
np.subdivision_1_iso_code
np.device
np.osfamily
```

Важно: берётся не самое частое значение, а все значения из `profile_value_summary_long`.

Если у профиля:

```text
np.subdivision_1_iso_code = RU-MOW, RU-SPE
np.device = smartphone, desktop
np.osfamily = android
```

то профиль попадёт в несколько `block_value`:

```text
RU-MOW|smartphone|android
RU-MOW|desktop|android
RU-SPE|smartphone|android
RU-SPE|desktop|android
```

Почему так: это повышает шанс не потерять дубль, если у профиля было несколько наблюдаемых значений. Цена — больше candidate pairs и больше потенциального шума.

После сборки ключей применяется ограничение размера блока:

```text
BLOCK_MIN_SIZE <= block_size <= BLOCK_MAX_SIZE
```

В текущей логике слишком маленькие блоки без пары и слишком большие шумные блоки не используются для генерации пар.

## 5. Считаем размер и вес блока

Блок определяется так:

```text
block_family + block_rule + block_value
```

Для каждого блока считаем:

```text
block_size = число уникальных profile_id в блоке
block_weight = 1 / log1p(block_size)
```

Пример:

```text
block_value = RU-MOW|smartphone|android
block_size = 3
block_weight = 1 / log1p(3)
```

Смысл: совпадение в маленьком блоке сильнее, чем совпадение в большом.

Если два профиля совпали в блоке из 2 профилей, это почти точечный сигнал. Если они совпали в блоке из 900 профилей, это слабый сигнал: там много случайных соседей.

## 6. Генерируем candidate pairs

Для каждого блока строим все пары профилей внутри него.

Пример:

```text
block_value = X
profiles = P1, P2, P3
```

Получаем:

```text
P1-P2
P1-P3
P2-P3
```

Если одна и та же пара попала в несколько блоков, она сначала появляется несколько раз. Это не ошибка. Потом эти срабатывания агрегируются в признаки пары.

## 7. Собираем pair evidence

Повторы одной пары схлопываются в одну строку.

Для пары считаются признаки:

```text
n_block_rules
n_block_families
min_block_size
sum_block_weight
hit_coverage_fallback
is_fallback_only
has_non_fallback_signal
has_strong_family
has_behavior
has_behavior_context
has_context
has_coverage_compound
only_weak_families
```

Расшифровка:

- `n_block_rules` — сколько разных правил нашли эту пару.
- `n_block_families` — сколько разных семейств правил подтвердили пару.
- `min_block_size` — самый маленький блок, через который пара была найдена.
- `sum_block_weight` — суммарная сила blocking-сигналов.
- `hit_coverage_fallback` — пара была найдена через fallback-правило.
- `is_fallback_only` — пара найдена только через fallback, без нормального сигнала.
- `has_non_fallback_signal` — есть хотя бы один сигнал кроме fallback.
- `has_strong_family` — сработало сильное семейство, например `behavior_context` или `identity_rescue`.
- `only_weak_families` — все сработавшие семейства считаются слабыми.

Эти признаки отвечают на вопрос:

```text
почему blocking вообще предложил сравнить эти два профиля?
```

## 8. Добавляем similarity-признаки

После evidence добавляем признаки сходства самих профилей:

```text
fs_total_jaccard
geo_total_jaccard
fs_shared_count
identity_email_match
identity_phone_match
identity_strong_match
```

`Jaccard` — доля общих значений среди всех значений двух профилей:

```text
|A ∩ B| / |A ∪ B|
```

Пример:

```text
profile A fs-values = {site_1, site_2}
profile B fs-values = {site_2, site_3}

intersection = {site_2} = 1
union = {site_1, site_2, site_3} = 3
Jaccard = 1 / 3 = 0.333
```

`identity_email_match` и `identity_phone_match` — простые флаги совпадения email/телефона.

## 9. Обучаем Decision Tree как edge policy

Decision Tree не назначает `entity_id` напрямую.

Он решает более узкую задачу:

```text
для пары profile_id A и profile_id B нужно поставить ребро или нет?
```

Если дерево считает пару хорошей, пара получает высокий score.

Дерево обучается на `valid` candidate pairs. Метка для обучения:

```text
label = 1, если у двух profile_id один entity_id
label = 0, если entity_id разные
```

В production `entity_id` не будет. Он нужен только для обучения и оценки.

Важное решение: `EXTENDED_TREE_FEATURES` удалены. Дерево использует только понятные evidence/similarity признаки, без искусственных рангов и graph-degree признаков.

## 10. Что такое threshold

Дерево выдаёт score пары. Но score сам по себе ещё не означает, что ребро точно ставится.

Мы проверяем сетку threshold:

```text
0.05, 0.10, 0.20, ..., 0.90
```

Для каждого threshold:

1. оставляем пары со score >= threshold;
2. применяем mutual top-K;
3. строим граф;
4. считаем итоговые метрики на `valid`.

Threshold выбирается на `valid` по правилу:

1. максимальный `correct_client_decisions_pct`;
2. если равно — меньший `wrong_merge_groups_pct`;
3. если равно — больший `existing_clients_found_pct`.

Потом выбранный threshold без изменений применяется к `test`.

## 11. Что такое mutual top-K

`mutual top-K` — это дополнительный фильтр рёбер после threshold.

В текущем коде:

```text
GRAPH_TOP_K = 1
```

То есть для каждого профиля оставляем только его top-1 соседа по score. Ребро между A и B остаётся только если условие взаимное:

```text
B входит в top-1 кандидатов A
и
A входит в top-1 кандидатов B
```

Пример.

Пусть после threshold есть scores:

```text
A-B = 0.91
A-C = 0.88
B-A = 0.91
B-D = 0.93
C-A = 0.88
D-B = 0.93
```

Top-1 соседи:

```text
A -> B
B -> D
C -> A
D -> B
```

Что останется:

```text
B-D останется, потому что B -> D и D -> B
A-B не останется, потому что A -> B, но B выбрал D
A-C не останется, потому что C -> A, но A выбрал B
```

Зачем это нужно: это защита от шумных профилей, которые похожи сразу на многих. Без mutual top-K один популярный/неоднозначный профиль может притянуть много слабых соседей и создать ложную большую компоненту.

Цена: можно потерять часть настоящих дублей, если у профиля несколько корректных дублей или score распределился неидеально.

## 12. Собираем граф через NetworkX

После threshold и mutual top-K оставшиеся пары становятся рёбрами графа.

```text
profile_id = вершина
разрешённая пара = ребро
```

NetworkX строит connected components.

Пример:

```text
P1 -- P2 -- P3
```

Все три профиля попадут в одну найденную группу, даже если прямого ребра `P1-P3` нет.

Это полезно, потому что один клиент может иметь цепочку совпадений:

```text
P1 похож на P2 по телефону
P2 похож на P3 по поведению
P1 и P3 напрямую слабее
```

Но это же риск: одно ошибочное ребро может соединить две разные группы.

## 13. Считаем итоговые метрики

Метрики специально сделаны простыми.

Для клиентов, у которых в разметке несколько профилей:

- `existing_clients_total` — сколько таких клиентов было;
- `existing_clients_found` — сколько нашли;
- `existing_clients_found_pct` — процент найденных;
- `existing_clients_missed` — сколько пропустили;
- `existing_clients_missed_pct` — процент пропущенных.

Для клиентов, у которых в разметке один профиль:

- `new_clients_total` — сколько таких клиентов было;
- `new_clients_correct` — сколько правильно оставили новыми;
- `new_clients_correct_pct` — процент правильно оставленных;
- `new_clients_wrongly_attached` — сколько ошибочно приклеили к кому-то;
- `new_clients_wrongly_attached_pct` — процент таких ошибок.

Для склеек графа:

- `predicted_merge_groups` — сколько непустых групп склейки построил граф;
- `wrong_merge_groups` — сколько групп смешали разные настоящие `entity_id`;
- `wrong_merge_groups_pct` — процент ошибочных групп склейки;
- `graph_edges` — сколько рёбер осталось в графе.

Итоговое summary:

- `correct_client_decisions_pct` — общий процент правильных решений по клиентам.

Его нельзя читать отдельно. Всегда надо смотреть рядом:

```text
existing_clients_found_pct
new_clients_correct_pct
wrong_merge_groups_pct
```

## 14. Что сохраняется

Основные файлы:

```text
data/processed/er_graph_tree_policy/tree_policy_selected_report.csv
data/processed/er_graph_tree_policy/tree_policy_full_threshold_report.csv
data/processed/er_graph_tree_policy/tree_policy_feature_importance.csv
data/processed/er_graph_tree_policy/tree_policy_rules.txt
```

Визуальный отчёт строится отдельным скриптом:

```text
visualize_graph_tree_policy.py
```

Он сохраняет:

```text
data/processed/er_graph_tree_policy/tree_policy_visual_report.html
```

## 15. Recommended blocking rules

Текущий `recommended_blocking_rules.csv` содержит 27 правил.

По семействам:

| block_family | rules |
|---|---:|
| behavior | 9 |
| behavior_context | 5 |
| context | 4 |
| coverage_compound | 4 |
| behavior_context_device | 3 |
| coverage_fallback | 1 |
| identity_rescue | 1 |

Эти правила дают:

- 1 084 747 строк выбранного `blocking_index`;
- 83.4% строк полного `blocking_index`;
- покрытие всех 61 927 профилей.

Расшифровка семейств:

- `context` — гео/context признаки. Хороши для покрытия, но сами по себе широкие.
- `behavior` — одиночные поведенческие fs/site-id признаки. Могут давать recall, но часто шумят.
- `behavior_context` — гео/context + behavior. Обычно чище, чем одиночный behavior.
- `behavior_context_device` — гео/context + behavior + OS/device. Более узкие правила.
- `coverage_compound` — coverage-композиты для сохранения охвата.
- `coverage_fallback` — технический fallback, чтобы профиль не выпал из процесса.
- `identity_rescue` — узкие identity-сигналы, например точный телефон.

## 16. Зачем добавлено `rule__identity_rescue__phone_digits`

Правило:

```text
rule__identity_rescue__phone_digits
```

Показатели:

```text
positive pairs captured: 78
candidate pairs: 78
pairs per positive captured: 1.0
max block size: 2
```

Оно почти не влияет на покрытие, но выглядит очень точным. Поэтому его добавили как high-precision rescue: если телефон совпал полностью, это сильный сигнал.

Это не backbone-правило, потому что телефон заполнен редко. Оно не заменяет context/behavior blocking, а только добавляет небольшой точный слой.

## Практический вывод

В текущей архитектуре главный bottleneck — не отсутствие покрытия профилей, а качество candidate pairs. Большая часть настоящих дублей находится только через слабые и шумные признаки, поэтому массовое добавление новых широких правил может ухудшить ложные склейки.

Правильная стратегия:

1. держать первый слой blocking достаточно широким, чтобы не терять кандидатов;
2. не считать слабое правило достаточным основанием для склейки;
3. усиливать второй слой: evidence, decision tree, mutual top-K, анализ ложных склеек;
4. добавлять новые правила точечно и проверять итоговые числа на test.
