# Независимое ревью Codex — 31.10.2026

База: `main@24481c10cd59e1abd2ea56f0b6043e030e4f80b5`, ветка `codex/review-1`.
Ревью начато с `5be1da5`; пришедшее во время работы обновление RESEARCH (`24481c1`, без изменений кода) прочитано и включено fast-forward.
Прочитаны правила, RULINGS, актуальный STATUS, RESEARCH, notes, код решения и эталонный harness.
Из STATUS: последний жюрийный прогон a6c3fe3 — 51,4/51,8 с, 401 MiB, десять проверок; это **чужие измерения**, здесь не повторённые.

## Дефекты по опасности

Строки относятся к исправленной ветке; существующие переключатели отключают соответствующий патч целиком.

| Риск | Место | Сценарий и исправление |
|---|---|---|
| P1, проверки 1/10 | `solution/warmup.py:446,499,626` | `join(60)` мог истечь; early-hash thread вообще не учитывался. Исключение настройки coordinator/запуска очередного helper оставляло уже запущенный native compiler жить после `_DONE`. Теперь отслеживаются early thread и успешно запущенные helpers; shutdown/drain ждут их завершения, `_DONE` публикуется после joins. CPU-тесты воспроизводят каждый путь. Зависший компилятор может задержать выход до внешнего таймаута harness. |
| P1, воспроизводимость/5–7 | `solution/warmup.py:223` | `warm_cublas` делал глобальный CUDA `randn` в фоне параллельно seed/model/Trainer setup. В Soup `train()` после ожидания warmup повторного seed нет. Заменён на `ones`; тестируется сохранность CPU/CUDA RNG. Фактическое нарушение допусков на T4 не доказано. |
| P1 при нехватке host RAM, проверка 1 | `solution/offload.py:158,213`; `solution/head_stream.py:267` | Recoverable отказ pinned allocation пробрасывался из HostStack/HeadStreamer. Prepin теперь прекращается чисто; stack сохраняет исходный GPU-тензор без повторных pin-попыток; голова выдаёт `HeadSkip` до мутации Parameter. Тесты инъецируют RuntimeError/MemoryError. Это не защита от Linux OOM-killer; resident fallback повышает GPU peak. |
| P1 перед сдачей, ручная приёмка | `README.md`; отсутствующий `SUBMISSION.md` | README остаётся starter kit: нет описания текущего решения, итоговой таблицы и реального публичного Kaggle notebook. Автотесты этого не закрывают. Заполнить перед сдачей; в рамках ревью содержание submission не выдумывалось. |
| P2, host peak | `solution/weight_stream.py:134` | После неудачного `cudaHostRegister` старый pageable buffer жил во время выделения нового pinned buffer. Добавлено освобождение до fallback: убран временный дубликат, для головы около 445 MiB. Проверка weakref доказывает порядок освобождения. |
| P2, fallback | `solution/fused_proj.py:270,351` | Fusion обходил instance `forward` у дочерних модулей и hooks у Identity dropout. Теперь отказывается от fusion. CPU-тесты и CUDA integration test. |
| P2, fallback | `solution/mlp16.py:40` | FP32 CUDA MLP вне autocast насильно получал FP16 input и падал на dtype mismatch. Возвращён original forward; CUDA-тест сравнивает выход и input gradient побитово. |
| P2, fallback/подготовка | `solution/trainer_build.py:94,135` | Memo кэшировал stateful fake tokenizer и shallow-копию nested messages; добавление vocabulary не инвалидировало ids. Memo ограничен штатным текстовым tokenizer/template, учитывает len/template/specials; instance override и nested content идут в original. Замена backend normalizer без смены signature остаётся вне поддержанного memo-контракта. |

**Побитовость:** арифметика штатного forward/backward новыми правками не меняется. Исправление RNG намеренно исключает посторонний случайный draw: при старой гонке dropout/adapter могли отличаться, поэтому обещать идентичный adapter прежнему main нельзя. Сам main уже допускает отличия от Soup из-за округлений объединённых GEMM и недетерминированного attention dQ. Численные допуски проверок 5–7 не расширялись.

## Проверенные границы и оставшийся риск

По коду и локальным ordering tests в fixed single-compute-stream пути проверен порядок allocate → side-copy → use/free: loader передаёт событие аллокации, ring ждёт free/ready, offload удерживает D2H source и защищает host reuse после H2D. Type ring освобождает NF4 slot после enqueue dequant; GEMM читает уже другой half buffer. Доказанной гонки здесь не найдено; произвольные параллельные model forwards этим выводом не покрываются.

Сумма длин четырёх строк ограничена 2048 токенами. План attention учитывает точный max sequence, alignment tail — отдельная последовательность, метрики раздельно используют attention_mask и shifted labels. Полностью отрезанный assistant должен отвергаться Soup до train; заменять это нулевым loss нельзя. Граничные truncation tests сравнивают FAST_ROWS с оригиналом.

Проверку 9 **нельзя гарантировать** по одному анализу. Сохраняются harness allocator и внешние замеры; никакие sleep, reset_peak или обход watcher не добавлены. Особо перепроверить setup/fallback после pin failure и длинный батч: краткий reserved peak может пережить освобождение физической памяти и быть пропущен nvidia-smi. Нужны два полных T4 прогона на допустимых длинных строках. Stack limit не менялся: [CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__CTX.html) допускает округление и автоматическое увеличение stack под kernel; это не доказательство приёмки настройки организаторами.

## Где искать время

Все оценки ниже **неизмеренные**, не складываются: CPU работа частично перекрыта GPU. Уже известные предложения RESEARCH отделены от новых идей.

| Приоритет | Изменение | Потенциал, мс/шаг | Риск/статус |
|---|---|---:|---|
| 1 | Один autograd.Function на decoder block с готовым LayerPlan | 10–30 стены, 40–100 CPU | Средний/высокий; предложение свежего Astra/RESEARCH, не новая находка этого ревью. Сначала предтест enqueue, затем переписывание backward. |
| 2 | CPU batch plan + prefetch следующего батча до scaler/clip; объединить чтение metrics/rewind с неизбежной точкой host sync | 10–20 | Средний; уже RESEARCH. Не убирать две синхронизации harness. |
| 3 | Закэшировать списки trainable parameters/модулей для Trainer FLOPs, clip, zero_grad, train; инвалидировать при изменении модели | 8–15 | Средний; уже night-boundary. Проверить его свежую ветку, не дублировать реализацию. |
| 4 | Сериализованный writer thread SQLite с прежними SQL, commit и PRAGMA; drain и ошибки до возврата train | 1–4 | Средний; новый способ закрыть известный RESEARCH fsync. Нельзя передавать живой sqlite connection другому потоку; read-after-write требует fence. |
| 5 | `SOUP_SOLUTION_STREAM_TIMING=0`: убрать diagnostic spans головы/attention-ring | 1–4 | Реализовано, opt-in; default `1`. Около 368 лишних Event.record/шаг плюс query/elapsed bookkeeping; ordering events и harness timing сохранены. |
| 6 | Кэшировать immutable часть `_attention_plan`/`_mlp_plan`, проверять signature/invalidation вместо повторных Params4bit attribute reads | 0,5–2 | Средний; уже известная идея. Не замораживать train/eval, hooks, adapters, dtype/device. |

Старт: повторно «сэкономить» 1,1 с dynamo/0,9 с trl/0,4 с transformers.data простым переносом import нельзя: эти зависимости нужны позже. Scope import diet уже имеется; без заглушек надёжного нового удаления этих импортов не найдено. CLI gap включает настоящий Soup setup/discovery/DB, а не бесплатную паузу. Не включать EARLY_PROBES: он запускает nvidia-smi subprocess из фона. Exit нельзя ускорять ценой незавершённой компиляции или `os._exit`; безопасные исправления выше ориентированы на exit 0.

## Дополнительные идеи, отсутствующие в solution/RESEARCH

Это направления экспериментов, не результаты и не готовые сторонние kernels.

| Механизм | Оценка выигрыша | Риск/проверка |
|---|---|---|
| Lossless encode/decode **активаций gate\|up** при offload: exponent-delta + escape, включая все FP16 bit patterns | 0–10 мс/шаг при сокращении 10–20% трафика | Средний/высокий; encode/decode и power могут съесть всё. Свои Triton kernels, bitwise round-trip, события, peak. В отличие от обсуждаемого сжатия головы, данные меняются каждый шаг. Принцип lossless exponent coding: [DFloat11](https://arxiv.org/abs/2504.11651); готовую библиотеку не импортировать. |
| Горизонтально объединить независимые RoPE-backward, compact-V copy и NF4 decode одним grid с program-id dispatch | 1,4–2,8 мс CPU/шаг; GPU выигрыш может быть нулевым | Средний; сохранить арифметику и ring release; дополнительная компиляция может не окупиться за 64 шага. |
| Adopt/register уже готовый CPU FP16 embedding storage вместо нового buffer + fill + copy | 0,05–0,2 с старта и до 445 MiB временной host RAM | Средний; нужен владелец source storage, исключение перекрывающихся регистраций и original fallback. |
| Write-combined собственные pinned staging/NF4 buffers, которые CPU только пишет | 0–3 мс/шаг, возможна регрессия при уже 12 ГБ/с | Средний; не трогать GPU allocator и не применять к head.host с CPU gather. [NVIDIA](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#write-combining-memory) предупреждает о дорогом CPU чтении WC memory. |

## Валидация

WSL, Python 3.13.18, torch 2.11.0+cu128, Triton 3.6.0 и версии CLAUDE; локальная карта RTX 3050 Ti 4 GiB. Локальные времена не сравниваются с Kaggle. Токены найдены по указанному пользователем пути и в Git не добавлялись. На T4 запущен [протокол жюри](https://www.kaggle.com/code/peagverr/soup-codex-r1-jury), `611cade`, другие публичные строки 10000–10255/50000–50255, seed 31, два повтора.

| Проверка | Локальный результат |
|---|---|
| `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` | **79 passed, exit 0**, с включённым warmup |
| `CUDA_VISIBLE_DEVICES='' python -m pytest solution -q` | **73 passed, 296 skipped** |
| Stream/warmup safety, forward guards, trainer_build, mlp16, fused_proj | **68 passed**, одна старая launch-count assertion |
| Длинные unpad/type-ring и CUDA RNG | **8 passed**: 1999/2048 токенов, повторное использование событий без timing, RNG CPU/CUDA |
| Streamed head `[hidden-2048]` | **1 passed**; test reference ограничен чанками и не удерживает лишние entropy graphs; runtime/допуски прежние |
| Baseline-контроль упавшей assertion | На чистом `5be1da5` то же: MLP forward 9→10, backward 11→12 launches; это не регрессия ветки. Тест/условия не ослаблялись. |

Static `source_kernels`: 15 своих kernels, запрещённых импортов/непарсящихся файлов/похожих переписываний примера нет; два example kernels корректно исключены. `compileall` и `git diff --check` проходят. `harness/`, `tests/`, `task/` не изменены.

Коммиты: `49c639b` — RNG/threads; `8c8f043` — fallback; `611cade` — pinned failures, timing switch и длинные тесты; последующая правка затрагивает только память тестового эталона головы. Итог T4 будет добавлен после завершения задания.
