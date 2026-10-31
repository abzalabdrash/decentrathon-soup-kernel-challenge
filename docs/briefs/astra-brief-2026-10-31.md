# Бриф для независимого эксперта: Soup Kernel Challenge, где спрятаны ещё 6–8 секунд

Ты независимый эксперт по производительности GPU-обучения (PyTorch, CUDA, Triton, cuBLAS, PCIe, память). Команда три дня оптимизировала фиксированный прогон QLoRA на Kaggle T4 с помощью агентов одной модели (Claude). У нас есть подозрение на слепые зоны: все агенты думают одинаково. Твоя задача — прочитать всё ниже (правила, разъяснения организаторов, архитектура решения, измерения, ресерч, что отвергнуто) и ответить на вопросы в конце. Думай долго. Нам нужны не общие советы, а конкретные механизмы с оценкой выигрыша в миллисекундах на шаг или секундах на прогон и с объяснением, почему это побитово (или в допуске) совпадёт с базой.

## 1. Задача и формула баллов

- Decentrathon 6.0, трек «Soup Kernel Challenge». Фиксированная команда `soup train` (пакет soup-cli 0.77.0 поверх transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0, torch 2.11, triton 3.6.0, Python 3.13): Qwen2.5-1.5B-Instruct в NF4 (bitsandbytes, двойное квантование, compute fp16), LoRA r=16, alpha 32, dropout 0.05 на q_proj и v_proj, batch 4, max_length 512, 64 шага, AdamW (foreach, не fused), fp16 autocast + GradScaler, logging каждый шаг, без eval внутри прогона (eval loss считается отдельно и не входит во время).
- Харнесс (`harness/run_task.py`, менять нельзя) запускает в одной сессии Kaggle T4: базовый Soup, затем Soup с нашим пакетом `solution/` (харнесс импортирует его и зовёт `apply()` в том же процессе до построения модели; `apply()` патчит transformers/peft/bitsandbytes/trl). Меряет стену всего процесса (от старта до конца процесса, снаружи) и пик nvidia-smi (опрос снаружи). Жюри гоняет базу и решение по два раза на скрытых строках и с другим seed.
- Баллы: 20 за все 10 проверок (иначе 0 за всё), 35·ln(s)/ln(s_лучшей команды) за скорость (s = время базы / наше), 30·ln(m)/ln(m_лучшей) за память (m = пик базы / наш), 15 README. Разница меньше 5 % считается нулём; при равенстве выше тот, у кого память лучше.
- 10 проверок: (1) все прогоны завершились; (2) нет запрещённых библиотек (unsloth, liger, xformers, flash-attn, torchao и др.); (3) собственные `@triton.jit`-ядра из файлов `solution/` запускаются в обучении; (4) они занимают ≥ 5 % времени шагов с шага 3; (5) те же выходы Soup (число шагов, формы адаптера, настройки тренера, токены на шаг, entropy/accuracy/rewind в пределах 3 %); (6) кривая loss: первый шаг в пределах 1 %, среднее по шагам 3 %; (7) eval loss не хуже базы более чем на 0.5 %; (8) не хуже базы по времени и памяти на 10 %; (9) чистое измерение: одна T4, пик nvidia-smi ≥ 0.97 × пика reserved torch (аллокатор харнесса `expandable_segments`); (10) 79 регрессионных тестов Soup с `SOUP_APPLY_SOLUTION=1` проходят (они зовут патченый код с фальшивыми модулями, поэтому каждый патч должен откатываться на оригинал при неожиданном входе).

## 2. Железо и среда

- Kaggle T4: sm_75, 40 SM, 16 ГБ GDDR6, лимит мощности 70 Вт. Под GEMM-нагрузкой частота падает до 600–900 МГц (максимум 1 590), реально достижимо 28–34 TFLOPS fp16 на наших формах (паспорт 65). Практическая полоса памяти ~220–240 ГБ/с (паспорт 320). PCIe pinned: ~12 ГБ/с в каждую сторону. Карта «горячая» после 87 с прогона базы: наши шаги на 10–15 % медленнее, чем в холодной сессии; порядок прогонов жюри (база, решение, решение, база) мы не контролируем.
- Triton 3.6 на sm_75 не использует тензорные ядра в `tl.dot`, поэтому все GEMM остаются в cuBLAS; свои ядра только для поэлементных операций, редукций, распаковки NF4, loss. int8 IMMA через `torch._int_mm` на T4 работает (CUTLASS i8816, ×1.6–1.9 на наших формах backward), но организаторам вопрос про int8 ещё не задан.
- CPU Kaggle медленный: наш шаг требует ~0.30 с CPU-времени (шаги короче 600 токенов идут 0.30–0.32 с независимо от числа токенов, это ~20 % шагов), ~1 800 запусков ядер за шаг.
- Данные: 256 строк alpaca-формата, 4 строки на шаг, после удаления паддинга в шаге 354–1 427 токенов (у жюри до ~1 970), в среднем ~830.

## 3. Что делает наше решение (все механизмы включены по умолчанию, каждый за переключателем `SOUP_SOLUTION_*`)

Слой (28 слоёв Qwen2, hidden 1536, intermediate 8960, 12 голов q и 2 головы k/v):
- Паддинг выброшен: батч упакован в одну последовательность, дополнен «хвостом выравнивания» до кратного 16 токенов (фиктивные позиции не влияют ни на что реальное, разрешено организаторами), attention одним вызовом `aten::_efficient_attention_forward/backward` с cu_seqlens (k/v расширяются до 12 голов внутри Function и не сохраняются).
- Распаковка NF4 своим Triton-ядром в общий fp16-буфер (q|k|v одной распаковкой), GEMM cuBLAS; q+k+v одним GEMM, LoRA через addmm, gate+up одним GEMM; RMSNorm и residual-add + RMSNorm слиты в свои ядра; RoPE своим ядром; x16 для backward восстанавливается из fp32-входа и rstd (не хранится); сумма градиентов адаптера по двум маскам dropout одним ядром.
- Память: пересчёт блока attention в backward (q,k,v, выход, lse, выход o_proj) побитово из того, что и так хранится (+16…35 мс/шаг); «split-sum» в пост-attention норме (хранит слагаемые, пересобирает сумму той же fp32-операцией); in-place backward SwiGLU; оффлоад gate|up (fp16, 36 КБ/токен/слой) и fp32-входов слоёв на pinned-хост по событиям на боковом потоке с ранним стартом копии сразу после GEMM и префетчем назад в один тензор; NF4-веса всех слоёв лежат в pinned-памяти хоста и стримятся на GPU кольцом (attention-группа: 2 слота по слоям; gate/up/down по одному слоту на тип матрицы), 1.25 ГБ H2D за шаг, ожидания 0.3 мс/шаг; эмбеддинг/lm_head (tied, 151 936 × 1 536 fp16, 445 МБ) живёт на хосте, выборка строк эмбеддинга на хосте, голова считается за один проход по словарю чанками по 4 096 строк через кольцо из 2 слотов (онлайн-softmax, loss, entropy, accuracy и dh в том же проходе, своими ядрами + cuBLAS), префетч первых чанков у финальной нормы; лимит стека CUDA 16 Б на поток (`cudaDeviceSetLimit`, −40 MiB вне аллокатора); освобождение рабочих областей cuBLAS в начале обучения.
- Старт: прогрев всех Triton-ядер в потоке процесса во время загрузки модели (лаунчеры с -O0); диета импортов (trl не делает второй скан importlib.metadata, не импортируются sklearn/wandb/Inductor-обёртки bitsandbytes); загрузка чекпоинта через pinned-буферы с кастом bf16→fp16 на GPU (побитово); дешёвый детерминированный отпечаток datasets; подготовка строк Soup без отбрасываемого рендера; слои и эмбеддинг уходят на хост прямо при загрузке (пик setup 64 MiB).
- Тренер: fused head + loss без (B,T,V) логитов, метрики TRL и rewind-лог Soup из fused head (`trainer_patch.py`).

## 4. Текущие числа (main a6c3fe3, холодная сессия Kaggle)

- База 87 с / 14 345 MiB. Мы: ~42.7 с / 401 MiB (s ≈ 2.04, m ≈ 35.8), все 10 проверок на публичных строках и на 8 наборах скрытых строк/seed. Главный соперник заявляет 29 с / 937 MiB (не проверено; второй заявляет 19 с / 613 MiB с eval на 1.3 % лучше базы, что ниже физического пола и помечено харнессом).
- Разложение наших 42.7 с: пролог харнесса 5.1 (import torch/transformers, CUDA-контекст; не наш) | apply() 3.5 (импорты 2.6: bitsandbytes 1.1 через torch._dynamo/inductor, trl 0.9, transformers.data 0.4; наш код 0.3) | щель Soup до setup 0.6 | загрузка модели 3.5 (токенизатор 0.9, цикл тензоров 2.24, остальное) | сборка тренера 1.3 | цикл 26.3 (64 шага, медиана 0.41 с; первый шаг 0.47) | завершение процесса 0.9 после нашей метки exit (освобождение ~4.3 ГБ pinned-памяти, контекст, финализация Python).
- GPU-время шага (холодная карта, конфигурация без пересчёта attention): ~0.32 с из 0.36: GEMM cuBLAS 150–185 мс (5.1 TFLOP на шаг при 830 токенах; forward 2·N·P, backward только градиент по входу 2·N·P, голова 2·V·d·N туда и обратно), распаковка NF4 32 мс (6.6 ГБ записи/чтения за шаг, 88 % практической полосы), attention 28 мс (fwd 5, bwd 23), SwiGLU примера 17 мс (на полосе), голова ~38 мс (упирается в PCIe: ждёт копии 7–20 мс/шаг), поэлементные ~25 мс, конец шага ~10. Простой GPU 12 %: ~26 мс внутри слоёв (CPU не успевает) и 26–40 мс на границе шага (синхронизации `.item()`, found_inf скейлера, четыре прохода по параметрам модели в Trainer, SQLite-коммит трекера Soup, DataLoader).
- Точка перелома против 29 с / 937 MiB: нам нужно ≤ 36.3 с при 401 MiB (или ≤ 38.3 с при 300 MiB). Память почти исчерпана: контекст CUDA 105 MiB + cuBLAS 28 вне аллокатора, статика 120 MiB (кольцо весов 26, LoRA+Adam 26, буфер распаковки 26, рабочие области cuBLAS 16, кольцо головы 24), фрагментация страниц по 20 MiB ~30–40; динамика худшего шага ~115 MiB.

## 5. Что уже измерено и отвергнуто (не предлагай повторно без нового аргумента)

Перебор алгоритмов cuBLAS/cuBLASLt (split-K, Lt для o/down) — в шуме (≤ 4 мс/шаг); распаковка следующего веса на боковом потоке — медленнее (+29 мс: конкуренция за SM и мощность); два потока для GEMM — не помогают (лимит мощности); чанкование длинных шагов — обрыва нет; оффлоад тензоров attention вместо пересчёта — на 18 мс/шаг медленнее при той же памяти (PCIe занята); моменты AdamW на хосте — 0 MiB в nvidia-smi из-за страниц аллокатора и +1 с; `empty_cache` по шагам +0.9 с; пауза GC — в шуме; `CUBLAS_WORKSPACE_CONFIG` — regex на каждом GEMM (+33…48 мс CPU/шаг); num_splits_key attention — в шуме; fused/capturable AdamW — другая арифметика; CUDA graphs и torch.compile — запрещены правилами; рекомпьют gate|up вместо оффлоада +50–90 мс/шаг; процесс-помощник компиляции — запрещён (только поток). Подробности и источники в разделах RESEARCH ниже.

## 6. Что сейчас в работе (четыре параллельных агента)

C1 — CPU-диета шага (планы на слой без чтений атрибутов через peft/bnb `__torch_function__`, меньше событий и запусков). B1 — граница шага без синхронизаций (счёт меченых токенов и план unpad с CPU-копии батча, метрики одной синхронизацией в точке лога, следующий батч в тени backward, проходы Trainer по модели один раз). S1 — старт (импорты, щель Soup, токенизатор, загрузчик, сборка тренера; цель первый шаг ≤ 7.0 с после apply вместо 9.2). E1 — завершение процесса (0.9 с) и lossless-сжатие fp16 весов головы для PCIe (DFloat11-подобное, своё decode-ядро). Плюс готовый, но выключенный int8 backward для GEMM градиента по входу gate|up и down (−0.55 с, численно чисто; ждём ответа организаторов).

## 7. Правила, которые нельзя нарушить

Не менять `harness/`, `tests/`, `task/`; не менять задачу (модель, данные, порядок батчей, число шагов, LoRA, оптимизатор, переключатели Soup, аллокатор, гиперпараметры); запрещены готовые библиотеки ядер, CUDA graphs для своих ядер, torch.compile как «свои ядра», подмена адаптера, кэши компиляции между прогонами, `sys.modules`-заглушки, `os._exit`, фоновые процессы (потоки внутри процесса можно), общая маска для всех метрик; разрешены fp16-хранение замороженных весов при побитово тех же значениях в операциях, собственный оффлоад/рекомпьют, удаление паддинга и хвост выравнивания, патчи функций, чтобы не брать ленивые импорты, кэш `packages_distributions()` внутри прогона. Эталон — побитовое совпадение с базой или допуски проверок 5–7.

## 8. Вопросы к тебе

1. **Слепые зоны.** Где в этой системе могут скрываться ещё 5–8 секунд на прогон, которых мы не видим? Пройди по каждой фазе (пролог, apply, щель Soup, загрузка, сборка тренера, первый шаг, шаг, граница шага, выход) и назови механизмы, которых нет ни в разделе 3, ни в разделе 5.
2. **Шаг.** При GEMM у стены мощности и распаковке у стены полосы, что ещё можно сделать с 0.32 с GPU-времени и 0.30 с CPU-времени на шаг? Оцени, в частности: уменьшение энергии на операцию (порядок ядер, раскладки, частота), слияния, которые мы пропустили, способы сократить 1 800 запусков без графов и compile, и как сделать CPU-путь слоя в 2 раза дешевле в eager PyTorch.
3. **Голова.** 445 МБ весов головы за шаг по PCIe. Есть ли способ лучше lossless-сжатия: иной порядок вычисления, частичная резидентность с учётом страниц аллокатора по 20 MiB, перекрытие с backward слоёв, использование того, что dh нужен только для меченых токенов?
4. **int8.** Если организаторы разрешат int8 только для GEMM градиента по входу в backward: как сделать это максимально безопасно численно (SwitchBack, per-token масштабы, Адамар) и где ещё int8 допустим без риска для допусков 1 %/3 %/0.5 %?
5. **Риски у жюри.** Что в нашей конструкции (оффлоад по событиям, стриминг весов, лимит стека, pinned-память 4 ГБ, прогрев в потоке, фрагментация и проверка 9 «nvidia-smi ≥ 0.97 reserved») может сломаться на скрытых строках до 2 000 токенов, другом seed, более горячей карте или другой последовательности прогонов? Что бы ты проверил до сдачи?
6. **Пол.** Дай свою оценку физического пола стены на этой задаче на T4 с этой математикой и объясни, из чего он складывается. Реалистично ли 29 с / 937 MiB у соперника и за счёт чего это могло бы быть достигнуто честно?
7. **Ранжирование.** Выдай топ-10 конкретных действий с ожидаемым выигрышем (мс/шаг или с/прогон), риском для проверок и объёмом работы в часах для сильного инженера с агентами.

Формат ответа: по пунктам, с числами и формулами, где возможно — со ссылками на источники. Если какая-то наша цифра кажется тебе неправдоподобной, скажи это прямо и объясни, что перемерить.

---

Далее приложены как есть: правила и проверки из README организаторов, разъяснения организаторов (`docs/RULINGS.md`), ресерч (`docs/RESEARCH.md`), состояние работ и измерения (`docs/STATUS.md`, выбранные разделы), точка входа решения (`solution/__init__.py`) и докстринги модулей решения.



---

# Приложение A. README организаторов: что меряется, правила, проверки, баллы

## What is measured

- **Time**: the wall time of the whole `soup train` command, from start to saved adapter, timed
  from outside the process. Triton compilation and autotuning count: every run starts with empty
  compiler caches. Base and solution run in the same session; the score uses the ratio.
- **Memory**: peak GPU memory, sampled by `nvidia-smi` from outside the process every 100 ms and
  summed over all GPUs. Both sides run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  so the peak follows the tensors that are alive rather than allocator fragmentation.
- **Quality**: `harness/eval_adapter.py`, the mean loss on the answer tokens of held-out rows. It
  loads your adapter with plain transformers + peft, without your code.
- **Your kernels' share**: inside the same measured runs, every launch of a `@triton.jit` kernel
  defined in `solution/` is timed with CUDA events from step 3 on, against the training time of
  those steps.

The jury runs the same scripts on hidden rows of the same dataset with another seed, base and
solution twice each in the order base, solution, solution, base, in a fresh copy of this kit with
only your `solution/`.

## Rules

Allowed: your own Triton kernels, autograd functions and patches; recomputation or offloading in
your own code (its time counts); `torch.compile` (its kernels are not yours); packages torch,
triton, transformers, peft, trl, bitsandbytes and `soup-cli==0.77.0`; ideas from other kernels
when you name the source.

Not allowed:

- ready-made kernel libraries: unsloth, liger-kernel, xformers, flash-attn, torchao and the rest
  of the list in `harness/common.py`, and installing packages during the run;
- copying other projects' kernel code as your own; kernels whose output training does not use
  (busy work) do not count;
- changing the task: model, data, batches and their order, number of steps, LoRA settings
  (dropout included), optimizer and schedule, Soup's switches (`backend: unsloth`,
  `gradient_checkpointing`,
  `activation_offloading`, `fast_lora`, `use_liger`, `use_cut_ce`, `packing`, `padding_free` and
  the like), and Soup's outputs. Removing padding inside a batch in your own code is fine; mixing
  rows across batches is not;
- getting around the measurement: another GPU, background processes, different behaviour when
  measured or tested, other allocator settings, replacing the saved adapter, prebuilt compiled
  caches;
- reading the hidden rows or any files outside your repository and the task data;
- changing `harness/`, `tests/` or `task/` (the jury uses its own copy anyway).

## Checks

`harness/check_rules.py` prints one line per check and writes `runs/checks.json`. Every check
must pass on the jury's run, and the jury reads your code: its review decides in disputes.

| # | Check | Limit |
|---|---|---|
| 1 | every base and solution run finished | all runs exit 0 |
| 2 | no ready-made kernel libraries | not imported in `solution/`; not loaded in the runs (torchao is checked in `solution/` only, peft may load it by itself); no packages installed during the run |
| 3 | your `@triton.jit` kernels in `solution/` run in training | launched in every solution run; the example's kernels do not count |
| 4 | your kernels together take enough of the training time | at least 5%, CUDA events, steps 3+, lowest run |
| 5 | same task and Soup outputs | steps, adapter config, adapter tensor names and shapes, trainer settings, logged metric names, tokens per step, rewind rows and token counts equal; entropy, accuracy and rewind row losses within 3% |
| 6 | loss curve follows the base | first step within 1%, mean of all steps within 3% |
| 7 | quality | eval loss at most 0.5% above the base |
| 8 | not worse than the base | time and memory at most 10% above the base |
| 9 | clean measurement | a Tesla T4, GPUs idle at start (under 1,000 MiB), one GPU, harness allocator, nvidia-smi peak at least 97% of PyTorch's reserved peak |
| 10 | Soup regression tests with your patches | 79 tests pass |

When a check fails, its line says why. Logs of every run are in `runs/<run>/train.log` and
`runs/eval_*.log`; per-step metrics in `runs/<run>/adapter/checkpoint-*/trainer_state.json`.

While iterating: `python harness/run_task.py --only solution --out runs_try` trains only your
solution (checks 1 and 5–8 need the base, so they fail there); `--profile` adds one profiled run with
the GPU time of every CUDA kernel in `runs/solution_profile/profile.json`.

## Scoring (100 points)

| Criterion | Points |
|---|---|
| Works and correct: every check passes and the code review agrees | 20 |
| Speed: `35 * ln(s) / ln(best s)`, s = base time / your time | 35 |
| Memory: `30 * ln(m) / ln(best m)`, m = base peak / your peak | 30 |
| README and reproducibility | 15 |

A gain under 5% counts as a tie (ratio 1.0). "Best" is the best admitted team. A team that fails a
check, or whose code or README the jury does not accept, is not admitted and gets no speed or
memory points. `harness/score.py` prints your ratios.

The jury gives the README points in three parts, up to 5 each: what you did, how to run it,
results and limits. If two teams have the same total, the one with the better memory result ranks
higher.



---

# Приложение B. docs/RULINGS.md (разъяснения организаторов)

# Разъяснения организаторов (чат хакатона, 30 октября) и как им соответствует решение

Выжимка из ответов организатора ({ nurikw }) в чате трека; проверено по коду `solution/`.

| Разъяснение | Статус у нас |
|---|---|
| Полные логиты хранить не нужно; достаточно loss с градиентом, `entropy`, `accuracy`, per-row loss для rewind; пересчёт логитов чанками разрешён; ядра этих метрик идут в зачёт 5 % | `fused_ce.py`, `head_stream.py`: голова за один проход по словарю, логитов целиком нет |
| Прямая запись токенного loss в `trainer._soup_chunked_step` без `stream_loss_chunk` | `trainer_patch.py` делает именно это |
| Маски метрик: `entropy` по `attention_mask` без сдвига; `loss`, `accuracy`, rewind по сдвинутым `labels` с −100; одна общая маска не допускается | `_log_trl_metrics`: entropy по attention_mask, accuracy по labelled; хвост выравнивания исключён из обоих |
| Замороженные веса (`embed_tokens`, `lm_head`) можно хранить в fp16, если значения и dtype, подаваемые в операции, бит-в-бит как у базы | эмбеддинг: fp16 на хосте, выход каста в fp32 (как fp32-gather базы); lm_head: fp16 как под autocast у базы |
| Патчить bitsandbytes под свою распаковку можно при сохранении NF4-весов | `nf4.py` |
| Предрасчёт вспомогательных тензоров NF4 на весь запуск разрешён (память в пике) | не используем (распаковка на лету, память меньше) |
| Удаление паддинга внутри батча своим кодом разрешено, смешивать строки батчей нельзя | `unpad.py`; хвост выравнивания разрешён (ответ 3) при полной изоляции от реальных токенов |
| Фоновая компиляция Triton в `threading.Thread` внутри процесса `soup train` с пустым кэшем разрешена | потоковый путь остаётся единственным: процесс-помощник запрещён (ответ 1) |
| Кэш `importlib.metadata.packages_distributions()` внутри запуска разрешён, возвращать копию, описать в README | `startup.py`: копия карты transformers для trl |
| Случайный `new_fingerprint` в `Dataset.map` разрешён, если не трогает глобальный RNG | `startup.py`: не случайный, а детерминированный редьюсер токенизатора (C-pickler), RNG не трогается |
| Не загружать ненужные библиотеки (sklearn) разрешено, патчем функции, которая делает ленивый импорт | `startup.py`: подмена `is_*_available` на время импорта |
| **Запрещено**: заглушки через `sys.modules` | не используем (`harness/_train.py` сам ставит `sys.modules["unsloth"] = None`, это харнесс) |
| **Запрещено**: `os._exit` после сохранения адаптера | не используем; наоборот, при выходе ждём компиляцию |
| **Запрещено**: кэши между запусками, предвычисленные тензоры между запусками | не используем |
| **Запрещено**: менять аллокатор PyTorch, гиперпараметры, батчи, проверочные скрипты | не трогаем |
| README обязан описывать все оптимизации, включая CPU-патчи старта; ноутбук Run All без правок | сделать при финализации |
| 5 % считаются с 3-го шага; подготовка до него в зачёт не идёт | своих ядер 13–16 % |
| «4 ГБ и 30 с было бы супер» (ориентир организатора) | у нас 1,35 ГБ и 45 с (s = 2,02, m = 10,6) на 30.10 22:25 |
| Совет: искать статьи и архитектурные ходы вне шаблона агентов | см. список отличий ниже |

## Ответы организаторов на наши четыре вопроса (30 октября, вечер)

1. Процесс-помощник для компиляции Triton: **нельзя**, это фоновый процесс (запрет из раздела «обход замера»); компилировать в потоке того же процесса или синхронно; пустой кэш и учёт времени запрет не снимают. Действие: прогрев возвращается в поток внутри процесса, код помощника убрать до сдачи.
2. Эмбеддинг и lm_head в pinned-памяти хоста со стримингом головы каждый шаг: **можно** как собственный CPU-оффлоад; копии и синхронизация в замере, значения, dtype и обязательные выходы сохраняются.
3. Хвост выравнивания до кратного 16: **можно** как внутреннее выравнивание вычислений; фиктивные позиции не должны влиять на реальные токены, attention, loss, метрики, rewind и учитываемое число токенов (тест на это добавляет T12).
4. `TORCHDYNAMO_DISABLE=1` на время импорта bitsandbytes: **можно**, если флаг действует только во время этого импорта, восстанавливается и не меняет путь обучения, харнесс и проверки.

## Вопросы организаторам (черновик, отправлены)

1. Можно ли компилировать свои Triton-ядра в процессе-помощнике, который `apply()` запускает в начале замеряемого прогона (кэш `TRITON_CACHE_DIR` пустой на старте, заполняется только нашим помощником в течение прогона, время входит в замер)? Или поток внутри процесса обязателен?
2. Разрешено ли держать замороженные `embed_tokens`/`lm_head` в pinned-памяти хоста весь прогон и стримить матрицу головы через GPU чанками каждый шаг (наш код, время в замере), при тех же значениях и dtype в операциях?
3. Паддинг внутри батча убираем; можно ли дополнять упакованную последовательность несколькими фиктивными токенами до кратного 16 (своя последовательность в attention, без loss, не влияют на реальные токены), ради выровненных GEMM?
4. Можно ли на время импорта bitsandbytes выставлять `TORCHDYNAMO_DISABLE=1` (torch-овский флаг, возвращается сразу после), чтобы bitsandbytes не строил `torch.compile`-обёртки своего CPU-бэкенда, которые прогон не вызывает?

## Чем решение отличается от шаблонного

Varlen-attention одним вызовом на слой с хвостом выравнивания (без цикла по строкам и без repeat_kv), голова за один проход по словарю с онлайн-перемасштабированием накопителя градиента (матрица головы читается один раз за шаг, никогда не резидентна на GPU), кольцо из двух слоёв NF4-весов, стримимых с хоста, оффлоад gate|up на событиях без record_stream, восстановление входов норм в backward своим ядром вместо хранения, компиляция ядер вне главного потока, диагностика пика по снимку аллокатора.



---

# Приложение C. docs/RESEARCH.md (ресерч: измерения, статьи, отвергнутое)

# Ресерч (30 октября, ночь): что говорят статьи и трассы про наш шаг на T4

Агент на Opus прочитал целиком 13 статей, исходники BLAS-пути PyTorch v2.11.0 и документацию cuBLAS, и заново разобрал нашу трассу T4 (диаг 21:07, шаги с N = 1 376 и 672 токенов). [V] проверено в источнике или в трассе, [I] вывод.

## Матмулы

- [V] T4 в реальности не даёт 65 TFLOPS: Jia et al. (1903.07486) намерили максимум 41,6 fp16 TFLOPS в cuBLAS при 1 590 МГц; частота падает до ~1 340 МГц на больших GEMM, пропускная способность до 17–28 TF, когда массивы превышают 4 МиБ (размер L2). Marlin видит то же на A10. Это совпадает с бенчмарком T19 (SM 527–1 397 МГц под нагрузкой).
- [V] По трассе наши GEMM идут на 27–34 TF при N = 672 и 20–30 при N = 1 376; cuBLAS уже сам делает split-K (grid z = 2–3) для выходов шириной 1 536 и выбирает одноволновые ядра на 36 CTA при малом N. Гипотеза волновой квантизации не подтвердилась (T19 тоже).
- [I] Реальные потери: длинные шаги на ~17 % медленнее на FLOP (переполнение L2: x16 4,2 МБ при N = 1 376, или мощность), ~1,5 мс на слой; маленькие qkv и o GEMM на 20–28 TF (1,65–2,4 волны).
- [V] PyTorch идёт в cuBLASLt при addmm с 1-D bias, beta = 1, row-major (наш qkv forward, отсюда «relu» в имени ядра); `preferred_blas_library("cublaslt")` переводит всё; Lt берёт один алгоритм по эвристике, workspace `CUBLASLT_WORKSPACE_SIZE` (по умолчанию 1 МиБ); legacy cuBLAS держит 8,1 МиБ workspace на поток. TunableOp на CUDA в 2.11 даёт только Default. Stream-K есть только в CUTLASS, не в cuBLAS.
- [I] Раскладка операндов (tn/nt/tt/nn) не важна: все в одной полосе 27–34 TF. Два потока большим GEMM не помогают (цепочка зависимостей); независимы только распаковка и LoRA-градиенты весов.

## Перестройка GEMM (без изменения математики)

- Свернуть LoRA-B в qkv GEMM конкатенацией по K: `[x | s·h_q | s·h_v] @ [[Wqkv]; [B_q]; [B_v]]ᵀ` (scaling = 2, умножение точное; B-слагаемое аккумулируется в fp32, различие на уровне округления): убирает два `addmm_` (41 + 12 мкс на слой при N = 1 376), ~1 мс на шаг. Обратное зеркало не окупается.
- Слияний через границу attention/MLP нет: строгая цепочка; при замороженных весах нет больших градиентов весов, которые можно было бы батчить.

## PCIe (FlexGen 2303.06865, ZeRO-Offload 2101.06840)

- [V] FlexGen: время слоя = max(H2D, D2H, compute) при идеальном перекрытии; без перекрытия −20 %. ZeRO-Offload: минимизировать трафик, активации лучше пересчитывать, чем возить.
- [I] Наш бюджет при 832 токенах: H2D веса 1,25 ГБ (104 мс) + голова 0,445 ГБ (37 мс) + возврат gate|up 0,83 ГБ (70 мс); D2H gate|up 0,83 ГБ (70 мс). Forward: 52 мс копий против ~100 мс счёта (запас); голова: 37 против 33 (чуть PCIe-bound); backward: 122 против ~125 мс (на грани). Совпадает с измеренными ожиданиями 5,5 + 8,1 + 8,5 мс.
- Два слота кольца достаточно, пока копия слоя (2,0 мс) короче его счёта (3,3–4,5 мс); замер слотов 3/4 в шуме это подтверждает.
- Балансировка backward по FlexGen: когда шаг PCIe-bound, пересчитывать gate|up для k слоёв вместо оффлоада (каждый ~1,4 мс GEMM против 2,5 мс PCIe); при N = 832 k ≈ 3 даёт около −8 мс; k выбирать по числу токенов шага.

## Выборочный пересчёт (Korthikanti et al. 2205.05198)

- [V] Пересчитывать только дешёвое по FLOP и дорогое по памяти (+2,7 %/1,6 % FLOP против 30–40 % у полного).
- [I] Для нас в backward Wqkv и Wo всё равно распаковываются; cuBLAS обещает побитовую воспроизводимость на той же архитектуре и конфигурации, значит бит-в-бит пересчитываемы: s (fp32, 4,5 MiB на слой при 767 токенах) = o_proj(сохранённый выход attention) + x за ~0,15 мс; q, k, v (3,0 MiB) = qkv GEMM + LoRA (h сохранён) + RoPE за ~0,2 мс; выход attention + lse (2,3 MiB) = mem-efficient forward (детерминирован) за ~0,18 мс; x (fp32) не пересчитывается дешевле 2 мс, оставить. Весь блок attention: ~0,5 мс на слой (+14 мс на шаг), −275 MiB при 767 и −510 MiB при 1 427 токенах в точке пика. Адаптивно (только в шагах, выходящих за целевой пик): стоимость в 2–3 раза меньше.

## Распаковка NF4

- [V] Уже на полосе памяти: 2,53 Б на параметр × 46,8 M × 56 ≈ 6,6 ГБ за шаг, при измеренных 220 ГБ/с это ~30 мс (измерено 32).
- [I] Держать вес в L2 нельзя: одна волна 128×256-плиток трогает 3–6 МБ W, пришлось бы считать на ≤ 20 CTA (вдвое медленнее). K-чанки с `addmm(beta=1)` численно нормальны, но каждый чанк перечитывает fp32 C (~45 мкс на 832×1536). FLUTE/QUICK/Marlin выигрывают только при batch ≲ 16–64 (memory-bound), у нас compute-bound.
- Рычаг: прятать распаковку следующего веса за хвостом GEMM на боковом потоке (двойной буфер, +20–26 MiB).

## CPU и границы шага

- [V] Трасса: простой 11,9 %, 105 мс за два шага; одна граница шага ~26,5 мс простоя (grad-norm, AdamW 3 мс, `.item()`, `_amp_update_scale_`, DataLoader `__next__` 2,8 мс); внутри слоёв самый большой источник `_QKVRopeBackward` 13 мс за два шага.
- [V] Eager PyTorch ~280 k операций/с (Horace He): 1 800 запусков это 10–20 мс CPU за шаг, прячется за GPU; простой даёт не число запусков, а синхронизации, осушающие очередь.

## int8 на тензорных ядрах

- [V] `torch._int_mm` (Blas.cpp 2.11): M > 16, K и N кратны 8, int32-выход, без эпилога; зовёт `cublasLtMatmul` с compute 32I; IMMA требует «TN», m и k кратны 4. bitsandbytes использует тот же путь без ветвления по архитектуре, HF заявляет T4 как поддерживаемый; для `_int_mm` на T4 не проверено (бенч агента имеет флаг `--int8`). [V] Jia et al.: int8 74,9 TOPS против 41,6 fp16 (×1,8).
- [V] Литература: SwitchBack (2304.13013) — per-row int8 для входов и градиентов, per-tensor для весов, forward и input-grad в int8, grad весов в fp16, совпадение с bf16 в пределах 0,1 п.п.; Jetfire (2403.12422) — per-token даёт +6…17 % к loss GPT-2, per-block 32×32 нужен свой GEMM; LLM.int8() — vector-wise absmax без декомпозиции +1,4–2 % loss на 1,3–2,7B; SmoothQuant +0,5–0,75 % perplexity на 7B; Xi et al. — градиенты при дообучении row-sparse, per-token подходит.
- [I] Оценка: int8 и forward, и backward ≈ −50 мс на шаг, но forward в int8 высокорисков для наших допусков (сдвиг loss 0,2–2 % против 1 % на первом шаге и 0,5 % eval). Backward-only (как SwitchBack) ≈ −20 мс на шаг при неизменных forward, первом шаге, eval и entropy; меняется только шум градиента; риск средний, не проверен. Несколько дней работы; при текущем бюджете времени это stretch-цель: сначала `--int8` на T4 и офлайн fake-quant на 64 шага.

## Ранжированные эксперименты

| № | Эксперимент | Ожидание | Риск |
|---|---|---|---|
| 1 | Конвейер границы шага: подготовить следующий батч (collate, план unpad, выборка эмбеддинга с хоста, копия на GPU) до синхронизаций scaler/clip; логирующие `.item()` читать на шаг позже | −15…−20 мс/шаг, побитово то же | низкий–средний (патч тренера) |
| 2 | Пересчёт блока attention (s, q, k, v, выход + lse), затем адаптивно | −275…−510 MiB, +14 мс/шаг (адаптивно ~+5), побитово | низкий (после T14b) |
| 3 | Распаковка следующего веса на боковом потоке, двойной буфер | −10…−25 мс/шаг, +20–26 MiB | средний (проверить `--overlap`) |
| 4 | GEMM чанками по токенам при N > порога, если свип покажет зависимость от M; Lt/без bias для qkv и o где быстрее | −10…−15 мс/шаг в среднем (до −40 на шагах по 1 400) | низкий |
| 5 | Выбор оффлоад/рекомпьют gate|up по шагу для баланса PCIe в backward | −5…−10 мс в PCIe-bound шагах | низкий |
| 6 | int8 только для input-grad GEMM в backward | ≈ −20 мс/шаг | большой объём, средний численный риск |
| 7 | Свернуть LoRA-B в qkv GEMM, bias qkv вынести из эпилога | −1…−3 мс/шаг | низкий |
| 8 | LoRA-градиенты весов на боковом потоке; `gc.disable()` в цикле | −1…−4 мс/шаг, +8 MiB workspace | низкий, не проверен |

## Прочитанные статьи

Stream-K 2301.03598; FlexGen 2303.06865; ZeRO-Offload 2101.06840; Korthikanti et al. 2205.05198; FLUTE 2407.10960; QUICK 2402.10076; Marlin 2408.11743; Jia et al. (T4 dissection) 1903.07486; SwitchBack 2304.13013; Jetfire 2403.12422; Xi et al. 2306.11987; LLM.int8() 2208.07339; SmoothQuant 2211.10438. Документация: cuBLAS; PyTorch v2.11.0 Blas.cpp, CUDABlas.cpp, CublasHandlePool.cpp, TunableOp README; PyTorch tuning guide; Horace He «Making Deep Learning Go Brrrr». Источники идей для README (правила просят называть источники).

## Ночная сессия (31 октября)

[V] измерено на T4 этой ночью или прочитано в источнике, [I] вывод.

### Измерения на T4

- [V] int8 тензорные ядра на T4 доступны из PyTorch 2.11 без своих расширений: `torch._int_mm` идёт в cuBLASLt, тот выбирает CUTLASS `cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16` (IMMA m8n8k16). Второй операнд обязан быть column-major («TN»); с row-major вторым операндом тот же вызов в 1,5–2,8 раза медленнее. На формах backward нашего шага (градиент по входу замороженных слоёв) при M = 832: gate|up ×1,88, down ×1,61, q|k|v ×1,17, o ×0,92; при M = 1 440: ×1,68 / ×1,61 / ×1,42 / ×1,21. Частота под нагрузкой 730–1 080 МГц при 53–72 Вт (лимит 70 Вт): int8 выигрывает не только вдвое большим пиком, но и меньшей энергией на операцию, то есть меньшим сбросом частоты. Квантование и масштабирование операциями torch съедают выигрыш (×0,3–0,7), нужны свои однопроходные ядра. `solution/bench_int8.py` в `claude/night-int8`.
- [V] Методика: первый прогон сессии Kaggle систематически быстрее по шагам (T4 холодная) и медленнее на старте (холодный дисковый кэш): soup-t14v base_1 22,73 с шагов 3+ против 24,1–24,4 у остальных девяти прогонов. Сравнения вариантов теперь с `--abba --warm-run`.
- [V] Переключатели T19b (split-K для q|k|v backward, cuBLASLt для o и down forward) в одной сессии variants в шуме: ≤ 4 мс на шаг.
- [V] Граница шага (трасса main под профилировщиком, шаги 8–9): ~40 мс простоя GPU между backward шага 8 и forward шага 9: add_ 5,9; .item 2,5 и 4,9; AdamW 2,75 (CPU 7 мс); `_amp_update_scale_` 6,8; копия на устройство 6,6; add 4,7 мс. Без профилировщика меньше, но это крупнейший резерв скорости, побитово нейтральный.

### Backward attention на sm_75 (субагент, исходники PyTorch v2.11.0; полные заметки в ночном отчёте)

- [V] На T4 backward идёт ядром `fmha_cutlassB_f16_aligned_128x64_k128_sm75` (54,8 КБ shared, 256 потоков, 1 CTA на SM, dK/dV в регистрах), сетка `(splits, 12 голов, n_seq)`; forward `fmha_cutlassF_f16_aligned_32x128_rf_sm75`. Источник: `aten/src/ATen/native/transformers/cuda/attention_backward.cu`, `mem_eff_attention/kernel_backward.h` (https://github.com/pytorch/pytorch/tree/v2.11.0/aten/src/ATen/native/transformers/cuda).
- [V] Forward побитово детерминирован (atomicMax только в shared для максимумов строк, суммы в фиксированном порядке, `kernel_forward.h:1241,1304`): это и даёт побитовый пересчёт N1.
- [V] По умолчанию splits = ceil(max_seqlen_k/64) до 8 (`attention_backward.cu:868-895`); единственная рабочая память это fp32-аккумулятор dQ (6–16 МБ), обнуляемый при splits > 1 (наши ~64 мкс на слой); dQ недетерминирован из-за спин-лока чтения-изменения-записи (`kernel_backward.h:2061-2148`), dK, dV детерминированы.
- [V] `num_splits_key` есть в схеме оп (`native_functions.yaml:15285`), передаётся ключевым аргументом в `torch.ops.aten._efficient_attention_backward`; 1 сплит делает backward детерминированным, без обнуления и замков (ветка `claude/night-attn`, `SOUP_SOLUTION_ATTN_SPLITS_KEY`). [I] Модель агента: при 1 сплите 48–60 CTA, каждая идёт по всей своей последовательности: примерно паритет на ровных батчах и ~1,7× медленнее, когда одна последовательность 512 упакована с короткими; замер на T4 в работе.
- [V] `max_seqlen` задаёт и число сплитов, и размер обнуляемого буфера: передавать точный максимум батча (у нас так и есть). `shared_storage_dqdkdv` при GQA ничего не даёт.
- [I] Путь через батчевые GEMM cuBLAS: ~260–390 мкс против ~450–550 мкс при N≈830 и паритет при N≈1500 (нет пропуска каузального треугольника), 11–77 МБ временной памяти на слой. Весь backward attention ≈ 1,5 с за прогон, так что потолок любого варианта 0,5–0,8 с.
- Источники: PyTorch PR #101847 (num_splits_key, https://github.com/pytorch/pytorch/pull/101847), PR #91979, xformers CHANGELOG и issue #1128 (https://github.com/facebookresearch/xformers/issues/1128), ssiu/flash-attention-turing (backward на T4 лишь в 1,35–1,51 раза быстрее этого ядра, https://github.com/ssiu/flash-attention-turing), FlashAttention-2 (arXiv 2307.08691), DASH (arXiv 2601.21824: детерминированный backward теряет до 38 % из-за сериализации dQ).

### Граница шага (субагент: исходники transformers 5.16.1, accelerate 1.14.0, torch 2.11.0, trl 0.29.1, soup-cli 0.77.0)

- [V] Около 29 ожиданий хоста за шаг. Три полных осушения очереди посреди шага: `int(labelled.sum().item())` в голове (`fused_ce.py:161` / `head_stream.py:603`), первый `.item()` метрик TRL в `trainer_patch.py:27` и проверка nan/inf потерь (`transformers/trainer.py:1773`); плюс `found_inf.item()` в `GradScaler._maybe_opt_step` (`torch/amp/grad_scaler.py:371`, оптимизатор foreach, не fused) и две синхронизации харнесса (`harness/_train.py:145,152`, их не трогаем). Backward без синхронизаций (оффлоад и стриминг весов на событиях).
- [V→I] Главная стоимость не синхронизации, а четыре прохода по всей модели за шаг на CPU при простое GPU: `Trainer.floating_point_ops` → `num_parameters` (`trainer.py:1784`, `modeling_utils.py:1026-1068`), `clip_grad_norm_(model.parameters())` (читает `.grad` у всех ~450 параметров дважды), `model.zero_grad()` (`trainer.py:1805`), `model.train()` (`trainer.py:1930`); каждое чтение атрибута 196 `Params4bit` идёт через Python `__torch_function__` bitsandbytes (~1 180 вызовов за шаг). Оценка агента: 12–15 мс на шаг. Сделано в `claude/night-boundary` (`solution/boundary.py`, `SOUP_SOLUTION_STEP_BOUNDARY=1`): списки обучаемых параметров и модулей один раз на прогон, сброс по глобальным хукам регистрации параметров/модулей torch; замер в работе.
- [I] Ещё: коммит SQLite трекера Soup на каждом логе (WAL, `synchronous=FULL`, вероятно fsync ~3 мс; менять только с разрешения организаторов), подготовка следующего батча в тени backward (−4…−6 мс), отложенное чтение метрик и rewind до точки логирования (−0,5…−1,5 мс). Не рекомендуется: fused/capturable AdamW (другая арифметика обновления), воркеры DataLoader (фоновые процессы запрещены).
- Источники: https://github.com/pytorch/pytorch/issues/80167, https://github.com/huggingface/accelerate/pull/2685, https://github.com/huggingface/transformers/pull/38029, https://github.com/huggingface/transformers/issues/42422, https://github.com/huggingface/transformers/pull/13619.

### Замеры этой ночи (одна сессия Kaggle на строку, ABBA где отмечено)

- [V] `num_splits_key` backward attention (peagverr, ABBA, 2 раунда): шаги 3+ base 24,63 с, 1 сплит 24,55, 2 сплита 24,46, 4 сплита 24,48: всё в шуме (≤ 2,7 мс на шаг); 1 сплит делает dq побитово воспроизводимым без потерь.
- [V] Прогрев в потоке внутри процесса (T15c, процесс-помощник удалён): первый шаг через 10,52–10,73 с после `apply begin` при 2 или 3 воркерах, со стартом в начале apply или с интервалом GIL 5 мс; у main с процессом-помощником в той же конфигурации 10,75 с. Удаление запрещённого помощника по времени ничего не стоит.
- [V] Пересчёт attention (N1): +16…35 мс на шаг за −490 MiB пика; split-sum без потерь по времени за −100 MiB.
- [V] Оффлоад входов слоёв (N1b) против пересчёта: оффлоад тензоров attention (q, k, v, выход, lse, half-ветка нормы) вместо пересчёта (N1c) даёт ту же память (563 против 561 MiB), но на 18 мс за шаг медленнее (peagverr, soup-night-oattn, ABBA): шина PCIe и так занята стримингом весов и копиями gate|up и входов, а пересчёт стоит ~0,75 мс GPU на слой. Пересчёт дешевле передачи, пока PCIe загружена.
- [V] Полоса T4 (peagverr, soup-night-nf4bw, прогретая карта): `torch` copy 240,7 ГБ/с, fill 227,4 ГБ/с (паспорт 320); наша распаковка NF4 211,5 ГБ/с на весе 8 960 × 1 536, то есть ~88 % практического потолка; перебор блоков на программу и варпов в пределах ±1 %, запись через 3-D индекс без reshape вдвое медленнее (теряется 16-байтная векторизация). Пол распаковки при двух распаковках каждого веса за шаг ~33 мс.
- [V] int8 backward с компиляцией в потоке после старта: первый шаг больше не ждёт компиляции (10,95 против 10,96 с от apply), но поток компиляции стоит ~0,3 с на шагах 0–4 (GIL), а выигрыш шагов между сессиями от −19 до −0,6 мс: в «горячих» сессиях, где T4 сильнее сбрасывает частоту, int8 полезнее. По стене от −1,2 до +0,2 с.
- [V] Комбинация ночи (`claude/night-combined`: split-sum, пересчёт всех слоёв, in-place backward MLP, оффлоад входов, префетч головы у нормы, очистка cuBLAS): полный прогон 47,1 с / 539 MiB (база 93,3 с / 14 345), жюри на других строках и seed 13 46,4 с / 579 MiB против 87,7 с у базы; все 10 проверок в обоих. Прогон решения сразу после 93 с базы идёт на горячей карте: медиана шага 0,4435 с против 0,383 в variants-сессиях той же конфигурации; порядок прогонов у жюри влияет на s сильнее, чем большинство наших оптимизаций шага.

### Память вне аллокатора и порядок копий (ночь, после 03:00)

- [V] Что nvidia-smi видит сверх `memory_reserved` на T4 (`solution/bench_ctx.py`, soup-night-ctx): контекст CUDA 104,9 MiB; первый GEMM cuBLAS +28 MiB (библиотека); attention, bitsandbytes и наши ядра Triton дальше ничего не добавляют; в обучении вне аллокатора ~167 MiB (ещё дескрипторы cuBLAS потоков autograd и прогрева, модули ядер). Самая большая регулируемая часть: резерв стека потоков, `cudaLimitStackSize` (1 КиБ по умолчанию) × 1 024 резидентных потока × 40 SM = 40 MiB.
- [V] Лимит стека 256 Б в начале обучения: −30 MiB пика nvidia-smi (479 → 449), время в шуме, eval тот же (soup-night-stack). По документации драйвера стек поднимается автоматически, если ядру нужно больше, и поднятый размер не сбрасывается («The driver automatically increases the per-thread stack size for each kernel launch as needed. This size isn't reset back to the original value after each launch»; флаг `CU_CTX_LMEM_RESIZE_TO_MAX` устарел, его поведение теперь по умолчанию): https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html. Значит, это безопасно по корректности, а пик за обучение показывает, что ни одно ядро прогона не требует больше 256 Б.
- [V] Порядок копий оффлоада важнее их объёма: копия gate|up на хост, начатая сразу после GEMM gate|up (а не после возврата forward MLP, когда хук упаковки видит сохранённый тензор), успевает пройти под SwiGLU, GEMM down и attention следующего слоя; ожидание копий перед каждым MLP на GPU тогда бесплатное, и gate|up двух слоёв больше не живут одновременно (539 → 499 MiB). Тот же принцип в backward: префетч в один тензор вместо окна слоя (499 → 479 MiB, +0,3 с на обе меры). Это прямое следствие «правильного времени выселения» из Capuchin (Peng et al., ASPLOS 2020, https://www.microsoft.com/en-us/research/publication/capuchin-tensor-based-gpu-memory-management-for-deep-learning/) на нашем масштабе: решает не что выселять, а когда начать копию относительно следующего выделения.
- [V] Сборщик мусора Python в цикле (`gc.freeze`/`gc.disable` на время обучения): в шуме, потому что startup.py уже замораживает объекты импорта в конце apply, а accelerate делает полную сборку до цикла; `gc.unfreeze()` при выходе стоит ~1 с (полная сборка по 650 тыс. объектов), его нельзя делать.

### Ресерч 03:10: новые рычаги времени и памяти (воркфлоу, 4 направления, скептическая проверка)

Четыре агента-исследователя (энергия и GEMM, память, ядра и attention, практики) принесли 23 находки. Каждую отдельный агент перепроверил по первоисточнику и нашим логам. Оставлены 3, ещё одна возвращена из отклонённых (стек 16 Б), к ним добавлен реплей снимка худшего шага (soup-night-wdiag). [V] — проверено в источнике, логе или снимке, [I] — вывод. Курс баллов сейчас: −20 MiB пика (4,5 %) ≈ 0,4 балла (нам, или столько же теряют соперники, если по памяти лидируем мы) ≈ 0,6 с стены.

**Что меняет оценку любого рычага памяти**
- [V] С `expandable_segments` large-пул (блоки больше 1 MiB) мапится страницами по 20 MiB, small-пул по 2 MiB, и за обучение ничего не размапливается. Reserved равен сумме максимумов двух пулов, даже если они приходятся на разные моменты (large — на голове, small 52 MiB — в `_multi_tensor_adam`). На худшем шаге large ≈ 242,5 MiB alloc = 13 страниц (260), плюс 52 small = reserved 312; nvidia-smi 449 = 312 + ~137 вне аллокатора. Срез large-пула виден, только если переводит через границу 20 MiB все фазы шага сразу; срез вне аллокатора доходит 1:1. https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/c10/core/AllocatorConfig.h, https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/c10/cuda/CUDACachingAllocator.cpp
- [V] Пик худшего шага приходится на начало головы. Реплей снимка soup-night-wdiag (шаг 8, 1 363 токена, alloc 274,9 MiB), large-пул по фазам:
  - голова 238,5: статика 88,5, кольцо головы 48, gate|up 27-го слоя 47,0, буфер логитов 21,3, сумма и fp32-выход финальной нормы 2 × 8,1, acc 6,3, x16 слоя 0 4,0, h16 4,0, wt 3,2;
  - финальная норма 219,9: кольцо, gate|up 27 и ещё не отпущенные s и выходы MLP 26-го и 27-го слоёв;
  - backward attention 27-го слоя 207,0: префетч gate|up 26 даёт 47,0, внутри оп 27,1, из них fp32-аккумулятор dQ 15,0;
  - MLP forward 199,4.

  Снимок fdiag (шаг 6) даёт тот же порядок: голова 227,6 > backward attention 199,2. Запись 19ad5b3 про 78,8 MiB внутри `_efficient_attention_backward` — ошибка атрибуции (и сплиты ключей пик не сдвинули). Резать голову глубже ~30 MiB бессмысленно, пока не срезаны финальная норма и backward attention.
- [I] Аккумулятор dQ (15,0 MiB) размерен по n_seq × max_seqlen (5 × 512 × 12 голов × 128 × 4 Б), а не по числу токенов. После этого раунда он и префетч gate|up предыдущего слоя — следующая цель в фазе backward attention.
- [V] CPU-пол: шаги короче ~600 токенов идут 0,30–0,32 с при любом числе токенов (8 прогонов решения, это ~20 % шагов). На таких шагах окупается только экономия CPU (запуски, Python); экономия GPU-времени и энергии там ничего не даёт.
- [V] Обрыва на длинных шагах нет: шаги 1 260–1 600 токенов укладываются в ±40 мс от линейной аппроксимации по 650–1 150 токенам во всех 8 прогонах, горячих и холодных. Обрыв cuBLAS за 4 МиБ L2 из Jia et al. (cuBLAS 2019 года, https://arxiv.org/pdf/1903.07486) у нас не воспроизводится. Пункт RESEARCH №4 (чанкование длинных шагов) закрыт.
- [V] Эффект горячей карты: на тех же строках цикл base_2 61,91 с против 55,59 у base_1, медиана шага решения 0,4333 с против 0,4181. Это утечка при лимите 70 Вт: горячий кристалл держит ту же мощность на меньшей частоте ещё до 85 °C (https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html). До теплового троттлинга Jia et al. грели карту ~1 900 с, а прогон жюри идёт ~5 мин.
- [V] Частота T4 пересчитывается каждые 20 мс по окну 80 мс (https://arxiv.org/html/2607.08307). Поэтому снятый простой GPU возвращается в стену не целиком. Перекрытие счёта со счётом не помогает (dqpipe +29 мс), перекрытие копий со счётом помогает.
- [V] Голова упирается в копии, а не в счёт. При N = 658 окно головы 41,4 мс, из них GPU занят 17,7. GEMM чанка стартует через 0,01 мс после конца его копии 24 MiB (2,04–2,36 мс, ~12 ГБ/с). Тюнинг ядер головы и CE ничего не даст; простой головы ~20 мс/шаг при N ≈ 826 возвращается только уменьшением числа байт.
- [V] `CUBLAS_WORKSPACE_CONFIG` не ставить. PyTorch 2.11 разбирает его std::regex при каждой выдаче хэндла, без кэша (`getChosenWorkspaceSize` внутри `setWorkspaceForHandle`), ~820 раз за шаг: +33…48 мс CPU на шаг (реплика: 42–59 мкс на вызов против 0,4–0,5 без переменной). Взамен 16 MiB, которые из-за страниц дают 0 или −20. https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/cuda/CublasHandlePool.cpp

**Оставлено, по баллу в час (переключатели и планы в выдаче воркфлоу)**
1. Лимит стека 16 Б (`SOUP_SOLUTION_STACK_LIMIT=16`, без кода; возвращён из отклонённых).
   - Проверка отвергла его порогом «< 10 MiB», но кода тут нет, а экономия вне аллокатора доходит до nvidia-smi 1:1.
   - 449 / 441 / 439 MiB при 256 / 64 / 16 Б ровно ложатся на формулу 40 SM × 1 024 потока × лимит: ни одно ядро стек обратно не поднимает, а если понадобится, драйвер поднимет его сам (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html, https://forums.developer.nvidia.com/t/cudadevicesetlimit-bug/320656).
   - Механизма замедления нет; «шаги 3+ на ~0,6 с медленнее» в stack2, вероятно, шум сессии. Перемерить ABBA.
   - Итог: −8…−10 MiB, 0 с.
2. Голова без лишних буферов (`SOUP_SOLUTION_HEAD_ALIAS=1` + `SOUP_SOLUTION_HEAD_ROWS=6144`).
   - Буфер логитов чанка становится видом буфера распаковки fused_proj (26,25 MiB fp16). Этот буфер простаивает от down-GEMM последнего слоя до первой распаковки его backward, всё идёт на одном потоке, а dh считается в том же проходе.
   - Кольцо головы 36 MiB вместо 48. Ядро то же (BLOCK = 8 192); алиас помещается для шагов до 2 240 токенов.
   - В фазе головы −34 MiB при 1 427 токенах и −43 при 1 971. В nvidia-smi реализуется −20…−40 MiB (13 страниц → 11–12); время 0…+0,06 с.
   - DRAIN_HEAD (−47 MiB ценой нескольких мс на шаг) сверх этого ничего не добавляет.
   - Опора: MEMO (последние слои не выселяются, их backward идёт сразу; https://arxiv.org/html/2407.12117v3), MsT (логиты S/M × V при плоском времени головы; https://arxiv.org/html/2407.15892).
3. Кольцо весов по матрицам (`SOUP_SOLUTION_WEIGHT_RING=type`).
   - gate, up и down распаковываются ровно раз за проход, поэтому им достаточно одного слота на тип. q/k/v/o пересчёт читает дважды, они остаются в двух слотах.
   - 25,7 MiB вместо 46: −20,3 MiB во всех фазах, значит всегда минус одна страница, −20 MiB.
   - Время 0…+3 мс/шаг, если H2D в backward (занят на 85–90 %) успевает.
   - SwapAdvisor: мелкая гранулярность пула даёт swap-in на шаг раньше (https://par.nsf.gov/servlets/purl/10191573). ZeRO-3 ограничивает живые веса по параметрам, а не по слоям (https://www.deepspeed.ai/docs/config-json/).
4. Пиновка 10 слабов оффлоада в начале apply() (`SOUP_SOLUTION_PIN_EARLY=1`, `SOUP_SOLUTION_OFFLOAD_PREPIN_MB=2560`).
   - Только при `sys.argv[:2] == ['soup', 'train']`, чтобы pytest не пинул; неиспользованные слабы отпустить в конце обучения.
   - Все 6 прогонов решения на жюри-строках (cjury, ejury2, fjury) пишут «pinned stack 2560 MiB in 10 slabs (6 prepinned)». Значит, 4 слаба по ~130 мс cudaHostAlloc пинуются на главном потоке в самом длинном шаге; этот шаг идёт 2,06–2,38× медианы, остаток над аппроксимацией 0,12–0,31 с.
   - Итог: −0,15…−0,35 с на жюри-подобных строках.
   - Регистрация pinned-памяти в torch 2.11 отпускает GIL (https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/csrc/cuda/shared/cudart.cpp). Сам PyTorch добавил cudaHostRegister против долгих блокировок cudaHostAlloc (https://github.com/pytorch/pytorch/pull/108488).
5. [I] Добивка до следующей страницы после п. 2–3, всё побитово (`SOUP_SOLUTION_DROP_DEAD=1`, `SOUP_SOLUTION_CUBLAS_IDLE_FREE=1`).
   - Финальная норма: до неё отпустить s и выходы MLP 26-го и 27-го слоёв, которые держат `hidden`, `branch` и `extra` в `unpad.varlen_model_forward` (−12 КБ/токен). Это механизм отклонённой находки «мёртвые тензоры»: её отвергли, потому что MLP forward не пик, но те же ссылки живы в фазе финальной нормы, а после п. 2 эта фаза определяет пик.
   - Голова: fp32-выход финальной нормы отпускать сразу после построения h16 (−6 КБ/токен).
   - Все фазы: одна рабочая область cuBLAS вместо двух, `torch._C._cuda_clearCublasWorkspaces()` на смене forward и backward (−8,1 MiB, https://docs.pytorch.org/docs/2.11/notes/cuda.html).
   - Все фазы: x16 слоя 0 (3 КБ/токен) живёт от forward до backward, потому что `norm_source` у слоя 0 возвращает None.
   - Оценка по страницам large-пула: 13 → 11–12 (п. 2) → 10–11 (п. 3) → 9–10 (п. 5). Вместе со стеком 16 Б публичные 449 → ~360–410 MiB.

**Отклонено**
- Ранжирование по джоулям на шаг и телеметрия энергии: методика без своего выигрыша. Её вопросы уже закрыты имеющимися логами (CPU-пол, нет обрыва на длинных шагах), а int8 ждёт решения организаторов (−0,55 с, fint8).
- `CUBLAS_WORKSPACE_CONFIG` (:4096:1, :16:8): regex на каждом GEMM (+0,4…1,0 с), по памяти 0 или −20 MiB, по правилам серая зона.
- DRAM/L2-рычаги GEMM (раскладки, fp32-выходы, L2-чанки распаковки, порядок ядер): DRAM даёт 1–4 % энергии GEMM при лимите мощности, потолок −3…−8 мс/шаг.
- Перебор алгоритмов cuBLASLt через ctypes: T19b в прогоне дал ≤ 4 мс/шаг (шум). Python на каждом GEMM утяжеляет короткие CPU-bound шаги, а побитовый пересчёт усложняет выбор алгоритма.
- fp16-аккумуляция, TunableOp, preferred_blas_library: не больше −6 мс/шаг при риске для проверок loss; у TunableOp на CUDA есть только Default.
- `empty_cache` в начале обучения: accelerate уже делает его в `free_memory()` (reserved 70 MiB при train begin). На каждом шаге он даёт те же 312 MiB и +0,9 с (soup-night-empty).
- Лимит стека ещё и в apply(): −30 MiB уже взяты в начале обучения, остался только хвост 16 Б (п. 1).
- Рабочая область cuBLAS 4 MiB и пулы хэндлов: env-переменная тянет regex и другой выбор ядер, пулы скорее дают 0 (cuBLAS 12.2+ выделяет лениво). Оставлена только «одна рабочая область» (п. 5).
- Моменты AdamW на хосте: измерено 0 MiB nvidia-smi и +1 с (+10…28 мс/шаг), потому что ленивый init на шаге 0 оставляет страницы small-пула замапленными. Повторять только с нулевыми видами на шаге 0 и копиями вне вычислительного потока (−16…−20 MiB small-пула).
- Балансировка пересчёт/оффлоад gate|up по шагу (RESEARCH №5): [I] пересчёт слоя (~46 GFLOP) стоит ~160 мДж при лимите мощности против ~3 мДж DRAM на копию; полный пересчёт MLP уже мерили, +50 мс/шаг.
- MLP чанками по токенам (Mini-Sequence): после среза головы MLP forward не пик (≤ −7,5 MiB), объём L.
- LoRA-путь на Triton (Philox-dropout, ранг 16, свёртка B в qkv): −5…−6,5 мс/шаг, но два новых ядра стоят ~1 с прогрева или +0,12…0,18 с ленивой компиляции. Нетто −0,1…−0,25 с за объём L.
- Нормы пишут x16 сразу (fp32-заглушка с нулевыми шагами): −0,15…−0,19 с минус компиляция варианта ядра, хрупкие фоллбэки, 0 MiB в пике.
- GQA-обвязка (k/v один раз, k12/v12 из пересчёта, сумма голов в rope) и общая распаковка Wqkv/Wo для пересчёта и backward: обе побитовые, по −0,09…−0,15 с каждая, ниже порога 0,3 с. Связкой ≈ −0,2 с, если агент простаивает. `num_splits_key=1` уже измерен, в шуме.
- Тратить запас памяти коротких шагов (без пересчёта в последних слоях, ранние слоты головы): обе добавки ложатся в фазу головы, то есть прямо в пик. −0,15…−0,25 с при риске фрагментации.
- Сжатие головы без потерь (DFloat11, https://arxiv.org/abs/2504.11651; 12 бит для bf16-чекпойнта, −25 % байт): +8…40 MiB в фазе пика, −0,2…−0,5 с, объём L, нужно согласие организаторов.
- N1c на прогретой карте: переключатели старше раннего оффлоада (при окне префетча 1 q/k/v/out приходят по требованию, +20–25 мс/шаг). Тезис «на горячей карте PCIe простаивает» не подтвердился: в жюри-порядке голова ждёт 6,0–6,7 мс/шаг.
- Часы фаз (psutil обрезает btime до секунды, харнесс вызывает nvidia-smi до Popen): это только учёт, до ~1 с переезжает между прологом и выходом в STATUS. 10 строк в phases.py стоят того ради честной таблицы в README, стене это 0 с. GC_PAUSE=freeze текущей версии не мерили (≤ 0,1–0,15 с, риск для проверки 5).
- Unsloth, RunLoRA, LoRAFusion, ALST TiledMLP, советы HF по QLoRA: всё уже есть в solution/ или неприменимо (dropout 0,05 зафиксирован, LoRA ~0,2 % FLOP шага, шаги ≤ 2 тыс. токенов).

_Примечание оркестратора (06:10): ниже синтез воркфлоу как есть. Уже применено после него: чанки головы по 4 096 строк (449 → 429 MiB, soup-night-head; кольцо головы и буфер логитов вместе ~35 MiB, это примерно то, что дал бы его п. 2 с 6 144 строками и алиасом), предварительная пиновка оффлоада 2 560 MiB (п. 4 без переноса в apply). Ошибка атрибуции «78,8 MiB внутри `_efficient_attention_backward`» была моей и исправлена в STATUS ветки `claude/night-combined`._

### int8 только в backward (субагент: симуляция на CPU с реальными весами Qwen2.5-1.5B в NF4, статьи, исходники)

- [M] Градиент gate|up (выход backward SwiGLU) внутри токена очень неровный: максимум строки в 25 раз больше RMS (до 127×), per-token int8 обнуляет 27 % элементов, ошибка GEMM 6,8 %; down и o спокойные (1,4–1,6 %). Длина свёртки роли не играет, важна неровность строк. Блочный Адамар по оси свёртки с обеих сторон (взаимно сокращается в GEMM) снижает ошибку всех четырёх GEMM с 14,1 % до 4,2 % (блок 16 уже 5,3 %). Сглаживание по каналам и fp16 для выбросов не работают: выбросы меняются от батча к батчу. Стохастическое округление хуже в 1,4–1,5 раза, per-tensor даёт 66 %.
- [M] Влияние на кривую loss в 10-шаговой симуляции даже без Адамара: 8,7e-5 относительного сдвига на шаг, eval −9,5e-5; с Адамаром ~3e-5. [V на T4] Наш полный прогон с int8 для down и gate|up без Адамара: все 10 проверок, кривая 0,01 % / 0,07 %, eval −0,013 % (как у fp16). Для допусков 1 % / 3 % / 0,5 % Адамар не нужен.
- [V] `torch._int_mm` (PyTorch v2.11.0 `Blas.cpp:859`, `CUDABlas.cpp:2227`): cuBLASLt, только int8 → int32, без эпилога, алгоритм и рабочая память по умолчанию; M > 16, K и N кратны 8, раскладка TN. В cuBLAS 12.8 на sm_75 нет int8 GEMM с fp16/fp32 выходом и масштабированием строка × столбец (https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html), поэтому деквантование вливается в соседние ядра (у нас в backward SwiGLU и отдельный эпилог). bitsandbytes 0.50.2 градиент по входу в int8 не считает, SwitchBack-модули удалены.
- Литература: градиенты требуют per-token масштабов (DeepSeek-V3: блочное масштабирование градиентов активаций расходилось, https://arxiv.org/abs/2412.19437), backward-only намного безопаснее forward (HALO, https://arxiv.org/abs/2501.02625), замороженные веса работают (HOT, https://arxiv.org/abs/2503.21261); SwitchBack https://arxiv.org/abs/2304.13013, Jetfire https://arxiv.org/abs/2403.12422, Xi et al. https://arxiv.org/abs/2306.11987, Quartet https://arxiv.org/abs/2505.14669, COAT https://arxiv.org/abs/2410.19313. Работ про int8 backward для QLoRA на NF4 не найдено.
- [V] Оффлоад входов слоёв (N1b, `claude/night-xoffload`): поверх N1 пик 801 → 581 MiB, с префетчем головы у финальной нормы 541 MiB (alloc 308), +0,85 с; оффлоадер возит 3 584 тензора за прогон, всё из префетча, PCIe хватает. Пик теперь в MLP любого слоя: статика 121 MiB + два gate|up (текущий и копия предыдущего в полёте) + hidden. Важная деталь реализации: у своей autograd-функции сохранённые тензоры упаковываются хуками после выхода из `forward`, поэтому метка для оффлоадера ставится по адресу тензора, а не контекстом вокруг `save_for_backward` (первая версия молча ничего не оффлоадила).
- [V] Распаковка следующего веса на боковом потоке (`claude/night-dqpipe`) на T4 медленнее: +29 мс на шаг и +40 MiB. Параллельная распаковка отнимает у GEMM SM и мощность (лимит 70 Вт), это хуже, чем выигрыш от перекрытия; совпадает с выводом T19 про два потока GEMM.
- [V] int8 backward MLP в обучении: медиана шага −12…−36 мс (сильнее в «горячих» сессиях с большим троттлингом), но три своих ядра при прогреве отодвигают первый шаг на ~1 с; стена −0,2…+1,2 с. Численно чисто. Чтобы выигрыш дошёл до стены, компиляцию надо спрятать (поток после старта обучения с fp16 до готовности) или сократить.



---

# Приложение D. docs/STATUS.md (разделы: итог ночи, кольцо весов, старт, полные прогоны, загрузчик, стек, оффлоад, int8, пересчёт, измерения 30 октября, таблица прогонов)

# Состояние работ

Обновляется главной сессией после каждого влитого модуля и каждого прогона на Kaggle. Для передачи эстафеты: читать этот файл, `docs/TASKS.md`, `README.md`.

## Ночная сессия (31 октября, с 00:20 по Астане): ход работ

**Итог ночи (06:45 по Астане).** Кандидат на влитие в main: **`claude/night-combined`** (итоговый код a0cb72e; main b098961 + все ночные ветки + T15c, ночные умолчания включены, `solution/__init__.py` изменён только T15c). Итоговые числа на T4:
- **полный прогон** (soup-night-cfull5, a0cb72e, до кольца по типам): все 10 проверок, **42,5 с / 421 MiB** против базы 77,5 с / 14 345 MiB: s = 1,83, **m = 34,1** (с кольцом по типам, 374ec0a: 46,1 с / **401 MiB**, m = 35,8, в горячей сессии) (main: 41,0 с / 1 311 MiB, m 10,9); eval −0,014 %, кривая loss 0,05 %; **pytest решения 331 из 331**; на предыдущем коммите (cfull4, горячая сессия) 47,3 с / 431 MiB при базе 89,6 с (s 1,89); первый шаг через ~9,2 с после apply (было 10,4–11,2);
- **жюри** на восьми прогонах с другими строками и seed (7 дважды, 13 дважды, 21, 42, 99, 123): все 10 проверок в каждом; seed 123 на итоговом коммите: 48,6 с против 85,9, **421 MiB**, eval +0,008 %; на итоговом коммите (строки seed 13): 47,05 с против 84,7 у базы, **459 MiB**, eval −0,003 % (те же строки в 02:00: 46,4 с / 579 MiB); на строках J1: 47,3 с против 87,0, 451 MiB (в 00:40 с одним N1: 47,4 с / 821 MiB);
- **эффект ночи в одной сессии** (soup-night-total2: все ночные умолчания против той же ветки с выключенными ночными переключателями, то есть конфигурации main): **+1,74 с (+3,9 %) за −880 MiB (−67 %)**; старт быстрее main на 0,9 с, цикл дороже на 2,3 с (пересчёт attention и оффлоад); eval одинаковый;
- финальная variants-сессия (soup-night-final): стартовые переключатели −1,2 с; лимит стека 16 Б ещё −10 MiB без потерь по времени (431 → 421 MiB), включён (a0cb72e); подтверждающие полный прогон и жюри на итоговом коммите прошли (выше).

**08:05: сверх итогового коммита влито и включено кольцо весов по типам матриц** (воркфлоу «реализация + ревью»; ревью «safe-to-enable»). Подтверждено на T4: **полный прогон 374ec0a: все 10 проверок, 46,1 с / 401 MiB (m = 35,8)**, eval −0,016 %; **жюри (строки seed 13): все 10 проверок, 439 MiB** против 459 без него, eval −0,006 %. Итоговый код ветки теперь **09ba9f8** (плюс правка трёх тестов под новое умолчание; финальный pytest soup-night-ctest8 идёт). Откат одной переменной: `SOUP_SOLUTION_WEIGHT_RING=layer`.

**Что включено по умолчанию** (каждое за своим переключателем `SOUP_SOLUTION_*`, все откатываются на оригинал при неожиданном входе): split-sum в пост-attention норме; пересчёт блока attention в backward во всех слоях (побитово); in-place backward SwiGLU; оффлоад входов слоёв; ранний старт копии gate|up на хост и ожидание копий перед каждым MLP, префетч backward в один тензор; очистка рабочих областей cuBLAS в начале обучения; префетч головы у финальной нормы и чанки головы по 4 096 строк; лимит стека CUDA 16 Б на поток (`cudaDeviceSetLimit` в начале обучения: −40 MiB вне аллокатора); загрузка модели через pinned-буферы с приведением bf16→fp16 на GPU (побитово, setup −1,7 с); предварительная пиновка оффлоада 2 560 MiB; лаунчеры Triton с -O0 (прогрев больше не держит первый шаг); подготовка строк Soup без отбрасываемого рендера и дешёвый детерминированный отпечаток datasets (строки идентичны).

**Выключено и почему:** int8 для GEMM градиента по входу в backward (готов, −0,55 с на итоговых умолчаниях, eval +0,013 %, но организаторам вопрос не задан); оффлоад attention вместо пересчёта (медленнее); моменты AdamW на хосте, `empty_cache` по шагам, пауза сборщика мусора, сплиты ключей backward attention, ожидание копий перед головой (выигрыша нет: пик reserved задают страницы по 20 MiB и фрагментация); ранние пробы nvidia-smi/entry points в потоке (малый выигрыш, подпроцесс из фонового потока); прочее см. STATUS ветки `claude/night-combined`.

**Решения для команды утром:** (1) влить `claude/night-combined` в main (`git merge origin/claude/night-combined`, main с вечера не двигался); (2) лимит стека CUDA — настройка контекста, не аллокатора и не гиперпараметр, корректность по документации драйвера не страдает; если есть сомнение, `SOUP_SOLUTION_STACK_LIMIT=0` вернёт +40 MiB; (3) int8 — спросить организаторов, включается одним переключателем; (4) README: строки про новые переключатели и итоговые числа. **Дальше** (ресерч ночи, RESEARCH.md): кольцо весов по типам матриц (−20 MiB во всех фазах), алиас буфера логитов головы на простаивающий буфер распаковки, отпустить мёртвые ссылки у финальной нормы, одна рабочая область cuBLAS; каждый срез большого пула виден в nvidia-smi только ступенью в 20 MiB.

| Ветка | Что | Результат на T4 | Что переключить |
|---|---|---|---|
| `claude/night-combined` | интеграция всех ночных веток + T15c; ночные умолчания включены | full (a0cb72e): 10 из 10, **42,5 с / 421 MiB** (s 1,83, m 34,1), pytest 331/331; жюри на 6 наборах: 10 из 10, последнее 47,3 с / 451 MiB | **влить в main** |
| `claude/night-recompute` | N1: пересчёт блока attention + split-sum + in-place backward MLP | пик 1 311 → 821 MiB (all), +16…35 мс/шаг за пересчёт; жюри J1 (seed 7): 10 из 10 | входит в combined |
| `claude/night-xoffload` | N1b: оффлоад входов слоёв; N1c: оффлоад тензоров attention вместо пересчёта | N1b: 801 → 581 MiB, с префетчем головы у нормы 541 (+0,85 с к N1-all); N1c: 563 MiB, но на 18 мс/шаг медленнее пересчёта | N1b входит в combined, N1c не включать |
| `claude/night-int8` | N2: int8 для GEMM градиента по входу MLP | численно чисто (10 из 10); шаги −12…−36 мс медианы, но компиляция трёх ядер при прогреве +1 с к старту; новая версия компилирует их в потоке после старта (первые 4 шага fp16, точка переключения не зависит от таймингов), замер soup-night-int8lazy в работе | не включать (вопрос организаторам) |
| `claude/night-early` | ранний старт копии gate|up на хост + ожидание копий перед каждым MLP + префетч backward в один тензор; переключатель сборщика мусора; диагностика градиентов | 539 → 499 → 479 MiB за +0,3 с; сборщик мусора в шуме; градиенты двух прогонов одного пути расходятся на ~3·10⁻⁶ (атомики dQ attention), пересчёт на том же уровне | входит в combined |
| `claude/night-nf4bw` | бенч полосы распаковки NF4 | 211 ГБ/с против 241 у copy: распаковка на ~88 % практической полосы, перебор конфигураций ±1 % | ничего |
| `claude/night-load` (подагент) | копии загрузки модели через pinned-буферы с потоком-подготовщиком, приведение bf16→fp16 на GPU | побитово (все 338 тензоров реального чекпойнта); setup 5,33 → 3,67 с, первый шаг −0,12 с (дальше ждёт прогрев 0,7–0,9 с), стена −0,3 с | входит в combined, включено |
| `claude/night-trinit` (подагент) | старт: лаунчеры Triton -O0, быстрая подготовка строк, дешёвый отпечаток datasets, ранние пробы; профиль старта | первый шаг 11,54 → 9,54 с от apply в одной сессии подагента, стена 47,28 → 44,59; Soup 79/79 | входит в combined: всё, кроме ранних проб |
| `claude/night-boundary` | граница шага без проходов по модели | в шуме | не включать |
| `claude/night-attn` | `num_splits_key` backward attention | в шуме; 1 сплит = детерминированный dQ | по желанию `=1` |
| `claude/night-startup` | main + T15c (помощник удалён) + 2 воркера | старт как у main с помощником | входит в combined |
| `claude/night-dqpipe` | распаковка следующего веса на боковом потоке | медленнее (конкуренция за SM и полосу) | не включать |
| `claude/night-jury` | N1 + режим жюри T18 | инструмент для N3 | входит в combined |

Ветки ночи (все от main b098961, в main не вливались): `claude/night-int8` (бенч int8, инструменты раннера), `claude/night-recompute` (N1). Общий коммит инструментов в каждой ветке: `tools/kaggle_run.py --mode bench --bench "<модуль> [аргументы]"` (без обучения), `--skip-pytest` (минус ~4 мин на сессию), `--abba` (чётные раунды в обратном порядке) и `--warm-run` (один холостой первый прогон). Причина последних двух: первый прогон сессии систематически быстрее по шагам (T4 ещё холодная) и медленнее на старте (холодный дисковый кэш): в soup-t14v base_1 дал 22,73 с шагов 3+ против 24,1–24,4 у всех последующих.

**Вечерние прогоны, забранные ночью.**
- soup-full-a (namedesign, f11e94b, full): уже в таблице (41,0 с, 1 311 MiB, 10 из 10).
- soup-t14v (namedesign, T19b 7ed1b27, variants, 2 раунда, шаги 3+, с): base 22,73 / 24,38; qkvsk 24,29 / 24,32; olt 24,31 / 24,26; downlt 24,21 / 24,10; all3 24,21 / 24,10. Без base_1 (холодный первый прогон) разница ≤ 0,28 с за прогон (≤ 4 мс на шаг, в шуме); пик 1 311 у всех. Вывод: переключатели T19b оставить выключенными, выигрыш не доказан.
- soup-t12 (namedesign, T20 fab2022, variants marks/profile): фазы setup на T4 от `setup_transformers begin`: токенизатор 0,84–1,05 с, до цикла загрузки 0,3–0,8 с, цикл загрузки 3,24–3,45 с, kbit 0,05, peft 0,14–0,46 с. В профиле (Python 3.13 cProfile видит все потоки) `.to` 1,75 с (страничные H2D-копии bf16) и 3,6 с хеширования triton_key потоком прогрева (на ветке T15c хеш уже вынесен раньше). Пиновка буферов load_stream 0,78 с в своём потоке. Резерв загрузчика по-прежнему ~0,3–0,5 с (pinned-буферы для копий).

**Диагностика main (namedesign, b098961, diag, 31.10 00:37).** Пики шагов по alloc: 406–1 072 MiB (худший шаг 40), статика 129 MiB (2 слота стриминга 46, буфер распаковки 26, LoRA и Adam ~26, 3 рабочих области cuBLAS по 8,1 = 24: прогрев, прямой поток и поток autograd), reserved 1 144, nvidia-smi 1 323. Состав пика шага 6 (767 токенов, 655 MiB, пик на down-GEMM MLP последнего слоя): суммы после attention 126 MiB (28 × 4,5), входы слоёв 121,5 (27 × 4,5), выход attention + lse 65, q 63, gate|up текущего и предыдущего слоя (оффлоад в полёте) 52,5, кольцо головы 48 (префетч у эмбеддинга живёт весь forward), hidden SwiGLU 13, k и v по 10,5. N1 нацелен ровно на суммы, q/k/v, выход и lse: −275 MiB при 767 токенах.

**N2, шаг 1: int8 на T4 работает** (`claude/night-int8`, `solution/bench_int8.py`, peagverr, сессия 31.10 00:30). `torch._int_mm` (cuBLASLt → CUTLASS `cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16`, нужен «TN»: второй операнд column-major; row-major второй операнд в 1,5–2,8 раза медленнее). Время одного GEMM градиента по входу, fp16 против int8 (без квантования): M = 832: q|k|v 232 → 198 мкс (×1,17), o 165 → 180 (×0,92), gate|up 2 118 → 1 123 (×1,88), down 1 041 → 645 (×1,61); M = 1 440: ×1,42 / ×1,21 / ×1,68 / ×1,61. Forward-формы для справки: down ×2,03, gate|up ×1,58–1,64. Квантование и масштабирование операциями torch съедают всё (×0,3–0,7): нужны свои ядра (распаковка NF4 сразу в int8 с поколоночными масштабами в транспонированной раскладке, квантование градиента по токенам за один проход, эпилог int32 → fp32). Ошибка per-token × per-column на синтетике с выбросами столбцов: 5–8 % относительной нормы (fp16: 0,02–0,04 %). Оценка: int8 только для gate|up и down в backward ≈ −39 мс GEMM на шаг при 832 токенах минус ~10–14 мс накладных = −25…−30 мс на шаг (7–8 %); численный риск средний (нужен Адамар по оси свёртки или проверка на реальных градиентах).


## Ночь, 08:05: кольцо весов по типам матриц влито и включено (`claude/night-wring` → 374ec0a)

Воркфлоу (реализация + независимое ревью): `SOUP_SOLUTION_WEIGHT_RING=type` в `weight_stream.py` и `fused_proj.py`: группа attention (q, k, v, o: 2,84 МБ на слой) остаётся в кольце двух слоёв, у gate, up и down по одному слоту на тип (каждая распаковывается ровно раз за проход; `acquire` перед распаковкой, `release` сразу после её постановки запускает копию той же матрицы следующего слоя по направлению прохода); слоты 25,7 вместо 46,0 MiB. Тесты `test_weight_ring.py` (10 случаев): побитово против резидентной модели с GPU-задержкой перед каждой распаковкой, три режима (пересчёт, SDPA math, MLP recompute), внеочередные вызовы и eval; полный pytest 341 из 341. T4 (namedesign, soup-night-wring, ABBA): пик **421 → 401 MiB**, reserved 292 → 272, alloc 254,9 → 234,6, стена 45,97 против 45,99 с, ожидания стримера 0,27 мс/шаг. Ревью: «safe-to-enable» (замечания только об усилении тестов). **Включено по умолчанию** (`SOUP_SOLUTION_WEIGHT_RING=layer` вернёт старое кольцо); **подтверждающий полный прогон** (peagverr, soup-night-cfull6, 374ec0a): **все 10 проверок, 46,1 с / 401 MiB** против базы 85,3 с: s = 1,85, **m = 35,8**; eval −0,016 %, кривая loss 0,05 %. В его pytest 338 прошли, 3 упали: тесты кольца по слою ждали ноль поздних загрузок и пустой глобальный реестр матриц при новом умолчании; исправлено в 09ba9f8 (эти тесты явно ставят `layer`, тест переключателя смотрит только свою модель), перепроверка soup-night-ctest8 идёт. **Жюри** (namedesign, soup-night-mjury, 374ec0a; строки seed 13, как cjury и kjury): **все 10 проверок**; база 95,5 / 96,0 с (очень горячая сессия), мы 51,3 / 52,6 с: s = 1,84; пик **439 MiB** против 459 на тех же строках без кольца по типам (−20 MiB, как на публичных данных): m = 32,7; eval −0,006 %; разрывы 0,04 % / 0,037 % / 0,07 % / 0,36 % / 0,17 %.

## Ночь: кольцо весов по типам матриц (ветка `claude/night-wring`, от `claude/night-combined` 8aab7d8)

**Идея.** Кольцо стриминга NF4-весов держало два целых слоя (2 × 24,1 МБ = 46 MiB) во всех фазах шага, хотя каждая матрица MLP слоя за проход распаковывается ровно один раз (forward: gate, up, down; backward с in-place: down, gate, up). `SOUP_SOLUTION_WEIGHT_RING=type` (по умолчанию `layer` — прежнее поведение, переключатель выключен): группа attention (q, k, v, o: 2,84 МБ на слой) остаётся в кольце из двух слотов по чётности слоя с прежними хуками слоя (пересчёт attention читает её в backward ещё раз), а у каждого типа матриц MLP (gate, up, down: по 7,1 МБ) один слот на все слои: 2 × 2,84 + 3 × 7,1 = 27,0 МБ (25,7 MiB) вместо 48,3 МБ (46,0 MiB), −20,3 MiB во всех фазах.

**Как сделано** (`solution/weight_stream.py`, `solution/fused_proj.py`; `solution/__init__.py` не менялся): Params4bit каждой матрицы (и её absmax, absmax второго уровня) навсегда смотрит в свой слот, меняются только байты. Слоты всех групп в одном плоском списке, у каждого свои события `_ready`/`_free` на одном потоке копий; копия группы — один непрерывный кусок раскладки слоя в pinned-памяти (порядок q, k, v, o, gate, up, down уже такой, слои, адоптированные у загрузчика, тоже). В `_MLP` fused_proj каждая распаковка большой матрицы обёрнута: `acquire(module, step)` — вычислительный поток ждёт копию этой матрицы (в слоте чужой слой — копия запускается на месте), `release(module, step)` сразу после постановки распаковки (GEMM читает общий half-буфер, не слот) — копия той же матрицы следующего слоя по направлению прохода (i+1 в forward, i−1 в backward). Режим recompute держит up между двумя его распаковками, int8-путь держит все три. MLP, читающий матрицы без acquire (не forward fused_proj или его откат: малое M у bitsandbytes, хуки, другой dtype), получает свои три матрицы перед запуском и ещё раз в начале backward слоя (хук `_LayerExit`). Eval и вызовы вне порядка грузят нужное на месте. Слой, который не делится на группы, или заданный `SOUP_SOLUTION_WEIGHT_STREAM_ONLY` — кольцо слоёв с причиной в логе. Копии и ожидания матриц MLP по умолчанию без CUDA-событий замера (`SOUP_SOLUTION_WEIGHT_RING_TIMING=1` включает: шесть лишних пар событий на слой и проход стоят CPU), группа attention замеряется как раньше; представления источника и приёмника каждой копии строятся один раз при установке.

**Тесты** (`solution/test_weight_ring.py`, только CUDA, 10 случаев; на T4 все проходят, 93249cb, вместе с 20 тестами `test_weight_stream.py`): Qwen2 с формами слоя 1,5B (4 слоя) через патчи слоя apply() с ночными умолчаниями (пересчёт attention всех слоёв, оффлоад MLP, LoRA dropout 0,05; один сплит ключей в backward attention ради воспроизводимости): выход, градиент входа первого слоя и градиенты LoRA двух шагов с кольцом слоёв и с кольцом типов побитно равны резидентной модели, хотя каждая распаковка стоит за ~3 мс сна GPU (варианты: пересчёт, SDPA math, `MLP_MEMORY=recompute`); слоты и счётчики (на шаг 2(n−2) копий attention и 2(n−1) копий каждой матрицы, ни одной поздней, по одному ожиданию на матрицу и проход, с замером матриц и без); память (кольцо 46,0 → 25,7 MiB, allocated ровно на 20,3 MiB меньше по requested bytes аллокатора); вызовы вне порядка (MLP напрямую в другом порядке, 8 токенов через собственный GEMM bitsandbytes без acquire, forward+backward одного MLP, eval всей модели через HF, затем шаги побитно); переключатель и откаты; MLP без acquire (Block из test_weight_stream) побитно, в том числе при GPU далеко позади CPU. Полный pytest variants-сессии 1 (44b51b6): 337 прошли, 2 упали — оба теста этой ветки с ошибкой в самом тесте (замер памяти между двумя разными состояниями; eval в режиме обучения без пересева dropout), исправлены в f9d7cd7/3fe1d15 и проходят на T4.

**Замер 1** (namedesign, soup-night-wring v1, 44b51b6 — все матрицы с CUDA-событиями замера; ABBA, 2 раунда + холостой, с pytest): base (кольцо слоёв) стена 45,88 / 44,28 с (45,08), шаги 3+ 27,41 / 26,30 (26,86), пик nvidia-smi 421 / 421 MiB, reserved 292, alloc 254,9; **type: стена 45,78 / 45,38 (45,58), шаги 3+ 27,48 / 27,05 (27,27), пик 401 / 401 MiB, reserved 272, alloc 234,6** (−20 MiB везде, как рассчитано: одна страница 20 MiB большого пула). Ожидания вычислительного потока (строка «compute stream waited»): base 19,8–20,3 мс за прогон (0,31 мс на шаг), type 58,2–59,3 мс (0,92 мс на шаг, **+0,6 мс на шаг**), из них матрицы MLP 40,5–41,4 мс на 10 752 ожиданиях (~4 мкс на ожидание — цена самой пары событий, реальных простоев нет); копий 13 701 (attn 3 330, gate/up/down по 3 457 — по одной на матрицу и проход), ни одной поздней, 0,53 мс на копию при 11,6 ГБ/с. Пошагово type − base: медиана +2,5 мс, среднее +6,6 мс, type медленнее в 43 шагах из 62: при +0,6 мс ожиданий на GPU остальное — CPU (четыре копии и четыре ожидания с замером на слой и проход вместо одного), отсюда версия без событий замера на матрицах (93249cb). eval 2,04938–2,04939 против 2,04935–2,04936 (разброс прогонов: атомики dQ в backward attention). Проверки 3, 4, 9 проходят, свои ядра 14,2–14,4 %.

**Замер 2** (namedesign, soup-night-wring v2, 93249cb — матрицы без событий замера; ABBA, 2 раунда + холостой, с pytest: **341 прошли, 5 пропущены, 0 упали**; горячая сессия, base_1 и base_2 по краям сошлись до 0,06 с): base стена 46,04 / 45,93 с (45,99), шаги 3+ 28,36 / 28,30 (28,33), пик 421 MiB, reserved 292, alloc 254,9; **type: стена 45,89 / 46,04 (45,97), шаги 3+ 28,30 / 28,38 (28,34), пик 401 MiB, reserved 272, alloc 234,6**; type с замером матриц (`SOUP_SOLUTION_WEIGHT_RING_TIMING=1`): 46,18 / 46,04 (46,11), шаги 3+ 28,52, пик 401. Пошагово type − base: медиана −1,4 мс, среднее +0,1 мс (медленнее в 24 шагах из 62) — в шуме; с замером матриц +1,1 / +3,1 мс: события замера и были ценой первого замера. Ожидания: base 19,9 мс за прогон (0,31 мс на шаг); type — группа attention 17,5 мс (0,27 мс на шаг, меньшие копии готовы раньше), матрицы не замеряются; type с замером 55,6–56,9 мс (0,88 мс на шаг), из них матрицы 38,0–38,9 мс на 10 752 ожиданиях (~3,6 мкс — пол замера, простоев нет). Ни одной поздней копии ни в одном прогоне. eval 2,04934–2,04940 против 2,04936–2,04938 (тот же разброс прогонов). Проверки 3, 4, 9 проходят, свои ядра 13,5–13,6 % против 13,4.

**Вывод: включить** `SOUP_SOLUTION_WEIGHT_RING=type` (в `apply()` или умолчанием `ring_from_env` в weight_stream.py): **пик nvidia-smi −20 MiB (421 → 401 на публичных строках), reserved −20, alloc −20,3 MiB**, по времени в шуме (версия 93249cb), результат побитно равен кольцу слоёв (тесты), откаты на кольцо слоёв и загрузку на месте покрыты. `SOUP_SOLUTION_WEIGHT_RING_TIMING` оставить выключенным (+3 мс на шаг CPU; только для диагностики).

## Ночь: старт до первого шага, сборка тренера и apply() (ветка `claude/night-trinit`, от `claude/night-combined` c1c51bf, затем слита с 940a5a6)

**Измерение** (`solution/startup_profile.py`, `SOUP_SOLUTION_STARTUP_PROFILE=1`, по умолчанию выключено; подключено из `phases.mark` и `startup.patch_soup_first_step_mark`, `solution/__init__.py` не менялся): семплер стеков всех Python-потоков каждые 4 мс от `apply begin` до конца первого шага (cProfile на 3.13 пишет все потоки на один стек и вдвое замедляет Python), доли CPU главного потока и остального процесса на каждой метке, метки внутри сборки тренера и начала train(), счётчик os.stat системы импорта. Первая версия сама замедляла импорт на треть (хэш code-объектов на каждом тике): исправлено, сейчас искажение apply ~0,25 с. Осторожно: «self» у кадров, отпускающих GIL (os.stat, чтение файлов, копии), завышен семплером; на T4 13 153 вызова os.stat импорта от apply до шага стоят всего 0,28 с.

**Окно (a), конец `_setup_transformers` → первый шаг, ~1,8 с** (T4, c1c51bf, сессия soup-night-trinit 1, база):
1. токенизация 256 строк Soup (`_map_text_sft_rows`) ~1,3 с: отпечаток datasets ~0,5 с (dill пиклит замыкание с токенизатором, `dill.detect.nestedglobals` печатает `dis.dis` в строку и парсит её), сами строки ~0,6 с (4 рендера шаблона и 4 токенизации на строку: первая — запрос маски ассистента, которую шаблон Qwen без `{% generation %}` не даёт; поштучная проверка токенов `_coerce_int_list`); под хвостом прогрева 0,8–1,0 с вместо 0,55;
2. accelerate `PartialState`: `nvidia-smi` для проверки P2P, 75 мс при 5 % CPU главного потока;
3. `importlib.metadata.entry_points()` в поиске плагинов Soup в начале train(), 70 мс;
4. assistant_masks 0,07; оптимизатор, даталоадер и on_train_begin 0,08; empty_cache 0,03; SFTTrainer+Trainer init 0,05 с.
С pinned-загрузкой (940a5a6) к этому окну добавилось ожидание прогрева 0,7–0,9 с на входе в train.

**Окно (b), apply() ~3,5 с:** импорты 2,6–2,7 с (bitsandbytes 1,1 — обёртки torch.compile грузят torch._dynamo и часть inductor, которые transformers.masking_utils грузит всё равно; trl.trainer.sft_trainer 0,9; transformers.data 0,4 — squad тянет BERT; loss_utils 0,2), всё это база платит тоже; наш код вне импортов ~0,3 с; поток хэша triton (415 МБ) 1,4 с CPU рядом.

**Прогрев против главного потока** (сессия 1, профиль): в окне загрузки модели два воркера и их дочерние gcc/ptxas — 11,5 поток-с, из них 5,3 с ожидание gcc (сборка лаунчеров с -O3, ~0,3 с CPU на каждый из 16–18 лаунчеров и cuda_utils), 0,9 с ptxas, остальное фронтенд и MLIR. Главный поток при этом 74 % времени на CPU и тратит на ту же работу на 24 % больше CPU (соседние гиперпотоки): загрузка модели 5,3 с против 3,5 без прогрева. Один воркер: загрузка на 1 с короче, но прогрев заканчивается на 2,4 с позже.

**Что сделано** (всё за переключателями, по умолчанию выключено; подключение из `warmup.start_early` и `startup.patch_soup_first_step_mark`):
- `SOUP_SOLUTION_LAUNCHER_OPT=0`, `SOUP_SOLUTION_LAUNCHER_NICE=10` (warmup.py): лаунчеры triton собираются через его собственный хук `triton.knobs.build.impl` тем же компилятором и с теми же флагами, но с -O0 вместо -O3 и под `nice`; кубины (ptxas) не меняются. Локально лаунчер 0,30 → 0,13 с. На T4 gcc в окне загрузки 5,2 → 2,9 поток-с, прогрев 6,7 → 5,0 с.
- `SOUP_SOLUTION_WARMUP_ORDER=longest` (warmup.py): очередь прогрева по измеренной стоимости, длинные первыми (по порядку регистрации последними шли head_stream и nf4_dequant3; симуляция двух воркеров 4,84 → 4,57 с).
- `SOUP_SOLUTION_FAST_ROWS=1` (trainer_build.py): для настоящего токенизатора с шаблоном без `{% generation %}` и строки с ответом ассистента `_apply_template_with_mask` сразу возвращает None (то, что Soup и так получает), список из точно int копируется без поштучной проверки; строки побитово те же.
- `SOUP_SOLUTION_FAST_FINGERPRINT=1`: глобальные имена функций для dill из сырого байткода вместо разбора текста `dis.dis`; тот же отпечаток.
- `SOUP_SOLUTION_CHEAP_FINGERPRINT=1`: отпечаток map Soup по модулю, имени и code-объекту функции вместо пиклинга замыкания (детерминированный, RNG не трогается; разъяснение разрешает даже случайный). Локально map+masks 0,71 → 0,34 с вместе с FAST_ROWS.
- `SOUP_SOLUTION_EARLY_PROBES=1`: проверка P2P accelerate (`nvidia-smi`) и `entry_points()` считаются в потоке во время загрузки модели, те же вызовы отвечают оттуда (entry points — пока sys.path тот же, до конца train()), как разрешённый кэш packages_distributions.
- `SOUP_SOLUTION_EARLY_ROWS=1`: токенизация строк Soup в потоке во время загрузки модели. **Проигрывает**: загрузка удлиняется на столько же, на сколько короче сборка (сессия 2: модель +0,69 с, сборка −0,46). Не включать.
- исправлено: флаг EARLY_ROWS затенялся списком строк, и EARLY_PROBES включал раннюю токенизацию тоже (вариант lo0rp сессии 3 поэтому не в счёт).

**Измерения на T4** (namedesign, ABBA, 2 раунда + холостой, --skip-pytest, если не сказано иное; первый шаг от `apply begin` / стена):
- сессия 1 (c1c51bf + профиль): base 11,11 / 48,74; профиль старой версии искажает apply (+2,3 с), поэтому только для состава; без прогрева загрузка модели 3,55 с против 5,72, шаг 0 9,0 с; один воркер ждёт прогрев 2,4 с на входе в train.
- сессия 2 (5419dc9): base 10,68 / 44,26; rows (FAST_ROWS+FAST_FINGERPRINT) 10,62 / 45,02 (сборка 1,48 против 1,59); early 10,79 / 45,01 (модель +0,69, сборка −0,46); **lo0** (LAUNCHER_OPT=0, NICE=10) **9,98** / 44,22, прогрев 4,99 с против 6,72; lo0+early+rows 10,41 / 44,61.
- сессия 3 (dd4beb3, с pytest): base 10,99 / 44,87; lo0 10,43 / 44,28; lo0 + старт прогрева в начале apply 10,25 / 43,98; lo0 + 1 воркер 10,80 / 44,61 (ожидание 0,61); вариант lo0rp не в счёт (ошибка флага). Pytest: 267 прошли, 8 упали — все в чужих тестах, те же, что в прогонах combined (offload ×2, recompute ×2, unpad, warmup/torchao, fused_proj память, int8 takeover); тесты этой ветки проходят.
- сессия 4 (dcafdd1 = ветка + combined 940a5a6, pinned-загрузка включена): base 10,70 / 44,47 (прогрев 6,97 с, ожидание на входе в train 0,64); **lo0 9,76 / 43,95** (прогрев 5,94 с, без ожидания); lo0 + старт в начале apply 10,17 / 44,25 (apply +1,15 с, модель −0,64); то же + порядок longest 10,00 / 44,05; без nice 10,04 / 44,23 (nice не важен); всё вместе со стартом в начале apply 9,88 / 44,03 (сборка 1,29 против 1,57, train→шаг 0,10 против 0,16: проверки P2P и entry points ушли из окна).
- бенч лаунчера на T4 (9752dbf, `tools/bench_launcher.py`, 20 000 запусков, лучшее из 5, по два процесса): -O3 19,86 / 19,49 мкс на запуск, -O0 19,09 / 18,84: у -O0 нет цены в цикле.
- сессия 5 (68a1025, с pytest: 330 прошли, 1 упал — прежний чужой `test_unpad::test_one_launch_each_way[True]`): base 10,10 / 46,87 (прогрев 6,65 с, ожидание 0,66); **lo0 9,20 / 45,84**; lo0 + порядок longest 9,32 / 45,74 (порядок отправил cublas в конец, где он встретил сборку тренера и пиннинг оффлоада: 0,95 с вместо 0,27; исправлено в 0c04921, cublas и step_ops остаются первыми); lo0 + порядок + rows + cheap + probes 9,08 / 45,38: сборка тренера 1,86 → 1,23 с, train→шаг 0,16 → 0,09, но загрузка модели +0,35 с (поток probes: скан entry points под конкуренцией 0,36 поток-с вместо 70 мс в главном потоке); в профиле токенизация теперь идёт под хвостом прогрева (главный поток 35 % CPU), прогрев кончается на +9,1 с.
- сессия 6 (68a1025, без порядка): base 11,54 / 47,28 (прогрев 7,59 с, ожидание 0,80); **lo0 10,27 / 45,86**; lo0 + FAST_ROWS + CHEAP_FINGERPRINT 9,73 / 45,73 (сборка 1,93 → 1,48 с); **+ EARLY_PROBES 9,54 / 44,59** (модель +0,16, сборка −0,16, train→шаг −0,07). Итого против base в той же сессии: первый шаг −2,0 с, стена −2,7 с (из них ~1 с шум цикла).
- не измерено: порядок с cublas/step_ops в начале (0c04921), остаётся выключенным.

**Включить:** `SOUP_SOLUTION_LAUNCHER_OPT=0` (главное: −0,9…−1,3 с до первого шага в каждой сессии на коде с pinned-загрузкой, цены в цикле нет; `SOUP_SOLUTION_LAUNCHER_NICE=10` нейтрален, можно с ним), `SOUP_SOLUTION_FAST_ROWS=1` и `SOUP_SOLUTION_CHEAP_FINGERPRINT=1` (−0,45…−0,55 с, строки побитово те же), `SOUP_SOLUTION_EARLY_PROBES=1` (−0,2 с в сессии 6). Оставить выключенными: `SOUP_SOLUTION_EARLY_ROWS` (проигрывает), `SOUP_SOLUTION_FAST_FINGERPRINT` (лишний при CHEAP), `SOUP_SOLUTION_WARMUP_ORDER` (не доказан), `SOUP_SOLUTION_WARMUP_START=apply_begin` (на коде с pinned-загрузкой хуже: apply +1,15 с). Включение: умолчания `_flag(...)` в `trainer_build.py` и `install_launcher_build` в `warmup.py` или переменные окружения в `apply()`. Для README: лаунчеры triton через его хук `knobs.build.impl` с -O0 (кубины те же), Soup-строки без доказуемо ненужного рендера и с повторным использованием одинакового вызова токенизации, детерминированный дешёвый отпечаток map (разрешено разъяснением о случайном), проверка P2P accelerate и `entry_points()` в потоке во время загрузки (как разрешённый кэш packages_distributions). Тесты: `solution/test_trainer_build.py` (10), `solution/test_launcher_build.py` (4) проходят локально и на T4; 79 тестов Soup проходят со всеми переключателями и без них.


## Ночь: полный прогон итоговых умолчаний `claude/night-combined` (c86b364)

**Состав умолчаний:** split-sum, пересчёт attention всех слоёв, in-place backward MLP, оффлоад входов слоёв, ранний оффлоад gate|up с префетчем в один тензор, очистка cuBLAS в начале обучения, префетч головы у финальной нормы, лимит стека CUDA 256 Б, загрузка модели через pinned-буферы (`load_pinned`). Выключены: int8 backward, моменты AdamW на хосте, пауза сборщика мусора, `empty_cache` по шагам, оффлоад attention вместо пересчёта.

**Полный прогон** (peagverr, soup-night-cfull2, `--mode full`): **все 10 проверок**. База 82,5 с / 14 345 MiB (быстрая база в этой сессии), мы **47,1 с / 449 MiB**: s = 1,75, **m = 31,9**; eval −0,013 %; кривая loss 0,01 % первый шаг, 0,05 % среднее; свои ядра 14,0 %; медиана шага 0,4346 с (горячая карта после базы); reserved 312 MiB. Pytest: 316 прошли, 1 упал (test_unpad: профилировщик в этой сессии не увидел ни одного ядра fmha даже за три попытки, проверка по диспетчеру проходит; профилировщику оставлена только верхняя граница).

**Жюри** (peagverr, soup-night-fjury, c86b364; строки 5000–5255 обучение / 15000–15255 eval, seed 99, base/solution/solution/base; самый длинный шаг 1 738 токенов): **все 10 проверок**. База 83,3 / 84,9 с (среднее 84,1), мы **45,7 / 46,0 с** (45,85): s = 1,83; пик 14 351 против **469 MiB**: m = 30,6; eval 2,06234 против 2,06242 (−0,004 %); разрывы против base_1: первый шаг 0,003 %, среднее 0,038 %, энтропия 0,08 %, точность 0,36–0,38 %, rewind 0,12 % (у base_2 0,00–0,06 %); свои ядра 13,9 %; 79 тестов Soup. Это четвёртый набор строк и seed (7, 13, 21, 99), на котором ночные умолчания проходят все 10 проверок.

**Эффект ночи в одной сессии** (peagverr, soup-night-total, b82a180, ABBA, 2 раунда + холостой; горячая сессия): ночные умолчания против той же ветки с выключенными ночными переключателями (`SPLIT_SUM=0, RECOMPUTE_ATTN=0, MLP_BWD_INPLACE=0, OFFLOAD_RESIDUAL=0, OFFLOAD_EARLY=0, STACK_LIMIT=0, LOAD_PINNED=0, HEAD_PREFETCH=embedding, CLEAR_CUBLAS_WS=0`, то есть конфигурация main плюс прогрев T15c): стена **48,67 против 46,56 с (+2,1 с, +4,5 %)**, шаги 3+ 30,51 против 28,24 с, пик nvidia-smi **449 против 1 311 MiB (−66 %)**, reserved 312 против 1 144, alloc 279 против 1 072; eval 2,04936–2,04939 против 2,04934–2,04938. По формуле баллов при s лучшей команды ~3 и нашей памяти лучшей: около −1,3 балла за скорость и +3,7 за память (против m 10,9 у main), плюс снижение баллов памяти соперников.

**Где теперь пик** (peagverr, soup-night-fdiag, c86b364, `--mode diag`; разбор трассы аллокатора шага 6 без torch: воспроизведение alloc/free от статики): пиковые alloc по шагам 195–275 MiB (худшие шаги 7 и 17), reserved максимум 312. На пике шага 6 (227,6 MiB) живы: статика 120,7 MiB (кольцо весов, LoRA и Adam, буфер распаковки, рабочие области cuBLAS), **78,8 MiB в трёх блоках внутри `_efficient_attention_backward` (unpad.py:256)**, 12,0 пересобранный выход attention (`_rebuild`), 5,5 + 4,5 буферы fused_proj, 1,6 голова. ~~То есть после раннего оффлоада пик сидит в backward attention~~ **Ошибка атрибуции**: мой реплей брал место выделения по адресу на конец трассы, а не на момент пика (адреса переиспользуются). Правильный реплей (исследовательский воркфлоу, тот же снимок): пик шага 6 в начале головы (227,6), backward attention 199,2; внутри оп backward attention 27,1 MiB, из них fp32-аккумулятор dQ 15,0 (по n_seq × max_seqlen). Перенесён переключатель `SOUP_SOLUTION_ATTN_SPLITS_KEY` из `claude/night-attn` (3dcf27a). Замер (peagverr, soup-night-splits, ABBA): 1 и 2 сплита дают тот же пик 449 MiB и alloc 279,1, время в шуме (46,73 / 47,11 против 46,75 с). Значит, максимум прогона (alloc 275–279 на самых длинных шагах 8 и 18) сидит не в backward attention шага 6; снимок худшего шага (soup-night-wdiag, шаг 8) снимается. Переключатель оставлен выключенным (0 = выбор ядра).

**Пик худшего шага** (peagverr, soup-night-wdiag, снимок шага 8: 274,9 MiB alloc; атрибуция по стеку выделения на момент пика): пик в начале головы, когда она выделяет буфер логитов чанка (21,3 MiB, `head_stream.py:428`); живы статика 120,8 MiB, **кольцо префетча головы 48 MiB** (2 × 24, выделено у финальной нормы), **gate|up последнего слоя 47 MiB** (копия на хост начата рано, но ожидания перед следующим MLP нет: слой последний, и в backward он нужен первым), остальное ~37 MiB. Рычаги: чанки головы по 4 096 строк (кольцо 2 × 12 и буфер логитов вдвое меньше, −33 MiB; T14 мерил такие чанки по времени в шуме) и `SOUP_SOLUTION_OFFLOAD_DRAIN_HEAD=1` (2c63b38: у финальной нормы дождаться копий и отпустить gate|up последнего слоя, backward возьмёт его с хоста; −47 MiB в этот момент ценой нескольких мс на шаг). Pytest на d73b8c7: 317 прошли, 0 упали. Замер (peagverr, soup-night-head, 2c63b38, ABBA): 8 192 строк (base) 449 MiB / alloc 279,1 / reserved 312, шаги 3+ 26,43 с; **4 096 строк 429 / 254,9 / 292, шаги 26,54** (в шуме); 2 048 строк 429 / 254,9 (стена в шуме сессии); 4 096 + `DRAIN_HEAD` 429 / 254,9, шаги 26,20. Голова больше не задаёт максимум, следующий момент 254,9 MiB alloc. **Включено: 4 096 строк по умолчанию** (head_stream.py `DEFAULT_ROWS`); `DRAIN_HEAD` выключен (ничего не добавляет). eval 2,04937–2,0494.

**Жюри с 4 096 строками** (peagverr, soup-night-gjury, c47f579; строки 12000–12255 / 22000–22255, seed 42; шаги до 1 662 токенов): **все 10 проверок**. База 89,3 / 87,7 с (88,5), мы 48,7 / 50,1 с (49,4; горячая сессия, медиана шага 0,47–0,50): s = 1,79; пик 14 409 против **451 MiB**: m = 31,9; eval −0,006 %; разрывы: первый шаг 0,009 %, среднее 0,03 %, энтропия 0,08 %, точность 0,29 %, rewind 0,16 %. В логе «pinned stack 2560 MiB in 10 slabs (6 prepinned)»: в этом коммите предварительная пиновка ещё 1 536 MiB, 4 слаба пинились во время шагов; с a169791 пинуется 2 560 MiB заранее.

**Полный прогон с 4 096 строками** (peagverr, soup-night-cfull3, c47f579, `--mode full`): **все 10 проверок**; база 86,5 с / 14 345 MiB, мы **47,4 с / 429 MiB**: s = 1,83, **m = 33,4**; eval −0,014 %; свои ядра 13,0 %; медиана шага 0,4605 (горячая карта); reserved 292; pytest 317 прошли, 0 упали.

**Итоговый полный прогон** (peagverr, soup-night-cfull4, ac9fc40 = все ночные умолчания вместе со стартом из `claude/night-trinit`): **все 10 проверок**; база 89,6 с / 14 345 MiB, мы **47,3 с / 431 MiB**: s = 1,89, **m = 33,3**; eval −0,014 %; кривая loss 0,01 % первый шаг, 0,05 % среднее; свои ядра 12,8 %; **pytest 331 прошли, 0 упали**. Фазы от старта процесса: apply 5,57 → 8,77 с, setup модели 9,30 → 12,81 (3,5 с, pinned-загрузка), начало обучения 14,55, **первый шаг 14,72 (9,15 с после начала apply; было 10,4–11,2)**, конец цикла 46,51 (цикл 31,3 с при медиане шага 0,468: горячая карта после базы). Оффлоад: «pinned stack 1792 MiB in 7 slabs (7 prepinned)», ни одного слаба во время шагов.

**Итоговое жюри на строках J1** (namedesign, soup-night-hjury, 2676242; строки 10000–10255 / 50000–50255, seed 7, как J1 в начале ночи): **все 10 проверок**. База 86,5 / 87,5 с (87,0), мы **46,8 / 47,8 с** (47,3): s = 1,84; пик 14 371 против **451 MiB**: m = 31,9; eval +0,000 %; разрывы: первый шаг 0,013 %, среднее 0,035 %, энтропия 0,07 %, точность 0,24 %, rewind 0,21 %. До/после на тех же строках: J1 с одним N1 (00:40) 47,4 с / 821 MiB, сейчас 47,3 с / 451 MiB: время то же, пик −45 %. Это шестой набор строк и seed (7 дважды, 13, 21, 42, 99), прошедший все 10 проверок.

**Финальная variants-сессия** (namedesign, soup-night-final, 2676242, ABBA, 2 раунда + холостой, pytest 331 прошли): итоговые умолчания 43,72 с / 431 MiB (шаги 3+ 26,13); без стартовых переключателей (`LAUNCHER_OPT=off, FAST_ROWS=0, CHEAP_FINGERPRINT=0`) 44,94 с / 431 (шаги 26,01): **старт даёт −1,2 с** при том же цикле; лимит стека 16 Б 43,66 с / **421 MiB** (шаги 26,02): −10 MiB без потерь по времени (в stack2 «замедление» было шумом). **Включено: лимит стека 16 Б по умолчанию.**

**Итоговый эффект ночи в одной сессии** (peagverr, soup-night-total2, d8b1ff3, ABBA): все ночные умолчания (тогда ещё стек 256 Б) против той же ветки со всеми ночными переключателями выключенными (конфигурация main + прогрев T15c): стена **46,03 против 44,29 с (+1,74 с, +3,9 %)**, пик **431 против 1 311 MiB (−67 %)**; первый шаг через 9,3 с после начала apply против 10,1–10,2 (старт быстрее main на 0,9 с), шаги 3+ 28,31 против 25,97 (пересчёт и оффлоад +2,3 с); eval 2,04936–2,04941 против 2,04935–2,04939.

**Подтверждающий полный прогон итогового коммита** (namedesign, soup-night-cfull5, a0cb72e: все ночные умолчания, стек 16 Б): **все 10 проверок**; база 77,5 с / 14 345 MiB (быстрая база в этой сессии), мы **42,5 с / 421 MiB**: s = 1,83, **m = 34,1**; eval −0,014 %; кривая loss 0,01 % / 0,05 %; свои ядра 14,7 %; медиана шага 0,4029; reserved 292; **pytest 331 прошли, 0 упали**.

**Подтверждающее жюри итогового коммита** (namedesign, soup-night-kjury, a0cb72e; строки 20000–20255 / 30000–30255, seed 13, как cjury в 02:00): **все 10 проверок**. База 83,9 / 85,6 с (84,7), мы **46,4 / 47,7 с** (47,05): s = 1,80; пик 14 345 против **459 MiB** (шаг в 1 733 токена): m = 31,3; eval −0,003 %; разрывы: первый шаг 0,041 %, среднее 0,04 %, энтропия 0,07 %, точность 0,36–0,38 %, rewind 0,16 %. На тех же строках в 02:00: 46,4 с / 579 MiB. Седьмой набор строк и seed с 10 из 10.

**Ещё одно жюри итогового коммита** (peagverr, soup-night-ljury, a0cb72e; строки 33000–33255 / 46000–46255, seed 123; шаги до 1 544 токенов): **все 10 проверок**. База 85,0 / 86,8 с (85,9), мы 48,4 / 48,75 с (48,6; горячая сессия, медиана шага 0,48): s = 1,77; пик 14 371 против **421 MiB**: m = 34,1; eval +0,008 %; разрывы: первый шаг 0,012 %, среднее 0,042 %, энтропия 0,06 %, точность 0,28–0,33 %, rewind 0,16 %.

**int8 поверх итоговых умолчаний** (peagverr, soup-night-fint8, b82a180, ABBA; `SOUP_SOLUTION_INT8_BWD=1` с компиляцией в потоке и фиксированной точкой переключения на 113-м backward MLP): стена 44,68 против 45,23 с (**−0,55 с**), шаги 3+ 26,47 против 26,99; по шагам: 0–4 +38, +7, +32, +47, +51 мс (поток компиляции), шаги 8+ в среднем −10,7 мс (медиана −15,8); пик 451 против 449 MiB; eval 2,04963 / 2,04969 против 2,04933–2,04943 (+0,013 %, допуск 0,5 %); свои ядра 18,9 %. Это готовый вариант на случай, если организаторы разрешат int8 для GEMM градиента по входу в backward; по умолчанию выключен.

## Ночь: копии загрузки модели через pinned-буферы (ветка `claude/night-load`, от `claude/night-combined` c1c51bf)

**Что сделано** (`solution/load_pinned.py`, `SOUP_SOLUTION_LOAD_PINNED=1`, по умолчанию выключено; подключено правкой `load_stream.py`: хук `spawn_materialize`, закрытие вместе с загрузкой, время цикла тензоров в его строке лога; `solution/__init__.py` не менялся). В цикле загрузки transformers (`convert_and_load_state_dict_in_model`, при квантовании на лету последовательный) заменён только шаг «тензор чекпоинта → GPU» и только в загрузках, которые ведёт load_stream (CUDA Qwen2ForCausalLM, bnb 4-bit на лету, одна карта, внутри setup Soup). Первый проход цикла записывает план: 337 срезов safetensors, идущих на GPU, в порядке второго прохода (эмбеддинг остаётся на пути load_stream, на хост). Со вторым проходом поток заполняет кольцо из 4 pinned-буферов по 26 MiB следующими тензорами; задание тензора выделяет GPU-тензор там же, где `.to()` (текущий поток, тот же размер), и копирует буфер `non_blocking`; событие после копии разрешает потоку перезаполнить буфер (без record_stream). Каст bf16→fp16: `SOUP_SOLUTION_LOAD_PINNED_CAST=gpu` (по умолчанию) — поток копирует сырые байты одним memmove, GPU кастует их на месте в самом тензоре (IEEE round-to-nearest-even, как CPU-каст у torch); `=cpu` — поток конвертирует тем же CPU-ядром, что `.to()` оригинала (блокирующая H2D-копия torch конвертирует на CPU, Copy.cu), биты те же по построению, но на T4 эта конверсия медленная. `SOUP_SOLUTION_LOAD_PINNED_WINDOW=W` (0 по умолчанию, до 2): W тензоров копируются заранее на боковом потоке, порядок событиями (пик загрузки +W тензоров; при W=0 выделения ровно как у оригинала). Квантование bnb (quantize_4bit на каждый вес) не тронуто. Любая неожиданность (тензор не по порядку, не default-поток, ошибка staging) — оригинальный путь.

**Тесты на T4** (pytest сессии soup-night-load v2): `solution/test_load_pinned.py` 42 из 42: GPU-каст против CPU-каста для всех 65 536 значений bf16 (NaN остаётся NaN); tiny Qwen2 (bf16 на диске, со значениями, которые fp16 переполняет и сбрасывает), окна 0/1/2 × оба каста: все упакованные веса, absmax, quant state, нормы, смещения и эмбеддинг побитово против обычной загрузки transformers и против того же setup без копий, пик загрузки при W=0 не выше оригинала; тот же шаг обучения побитово; настоящий Qwen2.5-1.5B: все 338 тензоров побитово против `_materialize_copy` и вся загрузка в setup побитово против обычной, оба каста; протокол на CPU с фейковым движком копий (мутации протокола ловятся). Весь pytest решения: 300 прошли, 8 упали — те же, что у c1c51bf без моей ветки (soup-night-ctest3: test_fused_proj, test_int8_bwd, test_offload ×2, test_recompute ×2, test_warmup/torchao; test_unpad[True] как в cfull). Локально 79 тестов Soup проходят с `SOUP_SOLUTION_LOAD_PINNED=1`.

**T4** (namedesign, soup-night-load v2, 7f9a017, ABBA, 2 раунда + холостой; стена / первый шаг от apply / setup модели / цикл тензоров / пик nvidia-smi):
- base: 46,87 / 11,22 / 5,33 / 3,91 с / 539 MiB;
- CPU-каст, W=1: 46,68 / 11,21 / 4,55 / 3,12 / 539 (поток конвертирует 1,9–2,1 с, цикл ждёт его 0,84–0,94 с);
- **GPU-каст, W=0: 46,58 / 11,10 / 3,67 / 2,24 / 539**; GPU-каст, W=1: 46,55 / 11,07 / 3,64 / 2,27 / 539 (memmove 1,0 с, цикл ждёт 0,16–0,21 с).
Сессия v1 (9423849, только CPU-каст) то же: цикл 3,90 → 3,21–3,24 с, setup 5,33 → 4,66 с, первый шаг 11,44 против 11,41–11,76. Вывод: setup −1,7 с, но первый шаг всего −0,1…−0,15 с: после setup главный поток ждёт прогрев (компиляция 16 ядер в потоке, 7,1–7,3 с от конца apply) 0,7–0,9 с; у base прогрев кончается как раз к началу обучения. Стена −0,3 с (в шуме сессии). Пик и шаги обучения без изменений.

**Что переключить:** `SOUP_SOLUTION_LOAD_PINNED=1` (по умолчанию каст GPU, W=0): бесплатно по памяти, побитово, −1,7 с setup; в стену это превращается, когда прогрев закончится раньше (работа по старту): тогда до −0,9 с к первому шагу. Оставшаяся CPU-работа загрузки: CPU-конверсия эмбеддинга в главном потоке (467 MB) и пиновка буферов load_stream (1,2–1,3 с в потоке).

## Ночь: память вне аллокатора, лимит стека CUDA (в `claude/night-combined`, включено)

**Диагностика** (`solution/bench_ctx.py`, peagverr, soup-night-ctx): на T4 память вне аллокатора torch (то, что nvidia-smi видит сверх `memory_reserved`): контекст 104,9 MiB, после первого GEMM cuBLAS 132,9 (+28: библиотека cuBLAS/cuBLASLt), дальше attention, bitsandbytes и наши ядра Triton ничего не добавляют. Лимиты контекста: стек 1 024 Б на поток (резерв 1 КиБ × 1 024 потока × 40 SM = 40 MiB), printf FIFO 1,25 MiB, куча malloc 8 MiB (не выделяется без device malloc). Лимит стека 512 Б: −20 MiB, 256 Б: −30 MiB; после этого attention forward/backward, распаковка NF4 и GEMM идут без ошибок и резерв не растёт обратно.

**Переключатель** `SOUP_SOLUTION_STACK_LIMIT` (`solution/cuda_limits.py`, по умолчанию **256**; ставится один раз в начале обучения после прогрева через `cudaDeviceSetLimit` той же библиотеки рантайма, что у torch; 0 оставляет как есть). По документации драйвера ядро, которому нужно больше стека, всё равно запустится: драйвер сам поднимет лимит, и поднятое значение останется. Тест `test_cuda_limits.py`.

**T4** (peagverr, soup-night-stack, 2e99a07, ABBA, 2 раунда + холостой; умолчания с ранним оффлоадом): стена / шаги 3+ / пик nvidia-smi: base 42,70 с / 24,38 / 479 MiB; стек 512 Б 42,82 / 24,56 / **459**; стек 256 Б 43,00 / 24,64 / **449**; reserved и alloc у всех 312 / 279,1; eval 2,04933–2,04939; лог: `CUDA stack limit 1024 -> 256 bytes per thread`. Разница по шагам в шуме сессии (два прогона base разошлись на 1,15 с). Итог: **−30 MiB (−6 %)**, m ≈ 31,9 на публичных данных.

Ниже 256 Б (peagverr, soup-night-stack2, f048391, ABBA): 64 Б 441 MiB, 16 Б 439 MiB (резерв почти весь снят: ни одно ядро не подняло лимит), но шаги 3+ 26,56 / 26,66 с против 25,95 у 256 Б; первый прогон 256 Б был заметно быстрее второго (25,39 против 26,50), так что это может быть шум. Оставлено 256 Б: проверено в двух сессиях без потерь по времени; ещё −8…10 MiB не стоят риска.


## Ночь: ранний оффлоад gate|up (ветка `claude/night-early`, влита в `claude/night-combined`, включена)

**Что сделано.** `SOUP_SOLUTION_OFFLOAD_EARLY` (offload.py, fused_proj.py; по умолчанию **1**): копия gate|up слоя на хост начинается сразу после GEMM gate|up (`offload_early`, хук упаковки после forward MLP берёт тот же дескриптор, копия одна), а каждый forward MLP сначала ждёт на GPU копии в полёте (`drain_inflight`): копия предыдущего слоя успевает пройти под SwiGLU, GEMM down и блоком attention следующего слоя, и его gate|up освобождается до того, как следующий слой выделит свой. С ним по умолчанию окно префетча backward в один тензор: распаковка gate|up слоя тянет его вход, распаковка входа (пост-attention норма) тянет gate|up предыдущего слоя, так что в backward MLP два gate|up больше не живут одновременно. Тест `test_xoffload.py::test_mlp_activation_copy_starts_early`. Попутно `SOUP_SOLUTION_GC_PAUSE=freeze|off` (сборщик мусора на время обучения, по умолчанию выключен, замер soup-night-gc идёт) и `solution/bench_ctx.py` (память вне аллокатора по стадиям, замер ещё не запускался).

**T4** (peagverr, soup-night-early, e1e31a6, ABBA, 2 раунда + холостой; горячая сессия, медиана шага ~0,5 с у всех): стена / шаги 3+ / пик nvidia-smi / reserved / alloc:
- комбинированные умолчания: 50,87 с / 32,17 / 539 / 372 / 298,7 MiB;
- + ранний оффлоад: 51,10 / 32,48 / **499** / 332 / 279,1;
- + префетч в один тензор: 51,15 / 32,63 / **479** / 312 / 279,1.
eval 2,04935–2,04941 у всех; оффлоадер: 3 584 тензора, 1 792 начаты рано, 0 по требованию, 0 ожиданий по потоку. Итог: **−60 MiB (−11 %) за +0,3 с** (в пределах шума горячей сессии); m при базе 14 345 MiB ≈ 29,9 против 26,6.

**Жюри с ранним оффлоадом** (peagverr, soup-night-ejury2, 299efca; строки 25000–25255 обучение / 35000–35255 eval, seed 21, base/solution/solution/base; шаги до 1 971 токена, пять шагов длиннее 1 427 публичных): **все 10 проверок**. База 106,0 / 97,1 с (очень горячая сессия, медиана шага базы 1,21 с), мы 52,5 / 53,1 с: s = 1,92; пик 14 347 против **541 MiB** (на самом длинном шаге этого набора; на публичных данных 479): m = 26,5; eval 2,04217 против 2,04184 (+0,016 %); разрывы против base_1: первый шаг 0,008 %, среднее 0,043 %, энтропия 0,13 %, точность 0,34 %, rewind 0,13 % (у base_2 0,00–0,16 %); свои ядра 12,0 %; 79 тестов Soup. Первая попытка (soup-night-ejury) не стартовала: eval-строки 40000+ публичные, `tools/jury_data.py` их отвергает.

**Сборщик мусора** (peagverr, soup-night-gc, 98476a1, ABBA, 2 раунда + холостой; умолчания уже с ранним оффлоадом): шаги 3+ base 28,08 с, `SOUP_SOLUTION_GC_PAUSE=freeze` 28,13, `off` 28,05: в шуме (startup.py и так замораживает объекты импорта в конце apply, а accelerate собирает мусор до цикла); стена у freeze/off на 1,1 с больше из-за `gc.unfreeze()` при выходе (полная сборка по 650 тыс. объектов), убрано в 299efca. Переключатель остаётся выключенным. Пик 479 MiB у всех (новые умолчания подтверждены второй сессией).

**Диагностика градиентов** (`solution/diag_recompute_test.py`, soup-night-diagrec): два прогона одного и того же пути (kept / kept) расходятся по градиенту входа слоя на ~3·10⁻⁶ относительной ошибки, по LoRA q на ~2·10⁻⁵ (dQ в backward attention складывается атомиками), LoRA v побитово; пересчёт (с проверкой и без, с удержанием выходов и без) и split-sum отличаются от kept на том же уровне. Ошибки в пересчёте нет; тесты, сравнивавшие градиенты двух прогонов через `torch.equal`, переведены на допуск 1e-4.

## Ночь: интеграционная ветка `claude/night-combined` (main + все ночные ветки, ночные умолчания включены)

**Состав.** main b098961 + `claude/night-startup` (T15c: прогрев только в потоке, WORKERS=2) + `claude/night-recompute` (N1) + `claude/night-xoffload` (N1b, N1c) + `claude/night-jury` (режим жюри T18) + `claude/night-int8` (N2, выключен). Умолчания (383289d): `SOUP_SOLUTION_SPLIT_SUM=1`, `SOUP_SOLUTION_RECOMPUTE_ATTN=1` (все слои), `SOUP_SOLUTION_MLP_BWD_INPLACE=1`, `SOUP_SOLUTION_OFFLOAD_RESIDUAL=1`, `SOUP_SOLUTION_CLEAR_CUBLAS_WS=1`, префетч головы у финальной нормы; int8, drain и оффлоад attention (N1c: на 18 мс за шаг медленнее пересчёта при той же памяти) выключены. Переключатели читаются в модулях решения; ночью `solution/__init__.py` не менялся, его изменение в ветке пришло из T15c (718f68e: процесс-помощник удалён, как требует ответ организаторов, прогрев только в потоке).

**Полный прогон** (namedesign, soup-night-cfull, 383289d, `--mode full`): **все 10 проверок**; база 93,3 с / 14 345 MiB, мы **47,1 с / 539 MiB**: s = 1,98, **m = 26,6**; eval −0,013 %, кривая loss 0,01 % первый шаг и 0,04 % среднее; свои ядра 13,4 %; медиана шага 0,4435 с (прогон решения идёт сразу после 93 с базы на горячей карте: в variants-сессиях та же конфигурация 0,383). Pytest: 253 прошли, 13 упали: 2 уже исправлены в 1fd8525 / 5088af1 (окна оффлоадера и кольцо головы следуют новым умолчаниям), 7 исправлены здесь (тест памяти пересчёта держал выходы и градиенты прошлых прогонов на GPU и сам держал сумму слоя; обёртка-счётчик в test_rmsnorm без именованных аргументов; число регистраций прогрева с ядром split; профилировщик T4 теряет ядро fmha, теперь лучшее из трёх попыток; доля ±1 у int8-распаковки против fp16-эталона 0,45 % при пороге 0,1 %), 2 прежних не наши (test_offload с `_sleep` по таймингу, test_warmup и torchao 0.10 на Kaggle).

**Прогон жюри** (namedesign, soup-night-cjury, bf8003d, строки 20000–20255 обучение / 30000–30255 eval, seed 13, base/solution/solution/base): **все 10 проверок**. База 91,0 / 84,5 с (среднее 87,7), мы **46,2 / 46,6 с** (46,4): s = 1,89; пик 14 345 против **579 MiB** в обоих прогонах: m = 24,8; eval 2,01766 против 2,01774 (−0,004 %); разрывы против base_1: loss первый шаг 0,041 %, среднее 0,04 %, энтропия 0,07 %, точность 0,33 %, rewind 0,17 % (допуски 1 % и 3 %; у второго прогона базы 0,00–0,09 %); самый длинный шаг 1 733 токена (в публичных данных 1 427).


## Ночь, N2: int8 для GEMM градиента по входу MLP в backward (ветка `claude/night-int8`)

**Шаг 1, бенч** (`solution/bench_int8.py`, peagverr): `torch._int_mm` на T4 работает через CUTLASS `i8816gemm` (нужен «TN»: второй операнд column-major). На формах backward при M = 832: gate|up ×1,88, down ×1,61, q|k|v ×1,17, o ×0,92; при M = 1 440: ×1,68 / ×1,61 / ×1,42 / ×1,21. Взяты только gate|up и down.

**Шаг 2, реализация** (`solution/int8_bwd.py`, `SOUP_SOLUTION_INT8_BWD=1`, по умолчанию выключено; подключение в `fused_proj._MLP.backward` для режима оффлоада/none): пять своих ядер (распаковка NF4 сразу в int8 W^T с масштабом на входной признак, квантование градиента по токенам, эпилог int32 слитый с backward SwiGLU и максимумом по токену, транспонирующее квантование dL/d(gate|up), эпилог int32 → fp32) и два `torch._int_mm`. int8-веса лежат в общем буфере распаковки (W^T gate|up в int8 ровно размера одного fp16-веса), int32-произведение dL/dhidden +18 КБ на токен временно. Тесты `solution/test_int8_bwd.py`.

**Численность на T4** (namedesign, soup-night-int8f, c6feced, `--mode full` с int8): **все 10 проверок**; кривая loss: первый шаг 0,01 %, среднее 0,07 % (у fp16 обычно 0,05 %); eval 2,04939 против базы 2,04966 (−0,013 %, ровно как у fp16-пути); метрики и rewind в допуске (проверка 5); 79 тестов Soup. То есть шум int8 в градиенте по входу MLP на исход обучения не виден.

**Скорость, первая версия:** прогон медленный (50,9 с, медиана шага 0,460 с): ядро распаковки NF4 → int8 читало по байту на элемент и занимало 79 мс на шаг (0,9 мс на вес вместо ~0,15); эпилог SwiGLU 17,8 мс на шаг (у примера 11,2), транспонирующее квантование 7,5. Исправлено в 86f3cbd (слова int32 по 8 кодов, как nf4.py).

**Скорость после исправлений** (ABBA, 2 раунда, шаги 3+ / медиана / стена / первый шаг от apply):
- soup-night-int8w (namedesign, 86f3cbd, 5 ядер): base 26,21 с / 0,412 / 45,94 / 11,15; int8 24,45 / 0,376 / 45,68 / 12,63 (горячая сессия: −1,76 с шагов, но +1,5 с старта: пять новых ядер компилируются при прогреве);
- микробенч `bench_int8_mlp` (fwd+bwd одного MLP 1.5B): 832 токена fp16 8,36 мс → int8 6,82; 1 440 токенов 11,92 → 10,97; эпилог SwiGLU по сетке тайлов лучший (32, 64, 10): 0,44 / 0,83 мс;
- soup-night-int8x (namedesign, 6cd5d13, 3 ядра, мелкие шаги на операциях torch): base 23,92 / 0,377 / 42,66 / 10,66; int8 23,69 / 0,365 / 43,91 / 11,94 (быстрая сессия: −0,23 с шагов, +1,3 с старта);
- soup-night-comb (ветка `claude/night-combined` с T15c, прогрев потоком): base 25,07 / 0,395 / 43,29 / 10,16; int8 23,89 / 0,369 / 43,13 / 11,11; N1-all 26,31 / 0,414 / 44,74 / 10,27; N1-all + int8 25,52 / 0,389 / 44,93 / 11,24.
- soup-night-int8lazy (peagverr, 8725017: компиляция трёх ядер в потоке с первого backward MLP, первые 112 backward MLP (4 шага) в fp16, на 113-м ожидание потока, точка переключения не зависит от таймингов; ABBA, 2 раунда + холостой): стена base 43,87 / int8 44,10 с; первый шаг от apply 10,96 / **10,95** (компиляция со старта ушла); шаги 3+ 24,60 / 24,76; по шагам int8 − base: шаги 0–4 +57, +10, +38, +50, +120 мс (поток компиляции конкурирует за GIL, на шаге 4 ожидание потока), шаги 8+ в среднем **−0,6 мс** (медиана −1,8); eval 2,0498 / 2,04963 против 2,04939 / 2,04942 (+0,02 %); пик 1 317 против 1 311.
Вывод: int8 стабильно ускоряет шаги (−12…−36 мс медианы, сильнее в «горячих» сессиях, где T4 сильнее сбрасывает частоту), численно чисто, но компиляция трёх своих ядер при прогреве отодвигает первый шаг на ~1 с, и выигрыш по стене −0,2…+1,2 с в зависимости от сессии. Пик +24 MiB (int32 dL/dhidden). **Оставить выключенным**: (1) вопрос организаторам не задан; (2) компиляция спрятана (8725017), но выигрыш шагов нестабилен между сессиями (−19 мс/шаг в горячей soup-night-comb, −0,6 мс в soup-night-int8lazy), а поток компиляции стоит ~0,3 с на шагах 0–4; итог по стене от −1,2 до +0,2 с.

## Ночь, N1: пересчёт блока attention и split-sum (ветка `claude/night-recompute`)

**Что сделано** (всё за переключателями, по умолчанию выключено):
- `SOUP_SOLUTION_SPLIT_SUM=1` (rmsnorm.py): пост-attention норма хранит для backward не fp32-сумму `residual + attn_out`, а её слагаемые: residual (вход слоя, его и так держит входная норма) и half-выход attention; backward пересобирает сумму в новом своём ядре `rms_norm_rows_bwd_split` той же одной fp32-операцией, что forward (побитово та же сумма). −2 байта на элемент на слой.
- `SOUP_SOLUTION_RECOMPUTE_ATTN=1|auto` (fused_proj.py, unpad.py): в пересчитываемом слое attention не хранит q, сжатые k/v, выход и lse, а пост-attention норма не хранит ни сумму, ни выход o_proj; в backward норма просит выход o_proj у объекта пересчёта (`_AttnRecompute`), и тот один раз прогоняет q|k|v GEMM + addmm адаптеров + RoPE, forward ядра attention и o_proj GEMM из того, что хранится и так (x16 из входной нормы, h, half-адаптеры, таблицы RoPE, cu_seqlens), на тех же операндах (тот же алгоритм cuBLAS, forward ядра attention без атомиков). `auto`: столько слоёв от 0, сколько нужно, чтобы память attention шага не превышала память шага в `SOUP_SOLUTION_RECOMPUTE_TOKENS` токенов (640) без пересчёта.
- `SOUP_SOLUTION_MLP_BWD_INPLACE=1` (fused_proj.py): градиент SwiGLU пишется поверх загруженного gate|up (оффлоадер отдаёт свежую копию и больше её не держит): −36 КБ на токен в начале backward слоя.
- Тесты `solution/test_recompute.py`: побитовая проверка всех пересобранных тензоров (x16, q, k, v, выход, lse по записанным строкам, выход o_proj) на крошечной модели и на слое 1.5B, градиенты против хранящего пути, байты на токен, RNG не трогается, политика auto; in-place MLP побитово.

**Замеры на T4.**
- Диагностика (namedesign, b9d2fd9, recompute=1 + split + inplace, снимок худшего шага 40, ~1 440 токенов): пик alloc 584 MiB против 1 072 у main (сессия b098961), reserved 634 против 1 144, **nvidia-smi 813 против 1 323 MiB**; eval loss 2,0493 (main 2,04935). Состав пика: входы слоёв 228 MiB (27 × 8,44), gate|up текущего и предыдущего слоя 98, кольцо головы 48, hidden 25, статика 130.
- Variants (peagverr, 57881e0, 2 раунда, без ABBA; второй раунд, оба прогрето): шаги 3+ base 26,20 с, split 26,37 (в шуме), auto 27,53 (+21 мс/шаг), all 28,42 (+35 мс/шаг); пики nvidia-smi 1 311 / 1 211 / 865 / 821 MiB; eval 2,04937–2,04939 у всех; проверки 3, 4, 9 проходят во всех прогонах (свои ядра 13,3–14,7 %). Первый раунд: all 28,05 против split 27,07 (+16 мс/шаг): пересчёт стоит 16–35 мс на шаг, ~0,75 мс GPU на слой плюс запуски CPU.
- Pytest на T4 (55 новых и старых тестов решения): 255 прошли; упали 2 моих теста (сравнение lse захватывало невыровненный хвост буфера, который ядро не пишет; тест памяти не учитывал, что выход слоя держит сумму): исправлено в ac9d7d1; 3 прежних падения не мои (test_offload record_stream, test_warmup torchao на Kaggle).

- **N3, прогон в стиле жюри** (namedesign, soup-night-jury, ветка `claude/night-jury` = эта ветка + режим жюри T18; recompute=1 + split + inplace; строки 10000–10255 обучение / 50000–50255 eval, seed 7, base/solution/solution/base): **все 10 проверок пройдены**. База 96,2 / 86,5 с (среднее 91,3), мы 46,5 / 48,3 с (47,4), s = 1,927; пик 14 371 против 821 MiB в обоих прогонах, m = 17,50; eval 2,06075 против 2,06077 (+0,001 %); кривая loss: первый шаг 0,01 %, среднее 0,04–0,05 % (допуск 1 % и 3 %, запас > 60×); метрики, rewind, токены совпадают (проверка 5); свои ядра 13,4 %; 79 тестов Soup (в ячейке проверок, без ночных переключателей). Самый длинный батч этого набора до 1 878 токенов (в публичных данных 1 427).

**Вывод и что переключить:** по формуле баллов память выгоднее времени: auto (865 MiB, +1,3 с) даёт m 16,6 против 10,9 и около +3,7 балла за память против −1 за скорость; all (821 MiB, +2,2 с) примерно так же. Рекомендация: `SOUP_SOLUTION_SPLIT_SUM=1`, `SOUP_SOLUTION_RECOMPUTE_ATTN=auto` (или 1 вместе с оффлоадом входов слоёв, N1b), `SOUP_SOLUTION_MLP_BWD_INPLACE=1` в `apply()` после прогона жюри (N3 запущен на namedesign: soup-night-jury). Дальше по памяти: оффлоад входов слоёв (N1b, ветка `claude/night-xoffload`), префетч головы у финальной нормы (−48 MiB на пике последнего слоя).

## Ночь, N1b: оффлоад входов слоёв (ветка `claude/night-xoffload`, поверх N1)

**Что сделано** (за переключателями, по умолчанию выключено):
- `SOUP_SOLUTION_OFFLOAD_RESIDUAL=1` (offload.py, rmsnorm.py): fp32-вход каждого слоя (сумма, которую входная норма держит для backward, 6 КБ на токен на слой) уходит на хост тем же оффлоадером, что gate|up: rmsnorm метит тензор по адресу (`offload.mark_residual`), хук упаковки забирает метку (у Python-функции сохранённые тензоры упаковываются после её forward, поэтому метка, а не контекст вокруг `save_for_backward`: первая версия не оффлоадила ничего); окна оффлоадера на 1 больше; вход остаётся на GPU после первой распаковки до последнего читателя в backward того же слоя (пост-attention норма берёт residual из узла входной нормы, без второй ссылки).
- `SOUP_SOLUTION_OFFLOAD_DRAIN_LAST=1`: перед MLP последнего слоя дождаться копий в полёте. `SOUP_SOLUTION_CLEAR_CUBLAS_WS=1`: в начале обучения освободить рабочие области cuBLAS (поток прогрева держит свою 8 MiB).

**T4** (peagverr, soup-night-xoff2, ba811a4, ABBA, 2 раунда + холостой; все варианты с split + recompute=1 + in-place): шаги 3+ / пик nvidia-smi / reserved / alloc:
- N1-all: 26,65 с / 801 / 634 / 584 MiB;
- + оффлоад входов: 27,20 с / **581** / 414 / 356 MiB;
- + drain последнего MLP: 27,30 с / 581 / 414 / 356 (пик теперь в MLP любого слоя, drain не помогает);
- + префетч головы у финальной нормы (`SOUP_SOLUTION_HEAD_PREFETCH=norm`): 27,49 с / **541** / 374 / 308 MiB.
Оффлоадер: 3 584 тензора за прогон, из них 1 792 входа слоёв, 0 по требованию, все остальные из префетча. Диагностика полной конфигурации (+ очистка cuBLAS, снимок шага 40, ~1 440 токенов): пик alloc **299 MiB** (main 1 072), reserved 352; состав пика: статика 121 (два слота весов 46, LoRA и Adam 26, буфер распаковки 26, cuBLAS 2 × 8), gate|up текущего и предыдущего слоя 98 (копия на хост ещё в полёте), hidden 25, суммы нормы 17. Цена по времени: +0,55 с за оффлоад входов и +0,3 с за префетч у нормы к N1-all.

**Что переключить:** `SOUP_SOLUTION_OFFLOAD_RESIDUAL=1` и `SOUP_SOLUTION_HEAD_PREFETCH=norm` вместе с N1 (m ≈ 26,5 против 15,3 у лидера); drain не нужен.

## 31 октября, 00:35 по Астане: чекпоинт на ночь

**T18 жюри** (`claude/t18-jury`, 060f366, T13): `--mode jury` в раннере; J1 (строки 10000–10255 для обучения и 50000–50255 для eval, seed 7, base/solution/solution/base) прошёл все 10 проверок с запасом не меньше 12× от допусков; J3 на самых крайних строках роняет базу по памяти (OOM 14 719 MiB), значит у жюри такого набора нет. **T19b** (`claude/t19b-gemm-tune`, 7ed1b27, T14): три GEMM-переключателя (`SOUP_SOLUTION_QKV_SPLITK`, `_O_LT`, `_DOWN_LT`), выключены до замера `soup-t14v`. **T15c** (`claude/t15c-startup`, 3f5b14c): процесс-помощник удалён, поток прогрева покрывает все ядра, первый шаг через 9,73 с после apply при `WORKERS=2` (10,46 по умолчанию).


main = f11e94b (+ docs): **41,0 с против 87,2 у базы (s = 2,126), пик 1 311 MiB (m = 10,94), все 10 проверок**, кривая loss 0,05 %, eval −0,013 %. Разложение: пролог харнесса 5,1 с, наше до обучения 10,5 с (apply 3,5, setup 5,2, тренер 1,8), цикл 23,8 с (шаг 0,360, первый 0,47), выход 1,0 с. Лидеры в чате заявляют 29 с / 937 MiB и 19 с / 613 MiB (второе ниже физического пола при той же математике и с eval на 1,3 % лучше базы, что харнесс помечает для жюри). Наши резервы по приоритету: граница шага (−15…−20 мс), распаковка на боковом потоке (−10…−25 мс), чанкование длинных шагов (−10…−40 мс на длинных), загрузчик (−1,5 с), прогрев в потоке (−1…−2 с), пересчёт attention (память), int8 backward (stretch). Ночная автономная сессия работает по `docs/NIGHT_BRIEF.md`; локальные подчаты остановлены с пушем своих веток (T12 `claude/t20-*` загрузчик, T14 `claude/t19b-*` GEMM, T13 `claude/t13*` режим жюри, T15 `claude/t15c-startup` прогрев в потоке), незабранные прогоны Kaggle перечислены в брифе.


## 30 октября, 21:20 по Астане: main после волны 1 (ba8c1b3)

Полный прогон (namedesign, обычная по скорости сессия для пролога, медленная для цикла: цикл базы 62,4 с против 55,5 обычно): **база 89,3 с, мы 45,4 с, s = 1,967; пик 14 345 против 1 371 MiB, m = 10,46; все 10 проверок**; кривая loss расходится с базой в среднем на 0,05 % (было 0,28 %: сработал фикс переполнения градиента головы на первом шаге), eval loss −0,013 %. Фазы: apply 5,19 → 9,22 с (4,0 с), setup модели до 15,36, train begin 16,35, первый шаг 0,64 с (было 1,39), цикл 27,5 с при медиане шага 0,403 с (в сессии нормальной скорости около 0,36). Своих ядер 13,6 %. Резерв torch 1 204 MiB, nvidia-smi 1 371: разница 167 MiB это контекст CUDA. alloc_watch: 0 повторов аллокаций, 0 освобождений сегментов, пул ни разу не сжимался. Стриминг весов: ожидания 5,3 мс (forward) и 7,8 мс (backward) на шаг; голова ждала копий 4,4 мс на шаг.

Диагностический прогон того же коммита (peagverr, медленная сессия, с накладными диагностики): пик nvidia-smi 1 383 MiB, резерв 1 204; статика после setup 151 MiB; пик шага от 420 (354 токена) до 1 091 MiB (1 427 токенов); **пик setup 1 146 MiB теперь выше худшего шага обучения** (все NF4-слои и эмбеддинг лежат на GPU между загрузкой и prepare_model_for_kbit_training): это следующая цель по памяти (T14). Состав пика шага 6 (767 токенов, 659 MiB): статика 151, fp32-пары резидуала 247 (55 × 4,5 MiB, теперь 62 % памяти слоя и пол без оффлоада), сохранённое attention 150 (q 63, выход с lse 65, k и v 21), кольцо головы 48 + 15, один gate|up в полёте 26. Трасса шагов 8–9: все GEMM в выровненных ядрах turing_*; копировальные движки HtoD 240 мс и DtoH 92 мс на шаг (веса 1,35 ГБ, голова 0,45 ГБ, оффлоад gate|up 0,73 ГБ в каждую сторону), в этой сессии стриминг весов ждал 31,8 мс на шаг в backward: PCIe стал спорным ресурсом, T13 перемеряет оффлоад против рекомпьюта. Простой GPU 12 % (ожидания потоков, Python в наших Function, конец шага ~8 мс). Прогрев: 14 ядер из дискового кэша, 4 компилируются в процессе (ядра головы и rebuild, их нет в списке процесса-помощника, T15 чинит).

Оценка пола на этой задаче: пролог харнесса ~6 с (не наш), наш старт ≥ 8 с, шаг ≥ 0,30 с GPU (матмулы 150–170 мс, распаковка NF4 32, attention 28, SwiGLU 17, голова ~38 с PCIe, поэлементные ~25) → цикл ≥ 19 с → всего ≈ 33–36 с, s ≈ 2,5. Память: пол слоя ≈ 14 MiB на 767 токенов (fp32-пара 9), худший шаг ≈ 1 000 MiB alloc, nvidia-smi ≈ 1 200 → m ≈ 12, при условии что пик setup уйдёт ниже.

Волна 2 (выдана 21:25): T12 часть 2 T16 (пересчёт x16 входа MLP, аудит запусков на слой), T13 (бюджет PCIe: оффлоад против рекомпьюта при потоковой голове, слоты стриминга, split gate|up по умолчанию, пиновка до шага 0), T14 (слои на хост при загрузке: пик setup ниже 500 MiB; префетч головы и приоритет потоков), T15 (остаток старта: apply 4 с, setup 4,6, тренер 1,05, шаг 0; помощник компилирует все зарегистрированные ядра).


## 30 октября, 20:15 по Астане: измерения нового оркестратора (коммит d9f60bb)

Что добавлено: `solution/diag.py` (только при `SOUP_SOLUTION_DIAG=1`: память на каждой фазе, проба PCIe, снимок аллокатора шага 6, chrome-трасса шагов 8–9), `tools/kaggle_run.py --mode diag`, `tools/mem_snapshot_report.py` (что живо в пике шага), `tools/trace_gaps_report.py` (простои GPU и что делал CPU). Прогоны идут из WSL (conda-окружение `soup`, torch 2.11, triton 3.6.0), токены обоих аккаунтов (namedesign, peagverr) у Абзала.

Чистый замер HEAD (full): база 89,5 с, мы 57,3 с (s = 1,563), пик 2 385 MiB (m = 6,02), цикл 30,4 с, медиана шага 0,416 с, первый шаг 1,39 с при полном прогреве (остаток: ленивая загрузка CUDA-модулей, cuBLAS). **Проверка 9 упала**: пик nvidia-smi 2 385 MiB меньше 0,97 от резерва torch 2 592 MiB. Причина: аллокатор при нехватке свободного блока ждёт события `record_stream` (копии оффлоада на боковых потоках), освобождает и размапливает сегменты, резерв «пилит», опрос раз в 100 мс пропускает вершину. Это надо убрать (без record_stream, ровный резерв с начала обучения), иначе любой прогон жюри может дать ноль.

Диагностический прогон (`--only solution`, с диагностикой 70 с):
- PCIe pinned: host→GPU 12,0 ГБ/с, GPU→host 12,8 ГБ/с; стриминг весов 11–12 ГБ/с, 2,1 мс на слой.
- nvidia-smi = alloc torch + ~160 MiB (контекст и библиотеки): при старте обучения alloc 510 / reserved 592 / nvidia-smi 745.
- Статика после setup 596 MiB: эмбеддинг fp16 445, буфер распакованных весов fused_proj 52,5, два слота стриминга 46, LoRA с Adam 48, норма и прочее ~5. Пик setup 2 513 MiB (fp32-копия эмбеддинга при prepare_model_for_kbit_training и перед embed16), ниже пика обучения только случайно: при пике обучения ниже 2,5 ГБ setup станет пиком, его тоже надо убрать (не давать peft кастовать эмбеддинг в fp32).
- Пик шага пропорционален числу токенов: от 982 MiB (354 токена) до 1 874 MiB (1 427 токенов); шаг 6 (767 токенов): 1 353 = 596 статики + 757 шага. Пик наступает на голове, при первом чанке логитов (148 MiB). Сохранённое на слой при 767 токенах: вход слоя fp32 4,5 + сумма после attention fp32 4,5 + x16 входа QKV 2,25 + q 2,25 + k и v после repeat_kv по 2,25 + выход attention с lse 2,3 ≈ 20 MiB, то есть 26 КБ на токен на слой; gate|up последнего слоя 26 ещё в полёте.
- Трасса шагов 8–9 (1 363 и 658 токенов): GPU простаивает 25 %: два стопа по 119 мс внутри `torch.empty` в forward MLP (аллокатор синхронизируется с событиями record_stream оффлоада), ~10 мс на конец шага (.item, amp scale, optimizer). Копировальные движки: HtoD 193 мс и DtoH 88 мс на шаг (веса 1,35 ГБ/шаг плюс gate|up 0,73 ГБ туда и обратно). GEMM: ядра cutlass `align1/align2` 232 мс/шаг против выровненных turing_* ~100 мс/шаг (N = 826 токенов не кратно 8: кратно только в 6 шагах из 64); nf4 32 мс; attention bwd 23, fwd 5; SwiGLU примера 17; поэлементные ~27.
- Импорты (`-X importtime`, apply 7,6–8,2 с): `trl.import_utils` 2,38 с self и `transformers.utils.import_utils` 2,31 с self (проверки наличия пакетов через importlib.metadata на огромном site-packages Kaggle); peft → bitsandbytes 1,4 с (из них torch._inductor 0,76); подмодули transformers, которые тянут наши патчи: generation.candidate_generator → sklearn 0,87, masking_utils → torch._dynamo 0,5, activations → hub_kernels 0,62; trl.trainer.base_trainer → wandb 0,46; datasets 0,36.

Решения по измерениям: оффлоад gate|up заменить на рекомпьют (ноль PCIe, без стопов аллокатора); паддинг N до кратного 16; attention одним varlen-вызовом без repeat_kv; стриминг lm_head с онлайн-накоплением dh (−445 статики, −150 чанк); диета старта (importlib.metadata, лишние импорты, прогрев в подпроцессе, крошечный разогрев CUDA-модулей). Оценка пола: шаг ~250 мс, старт ~16 с, итого ~33 с; память ~1,2–1,5 ГБ при худшем батче.


## Таблица прогонов

| Дата, время | Коммит | Режим | База, с | Мы, с | s | База, MiB | Мы, MiB | m | eval loss, % | Проверки |
|---|---|---|---|---|---|---|---|---|---|---|
| 31.10 00:30 | f11e94b | full (main с T17), namedesign | 87,2 | 41,0 | 2,126 | 14 345 | 1 311 | 10,94 | -0,013 | 10 из 10; цикл 23,8 с, шаг 0,360, первый 0,47; apply 3,5, train begin +15,58 (пролог 5,07); reserved 1 144; своих ядер 15,5 % |
| 30.10 23:45 | 88e1afd | full (main с T14b), peagverr | 90,7 | 44,5 | 2,038 | 14 345 | 1 309 | 10,96 | -0,012 | 10 из 10; цикл 25,8 с, шаг 0,379, первый 0,50; apply 3,7, train begin +17,26 (пролог 6,09); reserved 1 142; своих ядер 15,6 % |
| 30.10 22:25 | 75658f3 | full (main с T13b, T15b), namedesign | 90,7 | 44,9 | 2,021 | 14 345 | 1 351 | 10,62 | -0,013 | 10 из 10; сессия обычной скорости (цикл базы 55,5); наш цикл 25,7 с, шаг 0,380, первый шаг 0,51; apply 3,69 с, train begin +17,15 с (пролог 5,55); reserved 1 184; своих ядер 15,1 %; голова ждала 8,5 мс/шаг, веса 5,5 + 8,1 |
| 30.10 21:17 | ba8c1b3 | full (main после волны 1), namedesign | 89,3 | 45,4 | 1,967 | 14 345 | 1 371 | 10,46 | -0,013 | 10 из 10; кривая loss 0,05 %; цикл 27,5 с (медленная сессия: база 62,4), шаг 0,403, первый шаг 0,64, train begin +16,35 с; reserved 1 204 |
| 30.10 21:15 | ba8c1b3 | diag (--only solution), peagverr | | 58,3 | | | 1 383 | | | с диагностикой; статика 151, худший шаг 1 091 alloc, пик setup 1 146; HtoD 240 мс/шаг |
| 30.10 20:10 | 4744723 | full (main с T12), аккаунт peagverr | 97,8 | 52,4 | 1,866 | 14 345 | 2 151 | 6,67 | -0,011 | 10 из 10; медленная сессия Kaggle (цикл базы 64,2 с против 55,5 обычно), наш цикл 28,3 с, шаг 0,406 с; reserved 1 984; своих ядер 11,9 % |
| 30.10 19:50 | 4744723 | solution (T12: паддинг 16 + varlen attention) | | | | | 1 991 | | | шаг 0,329 с, цикл 23,6 с, alloc 1 688 / reserved 1 824, своих ядер 15,3 %; пролог харнесса в этой сессии 14,9 с |
| 30.10 20:05 | d9f60bb | full (HEAD: фиксы оффлоада и прогрева) | 89,5 | 57,3 | 1,563 | 14 345 | 2 385 | 6,02 | -0,017 | 9 из 10: №9 упала (nvidia-smi 2 385 < 0,97 × reserved 2 592); шаг 0,416 с; оффлоад gate|up работает (1 792 тензора), первый шаг 1,39 с при полном прогреве |
| 30.10 20:05 | d9f60bb | diag (--only solution) | | 70,1 | | | 2 191 | | | с диагностикой: PCIe 12,0/12,8 ГБ/с, статика 596 MiB, пик шага 982–1 874 MiB по токенам, см. раздел выше |
| 30.10 19:20 | f37bf98 | full (C: без слитых проекций) | 85,8 | не дочитан | | 14 345 | | | | шаг 0,424 с при базе 0,873 с (0,486), против 0,443–0,479 со слитыми проекциями: слитые проекции дают около 3–7% шага; torch alloc 2 513 MiB |
| 30.10 19:05 | f37bf98 | full (B: без стриминга весов) | 93,0 | 55,8 | 1,668 | 14 345 | 4 025 | 3,564 | -0,017 | 10 из 10; шаг 0,443 от базового против 0,479 со стримингом: стриминг стоит около 7% шага за минус 610 MiB; загрузка модели 4,9 с против 3,1 без фонового прогрева |
| 30.10 18:35 | f37bf98 | full (A: всё включено) | 90,3 | 56,2 | 1,606 | 14 345 | 3 415 | 4,201 | -0,017 | 10 из 10; стриминг весов минус 599 MiB; оффлоад gate/up не сработал (0 тензоров: фильтр grad_fn против слитого MLP); прогрев скомпилировал 7 ядер за загрузку модели, nf4 нет; первые два шага 2,4 с вместо 6,7 |
| 30.10 18:40 | f37bf98 | full (повтор A из-за ошибки --env) | 94,2 | 58,9 | 1,599 | 14 345 | 3 415 | 4,201 | -0,018 | 10 из 10; медленная сессия Kaggle (база 94 с) |
| 30.10 17:00 | 982f5f3 | full, repeats 2 | 87,5 и 84,4 | 58,8 и 59,1 | 1,459 | 14 345 | 4 341 | 3,304 | -0,006 | 10 из 10; разброс Kaggle: шаг базы 0,89 и 0,97 с, наш 0,44 с против 0,40 с в прошлой сессии |
| 30.10 16:40 | 5d091b4 | full | 81,8 | 54,1 | 1,512 | 14 345 | 4 341 | 3,304 | -0,007 | 10 из 10, свои ядра 12,2% (NF4 8,5%); шаг 0,398 с, но первые два шага 6,7 с (компиляция пяти ядер) |
| 30.10 16:30 | d649c7c | full | 88,2 | 56,8 | 1,552 | 14 345 | 4 341 | 3,304 | -0,001 | 9 из 10 (№4: 0,6%); чанк логитов 512 |
| 30.10 16:27 | 1274b24 | profile | 86,3 | 55,4 | 1,557 | 14 345 | 4 481 | 3,201 | -0,003 | 9 из 10, как выше |
| 30.10 16:17 | 1274b24 | full | 86,9 | 54,3 | 1,600 | 14 345 | 4 481 | 3,201 | -0,001 | 9 из 10: провал №4, свои ядра 1,2% времени шагов (нужно 5%), пример SwiGLU не считается |



---

# Приложение E. solution/__init__.py (точка входа, порядок патчей, переключатели)

```python
"""Our solution: Triton kernels and patches that make the fixed `soup train` run on a Kaggle T4
faster and lighter without changing what it trains.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every patch is a class-level method replacement looked
up at call time, guarded so that anything that is not the real CUDA Qwen2 training step (Soup's
regression tests with fake modules, eval, generation, CPU) takes the original code path.

Modules:
- fused_ce.py: lm_head + cross-entropy + TRL metrics without the (B, T, V) logits (own kernel);
- trainer_patch.py: TRL's compute_loss logging the same metrics from the fused head, and Soup's
  rewind recorder fed through its own hook for steps without logits;
- unpad.py: the layers run on the packed real tokens of the batch, attention per row;
- mlp16.py: the MLP's gate/up/SwiGLU in half storage (uses the kit's example SwiGLU kernel, which
  does not count as ours);
- embed16.py: the tied embedding / lm_head weight back in fp16 after Soup's setup;
- rmsnorm.py: RMSNorm and residual-add + RMSNorm in one Triton kernel each way (own kernels);
- nf4.py: NF4 dequantization in one Triton kernel, used by bitsandbytes' MatMul4Bit (own kernel);
- rope.py: rotary embedding of q and k in one Triton kernel each way (own kernel);
- fused_proj.py: q+k+v in one GEMM with LoRA by addmm, attention in half, gate+up in one GEMM;
- offload.py: the MLP's saved gate|up to pinned host memory on a side stream with events (prepinned
  slabs), or recomputed in the backward (SOUP_SOLUTION_MLP_MEMORY=recompute: +50 ms per step for
  24 MiB on the T4, so off by default);
- alloc_watch.py: the caching allocator's retries and segment frees, logged once per run;
- head_stream.py: the tied embedding / lm_head weight in pinned host memory for the whole run,
  the head computed in one streamed pass over the vocabulary (own kernels); switched on from
  fused_ce.patch_qwen2_causal_lm (SOUP_SOLUTION_HEAD_STREAM=0 turns it off);
- weight_stream.py: the layers' 4-bit weights in pinned host memory, a ring of K layers on the GPU;
- warmup.py: the kernels compiled by a background thread while Soup loads the model (triton's own
  hash started with apply());
- startup.py: apply()'s imports without work the run does not use (trl's second metadata scan,
  scikit-learn, wandb, bitsandbytes' Inductor), the imported objects frozen for the garbage
  collector, the tokenizer hashed fast by datasets in Soup's setup; a first-step phase mark; a
  profile of the trainer build (diagnostics);
- phases.py: wall-clock marks of the run's phases in the log;
- diag.py: diagnostics of a run (SOUP_SOLUTION_DIAG=1 only): memory at the marks, a PCIe probe,
  an allocation snapshot of one step, a chrome trace of two steps.

Switches (environment, for experiments): SOUP_SOLUTION_CE_CHUNK (rows of logits per chunk, default
512), SOUP_SOLUTION_UNPAD=0, SOUP_SOLUTION_MLP16=0, SOUP_SOLUTION_EMBED16=0, SOUP_SOLUTION_RMSNORM=0,
SOUP_SOLUTION_NF4=0, SOUP_SOLUTION_ROPE=0, SOUP_SOLUTION_FUSED_PROJ=0, SOUP_SOLUTION_MLP_MEMORY=
offload|recompute|none, SOUP_SOLUTION_WEIGHT_STREAM=0 (SOUP_SOLUTION_WEIGHT_SLOTS=2), SOUP_SOLUTION_WARMUP=0,
SOUP_SOLUTION_PAD_TOKENS=16 (0 turns the alignment tail off), SOUP_SOLUTION_VARLEN_ATTN=0,
SOUP_SOLUTION_REBUILD_X16=0, SOUP_SOLUTION_CARRY_RESIDUAL=0, SOUP_SOLUTION_HEAD_STREAM=0
(SOUP_SOLUTION_HEAD_ROWS, SOUP_SOLUTION_HEAD_SLOTS, SOUP_SOLUTION_HEAD_PREFETCH=norm|embedding|head),
SOUP_SOLUTION_IMPORT_DIET=0, SOUP_SOLUTION_DATASET_HASH=0; diagnostics SOUP_SOLUTION_PROFILE_STARTUP=1.
"""

import os


def _on(name):
    return os.environ.get(name, "1") not in ("0", "false", "no")


def apply():
    from .phases import mark, patch_soup_phases

    mark("apply begin")
    if _on("SOUP_SOLUTION_WARMUP"):
        from .warmup import start_early

        start_early()  # triton's own hash now, next to the imports, not next to the model load
    if _on("SOUP_SOLUTION_IMPORT_DIET"):
        from .startup import prepare_imports

        prepare_imports()  # trl without a second metadata scan, without sklearn and wandb
    from .embed16 import patch_soup_setup
    from .fused_ce import patch_qwen2_causal_lm
    from .mlp16 import patch_qwen2_mlp_half
    from .trainer_patch import patch_sft_trainer
    from .unpad import patch_qwen2_attention

    chunk = int(os.environ.get("SOUP_SOLUTION_CE_CHUNK", "512"))
    patch_qwen2_causal_lm(chunk=chunk, unpad_enabled=_on("SOUP_SOLUTION_UNPAD"))
    patch_qwen2_attention()
    patch_sft_trainer()
    if _on("SOUP_SOLUTION_MLP16"):
        patch_qwen2_mlp_half()
    else:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
    if _on("SOUP_SOLUTION_EMBED16"):
        patch_soup_setup()
    if _on("SOUP_SOLUTION_RMSNORM"):
        from .rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm

        patch_qwen2_rmsnorm()
        patch_qwen2_decoder_layer()
    if _on("SOUP_SOLUTION_NF4"):
        from .nf4 import patch_bnb_matmul4bit

        patch_bnb_matmul4bit()
    if _on("SOUP_SOLUTION_ROPE"):
        from .rope import patch_qwen2_rope

        patch_qwen2_rope()
    if _on("SOUP_SOLUTION_FUSED_PROJ"):
        from .fused_proj import patch_qwen2_attention_fused, patch_qwen2_mlp_fused

        patch_qwen2_attention_fused()  # after unpad's attention patch and rope: it falls back to them
        patch_qwen2_mlp_fused()  # after mlp16's patch: it falls back to it
    from .offload import patch_mlp_memory

    patch_mlp_memory()  # SOUP_SOLUTION_MLP_MEMORY=offload (default, event-based, prepinned)|recompute|none
    from .alloc_watch import patch_soup_alloc_watch

    patch_soup_alloc_watch()  # one log line per run: allocator retries, segment frees, reserved vs peak
    if _on("SOUP_SOLUTION_WEIGHT_STREAM"):
        from .weight_stream import patch_weight_stream

        patch_weight_stream(slots=int(os.environ.get("SOUP_SOLUTION_WEIGHT_SLOTS", "2")))
    patch_soup_phases()
    from .startup import (patch_soup_dataset_hashing, patch_soup_first_step_mark,
                          patch_soup_startup_profile)

    patch_soup_first_step_mark()
    if _on("SOUP_SOLUTION_DATASET_HASH"):
        patch_soup_dataset_hashing()  # datasets hashes the tokenizer in 0.09 s instead of 1 s
    if os.environ.get("SOUP_SOLUTION_PROFILE_STARTUP", "0") not in ("0", "false", "no", ""):
        patch_soup_startup_profile()  # off by default: cProfile of the trainer build, see startup.py
    if os.environ.get("SOUP_SOLUTION_DIAG", "0") not in ("0", "false", "no", ""):
        from .diag import patch_soup_diag

        patch_soup_diag()  # off by default: memory marks, PCIe probe, one step's allocation history, a trace
    if _on("SOUP_SOLUTION_WARMUP"):
        from .warmup import patch_soup_warmup, start_background_warmup, watch_compiles

        watch_compiles()  # logs any kernel compiled or loaded outside the warm-up
        patch_soup_warmup(start_in_setup=False)
        start_background_warmup()  # compiles every registered kernel in a background thread
    if _on("SOUP_SOLUTION_IMPORT_DIET"):
        from .startup import freeze_imported_objects

        freeze_imported_objects()  # later full garbage collections skip the imported objects
    mark("apply end")

```


---

# Приложение F. Докстринги модулей решения (обрезаны до 1 500 знаков каждый)

### `solution\__init__.py` (137 lines)

Our solution: Triton kernels and patches that make the fixed `soup train` run on a Kaggle T4
faster and lighter without changing what it trains.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every patch is a class-level method replacement looked
up at call time, guarded so that anything that is not the real CUDA Qwen2 training step (Soup's
regression tests with fake modules, eval, generation, CPU) takes the original code path.

Modules:
- fused_ce.py: lm_head + cross-entropy + TRL metrics without the (B, T, V) logits (own kernel);
- trainer_patch.py: TRL's compute_loss logging the same metrics from the fused head, and Soup's
  rewind recorder fed through its own hook for steps without logits;
- unpad.py: the layers run on the packed real tokens of the batch, attention per row;
- mlp16.py: the MLP's gate/up/SwiGLU in half storage (uses the kit's example SwiGLU kernel, which
  does not count as ours);
- embed16.py: the tied embedding / lm_head weight back in fp16 after Soup's setup;
- rmsnorm.py: RMSNorm and residual-add + RMSNorm in one Triton kernel each way (own kernels);
- nf4.py: NF4 dequantization in one Triton kernel, used by bitsandbytes' MatMul4Bit (own kernel);
- rope.py: rotary embedding of q and k in one Triton kernel each way (own kernel);
- fused_proj.py: q+k+v in one GEMM with LoRA by addmm, attention in half, gate+up in one GEMM;
- offload.py: the MLP's saved […]

### `solution\adam_host.py` (140 lines)

The AdamW moments of the training kept in pinned host memory between optimizer steps.

Soup's optimizer is torch's AdamW (`adamw_torch`, the foreach implementation) over the LoRA
parameters: 2.18 M fp32 values, so `exp_avg` and `exp_avg_sq` hold 17.4 MB on the GPU for the whole
run, through the forward and backward peaks where nothing reads them. Here a pre-step hook brings
them to the GPU right before each `optimizer.step()` and a post-step hook copies them back to pinned
host memory: they are on the GPU only during the step, after the backward, when the activations are
gone. All moments live in one flat fp32 buffer on each side (the state tensors are views into it),
so each way is one copy on the current stream (~1.5 ms at 12 GB/s) and the GPU side is one large
block the allocator can hand to the activations of the next step (a first version kept 112 separate
tensors: 112 small copies each way, +20 ms per step, and their small-pool blocks stayed reserved, so
nvidia-smi saw nothing). The step reads and writes the same values: parameters, moments and every
later step are bit for bit what they were.

SOUP_SOLUTION_ADAM_HOST=1 turns it on (off by default until measured); only torch.optim.AdamW whose
parameters are all on one CUDA device and whose moments are plain contiguous fp32 tensors is
touched; any other optimizer, or a state someone else replaced, runs as it is. Installed around
Soup's training by offload.py's train wrapper; when the training returns the moments are put […]

### `solution\alloc_watch.py` (177 lines)

The caching allocator's reserved pool during training: watched, and never left below its peak.

Check 9 asks for nvidia-smi's peak (sampled every 100 ms by the harness) >= 0.97 x
torch.cuda.max_memory_reserved(). nvidia-smi sees the pool plus the CUDA context (about 160 MiB on
the T4), so the check holds as long as the pool keeps every peak mapped for at least one sample. It
fails when the pool reaches a peak and gives it back within 100 ms: torch.cuda.empty_cache() after a
transient (setup), or the allocator's own release path (an allocation finds no free block and no
room to map, waits for every pending stream event, unmaps cached segments and retries). The
allocator setting (expandable_segments:True) is the harness's and stays.

This module installs a trainer callback on a real CUDA trainer (Soup's tests with fakes run the
original train). It reads host-side counters only, no synchronization:
- it logs the reserved MiB at train begin, after step 1, its maximum and at the end, how many steps
  ended with the pool smaller than its peak so far, and the allocator's retries (the release path)
  and segment unmaps during training;
- with the guard (SOUP_SOLUTION_ALLOC_GUARD=1, off by default), whenever the pool is more than
  GUARD_SLACK_MIB below torch.cuda.max_memory_reserved() (at train begin, after a setup transient;
  after any step; at train end), it maps that memory again: chunks are allocated until the pool is back at its peak, then
  freed, and the freed blocks stay […]

### `solution\cuda_limits.py` (82 lines)

The CUDA context's per-thread stack reservation, set lower at the start of training.

A context reserves local memory for the stack of every thread that can be resident at once:
`cudaLimitStackSize` (1 KiB by default) x 1,024 threads per SM x 40 SMs = 40 MiB on the T4, counted
by nvidia-smi outside torch's allocator. The kernels of this run do not use that much: cuBLAS and
cuBLASLt GEMMs, the memory-efficient attention forward and backward, torch's elementwise kernels and
our Triton kernels have no recursion and spill little. Measured on the T4 (solution/bench_ctx.py,
soup-night-ctx): device memory outside the allocator 132.9 MiB after a cuBLAS GEMM, 112.9 MiB with
the limit at 512 B, 102.9 MiB at 256 B, still 102.9 MiB after the attention forward/backward, the
NF4 dequantization and the GEMM ran again (none of them grew it back).

SOUP_SOLUTION_STACK_LIMIT: bytes per thread (16 by default), set once when training starts
(offload.py's train wrapper, after the warm-up); 0 leaves the context alone. T4, training
(soup-night-stack, ABBA): nvidia-smi peak 479 MiB at the default 1 KiB, 459 at 512 B, 449 at 256 B,
steps 3+ within the session's noise, eval loss unchanged; the peak never grew back during training.
16 B (soup-night-final, ABBA, against 256 B in the same session): 431 -> 421 MiB, wall 43.66 against
43.72 s: the reservation is nearly gone (40 SMs x 1,024 threads x 16 B) and no kernel raises it. The call goes to the CUDA runtime library
torch's wheels ship (the same […]

### `solution\embed16.py` (103 lines)

The tied embedding / lm_head weight stored in fp16 instead of fp32 (lossless on this run).

The checkpoint is fp16 and Soup loads it in fp16; peft's prepare_model_for_kbit_training then casts
every non-4-bit parameter to fp32, the 151,936 x 1536 tied embedding included: 933 MB that hold fp16
values. The base pays for it twice more: autocast makes an fp16 copy of the lm_head weight every
step (467 MB), and the embedding lookup is an fp32 gather.

After Soup has built the model (SFTTrainerWrapper._setup_transformers), the parameter goes back to
fp16 when the round trip is exact element for element (it is for an fp16 checkpoint; a bf16-loaded
model on an Ampere card is left alone). A forward hook casts the embedding output to fp32, so the
residual stream and the rotary tables keep the base's dtype; the fused head (fused_ce.py) uses the
fp16 weight directly. The parameter is frozen, so nothing else sees the change.

With head_stream.py on (the default), the weight is in pinned host memory before peft's cast and
never comes back to the GPU: this module then leaves it alone and only serves as the fallback.

### `solution\fused_ce.py` (296 lines)

Fused output head: lm_head matmul + cross-entropy + TRL metrics without the (B, T, V) logits.

Base Soup materializes fp16 logits for every position, upcasts them to fp32 for the loss, keeps the
log-softmax for the backward, and reads them three more times (TRL entropy, TRL token accuracy, Soup's
per-row rewind losses). With a 151,936-token vocabulary that is about 4 GB at the peak and several
passes over the memory bus per step.

Here the head runs in chunks of rows. For each chunk the logits are an fp16 GEMM (cuBLAS, tensor
cores), then one Triton kernel per row does two passes: pass 1 is an online reduction that yields the
log-sum-exp, the entropy, the argmax and the target logit (so the loss); pass 2, only for rows that
carry a label, overwrites the chunk in place with `(softmax - onehot) * grad_scale`, the gradient of
the loss with respect to the logits. The gradient with respect to the hidden states, `dL/dh = g @ W`,
is then a second GEMM on the labelled rows only, computed right away; the autograd backward only
rescales it. No tensor of shape (rows, V) survives the chunk.

Rows are reordered so that labelled rows come first: a chunk made of prompt or padding rows does
pass 1 only (TRL's entropy is averaged over every real position, so those rows still need the
reduction) and no gradient GEMM.

The per-token loss, entropy and correctness flags are handed to the trainer (see trainer_patch.py),
which logs TRL's `entropy`, `mean_token_accuracy` and `num_tokens` with TRL's […]

### `solution\fused_proj.py` (1219 lines)

Fused projections of a Qwen2 layer: q+k+v in one GEMM with the LoRA adapters added by addmm, the
attention in half from end to end, gate+up in one GEMM.

In the base, every 4-bit linear layer is its own chain: bitsandbytes casts the fp32 input to half,
rebuilds the weight, runs the GEMM and casts the result back to fp32; peft clones that result, runs
dropout on the fp32 input, autocast casts the dropped input and both adapter weights to half, two
small GEMMs, a multiply by `scaling` and an fp32 add. With unpad, rope and nf4 in place that is still
about 40 kernel launches per attention forward and more in the backward, and q, k, v live in fp32
until SDPA casts them back to half.

Attention here (inside `unpad.varlen`, the packed training step):
- the normed input is cast to half once (x16);
- the q, k and v weights are dequantized by `nf4.dequantize_nf4` into row slices of one shared
  (2048, 1536) buffer and one addmm with the concatenated bias gives qkv (N, 2048) in half;
- LoRA of q and v: the four adapter matrices go to half in one multi-tensor copy; dropout is drawn
  once for both adapters on x16 expanded to (2, N, D) (two independent masks, as the base has one
  dropout call per adapter), one bmm gives both h = dropout(x16) @ A.T, and
  `addmm_(qkv_slice, h, B.T, alpha=scaling)` adds the adapter output into the q and v columns of
  qkv with one rounding;
- the rotary embedding of q and k is one launch of rope.py's kernel (fp32 math, half out);
- q, k, v leave the […]

### `solution\head_stream.py` (915 lines)

The tied embedding / lm_head weight kept in pinned host memory for the whole run, and the output
head computed in one streamed pass over the vocabulary (own Triton kernels).

The 151,936 x 1536 weight of Qwen2.5-1.5B (445 MiB in fp16) was the largest static tensor of the
run, and the fused head's logits chunk (512 rows x V, 148 MiB) set the step's peak. Here neither
exists on the GPU: the weight lives in page-locked host memory and goes through the GPU once per
step in chunks of R vocabulary rows (4,096 by default: 12 MiB) on a ring of K slots (2 by default).

Setup (peft's `prepare_model_for_kbit_training` casts every fp16 parameter to fp32, the embedding
included: 933 MiB next to the fp16 copy, the setup's peak). Right before it is called inside Soup's
`SFTTrainerWrapper._setup_transformers`, the fp16 weight is copied to pinned host memory and the
Parameter (the same object for `embed_tokens.weight` and `lm_head.weight`) gets a placeholder as its
`.data`: one fp32 element on the GPU expanded to (V, D), 4 bytes. peft sees an fp32 parameter and
leaves it alone; `model.dtype`, `.device`, `.shape` and the parameter count stay what they are in the
base after the cast (fp32 on the GPU). Nothing reads the placeholder's values: the embedding and the
lm_head modules get instance-level `forward`s that read the host copy, and the fused head
(fused_ce.py) dispatches here when it is handed this Parameter. A `copy.deepcopy` of the Parameter
(peft's modules_to_save copies a module) […]

### `solution\int8_bwd.py` (289 lines)

int8 tensor cores for the MLP's input-gradient GEMMs in the backward (night task N2,
SOUP_SOLUTION_INT8_BWD=1, off by default: the forward is untouched, only the noise of the gradient
that flows back through the frozen MLP weights changes).

On the T4 an int8 GEMM (IMMA, `torch._int_mm` -> cuBLASLt -> CUTLASS i8816) runs the backward's two
big MLP products 1.6-1.9x faster than fp16 at the step's token counts (solution/bench_int8.py,
docs/RESEARCH.md): it does half the bytes and draws less power per operation, so the card throttles
less. The MLP backward of fused_proj.py becomes, with M the step's tokens (padded to 8):

1. grad (M, D) half -> per-token absmax int8 (torch ops: a small pass): g_q, s_g;
2. W_down (D, I) NF4 -> W_down^T (I, D) int8 with one scale per input feature, the column maxima
   of the weight (`nf4_int8_t_kernel`, the NF4 decode of nf4.py, then round(v * 127 / colmax));
3. acc_h (I, M) int32 = W_down^T_q @ g_q^T (`torch._int_mm`, "TN");
4. `int8_swiglu_bwd_kernel`: dL/dhidden = acc_h * s_w[i] * s_g[m] in fp32, the SwiGLU derivative
   with the saved (gate|up), dL/d(gate|up) written over (gate|up) in half, and every token's absmax
   of it per group of features (no atomics);
5. dL/d(gate|up) (2I, M) -> (M, 2I) int8 with the per-token scales, the max over the groups
   (`quant_t_int8_kernel`);
6. W_gate, W_up (I, D) NF4 -> [W_gate^T | W_up^T] (D, 2I) int8, one scale per input feature shared
   by both (the contraction runs over gate and up together);
7. […]

### `solution\load_pinned.py` (434 lines)

The checkpoint -> GPU copies of the model load go through pinned host buffers that a thread fills
ahead of transformers' loader (Soup's setup only, as part of load_stream.py's loads).

Measured on the T4 (session soup-t12): transformers' tensor loop
(`core_model_loading.convert_and_load_state_dict_in_model`) takes 3.2-3.5 s, about 1.75 s of it in
`_materialize_copy`'s `tensor.to(cuda, float16)` of each checkpoint tensor. With on-the-fly 4-bit
quantization the loop is sequential: its first pass only creates one deferred job per checkpoint
tensor (`spawn_materialize`), its second pass calls them one by one, and each `.to()` converts the
bf16 slice of the safetensors mmap to fp16 on the CPU into a fresh pageable tensor (a blocking copy
converts on the CPU: aten/src/ATen/native/cuda/Copy.cu, copy_kernel_cuda), then copies that to the
GPU from pageable memory and waits, while the GPU and the copy engine idle in between.

Here, for the loads load_stream.py handles (a CUDA Qwen2ForCausalLM quantized to 4 bits on the fly
by bitsandbytes on one GPU, inside Soup's `_setup_transformers`):
- `spawn_materialize` records every checkpoint slice that goes to this GPU, in the order the second
  pass will ask for them (the order the first pass creates their jobs);
- when the second pass begins, a thread fills a ring of K pinned host buffers
  (SOUP_SOLUTION_LOAD_PINNED_SLOTS, default 4, each the size of the largest tensor: 26 MiB for
  Qwen2.5-1.5B) with the next tensors, from the same […]

### `solution\load_stream.py` (363 lines)

The model's big weights go to pinned host memory while transformers loads them (Soup's setup
only), so that the setup never holds them all on the GPU.

Measured on the T4 (T14 diag, 30 Oct): between the load and peft's kbit preparation the GPU held all
28 layers' 4-bit weights (599 MiB) and the fp16 embedding (445 MiB): 1,100 MiB allocated, 1,176 MiB
reserved, a setup peak of 1,146 MiB, which becomes the run's peak once the training peak falls below
it. transformers' `caching_allocator_warmup` also reserves the model's whole size in one allocation
before the first weight is read.

With on-the-fly bitsandbytes quantization transformers 5.16 loads sequentially in the main thread
(`core_model_loading.convert_and_load_state_dict_in_model`): each checkpoint tensor is read to its
device (`spawn_materialize`, run when its converter runs), quantized there for a 4-bit linear layer
(`Bnb4bitQuantize` -> `Params4bit.to` -> `bnb.functional.quantize_4bit`), and set into its module
(`set_param_for_module`), in the natural order of the keys: the embedding, layer 0 .. 27, the final
norm. Here, inside Soup's `SFTTrainerWrapper._setup_transformers` and for a CUDA Qwen2ForCausalLM
loaded with 4-bit on-the-fly quantization on one GPU:

- `caching_allocator_warmup` is skipped: it reserves 1.1 GB for weights that no longer stay;
- the input embedding is read to the host (`get_device` answers "cpu" for its key), copied into
  pinned memory by head_stream.py and set as its 4-byte placeholder […]

### `solution\mlp16.py` (51 lines)

The MLP's big intermediates in fp16 instead of fp32 (lossless) and one cast instead of two.

In the base the residual stream is fp32 (prepare_model_for_kbit_training casts the norm weights to
fp32, and `fp32 weight * fp16 tensor` promotes), so the normed input of the MLP is fp32. bitsandbytes'
Linear4bit casts it to fp16 for the GEMM and casts the fp16 result back to fp32: `gate` and `up` are
fp32 tensors (2048 x 8960 x 4 bytes = 73 MB each on this task) that hold fp16 values, and SiLU and
the product run and save in fp32: three 73 MB tensors per layer stay alive for the backward, about
6 GB over 28 layers.

Here the input is cast to half once, the three projections run on half in and half out (no casts
inside bitsandbytes), the SwiGLU is the kit's example kernel (fp32 math, half storage), and only the
down projection's output goes back to the residual dtype. What is saved per layer: `gate` and `up`
in half, 73 MB instead of 219 MB. The values are the same as the base's: the GEMM outputs were
half before the base upcast them, and the base casts the fp32 product to half for the down GEMM.

### `solution\nf4.py` (258 lines)

NF4 weight dequantization in one Triton kernel, and bitsandbytes' 4-bit matmul running on it.

Every 4-bit linear layer of the base (7 per block, 28 blocks) rebuilds its fp16 weight on every forward
and every backward. With double quant (Soup's default), bitsandbytes 0.50.2 does it in three launches
plus two allocations: the 8-bit absmax codes are expanded to fp32 (`dequantize_blockwise`), the mean
offset is added (a separate torch add), then the 4-bit kernel writes the (N, K) weight. On a T4 the
training calls take this path: `gemm_4bit` sends M > 128 tokens to `_dequant_linear_fallback`
(dequantize, then `F.linear` in cuBLAS), and `MatMul4Bit.backward` always dequantizes.

Here one kernel reads the packed codes, the 8-bit absmax codes and the second-level scales, and writes
the fp16 weight in one pass. The layout it decodes (bitsandbytes 0.50.2, `quantize_4bit` with
`compress_statistics=True`):
- packed weight: uint8, (N * K + 1) // 2 bytes; byte i holds element 2i in the high nibble and
  element 2i + 1 in the low nibble of the row-major (N, K) weight;
- blocks of 64 elements (32 bytes): absmax of block b is `code2[q[b]] * absmax2[b // 256] + offset`,
  with q the uint8 `quant_state.absmax`, code2 = `state2.code` (256 fp32), absmax2 = `state2.absmax`
  (fp32, one per 256 blocks), offset = `quant_state.offset` (fp32 scalar);
- element value = `nf4[code] * absmax` in fp32, rounded to fp16, nf4 = `quant_state.code` (16 fp32).
The rounding steps are bitsandbytes' own (product […]

### `solution\offload.py` (763 lines)

The MLP's saved gate/up out of the GPU's peak: asynchronous offload to pinned host memory, or
recomputation in the backward. The rules allow both ("recomputation or offloading in your own code
(its time counts)"); Soup's own switches (activation_offloading, gradient_checkpointing) stay as the
task sets them.

With mlp16.py every layer keeps two fp16 tensors for the backward, gate and up (N x 8960, 14.8 MB
each at the 826 tokens of an average step after unpad.py), about 0.83 GB over 28 layers; with
fused_proj.py it is one, (gate|up) as (17920, N padded to 8). They are all alive at the end of the
forward, where the fused head adds its chunk of logits: the step's peak.

SOUP_SOLUTION_MLP_MEMORY picks one (patch_mlp_memory):

- recompute: with fused_proj.py's MLP, its autograd function saves only its half input x16 (N x
  1536, 2.25 MB at 767 tokens instead of 26 MB of (gate|up)) and its backward runs the (gate|up)
  GEMM again on the same operands, with the weights it rebuilds for the input gradient anyway (one
  GEMM more per layer, no dequantization more, bit-identical (gate|up)). Nothing crosses PCIe. On the
  T4 (main 4744723, diag runs of one commit) it costs ~50 ms of GPU per step (steps 3+: 24.86 s
  against 22.52 s with the offload) and does not lower the peak (1979 against 2003 MiB): the
  offload keeps one or two (gate|up) on the GPU, recompute keeps 28 x16. Worth it only when PCIe is
  saturated. Without the fused MLP: Qwen2MLP.forward saves only its fp16 input and […]

### `solution\phases.py` (70 lines)

Wall-clock marks of the phases of a run, printed to the training log.

The score is the wall time of the whole `soup train` command, and only part of it is the training
loop: imports, model load, trainer setup and the adapter save are the rest. These marks say where
our run spends that time (the harness records only the loop and `apply()`).

### `solution\rmsnorm.py` (642 lines)

Qwen2RMSNorm in one Triton pass each way, and the decoder layer's `residual + attn_out` fused into
the post-attention norm.

Qwen2RMSNorm.forward in eager PyTorch is five kernels per call: pow, mean, add + rsqrt, the row
scale and the weight multiply, each reading and writing the whole (rows, 1536) fp32 tensor. With 57
norms per forward (two per layer and the final one) and their backward graphs that is a lot of
memory traffic for a few FLOPs. Here a norm is one kernel that reads the rows once and writes the
output once; it keeps one fp32 rstd per row for the backward (eager keeps rsqrt's (rows, 1) output
and the fp32 input, the same amount). The backward is one kernel as well: dx in fp32 by the
closed-form RMSNorm gradient, and dweight only when the weight trains (on this task the norm weights
are frozen, so the reduction over rows is skipped entirely).

In the decoder layer the residual add that comes right before `post_attention_layernorm` is fused
into the same kernel: it reads the residual (fp32) and the attention output (fp16 out of
bitsandbytes' o_proj under autocast), writes the new residual and the normed rows in one pass, and
its backward writes the gradient of both addends at once.

Inside the packed training step (unpad.varlen_model_forward) the layer-end `residual + mlp_out` is
fused the same way, into the next layer's input norm (and the final norm): the layers hand the two
addends on unsummed. And `norm_source` / `rebuild` let a consumer of a norm's output […]

### `solution\rope.py` (195 lines)

Rotary position embedding for the Qwen2 attention: q and k in one Triton launch each way.

transformers' `apply_rotary_pos_emb` computes `q * cos + rotate_half(q) * sin` and the same for k:
per tensor two slices, a negation, a concatenation, two products and a sum, each one a kernel that
reads and writes the whole tensor, and as many again in the backward. With q (1, 12, 826, 128) fp32
that is about 5 MB moved per op, 28 layers, every step.

Here one launch reads q, k, cos and sin once and writes the rotated q and k. With x1, x2 the two
halves of a head vector and c1, c2, s1, s2 the matching halves of cos and sin:

    forward:   y1 = x1 * c1 - x2 * s1        y2 = x2 * c2 + x1 * s2
    backward:  g1 = d1 * c1 + d2 * s2        g2 = d2 * c2 - d1 * s1

(the backward is cos * dy plus the transpose of rotate_half, cat(x2, -x1), applied to sin * dy).
The two halves of cos and sin are read separately, so the kernel does not rely on them being equal.
Math in fp32. The drop-in returns the dtype the original would (`q * cos` type promotion: fp16 q
with fp32 tables gives fp32). `rotary_qk(..., out_dtype=torch.float16)` returns half directly for
an attention that feeds half q and k to SDPA: the same values as the original's fp32 result rounded
to half by autocast, rounded once, without the fp32 tensors and the casts.

Layout: q and k arrive as transposed views of the projections, (B, H, N, D) with the heads of one
token next to each other in memory. The kernel takes any strides over B, […]

### `solution\startup.py` (452 lines)

The time before the first training step: imports, the first CUDA work, and where the rest goes.

Imports (`prepare_imports`, first thing in apply())
---------------------------------------------------
apply() imports what the patches need: transformers' model and generation modules, trl's
SFTTrainer (and through it peft, bitsandbytes, datasets). Soup imports the same modules later in a
run without our package, so their cost moves, it does not add. Five parts of it are work that
this run does not use, and `prepare_imports` skips them (Kaggle T4, `python -X importtime` of
apply(), commits d9f60bb and ba8c1b3; SOUP_SOLUTION_IMPORT_DIET=0 switches all five off):

- 2.38 s: trl computes `importlib.metadata.packages_distributions()` when imported
  (trl/import_utils.py:25), a read of the metadata of every installed distribution, about a
  thousand on Kaggle. transformers computed the same mapping in this process a few seconds before
  (transformers/utils/import_utils.py:47, in the harness's `import transformers`), and since then
  nothing was installed (the rules forbid it; the harness compares `pip freeze` before and after)
  and only the repository root, which holds no distribution, was added to sys.path. While trl is
  imported, `packages_distributions` returns a copy of transformers' mapping. trl uses it only to
  find the version of liger_kernel and vllm (`_is_package_available(..., return_version=True)`).
- 0.84 s: two transformers modules import scikit-learn (and scipy) […]

### `solution\startup_profile.py` (462 lines)

Diagnostics of the time before the first training step: SOUP_SOLUTION_STARTUP_PROFILE=1 (off by
default; nothing in this file runs otherwise).

What it prints to the training log (lines start with `[solution] startup-profile`):
- at every phase mark (phases.mark), the wall time since `apply begin` and, since the previous mark,
  the CPU time of the main thread and of the rest of the process (warm-up workers, triton's hash,
  pinning threads, the tokenizer's and torch's native threads), and how late the sampler's ticks ran;
- extra marks from the end of Soup's model setup to the end of the first step (`install_marks`):
  the tokenization of the training rows, TRL's SFTTrainer and transformers' Trainer __init__, the end
  of Soup's setup, the Trainer's train() and its inner loop up to on_train_begin, the first step;
- the os.stat cost of each sys.path entry, and the import system's stat calls per sys.path entry
  (count, time) from `apply begin` to the first step;
- a sampling profile from `apply begin` to the end of the first step: a daemon thread reads
  sys._current_frames() every INTERVAL s and charges the time since its previous tick to the Python
  stack of the main thread and of every other thread. At the end of training it prints, per window
  between the main marks, the functions with the most time, inclusive (on the stack) and self (the
  innermost Python frame: that includes the C calls it makes and any wait for the GIL after them).

Why sampling and not cProfile: […]

### `solution\swiglu.py` (81 lines)

EXAMPLE: fused SwiGLU, silu(gate) * up, for the Qwen2 MLP, forward and backward in Triton.

It shows the whole path: a Triton kernel, a torch.autograd.Function around it, and a patch that
makes Soup's model call it. These kernels do not count as your own kernel, under any name: add your
own kernels next to them (keeping the example running is fine) or replace this file.

### `solution\trainer_build.py` (643 lines)

Soup's trainer build (the rest of SFTTrainerWrapper.setup after the model): the tokenization of the
training rows and datasets' fingerprint of it, with the same results and less work. Each part has
its own switch, off by default until measured on the T4; every replacement lives only while Soup's
`setup` runs and takes the original code for anything it does not recognize.

SOUP_SOLUTION_FAST_ROWS=1: Soup formats every row with `build_assistant_only_labels`
(soup_cli/data/loss_mask.py:273). It first asks the chat template for an assistant mask
(`_apply_template_with_mask`: a render with `return_assistant_tokens_mask=True` and a tokenization),
and Qwen2.5's template has no `{% generation %}` block, so transformers returns an all-zero mask
(transformers/utils/chat_template_utils.py:508, tokenization_utils_base.py:3133-3154: no generation
indices, `current_mask = [0] * len(ids)`) and Soup rejects it whenever the row has an assistant
message (loss_mask.py:146-147), then renders and tokenizes the conversation again for its incremental
fallback. Here, for a real transformers tokenizer whose template has no `{% generation %}` block
(transformers' own regex) and a conversation with an assistant message, `_apply_template_with_mask`
returns None at once: the value Soup computes, without the first render (one of the four renders
and tokenizations of each row). Every ids list then goes through `_coerce_int_list` (loss_mask.py:48),
which checks and converts each token in a Python loop; a […]

### `solution\trainer_patch.py` (113 lines)

TRL's SFTTrainer.compute_loss for a model that returns no logits.

TRL computes `entropy` and `mean_token_accuracy` from `outputs.logits` and raises without them. With
the fused head (fused_ce.py) the model leaves the per-token entropy, loss and correctness in
`fused_ce.LAST_STEP`; this compute_loss logs the same three columns with TRL's own formulas from those
values and hands Soup's rewind recorder the per-token losses through the attribute it already reads
(`_soup_chunked_step`, see soup_cli.trainer.rewind_hf: `row_losses_from_token_losses`).

Soup's trainer is a subclass of trl.SFTTrainer built at setup time; patching the attribute on the
base class reaches it because Python looks the method up on every call. A step that is not training,
has no labels, or where the model did not take the fused path goes through TRL's original code.

### `solution\unpad.py` (384 lines)

Padding removal for the training step (allowed by the rules: "removing padding inside a batch in
your own code is fine; mixing rows across batches is not").

The collator pads the 4 rows of a batch to the longest one; on the task data that makes 43% of the
positions padding, and the base runs every layer, the head and the loss on them. Here the real
tokens of the batch are packed into one sequence of N tokens (rows stay in order, each row keeps its
own positions 0..len-1 for the rotary embedding), the decoder layers run on (1, N, D), and attention
is causal within each row's slice: exactly the tokens a real token attends to in the base (a padded
row with a causal + padding mask). The fused head (fused_ce.py) scores the packed tokens and scatters
the per-token values back to (B, T) for the trainer's metrics and Soup's rewind log, so the logged
numbers keep their base meaning.

Alignment tail: N is rarely a multiple of 8 (6 of the 64 steps), and N is the contiguous extent of
several GEMM operands in the layers (the transposed gate|up product and its gradient, the LoRA and
weight-gradient products), where cuBLAS then falls back to its align1/align2 kernels at about half
the speed. `varlen_model_forward` appends a few dummy tokens (the last real id again, positions
0..pad-1, so a short row of its own) to make the layers' token count a multiple of
`SOUP_SOLUTION_PAD_TOKENS` (16; 0 or 1 turns it off) and cuts them off before the final norm, so the
head, the loss, the metrics and […]

### `solution\warmup.py` (831 lines)

Triton compilation of the run's kernels in a background thread of the training process, while
Soup loads the model.

Every measured run starts with empty compiler caches, so the first training step compiles every
kernel it meets (Kaggle T4, commit d649c7c: the first two steps took about 4.5 s against 0.9 s for
two ordinary ones; from the end of apply() to the first step there are about 6 s: the model load,
3.1 s, and the trainer setup, 2.9 s). Here they are compiled ahead, so that the first step finds
them in Triton's in-memory cache: the variants a real step compiles (rms_norm_rows_fwd and _bwd two
each, nf4_dequant_kernel, rope_qk_kernel forward and backward (half for fused_proj.py, fp32 for
rope.py's own path), the example SwiGLU forward and backward, the kernels other modules register:
fused_proj's rms_norm_rows_rebuild, head_stream's chunk and finish kernels; with head_stream off,
fused_ce_rows_kernel two) and a rarer third one of
fused_ce_rows_kernel.

What makes a launch compile (triton 3.6.0, triton/runtime/jit.py)
-----------------------------------------------------------------
`JITFunction.run` (jit.py:695) looks the launch up in a dict per device (`device_caches[device]`,
jit.py:707, 771) under `compute_cache_key(specialization, options)` (jit.py:563-583, called at
:712) and compiles only on a miss (jit.py:713-721). The key holds, per argument, what the binder
built by `create_function_from_signature` gives (jit.py:402-421; `specialize_impl` is the […]

### `solution\weight_stream.py` (939 lines)

The 4-bit weights of the decoder layers kept in pinned host memory and copied to the GPU one layer
ahead of the compute, so that only a ring of K layers (K = 2 or 3) holds them on the GPU instead of
all 28. Allowed by the rules: "recomputation or offloading in your own code (its time counts)";
Soup's own `stream_layers` switch stays off, nothing of it is used here.

What moves: for every bitsandbytes Linear4bit of a layer (q, k, v, o, gate, up, down) the packed NF4
weight (uint8), the 8-bit absmax codes of its blocks and, with double quant, the fp32 second-level
scales. On Qwen2.5-1.5B that is 46.8 M weights per layer, 24.1 MB with the scales, 675 MB over the
28 layers. The NF4 code table, the absmax code table, the offset and the biases are a few hundred
bytes and stay on the GPU.

The ring: slot s holds layer i when i % K == s, always the same slot for the same layer. The
Params4bit of layer i points (`param.data`) into slot i % K for good, and so do `quant_state.absmax`
and `quant_state.state2.absmax`; only the bytes in the slot change. This matters for the backward:
bitsandbytes' MatMul4Bit (and the patched one in nf4.py) saves `B.view(-1, 1)`, a view of the storage
the weight had during the forward, and reads it again in the backward. With a fixed slot per layer,
that view sees the layer's bytes again whenever the layer is loaded back into its slot.

Order of the copies (host -> GPU on a side stream, from pinned memory, so they overlap the compute):
- forward: before […]
