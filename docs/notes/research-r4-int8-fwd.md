# R4: int8 в forward для gate|up и down (Qwen2.5-1.5B, Tesla T4)

Дата: 31 октября 2026. Подчат R4 (веб-ресерч), вход для I1. Код в `solution/` не менялся.

Метки: **[V]** проверено по тексту статьи или исходнику (раздел или таблица в скобках); **[I]** вывод автора заметки; **[R]** наша внутренняя документация, извне не проверена; **[—]** не найдено.

## 0. Итог

1. **Опубликованных порядков величин на входах gate|up (после RMSNorm с весом) и down (после SwiGLU) для Qwen2.5 нет.** Есть пики hidden state и данные по другим моделям. Эти порядки надо измерить хуками на весах Qwen2.5-1.5B (рецепт в §5, около 20 минут).
2. У Qwen2.5-1.5B пик |a| = 7 968 в hidden state, но пик не разрежен по критерию Sun et al.: медиана в пиковом токене 13,9, отношение около 574 при пороге 1000 **[V 2605.15572, Table 1]**. Значит, выброс лежит в плотном токене, и нужно смотреть структуру по каналам отдельно.
3. Forward int8 без обработки выбросов опасен. На GPT-2 pretraining per-token даёт +5,9…+17,4 % valid loss к FP **[V Jetfire, Table 4]**. На OPT-6,7B per-token без сглаживания даёт 42,5 % против 64,9 % FP16 **[V SmoothQuant, Table 1]**.
4. Работает обработка выбросов: декомпозиция fp16 по каналам (LLM.int8) или сглаживание (SmoothQuant, O1 на OPT-175B: 66,5 % против 66,9 % FP16) **[V]**.
5. Условия `torch._int_mm` подтверждены в исходнике v2.11.0: M > 16, K и N кратны 8 **[V Blas.cpp:859–866]**. Наши формы (M ≈ 830, K ∈ {1536, 8960}, N ∈ {1536, 8960}) подходят.
6. Калибровку можно сделать внутри прогона: статистика первого шага по каналам, масштабы по истории (delayed scaling). Выбросы по каналам стабильны при fine-tune **[V An et al. §3]**.
7. Рекомендация (§5): схема B. Per-token int8 для X, per-output-channel int8 для W, плюс fp16-декомпозиция top-k каналов, выбранных на первом шаге. Сглаживание (схема C) — если B не проходит.

## 1. Выбросы у Qwen2.5

- **Пик по всем хукам** (embedding, hidden после residual, attention out, MLP out, SwiGLU gate pre-activation, финальная норма): Qwen2.5-1.5B 7 968 (base и Instruct, без изменений), 7B 13 248, 32B 30 848 (global M) **[V 2605.15572, Table 1, §C.2–C.3]**.
- **Локальная разреженность:** у 1.5B медиана |a| в пиковом токене 13,9, Top-100 7 776. Авторы называют это «large but relatively dense» и помечают 1.5B как провал критерия Sun **[V §3.2, Table 1]**. Точно где лежит пик (какой канал, какие токены) в статье не указано.
- **Форма по слоям:** у Qwen2.5 рост «jump-and-plateau»: резкий подъём в ранних или средних слоях и долгое плато **[V §4.2]**. Пиковый слой не обязан быть последним. Пик hidden state у 22 из 24 чекпойнтов лежит в residual stream **[V §4.3]**.
- **SFT** меняет только поздние слои: final-layer peak у 1.5B 1 536 → 840, а среднее плато стабильно (×1,000) **[V §C.3]**.
- **Опорная точка LLaMA2-7B (не Qwen):** массивные активации в фиксированных каналах 1415 и 2533, появляются с 2-го слоя и держатся до ~30-го, у стартового токена и первого «.»/«\n» **[V Sun et al. 2402.17762, §3]**.
- **Супер-веса:** всегда в `mlp.down_proj` раннего слоя у проверенных моделей (Llama, Llama2, Mistral, OLMo, Phi-3). Qwen в Table 2 нет **[V Yu et al. 2411.07191, Table 2, p.1]**. Поиск без данных: один forward, спайки входа и выхода down_proj **[V §3.1]**. Механизм: Hadamard-произведение gate и up даёт большую активацию перед down_proj, супер-вес её усиливает **[V §3.1]**. Для Qwen-1.5B — [—].
- **Выбросы в down_proj у LLaMA2-7B:** ограничены ранними и поздними слоями, связаны со стартовым токеном и слабыми токенами, фиксированные каналы **[V An et al. 2502.06415, §3.1]**. Выбросы сохраняются при fine-tune: величины и распределения почти не меняются **[V An et al., раздел о fine-tune]**.
- **Порядок на входе gate|up [I]:** RMSNorm даёт y_j = x_j·w_j / rms(x). Если один канал доминирует в rms (x_j = M, остальные малы), то rms ≈ M/√d и y_j ≈ √d·w_j. При d = 1536 это ≈ 39·|w_j|. Веса RMSNorm у Qwen2.5 не опубликованы, поэтому порядок надо мерить.
- **Порядок на входе down_proj:** [—] для Qwen. Для LLaMA2-7B выбросы из-за произведения gate и up см. выше.

## 2. Методы int8 в forward (гранулярность, ошибка, цена)

| Метод | Гранулярность (X / W) | Оставлено в высокой точности | Результат | Применимость к нам |
|---|---|---|---|---|
| LLM.int8() 2208.07339 | X per-row (per-token); W per-column; масштабы — внешний продукт | каналы с \|x\| ≥ 6,0 в ≥ 25 % слоёв и ≥ 6 % позиций (раздел 3) → fp16 | 16-битное качество при ≥ 6,7B; выбросов ~0,1 % каналов **[V §3]** | Декомпозиция top-k каналов — опора для схемы B. Авторы: накладные расходы квантования и декомпозиции «значительны» **[V раздел о скорости]** |
| SmoothQuant 2211.10438 | O1: W per-tensor, X per-token (dynamic), сглаживание s_j = max\|X_j\|^α / max\|W_j\|^(1−α), α = 0,5 (0,75 для GLM) | — | OPT-175B: FP16 66,9 %, O1 66,5 %, W8A8 без сглаживания 35,5 % **[V Table 3]**; OPT-6,7B: per-token 42,5 %, per-channel 64,8 %, FP16 64,9 % **[V Table 1]** | Сглаживание считается на первом шаге внутри прогона (калибровка 512 предложений в статье). Для gate|up s_j сворачивается в RMSNorm-вес, для down — в строки up_proj **[I]** |
| SwitchBack 2304.13013 | forward: X row-wise, W tensor-wise; backward: G row-wise, W tensor-wise; градиент W — 16 бит **[V Alg. 1]** | градиент весов | CLIP ViT-Huge: в пределах 0,1 п.п. bf16, ускорение 13–25 % **[V abstract]** | На LM pretraining по Jetfire Table 4 (там SwitchBack = per-token): valid loss +5,9 % (base), +9,5 % (medium), +17,4 % (large) **[V]**. Наш fine-tune с заморож. весами — [—] |
| Jetfire 2403.12422 | блоки 32×32 **[V, B = 32]**; int8 на входе и выходе линейного оператора | — | GPT-2 valid loss: Jetfire 2,8597 против FP 2,9074 (base) **[V Table 4]**; по MSE лучше per-token, на уровне per-channel **[V Fig. 5]** | Нужен свой GEMM по блокам, `torch._int_mm` такого не делает. Дорого |
| DeepSeek-V3 2412.19437 | forward: X 1×128 (per token, 128 каналов), W 128×128 блоки, онлайн max **[V §3, стр. 15]** | embedding, output head, MoE gating, normalization, attention **[V §3, стр. 15]** | Block-wise dgrad расходится на MoE 16B **[V App. B.2]** | Группы 128 по K не выразить через `torch._int_mm` без K/128 частичных GEMM. Ориентир для lm_head и норм **[I]** |
| QuaRot 2404.00456 | Адамар-ротация, без калибровки **[V abstract, через выжимку]** | — | 6/8-бит RTN на LLaMA2 без потерь без калибровочных данных; 4-бит LLaMA2-70B: ≤ 0,47 ppl WikiText-2 **[V]** | Ротация NF4-весов запрещена (NF4 не меняем), можно только на лету для распакованного W. WHT на 8960×1536 каждый шаг **[I]** |

- **Per-tensor для X плохой:** на hidden state 1.5B–7B max-abs per-tensor даёт около 10–14 дБ SQNR, 99,9 %-клиппинг — 0,2–0,4 дБ **[V 2605.15572, App. D.2]**.
- **Per-channel для X несовместим с INT8 GEMM** **[V SmoothQuant §3–4]**. Поэтому выбросы по каналам решают сглаживанием или декомпозицией.
- **QLoRA-подобный int8 forward:** работ не нашёл. В найденных руководствах QLoRA деквантует базу в 16 бит **[—]**.

## 3. Практика на T4 (sm_75)

- **Условия `torch._int_mm`** (PyTorch v2.11.0, `Blas.cpp:859–866`): dim = 2; `self.size(0)` > 16; `self.size(1)` (K) > 0 и кратно 8; `mat2.size(1)` (N) > 0 и кратно 8; результат int32, contiguous **[V]**. Внутри `int8_gemm` (`CUDABlas.cpp:2227`): `CUBLAS_COMPUTE_32I`, A и B в `CUDA_R_8I`, C в `CUDA_R_32I`, `transa`/`transb` по раскладке **[V]**.
- **Раскладка:** в исходнике `cublasCommonArgs` выбирает `transa`/`transb` по раскладке; код сам о «broken for some combination of transposed inputs» пишет комментарием **[V Blas.cpp:860]**. Что TN быстрее, мы мерили сами **[R, RESEARCH.md, ночная сессия]**.
- **Пиковые числа T4:** 130 INT8 TOPS, 65 FP16 TFLOPS (mixed precision), 320+ GB/s, 70 W **[V nvidia.com/en-eu/data-center/tesla-t4]**.
- **cuBLASLt против CUDA-ядер:** для int8 `cublasLtMatmul` (tensor cores) намного быстрее `cublasGemmEx` (CUDA cores). CUTLASS int8 и int4 на T4 не достигают 50 % теоретического пика **[V Jia et al. 1903.07486, §4.4]**.
- **Замечание к заметкам команды:** в извлечённом тексте таблицы 4.3 подписи строк, по-видимому, сдвинуты на одну строку (значения 253 и 7 174 похожи на FP64 и FP32). Значения 74 934 (INT8, GOPS) и 41 616 (FP16, GFLOPS) соответствуют заметке команды «74,9 против 41,6» **[R]**. Вероятно, верно, но подпись надо проверить на рендере PDF **[I]**.
- **Конфигурации sm75 в vLLM** (CUTLASS int8, по M): M > 256 — TileShape 128×128×64, smem 32 KB; M ∈ (128, 256] — 128×128×128, 64 KB; M ∈ (32, 64] — 64×128×128, 48 KB; M ∈ [1, 32] — 32×128×64, 48 KB **[V scaled_mm_c2x_sm75_dispatch.cuh, HF-зеркало RedHatAI]**. Для M ≈ 830 ориентир: 128×128×64 **[I]**.
- **Слияние квантования:** в `csrc/layernorm_quant_kernels.cu` (main, 31.10) fused RMSNorm + квантование только статический FP8 **[V, строка 5]**. Публичного fused int8-эпилога RMSNorm для sm75 не нашёл **[—]**. Jetfire делает квантование и деквантование на чипе в ядрах с int8 в памяти **[V §5]**. TensorRT-LLM описывает SmoothQuant W8A8 с динамическим per-token режимом и деквантование `static_cast<FP>(q)·s` **[вторичная выдача поиска, nvidia.github.io/TensorRT-LLM]**.
- **Вывод [I]:** квантование активаций в ядре RMSNorm или SwiGLU окупает схему. Наши torch-операции на квантование съедают выигрыш (×0,3–0,7 **[R, RESEARCH.md]**), поэтому нужно однопроходное ядро.

## 4. Калибровка внутри прогона

- **Per-token dynamic:** не требует калибровки. Нужна одна редукция max|x| по строке **[I]**.
- **Статистика первого шага:** per-channel max по первому батчу (4 строки, до 512 токенов) — одна редукция внутри прогона, что допустимо по правилам **[I]**. Опора: выбросы по каналам фиксированы (Sun, §3) и сохраняются при fine-tune (An et al.) **[V]**.
- **Delayed scaling (NVIDIA Transformer Engine, FP8):** масштаб берётся из истории amax предыдущих итераций (окно `amax_history_len`), текущий тензор не нужно просматривать дважды **[V docs.nvidia.com/transformer-engine, вторичная выдача]**. Для int8 идея та же **[I]**.
- **Критерий выбросов LLM.int8** (|x| ≥ 6, ≥ 25 % слоёв, ≥ 6 % позиций) считается на первом forward **[I]**. Для Qwen с пиками около 8 000 условие по величине выполнено, частоту надо проверить **[—]**.
- **Детекция супер-весов** (Yu et al. §3.1): один forward на одном промпте, спайки down_proj input и output **[V]**.
- **Цена декомпозиции [I]:** k каналов, выбранных по max за первый шаг. Доп. fp16 GEMM M × k × N: k = 16 на down (K = 8960) — около 0,2 % FLOP; на gate|up (K = 1536) — около 1 %.

## 5. Рекомендация для gate|up и down

**Измерения до внедрения** (это работа I1; даём рецепт, без скачивания весов):
1. Хуки на Qwen2.5-1.5B (весы уже у команды в WSL, путь проверить): X_gu = выход RMSNorm (с весом), X_down = выход SwiGLU. По 4–8 последовательностям по 512 токенов: max|x| по каналам, доля топ-16 каналов в Σx², max/медиана по токену, относительная ошибка GEMM для (A) per-token × per-channel и (B) то же + декомпозиция top-k.
2. Супер-веса: спайки down_proj input и output на первом forward (Yu et al.).
3. Оценка по порогам задачи: loss, entropy, accuracy ≤ 3 %, eval ≤ 0,5 %.

**Схемы (по нарастанию сложности):**
- **A.** Per-token X (dynamic) × per-output-channel W (из NF4 с нашим масштабом). Риск высокий на down при канальных выбросах **[V SmoothQuant Table 1]**.
- **B. (рекомендация)** = A + fp16-декомпозиция top-k каналов, выбранных на первом шаге (LLM.int8). k = 16 на down, около 8–16 на gate|up. Самая безопасная схема без калибровки **[I]**.
- **C.** = B + сглаживание SmoothQuant (α = 0,5) со s_j с первого шага. Для gate|up s_j сворачивается в RMSNorm-вес (без стоимости). Для down — в строки up_proj. Эффективный W меняется, NF4-байты нет. **Перед внедрением спросить организаторов**, считается ли это изменением замороженных весов **[I]**.

**Кандидаты на fp16:**
1. `down_proj` слоёв с супер-весом: в Llama, Mistral, OLMo и Phi-3 он в раннем слое **[V Yu et al., Table 2]**. Для Qwen [—], детектировать на первом шаге.
2. `down_proj` ранних и поздних слоёв: у LLaMA2-7B выбросы xdown ограничены такими слоями **[V An et al., §3.1]**. Для Qwen [—].
3. `gate|up` в слоях плато hidden state у Qwen2.5 **[V 2605.15572, §4.2]** — либо схема C.
4. `lm_head`: DeepSeek-V3 держит output head в высокой точности **[V §3.3]**. Организаторы разрешили int8 (ответ 29). Оставить fp16 как запасной вариант, если кривая eval не проходит.
5. Нормы, attention и embedding: не квантуем, как в DeepSeek-V3 и SmoothQuant (LayerNorm и softmax в FP16) **[V]**.

## 6. Что не закрыто и риски

- Нет публикаций про входы gate/up и down у Qwen2.5. Данные 2605.15572 — это hidden state и SwiGLU gate pre-activation, а не вход gate/up после RMSNorm **[V §2.3]**.
- Нет работ про int8 forward при fine-tune с заморожен. NF4-базой и LoRA. Есть только LM pretraining (Jetfire: +6…17 % loss, per-token) и CLIP (SwitchBack: ≤ 0,1 п.п.) **[V]**. Порог 3 % на 64 шагах надо проверять экспериментом **[I]**.
- Публичного fused int8-эпилога RMSNorm для sm75 не нашли **[—]**.
- Скорость int8 на T4 с нашими формами в этой сессии не мерили (нет T4). Цифры команды — **[R]**.

## Источники

- Chen et al., 2026, «Measuring Maximum Activations in Open LLMs», https://arxiv.org/abs/2605.15572 (Table 1, §2.3, §3.2, §4.2, §4.3, §C.2–C.3, App. D.2)
- Yu et al., 2024, «The Super Weight in LLMs», https://arxiv.org/abs/2411.07191 (§3.1, Table 2)
- Sun et al., 2024, «Massive Activations in LLMs», https://arxiv.org/abs/2402.17762 (§3)
- An et al., ICLR 2025, «Systematic Outliers in LLMs», https://arxiv.org/abs/2502.06415 (§3.1, раздел о fine-tune)
- Dettmers et al., 2022, «LLM.int8()», https://arxiv.org/abs/2208.07339 (§3, Table 5)
- Xiao et al., 2022, «SmoothQuant», https://arxiv.org/abs/2211.10438 (§3–5, Table 1–3)
- Wortsman et al., 2023, «SwitchBack», https://arxiv.org/abs/2304.13013 (Alg. 1, abstract)
- Xi et al., 2024, «Jetfire», https://arxiv.org/abs/2403.12422 (§5, §7.3, Table 4, Fig. 5)
- DeepSeek-AI, 2024, «DeepSeek-V3», https://arxiv.org/abs/2412.19437 (§3.3 стр. 15, App. B.2)
- Ashkboos et al., 2024, «QuaRot», https://arxiv.org/abs/2404.00456 (abstract, через выжимку)
- Jia et al., 2019, «Dissecting the NVIDIA Turing T4», https://arxiv.org/abs/1903.07486 (§4.4, Table 4.3, см. замечание в §3)
- NVIDIA, Tesla T4, https://www.nvidia.com/en-eu/data-center/tesla-t4/ (спецификации)
- PyTorch v2.11.0, `aten/src/ATen/native/cuda/Blas.cpp` и `aten/src/ATen/cuda/CUDABlas.cpp`, https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Blas.cpp
- vLLM, `csrc/layernorm_quant_kernels.cu` (main, 31.10.2026), https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu
- vLLM, `cutlass_w8a8/scaled_mm_c2x_sm75_dispatch.cuh` (HF-зеркало), https://huggingface.co/RedHatAI/quantization/blob/34013adbd7adab5a7f67cada4f61d4668bb1fb61/cutlass_w8a8/scaled_mm_c2x_sm75_dispatch.cuh
- NVIDIA Transformer Engine, FP8 delayed scaling, https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html
- NVIDIA TensorRT-LLM, numerical precision (вторичная выдача), https://nvidia.github.io/TensorRT-LLM/1.0.0rc4/reference/precision.html
- Наша документация: `docs/RESEARCH.md` (ночная сессия, int8 на тензорных ядрах), `docs/briefs/organizers-qa-2026-10-31.md` (ответы 22, 29, 31).
