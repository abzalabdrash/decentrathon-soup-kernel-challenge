# Submission: null-pointers

Decentrathon 6.0, трек 01 Soup Kernel Challenge. Описание решения, запуск, результаты и
ограничения: [README.md](README.md).

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers/commit/<FINAL_COMMIT>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  README.md): `<PUBLIC_NOTEBOOK_URL>`
- Demo video (up to 3 minutes): `<VIDEO_URL>`
- Presentation: `<PRESENTATION_URL>`

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| `<CAPTAIN_NAME>` | капитан | `<GITHUB_LOGIN>` | `<KAGGLE_LOGIN>` | `<CAPTAIN_CONTACT>` |
| `<NAME>` | `<ROLE>` | `<GITHUB_LOGIN>` | `<KAGGLE_LOGIN>` | |
| `<NAME>` | `<ROLE>` | `<GITHUB_LOGIN>` | `<KAGGLE_LOGIN>` | |
| `<NAME>` | `<ROLE>` | `<GITHUB_LOGIN>` | `<KAGGLE_LOGIN>` | |

## Our kernels
Доля — время ядра по CUDA-событиям харнесса на шагах 3+ к времени этих шагов (проверка 4), полный
прогон на Kaggle T4, коммит 834e5ad, 31 октября 2026 (soup-m11-full, `runs/solution_1/metrics.json`:
`own_kernel_ms` / `timed_loop_s` = 25,543 с). Финальные доли: `<FINAL_KERNEL_SHARES>`.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | распаковка NF4 → fp16 в bitsandbytes 0.50.2 (три запуска и два выделения на вес) одним проходом, побитово | 3,69 % |
| `gu_quant_kernel` | `solution/offload_int8.py` | SwiGLU forward (ядро примера) плюс int8-форма (gate, up) для оффлоада на хост в том же проходе | 3,03 % |
| `nf4_int8_t_kernel` | `solution/int8_bwd.py` | fp16-распаковку весов MLP перед GEMM градиента по входу: сразу транспонированный int8 с масштабом на признак | 2,82 % |
| `int8_swiglu_bwd_kernel` | `solution/int8_bwd.py` | backward SwiGLU и поэлементные операции torch вокруг него; эпилог int32-GEMM | 2,60 % |
| `head_chunk_kernel` | `solution/head_stream.py` | log-softmax, cross-entropy, энтропию и точность TRL по полным логитам (B, T, V): онлайн-статистики по чанку словаря | 1,43 % |
| `quant_t_int8_il_kernel` | `solution/int8_bwd.py` | транспонирующее квантование dL/d(gate, up) по токенам для второго int8-GEMM | 1,32 % |
| `rms_norm_rows_fwd` | `solution/rmsnorm.py` | Qwen2RMSNorm (5 ядер torch) и сложение residual перед нормой | 1,25 % |
| `nf4_dequant4_kernel` | `solution/fused_proj.py` | четыре распаковки q, k, v, o слоя одним запуском | 1,03 % |
| `rope_qk_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (срезы, отрицание, конкатенация, произведения) для q и k в обе стороны | 0,80 % |
| `rms_norm_rows_bwd_split` | `solution/rmsnorm.py` | backward пост-attention нормы по двум хранимым слагаемым вместо fp32-суммы | 0,77 % |
| `rms_norm_rows_bwd` | `solution/rmsnorm.py` | backward RMSNorm в autograd torch | 0,69 % |
| `masked_pair_sum_kernel` | `solution/fused_proj.py` | `mul` + `addcmul_` вклада двух адаптеров LoRA через маски dropout | 0,39 % |
| `rms_norm_rows_rebuild` | `solution/rmsnorm.py` | хранение half-выхода нормы для backward: восстанавливает его бит в бит | 0,27 % |
| `quant_t_int8_kernel` | `solution/int8_bwd.py` | квантование градиента из half (первые шаги и откаты) | 0,14 % |
| `gu_dequant_kernel` | `solution/offload_int8.py` | восстановление half (gate, up) из int8-формы там, где нужен half | 0,07 % |
| `head_finish_kernel` | `solution/head_stream.py` | финальные loss, энтропия, верность и dL/dh головы | 0,01 % |
| **Итого своих ядер** | | | **20,33 %** (на строках жюри 19,1 %) |

В коде есть и выключенные по умолчанию ядра: `fused_ce_rows_kernel` (`solution/fused_ce.py`, голова на
GPU), `nf4_dequant3_kernel` (`solution/fused_proj.py`), `head_unpack_kernel` (`solution/head_pack.py`,
12-битная голова), шесть ядер int8 forward в `solution/int8_fwd.py`. Ядра примера набора
(`solution/swiglu.py`) своими не считаем: они идут в первых шагах и откатах, 0,20 % в m11.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- кернелов других проектов не копировали; matmul остаётся в cuBLAS / cuBLASLt (`torch._int_mm`),
  attention — ядро PyTorch `aten::_efficient_attention_*` с `cu_seqlens` (ответ организаторов 35);
- голова с онлайн-softmax по чанкам словаря без материализации логитов: та же задача, что Cut
  Cross-Entropy (https://arxiv.org/abs/2411.09009), онлайн-нормализатор softmax
  (https://arxiv.org/abs/1805.02867);
- int8 в backward по входу с масштабами по токенам и признакам: SwitchBack (https://arxiv.org/abs/2304.13013),
  Jetfire (https://arxiv.org/abs/2403.12422), Xi et al. (https://arxiv.org/abs/2306.11987);
- оффлоад и перекрытие копий с вычислениями: FlexGen (https://arxiv.org/abs/2303.06865), ZeRO-Offload
  (https://arxiv.org/abs/2101.06840); выборочный пересчёт: Korthikanti et al. (https://arxiv.org/abs/2205.05198);
- раскладка NF4 с двойной квантизацией: bitsandbytes 0.50.2, формат из QLoRA (https://arxiv.org/abs/2305.14314);
- 12-битная голова (выключена): идея DFloat11 (https://arxiv.org/abs/2504.11651).
Полный список прочитанного: README.md, раздел 1.18, и `docs/RESEARCH.md`.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Сейчас — полный прогон 834e5ad (soup-m11-full, 31 октября 2026); после заморозки кода заменяется
прогоном публичного ноутбука на `<FINAL_COMMIT>`.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86,22 (`<FINAL_BASE_S>`) | 46,53 (`<FINAL_WALL_S>`) | speedup x1,853 (`<FINAL_S>`) |
| Peak GPU memory, MiB | 14 345 (`<FINAL_BASE_MIB>`) | 395 (`<FINAL_PEAK_MIB>`) | base/ours x36,32 (`<FINAL_M>`) |
| Eval loss (public rows) | 2,04966 (`<FINAL_BASE_EVAL>`) | 2,04972 (`<FINAL_EVAL>`) | +0,003 % vs base (`<FINAL_EVAL_PCT>`) |

Протокол жюри на других строках (834e5ad, обучение 22500–22755, eval 31000–31255, seed 117, порядок
base, solution, solution, base): база 92,8 с, решение 50,4 с (s = 1,840), пик 14 333 против 435 MiB
(m = 32,9), eval +0,001 %, 10 из 10 проверок. Все прогоны жюри и стресс-прогоны: README.md, разделы 3.2–3.3.

- [x] All 10 checks of `harness/check_rules.py` passed (834e5ad: полный прогон и прогон жюри; финальный коммит: `<FINAL_CHECKS>`)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
- какие операции ускорены и почему: удаление паддинга и varlen attention; слитые проекции и LoRA через
  addmm; свои ядра RMSNorm, RoPE, распаковки NF4; голова без логитов с матрицей на хосте; int8 на
  тензорных ядрах в backward MLP; граница шага без синхронизаций; старт команды;
- как достигнут пик 395 MiB против 14 345: NF4-веса слоёв и голова в pinned-памяти хоста с кольцами на
  GPU, пересчёт attention, оффлоад (gate, up) в int8-форме и входов слоёв, лимит стека CUDA;
- как работают ядра (онлайн-статистики головы, NF4 → int8 с масштабами, эпилоги int8-GEMM);
- результаты против базы: скорость, память, качество (eval, кривые, метрики TRL и rewind), прогоны жюри
  на скрытых строках и стресс-формы;
- что пробовали и отказались, ограничения и следующие шаги.

## Limitations
- s ≈ 1,85 на публичных строках и 1,74–1,91 на строках жюри: шаг ограничен PCIe (NF4-веса едут дважды
  за шаг, голова раз за шаг) и лимитом мощности T4.
- Пик GPU снижен ценой ~3,6 ГБ закреплённой памяти хоста, дополнительных передач по PCIe и пересчёта
  активаций; память хоста в зачёт не входит. Пик зависит от длин строк: 395 MiB на публичных, до 453
  на строках жюри, 431–471 на стресс-формах, которые база не проходит.
- int8-пути приближённые (не lossless): кривая loss 0,07 %, eval +0,003 %, метрики проверки 5 до 0,36 %
  против базы; побитового совпадения с базой нет.
- Решение специализировано под задачу: Qwen2, NF4 bitsandbytes, LoRA на q и v, fp16 на T4; другой вход
  идёт по исходному коду без ускорения.
- Старт ~17 с до первого шага, из них 5,7 с пролог харнесса и ~2,9 с отпечаток datasets — не наш код.
- Дальше: очередь копий по ближайшему потребителю, спекулятивная загрузка NF4, свой IMMA-GEMM со
  слитым эпилогом, тензорные ядра Turing из Triton через inline PTX `mma.sync`.
