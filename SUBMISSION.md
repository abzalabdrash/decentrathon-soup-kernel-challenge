# Submission: null-pointers

Decentrathon 6.0, трек 01 Soup Kernel Challenge. Описание решения, запуск, результаты и
ограничения: [README.md](README.md).

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers/commit/0b2cb75
  (код заморожен на 0b2cb75; папка `solution/` совпадает с 6c426d1)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  README.md): https://www.kaggle.com/code/slerfas/soup-final-public
- Demo video (up to 3 minutes): `<VIDEO_URL>` (сценарий: [docs/PRESENTATION.md](docs/PRESENTATION.md))
- Presentation: `<PRESENTATION_URL>` (план слайдов: [docs/PRESENTATION.md](docs/PRESENTATION.md))

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Нурлан Абдраш | капитан | abdrnurlan | nurlanjj | +77017783296, Telegram @Nurlan0006 |
| Абзал Абдраш | разработчик | abzalabdrash | namedesign | |
| Ерасыл Кудайберген | разработчик | peagverr | peagverr | |
| Руслан Дементов | участник | — | — | |

## Our kernels
Доля — время ядра по CUDA-событиям харнесса на шагах 3+ к времени этих шагов (проверка 4). Числа —
полный прогон на Kaggle T4 soup-m13-full (f152a56, 31 октября 2026; в нём уже все ядра финального кода):
`own_kernel_ms` / `timed_loop_s` из `runs/solution_1/metrics.json`, 24,433 с шагов 3+. Итог финального
прогона soup-final-full: `23,1 %`.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | распаковка NF4 → fp16 в bitsandbytes 0.50.2 (три запуска и два выделения на вес) одним проходом, побитово | 3,89 % |
| `gu_quant_kernel` | `solution/offload_int8.py` | SwiGLU forward (ядро примера) плюс int8-форма (gate, up) для оффлоада на хост в том же проходе | 3,20 % |
| `nf4_int8_t_kernel` | `solution/int8_bwd.py` | fp16-распаковку весов MLP перед GEMM градиента по входу: сразу транспонированный int8 с масштабом на признак | 3,04 % |
| `int8_swiglu_bwd_kernel` | `solution/int8_bwd.py` | backward SwiGLU и поэлементные операции torch вокруг него; эпилог int32-GEMM | 2,83 % |
| `head_chunk_kernel` | `solution/head_stream.py` | log-softmax, cross-entropy, энтропию и точность TRL по полным логитам (B, T, V): онлайн-статистики по чанку словаря | 1,65 % |
| `quant_t_int8_il_kernel` | `solution/int8_bwd.py` | транспонирующее квантование dL/d(gate, up) по токенам для второго int8-GEMM | 1,41 % |
| `rms_norm_rows_fwd` | `solution/rmsnorm.py` | Qwen2RMSNorm (5 ядер torch) и сложение residual перед нормой | 1,30 % |
| `nf4_dequant4_kernel` | `solution/fused_proj.py` | четыре распаковки q, k, v, o слоя одним запуском | 1,09 % |
| `rope_qk_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (срезы, отрицание, конкатенация, произведения) для q и k в обе стороны | 0,84 % |
| `rms_norm_rows_bwd_split` | `solution/rmsnorm.py` | backward пост-attention нормы по двум хранимым слагаемым вместо fp32-суммы | 0,82 % |
| `rms_norm_rows_bwd` | `solution/rmsnorm.py` | backward RMSNorm в autograd torch | 0,73 % |
| `masked_pair_sum_kernel` | `solution/fused_proj.py` | `mul` + `addcmul_` вклада двух адаптеров LoRA через маски dropout | 0,39 % |
| `dequant_rows_cols_kernel` | `solution/int8_glue.py` | `acc.float().mul_(scale_tok).mul_(scale_col)` деквантования градиента по входу MLP, побитово | 0,32 % |
| `rms_norm_rows_rebuild` | `solution/rmsnorm.py` | хранение half-выхода нормы для backward: восстанавливает его бит в бит | 0,28 % |
| `quant_rows_kernel` | `solution/int8_glue.py` | ~13 операций torch квантования градиента по токенам (abs, amax, деление, floor, clamp, касты) и каст fp32 → half, побитово | 0,18 % |
| `quant_t_int8_kernel` | `solution/int8_bwd.py` | квантование градиента из half (первые шаги и откаты) | 0,15 % |
| `gu_dequant_kernel` | `solution/offload_int8.py` | восстановление half (gate, up) из int8-формы там, где нужен half | 0,07 % |
| `head_finish_kernel` | `solution/head_stream.py` | финальные loss, энтропия, верность и dL/dh головы | 0,01 % |
| **Итого своих ядер** | | | **22,18 %** (на строках жюри 20,5 %) |

В коде есть и выключенные по умолчанию ядра: `fused_ce_rows_kernel` (`solution/fused_ce.py`, голова на
GPU), `nf4_dequant3_kernel` (`solution/fused_proj.py`), `head_unpack_kernel` (`solution/head_pack.py`,
12-битная голова), шесть ядер int8 forward в `solution/int8_fwd.py` (одно из них тоже называется
`quant_rows_kernel`). Ядра примера набора (`solution/swiglu.py`) своими не считаем: они идут в первых
шагах и откатах, 0,20 % в m13.

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
Публичный ноутбук soup-final-public на замороженном коде; справа для сравнения — последний полный
прогон до заморозки (soup-m13-full, f152a56).

| | Base Soup | Ours | Ratio | m13: base → ours |
|---|---|---|---|---|
| Time of `soup train`, s | `93,0` | `47,7` | speedup x`1,948` | 83,18 → 45,28 (x1,837) |
| Peak GPU memory, MiB | `14 345` | `395` | base/ours x`36,3` | 14 345 → 395 (x36,32) |
| Eval loss (public rows) | `2,04966` | `2,04964` | `-0,001 %` vs base | 2,04963 → 2,04980 (+0,008 %) |

Полный прогон soup-final-full: s = `1,875`, пик `395` MiB, eval `+0,002 %`,
`10 из 10`. Протокол жюри на других строках (base, solution, solution, base): soup-final-j1
(seed 67) s = `1,803`, пик `415` MiB, eval `+0,002 %`, `10 из 10`; soup-final-j2 (seed 89)
s = `1,841`, пик `415` MiB, eval `+0,002 %`, `10 из 10`. До заморозки — 10 из 10 на всех
18 наборах строк и seed из `docs/STATUS.md`; все прогоны и стресс-формы: README.md, разделы 3.1–3.3.

- [ ] All 10 checks of `harness/check_rules.py` passed (финальные прогоны: `10 из 10`; до заморозки 10 из 10 во всех 18 прогонах жюри и в полных прогонах main 31 октября)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
План слайдов и текст озвучки видео: [docs/PRESENTATION.md](docs/PRESENTATION.md).
- какие операции ускорены и почему: удаление паддинга и varlen attention; слитые проекции и LoRA через
  addmm; свои ядра RMSNorm, RoPE, распаковки NF4; голова без логитов с матрицей на хосте; int8 на
  тензорных ядрах в backward MLP и его клей в двух своих ядрах; граница шага без синхронизаций; старт
  и выход команды;
- как достигнут пик 395 MiB против 14 345: NF4-веса слоёв и голова в pinned-памяти хоста с кольцами на
  GPU, пересчёт attention, оффлоад (gate, up) в int8-форме и входов слоёв, лимит стека CUDA;
- как работают ядра (онлайн-статистики головы, NF4 → int8 с масштабами, эпилоги int8-GEMM);
- результаты против базы: скорость, память, качество (eval, кривые, метрики TRL и rewind), прогоны жюри
  на скрытых строках и стресс-формы;
- что пробовали и отказались, ограничения и следующие шаги.

## Limitations
- s ≈ 1,84–1,89 на публичных строках и 1,74–1,91 на строках жюри до заморозки. Короткие шаги ограничены
  CPU (скорость постановки ядер в очередь, включая события замера харнесса на каждом запуске своего
  ядра), длинные — GPU и лимитом мощности T4, голова — полосой PCIe.
- Пик GPU снижен ценой ~3,6 ГБ закреплённой памяти хоста, дополнительных передач по PCIe и пересчёта
  активаций; память хоста в зачёт не входит. Пик зависит от длин строк: 395 MiB на публичных, до 453
  на строках жюри, 431–471 на стресс-формах, которые база не проходит.
- int8-пути приближённые (не lossless): кривая loss 0,06–0,07 %, eval +0,003…+0,036 %, метрики проверки 5
  до 0,36 % против базы; побитового совпадения с базой нет.
- Решение специализировано под задачу: Qwen2, NF4 bitsandbytes, LoRA на q и v, fp16 на T4; другой вход
  идёт по исходному коду без ускорения.
- Старт ~17 с до первого шага, из них ~6 с пролог харнесса и ~2,9 с отпечаток datasets — не наш код;
  до первого шага процесс упирается в GIL (компиляция Triton его держит).
- Дальше: меньше Python на запуск ядра, спекулятивная загрузка NF4 там, где хост её окупает, свой
  IMMA-GEMM со слитым эпилогом, тензорные ядра Turing из Triton через inline PTX `mma.sync`.
