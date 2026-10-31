# I1: CPU-диета 2, подача вычислений (атрибуция, итог)

Ветка `claude/i1-cpu-diet2` от main f152a56 (с H2). Кода в итоге нет (единственная попытка откатана), заметка — атрибуция для следующих.

## Атрибуция CPU на main da895ed (Modal acc3, x16, `SOUP_SOLUTION_CPU_PROFILE=1`, шаги 5–12, публичные строки; `soup-cd-prof1`)

- Свои Triton-запуски: 598 на шаг (nf4_dequant 90, rope_qk 84, nf4_int8_t 79, rms_norm_rows_fwd 57, nf4_dequant4 55, head_chunk 38, rms_norm_rows_bwd / bwd_split / rebuild 28 / 28 / 27, masked_pair_sum 27, int8_swiglu_bwd 26, gu_quant 25, quant_t_int8_il 24); после H2 ещё ~56 его ядер вместо ~670 запусков torch.
- Путь запуска: обёртка харнесса `harness/_train.py:60 run` (два `torch.cuda.Event(enable_timing=True).record()` на каждый наш запуск, 1 196 на шаг) — 71 мс/шаг по сэмплеру, из них `Event.record` 70,5. Локально record стоит 8 мкс что с потоком, что без (current_stream 2,4 мкс, is_available 0,5): это сам `cudaEventRecord`, на T4-хосте ~2 ГГц один наш запуск ≈ 75–80 мкс CPU (run Triton ~30–35 + события харнесса ~40–50); на длинных шагах часть этого — ожидание места в очереди GPU.
- По самой внутренней функции solution (сэмплер, мс/шаг, вместе с их C-вызовами): weight_stream._load 27,4 и release 15,4 (зона P2), fused_proj._qkv_forward 26,7, int8_bwd.mlp_backward 21,0, _qkv_backward 21,0, int8_bwd._decode_t 18,3, attention _rebuild 15,9, _launch_many (dequant4, 28 аргументов) 14,8, rope._launch_strided 13,3, _quant_rows 13,2 и _inverse 6,9 (убраны H2), _expand_heads 8,0, _scaled_mm 7,3, _padded_half 6,7.
- `.item()` 226 на шаг, но синхронизаций 3 (grad_scaler в конце шага, lookup головы): остальные на CPU-тензорах.

Вывод: GO −30 мс на коротком шаге требует −300…−400 наших запусков на шаг (слияние ядер через модули) или столько же Python; быстрых правок такого размера в зоне подачи нет.

## Попытка: два декода int8 backward в одном запуске (`SOUP_SOLUTION_INT8_PAIR`, 1aeef7f, откатано в 6d0ca68)

W_gate^T и W_up^T одним `nf4_int8_t2_kernel` и все декоды из кэша операндов fused_proj (без чтений Params4bit), побитово то же (тест). Modal ABBA (`soup-cd-pair1`, 2 раунда + холостой): off 48,96 / 49,61 с, on 49,42 / 49,54 (**+0,20 с**); шаг 4 +204 мс (лишнее ядро в потоке компиляции int8 backward, переключение ждёт поток), шаги 10+ +3,3 мс/шаг (шум, выигрыша −28 запусков ≈ −2 мс не видно). Каждое новое ядро стоит компиляции на первых шагах, а экономия одного запуска на слой ~2 мс/шаг её не окупает.
