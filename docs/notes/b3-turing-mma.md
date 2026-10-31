# B3: тензорные ядра Turing в своих `@triton.jit` (31 октября, вечер, второй фронт)

Ветки: `claude/b3-turing-mma` (от main 0100e3a: хук, GEMM, слитый int8 backward) и `claude/b3-i1-fused` (от I1 0876091: слитый int8 gate|up forward). Аккаунт Kaggle shyraylym (`tools/kaggle_win.sh 4`).

## Что найдено

Triton 3.6 на T4 опускает `tl.dot` до FMA не потому, что не умеет Turing. В `libtriton.so` есть `callMmaTuringFp16` и `callMmaTuringInt8` (`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` и `m8n8k16...s32.s8.s8.s32`), `NvidiaMmaEncodingAttr::isTuring` (раскладка `nvidia_mma` версии 2.1). Отключён только проход AccelerateMatmul для cc < 80: строка «Dot op using MMA for compute capability 75 has been deprecated. It falls back to the FMA path» (проверено `strings` по `libtriton.so` из venv WSL с triton 3.6.0).

`solution/mma_gemm.py` `install()` ставит хук `triton.knobs.runtime.add_stages_inspection_hook` (его вызывает `CUDABackend.add_stages`, `triton/backends/nvidia/compiler.py:547`). Для ядер с именем `tc_*` стадия TTGIR идёт через штатный `make_ttgir` с capability 80, затем в тексте `versionMinor = 0` меняется на `1` и `"cuda:80"` на `"cuda:75"`, модуль парсится обратно (`triton.compiler.compiler.parse`). Остальное (LLVM, PTX, `ptxas --gpu-name=sm_75`) штатное. Запуск только с `num_stages=1`: конвейер Ampere выдал бы `cp.async`, которого на sm_75 нет. Прочие ядра прогона компилируются как раньше. Харнесс считает и меряет `tc_*` как обычные запуски `JITFunction.run` (`harness/_train.py:56-76`).

Проверка на T4 (`soup-b3-bench1`): в PTX каждого ядра есть `mma.sync.aligned.m16n8k8` (fp16) или `m8n8k16` (int8), есть `ldmatrix`, нет `cp.async` и нет `fma.rn.f32` в цикле; fp16 против `torch.mm` с ошибкой 0, int8 против `torch._int_mm` без расхождений.

Вопрос организаторам (это не буквально ответ 18, там был inline PTX): `docs/briefs/organizers-question-b3-2026-10-31.md`. До ответа всё за выключенными переключателями.

## Замеры T4 (бенч, чередование с cuBLAS в одной сессии; карта на 640–1300 МГц из-за лимита 70 Вт)

fp16 `tc_matmul_kernel` против cuBLAS, лучшая конфигурация из 8, скорость наша/cuBLAS: gate|up 0,87; down 0,73–1,07; qkv 0,67–0,77; o 0,66–0,82; голова 0,69–0,87; градиенты по входу 0,63–0,98. Медленнее cuBLAS на 10–30 %: без `cp.async` нет конвейера global→shared.

NF4 внутри тайла (`tc_nf4_matmul_kernel`, распаковка побитово как у `nf4_dequant_kernel`): медленнее «распаковка + cuBLAS» на всех формах, на 0,06–2,3 мс за GEMM (gate|up при 832 токенах 3,54 мс против 2,65). Причина: распаковка повторяется для каждого M-блока (7 раз при M = 832, BLOCK_M = 128), а GEMM упирается в тензорные ядра. Направление закрыто.

int8 `tc_matmul_i8_kernel` против `torch._int_mm` (cuBLASLt): gate|up forward 1,17 при 832 токенах, 1,03–1,06 при 1440–2048; down 0,63–0,83; gate|up backward 0,71 при 832, 1,06–1,09 при 1440–2048.

## Слитые int8-ядра (результат побитово тот же: int32-сумма точная, эпилог повторяет те же fp32-операции в том же порядке)

1. `tc_int8_swiglu_bwd_kernel` (шаги 3–4 `int8_bwd.mlp_backward`, `SOUP_SOLUTION_MMA_INT8_BWD`, по умолчанию 0). Бенч `soup-b3-bench2`, лучшая конфигурация (64, 128, 64, 4 варпа): 906 / 977 / 1811 / 2444 мкс против 726 / 921 / 1673 / 2207 при 512 / 832 / 1440 / 2048 токенах, то есть медленнее на 6–25 %; экономит int32-буфер 17,5 / 28,4 / 49,2 / 70,0 MiB. Тайлы 128 × 128 в 2–4 раза хуже. Польза только в памяти, если этот буфер стоит на пике.
2. `tc_int8_gate_up_kernel` (I1, MODE "gu": CHUNKS × `_int_mm` + `int8_gate_up_kernel` одним ядром, `SOUP_SOLUTION_MMA_INT8_FWD`, по умолчанию 0, ветка `claude/b3-i1-fused`). Локально (RTX 3050 Ti, путь sm75 через `TRITON_OVERRIDE_ARCH`): 1931 мкс против 2090 при 832 токенах, побитово во всех 6 конфигурациях, int32-буфер 11,4 MiB не нужен. T4: `soup-b3-bench3` (в работе).

## Что дальше

- ABBA `soup-b3-abba1` (main + слитый backward, тайлы 128 × 128, то есть неудачные): смотреть только пик памяти.
- ABBA I1 против I1 + слитый gate|up на T4, затем full и jury, если стена лучше.
- Если организаторы скажут «да»: остальные места, где за cuBLASLt идёт отдельный проход по int32 (int8-голова H1, шаг 7 backward), тем же приёмом.
