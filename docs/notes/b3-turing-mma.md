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

1. `tc_int8_swiglu_bwd_kernel` (шаги 3–4 `int8_bwd.mlp_backward`, `SOUP_SOLUTION_MMA_INT8_BWD`, по умолчанию 0). Перебор тайлов на T4 (`soup-b3-bench4`): (64, 64, 128, 4 варпа) 861 / 718 / 1666 / 2479 мкс против `_int_mm` + ядра 726 / 921 / 1618 / 2220 при 512 / 832 / 1440 / 2048 токенах, то есть при средней длине публичных строк быстрее, при 1440 вровень, на коротких и самых длинных медленнее на 5–19 %; этот тайл теперь по умолчанию. int32-буфер 17,5 / 28,4 / 49,2 / 70,0 MiB не нужен. Тайлы 128 в ширину в 2–4 раза хуже (`soup-b3-bench2`).
   ABBA `soup-b3-abba1` (main 6315426, тайлы 128 × 128, неудачные): base 45,17 с / 395 MiB / reserved 274; слитый backward (8 варпов) 46,38 с / **375 MiB** / 254, то есть пик минус одна страница 20 MiB; вариант с 4 варпами 54,2 с и **513 MiB**: spill регистров уходит в local memory, которую драйвер резервирует на все потоки, и nvidia-smi это видит (ловушка для любых своих ядер с большим числом регистров). eval у всех в шуме повторов base (2,04947–2,04964 у base, 2,04965–2,04989). ABBA с тайлом (64, 64, 128): `soup-b3-abba4`.
2. `tc_int8_gate_up_kernel` (I1, MODE "gu": CHUNKS × `_int_mm` + `int8_gate_up_kernel` одним ядром, `SOUP_SOLUTION_MMA_INT8_FWD`, по умолчанию 0, ветка `claude/b3-i1-fused`). T4 (`soup-b3-bench3`, побитово во всех конфигурациях): (64, 128, 64, 4 варпа) 1341 / 1670 / 2806 / 3757 мкс против пути I1 1512 / 1818 / 2962 / 4127 при 512 / 832 / 1440 / 2048 токенах, то есть −5…−11 % (−148 мкс на слой при 832: −4,1 мс на шаг, около −0,27 с на прогон; на длинных строках жюри до −10 мс на шаг); int32-буфер 7 / 11,4 / 19,7 / 28 MiB не нужен. 8 варпов и тайлы 128 в ширину медленнее. Ручная предвыборка следующего K-тайла в регистрах (без `cp.async`) локально в 20–80 раз медленнее (spill), отброшена.

## Что дальше

- `soup-b3-abba2`: fp16 / I1 (0876091) / I1 + слитый gate|up (два тайла), стена и пик.
- `soup-b3-abba4`: main + слитый backward с тайлом (64, 64, 128).
- Компиляция слитого gate|up с top-k на CPU WSL 1,6 с против 0,6 у `int8_gate_up_kernel` (зато `_int_mm` не нужен): поток компиляции I1 стартует на шаге 2, переключение после шага 8 (7313c94), запас есть; проверить на T4, что переключение не ждёт поток.
- Если организаторы скажут «да»: остальные места, где за cuBLASLt идёт отдельный проход по int32 (int8-голова H1 с её транспонированием 2,5 мс на шаг, шаг 7 backward), тем же приёмом.

## Строка для README (если организаторы ответят «да»)

Tensor cores from our own Triton kernels. Triton 3.6 lowers `tl.dot` to FMA code on sm_75 because its layout pass refuses MMA layouts below compute capability 80, although the library still contains the Turing lowering (`mma.sync.aligned.m16n8k8` for fp16, `m8n8k16` for int8). `solution/mma_gemm.py` registers Triton's stage hook (`triton.knobs.runtime.add_stages_inspection_hook`) for our kernels named `tc_*` only: their TTGIR is built with the Ampere layout pass and marked as a Turing (mma 2.1) cuda:75 module; LLVM lowering, PTX and ptxas run unchanged for sm_75, with `num_stages=1` (no `cp.async` on Turing). The kernels compile inside the measured run with empty caches and launch through `JITFunction.run`, so the harness counts and times them. Two of them replace a cuBLASLt int8 GEMM and the pass over its int32 result: the MLP's int8 gate|up forward with its SwiGLU epilogue, and the backward's int8 `W_down^T g` product with the SwiGLU derivative. Integer accumulation is exact and the epilogues repeat the same fp32 operations, so their outputs are bit-for-bit those of the two-kernel paths; the int32 buffers (up to 70 MiB at 2,048 tokens) are never allocated.
