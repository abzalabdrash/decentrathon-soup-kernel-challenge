# Вопрос организаторам (волна 3, B3): тензорные ядра Turing для своих `@triton.jit` через штатный путь Triton

Текст для чата трека (отправляет Абзал). Номера ответов по нашей нумерации, справка в `organizers-questions-2-2026-10-31.md`.

---

Продолжение ответа 18 (inline PTX `mma.sync` в своём `@triton.jit` засчитывается). Мы нашли способ получить те же инструкции без ручного PTX и хотим уточнить, можно ли так.

В Triton 3.6 уже есть генерация тензорных инструкций Turing: в `libtriton.so` есть `callMmaTuringFp16` и `callMmaTuringInt8` (`mma.sync.aligned.m16n8k8` для fp16 и `m8n8k16` для int8, раскладка `nvidia_mma` версии 2.1). Отключён только проход раскладки AccelerateMatmul для compute capability ниже 80 (сообщение «Dot op using MMA for compute capability 75 has been deprecated. It falls back to the FMA path»), поэтому `tl.dot` на T4 идёт в FMA.

Что делаем: в `apply()` регистрируем публичный хук Triton `triton.knobs.runtime.add_stages_inspection_hook`. Он меняет только стадию TTGIR и только для наших ядер (имя функции начинается с `tc_`, файлы в `solution/`): TTIR проходит штатный `make_ttgir` бэкенда с capability 80 (так AccelerateMatmul ставит MMA-раскладку), раскладка помечается как Turing (versionMinor 1), модуль обратно как cuda:75, дальше штатные LLVM-lowering, PTX и `ptxas --gpu-name=sm_75`. Ядра запускаются с `num_stages=1` (cp.async на sm_75 нет). Остальные ядра прогона, torch и харнесс компилируются как раньше.

Ядра наши (`solution/mma_gemm.py`), компилируются внутри замеряемого прогона с пустым кэшем, запускаются обычным `JITFunction.run` (харнесс их видит, считает и меряет CUDA-событиями), без CUDA graphs. Результат тот же `mma.sync`, что разрешён ответом 18, только код генерирует сам Triton.

Вопрос: можно ли так делать, и засчитываются ли такие ядра как свои Triton-ядра в 5 %?

---

Для Абзала: замеры T4 и чем это полезно, в `docs/notes/b3-turing-mma.md` (ветка `claude/b3-turing-mma`). До ответа всё за выключенным переключателем `SOUP_SOLUTION_MMA_INT8_BWD` (по умолчанию 0).
