# Бриф 5b (дополнение к брифу 5): профиль ядер шага текущего main на T4

Прогон soup-m10-prof (коммит d341d0c = main с влитым стартом L1, до влития int8-оффлоада P1), Kaggle T4, `harness/run_task.py --profile`: torch.profiler на 16 шагах (шаги 8–23), ожидание 6, прогрев 2. В той же сессии полный прогон: база 81,0 с, решение 48,5 с (s 1,671, медленная сессия), пик 395 MiB, eval 0,0 %. Профилированные шаги: стена 0,347–0,629 с, среднее **0,406 с/шаг**.

## Время GPU по категориям (сумма времени ядер и копий на шаг; потоки идут параллельно, поэтому сумма больше стены)

| Категория | мс/шаг | доля | запусков/шаг |
|---|---|---|---|
| Memcpy H2D (pinned → device) | **222,4** | 38 % | 313 |
| Memcpy D2H (device → pinned) | **76,7** | 13 % | 59 |
| GEMM cuBLAS (fp16 s1688 + int8 i8816) | 171,7 | 29 % | 742 |
| Распаковка NF4 (наши ядра) | 28,3 | 4,8 % | 223 |
| Поэлементные и редукции torch (dropout, копии, бинарные, fill…) | 24,8 | 4,2 % | 1 304 |
| Attention fmha (fwd + bwd) | 18,9 | 3,2 % | 84 |
| int8-ядра (наши: NF4→int8, SwiGLU bwd, квантование) | 17,5 | 3,0 % | 56 |
| RMSNorm (наши) | 10,3 | 1,8 % | 140 |
| SwiGLU forward (ядро примера набора) | 4,9 | 0,8 % | 28 |
| Голова (наши) | 4,6 | 0,8 % | 39 |
| RoPE (наши) | 2,2 | 0,4 % | 84 |
| Прочее наше | 1,3 | 0,2 % | 27 |
| Оптимизатор | 1,1 | 0,2 % | 26 |

**Вывод для обсуждения:** вычисления (всё, кроме копий) ≈ 286 мс/шаг, H2D ≈ 222 мс/шаг (шина занята ~55 % стены), D2H 77 мс. Стена 406 мс. Если бы копии полностью прятались под вычислениями, шаг был бы ~286 мс; разница **~120 мс/шаг (~7,7 с за 64 шага)** — это ожидания по цепочке зависимостей (вычисление ждёт копию веса/активации) и/или CPU на коротких шагах. Состав H2D по расчёту на N≈830: NF4 forward 0,65 ГБ + NF4 backward 0,65 + голова 0,445 + возврат gate|up 0,833 + входы слоёв 0,143 ≈ 2,72 ГБ ≈ 227 мс при 12 ГБ/с — совпадает с замером. После влития P1 (int8-оффлоад) возврат gate|up и D2H вдвое меньше: −35 мс H2D и −38 мс D2H по расчёту, стена в ABBA −1,64 с (−25 мс/шаг).

## Топ-45 ядер и копий (мс/шаг, запусков/шаг, имя)

| мс/шаг | запусков | ядро |
|---|---|---|
| 222.44 | 313.0 | `Memcpy HtoD (Pinned -> Device)` |
|  76.73 |  59.0 | `Memcpy DtoH (Device -> Pinned)` |
|  49.11 |  89.8 | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` |
|  38.99 |  56.0 | `void cutlass::Kernel2<cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16>(cutlass_75_tensorop_i8816gemm_s8_128x128_tn_a` |
|  16.47 |  21.0 | `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_tt` |
|  14.36 |  42.1 | `turing_s1688gemm_fp16_256x128_ldg8_stages_32x1_nn` |
|  13.74 |  84.0 | `nf4_dequant_kernel` |
|  12.17 |  28.0 | `fmha_cutlassB_f16_aligned_128x64_k128_sm75(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm75, cutlass:` |
|  11.15 |  28.0 | `int8_swiglu_bwd_kernel` |
|  10.77 |  84.0 | `nf4_int8_t_kernel` |
|  10.41 |   7.0 | `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_tn` |
|   7.99 |  45.5 | `turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn` |
|   6.77 |  56.0 | `fmha_cutlassF_f16_aligned_32x128_rf_sm75(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, t` |
|   6.34 |   7.0 | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tt` |
|   6.33 |  28.0 | `quant_t_int8_kernel` |
|   4.89 |  28.0 | `example_swiglu_fwd_kernel` |
|   4.56 |  38.0 | `head_chunk_kernel` |
|   4.14 |  57.0 | `rms_norm_rows_fwd` |
|   4.13 |  16.2 | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` |
|   4.09 |  56.0 | `void at::native::(anonymous namespace)::fused_dropout_kernel<c10::Half, float, unsigned int, -1, 1, bool>(at::cuda::deta` |
|   3.92 |  31.5 | `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_tn` |
|   3.84 | 224.0 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::T` |
|   3.74 |  55.0 | `nf4_dequant4_kernel` |
|   3.41 |  85.0 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, f` |
|   3.26 |  91.2 | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn` |
|   3.18 |  14.2 | `turing_s1688gemm_fp16_128x256_ldg8_stages_32x1_nn` |
|   2.79 |  28.0 | `rms_norm_rows_bwd_split` |
|   2.48 |  28.0 | `rms_norm_rows_bwd` |
|   2.29 |  88.0 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}:` |
|   2.15 |  84.0 | `rope_qk_kernel` |
|   1.82 |  15.8 | `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn` |
|   1.64 |  58.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda` |
|   1.61 |   3.5 | `turing_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_stages_32x1_tn` |
|   1.61 |  28.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, ` |
|   1.39 |   4.6 | `turing_fp16_s1688gemm_fp16_256x64_sliced1x2_ldg8_f2f_tn` |
|   1.34 |   7.0 | `turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn` |
|   1.33 |  27.0 | `masked_pair_sum_kernel` |
|   1.25 |  85.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorItera` |
|   1.24 |  56.0 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::Half, at::native::func_wrapper_t<c10::Half, at::native:` |
|   1.14 |  28.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::` |
|   1.12 |   7.0 | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` |
|   1.08 |  29.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int` |
|   1.05 |  28.0 | `turing_s1688gemm_fp16_128x128_ldg8_stages_32x1_nt` |
|   1.02 |   5.2 | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn` |
|   0.99 |  29.0 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}:` |

## Вопросы к брифу 5 по этому профилю

1. Где именно сидят ~120 мс/шаг ожиданий: какие копии оказываются на критическом пути (NF4 следующего слоя? голова? возврат gate|up в backward?), и что дешевле всего сделать в потолке 420 MiB: глубже префетч NF4 (второй слот на тип матрицы: +~20 MiB), порядок копий по ближайшему потребителю, разбиение больших копий, резидентность части слоёв, перенос распаковки/пересчёта под ожидание?
2. NF4 стримится дважды за шаг (1,3 ГБ, ~108 мс H2D). Есть ли разумный способ стримить его один раз (держать «разворот» forward → backward, переупорядочить вычисления backward), не поднимая память выше ~420 MiB?
3. 1 304 запуска поэлементных ядер torch на шаг (24,8 мс GPU + CPU на запуск): какие из них сливаются без смены RNG dropout (ответ 47 запрещает свой RNG), и стоит ли CUDA graph (ответ 17) ради них?
4. С учётом этого профиля: обнови оценку пути к 40 с и к 35 с при памяти 350–420 MiB.
