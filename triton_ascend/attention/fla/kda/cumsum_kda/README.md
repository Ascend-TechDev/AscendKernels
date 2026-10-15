# cumsum_kda

分块 cumsum：local/global × scalar/vector 共 4 个 kernel。其中 `chunk_local_cumsum_vector_kernel` 被采集到。

> 源文件 `cumsum_kda.py`（468 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `chunk_local_cumsum_scalar_kernel` | 31 | — |
| `chunk_local_cumsum_vector_kernel` | 85 | ✅ |
| `chunk_global_cumsum_scalar_kernel` | 142 | — |
| `chunk_global_cumsum_vector_kernel` | 200 | — |

对外入口：`chunk_local_cumsum_scalar` (L246)、`chunk_local_cumsum_vector` (L283)、`chunk_global_cumsum_scalar` (L326)、`chunk_global_cumsum_vector` (L357)、`chunk_global_cumsum` (L391)、`chunk_local_cumsum` (L428)

## 瓶颈数据（varlen 8k 用例）

固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，`safe_gate=True`，`lower_bound=-5.0`，bfloat16）

| kernel | 阶段 | 调用次数 | 总耗时(us) | 占比 | 瓶颈类别 |
|---|---|---|---|---|---|
| `chunk_local_cumsum_vector_kernel` | bwd | 1 | 278 | 0.6% | vec bound |

流水线占比明细（ratio = 该单元活跃时间 / 核活跃时间，取各 block 最大值）：

| kernel | vec | cube | scalar | mte_max | 备注 |
|---|---|---|---|---|---|
| `chunk_local_cumsum_vector_kernel` | 0.989 | — | 0.203 | 0.719 | vec 几乎满载（98.9%），反向 cumsum 纯向量计算 |

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/cumsum_kda.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
