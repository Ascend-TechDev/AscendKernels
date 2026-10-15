# gate

KDA 门控：前向门控、反向反传、以及门控与 cumsum 的融合 kernel。文件内 3 个 kernel。

> 源文件 `gate.py`（458 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `kda_gate_fwd_kernel` | 84 | — |
| `kda_gate_bwd_kernel` | 136 | ✅ |
| `kda_gate_chunk_cumsum_vector_kernel` | 360 | ✅ |

对外入口：`naive_kda_gate` (L24)、`naive_kda_lowerbound_gate` (L55)、`kda_gate_fwd` (L202)、`kda_gate_bwd` (L233)、`fused_kda_gate` (L313)、`kda_gate_chunk_cumsum` (L419)

## 瓶颈数据（varlen 8k 用例）

固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，`safe_gate=True`，`lower_bound=-5.0`，bfloat16）

| kernel | 阶段 | 调用次数 | 总耗时(us) | 占比 | 瓶颈类别 |
|---|---|---|---|---|---|
| `kda_gate_bwd_kernel` | bwd | 1 | 302 | 0.6% | MTE bound |
| `kda_gate_chunk_cumsum_vector_kernel` | fwd | 2 | 785 | 1.6% | vec bound |

流水线占比明细（ratio = 该单元活跃时间 / 核活跃时间，取各 block 最大值）：

| kernel | vec | cube | scalar | mte_max | 备注 |
|---|---|---|---|---|---|
| `kda_gate_bwd_kernel` | 0.435 | — | 0.102 | 0.562 | （aiv_mte3=0.147） |
| `kda_gate_chunk_cumsum_vector_kernel` | 0.972 | — | 0.159 | 0.438 | vec 占 97%，门控激活+cumsum 融合 |

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/gate.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
