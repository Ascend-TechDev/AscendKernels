# chunk_bwd

KDA 反向的两个重头 kernel：`dAv` 与 `wy/dq/dk/dg` 融合反传。

> 源文件 `chunk_bwd.py`（688 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `chunk_kda_bwd_kernel_dAv` | 38 | ✅ |
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | 114 | ✅ |

对外入口：`chunk_kda_bwd_dAv` (L413)、`chunk_kda_bwd_wy_dqkg_fused` (L459)、`chunk_kda_bwd` (L535)

## 瓶颈数据（varlen 8k 用例）

固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，`safe_gate=True`，`lower_bound=-5.0`，bfloat16）

| kernel | 阶段 | 调用次数 | 总耗时(us) | 占比 | 瓶颈类别 |
|---|---|---|---|---|---|
| `chunk_kda_bwd_kernel_dAv` | bwd | 1 | 539 | 1.1% | vec bound |
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | bwd | 1 | 6777 | 13.5% | vec bound |

流水线占比明细（ratio = 该单元活跃时间 / 核活跃时间，取各 block 最大值）：

| kernel | vec | cube | scalar | mte_max | 备注 |
|---|---|---|---|---|---|
| `chunk_kda_bwd_kernel_dAv` | 0.496 | 0.047 | 0.065 | 0.482 | vec 最高但偏低；aic_mte2=0.482、fixpipe=0.438 极接近（计算/访存均衡） |
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | 0.841 | 0.022 | 0.267 | 0.276 | vec 占 84%，融合反传大核，cube/mte 均低 |

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/chunk_bwd.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
