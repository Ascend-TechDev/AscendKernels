# wy_fast

WY 表示（W/U）的重算与反向准备。

> 源文件 `wy_fast.py`（346 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `recompute_w_u_fwd_kda_kernel` | 30 | ✅ |
| `prepare_wy_repr_bwd_kda_kernel` | 129 | — |

对外入口：`recompute_w_u_fwd` (L241)、`prepare_wy_repr_bwd` (L290)

## 瓶颈数据（varlen 8k 用例）

固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，`safe_gate=True`，`lower_bound=-5.0`，bfloat16）

| kernel | 阶段 | 调用次数 | 总耗时(us) | 占比 | 瓶颈类别 |
|---|---|---|---|---|---|
| `recompute_w_u_fwd_kda_kernel` | fwd | 2 | 3097 | 6.2% | vec bound |

流水线占比明细（ratio = 该单元活跃时间 / 核活跃时间，取各 block 最大值）：

| kernel | vec | cube | scalar | mte_max | 备注 |
|---|---|---|---|---|---|
| `recompute_w_u_fwd_kda_kernel` | 0.542 | 0.021 | 0.514 | 0.324 | vec 略高，scalar(0.514) 与之极接近（WY 表示重算控制流重） |

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/wy_fast.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
