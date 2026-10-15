# chunk_delta_h_kda

分块递推状态 `h` 的前向构建与反向反传（blockdim64）。文件内 2 个 kernel，fwd/bwd 各一。

> 源文件 `chunk_delta_h_kda.py`（661 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 36 | ✅ |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 280 | ✅ |

对外入口：`chunk_gated_delta_rule_fwd_h` (L541)、`chunk_gated_delta_rule_bwd_dhu` (L601)

## 瓶颈数据（varlen 8k 用例）

固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，`safe_gate=True`，`lower_bound=-5.0`，bfloat16）

| kernel | 阶段 | 调用次数 | 总耗时(us) | 占比 | 瓶颈类别 |
|---|---|---|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | fwd | 2 | 3773 | 7.5% | vec bound |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | bwd | 1 | 2333 | 4.6% | vec bound |

流水线占比明细（ratio = 该单元活跃时间 / 核活跃时间，取各 block 最大值）：

| kernel | vec | cube | scalar | mte_max | 备注 |
|---|---|---|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 0.626 | 0.026 | 0.061 | 0.410 | vec 主导，构建递推状态 h |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 0.673 | 0.036 | 0.079 | 0.393 | vec 最高；store mte3(aiv)=0.393 次之 |

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/chunk_delta_h_kda.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
