# l2norm_kda

q/k 的 L2 归一化前向与反向，含 1-D / 2-D 两套实现共 4 个 kernel。

> 源文件 `l2norm_kda.py`（286 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `l2norm_fwd_kernel1` | 26 | — |
| `l2norm_bwd_kernel1` | 56 | — |
| `l2norm_fwd_kernel` | 85 | — |
| `l2norm_bwd_kernel` | 117 | — |

对外入口：`l2norm_fwd` (L142)、`l2norm_bwd` (L191)、`l2norm` (L264)

## 瓶颈数据（varlen 8k 用例）

本算子不含上述 12 个采集 kernel，**未纳入本次 msprof 逐算子采集**。

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/l2norm_kda.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
