# chunk_intra_token_parallel

intra 的 token 并行变体——按 token 切分，替代 sub-chunk 的串行路径。

> 源文件 `chunk_intra_token_parallel.py`（176 行）。本目录是该算子的独立归档。

## 包含的 kernel

| kernel | 行号 | 是否纳入瓶颈采集 |
|---|---|---|
| `chunk_kda_fwd_kernel_intra_token_parallel` | 30 | — |

对外入口：`chunk_kda_fwd_intra_token_parallel` (L123)

## 瓶颈数据（varlen 8k 用例）

本算子不含上述 12 个采集 kernel，**未纳入本次 msprof 逐算子采集**。

## 测试

源仓无逐算子单测。本算子由端到端用例覆盖：
`triton_ascend/attention/fla/kda/tests/test_kda_chunk.py`（经公开入口 `chunk_kda` 驱动）。

## 来源

- 仓库 `triton-ascend-kernels`，分支 `kda_a5_v2`，commit `636a2de`
- 源路径 `src/triton_ascend_kernels/attention/fla/kda/chunk_intra_token_parallel.py`
- 改动：**仅将包内相对导入改写为仓内绝对导入**；文件头版权块与全部代码逻辑逐字节未动
