# _shared

`kda/` 下被多个算子共同依赖的叶子模块，集中放这一处，避免每个算子文件夹各存一份。

| 模块 | 被几个算子 import | 作用 |
|---|---|---|
| `fla_utils.py` | 11 | 断言/误差工具、`tensor_cache`、`input_guard`、device 选择、autocast 包装 |
| `utils.py` | 8 | `prepare_lens` / `prepare_chunk_indices` / `prepare_chunk_offsets`（chunk 索引准备）|
| `op.py` | 7 | `exp` / `exp2` / `log` / `log2` / `gather` 的 kernel 级实现（按环境变量分支）|
| `softplus.py` | 1 | softplus 及其变体的 kernel 生成 |
| `beta_sigmoid_kda.py` | 1 | beta 的 sigmoid 融合实现 |
| `naive.py` | 0（仅测试用）| `naive_recurrent_kda` / `naive_chunk_kda`，测试参考实现 |

> 本目录模块**无包内依赖**（或仅依赖同目录 `fla_utils.py`），是依赖图的叶子。
> 上游 `attention/fla/utils.py`（提供 `RCP_LN2`）不在本目录，按镜像路径置于
> `triton_ascend/attention/fla/utils.py`。

来源：`triton-ascend-kernels` @ `kda_a5_v2`(636a2de) 的 `src/triton_ascend_kernels/attention/fla/kda/`。
改动仅为相对导入 → 绝对导入。
