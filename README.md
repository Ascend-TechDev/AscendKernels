# AscendKernels

Ascend NPU operator implementations.

跨 **DSL** 的昇腾（Ascend NPU）算子集合。同一种算子可能有多套 DSL 实现，
按 DSL 分目录并列存放，便于横向对比。

## 目录约定

```
<dsl>/<category>/<subcategory>/<operator>/
```

- `<dsl>`：算子实现所用的编程模型，目前有 `triton_ascend`，后续会有 `ascendc`、`tilelang`
- `<category>` / `<subcategory>`：算子类别，与上游项目保持一致
- `<operator>`：**一个源文件一个文件夹**，内含实现、README 与相关测试

每个算子文件夹固定包含：

```
<operator>/
├── <operator>.py   # 算子实现
└── README.md       # 功能、包含的 kernel 清单、性能数据、来源回溯
```

## 当前内容

### `triton_ascend/attention/fla/kda/` —— KDA（Kimi Delta Attention）算子套件

共 **11 个算子、23 个 Triton kernel**，其中 7 个算子有 msprof 逐算子采集数据。

| 算子 | kernel 数 | 含采集数据 | 并入原分析文档 |
|---|---|---|---|
| [`chunk`](triton_ascend/attention/fla/kda/chunk/) | 0 | — |  |
| [`chunk_bwd`](triton_ascend/attention/fla/kda/chunk_bwd/) | 2 | ✅ |  |
| [`chunk_delta_h_kda`](triton_ascend/attention/fla/kda/chunk_delta_h_kda/) | 2 | ✅ |  |
| [`chunk_fwd`](triton_ascend/attention/fla/kda/chunk_fwd/) | 0 | — |  |
| [`chunk_intra`](triton_ascend/attention/fla/kda/chunk_intra/) | 4 | ✅ | 是 |
| [`chunk_intra_token_parallel`](triton_ascend/attention/fla/kda/chunk_intra_token_parallel/) | 1 | — |  |
| [`cumsum_kda`](triton_ascend/attention/fla/kda/cumsum_kda/) | 4 | ✅ |  |
| [`gate`](triton_ascend/attention/fla/kda/gate/) | 3 | ✅ |  |
| [`gla_chunk_kda`](triton_ascend/attention/fla/kda/gla_chunk_kda/) | 1 | ✅ |  |
| [`l2norm_kda`](triton_ascend/attention/fla/kda/l2norm_kda/) | 4 | — |  |
| [`wy_fast`](triton_ascend/attention/fla/kda/wy_fast/) | 2 | ✅ |  |

- 公开入口：`triton_ascend.attention.fla.kda.chunk_kda`，端到端编排 KDA 的 forward/backward
- 共享依赖收拢在 [`kda/_shared/`](triton_ascend/attention/fla/kda/_shared/)
- 测试位于 [`kda/tests/`](triton_ascend/attention/fla/kda/tests/)，41 个用例

性能数据的固定用例：`test_megatron_varlen[H32-cu[0, 2048, 4096, 8192]-safe_gateTrue-lb-5.0-torch.bfloat16]`
（varlen 3 段 2048/2048/4096，总 T=8192，H=32，D=128，bfloat16）。

## 安装与使用

```bash
pip install -e .
```

```python
from triton_ascend.attention.fla.kda import chunk_kda
```

## 来源与许可

本项目代码迁自 `triton-ascend-kernels`（分支 `kda_a5_v2`，commit `636a2de`）。

**迁移改动仅为：将包内相对导入改写为仓内绝对导入。** 文件头的版权块与全部代码逻辑逐字节未动。
每个算子的 README 末尾均记录其源路径与改动说明。

- 本项目基于 [MIT](LICENSE) 许可
- FLA 派生代码来自 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（MIT，
  Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li），源文件头部保留原始声明
- 其余上游归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)
