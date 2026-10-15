# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch

# Vendored from fla.ops.common.gate (fused_beta_sigmoid).
# Algorithm is identical to upstream: y = scale * sigmoid(x); the backward
# returns dy * scale * sigmoid(x) * (1 - sigmoid(x)).
# A pure torch implementation is used here (element-wise) which is bit-for-bit
# equivalent to the upstream fused triton kernel.


def fused_beta_sigmoid_fwd(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (torch.sigmoid(x.float()) * scale).to(x.dtype)


def fused_beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    x = x.float()
    s = torch.sigmoid(x)
    return (dy.float() * scale * s * (1.0 - s)).to(x.dtype)


def fused_beta_sigmoid(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``scale * sigmoid(x)``.

    Used to map raw beta logits into ``[0, scale]`` before launching the chunk
    kernel when ``use_beta_sigmoid_in_kernel=True``.
    """
    return fused_beta_sigmoid_fwd(x, scale)
