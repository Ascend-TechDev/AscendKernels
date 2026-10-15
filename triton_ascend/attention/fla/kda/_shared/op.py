# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from triton_ascend.attention.fla.kda._shared.fla_utils import IS_GATHER_SUPPORTED

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tldevice.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tldevice.fast_logf(x.to(tl.float32))
    @triton.jit
    def log2(x): return tldevice.fast_log2f(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tl.log(x.to(tl.float32))
    @triton.jit
    def log2(x): return tl.log2(x.to(tl.float32))


if not IS_GATHER_SUPPORTED:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather
