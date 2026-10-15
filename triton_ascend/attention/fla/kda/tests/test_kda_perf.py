# Copyright © 2026 Huawei Technologies Co., Ltd.

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from triton_ascend.attention.fla.kda import chunk_kda
from test_common import profiling_test


def get_device_type():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


device_type = get_device_type()
device = 'npu' if device_type == 'npu' else 'cuda'


@pytest.mark.parametrize(
    (
        "H", "HV", "D", "mask_p", "cu_seqlens", "dtype",
        "use_gate_in_kernel", "safe_gate", "disable_recompute", "chunk_size",
    ),
    [
        pytest.param(
            *test,
            id=(
                "H{}-HV{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}"
                "-safe_gate{}-disable_recompute{}-chunk_size{}"
            ).format(*test),
        )
        for test in [
            # (32, 32, 128, 0, [0, 2048, 4096, 8192], torch.bfloat16, True, True, False, 64), # intra_sub_chunk shape 4
            # (32, 32, 128, 0, [0, 2048, 4096, 8192], torch.bfloat16, True, False, False, 64), # other shape 4
            # (96, 96, 128, 0, [0, 2048, 4096, 8192], torch.bfloat16, True, True, False, 64), # shape 5

            # (32, 32, 128, 0, [0, 2048, 4096] + list(range(4097, 4608)), torch.bfloat16, True, True, False, 64) # intra_sub_chunk  shape 8
            # (32, 32, 128, 0, [0, 2048, 4096] + list(range(4097, 4608)), torch.bfloat16, True, False, False, 64) # other shape 8
            # (96, 96, 128, 0, [0, 1024, 2048] + list(range(4097, 4608)), torch.bfloat16, True, True, False, 64) # shape 9
            (96, 96, 128, 0, [0, 300, 700] + list(range(900, 1024)), torch.bfloat16, True, True, False, 64) # shape 10
        ]
    ],
)
def test_chunk_kda_varlen_perf(
    H: int,
    HV: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
    chunk_size: int,
):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype, device=device)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, HV, D), dtype=dtype, device=device)
    g = torch.randn(1, T, HV, D, dtype=torch.float if not use_gate_in_kernel else dtype, device=device)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, HV, 1, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(HV * D, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(g)
        g = g * (torch.rand_like(g) > mask_p)
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
    else:
        lower_bound = None

    beta = torch.rand(1, T, HV, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn((N, HV, D, D), dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    q_ = F.normalize(q.clone(), p=2, dim=-1).requires_grad_(True)
    k_ = k.clone().requires_grad_(True)
    v_ = v.clone().requires_grad_(True)
    g_ = g.clone().requires_grad_(True)
    beta_ = beta.clone().requires_grad_(True)
    h0_ = h0.clone().requires_grad_(True)

    def fn_triton():
        q_.grad = k_.grad = v_.grad = g_.grad = beta_.grad = h0_.grad = None
        tri, tri_ht = chunk_kda(
            q=q_, k=k_, v=v_, g=g_, beta=beta_,
            initial_state=h0_, output_final_state=True,
            cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=safe_gate, lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=disable_recompute,
            chunk_size=chunk_size,
        )
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)

    profiling_test(fn_triton, device_type, device)


@pytest.mark.parametrize(
    (
        "B", "T", "H", "HV", "D", "mask_p", "dtype",
        "use_gate_in_kernel", "safe_gate", "disable_recompute", "chunk_size",
    ),
    [
        pytest.param(
            *test,
            id=(
                "B{}-T{}-H{}-HV{}-D{}-mask_p{}-{}-gate{}"
                "-safe_gate{}-disable_recompute{}-chunk_size{}"
            ).format(*test),
        )
        for test in [
            # (2, 2048, 8, 8, 128, 0, torch.float16, True, True, False, 64), # intra_sub_chunk shape 1
            # (2, 2048, 8, 8, 128, 0, torch.float16, True, False, False, 64), # other shape 1
            # (1, 128, 2, 2, 64, 0, torch.float16, True, True, False, 64), # shape 2
            # (3, 512, 3, 3, 64, 0, torch.bfloat16, True, True, False, 64), # intra_sub_chunk shape 3
            # (3, 512, 3, 3, 64, 0, torch.bfloat16, True, False, False, 64), # other shape 3

            # NEW series
            # (2, 2048, 8, 8, 128, 0, torch.float16, True, True, False, 64), # intra_sub_chunk shape 1
            # (2, 2048, 8, 8, 128, 0, torch.float16, True, False, False, 64), # other shape 1
            # (10, 1024, 4, 4, 64, 0, torch.float16, True, True, False, 64), # new shape 2
            # (15, 1000, 3, 3, 64, 0, torch.float16, True, True, False, 64), # new shape 3 Compilation Err
            # (1, 4096, 32, 32, 128, 0, torch.bfloat16, True, True, False, 64), # shape 6
            # (3, 4096, 96, 96, 128, 0, torch.bfloat16, True, True, False, 64) # shape 7
            (1, 8192, 96, 96, 128, 0, torch.bfloat16, True, True, False, 64),
            (1, 16384, 96, 96, 128, 0, torch.bfloat16, True, True, False, 64),
        ]
    ],
)
def test_chunk_kda_perf(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    mask_p: float,
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
    chunk_size: int,
):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn((B, T, HV, D), dtype=dtype, device=device)
    g = torch.randn(B, T, HV, D, dtype=torch.float if not use_gate_in_kernel else dtype, device=device)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, HV, 1, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(HV * D, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(g)
        g = g * (torch.rand_like(g) > mask_p)
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
    else:
        lower_bound = None

    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn((B, HV, D, D), dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    q_ = F.normalize(q.clone(), p=2, dim=-1).requires_grad_(True)
    k_ = k.clone().requires_grad_(True)
    v_ = v.clone().requires_grad_(True)
    g_ = g.clone().requires_grad_(True)
    beta_ = beta.clone().requires_grad_(True)
    h0_ = h0.clone().requires_grad_(True)
    if use_gate_in_kernel:
        A_log_ = A_log.clone().requires_grad_(True)
        dt_bias_ = dt_bias.clone().requires_grad_(True)
    else:
        A_log_ = None
        dt_bias_ = None

    def fn_triton():
        q_.grad = k_.grad = v_.grad = g_.grad = beta_.grad = h0_.grad = None
        if use_gate_in_kernel:
            A_log_.grad = dt_bias_.grad = None
        tri, tri_ht = chunk_kda(
            q=q_,
            k=k_,
            v=v_,
            g=g_,
            beta=beta_,
            A_log=A_log_,
            dt_bias=dt_bias_,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            transpose_state_layout=True,
            disable_recompute=disable_recompute,
            chunk_size=chunk_size,
        )
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)

    profiling_test(fn_triton, device_type, device)
