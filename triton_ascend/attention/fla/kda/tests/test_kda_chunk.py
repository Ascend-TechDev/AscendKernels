# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import math
import pytest
import torch
import torch.nn.functional as F

from triton_ascend.attention.fla.kda import chunk_kda
from triton_ascend.attention.fla.kda.gate.gate import naive_kda_gate, naive_kda_lowerbound_gate
from triton_ascend.attention.fla.kda._shared.naive import naive_recurrent_kda
from triton_ascend.attention.fla.kda._shared.fla_utils import assert_close, device

# Actual deployment constants
D = 128
H_VALUES = [32, 64]


# ---------------------------------------------------------------------------
# Helpers: mimic Megatron's parameter initialization
# ---------------------------------------------------------------------------

def megatron_init_A_log(num_heads, A_init_range=(1, 16)):
    """A_log as initialized in KimiDeltaAttention.__init__ (kda.py:308-312)."""
    A = torch.empty(num_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
    return torch.log(A)


def megatron_init_dt_bias(qk_dim, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
    """dt_bias as initialized in KimiDeltaAttention.__init__ (kda.py:286-299)."""
    dt = torch.exp(
        torch.rand(qk_dim, device=device, dtype=torch.float32)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    return inv_dt


def megatron_chunk_kda(q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound,
                       cu_seqlens=None, cu_seqlens_cpu=None):
    """Call chunk_kda exactly as Megatron does (kda.py:794-809)."""
    return chunk_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )


# ---------------------------------------------------------------------------
# Test 1: Forward-only correctness (equal-length batch)
#
# No backward → can afford larger T. Reference via naive_recurrent_kda.
# Variable axes: H × safe_gate × lower_bound × dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("B", "T", "H", "safe_gate", "lower_bound", "dtype"),
    [
        pytest.param(
            *t,
            id="B{}-T{}-H{}-safe_gate{}-lb{}-{}".format(*t),
        )
        for t in [
            # safe_gate=False
            (1, 512, 32, False, None, torch.bfloat16),
            (1, 512, 64, False, None, torch.bfloat16),
            (2, 256, 32, False, None, torch.float16),
            # safe_gate=True, lower_bound=-5.0
            (1, 512, 32, True, -5.0, torch.bfloat16),
            (1, 512, 64, True, -5.0, torch.float16),
            # safe_gate=True, lower_bound=-3.0
            (1, 256, 32, True, -3.0, torch.bfloat16),
            # safe_gate=True, lower_bound=-1.0
            (1, 256, 64, True, -1.0, torch.bfloat16),
        ]
    ],
)
def test_megatron_forward(
    B: int, T: int, H: int,
    safe_gate: bool, lower_bound: float, dtype: torch.dtype,
):
    """Verify chunk_kda forward output against naive reference (no backward)."""
    torch.manual_seed(42)
    naive_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    gate_kwargs = dict(lower_bound=lower_bound) if safe_gate else {}

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    A_log = megatron_init_A_log(H)
    dt_bias = megatron_init_dt_bias(H * D)

    # Reference
    g_ref = naive_gate_fn(g.clone().float(), A_log.float(), dt_bias.float(), **gate_kwargs)
    ref, _ = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g_ref, beta=beta.clone(),
        scale=None, initial_state=None, output_final_state=False,
    )

    # Triton (Megatron convention)
    tri, _ = megatron_chunk_kda(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        A_log.clone(), dt_bias.clone(), safe_gate, lower_bound,
    )

    assert_close("o", ref, tri, 0.005)


# ---------------------------------------------------------------------------
# Test 2: Forward + Backward correctness (equal-length batch, training path)
#
# naive_recurrent_kda loops over T → use smaller T for H=32/64 to keep
# runtime reasonable.
# Variable axes: H × safe_gate × lower_bound × dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("B", "T", "H", "safe_gate", "lower_bound", "dtype"),
    [
        pytest.param(
            *t,
            id="B{}-T{}-H{}-safe_gate{}-lb{}-{}".format(*t),
        )
        for t in [
            # safe_gate=False
            (1, 256, 32, False, None, torch.bfloat16),
            (1, 256, 64, False, None, torch.bfloat16),
            (2, 128, 32, False, None, torch.float16),
            # safe_gate=True, lower_bound=-5.0 (boundary)
            (1, 256, 32, True, -5.0, torch.bfloat16),
            (1, 256, 64, True, -5.0, torch.float16),
            # safe_gate=True, lower_bound=-3.0 (mid-range)
            (1, 128, 32, True, -3.0, torch.bfloat16),
            # safe_gate=True, lower_bound=-1.0 (near zero)
            (1, 128, 64, True, -1.0, torch.bfloat16),
            # 8k input
            (1, 8192, 32, True, -5.0, torch.bfloat16),
            (1, 8192, 32, False, None, torch.bfloat16),
        ]
    ],
)
def test_megatron_backward(
    B: int, T: int, H: int,
    safe_gate: bool, lower_bound: float, dtype: torch.dtype,
):
    """Verify chunk_kda forward output and backward gradients against naive reference."""
    torch.manual_seed(42)
    naive_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    gate_kwargs = dict(lower_bound=lower_bound) if safe_gate else {}

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    A_log = megatron_init_A_log(H)
    dt_bias = megatron_init_dt_bias(H * D)

    q, k, v, g, beta = (x.requires_grad_(True) for x in (q, k, v, g, beta))
    A_log = A_log.requires_grad_(True)
    dt_bias = dt_bias.requires_grad_(True)

    do = torch.randn(B, T, H, D, dtype=dtype, device=device)

    # --- Reference (naive_recurrent_kda) ---
    g_ref = naive_gate_fn(g.clone(), A_log, dt_bias, **gate_kwargs)
    ref, _ = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g_ref, beta=beta.clone(),
        scale=None, initial_state=None, output_final_state=False,
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dq, ref_dk, ref_dv, ref_dg, ref_db = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    ref_dA, ref_dbias = A_log.grad.clone(), dt_bias.grad.clone()
    q.grad = k.grad = v.grad = g.grad = beta.grad = A_log.grad = dt_bias.grad = None

    # --- Triton (chunk_kda, Megatron convention) ---
    tri, _ = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        A_log=A_log.clone(), dt_bias=dt_bias.clone(),
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=safe_gate, lower_bound=lower_bound,
    )
    (tri * do).sum().backward(retain_graph=True)

    tri_dq, tri_dk, tri_dv, tri_dg, tri_db = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    tri_dA, tri_dbias = A_log.grad.clone(), dt_bias.grad.clone()
    q.grad = k.grad = v.grad = g.grad = beta.grad = A_log.grad = dt_bias.grad = None

    assert_close("o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("db", ref_db, tri_db, 0.02)
    assert_close("dA", ref_dA, tri_dA, 0.015)
    assert_close("dbias", ref_dbias, tri_dbias, 0.008)


# ---------------------------------------------------------------------------
# Test 3: Variable-length (packed sequences) forward + backward
#
# Variable axes: H × safe_gate × lower_bound × cu_seqlens × dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("H", "cu_seqlens", "safe_gate", "lower_bound", "dtype"),
    [
        pytest.param(
            *t,
            id="H{}-cu{}-safe_gate{}-lb{}-{}".format(*t),
        )
        for t in [
            # safe_gate=False
            (32, [0, 128, 256], False, None, torch.bfloat16),
            (64, [0, 64, 192, 256], False, None, torch.bfloat16),
            (32, [0, 15, 100, 200, 384], False, None, torch.float16),
            # safe_gate=True, lower_bound=-5.0
            (32, [0, 128, 256, 512], True, -5.0, torch.bfloat16),
            (64, [0, 128, 256], True, -5.0, torch.float16),
            # safe_gate=True, lower_bound=-3.0
            (32, [0, 128, 256], True, -3.0, torch.bfloat16),
            # 8k input
            (32, [0, 2048, 4096, 8192], True, -5.0, torch.bfloat16),
            (32, [0, 2048, 4096, 8192], False, None, torch.bfloat16),
            (2, [0, 477] + list(range(478, 627)), False, None, torch.bfloat16),
        ]
    ],
)
def test_megatron_varlen(
    H: int, cu_seqlens: list,
    safe_gate: bool, lower_bound: float, dtype: torch.dtype,
):
    """Verify chunk_kda with packed variable-length sequences as used in Megatron."""
    torch.manual_seed(42)
    naive_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    gate_kwargs = dict(lower_bound=lower_bound) if safe_gate else {}

    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
    cu_seqlens_cpu = cu_seqlens_t.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn(1, T, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    g = torch.randn(1, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(1, T, H, dtype=torch.float32, device=device).sigmoid()

    A_log = megatron_init_A_log(H)
    dt_bias = megatron_init_dt_bias(H * D)

    q, k, v, g, beta = (x.requires_grad_(True) for x in (q, k, v, g, beta))
    A_log = A_log.requires_grad_(True)
    dt_bias = dt_bias.requires_grad_(True)

    do = torch.randn(1, T, H, D, dtype=dtype, device=device)

    # --- Reference: loop over sequences ---
    ref_list = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        g_i = naive_gate_fn(g[:, s:e].clone().float(), A_log.float(), dt_bias.float(),
                            **gate_kwargs)
        ref_i, _ = naive_recurrent_kda(
            q=F.normalize(q[:, s:e].clone(), p=2, dim=-1),
            k=F.normalize(k[:, s:e].clone(), p=2, dim=-1),
            v=v[:, s:e].clone(), g=g_i, beta=beta[:, s:e].clone(),
            scale=None, initial_state=None, output_final_state=False,
        )
        ref_list.append(ref_i)
    ref = torch.cat(ref_list, 1)

    (ref * do).sum().backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    ref_dA, ref_dbias = A_log.grad.clone(), dt_bias.grad.clone()
    q.grad = k.grad = v.grad = g.grad = beta.grad = A_log.grad = dt_bias.grad = None

    # --- Triton: chunk_kda with cu_seqlens ---
    tri, _ = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        A_log=A_log.clone(), dt_bias=dt_bias.clone(),
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=safe_gate, lower_bound=lower_bound,
        cu_seqlens=cu_seqlens_t, cu_seqlens_cpu=cu_seqlens_cpu,
    )
    (tri * do).sum().backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    tri_dA, tri_dbias = A_log.grad.clone(), dt_bias.grad.clone()
    q.grad = k.grad = v.grad = g.grad = beta.grad = A_log.grad = dt_bias.grad = None

    assert_close("o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("db", ref_db, tri_db, 0.015)
    assert_close("dA", ref_dA, tri_dA, 0.015)
    assert_close("dbias", ref_dbias, tri_dbias, 0.008)


# ---------------------------------------------------------------------------
# Test 4: A_log / dt_bias initialization range fidelity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("H", "A_init_range", "dt_min", "dt_max"),
    [
        pytest.param(
            *t, id="H{}-A{}-dtmin{}-dtmax{}".format(*t),
        )
        for t in [
            (32, (1, 16), 0.001, 0.1),
            (64, (1, 16), 0.001, 0.1),
        ]
    ],
)
def test_megatron_param_init_ranges(
    H: int, A_init_range: tuple, dt_min: float, dt_max: float,
):
    """Verify that Megatron-style A_log / dt_bias initializations stay in expected ranges."""
    torch.manual_seed(42)

    A_log = megatron_init_A_log(H, A_init_range)
    A = A_log.exp()
    assert (A >= A_init_range[0]).all() and (A <= A_init_range[1]).all(), \
        f"A values out of range: {A.min().item():.4f} - {A.max().item():.4f}"

    dt_bias = megatron_init_dt_bias(H * D, dt_min, dt_max)
    dt = F.softplus(dt_bias)
    assert (dt >= dt_min - 1e-6).all() and (dt <= dt_max + 1e-6).all(), \
        f"dt values out of range: {dt.min().item():.6f} - {dt.max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 5: Megatron-realistic end-to-end shapes (smoke test)
#
# No naive reference; only checks shape, finiteness, no crash.
# Can afford full-scale B/T.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("B", "T", "H", "dtype"),
    [
        pytest.param(
            *t, id="B{}-T{}-H{}-{}".format(*t),
        )
        for t in [
            (1, 2048, 32, torch.bfloat16),
            (1, 4096, 32, torch.bfloat16),
            (1, 2048, 64, torch.bfloat16),
            (2, 1024, 32, torch.bfloat16),
            (1, 8192, 32, torch.bfloat16),
        ]
    ],
)
def test_megatron_realistic_shapes(
    B: int, T: int, H: int, dtype: torch.dtype,
):
    """Smoke test with realistic Megatron model shapes: no crash, finite output, correct shape."""
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    A_log = megatron_init_A_log(H)
    dt_bias = megatron_init_dt_bias(H * D)

    out, final_state = megatron_chunk_kda(
        q, k, v, g, beta, A_log, dt_bias,
        safe_gate=False, lower_bound=None,
    )

    assert out.shape == (B, T, H, D), f"Shape mismatch: {out.shape}"
    assert final_state is None
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ---------------------------------------------------------------------------
# Test 6: Determinism check (multiple runs produce identical results)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [32, 64])
def test_megatron_determinism(H: int):
    """Verify chunk_kda is deterministic across multiple runs with same inputs."""
    B, T = 1, 256
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    A_log = megatron_init_A_log(H)
    dt_bias = megatron_init_dt_bias(H * D)

    out1, _ = megatron_chunk_kda(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        A_log.clone(), dt_bias.clone(), safe_gate=False, lower_bound=None,
    )
    out2, _ = megatron_chunk_kda(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        A_log.clone(), dt_bias.clone(), safe_gate=False, lower_bound=None,
    )

    assert torch.equal(out1, out2), "chunk_kda is not deterministic"


# ---------------------------------------------------------------------------
# Test 7: Accuracy at the large deployment shapes from test_kda_perf
#
# Forward-only (the naive recurrent reference loops over T, so a backward pass
# at T=16384/H=96 would be prohibitive). Exercises the fused forward path used
# by the perf cases: use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel,
# use_gate_in_kernel + safe_gate, transpose_state_layout (state_v_first) with a
# non-zero initial_state. Reference: naive_recurrent_kda.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("T",),
    [
        pytest.param(8192, id="T8192"),
        pytest.param(16384, id="T16384"),
    ],
)
def test_megatron_perf_shapes_forward(T: int):
    """Accuracy check for the two test_chunk_kda_perf shapes (B=1, H=96, D=128,
    bf16, gate + safe_gate) against the naive recurrent reference."""
    torch.manual_seed(42)

    B, H, D = 1, 96, 128
    HV = H
    dtype = torch.bfloat16
    lower_bound = -5.0
    chunk_size = 64

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, HV, dtype=dtype, device=device)

    A_log = megatron_init_A_log(HV)
    dt_bias = megatron_init_dt_bias(HV * D)
    # transpose_state_layout=True: h0 is interpreted as V-first [B, HV, V, K]
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    # --- Reference (naive_recurrent_kda) ---
    # use_qk_l2norm_in_kernel  -> normalize q, k
    # use_beta_sigmoid_in_kernel -> sigmoid(beta) (scale=1.0)
    # use_gate_in_kernel + safe_gate -> naive_kda_lowerbound_gate
    q_ref = F.normalize(q.clone(), p=2, dim=-1)
    k_ref = F.normalize(k.clone(), p=2, dim=-1)
    beta_ref = torch.sigmoid(beta.clone().float()).to(dtype)
    g_ref = naive_kda_lowerbound_gate(g.clone().float(), A_log.float(), dt_bias.float(),
                                      lower_bound=lower_bound)
    # naive_recurrent_kda keeps state in K-first [B, HV, K, V]
    ref_o, ref_S = naive_recurrent_kda(
        q=q_ref, k=k_ref, v=v.clone(), g=g_ref, beta=beta_ref,
        scale=None, initial_state=h0.clone().transpose(-1, -2),
        output_final_state=True,
    )

    # --- Triton (perf-path flags) ---
    tri_o, tri_ht = chunk_kda(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
        A_log=A_log.clone(), dt_bias=dt_bias.clone(),
        initial_state=h0.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=lower_bound,
        transpose_state_layout=True,
        chunk_size=chunk_size,
    )

    assert_close("o", ref_o, tri_o, 0.005)
    # tri_ht is V-first [B, HV, V, K]; ref_S is K-first [B, HV, K, V]
    assert_close("ht", ref_S.transpose(-1, -2), tri_ht, 0.01)


# ---------------------------------------------------------------------------
# Test 8: Backward accuracy for the test_kda_perf config (H=96, perf-path flags)
#
# Same fused path as Test 7 (use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel,
# use_gate_in_kernel + safe_gate, transpose_state_layout/state_v_first with a
# non-zero initial_state), plus full gradient checking.
#
# Why smaller T: naive_recurrent_kda builds an O(T) autograd graph for backward
# (~[B,HV,K,V] saved per step), which OOMs at T=8192/H=96. The gradient code
# path is shape-independent, so a moderate T validates it; forward accuracy at
# the full perf shapes (T=8192/16384) is covered by Test 7.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("T",),
    [
        # pytest.param(256, id="T256"),
        # pytest.param(512, id="T512"),
        pytest.param(8192, id="T8192"),
        pytest.param(16384, id="T16384"),
    ],
)
def test_megatron_perf_shapes_backward(T: int):
    """Forward + backward accuracy for the test_kda_perf config (B=1, H=96,
    D=128, bf16, gate + safe_gate) against naive_recurrent_kda, at a T that
    keeps the naive backward graph memory-feasible."""
    torch.manual_seed(42)

    B, H, D = 1, 96, 128
    HV = H
    dtype = torch.bfloat16
    lower_bound = -5.0
    chunk_size = 64

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, HV, dtype=dtype, device=device)
    A_log = megatron_init_A_log(HV)
    dt_bias = megatron_init_dt_bias(HV * D)
    # transpose_state_layout=True: h0 is interpreted as V-first [B, HV, V, K]
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    q, k, v, g, beta = (x.requires_grad_(True) for x in (q, k, v, g, beta))
    A_log = A_log.requires_grad_(True)
    dt_bias = dt_bias.requires_grad_(True)
    h0 = h0.requires_grad_(True)

    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    dht = torch.rand_like(h0)   # V-first, matches tri_ht

    # --- Reference (naive_recurrent_kda) ---
    # use_qk_l2norm_in_kernel -> normalize q,k ; use_beta_sigmoid_in_kernel -> sigmoid(beta)
    # use_gate_in_kernel + safe_gate -> naive_kda_lowerbound_gate
    # state_v_first -> naive keeps K-first state, so transpose h0 in / ref_S out
    g_ref = naive_kda_lowerbound_gate(g.clone().float(), A_log, dt_bias, lower_bound=lower_bound)
    ref_o, ref_S = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g_ref,
        beta=torch.sigmoid(beta.clone().float()).to(dtype),
        scale=None, initial_state=h0.clone().transpose(-1, -2),
        output_final_state=True,
    )
    ((ref_o * do).sum() + (ref_S * dht.transpose(-1, -2)).sum()).backward()

    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    ref_dA, ref_dbias, ref_dh0 = A_log.grad.clone(), dt_bias.grad.clone(), h0.grad.clone()
    ref_o = ref_o.clone()
    ref_S = ref_S.clone()
    q.grad = k.grad = v.grad = g.grad = beta.grad = A_log.grad = dt_bias.grad = h0.grad = None

    # --- Triton (perf-path flags) ---
    tri_o, tri_ht = chunk_kda(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
        A_log=A_log.clone(), dt_bias=dt_bias.clone(),
        initial_state=h0.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=lower_bound,
        transpose_state_layout=True,
        chunk_size=chunk_size,
    )
    ((tri_o * do).sum() + (tri_ht * dht).sum()).backward()

    tri_dq, tri_dk, tri_dv, tri_dg, tri_dbeta = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), beta.grad.clone()
    )
    tri_dA, tri_dbias, tri_dh0 = A_log.grad.clone(), dt_bias.grad.clone(), h0.grad.clone()

    assert_close("o", ref_o, tri_o, 0.005)
    assert_close("ht", ref_S.transpose(-1, -2), tri_ht, 0.01)
    assert_close("dq", ref_dq, tri_dq, 0.01)
    assert_close("dk", ref_dk, tri_dk, 0.01)
    assert_close("dv", ref_dv, tri_dv, 0.01)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 0.02)
    assert_close("dA", ref_dA, tri_dA, 0.015)
    assert_close("dbias", ref_dbias, tri_dbias, 0.01)
    assert_close("dh0", ref_dh0, tri_dh0, 0.02)
