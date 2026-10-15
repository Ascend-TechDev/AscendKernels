# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch
import triton
import triton.language as tl

from triton_ascend.attention.fla.kda.chunk_delta_h_kda.chunk_delta_h_kda import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from triton_ascend.attention.fla.kda.chunk_intra.chunk_intra import chunk_kda_bwd_intra
from triton_ascend.attention.fla.kda.gate.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from triton_ascend.attention.fla.kda.wy_fast.wy_fast import recompute_w_u_fwd
from triton_ascend.attention.fla.kda.cumsum_kda.cumsum_kda import chunk_local_cumsum
from triton_ascend.attention.fla.kda._shared.utils import prepare_chunk_indices
from triton_ascend.attention.fla.utils import RCP_LN2
from triton_ascend.attention.fla.kda._shared.op import exp2
from triton_ascend.attention.fla.kda._shared.fla_utils import autotune_cache_kwargs

BK_LIST = [16, 32, 64]
BV_LIST = [16, 32, 64, 128]
CONST_TILING = [64]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({})],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    p_A = tl.make_block_ptr(A + (bos * HV + i_hv) * BT, (BT, T), (1, HV*BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, HV*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, b_v)
        # [BT, BV]
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_dA = tl.make_block_ptr(dA, (T, BT), (HV*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({'BK': 64, 'BV': 128, 'PRELOAD_DV': True})],
    key=['BT', 'HV', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PRELOAD_DV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K*V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K*V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    # FULL_TILE: host-proven alignment (T % BT == 0 and K % BK == 0 and
    # V % BV == 0). Under it every boundary check and the m_t/m_k masks are
    # all-true, so load/store mask evaluation compiles away entirely.
    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(() if FULL_TILE else (0,)))

    p_A = tl.make_block_ptr(A, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(() if FULL_TILE else (0, 1)))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    # dv is independent of i_k. Keep only dv resident across i_k to reduce
    # repeated global loads without retaining v_new and do in UB as well.
    if PRELOAD_DV and V // BV == 1:
        p_dv_pre0 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, 0), (BT, V), (1, 0))
        dv_pre0 = tl.load(p_dv_pre0, boundary_check=(() if FULL_TILE else (0, 1)))
        # Round 5: hoist the i_k-invariant b_v load to the prologue. b_v is
        # used only inside the i_k==0 scf.if, which sits AFTER the required
        # debug_barrier; loading it here lets MTE2 issue during iteration 0's
        # dots so the if-block starts immediately after the barrier clears
        # instead of waiting for a 16KB load.
        p_v_pre0 = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, 0), (BT, V), (1, 0))
        v_pre0 = tl.load(p_v_pre0, boundary_check=(() if FULL_TILE else (0, 1)))
    elif PRELOAD_DV and V // BV == 2:
        p_dv_pre0 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        dv_pre0 = tl.load(p_dv_pre0, boundary_check=(() if FULL_TILE else (0, 1)))
        p_dv_pre1 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, BV), (BT, BV), (1, 0))
        dv_pre1 = tl.load(p_dv_pre1, boundary_check=(() if FULL_TILE else (0, 1)))
    elif PRELOAD_DV and V // BV == 4:
        p_dv_pre0 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        dv_pre0 = tl.load(p_dv_pre0, boundary_check=(() if FULL_TILE else (0, 1)))
        p_dv_pre1 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, BV), (BT, BV), (1, 0))
        dv_pre1 = tl.load(p_dv_pre1, boundary_check=(() if FULL_TILE else (0, 1)))
        p_dv_pre2 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, 2 * BV), (BT, BV), (1, 0))
        dv_pre2 = tl.load(p_dv_pre2, boundary_check=(() if FULL_TILE else (0, 1)))
        p_dv_pre3 = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, 3 * BV), (BT, BV), (1, 0))
        dv_pre3 = tl.load(p_dv_pre3, boundary_check=(() if FULL_TILE else (0, 1)))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(() if FULL_TILE else (0, 1)))
        b_g = tl.load(p_g, boundary_check=(() if FULL_TILE else (0, 1))).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV*K + o_k
        if FULL_TILE:
            b_gn = tl.load(p_gn).to(tl.float32)
        else:
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        if PRELOAD_DV and V // BV == 1:
            # Round 6: hoist the pure-vector gate-chain front above the
            # required debug_barrier. Its inputs (b_g, b_gn, b_k, b_beta)
            # are all available here, and the barrier-gated post-barrier
            # segment is vec-gated (fresh profile: aiv_vec 0.628 top,
            # aic_mac 0.132; fresh TTIR: the two full-tile exp2s +
            # scalings sit after gpu.barrier). Issuing this vec work
            # pre-barrier overlaps it with the pre-barrier cube dots
            # instead of extending the post-barrier segment. Bit-identical
            # math; the barrier line stays in place and the scf.if dv2
            # block keeps its original b_dA/b_db accumulation order (its
            # dots still run after the barrier, before the gate-chain
            # tail).
            b_gk_exp = exp2(b_g)
            b_gb = b_gk_exp * b_beta[:, None]
            b_kg = b_k * b_gk_exp

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v_new = tl.load(p_v_new, boundary_check=(() if FULL_TILE else (0, 1)))
            b_do = tl.load(p_do, boundary_check=(() if FULL_TILE else (0, 1)))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(() if FULL_TILE else (0, 1)))
            b_dh = tl.load(p_dh, boundary_check=(() if FULL_TILE else (0, 1)))
            # [BT, BV] — use a resident dv tile when enabled.
            if PRELOAD_DV and V // BV == 1:
                b_dv = dv_pre0
            elif PRELOAD_DV and V // BV == 2:
                if i_v == 0:
                    b_dv = dv_pre0
                else:
                    b_dv = dv_pre1
            elif PRELOAD_DV and V // BV == 4:
                if i_v == 0:
                    b_dv = dv_pre0
                elif i_v == 1:
                    b_dv = dv_pre1
                elif i_v == 2:
                    b_dv = dv_pre2
                else:
                    b_dv = dv_pre3
            else:
                p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                b_dv = tl.load(p_dv, boundary_check=(() if FULL_TILE else (0, 1)))

            # Hoist b_h dtype conversion: b_do and b_v_new share the same
            # compute dtype, so convert b_h once for both b_dq and b_dw.
            b_h_cvt = b_h.to(b_do.dtype)
            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h_cvt)
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            # b_dv and b_v_new share compute dtype (both bf16);
            # the explicit .to() conversion is a no-op.
            b_dw += tl.dot(b_dv, b_h_cvt)
            if PRELOAD_DV and V // BV == 1:
                # Round 6 (cont.): the dot-output scales of the gate chain
                # still run pre-barrier — they depend on b_dq/b_dk/b_dgk,
                # which complete inside this loop body, and on b_gk_exp /
                # b_gb / b_kg hoisted above. Bit-identical math; the
                # required debug_barrier line below stays in place and the
                # scf.if dv2 block keeps its original accumulation order.
                b_dgk *= exp2(b_gn)
                b_dq = b_dq * b_gk_exp * scale
                if FULL_TILE:
                    b_dk = b_dk * exp2(b_gn[None, :] - b_g)
                else:
                    b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                if PRELOAD_DV and V // BV == 1:
                    # Round 5: b_v was hoisted to the prologue (v_pre0).
                    b_v = v_pre0
                elif FULL_TILE:
                    p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_v = tl.load(p_v)
                else:
                    p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_v = tl.load(
                        p_v,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )

                b_dA += tl.dot(b_dv, tl.trans(b_v))

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))

            if PRELOAD_DV and V // BV == 1:
                # Round 20: issue the b_dq store immediately post-barrier.
                # b_dq's value completes PRE-barrier (gate-chain scale above),
                # so its 16KB fp32 store needs no post-barrier dependency —
                # issuing it here lets MTE3 transfer during the post-barrier
                # dots (b_dA/b_dkgb) and the b_dw/b_dg vec chain instead of
                # queueing at the segment end behind them, shortening the
                # next PIPE_ALL barrier's MTE3 drain and the program tail.
                # Bit-identical: same address/value, pure store-issue reorder;
                # the required tl.debug_barrier() line and the dv2 scf.if
                # block are untouched; non-hot configs (where b_dq completes
                # in the post-loop gate chain) keep the original position.
                p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))

        if not (PRELOAD_DV and V // BV == 1):
            # Round 6: the gate-chain front below was hoisted above the
            # required debug_barrier for PRELOAD_DV and V//BV==1; other
            # configs keep the original post-loop position byte-identical.
            b_gk_exp = exp2(b_g)
            b_gb = b_gk_exp * b_beta[:, None]
            b_dgk *= exp2(b_gn)
            b_dq = b_dq * b_gk_exp * scale
            if FULL_TILE:
                b_dk = b_dk * exp2(b_gn[None, :] - b_g)
            else:
                b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)
            b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(() if FULL_TILE else (0, 1)))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dk = tl.make_block_ptr(dk, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        if not (PRELOAD_DV and V // BV == 1):
            # Round 20: hot config stores dq right after the required barrier
            # (value complete pre-barrier); non-hot configs keep the original
            # post-loop position byte-identical (b_dq completes in the
            # post-loop gate chain there).
            p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))

    if FULL_TILE:
        m_A = o_t[:, None] > o_t[None, :]
    else:
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    p_dA = tl.make_block_ptr(dA, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(() if FULL_TILE else (0, 1)))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(() if FULL_TILE else (0,)))


def chunk_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, HV, V = *k.shape, do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    CONST_TILING = 64
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
    dv = torch.empty_like(do)
    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_dAv[grid](
        q=q,
        k=k,
        v=v,
        A=A,
        do=do,
        dv=dv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dA, dv


def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # dq, dk are allocated at HV dimension; caller reduces to H if GVA
    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    # Aligned full-tile specialization: only valid when every tile axis divides
    # evenly. BK/BV mirror the single @triton.autotune config above. Var-len
    # chunks are per-chunk sized, so the specialization never applies there.
    full_tile = (
        cu_seqlens is None
        and T % BT == 0
        and K % 64 == 0
        and V % 128 == 0
    )
    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        FULL_TILE=full_tile,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    **kwargs,
):
    H, HV = q.shape[2], v.shape[2]
    G = HV // H

    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound
            )
        w, u, qg, kg = recompute_w_u_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]

    # dAqk = do @ v.T
    # dv = A @ do
    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate
    )

    # For GVA, reduce dq and dk from [B, T, HV, K] back to [B, T, H, K]
    if HV > H:
        dq = dq.view(*dq.shape[:2], H, G, dq.shape[-1]).sum(dim=3)
        dk = dk.view(*dk.shape[:2], H, G, dk.shape[-1]).sum(dim=3)

    dA, dbias = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound
        )

    return dq, dk, dv, db, dg, dh0, dA, dbias
