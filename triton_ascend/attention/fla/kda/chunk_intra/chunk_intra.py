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
import triton.language.extra.cann.extension as tle

from triton_ascend.attention.fla.kda.chunk_intra_token_parallel.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from triton_ascend.attention.fla.kda.wy_fast.wy_fast import recompute_w_u_fwd
from triton_ascend.attention.fla.kda._shared.utils import prepare_chunk_indices
from triton_ascend.attention.fla.kda._shared.op import exp2, gather
from triton_ascend.attention.fla.kda._shared.fla_utils import IS_GATHER_SUPPORTED, autotune_cache_kwargs

# TF32 is an NVIDIA-specific compute mode; not applicable on Ascend.
IS_TF32_SUPPORTED = False

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128})
    ],
    key=["H", "HV", "K", "BT", "BC", "NC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    # for i_tc1 >= T (partial last chunk with <= BC rows) the off-diagonal blocks
    # do not exist; the guarded work below is pure waste there (all stores masked,
    # Akk zero-initialized) so skip the tile loads, dots, and solve entirely.
    if i_tc1 < T:
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K

            p_k0 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            p_g0 = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
            b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if NC >= 3 and i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if NC >= 4 and i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b1 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3 and i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if NC >= 4 and i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (HV*BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    # b_Ai11 (and b_Ai22/b_Ai33) cover rows i_tc1.. (i_tc2.., i_tc3..): for chunks
    # with i_tc1 >= T (<= BC real rows) no real diagonal rows exist there, so skip
    # the load (zeros-init keeps the solve/substitution compile-safe; their stores
    # are guarded identically below).
    b_Ai11 = tl.zeros([BC, BC], dtype=tl.float32)
    if i_tc1 < T:
        p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (HV*BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.zeros([BC, BC], dtype=tl.float32)
    if NC >= 3:
        if i_tc2 < T:
            p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (HV*BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
            b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.zeros([BC, BC], dtype=tl.float32)
    if NC >= 4:
        if i_tc3 < T:
            p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (HV*BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
            b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        if NC >= 3:
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        if NC >= 4:
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
                b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        if NC >= 3:
            b_Ai22 += m_I
        if NC >= 4:
            b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    if i_tc1 < T:
        # we used tf32 to maintain matrix inverse's precision whenever possible.
        b_Ai10 = -tl.dot(
            tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai00,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )

        if NC >= 3:
            b_Ai21 = -tl.dot(
                tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
                b_Ai11,
                input_precision=SOLVE_TRIL_DOT_PRECISION
            )
            b_Ai20 = -tl.dot(
                b_Ai22,
                tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
                tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
                input_precision=SOLVE_TRIL_DOT_PRECISION
            )
        if NC >= 4:
            b_Ai32 = -tl.dot(
                tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
                b_Ai22,
                input_precision=SOLVE_TRIL_DOT_PRECISION
            )
            b_Ai31 = -tl.dot(
                b_Ai33,
                tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
                tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
                input_precision=SOLVE_TRIL_DOT_PRECISION
            )
            b_Ai30 = -tl.dot(
                b_Ai33,
                tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
                tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
                tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
                input_precision=SOLVE_TRIL_DOT_PRECISION
            )

        # store off-diagonal Akk_inv blocks to Akk
        p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        if NC >= 3:
            p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
            p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
            tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        if NC >= 4:
            p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
            p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
            p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
            tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    ################################################################################
    # store diagonal Akk_inv blocks to Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc1 < T:
        p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        if i_tc2 < T:
            p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
            tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        if i_tc3 < T:
            p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (HV*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))
            tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    # num_warps=8 / num_stages=1 
    configs=[triton.Config({})],
    key=['BK', 'NC', 'BT', 'HV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    I_I: tl.constexpr,
    FULL_CHUNK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_i = I_I

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * all + bos) * HV + i_hv

    p_g = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_b0 = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 = tl.dot(b_dAqk, b_kg, acc=b_dq2)
            b_dk2 = tl.dot(b_dAkk, b_kg, acc=b_dk2)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * HV*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q0 = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if SAFE_GATE:
        if USE_GATHER:
            b_gn_diag = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
            b_gn_diag = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag = tl.where(m_j_diag, b_g - b_gn_diag, 0.)
        exp_b_g_diag = tl.where(m_j_diag, exp2(b_g_diag), 0.)
        exp_neg_b_g_diag = tl.where(m_j_diag, exp2(-b_g_diag), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H*K
            p_gkj += HV*K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b0[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_dg2 = b_q0 * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            if FULL_CHUNK:
                b_gkn = exp2(b_gk - b_gn)
                b_qg = b_q * b_gkn
                b_kbg = b_kb * b_gkn
            else:
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                # [BC, BK]
                b_gkn = exp2(b_gk - b_gn)
                b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
                b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt = tl.dot(b_dAqk, b_qg, acc=b_dkt)
            b_dkt = tl.dot(b_dAkk, b_kbg, acc=b_dkt)
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * HV*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)

        b_q_exp = b_q0 * exp_b_g_diag
        b_kb_exp = b_k * b_b0[:, None] * exp_b_g_diag

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * HV*BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV*BT)
            # [BK,]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj += HV*K
            p_bj += HV
    p_dk = tl.make_block_ptr(dk, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({})],
    key=["BT", "BC", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_ti, 0), (BC, BK), (1, 0))

    # 2D [BC,1] beta block_ptr instead of a 1D strided block_ptr: the 1D form
    # lowers to a SIMT-mode hivm.hir.stride_load (2x [16] per program in the
    # 16-row split loop, no core_type=VECTOR), while the 2D form with inner
    # stride 1 routes the same values through the VECTOR-mode mte2 strided
    # load path used by the q/k/g blocks.
    p_beta = tl.make_block_ptr(beta, (T, 1), (HV, 1), (i_ti, 0), (BC, 1), (1, 0))
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (HV*BC, 1), (i_ti, 0), (BC, BC), (1, 0))

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]

    # Fully in-bounds sub-chunks (i_ti + BC <= T) skip all boundary machinery:
    # subview clipping, zero-fill branches, and row masks are dead work there and
    # dominate the scalar/MTE side of this small kernel. Values are identical to
    # the masked path; BK == K additionally guarantees in-bounds columns, which
    # is required for unchecked loads.
    if BK == K and (i_ti + BC) <= T:
        b_q = tl.load(p_q)
        b_k = tl.load(p_k)
        b_g = tl.load(p_g)
        b_beta = tl.load(p_beta)

        # Direct middle-row load instead of the UB gather: the gather lowers to
        # a [1, BK] index-tensor fill plus a SIMT-mode vgather over the loaded
        # block on every block iteration. The row (i_ti + BC//2) is in-bounds
        # on the fast path, and the bf16->f32 promotion is exact, so the value
        # is bit-identical to the gathered row while the work moves to mte2.
        # The column mask is statically dead when BK == K (fast-path guard).
        p_gn = g + (i_ti + BC // 2) * HV*K + tl.arange(0, BK)
        if BK > K:
            b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        else:
            b_gn = tl.load(p_gn)
        b_gn = b_gn[None, :]

        # current block, keep numerical stability by subtracting the left boundary
        # less than 85 to avoid overflow in exp2.
        # The k-side gate is written as exp2(b_gn - b_g) instead of exp2(-(b_g - b_gn)):
        # IEEE fp32 subtraction is anti-symmetric (fl(b-a) == -fl(a-b) exactly), so the
        # value is bit-identical while the expression needs no negation op.
        b_gq = exp2(b_g - b_gn)
        b_gk = exp2(b_gn - b_g)

        b_kgt = tl.trans(b_k * b_gk)

        b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
        b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta

        b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
        b_Akk = tl.where(m_Akk, b_Akk, 0.0)

        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))
        tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty))
    else:
        o_c = i_ti + tl.arange(0, BC)
        m_c = o_c < T

        # Row-masked raw-pointer loads instead of boundary_check: boundary_check
        # lowers to a [32,128] zero-fill that the compiler hoists to the top of
        # the per-block loop, where it executes on EVERY block iteration even
        # though only the slow path consumes it. An explicit row mask (plus the
        # column mask only when BK > K) keeps the same zero-fill semantics
        # (masked lanes yield other=0.0 and never touch memory) while letting
        # the fill live with its consumers in the slow branch. block_ptr loads
        # reject mask/other, so the addresses are expressed as raw pointers
        # (row stride H*K / HV*K, col stride 1 — identical addresses).
        m_col = (tl.arange(0, BK) < K)[None, :] if BK > K else None
        m_load = m_c[:, None] & m_col if m_col is not None else m_c[:, None]
        b_q = tl.load(q + o_c[:, None] * (H * K) + tl.arange(0, BK)[None, :], mask=m_load, other=0.0)
        b_k = tl.load(k + o_c[:, None] * (H * K) + tl.arange(0, BK)[None, :], mask=m_load, other=0.0)
        b_g = tl.load(g + o_c[:, None] * (HV * K) + tl.arange(0, BK)[None, :], mask=m_load, other=0.0)
        # [BC,1] masked beta load: same address math and mask semantics as the
        # 1D form (rows yield other=0.0 out of bounds), but the 2D shape with
        # inner stride 1 lowers to the VECTOR-mode mte2 load path instead of a
        # SIMT-mode stride_load (same change as the fast-path block_ptr).
        b_beta = tl.load(beta + o_c[:, None] * HV, mask=m_c[:, None], other=0.0)

        # Direct middle-row load instead of the UB gather (same reasoning as the
        # fast path): the clamped row index min(BC//2, T - i_ti - 1) is always
        # in-bounds, and the bf16 subtraction that consumes b_gn is identical to
        # the gathered value (bf16->f32 promotion is exact).
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + tl.arange(0, BK)
        if BK > K:
            b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        else:
            b_gn = tl.load(p_gn)
        b_gn = b_gn[None, :]

        # current block, keep numerical stability by subtracting the left boundary
        # less than 85 to avoid overflow in exp2.
        # b_gq needs no row mask: its out-of-bounds rows only feed dot output
        # rows that the (row-masked) stores never write, so an exp2 overflow
        # there stays confined (NaN confinement) and every stored value is
        # bit-identical. b_gk MUST keep its mask: a NaN in its out-of-bounds
        # rows would poison every column of the transposed k-side operand and
        # hence every stored dot output.
        b_gm = (b_g - b_gn).to(tl.float32)

        b_gq = exp2(b_gm)
        b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

        b_kgt = tl.trans(b_k * b_gk)

        b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
        b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta

        b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
        b_Akk = tl.where(m_Akk, b_Akk, 0.0)

        # Masked raw-pointer stores instead of block_ptr boundary_check stores:
        # boundary_check lowers to per-store row-clamp scalar chains
        # (subi/divsi/maxsi/minsi) plus extract_slice/subview plumbing that
        # recomputes the valid row range for every store (2x per program). A
        # masked store reuses the existing m_c row mask (masked lanes are never
        # written, identical to boundary_check) and drops the whole clamp. The
        # Aqk column range [i_i*BC, i_i*BC+BC) is always within the BT/BC buffer
        # when BT % BC == 0 (supported configs: 64/32, 64/16, 32/16), so the
        # column mask is statically dead there; it is kept for BT % BC != 0.
        o_j = i_i * BC + o_i
        m_col = (o_j < BT)[None, :] if BT % BC != 0 else None
        m_st = m_c[:, None] & m_col if m_col is not None else m_c[:, None]
        tl.store(Aqk + o_c[:, None] * (HV * BT) + o_j[None, :], b_Aqk.to(Aqk.dtype.element_ty), mask=m_st)
        tl.store(Akk + o_c[:, None] * (HV * BC) + o_i[None, :], b_Akk.to(Akk.dtype.element_ty), mask=m_c[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_chunk_solve(
    Akk,
    cu_seqlens,
    chunk_indices,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    Akk = Akk + (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_I = o_i[:, None] == o_i[None, :]

    p_Akk = tl.make_block_ptr(Akk, (T, BC), (HV*BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    b_Akk = tl.load(p_Akk, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tle.extract_slice(b_Akk, [i, 0], [1, BC], [1, 1]).reshape(BC)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai = tle.insert_slice(b_Ai, b_a.reshape(1, BC), [i, 0], [1, BC], [1, 1])
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    BC = 32 if (BT == 64 and (HV * K) & (HV * K - 1) != 0) else 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    # Step 1: Run token_parallel first to compute diagonal blocks into Akkd (fp32)
    # Step 1: compute diagonal blocks into Akk_diag (fp32)
    if safe_gate:
        grid = (NT, NC, B * HV)
        BK = triton.next_power_of_2(K)
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
        )
        chunk_kda_fwd_kernel_intra_sub_chunk_solve[grid](
            Akk=Akkd,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            HV=HV,
            BT=BT,
            BC=BC,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_SAFE_GATE=safe_gate,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    # BC 16->32：NC 4->2
    BC = min(16, BT) if (HV * K) & (HV * K - 1) == 0 else min(32, BT)
    BK = min(128, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK, NT, B * HV)
    for i_i in range(NC):
        chunk_kda_bwd_kernel_intra[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dq2=dq2,
            dk=dk,
            dk2=dk2,
            dg=dg,
            dg2=dg2,
            db=db2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
            I_I=i_i,
            SAFE_GATE=safe_gate,
            USE_GATHER=False,
            FULL_CHUNK=(cu_seqlens is None and T % BT == 0),
        )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg
