import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        Dk_ptr,
        V_block_ptr,
        Dv_ptr,
        block_index_q,
        softmax_scale,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        STAGE: tl.constexpr,
        offs_q: tl.constexpr,
        offs_kv: tl.constexpr,
        SEQ_LEN: tl.constexpr,
):
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (0, lo))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # Load K and Dk, then decompress K block
        K_part = tl.load(K_block_ptr)  # (RANK, BLOCK_SIZE_KV)
        Dk_block = tl.load(Dk_ptr)  # (RANK, HEAD_DIM)
        K_block = tl.dot(tl.trans(K_part), Dk_block)  # (BLOCK_SIZE_KV, HEAD_DIM)
        K_block = tl.trans(K_block)  # (HEAD_DIM, BLOCK_SIZE_KV)

        QK_block = tl.dot(Q_block, K_block.to(tl.float16))

        # online softmax
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # sum by rows of attention scores
        l_ij = tl.sum(P_block, 1)
        # correction
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # Load V and Dv, then compute full V
        V_part = tl.load(V_block_ptr)  # (RANK, BLOCK_SIZE_KV)
        Dv_block = tl.load(Dv_ptr)  # (RANK, HEAD_DIM)
        V_block = tl.dot(tl.trans(V_part), Dv_block).to(tl.float16)  # (BLOCK_SIZE_KV, HEAD_DIM)

        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_SIZE_KV))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM", "RANK"],
)
@triton.jit
def _attn_fwd(
        Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        K,  # BATCH_SIZE, SEQ_LEN, RANK
        Dk,  # BATCH_SIZE, RANK, NUM_HEADS * HEAD_DIM
        V,  # BATCH_SIZE, SEQ_LEN, RANK
        Dv,  # BATCH_SIZE, RANK, NUM_HEADS * HEAD_DIM
        softmax_scale,
        O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        stride_Q_batch,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_seq,
        stride_K_rank,
        stride_Dk_batch,
        stride_Dk_rank,
        stride_Dk_dim,
        stride_V_batch,
        stride_V_seq,
        stride_V_rank,
        stride_Dv_batch,
        stride_Dv_rank,
        stride_Dv_dim,
        stride_O_batch,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        NUM_HEADS,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        RANK: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    block_index_q = tl.program_id(0)
    head_index = tl.program_id(1)
    batch_index = tl.program_id(2)

    q_batch_head_offset = batch_index.to(tl.int64) * stride_Q_batch + head_index.to(tl.int64) * stride_Q_head
    o_batch_head_offset = batch_index.to(tl.int64) * stride_O_batch + head_index.to(tl.int64) * stride_O_head

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_batch_head_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # K and V are shared across heads, so no head offset
    K_block_ptr = tl.make_block_ptr(
        base=K + batch_index.to(tl.int64) * stride_K_batch,
        shape=(RANK, SEQ_LEN),
        strides=(stride_K_rank, stride_K_seq),
        offsets=(0, 0),
        block_shape=(RANK, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + batch_index.to(tl.int64) * stride_V_batch,
        shape=(RANK, SEQ_LEN),
        strides=(stride_V_rank, stride_V_seq),
        offsets=(0, 0),
        block_shape=(RANK, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # Dk and Dv - slice them per head inside
    Dk_ptr = tl.make_block_ptr(
        base=Dk + batch_index.to(tl.int64) * stride_Dk_batch,
        shape=(RANK, HEAD_DIM),  # We'll index into the right head slice
        strides=(stride_Dk_rank, stride_Dk_dim),
        offsets=(0, head_index * HEAD_DIM),  # Offset to the current head
        block_shape=(RANK, HEAD_DIM),
        order=(1, 0),
    )

    Dv_ptr = tl.make_block_ptr(
        base=Dv + batch_index.to(tl.int64) * stride_Dv_batch,
        shape=(RANK, HEAD_DIM),  # We'll index into the right head slice
        strides=(stride_Dv_rank, stride_Dv_dim),
        offsets=(0, head_index * HEAD_DIM),  # Offset to the current head
        block_shape=(RANK, HEAD_DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + o_batch_head_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i - running maximum
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i - running sum
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc - accumulator for the output
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        Dk_ptr,
        V_block_ptr,
        Dv_ptr,
        block_index_q,
        head_index,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        1,  # STAGE 1: left of diagonal
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        Dk_ptr,
        V_block_ptr,
        Dv_ptr,
        block_index_q,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        RANK,
        HEAD_DIM,
        NUM_HEADS,
        2,  # STAGE 2: diagonal block with masking
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    # scaling
    O_block = O_block / l_i[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def lowrank_flash_attention(Q, K, Dk, V, Dv, softmax_scale=None):
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    RANK = K.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    assert K.shape == (BATCH_SIZE, SEQ_LEN, RANK)
    assert Dk.shape == (BATCH_SIZE, RANK, NUM_HEADS * HEAD_DIM)
    assert V.shape == (BATCH_SIZE, SEQ_LEN, RANK)
    assert Dv.shape == (BATCH_SIZE, RANK, NUM_HEADS * HEAD_DIM)

    O = torch.empty_like(Q)

    grid = lambda args: (
        triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
        NUM_HEADS,
        BATCH_SIZE,
    )

    _attn_fwd[grid](
        Q=Q,
        K=K,
        Dk=Dk,
        V=V,
        Dv=Dv,
        softmax_scale=softmax_scale,
        O=O,
        stride_Q_batch=Q.stride(0),
        stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),
        stride_Q_dim=Q.stride(3),
        stride_K_batch=K.stride(0),
        stride_K_seq=K.stride(1),
        stride_K_rank=K.stride(2),
        stride_Dk_batch=Dk.stride(0),
        stride_Dk_rank=Dk.stride(1),
        stride_Dk_dim=Dk.stride(2),
        stride_V_batch=V.stride(0),
        stride_V_seq=V.stride(1),
        stride_V_rank=V.stride(2),
        stride_Dv_batch=Dv.stride(0),
        stride_Dv_rank=Dv.stride(1),
        stride_Dv_dim=Dv.stride(2),
        stride_O_batch=O.stride(0),
        stride_O_head=O.stride(1),
        stride_O_seq=O.stride(2),
        stride_O_dim=O.stride(3),
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
        RANK=RANK,
    )