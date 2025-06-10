"""
Transformer mixer block for DEQ transformers.
"""

import torch
from einops import rearrange
from torch import LongTensor, Tensor, nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


class LlamaRopeConfig:
    """ """

    def __init__(self, d_model: int, n_head: int, d_head: int, max_seq_len: int, rope_theta: float, rope_scaling=None):
        self.head_dim = d_head
        self.max_position_embeddings = max_seq_len
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_size = d_model
        self.num_attention_heads = n_head


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class CausalSelfAttention(nn.Module):
    """
    CausalSelfAttention from https://github.com/karpathy/nanoGPT/blob/master/model.py

    Llama3 https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        layer_idx: int | None = None,
        bias: bool = False,
        apply_rotary: bool = True,
        rope_theta: float = 500000.0,
        **kwargs,
    ):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        # n_kv_heads (used in Llama) == n_head and d_model % n_head == 0
        # because of the way we inject inputs TODO: remove this constraint
        assert d_model % n_head == 0
        self.n_head = n_head
        self.n_kv_heads = n_head
        self.d_head = d_model // n_head
        self.wq = nn.Linear(d_model, n_head * self.d_head, bias=bias)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        # output projection
        self.out_proj = nn.Linear(n_head * self.d_head, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.dropout = dropout
        self.layer_idx = layer_idx
        # we need to define d_in_proj so that the injection is scale to the right dimension
        self.d_in_proj = 3 * d_model
        self.kv_cache_size = 2 * d_model
        self.is_causal = True
        self.scaling = self.d_head ** -0.5

        # ROPE
        self.apply_rotary = apply_rotary
        self.rope_theta = rope_theta
        if self.apply_rotary:
            rope_config = LlamaRopeConfig(
                d_model=d_model,
                n_head=n_head,
                d_head=self.d_head,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                rope_scaling=None,
            )
            self.rotary_emb = LlamaRotaryEmbedding(rope_config)

    def forward(
        self,
        hidden_states: Tensor,
        injected_inputs: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        past_key_value: Tensor | None = None,
        cache_position: LongTensor | None = None,
        position_ids: Tensor | None = None,
        skip_kv_update: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Accepts tensors of shape
            u: (batch_size, seq_len, d_model)
            injected_inputs: (batch_size, seq_len, d_model) if seqlen is None
        Returns:
            y: (batch_size, seq_len, d_model)
        """
        # batch_size -> B, seq_len -> T, n_head -> nh, head_size -> hs
        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        if injected_inputs is not None:
            injected_q, injected_k, injected_v = torch.chunk(injected_inputs, 3, dim=-1)
            q = q + injected_q
            k = k + injected_k
            v = v + injected_v
        q = rearrange(q, "B T (nh hs) -> B nh T hs", nh=self.n_head, hs=self.d_head)
        k = rearrange(k, "B T (nh hs) -> B nh T hs", nh=self.n_head, hs=self.d_head)
        v = rearrange(v, "B T (nh hs) -> B nh T hs", nh=self.n_head, hs=self.d_head)

        if self.apply_rotary:
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                assert position_ids is not None, "position_ids must be provided when using rotary embeddings"
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                cos, sin = position_embeddings

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            if skip_kv_update:
                try:
                    cached_k, cached_v = past_key_value[self.layer_idx]
                except KeyError:
                    cached_k, cached_v = None, None

                if cached_k is not None:
                    k = torch.cat([cached_k, k], dim=-2)
                    v = torch.cat([cached_v, v], dim=-2)
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        attention_interface = sdpa_attention_forward
        y, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        y = rearrange(y, "B T nh hs -> B T (nh hs)").contiguous()
        y = self.resid_dropout(self.out_proj(y))
        return y
