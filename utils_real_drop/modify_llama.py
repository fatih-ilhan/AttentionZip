import math
from typing import Optional, Tuple
import time

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    rotate_half,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaForCausalLM
)
import types

__all__ = ["AttZipLlamaForCausalLM", "AttZipLlamaAttention",
           "AttZipLlamaAttention_streaming", "AttZipLlamaForCausalLM_streaming"]

recon_time = 0
cache_time = 0
att_time = 0

from transformers.configuration_utils import PretrainedConfig
from svd_utils import compute_SVD

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].repeat(bsz, 1, 1, 1)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)

    return x_embed


def apply_rotary_pos_emb_single_lr(cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    return cos, sin


class AttZipKVCache_LayerWise:
    def __init__(
        self,
        imp_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        scale_imp=False,
        svd_mode='t_hd'
    ):
        print(f"KVCache-LayerWise: {imp_size}, {recent_size}")
        self.imp_size = imp_size
        self.recent_size = recent_size
        self.cache_size = imp_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.imp_score = None
        self.counter = None
        self.scale_imp = scale_imp
        self.svd_mode = svd_mode
        self.is_full = False

    def __call__(self, past_key_values, attn_score_cache, idxs):

        self._update_imp_score(attn_score_cache, idxs)

        if past_key_values is None:
            return None
        
        bsz, num_heads = past_key_values[-1].shape[:2]

        if self.svd_mode == 't_hd' and len(past_key_values) == 5:
            recent_size = max(past_key_values[3].size(self.k_seq_dim), self.recent_size)
        else:
            recent_size = self.recent_size
        
        if len(past_key_values) == 5:
            if self.svd_mode == 't_hd':
                seq_len = past_key_values[0].size(1) + past_key_values[3].size(self.k_seq_dim)
            elif self.svd_mode == 'ht_d':
                T = past_key_values[0].shape[-2] // num_heads
                seq_len = T + past_key_values[3].size(self.k_seq_dim)
        elif len(past_key_values) == 4:
            if self.svd_mode == 't_hd':
                seq_len = past_key_values[0].size(1)
            elif self.svd_mode == 'ht_d':
                seq_len = past_key_values[0].shape[-2] // num_heads
        else:
            seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        self.is_full = True
        if self.svd_mode == 't_hd':
            select_imp_scores = self.imp_score[:, :seq_len - recent_size]
            select_counters = self.counter[:, :seq_len - recent_size]
        else:
            select_imp_scores = self.imp_score[:, :, :seq_len - recent_size]
            select_counters = self.counter[:, :, :seq_len - recent_size]

        if self.scale_imp:
            _, keep_topk = torch.topk(select_imp_scores / select_counters, self.imp_size, dim=-1)
        else:
            _, keep_topk = torch.topk(select_imp_scores, self.imp_size, dim=-1)
        keep_topk = keep_topk.sort().values

        if self.svd_mode == 't_hd':
            keep_recent = torch.arange(seq_len - recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        else:
            keep_recent = torch.arange(seq_len - recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.imp_score.shape, dtype=torch.bool, device=past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        if len(past_key_values) == 5:
            if self.svd_mode == 't_hd':
                k_imp_recent = torch.take_along_dim(past_key_values[0], keep_idx[:, :-past_key_values[3].shape[-2], None], dim=1)
            elif self.svd_mode == 'ht_d':
                T = past_key_values[0][0].shape[-2] // num_heads
                k_imp_recent = past_key_values[0][0].reshape(num_heads, T, -1)[:, mask[0, :T]].reshape(bsz, -1, past_key_values[0].shape[-1])
            else:
                raise NotImplementedError
        elif len(past_key_values) == 4:
            if self.svd_mode == 't_hd':
                k_imp_recent = torch.take_along_dim(past_key_values[0], keep_idx[..., None], dim=1)
        elif len(past_key_values) == 2:
            k_imp_recent = torch.take_along_dim(past_key_values[0], keep_idx.view(bsz, 1, -1, 1), dim=2)

        if self.svd_mode == 't_hd':
            v_imp_recent = torch.take_along_dim(past_key_values[-1], keep_idx.view(bsz, 1, -1, 1), dim=2)
            self.imp_score= torch.take_along_dim(self.imp_score, keep_idx, dim=1)
            self.counter = torch.take_along_dim(self.counter, keep_idx, dim=1)
            self.imp_score_var = torch.take_along_dim(self.imp_score_var, keep_idx, dim=1)
        else:
            v_imp_recent = past_key_values[-1][mask].reshape(bsz, num_heads, -1, past_key_values[-1].shape[-1])
            self.imp_score= self.imp_score[mask].view(bsz, num_heads, self.cache_size)
            self.counter = self.counter[mask].view(bsz, num_heads, self.cache_size)
            self.imp_score_var = self.imp_score_var[mask].view(bsz, num_heads, self.cache_size)

        if len(past_key_values) == 4:
            return (k_imp_recent, past_key_values[1], past_key_values[2], v_imp_recent)
        elif len(past_key_values) == 5:
            if self.svd_mode == 'ht_d':
                return (k_imp_recent, past_key_values[1], past_key_values[2],
                        past_key_values[3][0][mask[:, T:]].view(bsz, num_heads, -1, past_key_values[3].shape[-1]), v_imp_recent)
            else:
                return (k_imp_recent, past_key_values[1], past_key_values[2], past_key_values[3], v_imp_recent)
        else:
            return (k_imp_recent, v_imp_recent)

    def _update_imp_score(self, attn_score_cache, idxs):

        num_new_tokens = attn_score_cache.shape[2]

        if self.imp_score is None:
            if self.svd_mode == 't_hd':
                self.imp_score = attn_score_cache.sum(2).mean(dim=1)
                self.counter = torch.arange(attn_score_cache.shape[2], 0, -1, device=self.imp_score.device)[None, :].repeat(
                    attn_score_cache.shape[0], 1)
                if self.imp_score.shape[-1] != self.counter.shape[-1]:
                    print(self.counter.shape)
            else:
                self.imp_score = attn_score_cache.sum(2)
                self.counter = torch.arange(attn_score_cache.shape[2], 0, -1, device=self.imp_score.device)[None, None, :].repeat(
                    attn_score_cache.shape[0], attn_score_cache.shape[1], 1)
                if self.imp_score.shape[-1] != self.counter.shape[-1]:
                    print(self.counter.shape)
        else:
            if idxs is not None:
                if self.svd_mode == 't_hd':
                    attn_score_cache = attn_score_cache.sum(2).mean(dim=1)
                    self.imp_score = torch.cat([self.imp_score, attn_score_cache[..., -num_new_tokens:]], dim=-1)
                    self.imp_score.scatter_(-1, idxs[:, :-num_new_tokens], attn_score_cache[..., :-num_new_tokens], reduce='add')
                    
                    self.counter = torch.cat([self.counter, 
                                              torch.zeros(self.counter.shape[0], num_new_tokens, dtype=self.counter.dtype, device=self.counter.device)], dim=-1)
                    self.counter.scatter_(-1, idxs, 1, reduce='add')
                else:
                    attn_score_cache = attn_score_cache.sum(2)
                    self.imp_score = torch.cat([self.imp_score, attn_score_cache[..., -num_new_tokens:]], dim=-1)
                    self.imp_score[idxs.unsqueeze(1).repeat(1, self.imp_score.shape[1], 1)] += attn_score_cache.flatten()
                    
                    self.counter = torch.cat([self.counter, torch.zeros(*self.counter.shape[:2], num_new_tokens, device=self.counter.device)], dim=-1)
                    self.counter[idxs.unsqueeze(1).repeat(1, self.counter.shape[1], 1)] += 1
            else:
                if self.svd_mode == 't_hd':
                    attn_score_cache = attn_score_cache.sum(2).mean(dim=1)
                    attn_score_cache[:, :-num_new_tokens] += self.imp_score
                    counter = torch.ones((attn_score_cache.shape[0], attn_score_cache.shape[1]), dtype=self.counter.dtype, device=self.imp_score.device)
                    counter[:, :-num_new_tokens] += self.counter
                else:
                    attn_score_cache = attn_score_cache.sum(2)
                    attn_score_cache[:, :, :-num_new_tokens] += self.imp_score
                    counter = torch.ones((attn_score_cache.shape[0], attn_score_cache.shape[1], attn_score_cache.shape[2]), device=self.imp_score.device)
                    counter[:, :, :-num_new_tokens] += self.counter

                self.imp_score = attn_score_cache
                self.counter = counter

    def _clean_scores(self):
        self.imp_score = None
        self.counter = None


class AttZipLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rank_k = config.rank_k
        self.rank_v = config.rank_v
        self.svd_T = config.svd_T
        self.svd_mode = config.svd_mode
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = AttZipKVCache_LayerWise(
            imp_size=config.imp_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            scale_imp=config.scale_imp,
            svd_mode=config.svd_mode
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if len(past_key_value) == 5:
                if self.svd_mode == 't_hd':
                    kv_seq_len += past_key_value[0].shape[-2] + past_key_value[3].shape[-2]
                elif self.svd_mode == 'ht_d':
                    kv_seq_len += past_key_value[0].shape[-2] // self.num_heads + past_key_value[3].shape[-2]
                else:
                    raise NotImplementedError
            elif len(past_key_value) == 4:
                if self.svd_mode == 'ht_d':
                    kv_seq_len += past_key_value[0].shape[-2] // self.num_heads
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # remake causal mask
            attention_mask = _make_causal_mask(
                bsz=bsz,
                tgt_len=q_len,
                past_key_values_length=kv_seq_len-q_len if past_key_value is not None else 0,
                dtype=query_states.dtype,
                device=query_states.device,
            )

            if bsz > 1:
                val = -65504.
                for b in range(bsz): 
                    start = (position_ids[b] == 0).nonzero()
                    if start.numel() == 0:
                        if position_ids[b].min() < self.kv_cache.cache_size:
                            attention_mask[b, :, :, :self.kv_cache.cache_size - position_ids[b].min()] = val
                    else:
                        if start[0, 0] != 0:
                            attention_mask[b, :, :, :start[0, 0]] = val

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        # key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        idxs = None
        idxs_bool = None
        if past_key_value is not None:
            value_states = torch.cat([past_key_value[-1], value_states], dim=2)

        T = max(self.svd_T, q_len) if self.rank_k != 0 else max(self.kv_cache.cache_size, q_len)
        if kv_seq_len <= T or self.rank_k == 0:  # concat with past
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
        else:  # concat after reconstruction with past
            if self.svd_mode == 't_hd':
                recon = 0
                if recon and recon != 1:
                    p = self.kv_cache.imp_score[..., :past_key_value[0].shape[1]].mean(dim=1) / (self.kv_cache.counter[..., :past_key_value[0].shape[1]][:, 0] ** 1)
                    idxs = p.topk(recon, dim=-1)[1]
                    idxs_bool = torch.zeros_like(p, device=p.device).bool()
                    idxs_bool.scatter_(1, idxs, True)
                    sub_past_key_0 = past_key_value[0][idxs_bool.unsqueeze(-1).repeat(1, 1, past_key_value[0].shape[-1])].reshape(
                        bsz, -1, past_key_value[0].shape[-1])
                    cached = sub_past_key_0 * past_key_value[1].unsqueeze(-2) @ past_key_value[2]
                else:
                    cached = past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2]

                cached = cached.reshape(*cached.shape[:2], self.num_heads, -1).swapaxes(-2, -3)
                if len(past_key_value) == 4:
                    key_states = torch.cat([cached, key_states], dim=2)
                elif len(past_key_value) == 5:
                    key_states = torch.cat([cached, past_key_value[3], key_states], dim=2)

            elif self.svd_mode == 'ht_d':
                cached = past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2]
                cached = cached.reshape(cached.shape[0], self.num_heads, cached.shape[1] // self.num_heads, -1)
                if len(past_key_value) == 4:
                    key_states = torch.cat([cached, key_states], dim=2)
                elif len(past_key_value) == 5:
                    key_states = torch.cat([cached, past_key_value[3], key_states], dim=2)
            else:
                key_states = torch.cat([past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2], key_states], dim=2)

        past_key_states = key_states
            
        if self.rank_k != 0:
        
            if kv_seq_len == T:
                past_key_states = compute_SVD(key_states, self.svd_mode, self.rank_k)

            elif kv_seq_len > T:
                if self.svd_mode == 't_hd' or self.svd_mode == 'ht_d':
                    if len(past_key_value) == 4:
                        past_key_states = [*past_key_value[:-1], key_states[..., -q_len:, :]]
                    else:
                        if  past_key_value[3].shape[-2] == T:
                            past_key_states = compute_SVD(key_states, self.svd_mode, self.rank_k)
                        else:
                            past_key_states = [*past_key_value[:-2], torch.cat([past_key_value[3], key_states[..., -q_len:, :]], dim=-2)]
                else:
                    raise NotImplementedError

                if torch.isnan(key_states).any():
                    print(kv_seq_len)

        if isinstance(past_key_states, list):
            past_key_value = (*past_key_states, value_states) if use_cache else None
        else:
            past_key_value = (past_key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        if idxs is not None:

            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
            key_position_ids = torch.cat([key_position_ids[:, :idxs_bool.shape[-1]][idxs_bool].reshape(
                bsz, -1), key_position_ids[:, idxs_bool.shape[-1]:]], dim=-1)
            idxs_bool = torch.zeros(bsz, kv_seq_len, device=key_position_ids.device).bool()
            idxs_bool.scatter_(1, key_position_ids, True)
            attention_mask = attention_mask[idxs_bool[None, None, ...].repeat(*attention_mask.shape[:-1], 1)].reshape(
                *attention_mask.shape[:-1], -1)
            value_states = value_states[idxs_bool[None, ..., None].repeat(*value_states.shape[:-2], 1, value_states.shape[-1])].reshape(
                *value_states.shape[:-2], -1, value_states.shape[-1]) 
            
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        else:
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone(), idxs_bool)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AttZipLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = AttZipLlamaAttention(config, layer_idx)
    

## AttZip KV Cache dropping with Position rolling
class AttZipLlamaAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rank_k = config.rank_k
        self.layer_idx = layer_idx
        self.svd_T = config.svd_T
        self.svd_mode = config.svd_mode
        self.recon = config.recon

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = AttZipKVCache_LayerWise(
            imp_size=config.imp_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            scale_imp=config.scale_imp,
            svd_mode=config.svd_mode
        )

    def _init_rope(self):
        self.rotary_emb_k = None
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta)
            if self.rank_k != 0:
                self.rotary_emb_k = LlamaRotaryEmbedding(
                    self.rank_k,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        st = time.time()
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if len(past_key_value) == 5:
                if self.svd_mode == 't_hd':
                    kv_seq_len += past_key_value[0].shape[-2] + past_key_value[3].shape[-2]
                elif self.svd_mode == 'ht_d':
                    kv_seq_len += past_key_value[0].shape[-2] // self.num_heads + past_key_value[3].shape[-2]
                else:
                    raise NotImplementedError
            elif len(past_key_value) == 4:
                if self.svd_mode == 'ht_d':
                    kv_seq_len += past_key_value[0].shape[-2] // self.num_heads
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        
        if self.kv_cache.is_full or bsz == 1:
            # remake causal mask
            attention_mask = _make_causal_mask(
                bsz=bsz,
                tgt_len=q_len,
                past_key_values_length=kv_seq_len-q_len if past_key_value is not None else 0,
                dtype=query_states.dtype,
                device=query_states.device,
            )

            if bsz > 1:
                val = -65504.
                for b in range(bsz): 
                    start = (position_ids[b] == 0).nonzero()
                    if start.numel() == 0:
                        if position_ids[b].min() < self.kv_cache.cache_size:
                            attention_mask[b, :, :, :-position_ids[b].min()-1] = val
                    else:
                        if start[0, 0] != 0:
                            attention_mask[b, :, :, :start[0, 0]] = val

        if past_key_value is not None:
            value_states = torch.cat([past_key_value[-1], value_states], dim=2)

        st_ = time.time()
        idxs = None
        T = self.svd_T if self.rank_k != 0 else self.kv_cache.cache_size
        if past_key_value is not None:
            if kv_seq_len <= T or self.rank_k == 0:  # concat with past
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
            else:  # concat after reconstruction with past
                if self.svd_mode == 't_hd':
                    recon = 0
                    if past_key_value is not None and len(past_key_value) == 5 and past_key_value[-2].shape[-2] == T:
                        pass
                    else:
                        recon = int(past_key_value[0].shape[-2] * self.recon)
                    if recon != 0 and recon != 1 and recon != past_key_value[0].shape[-2]:

                        p = self.kv_cache.imp_score[..., :past_key_value[0].shape[1]] / (self.kv_cache.counter[..., :past_key_value[0].shape[1]] ** 1)
                        idxs = p.topk(recon, sorted=False, dim=-1)[1].sort(dim=-1)[0]

                        sub_past_key_0 = torch.take_along_dim(past_key_value[0], idxs[..., None], dim=1)
                        cached = sub_past_key_0 * past_key_value[1].unsqueeze(-2) @ past_key_value[2]
                    else:
                        cached = past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2]

                    cached = cached.reshape(*cached.shape[:2], self.num_heads, -1).swapaxes(-2, -3)
                    if len(past_key_value) == 4:
                        key_states = torch.cat([cached, key_states], dim=2)
                    elif len(past_key_value) == 5:
                        key_states = torch.cat([cached, past_key_value[3], key_states], dim=2)
                elif self.svd_mode == 'ht_d':
                    cached = past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2]
                    cached = cached.reshape(cached.shape[0], self.num_heads, cached.shape[1] // self.num_heads, -1)
                    if len(past_key_value) == 4:
                        key_states = torch.cat([cached, key_states], dim=2)
                    elif len(past_key_value) == 5:
                        key_states = torch.cat([cached, past_key_value[3], key_states], dim=2)
                else:
                    key_states = torch.cat([past_key_value[0] * past_key_value[1].unsqueeze(-2) @ past_key_value[2], key_states], dim=2)
        et_ = time.time()

        past_key_states = key_states
            
        if self.rank_k != 0:
        
            if kv_seq_len == T or (kv_seq_len > T and past_key_value is None):
                # -- low-rank decomposition --
                rank_k = min(self.rank_k, key_states.shape[-1])
                past_key_states = compute_SVD(key_states, self.svd_mode, rank_k)

            elif kv_seq_len > T:
                if self.svd_mode == 't_hd' or self.svd_mode == 'ht_d':
                    if len(past_key_value) == 4:
                        past_key_states = [*past_key_value[:-1], key_states[..., -q_len:, :]]
                    else:
                        if past_key_value[3].shape[-2] == T:
                            rank_k = min(self.rank_k, key_states.shape[-1])
                            past_key_states = compute_SVD(key_states, self.svd_mode, rank_k)
                        else:
                            past_key_states = [*past_key_value[:-2], torch.cat([past_key_value[3], key_states[..., -q_len:, :]], dim=-2)]
                else:
                    raise NotImplementedError
            
                if torch.isnan(key_states).any():
                    print(kv_seq_len)

        if isinstance(past_key_states, list):
            past_key_value = (*past_key_states, value_states) if use_cache else None
        else:
            past_key_value = (past_key_states, value_states) if use_cache else None

        if position_ids.shape[-1] == 1:
            for b in range(bsz):
                if idxs is not None:
                    # position_ids[b][0] = min(kv_seq_len - p.shape[-1] + idxs.shape[-1] - 1, position_ids[b][0])
                    position_ids[b][0] = min(kv_seq_len - 1, position_ids[b][0])
                else:
                    position_ids[b][0] = min(kv_seq_len - 1, position_ids[b][0])
        else:
            pass

        if idxs is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=idxs.shape[-1]+kv_seq_len-p.shape[-1])

        ### Shift Pos: query pos is min(cache_size, idx)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        ###

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        if idxs is not None:
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0).repeat(bsz, 1)
            idxs = torch.cat([idxs, key_position_ids[:, p.shape[-1]:]], dim=-1)
            key_position_ids = torch.take_along_dim(key_position_ids, idxs, dim=1)
            if not self.kv_cache.is_full:
                offset = position_ids.max(dim=-1)[0] - position_ids.max().item()
                key_position_ids = torch.clip(key_position_ids + offset.unsqueeze(-1), 0)
            else:
                offset = torch.clip(position_ids.max(dim=-1)[0] - kv_seq_len + 1, None, 0)
                key_position_ids = torch.clip(key_position_ids + offset.unsqueeze(-1), 0)
            attention_mask = torch.take_along_dim(attention_mask, idxs[:, None, None, :], dim=-1)
            value_states = torch.take_along_dim(value_states, idxs[:, None, :, None], dim=-2)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

        else:
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0).repeat(bsz, 1)

            if not self.kv_cache.is_full:
                offset = position_ids.max(dim=-1)[0] - position_ids.max().item()
            else:
                offset = torch.clip(position_ids.max(dim=-1)[0] - kv_seq_len + 1, None, 0)
            key_position_ids = torch.clip(key_position_ids + offset.unsqueeze(-1), 0)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

        assert torch.all(position_ids[:, -1] == key_position_ids[:, -1])

        if not isinstance(key_states, list):
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if len(past_key_value) == 5:
            assert past_key_value[0].shape[-2] + past_key_value[-2].shape[-2] == past_key_value[-1].shape[-2]
        else:
            assert past_key_value[0].shape[-2] == past_key_value[-1].shape[-2]

        past_key_value = self.kv_cache(past_key_value, attn_weights, idxs)

        if len(past_key_value) == 5:
            assert past_key_value[0].shape[-2] + past_key_value[-2].shape[-2] == past_key_value[-1].shape[-2]
        else:
            assert past_key_value[0].shape[-2] == past_key_value[-1].shape[-2]

        return attn_output, attn_weights, past_key_value


class AttZipLlamaForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = AttZipLlamaAttention_streaming(config, layer_idx)
