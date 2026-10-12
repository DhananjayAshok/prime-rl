"""Gemma 4 dense text model (e.g. gemma-4-31B-it) for PrimeRL training.

Text-only: the checkpoint's vision/audio towers are never built or loaded. Module paths
mirror the HF checkpoint (``model.language_model.*``), so weights load by name and LoRA
adapter keys use the names vLLM maps for ``Gemma4ForConditionalGeneration``.

Numerics follow ``transformers.models.gemma4.modeling_gemma4`` (scaled embeddings,
sandwich norms, q/k/v norms, attention scale 1.0, ``layer_scalar``, per-layer-type RoPE,
final logit softcapping in the LM head). Not supported: MoE blocks, per-layer input
embeddings and KV-shared layers (all off for the 31B dense model).

Attention runs over the packed batch: sliding layers (head_dim 256) use the varlen flash
kernels; full-attention layers (head_dim 512, above flash attention's limit) loop over
the packed documents with SDPA.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import flash_attn_2_varlen_op, flash_attn_3_varlen_op
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

FLASH_OPS = {"flash_attention_2": flash_attn_2_varlen_op, "flash_attention_3": flash_attn_3_varlen_op}


def _check_supported(config: Gemma4TextConfig) -> None:
    unsupported = {
        "enable_moe_block": config.enable_moe_block,
        "hidden_size_per_layer_input": config.hidden_size_per_layer_input,
        "num_kv_shared_layers": config.num_kv_shared_layers,
    }
    enabled = {name: value for name, value in unsupported.items() if value}
    if enabled:
        raise ValueError(f"PrimeRL's Gemma 4 supports the dense model only; this config sets {enabled}")


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int, attn_implementation: str):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        # Full-attention layers with attention_k_eq_v have no v_proj: values are the keys.
        self.k_eq_v = config.attention_k_eq_v and not self.is_sliding
        num_kv_heads = config.num_global_key_value_heads if self.k_eq_v else config.num_key_value_heads
        self.flash_op = FLASH_OPS.get(attn_implementation) if self.is_sliding else None

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            None
            if self.k_eq_v
            else nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(self, hidden_states: Tensor, position_embeddings, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        cos, sin = position_embeddings

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        raw_key = self.k_proj(hidden_states).view(hidden_shape)
        value = self.v_norm(self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else raw_key)
        key = apply_rotary_pos_emb(self.k_norm(raw_key), cos, sin, unsqueeze_dim=2)

        # [1, T, H, D] -> [T, H, D]
        query, key, value = query[0], key[0], value[0]
        if self.flash_op is not None:
            window = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
            out = self.flash_op(
                query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                softmax_scale=1.0, causal=True, window_size=window,
            )
        else:
            out = self._sdpa_per_document(query, key, value, cu_seqlens)
        return self.o_proj(out.reshape(1, out.shape[0], -1))

    def _sdpa_per_document(self, query: Tensor, key: Tensor, value: Tensor, cu_seqlens: Tensor) -> Tensor:
        bounds = cu_seqlens.tolist()
        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end == start:
                continue
            q, k, v = (t[start:end].transpose(0, 1) for t in (query, key, value))
            mask = None
            if self.sliding_window is not None and end - start > self.sliding_window:
                pos = torch.arange(end - start, device=q.device)
                diff = pos[:, None] - pos[None, :]
                mask = (diff >= 0) & (diff < self.sliding_window)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mask is None, scale=1.0, enable_gqa=True
            )
            outputs.append(out.transpose(0, 1))
        return torch.cat(outputs, dim=0)


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int, attn_implementation: str):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Gemma4Attention(config, layer_idx, attn_implementation)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden_states: Tensor, position_embeddings, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        # A buffer, so FSDP mixed precision leaves it fp32: cast it rather than promote the activations
        # (HF's in-place multiply keeps the activation dtype too).
        return hidden_states * self.layer_scalar.to(hidden_states.dtype)


class Gemma4LanguageModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig, attn_implementation: str):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i, attn_implementation) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

    def forward(self, input_ids, position_ids, inputs_embeds, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        if inputs_embeds is None:
            weight = self.embed_tokens.weight
            # HF keeps the scale as a tensor cast to the weight dtype (sqrt(5376) rounds in bf16).
            inputs_embeds = self.embed_tokens(input_ids) * torch.tensor(
                self.config.hidden_size**0.5, dtype=weight.dtype, device=weight.device
            )
        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings[layer.layer_type], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        return self.norm(hidden_states)


class Gemma4Backbone(nn.Module):
    """``model``: holds ``language_model`` so parameter names match the checkpoint, and
    exposes its ``layers``/``embed_tokens``/``norm`` where PrimeRL looks for them."""

    def __init__(self, config: Gemma4TextConfig, attn_implementation: str):
        super().__init__()
        self.language_model = Gemma4LanguageModel(config, attn_implementation)

    @property
    def layers(self) -> nn.ModuleList:
        return self.language_model.layers

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    @property
    def norm(self) -> nn.Module:
        return self.language_model.norm

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        reference = input_ids if input_ids is not None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(reference.shape[1], device=reference.device).unsqueeze(0)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=reference.device),
            total_tokens=None if seq_lens_are_pre_shard else reference.shape[1],
        )
        hidden_states = self.language_model(input_ids, position_ids, inputs_embeds, cu_seqlens, max_seqlen)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma4ForCausalLM(PreTrainedModelPrimeRL):
    config: Gemma4Config
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4DecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        text_config = config.get_text_config()
        _check_supported(text_config)
        self.model = Gemma4Backbone(text_config, config._attn_implementation)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(frozenset(), reason="Gemma 4 attention runs outside FlashAttention._compute_attention")

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Same key names as the HF checkpoint: never auto-convert.
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: Optional[Tensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        # Replaced by inject_prime_lm_head's forward in training; kept for direct use.
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return self.lm_head(outputs.last_hidden_state, labels, temperature=temperature)

    def init_buffers_post_meta(self) -> None:
        rotary_emb = self.model.language_model.rotary_emb
        device = next(self.parameters()).device
        fresh = Gemma4TextRotaryEmbedding(rotary_emb.config, device=device)
        for name, buffer in fresh.named_buffers():
            getattr(rotary_emb, name).copy_(buffer)


__all__ = ["Gemma4ForCausalLM"]
