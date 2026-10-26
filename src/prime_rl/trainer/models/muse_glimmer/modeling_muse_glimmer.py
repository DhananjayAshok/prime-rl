# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
"""Muse Glimmer text decoder (e.g. meta-models/Muse-Glimmer-30B) for PrimeRL training.

Text-only: the checkpoint's vision tower / adapter / projection are never built or loaded.
Module paths mirror the HF checkpoint (``model.language_model.*``, ``lm_head``), so weights load
by name and LoRA adapter keys use the names vLLM's ``MuseGlimmerForCausalLM`` maps (its
``hf_to_vllm_mapper`` turns ``model.language_model.`` into ``model.``, ``self_attn.gate_proj``
into its ``output_gate_proj`` and packs q/k/v and mlp gate/up).

Numerics follow vLLM's ``vllm/model_executor/models/muse_glimmer.py`` (the reference: transformers
has no Muse Glimmer model), a Gemma 2 derivative with these deltas:

* token embeddings pass through a weightless RMSNorm (no sqrt(hidden) scaling);
* sandwich norms ``x * (1 + w)``: the pre-norms use ``rms_norm_eps``, the post-norms
  ``post_norm_eps``; the final norm is ``x * w`` (no offset);
* SiLU-gated MLP;
* attention: weightless fp32 QK-norm over head_dim *before* RoPE, then the query is scaled by
  ``qk_scale_factor`` (already folded by 1/sqrt(head_dim) in the HF config) and softmax uses the
  standard 1/sqrt(head_dim); a sigmoid output gate ``sigmoid(gate_proj(x)) * attn`` (x is the
  layer's normed input) before ``o_proj``;
* iRoPE: NoPE layers (``no_rope_layers[i] == 0``, every 4th counted from the last) run full
  attention, the others run RoPE (rotate-half / NEOX layout, as the HF converter permutes q/k)
  with a sliding window;
* logits are multiplied by ``output_multiplier`` and then tanh-softcapped at
  ``final_logit_softcapping``. The multiplier is folded into the final norm (the LM head is
  linear, so ``lm_head(h * m) == lm_head(h) * m``), so PrimeRL's LM heads only need the softcap,
  which ``inject_prime_lm_head`` reads from the text config.

Attention runs over the packed batch with the varlen flash kernels (head_dim 128 everywhere),
or with SDPA per packed document when the attention implementation is not a flash one.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init
from transformers.modeling_outputs import BaseModelOutputWithPast

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig, MuseGlimmerTextConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

FLASH_VERSIONS = {"flash_attention_2": 2, "flash_attention_3": 3, "flash_attention_4": 4}


def check_supported(*, config: MuseGlimmerTextConfig) -> None:
    """Refuse configs this implementation would silently get wrong."""
    if config.hidden_activation != "silu":
        raise ValueError(
            f"Muse Glimmer uses a SiLU MLP; this config sets hidden_activation={config.hidden_activation!r}"
        )
    if config.attn_logit_softcapping is not None:
        raise ValueError("PrimeRL's Muse Glimmer does not support attention logit softcapping")
    rope_type = config.rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"PrimeRL's Muse Glimmer supports default RoPE only; this config sets rope_type={rope_type!r}")
    num_layers = config.num_hidden_layers
    if len(config.no_rope_layers) < num_layers or len(config.layer_types) < num_layers:
        raise ValueError("no_rope_layers and layer_types must cover every layer")
    for i in range(num_layers):
        expected = "sliding_attention" if config.no_rope_layers[i] == 1 else "full_attention"
        if config.layer_types[i] != expected:
            raise ValueError(
                f"layer {i}: layer_types says {config.layer_types[i]!r} but no_rope_layers says {expected!r} "
                "(Muse Glimmer runs RoPE layers with a sliding window and NoPE layers with full attention)"
            )
    if config.layer_rope_theta is not None:
        theta = config.rope_parameters["rope_theta"]
        for i in range(num_layers):
            layer_theta = config.layer_rope_theta[i]
            if (layer_theta == 0) != (config.no_rope_layers[i] == 0) or (layer_theta != 0 and layer_theta != theta):
                raise ValueError(
                    f"layer {i}: layer_rope_theta={layer_theta} does not match rope_theta={theta} "
                    f"and no_rope_layers={config.no_rope_layers[i]}"
                )


def query_prescale(*, config: MuseGlimmerTextConfig) -> float:
    """The post-QK-norm query scale, as vLLM's ``_muse_glimmer_query_prescale``: the HF config ships
    ``qk_scale_factor`` already divided by sqrt(head_dim) (~3.87); a raw native value (~43.78, at
    least sqrt(head_dim)) is divided here."""
    explicit = getattr(config, "scale_query_by", None)
    if explicit is not None:
        return float(explicit)
    qk_scale = float(config.qk_scale_factor)
    sqrt_head_dim = math.sqrt(config.head_dim)
    return qk_scale / sqrt_head_dim if qk_scale >= sqrt_head_dim else qk_scale


class MuseGlimmerRMSNorm(nn.Module):
    """RMSNorm in fp32, cast back to the input dtype: ``norm(x) * (w + weight_offset) * out_scale``,
    or ``norm(x) * out_scale`` when weightless (QK-norm and the token-embedding norm)."""

    def __init__(
        self,
        *,
        dim: int | None = None,
        eps: float,
        with_scale: bool = True,
        weight_offset: float = 0.0,
        out_scale: float = 1.0,
    ):
        super().__init__()
        self.eps = eps
        self.weight_offset = weight_offset
        self.out_scale = out_scale
        if with_scale:
            self.weight = nn.Parameter(torch.zeros(dim) if weight_offset else torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * (self.weight.float() + self.weight_offset)
        if self.out_scale != 1.0:
            out = out * self.out_scale
        return out.type_as(hidden_states)


class MuseGlimmerRotaryEmbedding(nn.Module):
    """Default RoPE (rotate-half layout) over the full head_dim."""

    inv_freq: Tensor

    def __init__(self, *, head_dim: int, rope_theta: float, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self.compute_inv_freq(device=device), persistent=False)

    def compute_inv_freq(self, *, device=None) -> Tensor:
        exponents = torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        return 1.0 / (self.rope_theta ** (exponents / self.head_dim))

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class MuseGlimmerAttention(FlashAttention):
    """PrimeRL's FlashAttention (q/k/v/o projections, flash kernel dispatch, optional qkv fusion)
    with Muse Glimmer's weightless QK-norm + query pre-scale, iRoPE and sigmoid output gate."""

    def __init__(self, *, config: MuseGlimmerTextConfig, layer_idx: int, attn_implementation: str):
        flash_version = FLASH_VERSIONS.get(attn_implementation)
        super().__init__(
            AttentionConfig(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                is_causal=True,
                attention_bias=config.attention_bias,
                use_qk_norm=False,  # Muse's QK-norm is weightless: applied in forward below
                rms_norm_eps=config.rms_norm_eps,
            ),
            flash_attn_version=flash_version or 2,
        )
        self.use_flash = flash_version is not None
        self.use_rope = config.no_rope_layers[layer_idx] == 1
        # FlashAttention._compute_attention reads this: RoPE layers slide, NoPE layers attend fully.
        self.sliding_window = config.sliding_window if self.use_rope else None
        self.qk_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False) if config.use_qk_norm else None
        self.scale_query_by = query_prescale(config=config)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
            if config.use_attn_output_gate
            else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[Tensor, None]:
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query, key, value = self.project_qkv(hidden_states)
        query, key, value = query.view(hidden_shape), key.view(hidden_shape), value.view(hidden_shape)

        if self.qk_norm is not None:
            query = (self.qk_norm(query) * self.scale_query_by).to(value.dtype)
            key = self.qk_norm(key).to(value.dtype)
        if self.use_rope:
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)

        # [1, T, H, D] -> [T, H, D]
        if self.use_flash:
            out = self._compute_attention(query[0], key[0], value[0], cu_seqlens, max_seqlen)
        else:
            out = self._sdpa_per_document(query=query[0], key=key[0], value=value[0], cu_seqlens=cu_seqlens)
        attn_output = out.reshape(1, out.shape[0], -1)
        if self.gate_proj is not None:
            attn_output = torch.sigmoid(self.gate_proj(hidden_states)) * attn_output
        return self.o_proj(attn_output), None

    def _sdpa_per_document(self, *, query: Tensor, key: Tensor, value: Tensor, cu_seqlens: Tensor) -> Tensor:
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
                q, k, v, attn_mask=mask, is_causal=mask is None, scale=self.scaling, enable_gqa=True
            )
            outputs.append(out.transpose(0, 1))
        return torch.cat(outputs, dim=0)


class MuseGlimmerMLP(nn.Module):
    def __init__(self, *, config: MuseGlimmerTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, *, config: MuseGlimmerTextConfig, layer_idx: int, attn_implementation: str):
        super().__init__()
        self.self_attn = MuseGlimmerAttention(
            config=config, layer_idx=layer_idx, attn_implementation=attn_implementation
        )
        self.mlp = MuseGlimmerMLP(config=config)
        hidden, eps, post_eps = config.hidden_size, config.rms_norm_eps, config.post_norm_eps
        self.input_layernorm = MuseGlimmerRMSNorm(dim=hidden, eps=eps, weight_offset=1.0)
        self.post_attention_layernorm = MuseGlimmerRMSNorm(dim=hidden, eps=post_eps, weight_offset=1.0)
        self.pre_feedforward_layernorm = MuseGlimmerRMSNorm(dim=hidden, eps=eps, weight_offset=1.0)
        self.post_feedforward_layernorm = MuseGlimmerRMSNorm(dim=hidden, eps=post_eps, weight_offset=1.0)

    def forward(self, hidden_states: Tensor, position_embeddings, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class MuseGlimmerLanguageModel(nn.Module):
    def __init__(self, *, config: MuseGlimmerTextConfig, attn_implementation: str):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False) if config.normalize_tok_embeddings else None
        )
        self.layers = nn.ModuleList(
            [
                MuseGlimmerDecoderLayer(config=config, layer_idx=i, attn_implementation=attn_implementation)
                for i in range(config.num_hidden_layers)
            ]
        )
        # Final norm: weight as the scale (no offset), with the logit output_multiplier folded in.
        self.norm = MuseGlimmerRMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps, out_scale=config.output_multiplier
        )
        self.rotary_emb = MuseGlimmerRotaryEmbedding(
            head_dim=config.head_dim, rope_theta=config.rope_parameters["rope_theta"]
        )

    def forward(self, input_ids, position_ids, inputs_embeds, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.embed_norm is not None:
                inputs_embeds = self.embed_norm(inputs_embeds)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.norm(hidden_states)


class MuseGlimmerBackbone(nn.Module):
    """``model``: holds ``language_model`` so parameter names match the checkpoint, and exposes
    its ``layers``/``embed_tokens``/``norm`` where PrimeRL looks for them."""

    def __init__(self, *, config: MuseGlimmerTextConfig, attn_implementation: str):
        super().__init__()
        self.language_model = MuseGlimmerLanguageModel(config=config, attn_implementation=attn_implementation)

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


class MuseGlimmerForCausalLM(PreTrainedModelPrimeRL):
    config: MuseGlimmerConfig
    config_class = MuseGlimmerConfig
    base_model_prefix = "model"
    _no_split_modules = ["MuseGlimmerDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: MuseGlimmerConfig):
        super().__init__(config)
        text_config = config.get_text_config()
        check_supported(config=text_config)
        self.model = MuseGlimmerBackbone(config=text_config, attn_implementation=config._attn_implementation)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        # transformers' generic init sets every *RMSNorm weight to 1; the sandwich norms store
        # w with the +1 offset applied at runtime, so their identity is 0.
        if isinstance(module, MuseGlimmerRMSNorm) and module.weight is not None and module.weight_offset:
            init.zeros_(module.weight)

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(frozenset(), reason="Muse Glimmer's sliding-window/NoPE attention is untested under CP")

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
        rotary_emb.inv_freq.copy_(rotary_emb.compute_inv_freq(device=rotary_emb.inv_freq.device))


__all__ = ["MuseGlimmerForCausalLM"]
