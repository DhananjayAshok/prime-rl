# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
"""Muse Glimmer configs (e.g. meta-models/Muse-Glimmer-30B) for PrimeRL training.

transformers 5.6 has no ``muse_glimmer`` model type, so ``AutoConfig`` cannot read the
checkpoint's config.json without these. The field sets and defaults follow vLLM's native copy
(``vllm/transformers_utils/configs/muse_glimmer.py``), so a checkpoint config deserializes the
same way in the trainer and in the inference server. Only the canonical (nested
``text_config`` / ``vision_config``) layout is supported; the vision config is parsed but the
trainer never builds the vision tower.
"""

from transformers.configuration_utils import PretrainedConfig


def default_no_rope_layers(*, num_hidden_layers: int) -> list[int]:
    """iRoPE mask (1 = RoPE + sliding window, 0 = NoPE + full attention): NoPE every 4th layer,
    counted backward from the last layer (as vLLM derives it)."""
    stride = 4
    return [0 if (num_hidden_layers - 1 - i) % stride == 0 else 1 for i in range(num_hidden_layers)]


class MuseGlimmerTextConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202_048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 16_384,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = 200_001,
        bos_token_id: int | None = 200_000,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        rope_theta: float | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = 2048,
        layer_types: list[str] | None = None,
        layer_rope_theta: list[float] | None = None,
        final_logit_softcapping: float | None = 20.0,
        attn_logit_softcapping: float | None = None,
        qk_scale_factor: float = 43.7840518911,
        use_qk_norm: bool = True,
        use_attn_output_gate: bool = True,
        output_multiplier: float = 0.19611613513818404,
        normalize_tok_embeddings: bool = True,
        post_norm_eps: float = 1e-8,
        no_rope_layers: list[int] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.qk_scale_factor = qk_scale_factor
        self.use_qk_norm = use_qk_norm
        self.use_attn_output_gate = use_attn_output_gate
        self.output_multiplier = output_multiplier
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.post_norm_eps = post_norm_eps
        self.layer_rope_theta = layer_rope_theta

        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta if rope_theta is not None else 500_000.0,
            }
        self.rope_parameters = rope_parameters

        self.no_rope_layers = (
            no_rope_layers
            if no_rope_layers is not None
            else default_no_rope_layers(num_hidden_layers=num_hidden_layers)
        )
        if layer_types is None:
            layer_types = ["full_attention" if flag == 0 else "sliding_attention" for flag in self.no_rope_layers]
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MuseGlimmerVisionConfig(PretrainedConfig):
    """Parsed so the composite config loads; the trainer does not build the vision tower."""

    model_type = "muse_glimmer_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        patch_size: int = 14,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 50,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        hidden_act: str = "gelu",
        merge_kernel_size: int = 2,
        rope_parameters: dict | None = None,
        max_position_embeddings: int = 32 * 32,
        output_dim: int = 6144,
        patch_temporal: int = 2,
        adapter_dim: int = 4096,
        layer_norm_eps: float = 1e-5,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.merge_kernel_size = merge_kernel_size
        self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings
        self.output_dim = output_dim
        self.patch_temporal = patch_temporal
        self.adapter_dim = adapter_dim
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)
        # Set after PretrainedConfig.__init__, whose validation rejects the vision tower's
        # "window_attention" layer type (as vLLM's copy does).
        self.layer_types = layer_types


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig, "vision_config": MuseGlimmerVisionConfig}

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 200092,
        video_token_id: int = 200091,
        **kwargs,
    ):
        if text_config is None and any(key in kwargs for key in ("hidden_size", "num_hidden_layers")):
            raise ValueError(
                "Flat (legacy-converter) Muse Glimmer configs are not supported by PrimeRL; "
                "use a checkpoint with a nested text_config."
            )
        self.text_config = (
            self.sub_configs["text_config"](**text_config) if isinstance(text_config, dict) else text_config
        ) or self.sub_configs["text_config"]()
        self.vision_config = (
            self.sub_configs["vision_config"](**vision_config) if isinstance(vision_config, dict) else vision_config
        ) or self.sub_configs["vision_config"]()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        # The LM head is untied: the top-level flag (read by the trainer's weight loading) must
        # follow the text config, not PretrainedConfig's default of True.
        kwargs["tie_word_embeddings"] = self.text_config.tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["MuseGlimmerConfig", "MuseGlimmerTextConfig", "MuseGlimmerVisionConfig", "default_no_rope_layers"]
