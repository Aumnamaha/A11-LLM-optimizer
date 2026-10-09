from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """Architecture config parsed from GGUF metadata (no hardcoded values)."""
    model_name: str
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_expert: int
    n_expert_used: int
    expert_ffn: int
    ffn: int
    rope_base: float
    rms_eps: float
    ctx_length: int
    vocab_size: int
    eos_token_id: int

    @classmethod
    def from_metadata(cls, metadata: Dict[str, str], vocab_size: int) -> "ModelConfig":
        arch = metadata.get("general.architecture", "qwen3moe")
        p = f"{arch}."

        def geti(key, default):
            v = metadata.get(key)
            try:
                return int(float(v)) if v is not None else default
            except (TypeError, ValueError):
                return default

        def getf(key, default):
            v = metadata.get(key)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        return cls(
            model_name=metadata.get("general.name", "unknown"),
            arch=arch,
            n_layer=geti(p + "block_count", 48),
            n_embd=geti(p + "embedding_length", 2048),
            n_head=geti(p + "attention.head_count", 32),
            n_head_kv=geti(p + "attention.head_count_kv", 4),
            head_dim=geti(p + "attention.key_length", 128),
            n_expert=geti(p + "expert_count", 128),
            n_expert_used=geti(p + "expert_used_count", 8),
            expert_ffn=geti(p + "expert_feed_forward_length", 768),
            ffn=geti(p + "feed_forward_length", 6144),
            rope_base=getf(p + "rope.freq_base", 1e7),
            rms_eps=getf(p + "layer_norm_rms_epsilon", 1e-6),
            # Clamp to a RAM-feasible context; GGUF advertises 262144
            ctx_length=min(geti(p + "context_length", 8192), 8192),
            vocab_size=vocab_size,
            eos_token_id=geti("tokenizer.ggml.eos_token_id", 151645),
        )
