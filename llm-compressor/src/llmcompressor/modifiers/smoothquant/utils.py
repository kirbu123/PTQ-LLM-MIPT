import functools
from collections import namedtuple
from typing import Dict, List, Tuple, Union

from loguru import logger

__all__ = [
    "get_layer_mappings_from_architecture",
    "MAPPINGS_REGISTRY",
    "DEFAULT_SMOOTHQUANT_MAPPINGS",
]

LayerMapType = Tuple[Union[List[str], str], Union[List[str], str]]
LayerMap: LayerMapType = namedtuple("LayerMap", ["balance_layers", "smooth_layers"])

DEFAULT_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
MIXTRAL_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
]
BLOOM_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*query_key_value"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*dense_h_to_4h"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]

# GPT-NeoX (GPTNeoXForCausalLM). EleutherAI Pythia families such as
# EleutherAI/pythia-{N}m (e.g. 14m, 31m, 70m, 160m, …), *-deduped variants, and
# pythia-*b sizes share this class. Same FFN split as Bloom — only dense_h_to_4h with
# post_attention_layernorm (not dense_4h_to_h), analogous to OPT fc1-only.
GPT_NEOX_SMOOTHQUANT_MAPPINGS: List[LayerMap] = BLOOM_SMOOTHQUANT_MAPPINGS
PHI3_VISION_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*qkv_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
WHISPER_V2_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*k_proj", "re:.*v_proj", "re:.*q_proj"],
        smooth_layers="re:.*self_attn_layer_norm",
    ),
    LayerMap(
        balance_layers=["re:.*fc1"],
        smooth_layers="re:.*final_layer_norm",
    ),
]

# Per-layer final_layer_norm only — avoids matching decoder.final_layer_norm.
OPT_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*self_attn_layer_norm",
    ),
    LayerMap(
        balance_layers=["re:.*fc1"],
        smooth_layers=r"re:.*layers\.\d+\.final_layer_norm",
    ),
]

DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*kv_a_proj_with_mqa"],
        smooth_layers="re:.*input_layernorm",
    ),
]


# Registry of layer mappings for different architectures
#   Add more mappings here
MAPPINGS_REGISTRY: Dict[str, List[LayerMap]] = {
    "LlamaForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MixtralForCausalLM": MIXTRAL_SMOOTHQUANT_MAPPINGS,
    "MistralForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Qwen2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "BloomForCausalLM": BLOOM_SMOOTHQUANT_MAPPINGS,
    "ChatGLMForConditionalGeneration": BLOOM_SMOOTHQUANT_MAPPINGS,
    "Phi3VForCausalLM": PHI3_VISION_SMOOTHQUANT_MAPPINGS,
    "WhisperForConditionalGeneration": WHISPER_V2_SMOOTHQUANT_MAPPINGS,
    "OPTForCausalLM": OPT_SMOOTHQUANT_MAPPINGS,
    # EleutherAI/pythia-*m, deduped variants, and other GPT-NeoX causal LMs
    "GPTNeoXForCausalLM": GPT_NEOX_SMOOTHQUANT_MAPPINGS,
    "DeepseekV2ForCausalLM": DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
}


def get_layer_mappings_from_architecture(architecture: str) -> List[LayerMap]:
    """
    :param architecture: str: The architecture of the model
    :return: list: The layer mappings for the given architecture
    """

    if architecture not in MAPPINGS_REGISTRY:
        logger.info(
            f"Architecture {architecture} not found in mappings. "
            f"Using default mappings: {DEFAULT_SMOOTHQUANT_MAPPINGS}"
        )

    return MAPPINGS_REGISTRY.get(architecture, DEFAULT_SMOOTHQUANT_MAPPINGS)


def handle_mapping_resolution_errors(func):
    """
    Decorator to catch any errors that occur when resolving mappings and provide a
    helpful error message to the user pointing them to the README
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as original_exception:
            readme_location = (
                "https://github.com/vllm-project/llm-compressor/tree/main/"
                "src/llmcompressor/modifiers/smoothquant"
            )
            raise RuntimeError(
                f"Error resolving mappings for given architecture."
                f"Please refer to the README at {readme_location} for more information."
            ) from original_exception

    return wrapper
