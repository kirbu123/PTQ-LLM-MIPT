import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import Conv1D
from deepspeed.compression.compress import init_compression
from deepspeed.compression.helper import convert_conv1d_to_linear
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
)


def dump_param_dtypes(model, out_path=None, per_param=False):
    lines = []
    if per_param:
        for name, p in model.named_parameters():
            lines.append(f"PARAM {name}: {p.dtype} - shape: {tuple(p.shape)}")
    else:
        for name, mod in model.named_modules():
            pname = name if name else "<root>"
            p_types = {str(p.dtype) for p in mod.parameters(recurse=False)}
            b_types = {str(b.dtype) for _, b in mod.named_buffers(recurse=False)}
            if p_types or b_types:
                lines.append(f"MODULE {pname} [{type(mod).__name__}] "
                             f"param_dtypes={sorted(p_types)} buffer_dtypes={sorted(b_types)}")
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)

def list_quantizers(model):
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "Quantizer" in cls or "Quant" in cls:
            print(f"Quant module: {name} -> {cls}")

def get_layer_dtype(layer):
    """Safely get the dtype of a layer by checking its parameters"""
    params = list(layer.parameters())
    if params:
        return params[0].dtype
    return "No parameters"

def log_model_quantization(model):
    # Count && log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    """Comprehensive analysis of model quantization including parameter counts"""

    # Initialize counters
    quant_counter = 0
    weight_counter = 0
    act_counter = 0
    module_count = 0
    
    # Parameter counters
    total_params = 0
    quantized_weight_params = 0
    quantized_activation_params = 0
    quantized_total_params = 0

    print("\n=== Detailed Quantizer Analysis ===")
    for name, module in model.named_modules():
        # Count parameters in this module
        module_params = sum(p.numel() for p in module.parameters())
        total_params += module_params

        # Check for quantizers
        weight_quantizer = getattr(module, 'weight_quantizer', None)
        activation_quantizer = getattr(module, 'activation_quantizer', None)

        has_weight_quant = weight_quantizer is not None
        has_act_quant = activation_quantizer is not None

        # Count quantized modules
        if has_weight_quant or has_act_quant:
            quant_counter += 1
            quantized_total_params += module_params
 
        if has_weight_quant:
            weight_counter += 1
            quantized_weight_params += module_params

        if has_act_quant:
            act_counter += 1
            # For activation quantization, we count the parameters that will pass through the quantizer
            # This is typically the same as the module parameters
            quantized_activation_params += module_params

        module_count += 1

    # Calculate percentages
    weight_quant_percentage = (quantized_weight_params / total_params) * 100 if total_params > 0 else 0
    act_quant_percentage = (quantized_activation_params / total_params) * 100 if total_params > 0 else 0
    total_quant_percentage = (quantized_total_params / total_params) * 100 if total_params > 0 else 0

    print(f"\n=== Quantization Summary ===")
    print(f"Total modules: {module_count}")
    print(f"Quantized modules: {quant_counter}")
    print(f"Weight quantized modules: {weight_counter}")
    print(f"Activation quantized modules: {act_counter}")

    print(f"\n=== Parameter Statistics ===")
    print(f"Total model parameters: {total_params:,}")
    print(f"Parameters in weight-quantized modules: {quantized_weight_params:,} ({weight_quant_percentage:.2f}%)")
    print(f"Parameters in activation-quantized modules: {quantized_activation_params:,} ({act_quant_percentage:.2f}%)")
    print(f"Parameters in any quantized modules: {quantized_total_params:,} ({total_quant_percentage:.2f}%)")

    # Additional breakdown by layer type
    print(f"\n=== Layer-wise Breakdown ===")
    layer_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_params = sum(p.numel() for p in module.parameters())
        
        if module_type not in layer_types:
            layer_types[module_type] = {
                'count': 0,
                'total_params': 0,
                'quantized_count': 0,
                'quantized_params': 0
            }
        
        layer_types[module_type]['count'] += 1
        layer_types[module_type]['total_params'] += module_params
        
        weight_quantizer = getattr(module, 'weight_quantizer', None)
        activation_quantizer = getattr(module, 'activation_quantizer', None)
        
        if weight_quantizer or activation_quantizer:
            layer_types[module_type]['quantized_count'] += 1
            layer_types[module_type]['quantized_params'] += module_params
    
    for layer_type, stats in layer_types.items():
        if stats['total_params'] > 0:  # Only show layers with parameters
            quant_percent = (stats['quantized_params'] / stats['total_params']) * 100
            print(f"{layer_type}: {stats['count']} modules, {stats['total_params']:,} params, "
                  f"{stats['quantized_count']} quantized ({quant_percent:.1f}% of params)")

    return {
        'total_modules': module_count,
        'quantized_modules': quant_counter,
        'weight_quantized_modules': weight_counter,
        'activation_quantized_modules': act_counter,
        'total_parameters': total_params,
        'quantized_weight_parameters': quantized_weight_params,
        'quantized_activation_parameters': quantized_activation_params,
        'quantized_total_parameters': quantized_total_params,
        'weight_quant_percentage': weight_quant_percentage,
        'act_quant_percentage': act_quant_percentage,
        'total_quant_percentage': total_quant_percentage
    }


if __name__ == "__main__":
    model_path = "/home/buka2004/PTQ-LLM-MIPT/DeepSpeedExamples/compression/gpt2/out/ZeroQuant/W8A8_quantization_lkd/quantized_model/best/pytorch_model.pt"
    model_name = "openai-community/gpt2-large"
    ds_config  = "/home/buka2004/PTQ-LLM-MIPT/DeepSpeedExamples/compression/gpt2/config/ds_config_W8A8_Qgroup64_fp32.json"

    # Load FP32 base and match training conversion
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = convert_conv1d_to_linear(model, Conv1D)

    # Load your weights (fixes path usage)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Attach ZeroQuant runtime quantization
    model = init_compression(model, ds_config)

    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2Attention,
        GPT2Block,
    )

    print("=== Checking Layer Dtypes ===")
    
    for name, m in model.transformer.named_modules():
        if isinstance(m, GPT2Block):
            # Get dtypes from parameters
            c_fc_dtype = get_layer_dtype(m.mlp.c_fc)
            c_proj_dtype = get_layer_dtype(m.mlp.c_proj)
            print(f"MLP {name}: c_fc={c_fc_dtype}, c_proj={c_proj_dtype}")
            
        elif isinstance(m, GPT2Attention):
            c_attn_dtype = get_layer_dtype(m.c_attn)
            c_proj_dtype = get_layer_dtype(m.c_proj)
            print(f"Attention {name}: c_attn={c_attn_dtype}, c_proj={c_proj_dtype}")

    # # Alternative: Check all parameter dtypes directly
    # print("\n=== All Parameter Dtypes (Direct Check) ===")
    # for name, param in model.named_parameters():
    #     if any(x in name for x in ['c_fc', 'c_proj', 'c_attn']):
    #         print(f"{name}: {param.dtype} - shape: {tuple(param.shape)}")

    # Check if weights are quantized by looking for quantizer modules
    print("\n=== Quantizer Modules ===")
    list_quantizers(model)

    quant_counter = 0
    weight_counter = 0
    act_counter = 0
    module_count = 0

    print("\n=== Detailed Quantizer Analysis ===")
    for name, module in model.named_modules():
        # Проверяем наличие квантизаторов
        weight_quantizer = getattr(module, 'weight_quantizer', None)
        activation_quantizer = getattr(module, 'activation_quantizer', None)

        print(f'{name}, weight_quantizer: {weight_quantizer is not None}, activation_quantizer: {activation_quantizer is not None}')
        if (weight_quantizer is not None) or (activation_quantizer is not None):
            quant_counter += 1
        if (weight_quantizer is not None):
            weight_counter += 1
        if (activation_quantizer is not None):
            act_counter += 1
        module_count += 1

    print(f'module_count: {module_count}')
    print(f'quant_counter: {quant_counter}')
    print(f'weight_counter: {weight_counter}')
    print(f'act_counter: {act_counter}')
