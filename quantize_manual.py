#!/usr/bin/env python3
"""
Step-Audio-R1.1 手动量化脚本
只量化 Qwen2 LLM 部分，保持 AudioEncoder 为 FP16
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def quantize_tensor_to_int4(tensor, group_size=128):
    """将 tensor 量化为 INT4"""
    original_shape = tensor.shape
    
    # 展平为 2D
    if len(original_shape) == 1:
        tensor = tensor.unsqueeze(0)
    
    rows, cols = tensor.shape[0], tensor.shape[-1]
    
    # 按组量化
    if cols % group_size != 0:
        # 填充
        pad_size = group_size - (cols % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        cols = tensor.shape[-1]
    
    tensor = tensor.reshape(-1, group_size)
    
    # 计算缩放因子
    max_vals = tensor.abs().max(dim=1, keepdim=True)[0]
    scales = max_vals / 7.0  # INT4 范围 [-8, 7]
    scales = scales.clamp(min=1e-8)
    
    # 量化
    quantized = torch.round(tensor / scales).clamp(-8, 7).to(torch.int8)
    
    return quantized, scales, original_shape

def dequantize_tensor(quantized, scales, original_shape, group_size=128):
    """反量化"""
    dequantized = quantized.float() * scales
    dequantized = dequantized.reshape(original_shape[0], -1)
    
    # 移除填充
    if dequantized.shape[-1] > original_shape[-1]:
        dequantized = dequantized[..., :original_shape[-1]]
    
    if len(original_shape) == 1:
        dequantized = dequantized.squeeze(0)
    
    return dequantized

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_path = "/model"
    output_path = "/workspace/Step-Audio-R1.1-INT4-Manual"
    
    print("=" * 60)
    print("Step-Audio-R1.1 手动 INT4 量化")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60)
    
    print("\n[1/4] 加载原始模型...")
    
    # 加载模型到 CPU 以节省 GPU 显存
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  模型类型: {type(model).__name__}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e9:.2f}B")
    
    print("\n[2/4] 量化 LLM 权重...")
    
    # 只量化 Qwen2 模型的线性层
    quantized_state_dict = {}
    scales_dict = {}
    
    state_dict = model.state_dict()
    
    # 需要量化的层
    layers_to_quantize = []
    for name in state_dict.keys():
        if 'model.layers' in name and 'weight' in name:
            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                        'gate_proj', 'up_proj', 'down_proj']):
                layers_to_quantize.append(name)
    
    print(f"  需要量化的层: {len(layers_to_quantize)}")
    
    # 量化
    for name in tqdm(layers_to_quantize, desc="量化进度"):
        tensor = state_dict[name].float()
        quantized, scales, orig_shape = quantize_tensor_to_int4(tensor)
        
        quantized_state_dict[name] = quantized
        scales_dict[name + ".scales"] = scales
    
    # 保存非量化的层
    for name, tensor in state_dict.items():
        if name not in quantized_state_dict:
            quantized_state_dict[name] = tensor
    
    print("\n[3/4] 保存量化模型...")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存量化权重
    torch.save(quantized_state_dict, os.path.join(output_path, "quantized_weights.pt"))
    torch.save(scales_dict, os.path.join(output_path, "scales.pt"))
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_path)
    
    # 复制配置文件
    src_path = Path(model_path)
    dst_path = Path(output_path)
    
    for f in ["config.json", "configuration_step_audio_2.py", "modeling_step_audio_2.py",
              "special_tokens_map.json", "added_tokens.json", "tokenizer_config.json", "vocab.json"]:
        src_file = src_path / f
        if src_file.exists():
            shutil.copy(src_file, dst_path / f)
    
    # 计算大小
    original_size = sum(f.stat().st_size for f in src_path.glob("*.safetensors")) / 1024**3
    quantized_size = (
        os.path.getsize(os.path.join(output_path, "quantized_weights.pt")) +
        os.path.getsize(os.path.join(output_path, "scales.pt"))
    ) / 1024**3
    
    print(f"\n[4/4] 量化完成！")
    print(f"  原始大小: {original_size:.1f} GB")
    print(f"  量化大小: {quantized_size:.1f} GB")
    print(f"  压缩比: {original_size/quantized_size:.1f}x")
    
    # 保存量化信息
    quant_info = {
        "method": "manual_int4",
        "group_size": 128,
        "quantized_layers": len(layers_to_quantize),
        "original_size_gb": original_size,
        "quantized_size_gb": quantized_size,
    }
    
    with open(dst_path / "quantization_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)
    
    print(f"\n✅ 量化模型保存在: {output_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
