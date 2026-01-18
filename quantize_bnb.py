#!/usr/bin/env python3
"""
Step-Audio-R1.1 量化脚本 - 使用 bitsandbytes INT4
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_path = "/model"
    output_path = "/workspace/Step-Audio-R1.1-INT4"
    
    print("=" * 60)
    print("Step-Audio-R1.1 INT4 量化 (bitsandbytes)")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory/1024**3:.1f} GB")
    print("=" * 60)
    
    # INT4 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\n[1/4] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  词表大小: {len(tokenizer)}")
    
    print("\n[2/4] 加载并量化模型 (这需要几分钟)...")
    print("  正在加载...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    print("\n[3/4] 显存使用统计:")
    total_mem = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        total_mem += mem
        print(f"  GPU {i}: {mem:.2f} GB")
    print(f"  总计: {total_mem:.2f} GB")
    
    print("\n[4/4] 保存量化模型...")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型和 tokenizer
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # 复制必要的配置文件
    src_path = Path(model_path)
    dst_path = Path(output_path)
    
    for f in ["configuration_step_audio_2.py", "modeling_step_audio_2.py"]:
        src_file = src_path / f
        if src_file.exists():
            shutil.copy(src_file, dst_path / f)
            print(f"  复制: {f}")
    
    # 计算文件大小
    original_size = sum(f.stat().st_size for f in src_path.glob("*.safetensors")) / 1024**3
    quantized_size = sum(f.stat().st_size for f in dst_path.glob("*.safetensors")) / 1024**3 if list(dst_path.glob("*.safetensors")) else 0
    
    print(f"\n✅ 量化完成！")
    print(f"  输出路径: {output_path}")
    print(f"  原始大小: {original_size:.1f} GB")
    print(f"  量化大小: {quantized_size:.1f} GB" if quantized_size > 0 else "  (动态量化，无独立权重文件)")
    
    # 保存量化配置
    quant_config = {
        "quantization_method": "bitsandbytes",
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
        "original_model": model_path,
        "total_gpu_memory_gb": total_mem,
    }
    
    with open(dst_path / "quantization_info.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    
    print("\n=== 测试量化模型 ===")
    # 简单测试
    test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=20)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"测试输入: Hello, how are you?")
    print(f"模型输出: {response[:100]}...")
    
    print("\n✅ 量化模型测试通过！")
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
