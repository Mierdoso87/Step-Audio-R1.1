#!/usr/bin/env python3
"""
Step-Audio-R1.1 量化脚本
将 BF16 模型量化为 INT4/INT8，降低显存需求

警告：这是实验性功能，可能影响模型质量
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    missing = []
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM
        print(f"✅ auto-gptq installed")
    except ImportError:
        missing.append("auto-gptq")
    
    try:
        from awq import AutoAWQForCausalLM
        print(f"✅ autoawq installed")
    except ImportError:
        missing.append("autoawq")
    
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        missing.append("bitsandbytes")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("\n安装命令:")
        print("pip install auto-gptq autoawq bitsandbytes accelerate")
        return False
    return True


def quantize_with_bitsandbytes(model_path: str, output_path: str, bits: int = 4):
    """
    使用 bitsandbytes 进行量化 (最简单的方法)
    
    注意：bitsandbytes 量化是动态的，不会保存量化后的权重
    主要用于推理时降低显存
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"\n=== 使用 bitsandbytes {bits}-bit 量化 ===")
    
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # normalized float 4
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # 二次量化，进一步压缩
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    print(f"加载模型: {model_path}")
    print("这可能需要几分钟...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 保存配置（bitsandbytes 不保存量化权重，但可以保存配置）
    os.makedirs(output_path, exist_ok=True)
    
    # 保存量化配置
    config = {
        "quantization_method": "bitsandbytes",
        "bits": bits,
        "quant_type": "nf4" if bits == 4 else "int8",
        "double_quant": bits == 4,
        "original_model": model_path,
    }
    
    with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ 量化配置已保存到: {output_path}")
    print("\n注意: bitsandbytes 是动态量化，每次加载时进行量化")
    print("要使用量化模型，在加载时指定 BitsAndBytesConfig")
    
    return model, tokenizer


def quantize_with_gptq(model_path: str, output_path: str, bits: int = 4):
    """
    使用 GPTQ 进行量化 (需要校准数据)
    
    GPTQ 会保存量化后的权重，可以直接加载
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("❌ 请先安装 auto-gptq: pip install auto-gptq")
        return None, None
    
    from transformers import AutoTokenizer
    
    print(f"\n=== 使用 GPTQ {bits}-bit 量化 ===")
    
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,  # 分组量化，平衡精度和压缩率
        desc_act=True,   # 激活值感知
        damp_percent=0.1,
    )
    
    print(f"加载模型: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备校准数据 (使用一些示例文本)
    calibration_data = [
        "Please transcribe the following audio content.",
        "What is the main topic discussed in this audio?",
        "Translate the speech to Chinese.",
        "Summarize the key points from this recording.",
        "The speaker is discussing artificial intelligence and machine learning.",
    ]
    
    # 加载模型进行量化
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )
    
    print("开始量化 (这可能需要 30-60 分钟)...")
    
    # 执行量化
    model.quantize(
        calibration_data,
        batch_size=1,
    )
    
    # 保存量化模型
    print(f"保存量化模型到: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ GPTQ 量化完成!")
    print(f"量化模型保存在: {output_path}")
    
    return model, tokenizer


def quantize_with_awq(model_path: str, output_path: str, bits: int = 4):
    """
    使用 AWQ 进行量化 (激活值感知量化)
    
    AWQ 通常比 GPTQ 更快，精度损失更小
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("❌ 请先安装 autoawq: pip install autoawq")
        return None, None
    
    from transformers import AutoTokenizer
    
    print(f"\n=== 使用 AWQ {bits}-bit 量化 ===")
    
    # AWQ 配置
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
        "version": "GEMM",  # 或 "GEMV" 用于更小的 batch size
    }
    
    print(f"加载模型: {model_path}")
    
    # 加载模型
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("开始 AWQ 量化 (这可能需要 20-40 分钟)...")
    
    # 执行量化
    model.quantize(
        tokenizer,
        quant_config=quant_config,
    )
    
    # 保存
    print(f"保存量化模型到: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ AWQ 量化完成!")
    
    return model, tokenizer


def estimate_memory(model_path: str):
    """估算不同量化方法的显存需求"""
    
    # 获取模型大小
    total_size = 0
    model_dir = Path(model_path)
    for f in model_dir.glob("*.safetensors"):
        total_size += f.stat().st_size
    
    total_gb = total_size / (1024**3)
    
    print(f"\n=== 显存需求估算 ===")
    print(f"原始模型大小: {total_gb:.1f} GB (BF16)")
    print()
    
    estimates = [
        ("BF16 (当前)", total_gb, total_gb * 1.3),
        ("INT8", total_gb * 0.5, total_gb * 0.5 * 1.3),
        ("INT4 (GPTQ/AWQ)", total_gb * 0.25, total_gb * 0.25 * 1.5),
    ]
    
    print(f"{'方法':<20} {'模型大小':<12} {'推理显存':<12} {'可用 GPU'}")
    print("-" * 65)
    
    for name, model_size, vram in estimates:
        if vram > 80:
            gpus = "4× L40S / 2× H100"
        elif vram > 46:
            gpus = "2× L40S / 1× H100 80GB"
        elif vram > 24:
            gpus = "1× L40S / 1× A100 40GB"
        else:
            gpus = "1× RTX 4090 24GB ⭐"
        
        print(f"{name:<20} {model_size:.1f} GB{'':<6} {vram:.1f} GB{'':<6} {gpus}")


def main():
    parser = argparse.ArgumentParser(description="Step-Audio-R1.1 量化工具")
    parser.add_argument("--model", default="/model", help="原始模型路径")
    parser.add_argument("--output", default="./Step-Audio-R1.1-quantized", help="输出路径")
    parser.add_argument("--method", choices=["gptq", "awq", "bnb"], default="bnb",
                        help="量化方法: gptq, awq, bnb (bitsandbytes)")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, help="量化位数")
    parser.add_argument("--estimate", action="store_true", help="仅估算显存需求")
    parser.add_argument("--check", action="store_true", help="检查依赖")
    
    args = parser.parse_args()
    
    if args.check:
        check_dependencies()
        return
    
    if args.estimate:
        estimate_memory(args.model)
        return
    
    print("=" * 60)
    print("Step-Audio-R1.1 量化工具")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"输出路径: {args.output}")
    print(f"量化方法: {args.method}")
    print(f"量化位数: {args.bits}-bit")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 执行量化
    if args.method == "bnb":
        quantize_with_bitsandbytes(args.model, args.output, args.bits)
    elif args.method == "gptq":
        quantize_with_gptq(args.model, args.output, args.bits)
    elif args.method == "awq":
        quantize_with_awq(args.model, args.output, args.bits)


if __name__ == "__main__":
    main()
