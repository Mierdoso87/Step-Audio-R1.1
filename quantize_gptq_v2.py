#!/usr/bin/env python3
"""
Step-Audio-R1.1 GPTQ 量化脚本
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main():
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    model_path = "/model"
    output_path = "/workspace/Step-Audio-R1.1-GPTQ-INT4"
    
    print("=" * 60)
    print("Step-Audio-R1.1 GPTQ INT4 量化")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory/1024**3:.1f} GB")
    print("=" * 60)
    
    # GPTQ 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        damp_percent=0.1,
    )
    
    print("\n[1/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  词表大小: {len(tokenizer)}")
    
    print("\n[2/5] 加载模型进行量化...")
    print("  这可能需要几分钟...")
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    print("\n[3/5] 准备校准数据...")
    # 校准数据 - 用于量化校准
    calibration_texts = [
        "Please transcribe the following audio content accurately.",
        "What is the main topic discussed in this audio recording?",
        "Translate the speech from English to Chinese.",
        "Summarize the key points from this audio file.",
        "The speaker is discussing artificial intelligence and machine learning.",
        "Can you identify the emotions expressed in this audio clip?",
        "What language is being spoken in this recording?",
        "Please provide a detailed analysis of the audio content.",
        "这段音频的主要内容是什么？请详细描述。",
        "请将这段语音翻译成英文，保持原意。",
        "音频中说话人的情绪是怎样的？是高兴还是悲伤？",
        "请总结这段录音的要点，列出关键信息。",
        "Hello, my name is John and I am here to discuss the quarterly results.",
        "The weather today is sunny with a high of 75 degrees Fahrenheit.",
        "In this lecture, we will explore the fundamentals of quantum computing.",
        "Thank you for joining us today. Let's begin with the agenda.",
    ]
    
    # 编码校准数据
    calibration_dataset = []
    for text in calibration_texts:
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        calibration_dataset.append(encoded.input_ids)
    
    print(f"  校准数据: {len(calibration_dataset)} 条")
    
    print("\n[4/5] 开始 GPTQ 量化...")
    print("  这是最耗时的步骤，可能需要 30-60 分钟")
    print("  请耐心等待...")
    
    model.quantize(calibration_dataset, batch_size=1)
    
    print("\n[5/5] 保存量化模型...")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)
    
    # 复制必要的配置文件
    src_path = Path(model_path)
    dst_path = Path(output_path)
    
    for f in ["configuration_step_audio_2.py", "modeling_step_audio_2.py", 
              "special_tokens_map.json", "added_tokens.json"]:
        src_file = src_path / f
        if src_file.exists():
            shutil.copy(src_file, dst_path / f)
            print(f"  复制: {f}")
    
    # 计算文件大小
    original_size = sum(f.stat().st_size for f in src_path.glob("*.safetensors")) / 1024**3
    quantized_files = list(dst_path.glob("*.safetensors")) + list(dst_path.glob("*.bin"))
    quantized_size = sum(f.stat().st_size for f in quantized_files) / 1024**3
    
    print(f"\n✅ GPTQ 量化完成！")
    print(f"  输出路径: {output_path}")
    print(f"  原始大小: {original_size:.1f} GB")
    print(f"  量化大小: {quantized_size:.1f} GB")
    print(f"  压缩比: {original_size/quantized_size:.1f}x" if quantized_size > 0 else "")
    
    # 保存量化信息
    quant_info = {
        "quantization_method": "GPTQ",
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
        "original_size_gb": original_size,
        "quantized_size_gb": quantized_size,
    }
    
    with open(dst_path / "quantization_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)
    
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
