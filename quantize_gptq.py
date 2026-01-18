#!/usr/bin/env python3
"""
Step-Audio-R1.1 GPTQ 量化脚本
在 Docker 容器内运行
"""

import os
import sys
import json
import torch
import gc
from pathlib import Path

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def find_model_path():
    """查找模型路径"""
    possible_paths = [
        "/root/.cache/huggingface/hub/models--stepfun-ai--Step-Audio-R1.1",
        "/root/.cache/huggingface/hub/models--stepfun-ai--Step-Audio-R1",
    ]
    
    for base in possible_paths:
        if os.path.exists(base):
            # 找到 snapshots 目录
            snapshots = Path(base) / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if (snap / "config.json").exists():
                        return str(snap)
    
    # 尝试直接下载
    return "stepfun-ai/Step-Audio-R1.1"

def quantize_with_gptq():
    """使用 GPTQ 进行 INT4 量化"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    model_path = find_model_path()
    output_path = "/workspace/Step-Audio-R1.1-GPTQ-INT4"
    
    print("=" * 60)
    print("Step-Audio-R1.1 GPTQ INT4 量化")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print("=" * 60)
    
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        damp_percent=0.1,
        static_groups=False,
    )
    
    print("\n[1/4] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    
    print("\n[2/4] 加载模型 (这需要几分钟)...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"\n模型加载完成，显存使用:")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {mem:.2f} GB")
    
    # 准备校准数据
    print("\n[3/4] 准备校准数据...")
    calibration_data = [
        "Please transcribe the following audio content accurately.",
        "What is the main topic discussed in this audio recording?",
        "Translate the speech from English to Chinese.",
        "Summarize the key points from this audio file.",
        "The speaker is discussing artificial intelligence and machine learning applications.",
        "Can you identify the emotions expressed in this audio clip?",
        "What language is being spoken in this recording?",
        "Please provide a detailed analysis of the audio content.",
        "这段音频的主要内容是什么？",
        "请将这段语音翻译成英文。",
        "音频中说话人的情绪是怎样的？",
        "请总结这段录音的要点。",
    ]
    
    # 对校准数据进行编码
    calibration_dataset = []
    for text in calibration_data:
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        calibration_dataset.append(encoded.input_ids)
    
    print(f"校准数据: {len(calibration_dataset)} 条")
    
    print("\n[4/4] 开始量化 (这可能需要 30-60 分钟)...")
    print("请耐心等待...")
    
    try:
        model.quantize(calibration_dataset, batch_size=1)
        
        print("\n量化完成！保存模型...")
        os.makedirs(output_path, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)
        
        # 复制必要的配置文件
        src_path = Path(model_path)
        dst_path = Path(output_path)
        
        for f in ["config.json", "configuration_step_audio_2.py", "modeling_step_audio_2.py", 
                  "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "added_tokens.json"]:
            src_file = src_path / f
            if src_file.exists():
                import shutil
                shutil.copy(src_file, dst_path / f)
                print(f"  复制: {f}")
        
        print(f"\n✅ 量化完成！")
        print(f"量化模型保存在: {output_path}")
        
        # 显示文件大小对比
        original_size = sum(f.stat().st_size for f in src_path.glob("*.safetensors")) / 1024**3
        quantized_size = sum(f.stat().st_size for f in dst_path.glob("*.safetensors")) / 1024**3
        
        print(f"\n文件大小对比:")
        print(f"  原始模型: {original_size:.1f} GB")
        print(f"  量化模型: {quantized_size:.1f} GB")
        print(f"  压缩比: {original_size/quantized_size:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quantize_with_bitsandbytes():
    """使用 bitsandbytes 进行动态量化 (备选方案)"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_path = find_model_path()
    output_path = "/workspace/Step-Audio-R1.1-BNB-INT4"
    
    print("=" * 60)
    print("Step-Audio-R1.1 BitsAndBytes INT4 量化")
    print("=" * 60)
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("\n加载并量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"\n显存使用:")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {mem:.2f} GB")
    
    # 保存配置
    os.makedirs(output_path, exist_ok=True)
    
    config = {
        "quantization_method": "bitsandbytes",
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
        "original_model": model_path,
    }
    
    with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ BitsAndBytes 量化配置已保存到: {output_path}")
    print("注意: BitsAndBytes 是动态量化，每次加载时进行量化")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gptq", "bnb"], default="gptq")
    args = parser.parse_args()
    
    if args.method == "gptq":
        success = quantize_with_gptq()
    else:
        success = quantize_with_bitsandbytes()
    
    sys.exit(0 if success else 1)
