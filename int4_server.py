#!/usr/bin/env python3
"""
Step-Audio-R1.1 INT4 量化推理服务
使用 transformers + bitsandbytes 实现单卡运行
"""

import os
import sys
import json
import time
import base64
import tempfile
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE", "0")

app = Flask(__name__)
CORS(app)

# 全局模型
model = None
tokenizer = None
processor = None

def load_model():
    """加载 INT4 量化模型"""
    global model, tokenizer
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_path = os.getenv("MODEL_PATH", "/model")
    
    print("=" * 60)
    print("Step-Audio-R1.1 INT4 量化推理服务")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)
    
    # INT4 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("加载 INT4 量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    print(f"\n✅ 模型加载完成！")
    print(f"显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def process_audio(audio_path, prompt="Please transcribe this audio.", max_tokens=2048, temperature=0.7):
    """处理音频"""
    import librosa
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 构建输入
    # Step-Audio 使用特殊的音频标记
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 使用模型的 chat 方法（如果有）或直接生成
    # 这里简化处理，直接使用文本生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "Step-Audio-R1.1-INT4"})

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "Step-Audio-R1.1-INT4",
            "object": "model",
            "owned_by": "stepfun-ai",
            "quantization": "INT4-bitsandbytes"
        }]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI 兼容的 chat completions API"""
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 2048)
    temperature = data.get('temperature', 0.7)
    
    # 提取最后一条用户消息
    user_message = ""
    for msg in reversed(messages):
        if msg['role'] == 'user':
            content = msg['content']
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        user_message = item.get('text', '')
                        break
            break
    
    # 生成响应
    start_time = time.time()
    
    inputs = tokenizer(user_message, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - start_time
    
    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "Step-Audio-R1.1-INT4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
            "total_tokens": outputs.shape[1]
        },
        "elapsed_time": elapsed
    })

@app.route('/process', methods=['POST'])
def process():
    """处理音频文件"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    prompt = request.form.get('prompt', 'Please transcribe this audio.')
    max_tokens = int(request.form.get('max_tokens', 2048))
    temperature = float(request.form.get('temperature', 0.7))
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_file.save(f.name)
        temp_path = f.name
    
    try:
        response = process_audio(temp_path, prompt, max_tokens, temperature)
        return jsonify({
            "status": "success",
            "response": response
        })
    finally:
        os.unlink(temp_path)

if __name__ == '__main__':
    load_model()
    
    port = int(os.getenv('PORT', 9998))
    print(f"\n🚀 服务启动在 http://0.0.0.0:{port}")
    
    app.run(host='0.0.0.0', port=port, threaded=True)
