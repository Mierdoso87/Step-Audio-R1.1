#!/usr/bin/env python3
"""Step-Audio-R1.1 全面 API 测试"""
import os, time, json, requests

API = "http://localhost:9100"
AUDIO_DIR = "/home/neo/upload/Step-Audio-R1/test_audio"
RESULTS = []

def test_api(audio, mode, **kwargs):
    """测试单个 API 调用"""
    path = os.path.join(AUDIO_DIR, audio)
    size_mb = os.path.getsize(path) / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"测试: {audio} | 模式: {mode} | 大小: {size_mb:.1f}MB")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        with open(path, 'rb') as f:
            data = {'mode': mode, **kwargs}
            resp = requests.post(f"{API}/api/process", files={'audio': f}, data=data, timeout=600)
        elapsed = time.time() - start
        result = resp.json()
        
        status = result.get('status', 'error')
        answer_len = len(result.get('answer', ''))
        thinking_len = len(result.get('thinking', ''))
        
        print(f"状态: {status} | 耗时: {elapsed:.2f}s")
        print(f"思考长度: {thinking_len} | 回答长度: {answer_len}")
        if answer_len > 0:
            print(f"回答预览: {result['answer'][:200]}...")
        
        RESULTS.append({
            'audio': audio, 'mode': mode, 'status': status,
            'elapsed': round(elapsed, 2), 'answer_len': answer_len,
            'thinking_len': thinking_len, 'size_mb': round(size_mb, 1)
        })
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"错误: {e} | 耗时: {elapsed:.2f}s")
        RESULTS.append({
            'audio': audio, 'mode': mode, 'status': 'error',
            'elapsed': round(elapsed, 2), 'error': str(e), 'size_mb': round(size_mb, 1)
        })
        return None

def test_audio_info(audio):
    """测试音频信息 API"""
    path = os.path.join(AUDIO_DIR, audio)
    print(f"\n--- 音频信息: {audio} ---")
    with open(path, 'rb') as f:
        resp = requests.post(f"{API}/api/audio/info", files={'audio': f})
    info = resp.json()
    print(f"时长: {info.get('duration', 0):.1f}s | 采样率: {info.get('sample_rate')} | 声道: {info.get('channels')}")
    return info

# 测试音频信息 API
print("\n" + "="*60)
print("第一部分: 音频信息 API 测试")
print("="*60)
for audio in ['5min.wav', '10min.wav', '30min.wav', '60min.wav', '85min.wav']:
    test_audio_info(audio)

# 测试不同时长 + s2t 模式
print("\n" + "="*60)
print("第二部分: 不同时长音频 s2t 模式测试")
print("="*60)
for audio in ['5min.wav', '10min.wav', '30min.wav', '60min.wav', '85min.wav']:
    test_api(audio, 's2t')

# 测试所有处理模式 (用 5min 音频)
print("\n" + "="*60)
print("第三部分: 所有处理模式测试 (5min.wav)")
print("="*60)
modes = [
    ('asr', {}),
    ('understand', {'question': '这段音频的主要内容是什么？讲了哪些技术要点？'}),
    ('translate', {'target_lang': 'English'}),
    ('summarize', {}),
]
for mode, kwargs in modes:
    test_api('5min.wav', mode, **kwargs)

# 输出汇总
print("\n" + "="*60)
print("测试结果汇总")
print("="*60)
print(f"{'音频':<15} {'模式':<12} {'状态':<8} {'耗时(s)':<10} {'回答长度':<10}")
print("-"*60)
for r in RESULTS:
    print(f"{r['audio']:<15} {r['mode']:<12} {r['status']:<8} {r['elapsed']:<10} {r.get('answer_len', 'N/A'):<10}")

# 保存结果
with open(os.path.join(AUDIO_DIR, 'test_results.json'), 'w') as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 test_results.json")
