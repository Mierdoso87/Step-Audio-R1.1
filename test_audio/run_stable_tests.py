#!/usr/bin/env python3
"""Step-Audio-R1.1 完整测试 - 稳定配置 (max_model_len=65536)"""
import os, time, json, requests

API = "http://localhost:9100"
AUDIO_DIR = "/home/neo/upload/Step-Audio-R1/test_audio"
RESULTS = []

def test_api(audio, mode, desc="", **kwargs):
    path = os.path.join(AUDIO_DIR, audio)
    size_mb = os.path.getsize(path) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"📊 测试: {audio} | 模式: {mode} | 大小: {size_mb:.1f}MB")
    if desc: print(f"📝 描述: {desc}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        with open(path, 'rb') as f:
            resp = requests.post(f"{API}/api/process", files={'audio': f}, 
                               data={'mode': mode, **kwargs}, timeout=600)
        elapsed = time.time() - start
        result = resp.json()
        
        status = result.get('status', 'error')
        answer = result.get('answer', '')
        thinking = result.get('thinking', '')
        
        print(f"✅ 状态: {status} | ⏱️ 耗时: {elapsed:.2f}s")
        print(f"💭 思考: {len(thinking)}字符 | 📄 回答: {len(answer)}字符")
        print(f"\n{'─'*70}\n📖 回答:\n{'─'*70}\n{answer[:2000]}{'...' if len(answer)>2000 else ''}")
        
        RESULTS.append({
            'audio': audio, 'mode': mode, 'desc': desc, 'status': status,
            'elapsed': round(elapsed, 2), 'answer_len': len(answer),
            'thinking_len': len(thinking), 'size_mb': round(size_mb, 1),
            'answer': answer, 'thinking': thinking[:500]
        })
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ 错误: {e}")
        RESULTS.append({'audio': audio, 'mode': mode, 'status': 'error', 
                       'elapsed': round(elapsed, 2), 'error': str(e)})
        return None

print("\n" + "="*70)
print("🎵 Step-Audio-R1.1 完整测试 (稳定配置)")
print("⚙️ 配置: max_model_len=65536, max_num_seqs=2")
print("="*70)

# 测试所有模式 (10分钟音频)
print("\n📋 所有处理模式测试 (elon_10min.wav)")
test_api('elon_10min.wav', 's2t', '语音转文字+摘要')
test_api('elon_10min.wav', 'asr', '纯语音识别')
test_api('elon_10min.wav', 'understand', '内容理解', 
         question='这段音频讨论了哪些关于AGI的观点？Elon Musk对AI发展有什么看法？')
test_api('elon_10min.wav', 'translate', '翻译为中文', target_lang='Chinese')
test_api('elon_10min.wav', 'summarize', '内容摘要')

# 不同时长测试 (s2t)
print("\n📋 不同时长音频测试")
for audio in ['elon_5min.wav', 'elon_30min.wav', 'elon_60min.wav']:
    test_api(audio, 's2t', f'语音转文字+摘要')

# 汇总
print("\n" + "="*70)
print("📊 测试结果汇总")
print("="*70)
print(f"{'音频':<18} {'模式':<12} {'状态':<8} {'耗时':<8} {'回答长度':<10}")
print("-"*70)
for r in RESULTS:
    print(f"{r['audio']:<18} {r['mode']:<12} {r['status']:<8} {r['elapsed']:<8} {r.get('answer_len','N/A'):<10}")

with open(os.path.join(AUDIO_DIR, 'elon_full_test.json'), 'w') as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print(f"\n💾 结果已保存")
