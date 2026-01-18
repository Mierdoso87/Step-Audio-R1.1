#!/usr/bin/env python3
"""Step-Audio-R1.1 全面 API 测试 - Elon Musk AGI 音频 (172分钟)"""
import os, time, json, requests

API = "http://localhost:9100"
AUDIO_DIR = "/home/neo/upload/Step-Audio-R1/test_audio"
RESULTS = []

def test_api(audio, mode, desc="", **kwargs):
    """测试单个 API 调用，返回完整结果"""
    path = os.path.join(AUDIO_DIR, audio)
    size_mb = os.path.getsize(path) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"📊 测试: {audio} | 模式: {mode} | 大小: {size_mb:.1f}MB")
    if desc:
        print(f"📝 描述: {desc}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        with open(path, 'rb') as f:
            data = {'mode': mode, **kwargs}
            resp = requests.post(f"{API}/api/process", files={'audio': f}, data=data, timeout=1200)
        elapsed = time.time() - start
        result = resp.json()
        
        status = result.get('status', 'error')
        answer = result.get('answer', '')
        thinking = result.get('thinking', '')
        
        print(f"✅ 状态: {status} | ⏱️ 耗时: {elapsed:.2f}s")
        print(f"💭 思考长度: {len(thinking)} 字符 | 📄 回答长度: {len(answer)} 字符")
        
        # 显示完整回答
        print(f"\n{'─'*70}")
        print("📖 完整回答:")
        print(f"{'─'*70}")
        print(answer if answer else "(无回答)")
        
        if thinking:
            print(f"\n{'─'*70}")
            print("🧠 思考过程 (前500字):")
            print(f"{'─'*70}")
            print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
        
        RESULTS.append({
            'audio': audio, 'mode': mode, 'desc': desc, 'status': status,
            'elapsed': round(elapsed, 2), 'answer_len': len(answer),
            'thinking_len': len(thinking), 'size_mb': round(size_mb, 1),
            'answer': answer, 'thinking': thinking[:1000]
        })
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ 错误: {e} | ⏱️ 耗时: {elapsed:.2f}s")
        RESULTS.append({
            'audio': audio, 'mode': mode, 'desc': desc, 'status': 'error',
            'elapsed': round(elapsed, 2), 'error': str(e), 'size_mb': round(size_mb, 1)
        })
        return None

def test_audio_info(audio):
    """测试音频信息 API"""
    path = os.path.join(AUDIO_DIR, audio)
    with open(path, 'rb') as f:
        resp = requests.post(f"{API}/api/audio/info", files={'audio': f})
    info = resp.json()
    duration_min = info.get('duration', 0) / 60
    print(f"  {audio}: {duration_min:.1f}分钟 | {info.get('sample_rate')}Hz | {info.get('channels')}ch")
    return info

# ============================================================
print("\n" + "="*70)
print("🎵 Step-Audio-R1.1 全面 API 测试")
print("📁 音频: Elon Musk AGI Timeline (172分钟)")
print("⚙️ 配置: max_model_len=131072, max_num_seqs=1")
print("="*70)

# 第一部分: 音频信息
print("\n" + "="*70)
print("📋 第一部分: 音频信息 API 测试")
print("="*70)
audios = ['elon_5min.wav', 'elon_10min.wav', 'elon_30min.wav', 
          'elon_60min.wav', 'elon_90min.wav', 'elon_120min.wav', 'elon_172min.wav']
for audio in audios:
    test_audio_info(audio)

# 第二部分: 不同时长 s2t 测试
print("\n" + "="*70)
print("📋 第二部分: 不同时长音频 s2t 模式测试")
print("="*70)
for audio in audios:
    test_api(audio, 's2t', f"语音转文字+智能摘要")

# 第三部分: 所有处理模式测试 (用 10min 音频)
print("\n" + "="*70)
print("📋 第三部分: 所有处理模式测试 (elon_10min.wav)")
print("="*70)

test_api('elon_10min.wav', 'asr', '纯语音识别 - 逐字转录')
test_api('elon_10min.wav', 'understand', '内容理解', 
         question='这段音频讨论了哪些关于AGI的观点？Elon Musk对AGI时间线有什么预测？')
test_api('elon_10min.wav', 'translate', '翻译为中文', target_lang='Chinese')
test_api('elon_10min.wav', 'summarize', '内容摘要')

# 第四部分: 长音频特殊测试
print("\n" + "="*70)
print("📋 第四部分: 长音频深度理解测试")
print("="*70)

test_api('elon_60min.wav', 'understand', '60分钟音频深度理解',
         question='请详细分析这段音频中关于AI发展的所有观点，包括技术预测、风险评估和社会影响。')

test_api('elon_90min.wav', 'summarize', '90分钟音频摘要')

# 输出汇总
print("\n" + "="*70)
print("📊 测试结果汇总表")
print("="*70)
print(f"{'音频':<18} {'模式':<12} {'状态':<8} {'耗时(s)':<10} {'回答长度':<10} {'思考长度':<10}")
print("-"*70)
for r in RESULTS:
    print(f"{r['audio']:<18} {r['mode']:<12} {r['status']:<8} {r['elapsed']:<10} {r.get('answer_len', 'N/A'):<10} {r.get('thinking_len', 'N/A'):<10}")

# 保存结果
output_file = os.path.join(AUDIO_DIR, 'elon_test_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print(f"\n💾 详细结果已保存到: {output_file}")
