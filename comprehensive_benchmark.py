#!/usr/bin/env python3
"""
Step-Audio-R1.1 全面基准测试
测试所有音频长度 × 所有处理模式，生成详细报告
"""
import os
import sys
import json
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:9100"

# 测试音频（按时长排序）
AUDIO_FILES = [
    ("5min", "test_audio/elon_5min.wav", 300),
    ("10min", "test_audio/elon_10min.wav", 600),
    ("30min", "test_audio/elon_30min.wav", 1800),
    ("60min", "test_audio/elon_60min.wav", 3600),
    ("85min", "test_audio/elon_85min.wav", 5100),
]

# 172分钟需要分段处理，单独测试
LONG_AUDIO = ("172min", "test_audio/elon_172min.wav", 10330)

# 所有处理模式
MODES = ["asr", "s2t", "translate", "summarize", "understand"]

# 存储所有结果
ALL_RESULTS = {}

def process_single_audio(audio_path, mode, extra_params=None):
    """处理单个音频文件"""
    start = time.time()
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"mode": mode}
            if mode == "translate":
                data["target_lang"] = "Chinese"
            elif mode == "understand":
                data["question"] = "Please analyze the main topics, key arguments, and important insights discussed in this audio. Provide a comprehensive summary."
            if extra_params:
                data.update(extra_params)
            
            resp = requests.post(f"{API_URL}/api/process", files=files, data=data, timeout=1800)
        
        elapsed = time.time() - start
        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("answer", "")
            thinking = result.get("thinking", "")
            return {
                "success": True,
                "elapsed": elapsed,
                "answer": answer,
                "thinking": thinking,
                "answer_len": len(answer),
                "thinking_len": len(thinking),
            }
        else:
            return {"success": False, "elapsed": elapsed, "error": resp.text[:200]}
    except Exception as e:
        return {"success": False, "elapsed": time.time() - start, "error": str(e)[:200]}

def test_standard_audio(name, audio_path, duration_sec, mode):
    """测试标准长度音频（<=85分钟）"""
    print(f"    处理中: {name} × {mode}...", end=" ", flush=True)
    result = process_single_audio(audio_path, mode)
    
    if result["success"]:
        print(f"✅ {result['elapsed']:.1f}s, 答案:{result['answer_len']}字符")
    else:
        print(f"❌ {result['error'][:50]}")
    
    return result

def test_long_audio_segmented(audio_path, duration_sec, mode):
    """测试超长音频（分段并行处理）"""
    import subprocess
    import tempfile
    import shutil
    
    # 分段参数
    SEGMENT_DURATION = 3600  # 60分钟/段
    OVERLAP = 30  # 30秒重叠
    
    # 计算分段
    segments = []
    start = 0
    seg_idx = 0
    while start < duration_sec:
        end = min(start + SEGMENT_DURATION, duration_sec)
        segments.append((seg_idx, start, end))
        start = end - OVERLAP if end < duration_sec else end
        seg_idx += 1
    
    print(f"    分成 {len(segments)} 段并行处理...")
    
    # 创建临时目录存放分段
    temp_dir = tempfile.mkdtemp(prefix="audio_seg_")
    seg_files = []
    
    try:
        # 切分音频
        for idx, seg_start, seg_end in segments:
            seg_file = os.path.join(temp_dir, f"seg_{idx}.wav")
            cmd = ["ffmpeg", "-y", "-i", audio_path, "-ss", str(seg_start), 
                   "-t", str(seg_end - seg_start), "-c", "copy", seg_file]
            subprocess.run(cmd, capture_output=True)
            seg_files.append((idx, seg_file, seg_start, seg_end))
        
        # 并行处理
        start_time = time.time()
        seg_results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for idx, seg_file, seg_start, seg_end in seg_files:
                future = executor.submit(process_single_audio, seg_file, mode)
                futures[future] = (idx, seg_start, seg_end)
            
            for future in as_completed(futures):
                idx, seg_start, seg_end = futures[future]
                result = future.result()
                seg_results.append((idx, seg_start, seg_end, result))
                status = "✅" if result["success"] else "❌"
                print(f"      段{idx+1}: {status} {result['elapsed']:.1f}s")
        
        total_elapsed = time.time() - start_time
        
        # 按顺序排列结果
        seg_results.sort(key=lambda x: x[0])
        
        # 合并答案
        merged_answer = []
        for idx, seg_start, seg_end, result in seg_results:
            if result["success"] and result["answer"]:
                time_marker = f"[{int(seg_start//60)}:{int(seg_start%60):02d} - {int(seg_end//60)}:{int(seg_end%60):02d}]"
                merged_answer.append(f"\n{time_marker}\n{result['answer']}")
        
        combined_answer = "\n".join(merged_answer)
        
        return {
            "success": all(r[3]["success"] for r in seg_results),
            "elapsed": total_elapsed,
            "answer": combined_answer,
            "answer_len": len(combined_answer),
            "segments": len(segments),
            "segment_results": [(r[0], r[3]["elapsed"], r[3]["success"], r[3].get("answer_len", 0)) for r in seg_results]
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def run_full_benchmark():
    """运行完整基准测试"""
    print("=" * 70)
    print("🚀 Step-Audio-R1.1 全面基准测试")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {"standard": {}, "long_audio": {}, "meta": {}}
    results["meta"]["start_time"] = datetime.now().isoformat()
    
    # 第一部分：标准音频测试（5-85分钟）
    print("\n📊 第一部分: 标准音频测试 (5-85分钟)")
    print("-" * 70)
    
    for name, audio_path, duration in AUDIO_FILES:
        print(f"\n🎵 测试 {name} ({duration//60}分钟):")
        results["standard"][name] = {}
        
        for mode in MODES:
            result = test_standard_audio(name, audio_path, duration, mode)
            results["standard"][name][mode] = result
            time.sleep(1)  # 短暂间隔
    
    # 第二部分：超长音频分段测试（172分钟）
    print("\n" + "=" * 70)
    print("📊 第二部分: 超长音频分段测试 (172分钟)")
    print("-" * 70)
    
    name, audio_path, duration = LONG_AUDIO
    results["long_audio"][name] = {}
    
    for mode in MODES:
        print(f"\n🎵 测试 {name} × {mode} (分段并行):")
        result = test_long_audio_segmented(audio_path, duration, mode)
        results["long_audio"][name][mode] = result
        print(f"    总耗时: {result['elapsed']:.1f}s, 合并答案: {result['answer_len']}字符")
        time.sleep(2)
    
    results["meta"]["end_time"] = datetime.now().isoformat()
    
    # 保存原始结果
    with open("test_audio/benchmark_raw_results.json", "w", encoding="utf-8") as f:
        # 简化保存（不保存完整答案文本，太大）
        save_results = {"standard": {}, "long_audio": {}, "meta": results["meta"]}
        for name, modes in results["standard"].items():
            save_results["standard"][name] = {}
            for mode, r in modes.items():
                save_results["standard"][name][mode] = {
                    "success": r.get("success"),
                    "elapsed": r.get("elapsed"),
                    "answer_len": r.get("answer_len", 0),
                    "thinking_len": r.get("thinking_len", 0),
                }
        for name, modes in results["long_audio"].items():
            save_results["long_audio"][name] = {}
            for mode, r in modes.items():
                save_results["long_audio"][name][mode] = {
                    "success": r.get("success"),
                    "elapsed": r.get("elapsed"),
                    "answer_len": r.get("answer_len", 0),
                    "segments": r.get("segments"),
                    "segment_results": r.get("segment_results"),
                }
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    return results

def analyze_and_report(results):
    """分析结果并生成报告"""
    print("\n" + "=" * 70)
    print("📋 基准测试结果分析")
    print("=" * 70)
    
    # 1. 处理时间分析
    print("\n### 1. 处理时间分析 (秒)")
    print("-" * 70)
    header = f"{'音频':<10}" + "".join(f"{m:<12}" for m in MODES)
    print(header)
    print("-" * 70)
    
    for name in ["5min", "10min", "30min", "60min", "85min"]:
        if name in results["standard"]:
            row = f"{name:<10}"
            for mode in MODES:
                r = results["standard"][name].get(mode, {})
                elapsed = r.get("elapsed", 0)
                row += f"{elapsed:<12.1f}"
            print(row)
    
    # 172分钟
    if "172min" in results["long_audio"]:
        row = f"{'172min':<10}"
        for mode in MODES:
            r = results["long_audio"]["172min"].get(mode, {})
            elapsed = r.get("elapsed", 0)
            row += f"{elapsed:<12.1f}"
        print(row + " (分段)")
    
    # 2. 输出长度分析
    print("\n### 2. 输出长度分析 (字符)")
    print("-" * 70)
    print(header)
    print("-" * 70)
    
    for name in ["5min", "10min", "30min", "60min", "85min"]:
        if name in results["standard"]:
            row = f"{name:<10}"
            for mode in MODES:
                r = results["standard"][name].get(mode, {})
                ans_len = r.get("answer_len", 0)
                row += f"{ans_len:<12}"
            print(row)
    
    if "172min" in results["long_audio"]:
        row = f"{'172min':<10}"
        for mode in MODES:
            r = results["long_audio"]["172min"].get(mode, {})
            ans_len = r.get("answer_len", 0)
            row += f"{ans_len:<12}"
        print(row)
    
    # 3. 处理效率分析
    print("\n### 3. 处理效率 (秒/分钟音频)")
    print("-" * 70)
    
    durations = {"5min": 5, "10min": 10, "30min": 30, "60min": 60, "85min": 85, "172min": 172}
    
    for mode in MODES:
        print(f"\n{mode.upper()}:")
        for name in ["5min", "10min", "30min", "60min", "85min"]:
            if name in results["standard"]:
                r = results["standard"][name].get(mode, {})
                elapsed = r.get("elapsed", 0)
                dur = durations[name]
                efficiency = elapsed / dur if dur > 0 else 0
                print(f"  {name}: {efficiency:.2f} 秒/分钟")
        
        if "172min" in results["long_audio"]:
            r = results["long_audio"]["172min"].get(mode, {})
            elapsed = r.get("elapsed", 0)
            efficiency = elapsed / 172
            print(f"  172min: {efficiency:.2f} 秒/分钟 (分段并行)")

def save_sample_outputs(results):
    """保存样本输出用于质量分析"""
    output_dir = "test_audio/benchmark_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个模式的样本输出
    for name, modes in results["standard"].items():
        for mode, r in modes.items():
            if r.get("success") and r.get("answer"):
                filename = f"{output_dir}/{name}_{mode}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# {name} - {mode}\n")
                    f.write(f"# 处理时间: {r['elapsed']:.1f}s\n")
                    f.write(f"# 答案长度: {r['answer_len']} 字符\n\n")
                    f.write(r["answer"])
    
    # 保存172分钟合并结果
    if "172min" in results["long_audio"]:
        for mode, r in results["long_audio"]["172min"].items():
            if r.get("success") and r.get("answer"):
                filename = f"{output_dir}/172min_{mode}_merged.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# 172min - {mode} (分段合并)\n")
                    f.write(f"# 总处理时间: {r['elapsed']:.1f}s\n")
                    f.write(f"# 分段数: {r['segments']}\n")
                    f.write(f"# 合并答案长度: {r['answer_len']} 字符\n\n")
                    f.write(r["answer"])
    
    print(f"\n📁 样本输出已保存到: {output_dir}/")

if __name__ == "__main__":
    results = run_full_benchmark()
    analyze_and_report(results)
    save_sample_outputs(results)
    
    print("\n" + "=" * 70)
    print("✅ 基准测试完成!")
    print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
