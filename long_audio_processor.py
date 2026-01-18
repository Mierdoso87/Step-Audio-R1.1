#!/usr/bin/env python3
"""
长音频分段处理器
支持处理 1-2 小时甚至更长的音频文件
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
API_URL = os.getenv("STEP_AUDIO_API", "http://localhost:9100")
SEGMENT_DURATION = 300  # 5分钟一段（安全值）
OVERLAP_DURATION = 10   # 段之间重叠10秒，避免切断句子

def get_audio_duration(audio_path: str) -> float:
    """获取音频时长"""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "json", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def split_audio(audio_path: str, output_dir: str, segment_duration: int = SEGMENT_DURATION) -> list:
    """将音频分割成多个片段"""
    duration = get_audio_duration(audio_path)
    segments = []
    
    start = 0
    segment_idx = 0
    
    while start < duration:
        end = min(start + segment_duration, duration)
        output_path = os.path.join(output_dir, f"segment_{segment_idx:04d}.wav")
        
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(end - start),
            "-ar", "16000", "-ac", "1",  # 统一采样率和声道
            output_path
        ]
        subprocess.run(cmd, capture_output=True)
        
        segments.append({
            "index": segment_idx,
            "path": output_path,
            "start_time": start,
            "end_time": end,
            "duration": end - start
        })
        
        # 下一段开始位置（有重叠）
        start = end - OVERLAP_DURATION if end < duration else duration
        segment_idx += 1
    
    return segments

def process_segment(segment: dict, mode: str = "s2t", **kwargs) -> dict:
    """处理单个音频片段"""
    try:
        with open(segment["path"], "rb") as f:
            files = {"audio": f}
            data = {"mode": mode, **kwargs}
            response = requests.post(f"{API_URL}/api/process", files=files, data=data, timeout=300)
            result = response.json()
        
        return {
            "index": segment["index"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "status": result.get("status", "error"),
            "answer": result.get("answer", ""),
            "thinking": result.get("thinking", ""),
            "error": result.get("error", "")
        }
    except Exception as e:
        return {
            "index": segment["index"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "status": "error",
            "error": str(e)
        }

def format_timestamp(seconds: float) -> str:
    """格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def process_long_audio(
    audio_path: str,
    mode: str = "s2t",
    output_file: str = None,
    parallel: int = 1,
    **kwargs
) -> dict:
    """
    处理长音频文件
    
    Args:
        audio_path: 音频文件路径
        mode: 处理模式 (s2t, asr, understand, translate, summarize)
        output_file: 输出文件路径
        parallel: 并行处理数量
        **kwargs: 其他参数传递给 API
    
    Returns:
        处理结果字典
    """
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}
    
    # 获取音频信息
    duration = get_audio_duration(audio_path)
    print(f"📁 音频文件: {audio_path}")
    print(f"⏱️  总时长: {format_timestamp(duration)} ({duration:.1f}秒)")
    
    # 如果音频较短，直接处理
    if duration <= SEGMENT_DURATION:
        print("✅ 音频较短，直接处理...")
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            data = {"mode": mode, **kwargs}
            response = requests.post(f"{API_URL}/api/process", files=files, data=data, timeout=300)
            return response.json()
    
    # 分段处理
    print(f"📊 音频较长，将分成 {int(duration / SEGMENT_DURATION) + 1} 段处理...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 分割音频
        print("✂️  正在分割音频...")
        segments = split_audio(audio_path, temp_dir)
        print(f"   分割完成: {len(segments)} 段")
        
        # 处理每个片段
        results = []
        
        if parallel > 1:
            print(f"🚀 并行处理 (workers={parallel})...")
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(process_segment, seg, mode, **kwargs): seg 
                    for seg in segments
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f"   ✅ 段 {result['index']+1}/{len(segments)} 完成 [{format_timestamp(result['start_time'])} - {format_timestamp(result['end_time'])}]")
        else:
            print("🔄 顺序处理...")
            for i, segment in enumerate(segments):
                print(f"   处理段 {i+1}/{len(segments)} [{format_timestamp(segment['start_time'])} - {format_timestamp(segment['end_time'])}]...", end=" ", flush=True)
                result = process_segment(segment, mode, **kwargs)
                results.append(result)
                print("✅" if result["status"] == "success" else f"❌ {result.get('error', '')}")
        
        # 按顺序排列结果
        results.sort(key=lambda x: x["index"])
        
        # 合并结果
        combined = {
            "status": "success",
            "total_duration": duration,
            "segments": len(results),
            "mode": mode,
            "results": []
        }
        
        full_transcript = []
        for r in results:
            combined["results"].append({
                "time_range": f"{format_timestamp(r['start_time'])} - {format_timestamp(r['end_time'])}",
                "content": r.get("answer", "")
            })
            if r.get("answer"):
                full_transcript.append(f"[{format_timestamp(r['start_time'])}] {r['answer']}")
        
        combined["full_transcript"] = "\n\n".join(full_transcript)
        
        # 保存结果
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到: {output_file}")
        
        return combined

def main():
    parser = argparse.ArgumentParser(description="长音频分段处理器")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("-m", "--mode", default="s2t", 
                        choices=["s2t", "asr", "understand", "translate", "summarize"],
                        help="处理模式")
    parser.add_argument("-o", "--output", help="输出文件路径 (JSON)")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="并行处理数量")
    parser.add_argument("--segment-duration", type=int, default=300, help="每段时长(秒)")
    parser.add_argument("--target-lang", default="Chinese", help="翻译目标语言")
    parser.add_argument("--question", help="理解模式的问题")
    
    args = parser.parse_args()
    
    global SEGMENT_DURATION
    SEGMENT_DURATION = args.segment_duration
    
    kwargs = {}
    if args.target_lang:
        kwargs["target_lang"] = args.target_lang
    if args.question:
        kwargs["question"] = args.question
    
    result = process_long_audio(
        args.audio,
        mode=args.mode,
        output_file=args.output,
        parallel=args.parallel,
        **kwargs
    )
    
    if not args.output:
        print("\n" + "="*60)
        print("处理结果:")
        print("="*60)
        if "full_transcript" in result:
            print(result["full_transcript"])
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
