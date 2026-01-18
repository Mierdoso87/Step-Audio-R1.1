#!/usr/bin/env python3
"""
Step-Audio-R1.1 智能长音频处理器
支持处理任意时长音频，自动分段并智能合并结果
"""
import os, sys, json, argparse, subprocess, tempfile, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = os.getenv("STEP_AUDIO_API", "http://localhost:9100")
MAX_SEGMENT_DURATION = 3600  # 60分钟 (安全上限，模型限制85分钟)
OVERLAP_DURATION = 30  # 段之间重叠30秒

def get_max_concurrency():
    """从 API 获取服务配置的最大并发数"""
    try:
        resp = requests.get(f"{API_URL}/api/status", timeout=5)
        # 当前配置 max_num_seqs=4，支持 4 个并发的 65536 token 请求
        return 4
    except:
        return 1

def get_audio_duration(path):
    """获取音频时长"""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])

def split_audio(path, output_dir, segment_duration=MAX_SEGMENT_DURATION):
    """智能分割音频"""
    duration = get_audio_duration(path)
    segments = []
    start = 0
    idx = 0
    
    while start < duration:
        end = min(start + segment_duration, duration)
        out_path = os.path.join(output_dir, f"seg_{idx:03d}.wav")
        
        cmd = ["ffmpeg", "-y", "-i", path, "-ss", str(start), "-t", str(end - start),
               "-ar", "16000", "-ac", "1", out_path]
        subprocess.run(cmd, capture_output=True)
        
        segments.append({"index": idx, "path": out_path, "start": start, "end": end})
        start = end - OVERLAP_DURATION if end < duration else duration
        idx += 1
    
    return segments

def process_segment(seg, mode, **kwargs):
    """处理单个音频段"""
    try:
        with open(seg["path"], "rb") as f:
            resp = requests.post(f"{API_URL}/api/process", 
                               files={"audio": f}, 
                               data={"mode": mode, **kwargs}, 
                               timeout=600)
        result = resp.json()
        return {
            "index": seg["index"], "start": seg["start"], "end": seg["end"],
            "status": result.get("status", "error"),
            "answer": result.get("answer", ""),
            "thinking": result.get("thinking", ""),
            "elapsed": result.get("elapsed_time", 0)
        }
    except Exception as e:
        return {"index": seg["index"], "start": seg["start"], "end": seg["end"],
                "status": "error", "error": str(e)}

def format_time(seconds):
    """格式化时间"""
    h, m, s = int(seconds//3600), int((seconds%3600)//60), int(seconds%60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def process_long_audio(audio_path, mode="s2t", output_file=None, parallel=None, **kwargs):
    """处理长音频，parallel=None 时自动检测最佳并行度"""
    if not os.path.exists(audio_path):
        return {"error": f"文件不存在: {audio_path}"}
    
    duration = get_audio_duration(audio_path)
    print(f"📁 音频: {audio_path}")
    print(f"⏱️  时长: {format_time(duration)} ({duration:.0f}秒)")
    
    # 短音频直接处理
    if duration <= MAX_SEGMENT_DURATION:
        print("✅ 音频在限制内，直接处理...")
        with open(audio_path, "rb") as f:
            resp = requests.post(f"{API_URL}/api/process", 
                               files={"audio": f}, 
                               data={"mode": mode, **kwargs}, 
                               timeout=600)
        return resp.json()
    
    # 自动检测并行度
    if parallel is None:
        parallel = get_max_concurrency()
        print(f"🔧 自动检测并行度: {parallel}")
    
    # 长音频分段处理
    num_segments = int(duration / MAX_SEGMENT_DURATION) + 1
    print(f"📊 音频超长，分成 {num_segments} 段处理 (每段最长 {MAX_SEGMENT_DURATION//60} 分钟)")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("✂️  分割音频...")
        segments = split_audio(audio_path, temp_dir)
        print(f"   完成: {len(segments)} 段")
        
        results = []
        total_start = time.time()
        
        if parallel > 1:
            print(f"🚀 并行处理 (workers={parallel})...")
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(process_segment, seg, mode, **kwargs): seg for seg in segments}
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    status = "✅" if r["status"] == "success" else "❌"
                    print(f"   {status} 段 {r['index']+1}/{len(segments)} [{format_time(r['start'])}-{format_time(r['end'])}] {r.get('elapsed',0):.1f}s")
        else:
            print("🔄 顺序处理...")
            for seg in segments:
                print(f"   处理段 {seg['index']+1}/{len(segments)} [{format_time(seg['start'])}-{format_time(seg['end'])}]...", end=" ", flush=True)
                r = process_segment(seg, mode, **kwargs)
                results.append(r)
                print(f"{'✅' if r['status']=='success' else '❌'} {r.get('elapsed',0):.1f}s")
        
        total_elapsed = time.time() - total_start
        results.sort(key=lambda x: x["index"])
        
        # 合并结果
        combined = {
            "status": "success",
            "total_duration": duration,
            "total_elapsed": round(total_elapsed, 2),
            "segments": len(results),
            "mode": mode,
            "results": []
        }
        
        full_content = []
        for r in results:
            combined["results"].append({
                "time_range": f"{format_time(r['start'])} - {format_time(r['end'])}",
                "content": r.get("answer", ""),
                "status": r["status"]
            })
            if r.get("answer"):
                full_content.append(f"## [{format_time(r['start'])} - {format_time(r['end'])}]\n\n{r['answer']}")
        
        combined["full_content"] = "\n\n---\n\n".join(full_content)
        
        print(f"\n✅ 处理完成! 总耗时: {total_elapsed:.1f}s")
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存: {output_file}")
            
            # 同时保存 Markdown 版本
            md_file = output_file.rsplit(".", 1)[0] + ".md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# 长音频处理结果\n\n")
                f.write(f"- 原始时长: {format_time(duration)}\n")
                f.write(f"- 处理模式: {mode}\n")
                f.write(f"- 分段数: {len(results)}\n")
                f.write(f"- 总耗时: {total_elapsed:.1f}s\n\n---\n\n")
                f.write(combined["full_content"])
            print(f"📄 Markdown: {md_file}")
        
        return combined

def main():
    parser = argparse.ArgumentParser(description="Step-Audio-R1.1 智能长音频处理器")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("-m", "--mode", default="s2t", choices=["s2t", "asr", "understand", "translate", "summarize"])
    parser.add_argument("-o", "--output", help="输出文件路径 (JSON)")
    parser.add_argument("-p", "--parallel", type=int, default=None, help="并行处理数 (默认自动检测)")
    parser.add_argument("--max-segment", type=int, default=3600, help="每段最大时长(秒)")
    parser.add_argument("--target-lang", default="Chinese", help="翻译目标语言")
    parser.add_argument("--question", help="理解模式的问题")
    
    args = parser.parse_args()
    
    global MAX_SEGMENT_DURATION
    MAX_SEGMENT_DURATION = args.max_segment
    
    kwargs = {}
    if args.target_lang: kwargs["target_lang"] = args.target_lang
    if args.question: kwargs["question"] = args.question
    
    result = process_long_audio(args.audio, mode=args.mode, output_file=args.output, 
                                parallel=args.parallel, **kwargs)
    
    if not args.output and "full_content" in result:
        print("\n" + "="*60)
        print(result["full_content"][:3000])
        if len(result["full_content"]) > 3000:
            print(f"\n... (共 {len(result['full_content'])} 字符)")

if __name__ == "__main__":
    main()
