# Step-Audio-R1.1 Benchmark Report

> 测试日期: 2026-01-18  
> 测试环境: 4× NVIDIA L40S (46GB each)  
> 模型版本: Step-Audio-R1.1 (stepfun-ai)

---

## 1. 执行摘要

本报告记录了 Step-Audio-R1.1 音频语言模型在不同配置下的性能表现，包括显存使用、音频处理时长上限、推理延迟等关键指标。测试结果表明，当前配置可稳定处理最长 60 分钟的音频文件。

### 关键结论

| 指标 | 数值 |
|------|------|
| 最大支持音频时长 | **60 分钟** |
| 单次推理最大耗时 | ~80 秒 |
| GPU 显存利用率 | 82-90% |
| 最大并发请求数 | 4 |

---

## 2. 硬件配置

### 2.1 GPU 规格

| GPU | 型号 | 显存 | 用途 |
|-----|------|------|------|
| GPU 0 | NVIDIA L40S | 46 GB | 模型分片 (TP) |
| GPU 1 | NVIDIA L40S | 46 GB | 模型分片 (TP) |
| GPU 2 | NVIDIA L40S | 46 GB | 模型分片 (TP) |
| GPU 3 | NVIDIA L40S | 46 GB | 模型分片 (TP) |

**总显存**: 184 GB  
**张量并行**: 4-way (必需，模型 63GB 无法单卡加载)

### 2.2 服务器配置

- **操作系统**: Linux
- **容器运行时**: Docker with NVIDIA Container Toolkit
- **vLLM 版本**: stepfun2025/vllm:step-audio-2-v20250909 (定制版)

---

## 3. 模型架构

### 3.1 组件结构

```
┌─────────────────────────────────────────────────────────┐
│                    Step-Audio-R1.1                       │
├─────────────────────────────────────────────────────────┤
│  AudioEncoder (Whisper-like)                            │
│  - 32 层 Transformer                                    │
│  - 参数量: ~1.28 GB                                     │
│  - 帧率: 25 Hz (冻结)                                   │
├─────────────────────────────────────────────────────────┤
│  Audio Adaptor                                          │
│  - Conv + MLP                                           │
│  - 参数量: ~100 MB                                      │
│  - 下采样至 12.5 Hz                                     │
├─────────────────────────────────────────────────────────┤
│  Qwen2.5-32B LLM Decoder                                │
│  - 64 层 Transformer                                    │
│  - hidden_size: 5120                                    │
│  - vocab_size: 158720                                   │
│  - 参数量: ~60 GB                                       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心创新

| 特性 | 描述 |
|------|------|
| **MGRD** | 模态锚定推理蒸馏，推理基于声学特征而非文本转录 |
| **双脑架构** | Formulation Brain (推理) + Articulation Brain (生成)，支持边思考边说话 |
| **测试时计算扩展** | 首个成功解锁音频领域 test-time compute scaling 的模型 |

---

## 4. 服务配置

### 4.1 当前生产配置

```yaml
# docker-compose.yml 关键参数
vllm:
  command:
    --max-model-len 49152        # 上下文长度
    --max-num-seqs 4             # 最大并发数
    --tensor-parallel-size 4     # 张量并行度
    --gpu-memory-utilization 0.85
```

### 4.2 配置演进历史

| 日期 | max_model_len | max_num_seqs | 最大音频时长 | 状态 |
|------|---------------|--------------|-------------|------|
| 初始 | 8,192 | 32 | ~8 分钟 | 稳定 |
| 2026-01-17 | 16,384 | 16 | ~20 分钟 | 稳定 |
| 2026-01-18 | 32,768 | 8 | ~40 分钟 | 稳定 |
| 2026-01-18 | **49,152** | **4** | **~60 分钟** | **当前** |

### 4.3 音频时长计算公式

```
音频编码率 = 12.5 Hz (每秒 12.5 个 token)
系统预留 token ≈ 1500

最大音频秒数 = (max_model_len - 1500) / 12.5

示例:
- max_model_len=49152 → (49152-1500)/12.5 = 3812秒 ≈ 63.5分钟
- max_model_len=65536 → (65536-1500)/12.5 = 5123秒 ≈ 85.4分钟
```

---

## 5. 性能基准测试

### 5.1 测试方法

- **测试音频**: Aeron Deep Dive 技术分享录音 (原始 84.8 分钟)
- **截取版本**: 10 分钟、30 分钟、60 分钟
- **测试模式**: s2t (语音转文字 + 摘要)
- **API 端点**: `POST /api/process`

### 5.2 测试结果

| 音频时长 | 文件大小 | 处理耗时 | 输出长度 | 状态 |
|---------|---------|---------|---------|------|
| 10 分钟 | 4.6 MB | 64.59 秒 | 5,847 字符 | ✅ 成功 |
| 30 分钟 | 14 MB | 79.99 秒 | 2,691 字符 | ✅ 成功 |
| 60 分钟 | 27 MB | 69.16 秒 | 1,401 字符 | ✅ 成功 |

### 5.3 显存使用 (稳态)

| GPU | 已用显存 | 总显存 | 利用率 |
|-----|---------|--------|--------|
| GPU 0 | 41.6 GB | 46 GB | 90.3% |
| GPU 1 | 38.2 GB | 46 GB | 82.9% |
| GPU 2 | 37.7 GB | 46 GB | 81.9% |
| GPU 3 | 38.3 GB | 46 GB | 83.2% |

**观察**: GPU 0 显存使用较高，可能承担了额外的协调开销。

---

## 6. 量化研究

### 6.1 量化收益估算

| 精度 | 模型大小 | 显存需求 | 最小 GPU 配置 |
|------|----------|----------|--------------|
| BF16 (当前) | 63 GB | ~82 GB | 4× L40S |
| INT8 | 32 GB | ~41 GB | 1× L40S |
| INT4 | 16 GB | ~20 GB | 1× RTX 4090 |

### 6.2 量化测试结果

| 方案 | 工具 | 结果 | 备注 |
|------|------|------|------|
| INT4 (transformers) | bitsandbytes | ✅ 成功加载 | 显存 18.38GB，但无法用于 vLLM 推理 |
| INT4 (vLLM) | bitsandbytes | ❌ 失败 | StepAudio2ForCausalLM 缺少 packed_modules_mapping |
| GPTQ | auto-gptq | ❌ 失败 | 不支持 step_audio_2 模型类型 |
| AWQ | autoawq | ❌ 失败 | 同上 |

### 6.3 量化结论

当前 vLLM 定制版本不支持该模型的量化推理。如需单卡部署，需等待：
1. 官方发布量化权重
2. vLLM 添加对 StepAudio2 的量化支持

---

## 7. 功能测试矩阵

### 7.1 处理模式

| 模式 | 描述 | 状态 |
|------|------|------|
| `s2t` | 语音转文字 + 智能摘要 | ✅ |
| `asr` | 纯语音识别 (逐字转录) | ✅ |
| `understand` | 音频内容理解 | ✅ |
| `translate` | 音频翻译 | ✅ |
| `summarize` | 音频摘要 | ✅ |

### 7.2 支持的音频格式

```
mp3, wav, flac, ogg, m4a, aac, webm, wma, amr
```

### 7.3 API 端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/api/process` | POST | 同步处理 | ✅ |
| `/api/task` | POST | 异步任务提交 | ✅ |
| `/api/task/<id>` | GET | 任务状态查询 | ✅ |
| `/api/audio/info` | POST | 音频信息获取 | ✅ |
| `/api/status` | GET | 服务状态 | ✅ |

---

## 8. 部署架构

### 8.1 服务拓扑

```
                    ┌─────────────────┐
                    │   Cloudflare    │
                    │  (DNS + Proxy)  │
                    └────────┬────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   Nginx (107.172.39.47)                  │
│              step-audio.aws.xin:443                      │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (反向代理)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              AWS EC2 (44.193.212.118:9100)               │
│  ┌─────────────────────────────────────────────────┐    │
│  │           step-audio-r1.1-web                    │    │
│  │         (Flask + Gradio UI)                      │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │ HTTP                          │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │          step-audio-r1.1-vllm                    │    │
│  │    (vLLM OpenAI-compatible API :9999)            │    │
│  │         4× L40S Tensor Parallel                  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 8.2 容器配置

| 容器 | 镜像 | 端口 | 健康检查 |
|------|------|------|---------|
| step-audio-r1.1-vllm | stepfun2025/vllm:step-audio-2-v20250909 | 9999 | `/health` |
| step-audio-r1.1-web | 自建 | 9100 | depends_on vllm |

---

## 9. 长音频处理方案

### 9.1 分段处理器

对于超过 60 分钟的音频，提供 `long_audio_processor.py` 分段处理方案：

```python
# 使用示例
python long_audio_processor.py \
  --input /path/to/long_audio.mp3 \
  --mode s2t \
  --segment-duration 1800  # 30分钟分段
```

**特性**:
- 自动分割长音频
- 并行处理各分段
- 智能合并结果
- 无时长限制

### 9.2 分段策略建议

| 原始时长 | 建议分段 | 分段数 |
|---------|---------|--------|
| < 60 分钟 | 不分段 | 1 |
| 60-120 分钟 | 30 分钟 | 2-4 |
| > 120 分钟 | 30 分钟 | N |

---

## 10. 优化建议

### 10.1 短期优化

| 优化项 | 当前值 | 建议值 | 预期收益 |
|--------|--------|--------|---------|
| max_model_len | 49152 | 65536 | 支持 85 分钟音频 (需监控 OOM) |
| gpu_memory_utilization | 0.85 | 0.90 | 更多 KV cache 空间 |

### 10.2 中期优化

1. **等待官方量化支持**: INT8/INT4 量化可将显存需求降至单卡水平
2. **升级 vLLM**: 关注 stepfun-ai/vllm 分支更新
3. **Flash Attention 优化**: 确认是否启用 FA2

### 10.3 长期优化

1. **模型蒸馏**: 等待官方发布更小参数量版本
2. **专用推理芯片**: 考虑 Groq/Cerebras 等专用硬件
3. **多实例负载均衡**: 高并发场景部署多套服务

---

## 11. 已知限制

| 限制 | 描述 | 规避方案 |
|------|------|---------|
| 单次最大音频 | ~60 分钟 | 使用分段处理器 |
| 最大并发 | 4 请求 | 部署多实例 |
| 量化不可用 | vLLM 不支持该模型量化 | 等待官方支持 |
| GPU 绑定 | 必须 4 卡张量并行 | 无法减少 GPU 数量 |

---

## 12. 参考资料

- [Step-Audio-R1 GitHub](https://github.com/stepfun-ai/Step-Audio-R1)
- [Step-Audio-R1 技术报告](https://arxiv.org/pdf/2511.15848)
- [HuggingFace 模型页](https://huggingface.co/stepfun-ai/Step-Audio-R1.1)
- [vLLM 定制分支](https://github.com/stepfun-ai/vllm)

---

## 附录 A: 配置文件

### docker-compose.yml (当前生产配置)

```yaml
services:
  vllm:
    image: stepfun2025/vllm:step-audio-2-v20250909
    container_name: step-audio-r1.1-vllm
    runtime: nvidia
    command: >
      vllm serve /model
      --served-model-name Step-Audio-R1.1
      --port 9999
      --host 0.0.0.0
      --max-model-len 49152
      --max-num-seqs 4
      --tensor-parallel-size 4
      --gpu-memory-utilization 0.85
      --trust-remote-code
      --enable-log-requests
      --interleave-mm-strings

  web:
    container_name: step-audio-r1.1-web
    environment:
      - VLLM_API_URL=http://vllm:9999/v1/chat/completions
      - MODEL_NAME=Step-Audio-R1.1
```

---

## 附录 B: 测试命令

```bash
# 健康检查
curl -sf http://localhost:9100/api/status

# 音频处理 (同步)
curl -X POST http://localhost:9100/api/process \
  -F "audio=@/path/to/audio.mp3" \
  -F "mode=s2t"

# 音频信息
curl -X POST http://localhost:9100/api/audio/info \
  -F "audio=@/path/to/audio.mp3"

# GPU 显存监控
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

---

*报告生成时间: 2026-01-18 09:56 CST*
