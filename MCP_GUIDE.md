# Step-Audio-R1.1 MCP Guide

## Overview

Step-Audio-R1.1 provides MCP (Model Context Protocol) interface for programmatic access to audio processing capabilities.

## Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "step-audio-r1": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "VLLM_API_URL": "http://localhost:9101/v1/chat/completions",
        "MODEL_NAME": "Step-Audio-R1.1"
      }
    }
  }
}
```

## Available Tools

### 1. transcribe_audio
Transcribe audio file to text with optional instructions.

```python
result = await mcp_client.call_tool(
    "transcribe_audio",
    {
        "audio_path": "/path/to/audio.wav",
        "instruction": "Transcribe in detail",
        "max_tokens": 2048,
        "temperature": 0.7
    }
)
```

### 2. understand_audio
Answer questions about audio content.

```python
result = await mcp_client.call_tool(
    "understand_audio",
    {
        "audio_path": "/path/to/audio.wav",
        "question": "What emotion is expressed in this audio?",
        "max_tokens": 2048,
        "temperature": 0.7
    }
)
```

### 3. translate_audio
Translate audio content to target language.

```python
result = await mcp_client.call_tool(
    "translate_audio",
    {
        "audio_path": "/path/to/audio.wav",
        "target_language": "Chinese",
        "max_tokens": 2048,
        "temperature": 0.7
    }
)
```

Supported languages: Chinese, English, Japanese, Korean, French, German, Spanish

### 4. summarize_audio
Summarize audio content.

```python
result = await mcp_client.call_tool(
    "summarize_audio",
    {
        "audio_path": "/path/to/audio.wav",
        "max_tokens": 1024,
        "temperature": 0.7
    }
)
```

### 5. get_audio_info
Get audio file metadata.

```python
result = await mcp_client.call_tool(
    "get_audio_info",
    {"audio_path": "/path/to/audio.wav"}
)
# Returns: duration, sample_rate, channels, sample_width, frame_count
```

### 6. process_audio_base64
Process base64-encoded audio data (useful when file path is not available).

```python
import base64

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = await mcp_client.call_tool(
    "process_audio_base64",
    {
        "audio_base64": audio_b64,
        "audio_format": "wav",
        "mode": "transcribe",  # transcribe, understand, translate, summarize
        "instruction": "",
        "max_tokens": 2048
    }
)
```

## Response Format

All tools return a dict with:

```json
{
    "status": "success",
    "thinking": "Model's reasoning process...",
    "answer": "Final answer...",
    "full_response": "Complete response including thinking..."
}
```

On error:
```json
{
    "status": "error",
    "error": "Error message"
}
```

## Comparison with REST API

| Feature | MCP | REST API |
|---------|-----|----------|
| Use Case | Programmatic/Agent access | Web/HTTP clients |
| File Input | Local path or base64 | File upload |
| Async | Native | Via /api/task endpoint |
| Streaming | Not supported | Not supported |

## Running Standalone

```bash
# Set environment variables
export VLLM_API_URL=http://localhost:9101/v1/chat/completions
export MODEL_NAME=Step-Audio-R1.1

# Run MCP server
python mcp_server.py
```
