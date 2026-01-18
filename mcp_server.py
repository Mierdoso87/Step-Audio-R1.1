"""
Step-Audio-R1.1 MCP Server
Model Context Protocol interface for programmatic access
"""
import os
import tempfile
import base64
from typing import Optional
from fastmcp import FastMCP

from stepaudior1vllm import StepAudioR1, AudioService

mcp = FastMCP("step-audio-r1")

# Config
VLLM_API_URL = os.environ.get('VLLM_API_URL', 'http://localhost:9999/v1/chat/completions')
MODEL_NAME = os.environ.get('MODEL_NAME', 'Step-Audio-R1.1')
UPLOAD_DIR = '/tmp/step-audio-r1'

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Lazy model loading
_model = None

def get_model():
    global _model
    if _model is None:
        _model = StepAudioR1(VLLM_API_URL, MODEL_NAME)
    return _model


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    instruction: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> dict:
    """
    Transcribe audio file to text with optional instructions.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, FLAC, M4A)
        instruction: Additional instruction for processing
        max_tokens: Maximum output tokens (default: 2048)
        temperature: Sampling temperature (default: 0.7)
    
    Returns:
        dict with 'thinking', 'answer', and 'full_response'
    """
    if not os.path.exists(audio_path):
        return {'status': 'error', 'error': f'File not found: {audio_path}'}
    
    content = [{"type": "audio", "audio": audio_path}]
    if instruction:
        content.insert(0, {"type": "text", "text": instruction})
    
    messages = [
        {"role": "human", "content": content},
        {"role": "assistant", "content": "<think>\n", "eot": False}
    ]
    
    try:
        model = get_model()
        full_text = ""
        for _, text, _ in model.stream(messages, max_tokens=max_tokens, temperature=temperature, stop_token_ids=[151665]):
            if text:
                full_text += text
        
        thinking, answer = _parse_response(full_text)
        return {
            'status': 'success',
            'thinking': thinking,
            'answer': answer,
            'full_response': full_text
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@mcp.tool()
def understand_audio(
    audio_path: str,
    question: str,
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> dict:
    """
    Answer questions about audio content.
    
    Args:
        audio_path: Path to the audio file
        question: Question about the audio content
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    
    Returns:
        dict with 'thinking', 'answer', and 'full_response'
    """
    if not os.path.exists(audio_path):
        return {'status': 'error', 'error': f'File not found: {audio_path}'}
    
    messages = [
        {"role": "system", "content": "You are an expert audio analyst. Answer questions about audio content."},
        {"role": "human", "content": [
            {"type": "text", "text": question},
            {"type": "audio", "audio": audio_path}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False}
    ]
    
    try:
        model = get_model()
        full_text = ""
        for _, text, _ in model.stream(messages, max_tokens=max_tokens, temperature=temperature, stop_token_ids=[151665]):
            if text:
                full_text += text
        
        thinking, answer = _parse_response(full_text)
        return {
            'status': 'success',
            'thinking': thinking,
            'answer': answer,
            'full_response': full_text
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@mcp.tool()
def translate_audio(
    audio_path: str,
    target_language: str = "Chinese",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> dict:
    """
    Translate audio content to target language.
    
    Args:
        audio_path: Path to the audio file
        target_language: Target language (Chinese, English, Japanese, Korean, French, German, Spanish)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    
    Returns:
        dict with 'thinking', 'answer', and 'full_response'
    """
    if not os.path.exists(audio_path):
        return {'status': 'error', 'error': f'File not found: {audio_path}'}
    
    messages = [
        {"role": "system", "content": f"You are a professional translator. Listen to the audio and translate it to {target_language}."},
        {"role": "human", "content": [
            {"type": "text", "text": f"Translate the following audio to {target_language}:"},
            {"type": "audio", "audio": audio_path}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False}
    ]
    
    try:
        model = get_model()
        full_text = ""
        for _, text, _ in model.stream(messages, max_tokens=max_tokens, temperature=temperature, stop_token_ids=[151665]):
            if text:
                full_text += text
        
        thinking, answer = _parse_response(full_text)
        return {
            'status': 'success',
            'thinking': thinking,
            'answer': answer,
            'full_response': full_text
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@mcp.tool()
def summarize_audio(
    audio_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> dict:
    """
    Summarize audio content.
    
    Args:
        audio_path: Path to the audio file
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    
    Returns:
        dict with 'thinking', 'answer', and 'full_response'
    """
    if not os.path.exists(audio_path):
        return {'status': 'error', 'error': f'File not found: {audio_path}'}
    
    messages = [
        {"role": "system", "content": "You are an expert at summarizing audio content. Provide a concise summary."},
        {"role": "human", "content": [
            {"type": "text", "text": "Summarize the following audio:"},
            {"type": "audio", "audio": audio_path}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False}
    ]
    
    try:
        model = get_model()
        full_text = ""
        for _, text, _ in model.stream(messages, max_tokens=max_tokens, temperature=temperature, stop_token_ids=[151665]):
            if text:
                full_text += text
        
        thinking, answer = _parse_response(full_text)
        return {
            'status': 'success',
            'thinking': thinking,
            'answer': answer,
            'full_response': full_text
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@mcp.tool()
def get_audio_info(audio_path: str) -> dict:
    """
    Get audio file information (duration, sample rate, channels).
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        dict with audio metadata
    """
    if not os.path.exists(audio_path):
        return {'status': 'error', 'error': f'File not found: {audio_path}'}
    
    try:
        info = AudioService.get_audio_info(audio_path)
        return {'status': 'success', **info}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@mcp.tool()
def process_audio_base64(
    audio_base64: str,
    audio_format: str = "wav",
    mode: str = "transcribe",
    instruction: str = "",
    question: str = "",
    target_language: str = "Chinese",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> dict:
    """
    Process base64-encoded audio data.
    
    Args:
        audio_base64: Base64-encoded audio data
        audio_format: Audio format (wav, mp3, flac, m4a)
        mode: Processing mode (transcribe, understand, translate, summarize)
        instruction: Additional instruction (for transcribe mode)
        question: Question about audio (for understand mode)
        target_language: Target language (for translate mode)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    
    Returns:
        dict with processing result
    """
    try:
        # Decode and save to temp file
        audio_data = base64.b64decode(audio_base64)
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{os.getpid()}.{audio_format}")
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        # Process based on mode
        if mode == 'transcribe':
            result = transcribe_audio(temp_path, instruction, max_tokens, temperature)
        elif mode == 'understand':
            result = understand_audio(temp_path, question, max_tokens, temperature)
        elif mode == 'translate':
            result = translate_audio(temp_path, target_language, max_tokens, temperature)
        elif mode == 'summarize':
            result = summarize_audio(temp_path, max_tokens, temperature)
        else:
            result = {'status': 'error', 'error': f'Unknown mode: {mode}'}
        
        # Cleanup
        os.remove(temp_path)
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def _parse_response(text: str) -> tuple:
    """Parse thinking and answer from response"""
    thinking = ""
    answer = text
    
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
    elif "<think>" in text:
        thinking = text.replace("<think>", "").strip()
        answer = ""
    
    return thinking, answer


if __name__ == "__main__":
    mcp.run()
