"""
Step-Audio-R1.1 Web UI + API Server
Provides: Web Interface, REST API, Swagger Docs
"""
import os
import json
import uuid
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flasgger import Swagger
from werkzeug.utils import secure_filename

from stepaudior1vllm import StepAudioR1, AudioService

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Step-Audio-R1.1 API',
    'version': '1.0',
    'description': 'Audio Language Model API with reasoning capabilities'
}
swagger = Swagger(app)

# Config
VLLM_API_URL = os.environ.get('VLLM_API_URL', 'http://localhost:9999/v1/chat/completions')
MODEL_NAME = os.environ.get('MODEL_NAME', 'Step-Audio-R1.1')
UPLOAD_FOLDER = '/tmp/step-audio-r1/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac', 'webm', 'wma', 'amr'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state
model = None
tasks = {}
stats = {'processing': 0, 'waiting': 0, 'completed': 0, 'total_time': 0}

def get_model():
    global model
    if model is None:
        model = StepAudioR1(VLLM_API_URL, MODEL_NAME)
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============== Web UI ==============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# ============== Health Check ==============
@app.route('/health')
def health():
    """
    Health check endpoint
    ---
    tags: [System]
    responses:
      200:
        description: Service is healthy
    """
    return jsonify({'status': 'healthy', 'model': MODEL_NAME})

@app.route('/api/status')
def api_status():
    """
    Get service status and statistics
    ---
    tags: [System]
    responses:
      200:
        description: Service status
    """
    return jsonify({
        'model': MODEL_NAME,
        'vllm_url': VLLM_API_URL,
        'stats': stats
    })

# ============== Main API ==============
@app.route('/api/process', methods=['POST'])
def process_audio():
    """
    Process audio file with Step-Audio-R1.1
    ---
    tags: [Audio Processing]
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: Audio file (WAV, MP3, FLAC, M4A)
      - name: mode
        in: formData
        type: string
        enum: [s2t, understand, translate, summarize, asr]
        default: s2t
        description: Processing mode
      - name: instruction
        in: formData
        type: string
        description: Additional instruction
      - name: target_lang
        in: formData
        type: string
        default: Chinese
        description: Target language for translation
      - name: question
        in: formData
        type: string
        description: Question about the audio
      - name: max_tokens
        in: formData
        type: integer
        default: 2048
      - name: temperature
        in: formData
        type: number
        default: 0.7
    responses:
      200:
        description: Processing result
      400:
        description: Invalid request
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    print(f"[DEBUG] Received file: '{file.filename}', content_type: {file.content_type}")
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'none'
        return jsonify({'error': f'Invalid file type: .{ext}. Allowed: wav, mp3, flac, m4a, ogg, aac, webm, wma, amr'}), 400
    
    # Save file
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Get parameters
    mode = request.form.get('mode', 's2t')
    instruction = request.form.get('instruction', '')
    target_lang = request.form.get('target_lang', 'Chinese')
    question = request.form.get('question', '')
    max_tokens = int(request.form.get('max_tokens', 2048))
    temperature = float(request.form.get('temperature', 0.7))
    top_p = float(request.form.get('top_p', 0.9))
    repetition_penalty = float(request.form.get('repetition_penalty', 1.0))
    
    # Build messages based on mode
    messages = build_messages(mode, filepath, instruction, target_lang, question)
    
    # Process
    stats['processing'] += 1
    start_time = time.time()
    
    try:
        m = get_model()
        full_text = ""
        for response, text, audio in m.stream(
            messages, 
            max_tokens=max_tokens, 
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[151665]
        ):
            if text:
                full_text += text
        
        elapsed = time.time() - start_time
        stats['completed'] += 1
        stats['total_time'] += elapsed
        
        # Parse thinking and answer
        thinking, answer = parse_response(full_text)
        
        return jsonify({
            'status': 'success',
            'thinking': thinking,
            'answer': answer,
            'full_response': full_text,
            'elapsed_time': round(elapsed, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    finally:
        stats['processing'] -= 1
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass

def build_messages(mode, audio_path, instruction='', target_lang='Chinese', question=''):
    """Build messages based on processing mode"""
    
    if mode == 'asr':
        # Pure ASR transcription
        messages = [
            {"role": "system", "content": "You are an expert speech recognition system. Transcribe the audio accurately."},
            {"role": "human", "content": [
                {"type": "audio", "audio": audio_path}
            ]}
        ]
    elif mode == 'translate':
        messages = [
            {"role": "system", "content": f"You are a professional translator. Listen to the audio and translate it to {target_lang}."},
            {"role": "human", "content": [
                {"type": "text", "text": f"Translate the following audio to {target_lang}:"},
                {"type": "audio", "audio": audio_path}
            ]}
        ]
    elif mode == 'summarize':
        messages = [
            {"role": "system", "content": "You are an expert at summarizing audio content. Provide a concise summary."},
            {"role": "human", "content": [
                {"type": "text", "text": "Summarize the following audio:"},
                {"type": "audio", "audio": audio_path}
            ]}
        ]
    elif mode == 'understand':
        q = question or "What is the main content of this audio?"
        messages = [
            {"role": "system", "content": "You are an expert audio analyst. Answer questions about audio content."},
            {"role": "human", "content": [
                {"type": "text", "text": q},
                {"type": "audio", "audio": audio_path}
            ]}
        ]
    else:  # s2t default
        content = [{"type": "audio", "audio": audio_path}]
        if instruction:
            content.insert(0, {"type": "text", "text": instruction})
        messages = [
            {"role": "human", "content": content}
        ]
    
    # Add thinking prompt
    messages.append({"role": "assistant", "content": "<think>\n", "eot": False})
    return messages

def parse_response(text):
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

# ============== Async Task API ==============
@app.route('/api/task', methods=['POST'])
def create_task():
    """
    Create async processing task
    ---
    tags: [Async Tasks]
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Task created
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'pending',
        'filepath': filepath,
        'params': dict(request.form),
        'result': None,
        'created_at': time.time()
    }
    
    # Start async processing
    thread = threading.Thread(target=process_task, args=(task_id,))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'pending'})

@app.route('/api/task/<task_id>')
def get_task(task_id):
    """
    Get task status
    ---
    tags: [Async Tasks]
    parameters:
      - name: task_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Task status
      404:
        description: Task not found
    """
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(tasks[task_id])

def process_task(task_id):
    """Background task processor"""
    task = tasks[task_id]
    task['status'] = 'processing'
    
    try:
        params = task['params']
        mode = params.get('mode', 's2t')
        messages = build_messages(
            mode, 
            task['filepath'],
            params.get('instruction', ''),
            params.get('target_lang', 'Chinese'),
            params.get('question', '')
        )
        
        m = get_model()
        full_text = ""
        for _, text, _ in m.stream(messages, max_tokens=2048, temperature=0.7, stop_token_ids=[151665]):
            if text:
                full_text += text
        
        thinking, answer = parse_response(full_text)
        task['result'] = {'thinking': thinking, 'answer': answer}
        task['status'] = 'completed'
    except Exception as e:
        task['status'] = 'failed'
        task['error'] = str(e)
    finally:
        try:
            os.remove(task['filepath'])
        except:
            pass

# ============== Utility APIs ==============
@app.route('/api/audio/info', methods=['POST'])
def audio_info():
    """
    Get audio file information
    ---
    tags: [Utilities]
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Audio information
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        info = AudioService.get_audio_info(filepath)
        return jsonify(info)
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9100))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
