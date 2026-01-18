# Step-Audio-R1.1 All-in-One Docker Image
# Base: stepfun vLLM with Step-Audio support
FROM stepfun2025/vllm:step-audio-2-v20250909

WORKDIR /app

# Install dependencies for Web UI and MCP
RUN pip install --no-cache-dir --ignore-installed \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    flasgger==0.9.7.1 \
    pydub==0.25.1 \
    fastmcp==0.1.0 \
    gunicorn==21.2.0

# Copy application code
COPY app.py /app/
COPY stepaudior1vllm.py /app/
COPY mcp_server.py /app/
COPY static/ /app/static/
COPY templates/ /app/templates/

# Create necessary directories
RUN mkdir -p /app/uploads /app/request_logs /tmp/step-audio-r1

# Environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

EXPOSE 9100

CMD ["python3", "app.py"]
