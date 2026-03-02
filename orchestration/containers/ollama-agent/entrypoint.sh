#!/usr/bin/env bash
set -e

if [ -z "$LLM_NAME" ]; then
  echo "LLM_NAME is not set !!!"
  exit 1
fi

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done

echo "Downloading model if not already exist: $LLM_NAME"
ollama pull "$LLM_NAME"
ollama run $LLM_NAME ""     # dummy run to load into VRAM

echo "Starting Chainlit agent..."
exec chainlit run /app/orchestration/containers/ollama-agent/chainlit_server.py --host 0.0.0.0 --port 8000
