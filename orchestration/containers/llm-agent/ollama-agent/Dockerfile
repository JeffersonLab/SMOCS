FROM llm-agent-base:latest

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

COPY ./ollama-agent/requirements.txt ./requirements-agent.txt
RUN pip install --no-cache-dir -r requirements-agent.txt

ENTRYPOINT ["bash", "/app/orchestration/containers/llm-agent/ollama-agent/entrypoint.sh"]
