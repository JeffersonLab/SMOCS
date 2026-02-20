FROM python:3.12-slim

# Install system tools
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    bash \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama ==> Does not start it
RUN curl -fsSL https://ollama.com/install.sh | bash

WORKDIR /app

# Copy all following files under /app: 1) requirements.txt, 2) entrypoint.sh, 3) chainlit_server.py, 4) mcp_server.py
# Remember that the build context is the "ollama-agent" directory in the local host under which all these files actually exist
#COPY ./requirements.txt ./entrypoint.sh ./chainlit_server.py ./mcp_server.py /app/

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["bash", "/app/orchestration/containers/ollama-agent/entrypoint.sh"]