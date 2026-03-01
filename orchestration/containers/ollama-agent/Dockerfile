FROM python:3.12-slim

# Fix corporate SSL ==> IMPORTANT NOTE: copied from /etc/ssl/certs/ca-bundle.crt
COPY ./ca-bundle.crt /usr/local/share/ca-certificates/ca-bundle.crt
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    bash \
    zstd \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["bash", "/app/orchestration/containers/ollama-agent/entrypoint.sh"]