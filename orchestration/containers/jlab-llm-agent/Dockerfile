FROM python:3.12-slim

# Fix corporate SSL ==> IMPORTANT NOTE: copied from /etc/ssl/certs/ca-bundle.crt
COPY ./ca-bundle.crt /usr/local/share/ca-certificates/ca-bundle.crt
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    update-ca-certificates

RUN apt-get update && apt-get install -y \
    curl \
    bash \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI + Compose plugin from Docker's official apt repo ==> To enable launching sibling containers
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
        https://download.docker.com/linux/debian \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["chainlit", "run", "/app/orchestration/containers/jlab-llm-agent/chainlit_server.py",  "--host", "0.0.0.0", "--port", "8000"]