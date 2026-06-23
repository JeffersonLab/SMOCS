# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Step 1: Install ca-certificates package (needed for update-ca-certificates; Debian apt repos are HTTP so this is safe before the cert is set up)
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Step 2: Corporate SSL certificate (JLab-specific). Typically copied from /etc/ssl/certs/ to the orchestration directory. Skipped automatically if ca-bundle.crt is absent or empty.
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -f /ctx/ca-bundle.crt ] && [ -s /ctx/ca-bundle.crt ]; then \
        cp /ctx/ca-bundle.crt /usr/local/share/ca-certificates/ca-bundle.crt && \
        update-ca-certificates; \
    fi

# Step 3: Install remaining tools — curl/bash/zstd now use the updated cert store for any HTTPS calls
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

COPY ./containers/llm-agent/requirements.txt ./requirements-shared.txt
RUN pip install --no-cache-dir -r requirements-shared.txt
