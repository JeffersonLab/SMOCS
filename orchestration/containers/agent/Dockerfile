FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      python3 \
      python3-pip \
      mysql-server \
      default-mysql-client \
      build-essential \
      libgl1 \
      curl \
      git \
      && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy base requirements and install minimal dependencies
COPY orchestration/containers/agent/base-requirements.txt ./base-requirements.txt
RUN pip3 install --no-cache-dir -r base-requirements.txt

# Copy all agent-specific requirements files
COPY orchestration/containers/agent/*-requirements.txt ./

# Copy entrypoint script
COPY orchestration/containers/agent/entrypoint.sh /entrypoint.sh

# Copy the entire smocs package and config
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml

# Optional: copy SQL init script (agents can override this)
COPY orchestration/containers/agent/init.sql /docker-entrypoint-initdb.d/init.sql

# Set Python path
ENV PYTHONPATH=/app
RUN chmod +x /entrypoint.sh