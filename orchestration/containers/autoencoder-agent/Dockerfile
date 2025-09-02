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

# Optional: copy SQL init script
COPY orchestration/containers/autoencoder-agent/init.sql /docker-entrypoint-initdb.d/init.sql

# Set working directory
WORKDIR /app

# Copy requirements files and install Python dependencies
COPY orchestration/containers/autoencoder-agent/requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy entrypoint script
COPY orchestration/containers/autoencoder-agent/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the entire smocs package, config, and autoencoder agent
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml
COPY smocs/agents/autoencoder_agent.py ./autoencoder_agent.py

# Set Python path
ENV PYTHONPATH=/app