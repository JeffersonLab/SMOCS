FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
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

# Install Python ML packages
# RUN pip3 install --upgrade pip --break-system-packages

# Optional: copy SQL init script
COPY orchestration/containers/agent/init.sql /docker-entrypoint-initdb.d/init.sql

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY orchestration/containers/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestration/containers/agent/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml