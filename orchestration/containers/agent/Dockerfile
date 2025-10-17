FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.9
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    mysql-server \
    default-mysql-client \
    build-essential \
    libgl1 \
    curl \
    git \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Upgrade pip for Python 3.9
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy base requirements and install minimal dependencies
COPY orchestration/containers/agent/base-requirements.txt ./base-requirements.txt
RUN pip3 install --no-cache-dir -r base-requirements.txt

# Install jlab_opt_control (SciOptControlToolkit)
# This must be done before agent-specific requirements
RUN git clone https://github.com/JeffersonLab/SciOptControlToolkit.git /tmp/jlab && \
    cd /tmp/jlab && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -e . && \
    mkdir -p /app/jlab_configs_default && \
    cp -r /tmp/jlab/jlab_opt_control/cfgs/* /app/jlab_configs_default/ && \
    cd / && \
    rm -rf /tmp/jlab/.git

# Copy all agent-specific requirements files
COPY orchestration/containers/agent/*-requirements.txt ./

# Copy entrypoint script
COPY orchestration/containers/agent/entrypoint.sh /entrypoint.sh

# Copy the entire smocs package and config
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml

# Copy jlab configuration files for RL agents
COPY orchestration/jlab_configs ./jlab_configs

# Optional: copy SQL init script (agents can override this)
COPY orchestration/containers/agent/init.sql /docker-entrypoint-initdb.d/init.sql

# Set Python path
ENV PYTHONPATH=/app

RUN chmod +x /entrypoint.sh