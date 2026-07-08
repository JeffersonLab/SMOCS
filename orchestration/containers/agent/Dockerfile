# syntax=docker/dockerfile:1
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Step 1: Install ca-certificates package
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Step 2: Corporate SSL certificate (JLab-specific). Skipped automatically if ca-bundle.crt is absent or empty.
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -f /ctx/orchestration/ca-bundle.crt ] && [ -s /ctx/orchestration/ca-bundle.crt ]; then \
        cp /ctx/orchestration/ca-bundle.crt /usr/local/share/ca-certificates/ca-bundle.crt && \
        update-ca-certificates; \
    fi

# Install system dependencies including Python 3.11
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    mysql-server \
    default-mysql-client \
    build-essential \
    libgl1 \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Upgrade pip for Python 3.11
RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir mujoco==3.1.1

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
    pip3 install --no-cache-dir -e .

# Backup SOCT's original configs to a safe location
# The configs are in the source directory at jlab_opt_control/cfgs
RUN mkdir -p /app/soct_configs_default && \
    cp -r /tmp/jlab/jlab_opt_control/cfgs/* /app/soct_configs_default/ && \
    echo "Backed up SOCT default configs to /app/soct_configs_default" && \
    ls -la /app/soct_configs_default/

# Copy custom SOCT configs to temp location
COPY orchestration/soct_configs /tmp/custom_soct_configs

# Replace SOCT's default configs with custom ones
# SOCT configs are in the source tree at /tmp/jlab/jlab_opt_control/cfgs/
RUN echo "Replacing SOCT configs with custom configs..." && \
    echo "Custom configs to install:" && \
    ls -la /tmp/custom_soct_configs/ && \
    echo "Copying to /tmp/jlab/jlab_opt_control/cfgs/" && \
    cp -v /tmp/custom_soct_configs/* /tmp/jlab/jlab_opt_control/cfgs/ && \
    echo "Custom SOCT configs installed successfully" && \
    echo "Current configs in SOCT directory:" && \
    ls -la /tmp/jlab/jlab_opt_control/cfgs/

# Also keep a copy in /app/soct_configs for reference/documentation
RUN mkdir -p /app/soct_configs && \
    cp /tmp/custom_soct_configs/* /app/soct_configs/

# Clean up temp custom configs directory (but keep /tmp/jlab for now)
RUN rm -rf /tmp/custom_soct_configs

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

# Add a verification step to confirm configs were copied and are accessible
RUN echo "=== Configuration Verification ===" && \
    echo "SOCT installation configs (replaced with custom):" && \
    ls -lh /tmp/jlab/jlab_opt_control/cfgs/ && \
    echo "" && \
    echo "Custom configs backup in /app/soct_configs:" && \
    ls -lh /app/soct_configs/ && \
    echo "" && \
    echo "Original SOCT configs backup in /app/soct_configs_default:" && \
    ls -lh /app/soct_configs_default/ && \
    echo "" && \
    echo "Comparing custom keras_td3.cfg with original (if exists):" && \
    (diff /app/soct_configs/keras_td3.cfg /app/soct_configs_default/keras_td3.cfg || echo "Files differ (this is expected)") && \
    echo "=== Verification Complete ==="

RUN chmod +x /entrypoint.sh