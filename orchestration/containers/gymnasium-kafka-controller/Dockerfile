FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY orchestration/containers/gymnasium-kafka-controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml

# Install pyepics only when the configured environment talks to EPICS.
# When adding a new EPICS-backed environment, append its gym id to the tuple below.
RUN if python -c "import sys, yaml; env = yaml.safe_load(open('config.yaml')).get('gymnasium', {}).get('environment'); sys.exit(env not in ('UITF-v0',))"; then \
        pip install --no-cache-dir pyepics==3.5.8; \
    fi

# Set Python path
ENV PYTHONPATH=/app