FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY orchestration/containers/gymnasium-kafka-controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml

# Set Python path
ENV PYTHONPATH=/app