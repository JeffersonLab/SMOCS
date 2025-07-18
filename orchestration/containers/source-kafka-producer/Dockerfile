FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY orchestration/containers/source-kafka-producer/requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs

# Copy configuration file
COPY orchestration/config.yaml ./config.yaml

# Set Python path
ENV PYTHONPATH=/app
