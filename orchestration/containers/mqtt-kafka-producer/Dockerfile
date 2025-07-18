FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY orchestration/containers/mqtt-kafka-producer/requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs

# Set Python path
ENV PYTHONPATH=/app
