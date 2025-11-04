FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY orchestration/containers/csv-kafka-producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire smocs package (from root build context)
COPY smocs ./smocs
COPY orchestration/config.yaml ./config.yaml

# Create directory for CSV files
RUN mkdir -p /app/data

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "smocs.producers.csv_kafka_producer"]