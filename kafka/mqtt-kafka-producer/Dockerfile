FROM python:3.10-slim
WORKDIR /app

# Copy and install requirements first (for better caching)
COPY mqtt-kafka-producer/requirements.txt .
RUN pip install -r requirements.txt

# Copy both cores and producers directories (now they're in context)
COPY cores ./cores
COPY mqtt-kafka-producer/mqtt_kafka_producer.py ./

# Set PYTHONPATH to include current directory
ENV PYTHONPATH=/app

CMD ["python", "mqtt_kafka_producer.py"]