FROM python:3.10-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy both cores and producers directories
COPY ../cores ./cores
COPY ../producers ./producers

# Copy the main producer file
COPY mqtt_kafka_producer.py ./

# Set PYTHONPATH to include current directory
ENV PYTHONPATH=/app

CMD ["python", "mqtt_kafka_producer.py"]