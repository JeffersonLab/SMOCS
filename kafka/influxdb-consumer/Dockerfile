FROM python:3.10-slim

WORKDIR /app

# Copy and install requirements first (for better caching)
COPY influxdb-consumer/requirements.txt .
RUN pip install -r requirements.txt

# Copy cores directory (contains base classes)
COPY cores ./cores

COPY influxdb-consumer/consumer.py ./

ENV PYTHONPATH=/app

CMD ["python", "consumer.py"]