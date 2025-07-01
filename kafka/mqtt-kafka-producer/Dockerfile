FROM python:3.9-slim

WORKDIR /app

COPY mqtt_kafka_producer.py ./

# Install required packages
RUN pip install kafka-python==2.0.2 paho-mqtt==1.6.1

CMD ["python", "mqtt_kafka_producer.py"]