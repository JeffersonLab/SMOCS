FROM python:3.9-slim

WORKDIR /app

COPY producer.py ./

RUN pip install kafka-python==2.0.2

CMD ["python", "producer.py"]