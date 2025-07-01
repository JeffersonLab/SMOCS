FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY consumer.py ./

# Install kafka-python with specific version for better compatibility
RUN pip install kafka-python==2.0.2

# Run the consumer
CMD ["python", "consumer.py"]