from kafka import KafkaProducer
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:19092')
    
    logging.info(f"Connecting to Kafka broker at: {broker_url}")
    
    producer = KafkaProducer(
        bootstrap_servers=[broker_url],
        retries=5,
        retry_backoff_ms=300,
        request_timeout_ms=30000,
        metadata_max_age_ms=30000,
        api_version=(2, 8, 0)
    )
    
    try:
        for i in range(10):
            message = f"Message {i}"
            future = producer.send('mytopic', value=message.encode('utf-8'))
            
            record_metadata = future.get(timeout=10)
            
            logging.info(f'Sent: {message} to partition {record_metadata.partition} at offset {record_metadata.offset}')
            time.sleep(1)
            
    except Exception as e:
        logging.error(f"Error sending messages: {e}")
    finally:
        producer.flush()
        producer.close()
        logging.info("Producer finished and closed")

if __name__ == "__main__":
    main()