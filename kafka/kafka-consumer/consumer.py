from kafka import KafkaConsumer
import logging
import os

logging.basicConfig(level=logging.INFO)

def main():
    broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:19092')
    
    logging.info(f"Connecting to Kafka broker at: {broker_url}")
    
    consumer = KafkaConsumer(
        'mytopic',                          # Topic to subscribe to
        bootstrap_servers=[broker_url],
        auto_offset_reset='earliest',       # Start from the beginning if no offset
        enable_auto_commit=True,            # Automatically commit offsets
        group_id='my-group',               # Consumer group ID
        value_deserializer=lambda x: x.decode('utf-8') if x else None,
        api_version=(2, 8, 0),             # Specify API version for compatibility
        consumer_timeout_ms=1000,          # Timeout for polling
        session_timeout_ms=30000,          # Session timeout
        heartbeat_interval_ms=3000,        # Heartbeat interval
        metadata_max_age_ms=30000          # Metadata refresh interval
    )
    
    logging.info("Kafka Consumer started. Listening for messages...")
    
    try:
        for message in consumer:
            if message.value:
                logging.info(f'Received: {message.value} from partition {message.partition} at offset {message.offset}')
            else:
                logging.info("Received empty message")
                
    except KeyboardInterrupt:
        logging.info("Consumer interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        consumer.close()
        logging.info("Kafka Consumer stopped.")

if __name__ == "__main__":
    main()