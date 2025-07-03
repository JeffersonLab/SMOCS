from kafka import KafkaConsumer, KafkaClient
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
    logging.info(f"Connecting to Kafka broker at: {broker_url}")
    
    # Wait for broker to be available
    for i in range(10):
        try:
            client = KafkaClient(bootstrap_servers=[broker_url])
            client.close()
            logging.info("Kafka broker is available")
            break
        except:
            logging.info(f"Waiting for Kafka broker... ({i+1}/10)")
            time.sleep(5)
    
    consumer = KafkaConsumer(
        bootstrap_servers=[broker_url],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='all-topics-consumer',
        value_deserializer=lambda m: m.decode('utf-8') if m else None,
        api_version=(2, 8, 0),
        metadata_max_age_ms=5000,
        request_timeout_ms=30000
    )
    
    consumer.subscribe(pattern=r'.*')  # Subscribe to all topics
    logging.info("Subscribed to all topics")
    
    logging.info("Starting consumer for all topics...")
    
    try:
        message_count = 0
        
        for message in consumer:
            message_count += 1
            logging.info(f"Message #{message_count} - Topic: '{message.topic}' "
                        f"(partition {message.partition}, offset {message.offset}): "
                        f"{message.value[:200]}{'...' if len(message.value) > 200 else ''}")
            
    except KeyboardInterrupt:
        logging.info("Shutting down consumer...")
    except Exception as e:
        logging.error(f"Error consuming messages: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
    finally:
        consumer.close()
        logging.info("Consumer closed")

if __name__ == "__main__":
    main()