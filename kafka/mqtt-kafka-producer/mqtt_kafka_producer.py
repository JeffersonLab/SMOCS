import paho.mqtt.client as mqtt
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

class MQTTKafkaProducer:
    def __init__(self):
        # MQTT Configuration
        self.mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.mqtt_username = os.getenv('MQTT_USERNAME')
        self.mqtt_password = os.getenv('MQTT_PASSWORD')
        
        # Validate required credentials
        if not self.mqtt_username or not self.mqtt_password:
            raise ValueError("MQTT_USERNAME and MQTT_PASSWORD environment variables are required")
        
        # Support multiple MQTT topics (comma separated)
        mqtt_topics_str = os.getenv('MQTT_TOPICS', 'test/topic/#')
        self.mqtt_topics = [topic.strip() for topic in mqtt_topics_str.split(',')]
        
        # Kafka Configuration
        self.kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        
        # Initialize clients
        self.mqtt_client = None
        self.kafka_producer = None
        self.kafka_admin = None
        self.created_topics = set()  # Track created topics
        
        logging.info(f"MQTT: {self.mqtt_broker}:{self.mqtt_port}")
        logging.info(f"MQTT Topics: {self.mqtt_topics}")
        logging.info(f"Kafka: {self.kafka_broker_url}")
    
    def setup_kafka_producer(self):
        self.kafka_admin = KafkaAdminClient(
            bootstrap_servers=[self.kafka_broker_url],
            api_version=(2, 8, 0)
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker_url],
            retries=5,
            retry_backoff_ms=300,
            request_timeout_ms=30000,
            metadata_max_age_ms=30000,
            api_version=(2, 8, 0),
            acks='all',
            max_block_ms=60000
        )
        logging.info("Kafka producer and admin client connected")
    
    def create_topic_if_not_exists(self, topic_name):
        """Create Kafka topic if it doesn't exist"""
        if topic_name in self.created_topics:
            return
            
        try:
            # Check if topic exists
            existing_topics = self.kafka_admin.list_topics()
            if topic_name in existing_topics:
                self.created_topics.add(topic_name)
                return
                
            # Create topic
            topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
            self.kafka_admin.create_topics([topic])
            self.created_topics.add(topic_name)
            logging.info(f"Created Kafka topic: {topic_name}")
            
        except Exception as e:
            logging.warning(f"Could not create topic {topic_name}: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker successfully!")
            for topic in self.mqtt_topics:
                result = client.subscribe(topic)
                logging.info(f"Subscribed to: {topic} - Result: {result}")
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier", 
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Unknown error code: {rc}")
            logging.error(f"MQTT connection failed: {error_msg}")
    
    def sanitize_topic_name(self, mqtt_topic):
        """Convert MQTT topic to valid Kafka topic name"""
        # Replace forward slashes with dots and remove/replace invalid characters
        kafka_topic = mqtt_topic.replace('/', '.').replace('#', 'wildcard').replace('+', 'plus')
        # Remove any other invalid characters and ensure it starts with alphanumeric
        kafka_topic = ''.join(c if c.isalnum() or c in '.-_' else '_' for c in kafka_topic)
        # No starting with a dot or dash
        if kafka_topic.startswith('.') or kafka_topic.startswith('-'):
            kafka_topic = 'topic_' + kafka_topic
        return kafka_topic
    
    def on_mqtt_message(self, client, userdata, msg):
        try:
            message = msg.payload.decode('utf-8')
            mqtt_topic = msg.topic
            
            # Convert MQTT topic to valid Kafka topic name
            kafka_topic = self.sanitize_topic_name(mqtt_topic)
            
            logging.info(f"MQTT received from '{mqtt_topic}': {message[:100]}{'...' if len(message) > 100 else ''}")
            
            self.create_topic_if_not_exists(kafka_topic)
            
            future = self.kafka_producer.send(kafka_topic, value=message.encode('utf-8'))
            record_metadata = future.get(timeout=10)
            
            logging.info(f'Forwarded to Kafka topic "{kafka_topic}" (from MQTT "{mqtt_topic}") - partition {record_metadata.partition}, offset {record_metadata.offset}')
            
        except Exception as e:
            logging.error(f"Error forwarding message: {e}")
    
    def setup_mqtt_client(self):
        self.mqtt_client = mqtt.Client(clean_session=True)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(username=self.mqtt_username, password=self.mqtt_password)
            logging.info("MQTT credentials set")
    
    def start(self):
        try:
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            logging.info("Setting up MQTT client...")
            self.setup_mqtt_client()
            
            logging.info(f"Connecting to MQTT broker: {self.mqtt_broker}:{self.mqtt_port}")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            
            logging.info("Starting MQTT loop...")
            self.mqtt_client.loop_forever()
            
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error: {e}")
            self.cleanup()
            raise
    
    def cleanup(self):
        if self.mqtt_client:
            self.mqtt_client.disconnect()
            logging.info("MQTT client disconnected")
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
            logging.info("Kafka producer closed")
        if self.kafka_admin:
            self.kafka_admin.close()
            logging.info("Kafka admin client closed")

def main():
    logging.info("Starting MQTT-to-Kafka bridge with topic preservation")
    time.sleep(10)
    
    bridge = MQTTKafkaProducer()
    bridge.start()

if __name__ == "__main__":
    main()