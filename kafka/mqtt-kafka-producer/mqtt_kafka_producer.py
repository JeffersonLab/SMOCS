import paho.mqtt.client as mqtt
from kafka import KafkaProducer
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

class MQTTKafkaProducer:
    def __init__(self):
        # MQTT Configuration
        self.mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.mqtt_username = os.getenv('MQTT_USERNAME', '')
        self.mqtt_password = os.getenv('MQTT_PASSWORD', '')
        self.mqtt_topic = os.getenv('MQTT_TOPIC', 'sensor/data')
        
        # Kafka Configuration - using same topic as your existing consumer
        self.kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        
        # Initialize clients
        self.mqtt_client = None
        self.kafka_producer = None
        
        logging.info(f"MQTT: {self.mqtt_broker}:{self.mqtt_port}, Topic: {self.mqtt_topic}")
        logging.info(f"Kafka: {self.kafka_broker_url}, Topic: mytopic")
        if self.mqtt_username:
            logging.info(f"MQTT Username: {self.mqtt_username}")
        else:
            logging.info("No MQTT username provided")
    
    def setup_kafka_producer(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker_url],
            retries=5,
            retry_backoff_ms=300,
            request_timeout_ms=30000,
            metadata_max_age_ms=30000,
            api_version=(2, 8, 0)
        )
        logging.info("Kafka producer connected")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker successfully!")
            result = client.subscribe(self.mqtt_topic)
            logging.info(f"Subscribed to: {self.mqtt_topic} - Result: {result}")
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
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        if rc != 0:
            logging.warning(f"Unexpected MQTT disconnection: {rc}")
        else:
            logging.info("MQTT disconnected")
    
    def on_mqtt_subscribe(self, client, userdata, mid, granted_qos):
        logging.info(f"Subscription confirmed - QoS: {granted_qos}")
    
    def on_mqtt_message(self, client, userdata, msg):
        try:
            # Get the message payload
            message = msg.payload.decode('utf-8')
            topic = msg.topic
            
            logging.info(f"MQTT received from '{topic}': {message[:100]}{'...' if len(message) > 100 else ''}")
            
            # Send directly to Kafka mytopic (same as your existing consumer)
            future = self.kafka_producer.send('mytopic', value=message.encode('utf-8'))
            record_metadata = future.get(timeout=10)
            
            logging.info(f'Forwarded to Kafka mytopic - partition {record_metadata.partition}, offset {record_metadata.offset}')
            
        except Exception as e:
            logging.error(f"Error forwarding message: {e}")
    
    def on_mqtt_log(self, client, userdata, level, buf):
        logging.info(f"MQTT Log: {buf}")
    
    def setup_mqtt_client(self):
        # Create client with clean session
        self.mqtt_client = mqtt.Client(clean_session=True)
        
        # Set all callbacks
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_subscribe = self.on_mqtt_subscribe
        self.mqtt_client.on_log = self.on_mqtt_log
        
        # Set credentials if provided (both must be non-empty)
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(username=self.mqtt_username, password=self.mqtt_password)
            logging.info("MQTT credentials set")
        else:
            logging.info("No MQTT credentials provided")
    
    def start(self):
        try:
            # Setup Kafka first
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            # Setup MQTT client
            logging.info("Setting up MQTT client...")
            self.setup_mqtt_client()
            
            # Connect to MQTT broker with timeout
            logging.info(f"Attempting to connect to MQTT broker: {self.mqtt_broker}:{self.mqtt_port}")
            try:
                result = self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
                if result != 0:
                    logging.error(f"MQTT connect() returned error code: {result}")
                    return
                logging.info("MQTT connect() call successful, waiting for connection...")
            except Exception as e:
                logging.error(f"Exception during MQTT connect: {e}")
                return
            
            # Start the loop with a timeout to see if connection works
            logging.info("Starting MQTT loop...")
            self.mqtt_client.loop_start()
            
            # Wait a bit to see if connection establishes
            time.sleep(5)
            
            # Check if we're connected
            if self.mqtt_client.is_connected():
                logging.info("MQTT connection successful! Switching to blocking loop...")
                self.mqtt_client.loop_stop()
                self.mqtt_client.loop_forever()
            else:
                logging.error("MQTT connection failed after 5 seconds")
                self.cleanup()
                return
            
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
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

def main():
    logging.info("Starting MQTT-to-Kafka bridge")
    
    # Wait for Kafka to be ready
    logging.info("Waiting for Kafka to be ready...")
    time.sleep(10)
    
    bridge = MQTTKafkaProducer()
    bridge.start()

if __name__ == "__main__":
    main()