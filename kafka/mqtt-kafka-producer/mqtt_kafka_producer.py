import paho.mqtt.client as mqtt
import os
import logging
import sys
from pathlib import Path

from cores import KafkaProducerBase

logging.basicConfig(level=logging.INFO)


class MQTTKafkaProducer(KafkaProducerBase):
    """
    MQTT to Kafka producer that inherits from KafkaProducerBase.
    
    This producer connects to an MQTT broker, subscribes to specified topics,
    and forwards all received messages to corresponding Kafka topics.
    """
    
    def __init__(self):
        """
        Initialize the MQTT to Kafka producer.
        Uses environment variables for configuration.
        """
        # Get Kafka broker URL from environment
        kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        
        # Initialize base class
        super().__init__(kafka_broker_url)
        
        # MQTT Configuration from environment
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
        
        # MQTT client
        self.mqtt_client = None
        
        logging.info(f"MQTT: {self.mqtt_broker}:{self.mqtt_port}")
        logging.info(f"MQTT Topics: {self.mqtt_topics}")
    
    def sanitize_topic_name(self, mqtt_topic):
        """
        Convert MQTT topic to valid Kafka topic name.
        
        Args:
            mqtt_topic (str): Original MQTT topic name
            
        Returns:
            str: Sanitized Kafka topic name
        """
        # Replace forward slashes with dots and remove/replace invalid characters
        kafka_topic = mqtt_topic.replace('/', '.').replace('#', 'wildcard').replace('+', 'plus')
        # Remove any other invalid characters and ensure it starts with alphanumeric
        kafka_topic = ''.join(c if c.isalnum() or c in '.-_' else '_' for c in kafka_topic)
        # No starting with a dot or dash
        if kafka_topic.startswith('.') or kafka_topic.startswith('-'):
            kafka_topic = 'topic_' + kafka_topic
        return kafka_topic
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """
        Callback for when MQTT client connects to the broker.
        
        Args:
            client: MQTT client instance
            userdata: User data passed to callbacks
            flags: Response flags sent by the broker
            rc: Connection result code
        """
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
    
    def on_mqtt_message(self, client, userdata, msg):
        """
        Callback for when an MQTT message is received.
        
        Args:
            client: MQTT client instance
            userdata: User data passed to callbacks
            msg: MQTT message object
        """
        try:
            message = msg.payload.decode('utf-8')
            mqtt_topic = msg.topic
            
            # Convert MQTT topic to valid Kafka topic name
            kafka_topic = self.sanitize_topic_name(mqtt_topic)
            
            logging.info(f"MQTT received from '{mqtt_topic}': {message[:100]}{'...' if len(message) > 100 else ''}")
            
            # Send to Kafka using base class method
            record_metadata = self.send_to_kafka(kafka_topic, message)
            
            logging.info(f'Forwarded to Kafka topic "{kafka_topic}" (from MQTT "{mqtt_topic}") - partition {record_metadata.partition}, offset {record_metadata.offset}')
            
        except Exception as e:
            logging.error(f"Error forwarding message: {e}")
    
    def setup_mqtt_client(self):
        """
        Set up MQTT client with callbacks and credentials.
        """
        self.mqtt_client = mqtt.Client(clean_session=True)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(username=self.mqtt_username, password=self.mqtt_password)
            logging.info("MQTT credentials set")
    
    def start(self):
        """
        Start the MQTT to Kafka producer.
        
        This method:
        1. Sets up Kafka producer (via base class)
        2. Sets up MQTT client
        3. Connects to MQTT broker
        4. Starts the main processing loop
        """
        try:
            logging.info("Starting MQTT-to-Kafka bridge with topic preservation")
            
            # Setup Kafka producer using base class method
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            # Setup MQTT client
            logging.info("Setting up MQTT client...")
            self.setup_mqtt_client()
            
            # Connect to MQTT broker
            logging.info(f"Connecting to MQTT broker: {self.mqtt_broker}:{self.mqtt_port}")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            
            # Start MQTT loop
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
        """
        Clean up MQTT and Kafka resources.
        """
        # Disconnect MQTT client
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logging.info("MQTT client disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting MQTT client: {e}")
        
        # Call base class cleanup for Kafka resources
        super().cleanup()


def main():
    """
    Main entry point for the MQTT to Kafka producer.
    """
    logging.info("Starting MQTT-to-Kafka bridge with topic preservation")
    
    producer = MQTTKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()