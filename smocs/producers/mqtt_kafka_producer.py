import paho.mqtt.client as mqtt
import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from smocs.cores import KafkaProducerBase
from smocs.utils import ConfigLoader

logging.basicConfig(level=logging.INFO)


class MQTTKafkaProducer(KafkaProducerBase):
    """
    MQTT to Kafka producer with configuration-driven message parsing.
    
    This producer connects to an MQTT broker, subscribes to configured topics,
    parses messages according to their specific configurations, and forwards
    structured data to corresponding Kafka topics.
    """
    
    def __init__(self):
        """
        Initialize the MQTT to Kafka producer with configuration.
        """
        # Load configuration first
        config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')
        self.config_loader = ConfigLoader(config_path)
        
        # Validate MQTT configuration exists
        if not self.config_loader.has_config(name='mqtt'):
            raise ValueError("No MQTT configuration found in config file")
        
        # Initialize base class
        kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        super().__init__(kafka_broker_url)
        
        # MQTT Configuration from environment
        self.mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.mqtt_username = os.getenv('MQTT_USERNAME')
        self.mqtt_password = os.getenv('MQTT_PASSWORD')
        
        # Validate required credentials
        if not self.mqtt_username or not self.mqtt_password:
            raise ValueError("MQTT_USERNAME and MQTT_PASSWORD environment variables are required")
        
        # Get configured topics
        self.topic_configs = {
            config['topic']: config 
            for config in self.config_loader.get_mqtt_topic_configs()
        }
        
        # MQTT client
        self.mqtt_client = None
        
        logging.info(f"MQTT: {self.mqtt_broker}:{self.mqtt_port}")
        logging.info(f"Loaded {len(self.topic_configs)} topic configurations:")
        for topic in self.topic_configs.keys():
            logging.info(f"  - {topic}")
    
    def extract_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Extract value from nested dictionary using dot notation path.
        
        Args:
            data: Source dictionary
            path: Dot notation path (e.g., 'res1.value')
            
        Returns:
            Extracted value
            
        Raises:
            ValueError: If path cannot be extracted
        """
        keys = path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to extract path '{path}': {e}")
    
    def parse_mqtt_message(self, message: str, topic_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse MQTT message according to topic configuration.
        
        Args:
            message: Raw MQTT message string
            topic_config: Configuration for this topic
            
        Returns:
            Structured message for Kafka
            
        Raises:
            ValueError: If parsing fails
        """
        try:
            # Parse JSON message
            data = json.loads(message)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in MQTT message: {e}")
        
        # Extract timestamp if configured
        timestamp = None
        if 'timestamp_path' in topic_config:
            try:
                timestamp = self.extract_nested_value(data, topic_config['timestamp_path'])
            except ValueError as e:
                raise ValueError(f"Failed to extract timestamp: {e}")
        
        # Extract channels
        channels = {}
        for channel_name, channel_path in topic_config['channel_paths'].items():
            try:
                channels[channel_name] = self.extract_nested_value(data, channel_path)
            except ValueError as e:
                raise ValueError(f"Failed to extract channel '{channel_name}': {e}")
        
        # Build output message
        output_message = {
            "channels": channels,
            "source_topic": topic_config['topic']
        }
        
        if timestamp:
            output_message["timestamp"] = timestamp
        
        return output_message
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """
        Callback for when MQTT client connects to the broker.
        """
        if rc == 0:
            logging.info("Connected to MQTT broker successfully!")
            
            # Subscribe only to configured topics
            for topic in self.topic_configs.keys():
                result = client.subscribe(topic)
                logging.info(f"Subscribed to configured topic: {topic} - Result: {result}")
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
            mqtt_topic = msg.topic
            message = msg.payload.decode('utf-8')
            
            # Only process configured topics
            if mqtt_topic not in self.topic_configs:
                logging.debug(f"Skipping non-configured topic: {mqtt_topic}")
                return
            
            topic_config = self.topic_configs[mqtt_topic]
            
            logging.info(f"Processing configured topic '{mqtt_topic}'")
            logging.debug(f"Raw message: {message}'")
            
            try:
                parsed_message = self.parse_mqtt_message(message, topic_config)
            except ValueError as e:
                logging.error(f"Failed to parse message from topic '{mqtt_topic}': {e}")
                logging.error(f"Raw message: {message}")
                raise
            
            # Convert to Kafka topic name
            kafka_topic = self.sanitize_topic_name(mqtt_topic)
            
            # Send to Kafka
            kafka_message = json.dumps(parsed_message)
            record_metadata = self.send_to_kafka(kafka_topic, kafka_message)
            
            logging.info(f'Successfully processed and sent to Kafka topic "{kafka_topic}" - partition {record_metadata.partition}, offset {record_metadata.offset}')
            logging.debug(f'Kafka message: {kafka_message}')
            
        except Exception as e:
            logging.error(f"Critical error processing message from topic {mqtt_topic}: {e}")
            # For configured topics, we want to error out rather than continue
            raise
    
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
            logging.info("Starting configurable MQTT-to-Kafka bridge")
            
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
    logging.info("Starting configurable MQTT-to-Kafka bridge")
    
    producer = MQTTKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()