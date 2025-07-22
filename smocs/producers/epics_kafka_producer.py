import epics
print(f"find libca: {epics.ca.find_libca()}")
epics.ca.initialize_libca()
import os
import logging
import sys
import yaml
from pathlib import Path
import time

from smocs.cores import KafkaProducerBase

logging.basicConfig(level=logging.INFO)


class EpicsKafkaProducer(KafkaProducerBase):
    """
    EPICS to Kafka producer that inherits from KafkaProducerBase.
    
    This producer connects to an EPICS broker, subscribes to specified topics,
    and forwards all received messages to corresponding Kafka topics.
    """
    
    def __init__(self):
        """
        Initialize the producer.
        Uses environment variables for configuration.
        """
        
        # Get Kafka broker URL from environment
        kafka_broker_url = os.getenv('KAFKA_BROKER_URL', 'kafka-broker:9092')
        # Initialize base class
        super().__init__(kafka_broker_url)
        
        config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')
        try:
            with open(config_path, 'rb') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            logging.error("EPICS Kafka Producer: error in reading config file")
            print(e)
            sys.exit(1)
        
        self.sensors = config['epics']['sensors']
        self.sensors_pv_objects = {}
        for sensor in self.sensors:
            self.sensors_pv_objects[sensor] = []
            for pv in self.sensors[sensor]:
                self.sensors_pv_objects[sensor].append(epics.PV(pv))
        
        logging.info(f"EPICS: {os.environ['EPICS_CA_ADDR_LIST']}")
        logging.info(f"EPICS Topics: {self.topics}")
        
    
    def sanitize_topic_name(self, topics):
        """
        Convert MQTT topic to valid Kafka topic name.
        
        Args:
            mqtt_topic (str): Original MQTT topic name
            
        Returns:
            str: Sanitized Kafka topic name
        """
        # Replace forward slashes with dots and remove/replace invalid characters
        kafka_topic = topics.replace('/', '.').replace('#', 'wildcard').replace('+', 'plus')
        # Remove any other invalid characters and ensure it starts with alphanumeric
        kafka_topic = ''.join(c if c.isalnum() or c in '.-_' else '_' for c in kafka_topic)
        # No starting with a dot or dash
        if kafka_topic.startswith('.') or kafka_topic.startswith('-'):
            kafka_topic = 'topic_' + kafka_topic
            
        return kafka_topic
    
    
    def start(self):
        """
        Start the EPICS to Kafka producer.
        
        This method:
        1. Sets up Kafka producer (via base class)
        2. Configures epics sensor/pv information
        4. Starts the main processing loop
        """
        try:
            logging.info("Starting EPICS-to-Kafka bridge with set Sensor/PVs")
            
            # Setup Kafka producer using base class method
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            # Start MQTT loop
            logging.info("Starting EPICS loop...")
            while True:
                time.sleep(1)
                for sensor in self.sensors_pv_objects:
                    topic = sensor
                    pv_list = self.sensors[sensor]
                    channels = {pv_list[i]:self.sensors_pv_objects[i].get() for i in range(len(pv_list))}
                    timestamp = self.sensors_pv_objects[0].timestamp
                    message = {'timestamp': timestamp, 'channels':channels, 'source_topic':topic}
                    
                    # Convert EPICS topic to valid Kafka topic name
                    kafka_topic = self.sanitize_topic_name(mqtt_topic)
                    
                    logging.info(f"EPICS received from '{topic}' timestamp {timestamp}: ", channels)
                    
                    # Send to Kafka using base class method
                    record_metadata = self.send_to_kafka(kafka_topic, message)
                    
                    logging.info(f'Forwarded to Kafka topic "{kafka_topic}" (from EPICS "{topic}") - partition {record_metadata.partition}, offset {record_metadata.offset}')
                    
            
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
        
        try:
            for sensor in self.sensors_pv_objects:
                for pv in self.sensors_pv_objects[sensor]:
                    pv.disconnect()
            logging.info("EPICS PVs disconnected")
        except Exception as e:
            logging.error(f"Error disconnecting EPICS: {e}")
        
        # Call base class cleanup for Kafka resources
        super().cleanup()


def main():
    """
    Main entry point for the EPICS to Kafka producer.
    """
    
    logging.info("Starting EPICS-to-Kafka bridge with topic preservation")
    
    producer = EpicsKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()