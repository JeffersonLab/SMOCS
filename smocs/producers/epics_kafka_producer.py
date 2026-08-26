import epics
epics.ca.initialize_libca()
import os
import logging
import sys
import yaml
import json
import time

from smocs.cores import KafkaProducerBase
from smocs.utils import setup_logging

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
            logging.error(f"EPICS Kafka Producer: error in reading config file: {e}")
            sys.exit(1)
        
        self.pv_list = config['epics']['PVs']
        self.source = config['epics']['source']
        
        
        logging.info(f"EPICS: {os.environ['EPICS_CA_ADDR_LIST']}")
        logging.info(f"EPICS reading pvs: {self.pv_list}")
    
    
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
                
                    
                all_pv_values = epics.caget_many(self.pv_list)
                channels = {self.pv_list[i]:all_pv_values[i] for i in range(len(self.pv_list))}
                timestamp = time.time()
                message = {'timestamp': timestamp, 'channels':channels, 'source_topic':self.source}
                message = json.dumps(message)
                # Convert EPICS topic to valid Kafka topic name
                kafka_topic = self.sanitize_topic_name(self.source)
                    
                logging.debug(f"EPICS received from '{self.source}' timestamp {timestamp}: ", channels)
                
                logging.debug(f"Type of kafka message: {type(message)}")
                # Send to Kafka using base class method
                record_metadata = self.send_to_kafka(kafka_topic, message)
                
                logging.info(f'Forwarded to Kafka topic "{kafka_topic}" (from EPICS "{self.source}") - timestamp {timestamp} -channels {channels} - partition {record_metadata.partition}, offset {record_metadata.offset}')
                    
            
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error: {e}")
            self.cleanup()
            raise


def main():
    """
    Main entry point for the EPICS to Kafka producer.
    """
    setup_logging()
    logging.info("Starting EPICS-to-Kafka bridge with topic preservation")
    
    producer = EpicsKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()