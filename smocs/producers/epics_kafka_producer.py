import epics
epics.ca.initialize_libca()
import os
import logging
import sys
import yaml
import json
from pathlib import Path
import time
import threading

from smocs.cores import KafkaProducerBase
from smocs.utils import setup_logging, EpicsCLIController


class EpicsKafkaProducer(KafkaProducerBase):
    """
    EPICS to Kafka producer that inherits from KafkaProducerBase.
    
    This producer connects to an EPICS broker, subscribes to specified PVs,
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
        
        self.pv_list = config['epics']['PVs'].copy()
        self.source = config['epics']['source']
        
        logging.info(f"EPICS: {os.environ.get('EPICS_CA_ADDR_LIST', 'Not set')}")
        logging.info(f"EPICS reading pvs: {self.pv_list}")
        logging.info(f"EPICS source/topic: {self.source}")
        
        # CLI Control setup
        if os.getenv('ENABLE_CLI_CONTROL', 'false').lower() == 'true':
            logging.info("CLI control is enabled, initializing...")
            self.pv_lock = threading.Lock()
            self.cli_controller = EpicsCLIController(self)
            self._register_cli_commands()
        else:
            logging.info("CLI control is disabled")
            self.cli_controller = None
            self.pv_lock = None
    
    def _register_cli_commands(self):
        """Register CLI commands with the controller."""
        self.cli_controller.register_command(
            'add_pv',
            self.cli_add_pv,
            'Add a PV to monitor'
        )
        self.cli_controller.register_command(
            'remove_pv',
            self.cli_remove_pv,
            'Remove a PV from monitoring'
        )
        self.cli_controller.register_command(
            'list_pvs',
            self.cli_list_pvs,
            'List all monitored PVs'
        )
        self.cli_controller.register_command(
            'set_source',
            self.cli_set_source,
            'Change Kafka topic name'
        )
        self.cli_controller.register_command(
            'status',
            self.cli_status,
            'Show current producer status'
        )
        logging.info("CLI commands registered")
    
    def cli_add_pv(self, params):
        """Add a PV to the monitoring list."""
        pv = params.get('pv')
        
        if not pv:
            return {
                'status': 'error',
                'message': 'No PV specified'
            }
        
        with self.pv_lock:
            if pv in self.pv_list:
                return {
                    'status': 'ok',
                    'message': f'PV {pv} already exists',
                    'pvs': self.pv_list.copy(),
                    'source': self.source
                }
            
            self.pv_list.append(pv)
            logging.info(f"CLI: Added PV {pv}")
            
            return {
                'status': 'ok',
                'message': f'Added PV {pv}',
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def cli_remove_pv(self, params):
        """Remove a PV from the monitoring list."""
        pv = params.get('pv')
        
        if not pv:
            return {
                'status': 'error',
                'message': 'No PV specified'
            }
        
        with self.pv_lock:
            if pv not in self.pv_list:
                return {
                    'status': 'ok',
                    'message': f'PV {pv} does not exist',
                    'pvs': self.pv_list.copy(),
                    'source': self.source
                }
            
            self.pv_list.remove(pv)
            logging.info(f"CLI: Removed PV {pv}")
            
            return {
                'status': 'ok',
                'message': f'Removed PV {pv}',
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def cli_list_pvs(self, params):
        """List all monitored PVs."""
        with self.pv_lock:
            return {
                'status': 'ok',
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def cli_set_source(self, params):
        """Change the Kafka topic name."""
        new_source = params.get('source')
        
        if not new_source:
            return {
                'status': 'error',
                'message': 'No source specified'
            }
        
        with self.pv_lock:
            old_source = self.source
            self.source = new_source
            logging.info(f"CLI: Changed source from {old_source} to {new_source}")
            
            return {
                'status': 'ok',
                'message': f'Source changed from {old_source} to {new_source}',
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def cli_status(self, params):
        """Get current producer status."""
        with self.pv_lock:
            return {
                'status': 'ok',
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def get_state(self):
        """Get current state (thread-safe)."""
        if self.pv_lock:
            with self.pv_lock:
                return {
                    'pvs': self.pv_list.copy(),
                    'source': self.source
                }
        else:
            return {
                'pvs': self.pv_list.copy(),
                'source': self.source
            }
    
    def start(self):
        """
        Start the EPICS to Kafka producer.
        
        This method:
        1. Sets up Kafka producer (via base class)
        2. Starts CLI controller if enabled
        3. Starts the main EPICS polling loop
        """
        try:
            logging.info("Starting EPICS-to-Kafka bridge")
            
            # Setup Kafka producer using base class method
            logging.info("Setting up Kafka producer...")
            self.setup_kafka_producer()
            
            # Start CLI controller if enabled
            if self.cli_controller:
                logging.info("Starting CLI controller...")
                self.cli_controller.start()  # Fatal if fails
            
            # Start EPICS loop
            logging.info("Starting EPICS polling loop...")
            while True:
                time.sleep(1)
                
                # Thread-safe copy of PV list
                if self.pv_lock:
                    with self.pv_lock:
                        pv_list_copy = self.pv_list.copy()
                        source_copy = self.source
                else:
                    pv_list_copy = self.pv_list.copy()
                    source_copy = self.source
                
                # Skip if no PVs to monitor
                if not pv_list_copy:
                    logging.debug("No PVs to monitor, skipping...")
                    continue
                
                # Get all PV values
                all_pv_values = epics.caget_many(pv_list_copy)
                channels = {pv_list_copy[i]: all_pv_values[i] for i in range(len(pv_list_copy))}
                timestamp = time.time()
                
                message = {
                    'timestamp': timestamp,
                    'channels': channels,
                    'source_topic': source_copy
                }
                message_json = json.dumps(message)
                
                # Convert EPICS topic to valid Kafka topic name
                kafka_topic = self.sanitize_topic_name(source_copy)
                
                logging.debug(f"EPICS received from '{source_copy}' timestamp {timestamp}: {channels}")
                
                # Send to Kafka using base class method
                record_metadata = self.send_to_kafka(kafka_topic, message_json)
                
                logging.info(f'Forwarded to Kafka topic "{kafka_topic}" - timestamp {timestamp} - partition {record_metadata.partition}, offset {record_metadata.offset}')
                    
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error: {e}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """Clean up resources."""
        if self.cli_controller:
            self.cli_controller.stop()
        super().cleanup()


def main():
    """
    Main entry point for the EPICS to Kafka producer.
    """
    setup_logging()
    logging.info("Starting EPICS-to-Kafka bridge")
    
    producer = EpicsKafkaProducer()
    producer.start()


if __name__ == "__main__":
    main()