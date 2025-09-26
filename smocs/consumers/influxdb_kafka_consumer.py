import os
import logging
import json
import ast
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from smocs.cores import KafkaConsumerBase
from smocs.utils import ChannelFilter, setup_logging

class InfluxDBConsumer(KafkaConsumerBase):
    """
    InfluxDB consumer that inherits from KafkaConsumerBase.
    
    This consumer subscribes to all Kafka topics and writes messages
    with the 'blinky-mqtt' type to InfluxDB.
    """
    
    def __init__(self):
        """
        Initialize the InfluxDB consumer.
        Uses environment variables for configuration.
        """
        # Kafka configuration from environment
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = 'grafana-consumer'
        topics_pattern = r'.*'  # Subscribe to all topics
        
        # Initialize base class
        super().__init__(kafka_broker_url, group_id, topics_pattern)
        
        # InfluxDB configuration from environment
        self.influxdb_url = os.environ.get('INFLUXDB_URL', 'http://influxdb:8086')
        self.influxdb_token = os.environ.get('INFLUXDB_TOKEN')
        self.influxdb_org = os.environ.get('INFLUXDB_ORG', 'myorg')
        self.influxdb_bucket = os.environ.get('INFLUXDB_BUCKET', 'kafka_data')
        
        # InfluxDB client
        self.influxdb_client = None
        self.write_api = None
        
        logging.info(f"InfluxDB URL: {self.influxdb_url}")
        logging.info(f"InfluxDB Bucket: {self.influxdb_bucket}")
        
        # Setup InfluxDB client
        self.setup_influxdb_client()
    
    def setup_influxdb_client(self):
        """
        Set up InfluxDB client and write API.
        """
        try:
            self.influxdb_client = InfluxDBClient(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            self.write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            logging.info("InfluxDB client connected successfully")
        except Exception as e:
            logging.error(f"Failed to setup InfluxDB client: {e}")
            raise
    
    def process_message(self, message, topic, partition, offset):
        """
        Process a single Kafka message and write to InfluxDB.
        
        Args:
            message (str): The message value
            topic (str): The topic name
            partition (int): The partition number
            offset (int): The message offset
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            logging.debug(f"Processing message from topic {topic}: {message}")
            
            # Skip test messages
            if message and message.startswith("Message"):
                logging.debug("Skipping test message")
                return True
            
            # Parse message data
            parsed_data = self.parse_message_data(message)
            if not parsed_data:
                return False
            
            # Create InfluxDB point
            point = self.create_influx_point(parsed_data, topic)
            if not point:
                return False
            
            # # Write to InfluxDB
            self.write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=point)
            logging.info(f"Successfully wrote {topic} with data {parsed_data} to InfluxDB bucket: {self.influxdb_bucket}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing message from topic {topic}: {e}")
            return False
    
    def parse_message_data(self, message):
        """
        Parse message data from string to dictionary.
        
        Args:
            message (str): Raw message string
            
        Returns:
            dict or None: Parsed message data or None if parsing failed
        """
        try:
            if not message:
                return None
            
            # Handle different message formats
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            elif isinstance(message, str):
                # Try to parse as JSON first, then as Python literal
                try:
                    return json.loads(message)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(message)
                    except (ValueError, SyntaxError):
                        logging.warning(f"Failed to parse message: {message[:100]}...")
                        return None
            else:
                logging.warning(f"Received message of unexpected type: {type(message)}")
                return None
                
        except Exception as e:
            logging.error(f"Error parsing message data: {e}")
            return None

    def create_influx_point(self, message, topic):
        """
        Create an InfluxDB point from message data using channel filtering.
        
        Args:
            message (dict): Parsed message data
            topic (str): Kafka topic name
            
        Returns:
            Point or None: InfluxDB point or None if creation failed
        """
        try:
            # Use ChannelFilter to extract all numeric channels
            filtered_result = ChannelFilter.extract_all_channels(message)
            
            if filtered_result is None:
                logging.warning(f"No valid numeric channels found for topic {topic}")
                return None
            
            channel_names, channel_values = filtered_result
            
            # Create InfluxDB point
            point = Point(topic)
            
            # Add all numeric fields to the point
            for name, value in zip(channel_names, channel_values):
                point.field(name, value)
            
            # Optionally use the original timestamp
            # if 'timestamp' in message:
            #     point.time(message['timestamp'])
            
            logging.debug(f"Created InfluxDB point with {len(channel_values)} fields for topic {topic}")
            return point
            
        except Exception as e:
            logging.error(f"Error creating InfluxDB point: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up InfluxDB and Kafka resources.
        """
        # Close InfluxDB client
        if self.influxdb_client:
            try:
                self.influxdb_client.close()
                logging.info("InfluxDB client closed")
            except Exception as e:
                logging.error(f"Error closing InfluxDB client: {e}")
        
        # Call base class cleanup for Kafka resources
        super().cleanup()


def main():
    """
    Main entry point for the InfluxDB consumer.
    """
    setup_logging()
    logging.info("Starting InfluxDB consumer...")
    
    consumer = InfluxDBConsumer()
    consumer.start()


if __name__ == "__main__":
    main()