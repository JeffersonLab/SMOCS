from abc import ABC, abstractmethod
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import logging
import json
import re
from datetime import datetime
from typing import Union


class KafkaProducerBase(ABC):
    """
    Abstract base class for Kafka producers.
    
    This class provides common Kafka functionality including:
    - Kafka producer and admin client setup
    - Topic creation
    - Message and topic format validation
    - Resource cleanup
    
    Subclasses must implement the start() method to define their specific
    data source integration (MQTT, HTTP, file, etc.).
    """
    
    def __init__(self, kafka_broker_url='kafka-broker:9092'):
        """
        Initialize the base Kafka producer.
        
        Args:
            kafka_broker_url (str): Kafka broker URL
        """
        self.kafka_broker_url = kafka_broker_url
        self.kafka_producer = None
        self.kafka_admin = None
        self.created_topics = set()  # Track created topics
        
        logging.info(f"Kafka broker URL: {self.kafka_broker_url}")
    
    def validate_topic_format(self, topic: str) -> bool:
        """
        Validate topic follows the required hierarchical format.
        
        Args:
            topic (str): Topic name to validate
            
        Returns:
            bool: True if topic format is valid
            
        Raises:
            ValueError: If topic format is invalid
        """
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"Topic must be a non-empty string, got: {type(topic)}")
        
        return True
    
    def validate_message_format(self, message: Union[str, bytes]) -> bool:
        """
        Validate message has required timestamp and channels structure.
        Expected format: {"timestamp": "2025-01-XX", "channels": {...}}
        
        Args:
            message: Message to validate (string or bytes)
            
        Returns:
            bool: True if message format is valid
            
        Raises:
            ValueError: If message format is invalid
        """
        try:
            # Convert bytes to string if necessary
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            # Parse JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                raise ValueError(f"Message is not valid JSON: {e}")
            
            # Check if data is a dictionary
            if not isinstance(data, dict):
                raise ValueError("Message must be a JSON object")
            
            # Check for required timestamp field
            if 'timestamp' not in data:
                raise ValueError("Message must contain 'timestamp' field")
            
            # Validate timestamp can be parsed (flexible format)
            timestamp = data['timestamp']
            if timestamp is not None:  # Allow None timestamps
                try:
                    # Try various common timestamp formats
                    if isinstance(timestamp, (int, float)):
                        # Unix timestamp
                        datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        # Try ISO format first, then other common formats
                        try:
                            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            # Try other formats
                            from dateutil import parser
                            parser.parse(timestamp)
                    else:
                        raise ValueError(f"Timestamp must be string, number, or null, got: {type(timestamp)}")
                except (ValueError, OverflowError) as e:
                    raise ValueError(f"Invalid timestamp format: {e}")
            
            # Check for required channels field
            if 'channels' not in data:
                raise ValueError("Message must contain 'channels' field")
            
            # Validate channels is a dictionary (content can be anything)
            if not isinstance(data['channels'], dict):
                raise ValueError("'channels' field must be a JSON object")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            else:
                raise ValueError(f"Message validation failed: {e}")
    
    def setup_kafka_producer(self):
        """
        Set up Kafka producer and admin client with robust configuration.
        
        Raises:
            Exception: If Kafka connection fails
        """
        try:
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
            
            logging.info("Kafka producer and admin client connected successfully")
            
        except Exception as e:
            logging.error(f"Failed to setup Kafka producer: {e}")
            raise
    
    def create_topic_if_not_exists(self, topic_name):
        """
        Create Kafka topic if it doesn't exist.
        
        Args:
            topic_name (str): Name of the topic to create
        """
        if topic_name in self.created_topics:
            return
            
        try:
            # Check if topic exists
            existing_topics = self.kafka_admin.list_topics()
            if topic_name in existing_topics:
                self.created_topics.add(topic_name)
                logging.debug(f"Topic {topic_name} already exists")
                return
                
            # Create topic
            topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
            self.kafka_admin.create_topics([topic])
            self.created_topics.add(topic_name)
            logging.info(f"Created Kafka topic: {topic_name}")
            
        except Exception as e:
            logging.warning(f"Could not create topic {topic_name}: {e}")
            
    def sanitize_topic_name(self, topics):
        """
        Convert generic topic to valid Kafka topic name.
        
        Args:
            topic (str): Original Source topic name
            
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
    
    def send_to_kafka(self, topic_name, message, key=None):
        """
        Send a message to Kafka topic with validation.
        
        Args:
            topic_name (str): Kafka topic name
            message (str or bytes): Message to send
            key (str or bytes, optional): Message key
            
        Returns:
            RecordMetadata: Metadata about the sent record
            
        Raises:
            ValueError: If topic or message format is invalid
            Exception: If sending fails
        """
        try:
            # Validate topic format
            self.validate_topic_format(topic_name)
            
            # Validate message format
            self.validate_message_format(message)
            
            # Ensure topic exists
            self.create_topic_if_not_exists(topic_name)
            
            # Convert message to bytes if it's a string
            if isinstance(message, str):
                message = message.encode('utf-8')
            
            # Convert key to bytes if provided and is a string
            if key is not None and isinstance(key, str):
                key = key.encode('utf-8')
            
            # Send message
            future = self.kafka_producer.send(topic_name, value=message, key=key)
            record_metadata = future.get(timeout=10)
            
            logging.debug(f'Message sent to Kafka topic "{topic_name}" - partition {record_metadata.partition}, offset {record_metadata.offset}')
            
            return record_metadata
            
        except ValueError as e:
            logging.error(f"Validation failed for topic '{topic_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"Error sending message to Kafka topic {topic_name}: {e}")
            raise
    
    def start(self):
        """
        Default start implementation that just sets up the producer.
        Subclasses can override this if they need custom startup logic.
        """
        logging.info("Starting Kafka producer...")
        self.setup_kafka_producer()
        logging.info("Kafka producer ready")
    
    def cleanup(self):
        """
        Clean up Kafka resources.
        
        This method should be called by subclasses in their cleanup methods.
        """
        if self.kafka_producer:
            try:
                self.kafka_producer.flush()
                self.kafka_producer.close()
                logging.info("Kafka producer closed")
            except Exception as e:
                logging.error(f"Error closing Kafka producer: {e}")
        
        if self.kafka_admin:
            try:
                self.kafka_admin.close()
                logging.info("Kafka admin client closed")
            except Exception as e:
                logging.error(f"Error closing Kafka admin client: {e}")