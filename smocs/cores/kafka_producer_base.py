from abc import ABC
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import logging

from smocs.utils import validate_topic_format, validate_message_format

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
            validate_topic_format(topic_name)
            
            # Validate message format
            validate_message_format(message)
            
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