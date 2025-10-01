from abc import ABC, abstractmethod
from kafka import KafkaConsumer
import logging
import time
import re
import json
from datetime import datetime
from typing import Union

from smocs.utils import validate_topic_format, validate_message_format

class KafkaConsumerBase(ABC):
    """
    Abstract base class for Kafka consumers.
    
    This class provides common Kafka consumer functionality including:
    - Kafka consumer setup with sensible defaults
    - Main consumption loop with error handling
    - Message and topic format validation
    - Resource cleanup
    
    Subclasses must implement the process_message() method to define their specific
    message processing logic.
    """
    
    def __init__(self, kafka_broker_url, group_id, topics_or_pattern):
        """
        Initialize the base Kafka consumer.
        
        Args:
            kafka_broker_url (str): Kafka broker URL
            group_id (str): Consumer group ID
            topics_or_pattern (str or list): Either a regex pattern string or list of topic names
        """
        self.kafka_broker_url = kafka_broker_url
        self.group_id = group_id
        self.topics_or_pattern = topics_or_pattern
        self.consumer = None
        self.running = False
        
        logging.info(f"Kafka consumer initialized - Broker: {kafka_broker_url}, Group: {group_id}")
        
        # Determine if we're using pattern or specific topics
        self.is_pattern = isinstance(topics_or_pattern, str)
        if self.is_pattern:
            logging.info(f"Will subscribe to topic pattern: {topics_or_pattern}")
        else:
            logging.info(f"Will subscribe to topics: {topics_or_pattern}")
    
    def setup_kafka_consumer(self):
        """
        Set up Kafka consumer with sensible defaults.
        
        Raises:
            Exception: If Kafka consumer setup fails
        """
        consumer_config = {
            'bootstrap_servers': [self.kafka_broker_url],
            'group_id': self.group_id,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'value_deserializer': lambda m: m.decode('utf-8') if m else None
        }
        
        # Allow subclasses to customize configuration
        custom_config = self.get_consumer_config()
        if custom_config:
            consumer_config.update(custom_config)
        
        # Retry connection setup
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.consumer = KafkaConsumer(**consumer_config)
                logging.info("Kafka consumer connected successfully")
                return
            except Exception as e:
                logging.error(f"Failed to connect to Kafka (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def get_consumer_config(self):
        """
        Override this method to provide custom consumer configuration.
        
        Returns:
            dict: Additional consumer configuration options
        """
        return {}
    
    def subscribe_to_topics(self):
        """
        Subscribe to topics or topic pattern.
        """
        if self.is_pattern:
            # Subscribe using pattern
            pattern = re.compile(self.topics_or_pattern)
            self.consumer.subscribe(pattern=pattern)
            logging.info(f"Subscribed to topic pattern: {self.topics_or_pattern}")
        else:
            # Subscribe to specific topics
            topics = self.topics_or_pattern if isinstance(self.topics_or_pattern, list) else [self.topics_or_pattern]
            self.consumer.subscribe(topics)
            logging.info(f"Subscribed to topics: {topics}")
    
    def start(self):
        """
        Start the consumer and begin processing messages.
        
        This method:
        1. Sets up the Kafka consumer
        2. Subscribes to topics
        3. Starts the main consumption loop
        """
        try:
            logging.info("Starting Kafka consumer...")
            
            # Setup consumer
            self.setup_kafka_consumer()
            
            # Subscribe to topics
            self.subscribe_to_topics()
            
            # Start consuming
            self.running = True
            self.consume_messages()
            
        except KeyboardInterrupt:
            logging.info("Received shutdown signal...")
            self.stop()
        except Exception as e:
            logging.error(f"Error in consumer: {e}")
            self.cleanup()
            raise
    
    def consume_messages(self):
        """
        Main consumption loop that polls for messages, validates them, and processes them.
        Invalid messages are logged but skipped to avoid blocking processing.
        """
        logging.info("Starting message consumption loop...")
        
        while self.running:
            try:
                # Poll for messages with timeout
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process each message in the batch
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Validate topic format
                            try:
                                validate_topic_format(message.topic)
                            except ValueError as e:
                                logging.error(f"Invalid topic format from {message.topic}:{message.partition}:{message.offset}: {e}")
                                continue
                            
                            # Validate message format
                            try:
                                validate_message_format(message.value)
                            except ValueError as e:
                                logging.error(f"Invalid message format from {message.topic}:{message.partition}:{message.offset}: {e}")
                                logging.debug(f"Invalid message content: {message.value}")
                                continue
                            
                            # Process the validated message
                            success = self.process_message(
                                message=message.value,
                                topic=message.topic,
                                partition=message.partition,
                                offset=message.offset
                            )
                            
                            if not success:
                                logging.warning(f"Message processing failed for topic {message.topic}, offset {message.offset}")
                            
                        except Exception as e:
                            logging.error(f"Error processing message from topic {message.topic}: {e}")
                            self.handle_processing_error(e, message)
                
            except Exception as e:
                logging.error(f"Error in consumption loop: {e}")
                # Brief pause before retrying
                time.sleep(1)
    
    def handle_processing_error(self, exception, message):
        """
        Handle errors that occur during message processing.
        
        Args:
            exception (Exception): The exception that occurred
            message: The Kafka message that caused the error
        """
        logging.error(f"Processing error for message {message.topic}:{message.partition}:{message.offset} - {exception}")
        # Default behavior: log and continue
        # Subclasses can override for custom error handling
    
    @abstractmethod
    def process_message(self, message, topic, partition, offset):
        """
        Process a single message. Must be implemented by subclasses.
        
        Args:
            message (str): The message value (already deserialized and validated)
            topic (str): The topic name (already validated)
            partition (int): The partition number
            offset (int): The message offset
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        pass
    
    def stop(self):
        """
        Signal the consumer to stop processing messages.
        """
        logging.info("Stopping Kafka consumer...")
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up Kafka consumer resources.
        
        This method should be called by subclasses in their cleanup methods.
        """
        if self.consumer:
            try:
                self.consumer.close()
                logging.info("Kafka consumer closed")
            except Exception as e:
                logging.error(f"Error closing Kafka consumer: {e}")