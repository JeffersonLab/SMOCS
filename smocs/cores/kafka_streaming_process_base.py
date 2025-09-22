from abc import ABC, abstractmethod
import logging
import time
from typing import List, Tuple
from smocs.cores import KafkaConsumerBase, KafkaProducerBase
from smocs.utils import validate_topic_format, validate_message_format

class KafkaStreamingProcessBase(KafkaConsumerBase):
    """
    Abstract base class for Kafka stream processors.
    
    This class extends KafkaConsumerBase to add producer functionality,
    enabling bidirectional Kafka communication. It consumes messages,
    processes them, and can send results back to Kafka topics.
    
    The stream processor maintains the same single-threaded architecture
    as the consumer but adds the ability to produce messages based on
    processing results. Both input and output messages are validated
    according to the required format.
    
    Uses composition with a KafkaProducerBase instance to provide
    producer functionality without code duplication.
    """
    
    def __init__(self, kafka_broker_url, group_id, topics_or_pattern):
        """
        Initialize the stream processor.
        
        Args:
            kafka_broker_url (str): Kafka broker URL
            group_id (str): Consumer group ID
            topics_or_pattern (str or list): Either a regex pattern string or list of topic names
        """
        # Initialize consumer base (which includes validation methods)
        super().__init__(kafka_broker_url, group_id, topics_or_pattern)
        
        # Direct composition with KafkaProducerBase (which also includes validation methods)
        self.producer = KafkaProducerBase(kafka_broker_url)
        
        logging.info("Kafka stream processor initialized with validation-enabled consumer and producer")
    
    def start(self):
        """
        Start the stream processor.
        
        This method:
        1. Sets up the Kafka consumer
        2. Sets up the Kafka producer (via KafkaProducerBase)
        3. Subscribes to topics
        4. Starts the main processing loop
        """
        try:
            logging.info("Starting Kafka stream processor...")
            
            # Setup consumer (inherited from KafkaConsumerBase)
            self.setup_kafka_consumer()
            
            # Setup producer (KafkaProducerBase.start() calls setup_kafka_producer())
            self.producer.start()
            
            # Subscribe to topics (inherited from KafkaConsumerBase)
            self.subscribe_to_topics()
            
            # Start consuming and processing
            self.running = True
            self.consume_messages()
            
        except KeyboardInterrupt:
            logging.info("Received shutdown signal...")
            self.stop()
        except Exception as e:
            logging.error(f"Error in stream processor: {e}")
            self.cleanup()
            raise
    
    def consume_messages(self):
        """
        Main consumption loop that polls for messages, validates them, processes them,
        validates outputs, and sends results back to Kafka.
        
        This extends the consumer's consume_messages to handle the
        new return format from process_message and use the composed
        KafkaProducerBase for sending messages with validation.
        """
        logging.info("Starting stream processing loop with validation...")
        
        while self.running:
            try:
                # Poll for messages with timeout (inherited from KafkaConsumerBase)
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process each message in the batch
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Validate input topic format
                            try:
                                validate_topic_format(message.topic)
                            except ValueError as e:
                                logging.error(f"Invalid input topic format from {message.topic}:{message.partition}:{message.offset}: {e}")
                                continue
                            
                            # Validate input message format
                            try:
                                validate_message_format(message.value)
                            except ValueError as e:
                                logging.error(f"Invalid input message format from {message.topic}:{message.partition}:{message.offset}: {e}")
                                logging.debug(f"Invalid message content: {message.value}")
                                continue
                            
                            # Call subclass process_message with tuple return format
                            success, outputs = self.process_message(
                                message=message.value,
                                topic=message.topic,
                                partition=message.partition,
                                offset=message.offset
                            )
                            
                            if not success:
                                logging.warning(f"Message processing failed for topic {message.topic}, offset {message.offset}")
                                continue
                            
                            # Send each output message to Kafka with validation
                            if outputs:
                                for output in outputs:
                                    try:
                                        # Unpack tuple: (topic, message) or (topic, message, key)  
                                        if len(output) == 2:
                                            topic, output_message = output
                                            key = None
                                        elif len(output) == 3:
                                            topic, output_message, key = output
                                        else:
                                            raise ValueError(f"Invalid output tuple length: {len(output)}. Expected 2 or 3 elements.")
                                        
                                        # Use composed KafkaProducerBase to send message (includes validation)
                                        record_metadata = self.producer.send_to_kafka(topic, output_message, key)
                                        logging.debug(f"Sent validated output to topic '{topic}' - partition {record_metadata.partition}, offset {record_metadata.offset}")
                                        
                                    except ValueError as e:
                                        # Validation errors from producer
                                        logging.error(f"Output validation failed for topic '{topic if 'topic' in locals() else 'unknown'}': {e}")
                                        logging.debug(f"Invalid output content: {output}")
                                    except Exception as e:
                                        logging.error(f"Failed to send output tuple {output}: {e}")
                            
                        except Exception as e:
                            logging.error(f"Error processing message from topic {message.topic}: {e}")
                            self.handle_processing_error(e, message)
                
            except Exception as e:
                logging.error(f"Error in consumption loop: {e}")
                time.sleep(1)
    
    @abstractmethod
    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process a single message and return results for publishing.
        
        This method must be implemented by subclasses to define their specific
        stream processing logic.
        
        Args:
            message (str): The message value (already deserialized and validated)
            topic (str): The topic name (already validated)
            partition (int): The partition number
            offset (int): The message offset
            
        Returns:
            Tuple[bool, List[Tuple]]: 
                - bool: True if processing was successful, False otherwise
                - List[Tuple]: List of tuples to send to Kafka topics
                  Each tuple can be:
                  - (topic, message) - topic and message only
                  - (topic, message, key) - topic, message, and optional key
                  
                  Note: All output topics and messages will be validated before sending.
        """
        pass
    
    def cleanup(self):
        """
        Clean up both consumer and producer resources.
        
        Uses the KafkaProducerBase instance for producer cleanup
        and calls the parent KafkaConsumerBase cleanup for consumer resources.
        """
        # Cleanup producer using KafkaProducerBase
        if self.producer:
            self.producer.cleanup()
        
        # Call base class cleanup for consumer resources
        super().cleanup()