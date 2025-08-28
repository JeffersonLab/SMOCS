import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from smocs.cores import KafkaConsumerBase
from smocs.db.mysql_api_v0 import DBManager

logging.basicConfig(level=logging.INFO)

class DataIngestThreadBase(KafkaConsumerBase, ABC):
    """
    Base class for sensor data ingestion thread.
    Inherits from KafkaConsumerBase to consume messages from Kafka.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the data ingest thread.
        
        Args:
            agent_id: Unique identifier for the parent agent
            config: Agent configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        
        # Setup Kafka consumer
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = f"{agent_id}-data-ingest"
        input_topic = config.get('kafka_topics', {}).get('input', 'sensor-data')
        
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        # Setup database connection
        self.db_manager = self._setup_db_connection()
        
        logging.info(f"Data Ingest Thread initialized for agent {agent_id}")
    
    def _setup_db_connection(self) -> DBManager:
        """Setup database connection for this thread."""
        db_config = {
            'agent_id': self.agent_id,
            'host': os.environ.get('MYSQL_HOST', 'localhost'),
            'port': int(os.environ.get('MYSQL_PORT', 3307)),
            'user': os.environ.get('MYSQL_USER', 'root'),
            'pwd': os.environ['MYSQL_ROOT_PASSWORD'],
            'database': os.environ.get('MYSQL_DATABASE', 'agentdb')
        }
        return DBManager(db_config)
    
    def process_message(self, message, topic, partition, offset):
        """
        Process incoming Kafka message and store to database.
        Calls the abstract store_message method.
        
        Args:
            message: The message value
            topic: The topic name  
            partition: The partition number
            offset: The message offset
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            success = self.store_message(message, topic, partition, offset)
            if success:
                logging.debug(f"Successfully processed message from {topic}:{partition}:{offset}")
            return success
        except Exception as e:
            logging.error(f"Error in store_message: {e}")
            return False
    
    @abstractmethod
    def store_message(self, message, topic, partition, offset) -> bool:
        """
        Store the processed message data to the database.
        Must be implemented by subclasses.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number  
            offset: The message offset
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()
        super().cleanup()