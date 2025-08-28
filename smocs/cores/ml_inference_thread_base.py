import time
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

from smocs.cores import KafkaStreamingProcessBase
from smocs.db.mysql_api_v0 import DBManager

logging.basicConfig(level=logging.INFO)

class MLInferenceThreadBase(KafkaStreamingProcessBase, ABC):
    """
    Base class for ML inference thread.
    Inherits from KafkaStreamingProcessBase to consume and produce Kafka messages.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the ML inference thread.
        
        Args:
            agent_id: Unique identifier for the parent agent
            config: Agent configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        
        # Setup Kafka streaming
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = f"{agent_id}-ml-inference"
        input_topic = config.get('kafka_topics', {}).get('input', 'sensor-data')
        self.output_topic = config.get('kafka_topics', {}).get('output', 'inference-results')
        
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        # Setup database connection
        self.db_manager = self._setup_db_connection()
        
        # Load initial model
        self.load_model()
        
        logging.info(f"ML Inference Thread initialized for agent {agent_id}")
    
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
    
    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process incoming message and return inference results.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            Tuple[bool, List[Tuple]]: Success status and list of outputs to send
        """
        try:
            # Parse inference request
            inference_request = self.parse_inference_request(message, topic, partition, offset)
            
            if inference_request is None:
                return False, []
            
            # Perform inference
            inference_result = self.perform_inference(inference_request)
            
            if inference_result is None:
                return False, []
            
            # Store inference result to database
            self._store_inference_result(inference_request, inference_result)
            
            # Format result for Kafka
            output_message = {
                'agent_id': self.agent_id,
                'timestamp': time.time(),
                'inference_result': inference_result,
                'original_message': message
            }
            
            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
            
        except Exception as e:
            logging.error(f"Error processing inference message: {e}")
            return False, []
    
    def _store_inference_result(self, inference_request: Any, inference_result: Any):
        """Store inference result to database."""
        try:
            # This would use DBManager to store the inference result
            # Implementation depends on specific data structure
            pass
        except Exception as e:
            logging.error(f"Error storing inference result: {e}")
    
    @abstractmethod
    def load_model(self):
        """Load the latest model from database."""
        pass
    
    @abstractmethod
    def parse_inference_request(self, message, topic, partition, offset) -> Optional[Any]:
        """
        Parse incoming message into inference request.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            Parsed inference request or None if parsing failed
        """
        pass
    
    @abstractmethod
    def perform_inference(self, inference_request: Any) -> Optional[Any]:
        """
        Perform inference on the request.
        
        Args:
            inference_request: Parsed inference request
            
        Returns:
            Inference result or None if inference failed
        """
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()
        super().cleanup()