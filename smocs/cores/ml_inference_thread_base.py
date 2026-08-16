import time
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

from smocs.cores import KafkaStreamingProcessBase
from smocs.db.mysql_api_v0 import DBManager
from smocs.utils import ChannelFilter

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
        self.switch_fn = config['switch_function'] if 'switch_function' in config else None
        
        # Setup channel filter if configured
        input_channels = config.get('model_input', {}).get('channels')
        self.channel_filter = ChannelFilter(input_channels) if input_channels else None
        
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
        
        logging.info(f"MLInfernceThread: ML Inference Thread initialized for agent {agent_id}")
        logging.info(f"MLInfernceThread: Channel filter: {'enabled' if self.channel_filter else 'disabled'}")
        if self.channel_filter:
            logging.info(f"MLInfernceThread: Required channels: {self.channel_filter.get_required_channels()}")
    
    def _setup_db_connection(self) -> DBManager:
        """Setup database connection for this thread."""
        model_input = self.config.get('model_input', {})
        db_config = {
            'agent_id': self.agent_id,
            'host': os.environ.get('MYSQL_HOST', 'localhost'),
            'port': int(os.environ.get('MYSQL_PORT', 3307)),
            'user': os.environ.get('MYSQL_USER', 'root'),
            'pwd': os.environ['MYSQL_ROOT_PASSWORD'],
            'context_cols': model_input.get('context_channels', []),
            'max_gap_seconds': self.config.get('max_gap_seconds', float('inf')),
        }
        # Schema (including any context_cols columns) is already fully established
        # by the owning agent's _ensure_sensor_schema before this thread is ever
        # constructed - see AgentBase._ensure_sensor_schema's docstring - so this
        # connection has no need to call create_tables() itself.
        return DBManager(db_config)
    
    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process incoming message with optional channel filtering and return inference results.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            Tuple[bool, List[Tuple]]: Success status and list of outputs to send
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            message_data = json.loads(message)
            
            # Apply channel filtering if configured
            if self.channel_filter:
                filtered_result = self.channel_filter.filter_channels(message_data)
                if filtered_result is None:
                    # Skip message due to missing/invalid channels
                    logging.debug(f"MLInfernceThread: Skipping message from {topic}:{partition}:{offset} due to channel filtering")
                    return True, []  # Continue processing but don't send outputs
                
                # Replace the channels in the message with filtered ones
                channel_names, channel_values = filtered_result
                message_data['channels'] = dict(zip(channel_names, channel_values))
                logging.debug(f"MLInfernceThread: Applied channel filtering, extracted {len(channel_values)} channels")
            
            # Parse inference request with filtered data
            inference_request = self.parse_inference_request(message_data, topic, partition, offset)
            
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
            
        except json.JSONDecodeError as e:
            logging.error(f"MLInfernceThread: JSON decode error for message from {topic}:{partition}:{offset}: {e}")
            return False, []
        except Exception as e:
            logging.error(f"MLInfernceThread: Error processing inference message: {e}")
            return False, []
    
    @abstractmethod
    def _store_inference_result(self, inference_request: Any, inference_result: Any):
        """Store inference result to database."""
        try:
            # This would use DBManager to store the inference result
            # Implementation depends on specific data structure
            pass
        except Exception as e:
            logging.error(f"MLInfernceThread: Error storing inference result: {e}")
    
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