import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from smocs.cores import KafkaConsumerBase
from smocs.db.mysql_api_v0 import DBManager
from smocs.utils import ChannelFilter

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
        self.switch_fn = config["switch_function"] if "switch_function" in config else None
        
        # Setup channel filter if configured
        input_channels = config.get('model_input', {}).get('channels')
        self.channel_filter = ChannelFilter(input_channels) if input_channels else None
        
        # Setup Kafka consumer
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = f"{agent_id}-data-ingest"
        input_topic = config.get('kafka_topics', {}).get('input', 'sensor-data')
        
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        # Setup database connection
        self.db_manager = self._setup_db_connection()
        
        logging.info(f"DataIngestThread: Data Ingest Thread initialized for agent {agent_id}")
        logging.info(f"DataIngestThread: Channel filter: {'enabled' if self.channel_filter else 'disabled'}")
        if self.channel_filter:
            logging.info(f"DataIngestThread: Required channels: {self.channel_filter.get_required_channels()}")
    
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
        db_manager = DBManager(db_config)
        # Among the several DBManager instances an agent constructs, this particular
        # instance is the only one that ever calls record_sensor_data() - only the
        # ingest thread writes sensor data; the training and inference threads only
        # ever read. Consequently, this is the one instance whose in-memory
        # "latest row" cache (used to determine each new row's block_id) must be
        # seeded from the database's actual history before any writing begins.
        # Without this call, that cache would instead start out empty, causing the
        # first row written after every process restart to be assigned block_id 0,
        # as though it were the very first row this agent had ever recorded - even
        # though rows from prior runs may already occupy that same block_id, which
        # would corrupt the guarantee that block_id uniquely and consistently
        # identifies a single, contiguous period of operation.
        db_manager.refresh_latest_row_cache()
        return db_manager
    
    def process_message(self, message, topic, partition, offset):
        """
        Process incoming Kafka message with optional channel filtering and store to database.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            message_data = json.loads(message)
            
            # Apply channel filtering or extract all channels
            if self.channel_filter:
                # Use configured channel filtering
                filtered_result = self.channel_filter.filter_channels(message_data)
                if filtered_result is None:
                    logging.debug(f"DataIngestThread: Skipping message from {topic}:{partition}:{offset} due to channel filtering")
                    return True
                
                channel_names, channel_values = filtered_result
            else:
                # Extract all numeric channels when no filter configured
                filtered_result = ChannelFilter.extract_all_channels(message_data)
                if filtered_result is None:
                    logging.debug(f"DataIngestThread: Skipping message from {topic}:{partition}:{offset} - no valid channels")
                    return True
                
                channel_names, channel_values = filtered_result
            
            # Create clean channel dictionary for agent processing
            filtered_channels = dict(zip(channel_names, channel_values))
            message_data['channels'] = filtered_channels
            
            logging.debug(f"DataIngestThread: Extracted {len(channel_values)} channels for processing")
            
            # Call subclass implementation with processed data
            success = self.store_message(message_data, topic, partition, offset)
            if success:
                logging.debug(f"DataIngestThread: Successfully processed message from {topic}:{partition}:{offset}")
            return success
            
        except json.JSONDecodeError as e:
            logging.error(f"DataIngestThread: JSON decode error for message from {topic}:{partition}:{offset}: {e}")
            return False
        except Exception as e:
            logging.error(f"DataIngestThread: Error in process_message: {e}")
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