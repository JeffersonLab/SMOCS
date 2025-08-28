import time
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from smocs.cores import KafkaProducerBase
from smocs.db.mysql_api_v0 import DBManager

logging.basicConfig(level=logging.INFO)

class MLTrainingThreadBase(KafkaProducerBase, ABC):
    """
    Base class for ML training thread.
    Inherits from KafkaProducerBase to send training results to Kafka.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the ML training thread.
        
        Args:
            agent_id: Unique identifier for the parent agent
            config: Agent configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        self.running = False
        
        # Setup Kafka producer
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        super().__init__(kafka_broker_url)
        
        # Setup database connection
        self.db_manager = self._setup_db_connection()
        
        # Build initial model
        self.build_model()
        
        logging.info(f"ML Training Thread initialized for agent {agent_id}")
    
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
    
    def start(self):
        """Start the training thread and begin the training loop."""
        try:
            logging.info("Starting ML Training Thread...")
            self.setup_kafka_producer()
            self.running = True
            self.training_loop()
        except Exception as e:
            logging.error(f"Error in ML Training Thread: {e}")
            self.cleanup()
            raise
    
    def stop(self):
        """Stop the training loop."""
        logging.info("Stopping ML Training Thread...")
        self.running = False
        self.cleanup()
    
    def training_loop(self):
        """
        Main training loop that continuously checks for new data and trains models.
        """
        while self.running:
            try:
                # Get training data from database
                training_data = self.get_training_data()
                
                if training_data is not None:
                    # Train model with training data
                    model_metrics = self.train_model(training_data)
                    
                    # Evaluate model
                    eval_results = self.eval_model()
                    
                    # Save model to database
                    self.save_model(model_metrics, eval_results)
                    
                    # Send training results to Kafka
                    self._send_training_results(model_metrics, eval_results)
                
            except Exception as e:
                logging.error(f"Error in training loop: {e}")
                time.sleep(1)
    
    def _send_training_results(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """Send training results to Kafka."""
        try:
            output_topic = self.config.get('kafka_topics', {}).get('training_output', 'training-results')
            message = {
                'agent_id': self.agent_id,
                'timestamp': time.time(),
                'model_metrics': model_metrics,
                'eval_results': eval_results
            }
            kafka_topic = self.sanitize_topic_name(output_topic)
            self.send_to_kafka(kafka_topic, json.dumps(message))
        except Exception as e:
            logging.error(f"Error sending training results to Kafka: {e}")
    
    @abstractmethod
    def build_model(self):
        """Build/initialize the model. Called during __init__."""
        pass
    
    @abstractmethod
    def get_training_data(self) -> Optional[Any]:
        """
        Retrieve training data from database.
        
        Returns:
            Training data if available, None otherwise
        """
        pass
    
    @abstractmethod
    def train_model(self, training_data: Any) -> Dict[str, Any]:
        """
        Train the model with provided data.
        
        Args:
            training_data: Data to train the model with
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def eval_model(self) -> Dict[str, Any]:
        """
        Evaluate the current model.
        
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    @abstractmethod
    def save_model(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """
        Save the trained model to database.
        
        Args:
            model_metrics: Training metrics
            eval_results: Evaluation results
        """
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()
        super().cleanup()