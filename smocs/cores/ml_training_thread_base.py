import time
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from smocs.cores import KafkaProducerBase
from smocs.db.mysql_api_v0 import DBManager

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
        
        logging.info(f"MLTrainingThread: ML Training Thread initialized for agent {agent_id}")
    
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
            logging.info("MLTrainingThread: Starting ML Training Thread...")
            self.setup_kafka_producer()
            self.running = True
            self.training_loop()
        except Exception as e:
            logging.error(f"MLTrainingThread: Error in ML Training Thread: {e}")
            self.cleanup()
            raise
    
    def stop(self):
        """Stop the training loop."""
        logging.info("MLTrainingThread: Stopping ML Training Thread...")
        self.running = False
        self.cleanup()
    
    def training_loop(self):
        """
        Main training loop that continuously checks for new data, trains, evaluates,
        and saves models.

        Training and evaluation are independent outcomes, each handled on its own:
          - No training data available: wait and retry next cycle; nothing else runs.
          - train_model() fails: nothing is saved or reported to Kafka, and
            committed_last_training_count is deliberately NOT advanced - see
            get_training_data()'s docstring - so this same data is retried on
            the next cycle instead of being silently skipped forever.
          - train_model() succeeds: committed_last_training_count IS advanced, and
            eval_model() then runs regardless of what it returns.
          - eval_model() succeeds: both training and evaluation info are saved.
          - eval_model() fails (e.g. no fresh data to score against at that moment):
            the newly trained model is still saved - training succeeded and is worth
            keeping - but its eval_metrics records that evaluation was skipped this
            cycle, rather than a stale or default threshold silently being assumed.
        """
        logging.info("MLTrainingThread: Starting training loop...")

        while self.running:
            try:
                logging.debug("MLTrainingThread: Checking for training data...")

                # Get training data from database
                training_data = self.get_training_data()

                if training_data is not None:
                    logging.info(f"MLTrainingThread: Found training data with shape {training_data.shape}")

                    # Train model with training data
                    logging.info("MLTrainingThread: Starting model training...")
                    model_metrics = self.train_model(training_data)
                    logging.info(f"MLTrainingThread: Training completed with metrics: {model_metrics}")

                    if 'error' in model_metrics:
                        # Training failed: nothing to evaluate or save.
                        # committed_last_training_count is intentionally left untouched -
                        # get_training_data() only staged the candidate value in
                        # self._pending_last_training_count, never committed it - so this
                        # same data is considered "new" again and retried next cycle
                        # instead of being silently skipped forever.
                        logging.error(f"MLTrainingThread: Training failed, skipping this cycle: {model_metrics['error']}")
                        time.sleep(30)
                        continue

                    # Training succeeded - commit the pending count only now.
                    self.committed_last_training_count = self._pending_last_training_count

                    # Evaluate model
                    logging.info("MLTrainingThread: Starting model evaluation...")
                    eval_results = self.eval_model()
                    logging.info(f"MLTrainingThread: Evaluation completed: {eval_results}")

                    # Save model - always, now that training has succeeded, regardless of
                    # whether evaluation also succeeded; save_model() persists whatever
                    # eval_results contains, including a "skipped" status if evaluation
                    # itself did not succeed this cycle.
                    logging.info("MLTrainingThread: Saving model...")
                    self.save_model(model_metrics, eval_results)

                    # Send training results to Kafka
                    logging.info("MLTrainingThread: Sending training results to Kafka...")
                    self._send_training_results(model_metrics, eval_results)

                    # Sleep after successful training to avoid continuous training
                    logging.info("MLTrainingThread: Training cycle completed, sleeping...")
                    time.sleep(30)  # Wait 30 seconds before next training cycle

                else:
                    logging.info("MLTrainingThread: Training data unavailable, waiting...")
                    time.sleep(30)  # Wait 30 seconds before checking again

            except Exception as e:
                logging.error(f"MLTrainingThread: Error in training loop: {e}")
                logging.error(f"MLTrainingThread: Exception details: {type(e).__name__}: {str(e)}")
                time.sleep(5)  # Wait 5 seconds before retrying on error
    
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
            logging.error(f"MLTrainingThread: Error sending training results to Kafka: {e}")
    
    @abstractmethod
    def build_model(self):
        """Build/initialize the model"""
        pass
    
    @abstractmethod
    def get_training_data(self) -> Optional[Any]:
        """
        Retrieve training data from database.

        Implementations that gate on a "new since last training" count (e.g.
        self.committed_last_training_count) must NOT commit that count here - stash
        the candidate value in self._pending_last_training_count instead.
        training_loop() commits it to self.committed_last_training_count only after
        train_model() actually succeeds. Committing it here unconditionally would
        advance the count even when training subsequently fails, permanently
        skipping this data instead of retrying it on the next cycle.

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