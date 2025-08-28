import threading
import time
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any
import pickle

from smocs.utils import ConfigLoader
from smocs.db.mysql_api_v0 import DBManager

logging.basicConfig(level=logging.INFO)

class AgentBase(ABC):
    """
    Base class for SMOCS agents.
    Manages the three main threads: data ingest, ML training, and ML inference.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Generate unique agent ID
        self.agent_id = str(uuid.uuid4())
        
        # Load configuration
        config_path = config_path or os.getenv('CONFIG_PATH', '/app/config.yaml')
        self.config_loader = ConfigLoader(config_path)
        
        if not self.config_loader.has_config('agent'):
            raise ValueError("No agent configuration found in config file")
        
        self.config = self.config_loader.config.get('agent', {})
        
        # Setup database connection for agent registration
        self.db_manager = self._setup_db_connection()
        
        # Initialize thread objects (but don't start them yet)
        self.data_ingest_thread = None
        self.ml_training_thread = None
        self.ml_inference_thread = None
        
        # Thread objects for monitoring
        self.thread_objects = {}
        
        logging.info(f"Agent {self.agent_id} initialized")
    
    def _setup_db_connection(self) -> DBManager:
        """Setup database connection for agent registration."""
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
        """Start the agent and all its threads."""
        try:
            logging.info(f"Starting agent {self.agent_id}...")
            
            # Register agent in database
            self._register_agent()
            
            # Create component threads
            self._create_component_threads()
            
            # Start component threads  
            self._start_component_threads()
            
            # Begin thread monitoring
            self._thread_monitoring()
            
        except KeyboardInterrupt:
            logging.info("Received shutdown signal...")
            self.stop()
        except Exception as e:
            logging.error(f"Error starting agent: {e}")
            self.cleanup()
            raise
    
    def _register_agent(self):
        """Register this agent in the database."""
        try:
            agent_data = {
                'registered_id': self.agent_id,
                'agent_name': self.config.get('name', 'unnamed_agent'),
                'config': pickle.dumps(self.config),
                'info': pickle.dumps({
                    'startup_time': time.time(),
                    'status': 'starting'
                })
            }
            
            # Insert into agent_information table
            query = """INSERT INTO agent_information 
                      (registered_id, agent_name, config, info) 
                      VALUES (%s, %s, %s, %s)"""
            values = (
                agent_data['registered_id'],
                agent_data['agent_name'], 
                agent_data['config'],
                agent_data['info']
            )
            
            status = self.db_manager._DBManager__execute_and_commit(query, values)
            if status == 0:
                logging.info(f"Agent {self.agent_id} registered in database")
            else:
                logging.error("Failed to register agent in database")
                
        except Exception as e:
            logging.error(f"Error registering agent: {e}")
    
    def _create_component_threads(self):
        """Create the three component threads."""
        # Create thread instances using abstract factory methods
        self.data_ingest_thread = self.create_data_ingest_component()
        self.ml_training_thread = self.create_ml_training_component()
        self.ml_inference_thread = self.create_ml_inference_component()
        
        logging.info("Component threads created")
    
    def _start_component_threads(self):
        """Start all component threads."""
        # Start data ingest thread
        data_ingest_thread_obj = threading.Thread(
            target=self.data_ingest_thread.start,
            name=f"{self.agent_id}-data-ingest"
        )
        data_ingest_thread_obj.daemon = True
        data_ingest_thread_obj.start()
        self.thread_objects['data_ingest'] = data_ingest_thread_obj
        
        # Start ML training thread  
        ml_training_thread_obj = threading.Thread(
            target=self.ml_training_thread.start,
            name=f"{self.agent_id}-ml-training"
        )
        ml_training_thread_obj.daemon = True
        ml_training_thread_obj.start()
        self.thread_objects['ml_training'] = ml_training_thread_obj
        
        # Start ML inference thread
        ml_inference_thread_obj = threading.Thread(
            target=self.ml_inference_thread.start,
            name=f"{self.agent_id}-ml-inference"
        )
        ml_inference_thread_obj.daemon = True
        ml_inference_thread_obj.start()
        self.thread_objects['ml_inference'] = ml_inference_thread_obj
        
        # Wait for threads to start
        time.sleep(2)
        
        logging.info("All component threads started")
    
    def _thread_monitoring(self):
        """Monitor thread health and restart if necessary."""
        logging.info("Starting thread monitoring...")
        
        while True:
            try:
                self._check_thread_health()
                time.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                logging.info("Thread monitoring interrupted")
                break
            except Exception as e:
                logging.error(f"Error in thread monitoring: {e}")
                time.sleep(1)
    
    def _check_thread_health(self):
        """Check if all threads are alive and restart if needed."""
        for thread_name, thread_obj in self.thread_objects.items():
            if not thread_obj.is_alive():
                logging.warning(f"Thread {thread_name} is not alive. Attempting restart...")
                try:
                    # Restart the specific thread
                    if thread_name == 'data_ingest':
                        self.data_ingest_thread = self.create_data_ingest_component()
                        new_thread = threading.Thread(
                            target=self.data_ingest_thread.start,
                            name=f"{self.agent_id}-data-ingest"
                        )
                    elif thread_name == 'ml_training':
                        self.ml_training_thread = self.create_ml_training_component()
                        new_thread = threading.Thread(
                            target=self.ml_training_thread.start,
                            name=f"{self.agent_id}-ml-training"
                        )
                    elif thread_name == 'ml_inference':
                        self.ml_inference_thread = self.create_ml_inference_component()
                        new_thread = threading.Thread(
                            target=self.ml_inference_thread.start,
                            name=f"{self.agent_id}-ml-inference"
                        )
                    
                    new_thread.daemon = True
                    new_thread.start()
                    self.thread_objects[thread_name] = new_thread
                    logging.info(f"Thread {thread_name} restarted successfully")
                    
                except Exception as e:
                    logging.error(f"Failed to restart thread {thread_name}: {e}")
    
    def stop(self):
        """Stop the agent and all threads."""
        logging.info("Stopping agent...")
        
        # Stop all thread components
        if self.data_ingest_thread and hasattr(self.data_ingest_thread, 'stop'):
            self.data_ingest_thread.stop()
        if self.ml_training_thread and hasattr(self.ml_training_thread, 'stop'):
            self.ml_training_thread.stop()
        if self.ml_inference_thread and hasattr(self.ml_inference_thread, 'stop'):
            self.ml_inference_thread.stop()
        
        # Wait for threads to finish
        for thread_obj in self.thread_objects.values():
            if thread_obj.is_alive():
                thread_obj.join(timeout=5)
        
        self.cleanup()
        logging.info("Agent stopped")
    
    def cleanup(self):
        """Clean up agent resources."""
        if self.db_manager:
            self.db_manager.close()
    
    @abstractmethod
    def create_data_ingest_component(self):
        """
        Create the data ingest thread component.
        Must be implemented by subclasses.
        
        Returns:
            DataIngestThreadBase: Instance of data ingest thread
        """
        pass
    
    @abstractmethod
    def create_ml_training_component(self):
        """
        Create the ML training thread component.
        Must be implemented by subclasses.
        
        Returns:
            MLTrainingThreadBase: Instance of ML training thread
        """
        pass
    
    @abstractmethod
    def create_ml_inference_component(self):
        """
        Create the ML inference thread component.
        Must be implemented by subclasses.
        
        Returns:
            MLInferenceThreadBase: Instance of ML inference thread
        """
        pass