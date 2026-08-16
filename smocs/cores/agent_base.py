import threading
import time
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any

from smocs.db.mysql_api_v0 import DBManager

class AgentBase(ABC):
    """
    Base class for SMOCS agents.
    Manages the three main threads: data ingest, ML training, and ML inference.
    """
    
    def __init__(self, agent_name: str = "unnamed_agent"):
        """
        Initialize the agent.
        
        Args:
            agent_name: Name for this agent instance
        """
        # Generate unique agent ID
        self.agent_id = str(uuid.uuid4())
        self.agent_name = agent_name
        
        # Setup database connection for agent registration
        self.db_manager = self._setup_db_connection()
        
        # Initialize thread objects (but don't start them yet)
        self.data_ingest_thread = None
        self.ml_training_thread = None
        self.ml_inference_thread = None
        
        # Thread objects for monitoring
        self.thread_objects = {}
        
        logging.info(f"Agent {self.agent_id} ({self.agent_name}) initialized")
    
    def _setup_db_connection(self) -> DBManager:
        """
        Establishes the database connection used for agent registration and status
        tracking (the agent_information table). This connection is distinct from,
        and unrelated to, the DBManager instances each thread establishes
        independently for sensor data storage and retrieval.

        At this point in the initialization sequence, the agent's configuration has
        not yet been loaded - subclasses load it only after invoking this base
        class's constructor - so this connection cannot be supplied with
        context_cols or max_gap_seconds. Those parameters are instead supplied
        later, when the ingest, training, and inference threads establish their
        own connections. This does not result in an inconsistent schema, since
        create_tables() performs an idempotent, additive migration: whichever
        connection runs it first establishes the base schema, and any connection
        that runs it subsequently adds only what is still missing.

        Returns:
            DBManager: A connected DBManager instance with the required tables
                already ensured to exist.
        """
        db_config = {
            'agent_id': self.agent_id,
            'host': os.environ.get('MYSQL_HOST', 'localhost'),
            'port': int(os.environ.get('MYSQL_PORT', 3307)),
            'user': os.environ.get('MYSQL_USER', 'root'),
            'pwd': os.environ['MYSQL_ROOT_PASSWORD'],
        }
        db_manager = DBManager(db_config)
        db_manager.create_tables()
        return db_manager

    def _ensure_sensor_schema(self):
        """
        Hook that allows a subclass to establish, once and in full - including any
        per-agent context_cols columns - the database schema used for sensor data
        storage and retrieval (the agent_inferences table), before any of the three
        component threads are constructed.

        This exists because the connection _setup_db_connection establishes, above,
        is opened before this agent's configuration has been loaded, and therefore
        cannot be supplied with context_cols or max_gap_seconds. By the time
        _create_component_threads (which invokes this hook) runs, in contrast, the
        subclass's own constructor has already loaded that configuration. Ensuring
        the schema here, exactly once, up front, means that the ingest, training,
        and inference threads' own DBManager connections - each established
        independently, when that thread's __init__ runs - can rely on the full
        schema, context columns included, already being present, and therefore have
        no need to invoke create_tables() themselves.

        The default implementation here does nothing, since AgentBase itself has no
        notion of context_cols. Subclasses whose configuration includes a
        model_input.context_channels entry (or an equivalent) should override this
        method to construct a DBManager configured with that agent's actual
        context_cols and max_gap_seconds and call create_tables() on it.
        """
        pass

    def _prepare_agent_data(self, custom_config=None, custom_info=None):
        """
        Prepare agent data for registration.
        Can be overridden by subclasses to provide custom configuration and info.
        
        Args:
            custom_config (dict, optional): Custom configuration to use instead of default
            custom_info (dict, optional): Custom info to use instead of default
            
        Returns:
            tuple: (config_dict, info_dict)
        """
        # Default configuration - can be overridden by subclasses
        config = custom_config if custom_config is not None else {
            'agent_type': self.__class__.__name__,
            'initialization_params': {
                'agent_name': self.agent_name
            }
        }
        
        # Default info - can be overridden by subclasses  
        info = custom_info if custom_info is not None else {
            'registration_time': time.time(),
            'status': 'starting',
            'agent_class': self.__class__.__name__,
            'agent_module': self.__class__.__module__
        }
        
        return config, info
    
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
        """Register this agent in the database using DBManager's register_agent method."""
        try:
            # Prepare agent data (can be customized by subclasses)
            config, info = self._prepare_agent_data()
            
            # Use DBManager's register_agent method
            status = self.db_manager.register_agent(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                config=config,
                info=info
            )
            
            if status != 0:
                logging.error("Failed to register agent in database")
                raise Exception("Agent registration failed")
                
        except Exception as e:
            logging.error(f"Error registering agent: {e}")
            raise e
    
    def update_agent_status(self, status_updates):
        """
        Update agent status information in the database.
        
        Args:
            status_updates (dict): Dictionary containing status updates to merge with existing info
        """
        try:
            self.db_manager.update_agent_info(self.agent_id, status_updates)
        except Exception as e:
            logging.error(f"Error updating agent status: {e}")
    
    def get_agent_info(self):
        """
        Retrieve agent information from the database.
        
        Returns:
            dict: Agent information or None if not found
        """
        try:
            return self.db_manager.get_agent_info(self.agent_id)
        except Exception as e:
            logging.error(f"Error retrieving agent info: {e}")
            return None
    
    def _create_component_threads(self):
        """Create the three component threads."""
        # Ensure the full sensor-data schema - including any per-agent context
        # columns - exists before any thread component below is constructed, so
        # that none of those threads' own DBManager connections need to perform
        # schema migration themselves; see _ensure_sensor_schema's docstring.
        self._ensure_sensor_schema()

        # Create thread instances using abstract factory methods
        self.data_ingest_thread = self.create_data_ingest_component()
        self.ml_training_thread = self.create_ml_training_component()
        self.ml_inference_thread = self.create_ml_inference_component()
        
        logging.info("Component threads created")
    
    def _start_component_threads(self):
        """Start all component threads."""
        # Start data ingest thread if it exists
        if self.data_ingest_thread is not None:
            data_ingest_thread_obj = threading.Thread(
                target=self.data_ingest_thread.start,
                name=f"{self.agent_id}-data-ingest"
            )
            data_ingest_thread_obj.daemon = True
            data_ingest_thread_obj.start()
            self.thread_objects['data_ingest'] = data_ingest_thread_obj
        else:
            logging.info("Data ingest thread not enabled, skipping...")
        
        # Start ML training thread if it exists
        if self.ml_training_thread is not None:
            ml_training_thread_obj = threading.Thread(
                target=self.ml_training_thread.start,
                name=f"{self.agent_id}-ml-training"
            )
            ml_training_thread_obj.daemon = True
            ml_training_thread_obj.start()
            self.thread_objects['ml_training'] = ml_training_thread_obj
        else:
            logging.info("ML training thread not enabled, skipping...")
        
        # Start ML inference thread if it exists
        if self.ml_inference_thread is not None:
            ml_inference_thread_obj = threading.Thread(
                target=self.ml_inference_thread.start,
                name=f"{self.agent_id}-ml-inference"
            )
            ml_inference_thread_obj.daemon = True
            ml_inference_thread_obj.start()
            self.thread_objects['ml_inference'] = ml_inference_thread_obj
        else:
            logging.info("ML inference thread not enabled, skipping...")
        
        # Wait for threads to start (only if any threads were started)
        if self.thread_objects:
            time.sleep(2)
        
        # Update agent status to running
        self.update_agent_status({
            'status': 'running', 
            'threads_started_time': time.time(),
            'active_threads': list(self.thread_objects.keys())
        })
        
        logging.info(f"Component threads started: {list(self.thread_objects.keys())}")
    
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
                    if thread_name == 'data_ingest' and self.data_ingest_thread is not None:
                        self.data_ingest_thread = self.create_data_ingest_component()
                        if self.data_ingest_thread is not None:
                            new_thread = threading.Thread(
                                target=self.data_ingest_thread.start,
                                name=f"{self.agent_id}-data-ingest"
                            )
                    elif thread_name == 'ml_training' and self.ml_training_thread is not None:
                        self.ml_training_thread = self.create_ml_training_component()
                        if self.ml_training_thread is not None:
                            new_thread = threading.Thread(
                                target=self.ml_training_thread.start,
                                name=f"{self.agent_id}-ml-training"
                            )
                    elif thread_name == 'ml_inference' and self.ml_inference_thread is not None:
                        self.ml_inference_thread = self.create_ml_inference_component()
                        if self.ml_inference_thread is not None:
                            new_thread = threading.Thread(
                                target=self.ml_inference_thread.start,
                                name=f"{self.agent_id}-ml-inference"
                            )
                    else:
                        logging.info(f"Thread {thread_name} is disabled, removing from monitoring")
                        # Remove from thread_objects since it's disabled
                        del self.thread_objects[thread_name]
                        continue
                    
                    if 'new_thread' in locals():
                        new_thread.daemon = True
                        new_thread.start()
                        self.thread_objects[thread_name] = new_thread
                        logging.info(f"Thread {thread_name} restarted successfully")
                        
                        # Update agent status with thread restart info
                        self.update_agent_status({
                            f'{thread_name}_restart_time': time.time(),
                            'status': 'running'
                        })
                    
                except Exception as e:
                    logging.error(f"Failed to restart thread {thread_name}: {e}")
                    # Update agent status with error info
                    self.update_agent_status({
                        'status': 'error',
                        'last_error': str(e),
                        'error_time': time.time()
                    })
    
    def stop(self):
        """Stop the agent and all threads."""
        logging.info("Stopping agent...")
        
        # Update agent status to stopping
        self.update_agent_status({'status': 'stopping', 'stop_time': time.time()})
        
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
        
        # Update agent status to stopped
        self.update_agent_status({'status': 'stopped', 'stopped_time': time.time()})
        
        self.cleanup()
        logging.info("Agent stopped")
    
    def is_switch_on(self, message):
        # print(f"Agent config inputs: {self.agent_config['model_input']}")
        if 'switch' in self.agent_config['model_input']:
            switch_dict = self.agent_config['model_input']['switch']
            switch_positions = []
            # print(f"Switch Dict: {switch_dict}")
            # print(f"Message: {message}")
            for var_name in switch_dict:
                position = True
                if 'greater_than' in switch_dict[var_name]:
                    if switch_dict[var_name]['greater_than'] > message[var_name]:
                        position = False
                if 'smaller_than' in switch_dict[var_name]:
                    if switch_dict[var_name]['smaller_than'] < message[var_name]:
                        position = False
                switch_positions.append(position)
            return all(switch_positions)
        else:
            return True
    
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