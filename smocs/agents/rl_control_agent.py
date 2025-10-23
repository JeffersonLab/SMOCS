import os
import json
import time
import logging
import argparse
import threading
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# SMOCS imports
import smocs.control_plane # For custom environments
from smocs.cores import AgentBase, KafkaConsumerBase, MLTrainingThreadBase, KafkaStreamingProcessBase
from smocs.utils import ConfigLoader, setup_logging

# JLab opt control imports
import jlab_opt_control.agents

# Gymnasium for environment instantiation (utilized for SOCT agent init)
import gymnasium as gym

class RLDataIngestThread(KafkaConsumerBase):
    """
    Data ingestion thread for RL control agent.
    Consumes SARSA tuples from Kafka and stores to jlab agent buffer.
    
    Thread Safety:
    - Acquires agent lock before calling agent.memory()
    - Uses blocking acquisition (must store experiences)
    - Waits for inference before processing SARSA
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], jlab_agent: Any, 
                 agent_lock: threading.Lock, inference_done_event: threading.Event,
                 ingestion_done_event: threading.Event):
        """
        Initialize the data ingestion thread.
        
        Args:
            agent_id: Unique identifier for the parent agent
            config: Agent configuration dictionary
            jlab_agent: Instance of jlab_opt_control agent
            agent_lock: Shared lock for thread-safe access to jlab_agent
            inference_done_event: Event to wait for (inference completion)
            ingestion_done_event: Event to signal (ingestion completion)
        """
        self.agent_id = agent_id
        self.config = config
        self.jlab_agent = jlab_agent
        self.agent_lock = agent_lock
        self.inference_done_event = inference_done_event  # Wait for this
        self.ingestion_done_event = ingestion_done_event  # Signal this
        
        # Ingestion configuration
        ingest_config = config.get('data_ingest', {})
        self.use_pipeline_sync = ingest_config.get('use_pipeline_sync', False)
        self.pipeline_timeout = ingest_config.get('pipeline_timeout_sec', 10.0)
        
        # Kafka configuration
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = f"{agent_id}-rl-data-ingest"
        input_topic = config.get('kafka_topics', {}).get('input_sarsa', 'gymnasium-sarsa')
        
        # Initialize base class
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        # Metrics
        self.experiences_stored = 0
        self.lock_wait_times = []
        self.pipeline_wait_timeouts = 0
        
        logging.info(f"RLDataIngestThread: Initialized for agent {agent_id}")
        logging.info(f"RLDataIngestThread: Subscribing to topic: {input_topic}")
        if self.use_pipeline_sync:
            logging.info(f"RLDataIngestThread: Using pipeline synchronization")
    
    def process_message(self, message, topic, partition, offset) -> bool:
        """
        Process SARSA tuple from Kafka and store to jlab agent buffer.
        
        Args:
            message: The message value (JSON string)
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            bool: True if processing was successful
        """
        logging.debug(f"[INGESTION-RECV] Received SARSA message at offset {offset}")
        try:
            # Wait for inference to complete before processing SARSA
            if self.use_pipeline_sync:
                logging.debug("PIPELINE STEP 2: DATA INGESTION - Waiting for inference to complete...")
                logging.debug(f"RLDataIngestThread: inference_done_event.is_set() = {self.inference_done_event.is_set()}")
                
                wait_start = time.time()
                if not self.inference_done_event.wait(timeout=self.pipeline_timeout):
                    self.pipeline_wait_timeouts += 1
                    logging.warning(f"RLDataIngestThread: Timeout waiting for inference "
                                  f"(total timeouts: {self.pipeline_wait_timeouts})")
                    return False
                
                wait_duration = time.time() - wait_start
                logging.debug(f"PIPELINE STEP 2: DATA INGESTION - Inference complete (waited {wait_duration:.3f}s)")
                
                # Clear the inference_done event so we wait next time
                self.inference_done_event.clear()
                logging.debug("PIPELINE STEP 2: DATA INGESTION - Cleared inference_done_event, proceeding with ingestion")
                
                # Small sleep to ensure log ordering
                time.sleep(0.01)
            
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            message_data = json.loads(message)
            
            # Extract SARSA tuple from message
            sarsa_tuple = self._parse_sarsa_message(message_data)
            
            if sarsa_tuple is None:
                logging.error(f"RLDataIngestThread: Failed to parse SARSA from message")
                return False
            
            result = self._save_agent_buffer(sarsa_tuple)

            if result is None:
                logging.error("RLDataIngestThread: Failed to save to buffer")
                return False
            
            # Signal that ingestion is done
            if result and self.use_pipeline_sync:
                logging.debug("PIPELINE STEP 2: DATA INGESTION - Successfully saved SARSA to buffer")
                logging.debug(f"PIPELINE STEP 2: DATA INGESTION - Setting ingestion_done_event (experiences stored: {self.experiences_stored})")
                self.ingestion_done_event.set()
                
                # Small sleep to ensure log ordering
                time.sleep(0.01)
            
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"RLDataIngestThread: JSON decode error: {e}")
            return False
        except Exception as e:
            logging.error(f"RLDataIngestThread: Error processing message: {e}")
            logging.error(f"RLDataIngestThread: Message content: {message[:200]}")
            return False
    
    def _save_agent_buffer(self, sarsa_tuple):
        """ Function to save the SARSA tuple to the agents buffer, returns True is successful else error out """
        
        state, action, reward, next_state, done = sarsa_tuple

        logging.info(f"SARSA Tuple #{self.experiences_stored}:")
        logging.info(f"  State: {state}")
        logging.info(f"  Action: {action}")
        logging.info(f"  Reward: {reward:.3f}")
        logging.info(f"  Next_State: {next_state}")
        logging.info(f"  Done: {done}")
        logging.info(f"  State vs Next_State diff: {np.linalg.norm(next_state - state):.3f}")
    

        try:    
            # Acquire lock and store to buffer
            lock_wait_start = time.time()
            with self.agent_lock:
                lock_wait_time = time.time() - lock_wait_start
                self.lock_wait_times.append(lock_wait_time)
                
                # Log if lock wait was significant
                if lock_wait_time > 0.05:  # 50ms threshold
                    logging.warning(f"RLDataIngestThread: Lock wait time: {lock_wait_time:.3f}s")
                
                # Store experience in jlab agent buffer
                self.jlab_agent.memory((state, action, reward, next_state, done))
                
                self.experiences_stored += 1

            # Log periodically
            if self.experiences_stored % 100 == 0:
                avg_lock_wait = np.mean(self.lock_wait_times[-100:]) if self.lock_wait_times else 0
                logging.info(f"RLDataIngestThread: Stored {self.experiences_stored} experiences, "
                            f"avg lock wait: {avg_lock_wait:.4f}s, buffer size: {self.jlab_agent.buffer.size()}, "
                            f"pipeline timeouts: {self.pipeline_wait_timeouts}")
            return True
        except Exception as e:
            logging.error(f"RLDataIngestThread: Error saving SARSA in agent buffer: {e}")
            return None
        
    def _parse_sarsa_message(self, message_data: Dict[str, Any]) -> Optional[Tuple]:
        """
        Parse SARSA tuple from Kafka message with base64-encoded numpy arrays.

        Expected SARSA message format from Gymnasium wrapper:
        {
            "channels": {
                "state": {"_numpy_": True, "data": "...", "dtype": "...", "shape": [...]},
                "action": {"_numpy_": True, "data": "...", "dtype": "...", "shape": [...]},
                "reward": float,
                "next_state": {"_numpy_": True, "data": "...", "dtype": "...", "shape": [...]},
                "done": bool,
                "truncated": bool,
                ...
            },
            "timestamp": float
        }

        Args:
            message_data: Parsed JSON message

        Returns:
            Tuple of (state, action, reward, next_state, done) or None if parsing fails
        """
        try:
            channels = message_data.get('channels', {})
            
            if not channels:
                logging.error("RLDataIngestThread: No 'channels' field in message")
                return None
            
            # Extract components
            state_json = channels.get('state')
            action_json = channels.get('action')
            reward = channels.get('reward')
            next_state_json = channels.get('next_state')
            done = channels.get('done', False)
            truncated = channels.get('truncated', False)
            
            # Validate all components exist
            if state_json is None or action_json is None or reward is None or next_state_json is None:
                logging.error("RLDataIngestThread: Missing required SARSA components")
                logging.error(f"RLDataIngestThread: state: {state_json is not None}, action: {action_json is not None}, "
                           f"reward: {reward is not None}, next_state: {next_state_json is not None}")
                return None
             
            # Convert to numpy arrays (JSON removes all data formats)
            state = np.array(state_json, dtype=np.float32)
            action = np.array(action_json, dtype=np.float32)
            reward = float(reward)
            next_state = np.array(next_state_json, dtype=np.float32)
            
            # Combine done and truncated into single done flag
            done = bool(done or truncated)
            
            # Validate shapes
            if state.ndim != 1 or next_state.ndim != 1:
                logging.error(f"RLDataIngestThread: Invalid state dimensions: state={state.shape}, next_state={next_state.shape}")
                return None
            
            if action.ndim != 1:
                logging.error(f"RLDataIngestThread: Invalid action dimension: {action.shape}")
                return None
            
            # Validate dtypes are preserved
            if state.dtype != np.float32:
                logging.warning(f"RLDataIngestThread: State dtype is {state.dtype}, expected float32. Attempting converstion")
                state = state.astype(np.float32)
            
            if action.dtype != np.float32:
                logging.warning(f"RLDataIngestThread: Action dtype is {action.dtype}, expected float32. Attempting converstion")
                action = action.astype(np.float32)
                
            if next_state.dtype != np.float32:
                logging.warning(f"RLDataIngestThread: Next_state dtype is {next_state.dtype}, expected float32. Attempting converstion")
                next_state = next_state.astype(np.float32)
            
            return (state, action, reward, next_state, done)
            
        except Exception as e:
            logging.error(f"RLDataIngestThread: Error parsing SARSA message: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

class RLTrainingThread(MLTrainingThreadBase):
    """
    ML training thread for RL control agent.
    Trains the jlab agent when data is ready.
    
    Thread Safety:
    - Waits for ingestion to complete before training
    - Signals when training is complete
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], jlab_agent: Any, 
                 agent_lock: threading.Lock, ingestion_done_event: threading.Event,
                 training_done_event: threading.Event, tensorboard_writer):
        self.agent_id = agent_id
        self.config = config
        self.jlab_agent = jlab_agent
        self.agent_lock = agent_lock
        self.ingestion_done_event = ingestion_done_event  # Wait for this
        self.training_done_event = training_done_event    # Signal this
        self.tb_writer = tensorboard_writer
        
        # Training configuration
        training_config = config.get('training', {})
        self.check_interval_ms = training_config.get('check_interval_ms', 10)
        self.use_pipeline_sync = training_config.get('use_pipeline_sync', False)
        self.pipeline_timeout = training_config.get('pipeline_timeout_sec', 10.0)
        
        # Metrics
        self.training_updates = 0
        self.pipeline_wait_timeouts = 0
        
        # Initialize base class
        super().__init__(agent_id, config)
        
        logging.info(f"RLTrainingThread: Initialized for agent {agent_id}")
        if self.use_pipeline_sync:
            logging.info(f"RLTrainingThread: Using pipeline synchronization")
    
    def start(self):
        """Start the training thread with custom training loop."""
        try:
            logging.info("RLTrainingThread: Starting training loop...")
            self.running = True
            self.training_loop()
        except Exception as e:
            logging.error(f"RLTrainingThread: Error in training thread: {e}")
    
    def training_loop(self):
        """Main training loop with pipeline synchronization."""
        
        logging.info("RLTrainingThread: Training loop started")
        
        while self.running:
            try:
                if self.use_pipeline_sync:
                    # Wait for ingestion to complete before training
                    logging.debug("PIPELINE STEP 3: TRAINING - Waiting for ingestion to complete...")
                    logging.debug(f"RLTrainingThread: ingestion_done_event.is_set() = {self.ingestion_done_event.is_set()}")
                    
                    wait_start = time.time()
                    if not self.ingestion_done_event.wait(timeout=self.pipeline_timeout):
                        self.pipeline_wait_timeouts += 1
                        if self.pipeline_wait_timeouts % 10 == 0:
                            logging.warning(f"RLTrainingThread: Timeout waiting for ingestion "
                                          f"(total timeouts: {self.pipeline_wait_timeouts})")
                        time.sleep(self.check_interval_ms / 1000.0)
                        continue
                    
                    wait_duration = time.time() - wait_start
                    logging.debug(f"PIPELINE STEP 3: TRAINING - Ingestion complete (waited {wait_duration:.3f}s)")
                    
                    # Clear the ingestion_done event so we wait next time
                    self.ingestion_done_event.clear()
                    logging.debug("PIPELINE STEP 3: TRAINING - Cleared ingestion_done_event, proceeding with training")
                    
                    # Small sleep to ensure log ordering
                    time.sleep(0.01)
                    
                    # Acquire lock and train
                    logging.debug("PIPELINE STEP 3: TRAINING - Attempting to acquire agent lock...")
                    acquired = self.agent_lock.acquire(blocking=True, timeout=2.0)
                    
                    if acquired:
                        try:
                            logging.debug("PIPELINE STEP 3: TRAINING - Lock acquired, starting training...")
                            train_start = time.time()
                            
                            with self.tb_writer.as_default():
                                self.jlab_agent.train()
                                self.tb_writer.flush()
                            
                            train_duration = time.time() - train_start
                            self.training_updates += 1
                            
                            logging.debug(f"PIPELINE STEP 3: TRAINING - Training complete (duration: {train_duration:.3f}s, "
                                       f"update #{self.training_updates}, buffer size: {self.jlab_agent.buffer.size()})")
                            
                            # Log periodically
                            if self.training_updates % 10 == 0:
                                logging.info(f"RLTrainingThread: Completed {self.training_updates} training updates, "
                                        f"buffer size: {self.jlab_agent.buffer.size()}, "
                                        f"pipeline timeouts: {self.pipeline_wait_timeouts}")
                        
                        finally:
                            self.agent_lock.release()
                            logging.debug("PIPELINE STEP 3: TRAINING - Released agent lock")
                        
                        # Signal that training is done
                        logging.debug("PIPELINE STEP 3: TRAINING - Setting training_done_event")
                        self.training_done_event.set()
                        
                        # Small sleep to ensure log ordering
                        time.sleep(0.01)
                    else:
                        logging.warning("RLTrainingThread: Failed to acquire lock within timeout")
                
                else:
                    # Original non-synchronized mode - train whenever possible
                    logging.warning("RLTrainingThread: Non-pipeline mode not fully implemented")
                    time.sleep(1.0)
                
                # Sleep before next iteration
                time.sleep(self.check_interval_ms / 1000.0)
                
            except Exception as e:
                logging.error(f"RLTrainingThread: Error in training loop: {e}")
                import traceback
                logging.error(traceback.format_exc())
                time.sleep(1)
    
    # Override base class abstract methods (we don't use them with the SOCT agents for now)
    def build_model(self):
        """Not used - jlab agent manages its own models."""
        pass
    
    def get_training_data(self) -> Optional[Any]:
        """Not used - jlab agent manages its own buffer."""
        return None
    
    def train_model(self, training_data: Any) -> Dict[str, Any]:
        """Not used - jlab agent has its own train() method."""
        return {}
    
    def eval_model(self) -> Dict[str, Any]:
        """Not used - jlab agent manages evaluation internally."""
        return {}
    
    def save_model(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """Not used - model saving disabled for now."""
        pass

class RLInferenceThread(KafkaStreamingProcessBase):
    """
    ML inference thread for RL control agent.
    Consumes states from Kafka, generates actions using jlab agent, publishes to Kafka.
    
    Thread Safety:
    - Acquires agent lock before calling agent.action()
    - Uses blocking acquisition
    - Waits for training to complete before processing next state
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], jlab_agent: Any, 
                 agent_lock: threading.Lock, training_done_event: threading.Event,
                 inference_done_event: threading.Event, tensorboard_writer):
        self.agent_id = agent_id
        self.config = config
        self.jlab_agent = jlab_agent
        self.agent_lock = agent_lock
        self.training_done_event = training_done_event  # Wait for this
        self.inference_done_event = inference_done_event  # Signal this
        self.tb_writer = tensorboard_writer
        
        # Inference configuration
        inference_config = config.get('inference', {})
        self.train_mode = inference_config.get('train_mode', True)
        self.log_lock_wait_threshold_ms = inference_config.get('log_lock_wait_threshold_ms', 100)
        self.use_pipeline_sync = inference_config.get('use_pipeline_sync', False)
        self.pipeline_timeout = inference_config.get('pipeline_timeout_sec', 10.0)
        
        # Kafka configuration
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = f"{agent_id}-rl-inference"
        input_topic = config.get('kafka_topics', {}).get('input_state', 'gymnasium-state')
        self.output_topic = config.get('kafka_topics', {}).get('output_action', 'gymnasium-action')
        
        # Metrics
        self.actions_generated = 0
        self.lock_wait_times = []
        self.pipeline_wait_timeouts = 0
        
        # Initialize base class
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        logging.info(f"RLInferenceThread: Initialized for agent {agent_id}")
        if self.use_pipeline_sync:
            logging.info(f"RLInferenceThread: Using pipeline synchronization")
    
    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """Process state from Kafka, generate action."""
        try:
            # STEP 1: Wait for training to complete before processing this state
            # (For the first message, training_done is already set)
            # (For subsequent messages, this waits until the previous cycle completes)
            if self.use_pipeline_sync:
                logging.debug("PIPELINE STEP 1: INFERENCE - Waiting for training cycle to complete...")
                logging.debug(f"RLInferenceThread: training_done_event.is_set() = {self.training_done_event.is_set()}")
                
                wait_start = time.time()
                if not self.training_done_event.wait(timeout=self.pipeline_timeout):
                    self.pipeline_wait_timeouts += 1
                    logging.warning(f"RLInferenceThread: Timeout waiting for training "
                                f"(total timeouts: {self.pipeline_wait_timeouts})")
                    return False, []
                
                wait_duration = time.time() - wait_start
                logging.debug(f"PIPELINE STEP 1: INFERENCE - Training complete (waited {wait_duration:.3f}s)")
                
                logging.debug("PIPELINE STEP 1: INFERENCE - Proceeding with inference (will clear training_done_event after)")
                
                # Small sleep to ensure log ordering
                time.sleep(0.01)
            
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            message_data = json.loads(message)
            
            # Extract state
            state = self._parse_state_message(message_data)
            
            if state is None:
                logging.error("RLInferenceThread: Failed to parse state from message")
                return False, []
            
            action_list = self._agent_action(state)
            
            if action_list is None:
                logging.error("RLInferenceThread: Failed to generate action")
                return False, []

            logging.debug(f"PIPELINE STEP 1: INFERENCE - Generated action: {action_list[:3]}... (showing first 3 elements)")

            # Create output message in channels format for consistency
            output_message = {
                'channels': {
                    'action': action_list
                },
                'timestamp': time.time()
            }
            
            # STEP 2: Signal that inference is done, THEN clear training_done
            if self.use_pipeline_sync:
                logging.debug(f"PIPELINE STEP 1: INFERENCE - Inference complete (action #{self.actions_generated})")
                logging.debug("PIPELINE STEP 1: INFERENCE - Setting inference_done_event")
                self.inference_done_event.set()
                
                # Small sleep to ensure log ordering
                time.sleep(0.01)
                
                # NOW clear training_done_event so we wait for the next cycle
                logging.debug("PIPELINE STEP 1: INFERENCE - Clearing training_done_event (ready for next cycle)")
                self.training_done_event.clear()
                
                # Small sleep to ensure log ordering
                time.sleep(0.01)
            
            # Log periodically
            if self.actions_generated % 100 == 0:
                avg_lock_wait = np.mean(self.lock_wait_times[-100:]) if self.lock_wait_times else 0
                logging.info(f"RLInferenceThread: Generated {self.actions_generated} actions, "
                        f"avg lock wait: {avg_lock_wait:.4f}s, buffer size: {self.jlab_agent.buffer.size()}, "
                        f"pipeline timeouts: {self.pipeline_wait_timeouts}")
            
            # Sanitize topic name and return
            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
            
        except json.JSONDecodeError as e:
            logging.error(f"RLInferenceThread: JSON decode error: {e}")
            return False, []
        except Exception as e:
            logging.error(f"RLInferenceThread: Error processing message: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False, []
    
    def _agent_action(self, state):
        """Input state, generate action list by calling SOCT agent"""
        try:
            # Acquire lock and generate action
            logging.debug("PIPELINE STEP 1: INFERENCE - Attempting to acquire agent lock...")
            lock_wait_start = time.time()
            with self.agent_lock:
                lock_wait_time = time.time() - lock_wait_start
                self.lock_wait_times.append(lock_wait_time)
                
                logging.debug(f"PIPELINE STEP 1: INFERENCE - Lock acquired (wait time: {lock_wait_time:.3f}s)")
                
                if lock_wait_time > (self.log_lock_wait_threshold_ms / 1000.0):
                    logging.warning(f"RLInferenceThread: Lock wait time: {lock_wait_time:.3f}s")
                
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                
                action_start = time.time()
                with self.tb_writer.as_default():
                    action, _ = self.jlab_agent.action(state_tensor, train=self.train_mode)
                    self.tb_writer.flush()
                
                action_duration = time.time() - action_start
                self.actions_generated += 1
                
                logging.debug(f"PIPELINE STEP 1: INFERENCE - Action generated (duration: {action_duration:.3f}s)")
            
            logging.debug("PIPELINE STEP 1: INFERENCE - Released agent lock")
            
            # Convert action to list for JSON serialization
            action_list = action.numpy().tolist() if hasattr(action, 'numpy') else (action.tolist() if isinstance(action, np.ndarray) else list(action))
            
            logging.debug(f"RLInferenceThread: Action list: {action_list}")

            return action_list
        
        except Exception as e:
            logging.error(f"RLInferenceThread: Error performing agent action: {e}")
            return None

    def _parse_state_message(self, message_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Parse state from Kafka message with base64-encoded numpy array.
        
        Expected message format from Gymnasium wrapper:
        {
            "channels": {
                "state": {"_numpy_": True, "data": "...", "dtype": "...", "shape": [...]}
            },
            "timestamp": float
        }
        
        Args:
            message_data: Parsed JSON message
            
        Returns:
            State as numpy array or None if parsing fails
        """
        try:
            channels = message_data.get('channels', {})
            
            if not channels:
                logging.error("RLInferenceThread: No 'channels' field in message")
                return None
            
            state_json = channels.get('state')

            if state_json is None:
                logging.error("RLInferenceThread: No 'state' field in channels")
                return None
            
            # Convert to numpy array
            state = np.array(state_json, dtype=np.float32)

            # Validate dimensions (only using BOX for now)
            if state.ndim != 1:
                logging.error(f"RLInferenceThread: Invalid state dimension: {state.shape}, expected 1D array")
                return None
            
            # Validate dtype
            if state.dtype != np.float32:
                logging.warning(f"RLInferenceThread: State dtype is {state.dtype}, expected float32. Converting...")
                state = state.astype(np.float32)
            
            return state
            
        except Exception as e:
            logging.error(f"RLInferenceThread: Error parsing state message: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None

class RLControlAgent(AgentBase):
    """
    RL Control Agent that wraps jlab_opt_control agents (TD3, SAC, etc.)
    and interfaces with Gymnasium Kafka wrapper.
    
    Architecture:
    - Three threads: data ingestion, training, inference
    - Shared jlab_agent instance protected by single lock
    - Inference and data ingestion use blocking lock (critical path)
    - Training uses non-blocking lock (opportunistic)
    """
    
    def __init__(self, config_path: str = None, config_key: str = None):
        """
        Initialize the RL control agent.
        
        Args:
            config_path: Path to configuration file
            config_key: Key for agent configuration in config file
        """
        super().__init__("RLControlAgent")
        
        # Load configuration
        if config_path:
            config_loader = ConfigLoader(config_path)
            if config_key is None:
                config_key = "rl_control_agent1"
            
            if not config_loader.has_config(name=config_key):
                raise ValueError(f"No configuration found for key: {config_key}")
            
            self.agent_config = config_loader.config.get(config_key, {})
        else:
            raise ValueError("config_path is required for RLControlAgent")
        
        # Extract configuration
        self.environment_id = self.agent_config.get('environment', 'CartPole-v1')
        self.jlab_agent_type = self.agent_config.get('jlab_agent_type', 'KerasTD3-v0')
        self.jlab_agent_config_path = self.agent_config.get('jlab_agent_config_path')
        self.buffer_type = self.agent_config.get('buffer_type', 'ER-v0')
        self.buffer_size = self.agent_config.get('buffer_size', 1000000)
        
        # Get enabled threads
        self.enabled_threads = self.agent_config.get('enabled_threads', ['ingest', 'training', 'inference'])
        
        logging.info(f"RLControlAgent: Initializing with environment: {self.environment_id}")
        logging.info(f"RLControlAgent: JLab agent type: {self.jlab_agent_type}")
        logging.info(f"RLControlAgent: Buffer type: {self.buffer_type}, size: {self.buffer_size}")
        logging.info(f"RLControlAgent: Enabled threads: {self.enabled_threads}")
        
        # Create local environment instance for initialization
        self.env = self._create_local_environment()
        
        # Create jlab agent instance
        self.jlab_agent = self._create_jlab_agent()
        
        # Create shared lock for thread-safe access to jlab_agent
        self.jlab_agent_lock = threading.Lock()

        #  Three-way synchronization events for default RL blcoking training pipeline
        self.inference_done_event = threading.Event()
        self.ingestion_done_event = threading.Event()
        self.training_done_event = threading.Event()
        
        # Set training_done initially so inference can start first
        self.training_done_event.set()
        
        logging.debug("PIPELINE INITIALIZATION COMPLETE")
        logging.debug("Initial state: training_done_event is SET (inference can start)")        
        logging.info(f"RLControlAgent: Agent {self.agent_id} initialized successfully")
    
    def _create_local_environment(self) -> gym.Env:
        """
        Create a local instance of the Gymnasium environment.
        This is used only for initialization purposes (getting observation/action spaces).
        The actual environment is managed by the Gymnasium Kafka wrapper.
        
        Returns:
            gym.Env: Local environment instance
        """
        try:
            env = gym.make(self.environment_id)
            logging.info(f"RLControlAgent: Created environment: {self.environment_id}")
    
            # Log environment details
            logging.info(f"RLControlAgent: Observation space: {env.observation_space}")
            logging.info(f"RLControlAgent: Action space: {env.action_space}")
            
            return env
            
        except Exception as e:
            logging.error(f"RLControlAgent: Failed to create environment '{self.environment_id}': {e}")
            raise
    
    def _create_jlab_agent(self):
        """
        Create an instance of the jlab_opt_control agent.
        
        Returns:
            jlab_opt_control agent instance (e.g., KerasTD3)
        """
        try:
            # Get logdir from config or use default
            logdir_base = self.agent_config.get('logdir', f'./logs/rl_agent_{self.agent_id}')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            logdir = os.path.join(logdir_base, f'{timestamp}')

            # Ensure logdir exists
            os.makedirs(logdir, exist_ok=True)
            
            logging.info(f"RLControlAgent: Creating jlab agent with logdir: {logdir}")
            
            # Prepare config path
            cfg_arg = None
            if self.jlab_agent_config_path:
                # If it's already an absolute path, use it directly
                if os.path.isabs(self.jlab_agent_config_path):
                    cfg_path = self.jlab_agent_config_path
                else:
                    # Extract just the filename
                    config_filename = os.path.basename(self.jlab_agent_config_path)
                    
                    # Look in these locations in order:
                    # 1. /app/jlab_configs/ (user-provided configs)
                    # 2. /app/jlab_configs_default/ (default jlab configs)
                    search_paths = [
                        os.path.join('/app/jlab_configs', config_filename),
                        os.path.join('/app/jlab_configs_default', config_filename)
                    ]
                    
                    cfg_path = None
                    for path in search_paths:
                        if os.path.exists(path):
                            cfg_path = path
                            break
                    
                    if cfg_path is None:
                        raise FileNotFoundError(
                            f"JLab config file '{config_filename}' not found in any of: {search_paths}"
                        )
                
                logging.info(f"RLControlAgent: Using jlab config: {cfg_path}")
                cfg_arg = cfg_path
            
            # Create jlab agent
            # Note: jlab agents.make() expects: (agent_id, env, logdir, buffer_type, buffer_size, cfg)
            jlab_agent = jlab_opt_control.agents.make(
                self.jlab_agent_type,
                env=self.env,
                logdir=logdir,
                buffer_type=self.buffer_type,
                buffer_size=self.buffer_size,
                cfg=cfg_arg
            )
            
            self.tensorboard_writer = tf.summary.create_file_writer(f"{logdir}/metrics")
            logging.info(f"RLControlAgent: Created writer reference for {logdir}/metrics")

            logging.info(f"RLControlAgent: Successfully created jlab agent: {self.jlab_agent_type}")
            logging.info(f"RLControlAgent: Buffer capacity: {jlab_agent.buffer.buffer_capacity}")
            logging.info(f"RLControlAgent: Warmup size: {jlab_agent.warmup_size}")
            logging.info(f"RLControlAgent: Batch size: {jlab_agent.batch_size}")

            # Debug: Check what bounds were set
            logging.info(f"RLControlAgent: JLab agent action bounds:")
            logging.info(f"  lower_bound: {jlab_agent.lower_bound}")
            logging.info(f"  upper_bound: {jlab_agent.upper_bound}")
            logging.info(f"  Environment action_space.low: {self.env.action_space.low}")
            logging.info(f"  Environment action_space.high: {self.env.action_space.high}")
            
            return jlab_agent
            
        except Exception as e:
            logging.error(f"RLControlAgent: Failed to create jlab agent: {e}")
            raise
    
    def create_data_ingest_component(self):
        """Create the data ingestion thread component."""
        if 'ingest' in self.enabled_threads:
            return RLDataIngestThread(
                self.agent_id,
                self.agent_config,
                self.jlab_agent,
                self.jlab_agent_lock,
                self.inference_done_event,  # Wait for this
                self.ingestion_done_event   # Signal this
            )
        return None
    
    def create_ml_training_component(self):
        """Create the ML training thread component."""
        if 'training' in self.enabled_threads:
            return RLTrainingThread(
                self.agent_id,
                self.agent_config,
                self.jlab_agent,
                self.jlab_agent_lock,
                self.ingestion_done_event,  # Wait for this
                self.training_done_event,   # Signal this
                self.tensorboard_writer
            )
        return None
    
    def create_ml_inference_component(self):
        """Create the ML inference thread component."""
        if 'inference' in self.enabled_threads:
            return RLInferenceThread(
                self.agent_id,
                self.agent_config,
                self.jlab_agent,
                self.jlab_agent_lock,
                self.training_done_event,   # Wait for this
                self.inference_done_event,  # Signal this
                self.tensorboard_writer
            )
        return None
    
    def _prepare_agent_data(self, custom_config=None, custom_info=None):
        """
        Prepare agent data for database registration.
        
        Args:
            custom_config: Custom configuration dict
            custom_info: Custom info dict
            
        Returns:
            Tuple of (config_dict, info_dict)
        """
        config = custom_config if custom_config is not None else {
            'agent_type': 'RLControlAgent',
            'jlab_agent_type': self.jlab_agent_type,
            'environment': self.environment_id,
            'buffer_type': self.buffer_type,
            'buffer_size': self.buffer_size,
            'enabled_threads': self.enabled_threads
        }
        
        info = custom_info if custom_info is not None else {
            'registration_time': time.time(),
            'status': 'starting',
            'agent_class': self.__class__.__name__,
            'jlab_agent_warmup_size': self.jlab_agent.warmup_size,
            'jlab_agent_batch_size': self.jlab_agent.batch_size
        }
        
        return config, info

def main():
    """Main entry point for RL control agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config", help="Key for agent configuration in config file", 
                       type=str, default='rl_control_agent1')
    args = parser.parse_args()
    
    setup_logging()
    
    config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')
    config_key = args.agent_config
    
    logging.info(f"Starting RLControlAgent with config key: {config_key}")
    
    try:
        agent = RLControlAgent(config_path, config_key)
        agent.start()
    except KeyboardInterrupt:
        logging.info("Shutting down RL control agent...")
    except Exception as e:
        logging.error(f"Error running RL control agent: {e}")
        raise

if __name__ == "__main__":
    main()