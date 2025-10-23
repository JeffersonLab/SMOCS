import os
import time
import json
import logging
import numpy as np
import tensorflow as tf
import gymnasium as gym
from datetime import datetime
from typing import List, Tuple, Union, Callable, Any

import smocs.control_plane
from smocs.cores import KafkaStreamingProcessBase
from smocs.utils import ConfigLoader, setup_logging

class KafkaGymWrapper(KafkaStreamingProcessBase):
    """
    Kafka-controlled Gymnasium environment wrapper with configuration support.
    
    This wrapper allows any Gymnasium environment to be controlled via Kafka messages.
    It can operate in two modes:
    
    1. Blocking mode: Waits for actions from Kafka before stepping the environment
    2. Default action mode: Runs continuously using default actions when no Kafka action is received
    
    The wrapper sends three types of messages to Kafka after each environment step:
    1. SARSA topic: Complete RL transition tuples with native formats
    2. State topic: Current state only for downstream consumers
    3. Decomposed topic: Flattened data for logging/monitoring
    
    Logs comprehensive metrics to TensorBoard including per-step rewards, episode metrics, and more.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Kafka Gym wrapper with configuration.
        
        Args:
            config_path: Path to configuration file (uses environment variable if None)
        """
        # Load configuration
        config_path = config_path or os.getenv('CONFIG_PATH', '/app/config.yaml')
        self.config_loader = ConfigLoader(config_path)
        
        # Validate gymnasium configuration exists
        if not self.config_loader.has_config(name='gymnasium'):
            raise ValueError("No gymnasium configuration found in config file")
        
        # Get gymnasium configuration
        self.gym_config = self.config_loader.get_gymnasium_config()
        
        # Kafka configuration from environment
        kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
        group_id = os.environ.get('KAFKA_GROUP_ID', 'gym-wrapper')
        
        # Extract topics from config
        input_topic = self.gym_config['input_topic']

        # ConfigLoader now always returns output_topics as a dictionary
        self.output_topics = self.gym_config['output_topics']
        
        # Store configuration
        self.blocking_mode = self.gym_config['blocking_mode']
        self.default_action_strategy = self.gym_config['default_action_strategy']
        self.step_delay = self.gym_config['step_delay']
        self.reset_on_start = self.gym_config['reset_on_start']
        
        # Initialize streaming processor to listen to input topic
        super().__init__(kafka_broker_url, group_id, [input_topic])
        
        # Initialize gymnasium environment
        self.env = self._create_environment()
        
        # Environment state
        self.current_obs = None
        self.episode_step = 0
        self.episode_num = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_start_time = None
        
        # TensorBoard setup
        self._setup_tensorboard()
        
        logging.info(f"Kafka Gym Wrapper initialized:")
        logging.info(f"  Environment: {self.gym_config['environment']}")
        logging.info(f"  Input topic: {input_topic}")
        logging.info(f"  Output topics:")
        logging.info(f"    SARSA: {self.output_topics['sarsa']}")
        logging.info(f"    State: {self.output_topics['state']}")
        logging.info(f"    Decomposed: {self.output_topics['decomposed']}")
        logging.info(f"  Blocking mode: {self.blocking_mode}")
        logging.info(f"  Default action strategy: {self.default_action_strategy}")
        logging.info(f"  Step delay: {self.step_delay}s")
        logging.info(f"  Action space: {self.env.action_space}")
        logging.info(f"  Observation space: {self.env.observation_space}")
        logging.info(f"  TensorBoard logdir: {self.tensorboard_logdir}")
    
    def _setup_tensorboard(self):
        """
        Set up TensorBoard logging.
        Creates the logging directory and file writer with timestamp.
        """
        # Get logdir from config, default to ./logs/env if not specified
        logdir_base = self.gym_config.get('logdir', './logs/env')
        
        # Add timestamp to avoid overriding previous runs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create metrics subdirectory with timestamp
        self.tensorboard_logdir = os.path.join(logdir_base, f'{timestamp}/metrics')
        
        try:
            os.makedirs(self.tensorboard_logdir, exist_ok=True)
            logging.info(f"Created TensorBoard logging directory: {self.tensorboard_logdir}")
        except OSError as error:
            logging.error(f'Error creating TensorBoard directory: {error}')
            raise
        
        # Create TensorBoard file writer
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_logdir)
        self.file_writer.set_as_default()
        
        logging.info(f"TensorBoard file writer initialized at {self.tensorboard_logdir}")
    
    def _create_environment(self) -> gym.Env:
        """
        Create and configure the gymnasium environment.
        
        Returns:
            Configured gymnasium environment
        """
        try:
            env_name = self.gym_config['environment']
            render_mode = self.gym_config['render_mode']
            max_episode_steps = self.gym_config['max_episode_steps']
            
            # Create environment with optional render mode
            env_kwargs = {}
            if render_mode:
                env_kwargs['render_mode'] = render_mode
            
            env = gym.make(env_name, **env_kwargs)
            
            # Set max episode steps if specified
            if max_episode_steps is not None:
                env._max_episode_steps = max_episode_steps
            
            logging.info(f"Created gymnasium environment: {env_name}")
            if render_mode:
                logging.info(f"Render mode: {render_mode}")
            if max_episode_steps:
                logging.info(f"Max episode steps: {max_episode_steps}")
            
            return env
            
        except Exception as e:
            logging.error(f"Failed to create gymnasium environment '{self.gym_config['environment']}': {e}")
            raise
    
    def get_default_action(self):
        """
        Generate a default action based on the configured strategy.
        
        Returns:
            Action compatible with the environment's action space
        """
        if self.default_action_strategy == 'random':
            return self.env.action_space.sample()
        elif self.default_action_strategy == 'zero':
            if hasattr(self.env.action_space, 'shape'):
                return np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
            else:
                # Discrete action space
                return 0
        else:
            raise ValueError(f"Unknown default action strategy: {self.default_action_strategy}")
    
    def parse_action(self, message: str):
        """
        Parse action from Kafka message.
        
        Accepts two formats:
        1. {"channels": {"action": [...]}, "timestamp": ...}  (preferred)
        2. {"action": [...], "timestamp": ...}  (legacy)
        
        Args:
            message: JSON string containing action data
            
        Returns:
            Action compatible with environment's action space
            
        Raises:
            ValueError: If message cannot be parsed or action is invalid
        """
        try:
            data = json.loads(message)
            
            # Handle different message formats
            # Format 1: Channels format (preferred for consistency)
            if 'channels' in data and isinstance(data['channels'], dict):
                if 'action' in data['channels']:
                    action = data['channels']['action']
                else:
                    raise ValueError(f"No 'action' field found in channels: {data}")
            # Format 2: Direct action field (legacy)
            elif 'action' in data:
                action = data['action']
            # Format 3: Raw list/scalar (legacy)
            elif isinstance(data, (list, int, float)):
                action = data
            else:
                raise ValueError(f"No 'action' field found in message: {data}")
            
            # Convert to numpy array with correct dtype for continuous spaces
            if hasattr(self.env.action_space, 'shape'):
                if not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)
                elif action.dtype != np.float32:
                    action = action.astype(np.float32)
            
            # Validate action is in action space
            if not self.env.action_space.contains(action):
                logging.warning(f"Action {action} not in action space {self.env.action_space}")
                # Use default action instead
                return self.get_default_action()
            
            return action
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in action message: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing action: {e}")
    
    def convert_for_json(self, obj):
        """
        Convert numpy arrays and types to JSON-serializable formats.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_for_json(item) for item in obj]
        else:
            return obj
    
    def log_step_metrics(self, action, reward):
        """
        Log per-step metrics to TensorBoard.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        with self.file_writer.as_default():
            # Log step reward
            tf.summary.scalar('Step/Reward', data=reward, step=self.total_steps)
            
            # Log cumulative episode reward
            tf.summary.scalar('Step/Cumulative_Episode_Reward', 
                            data=self.episode_reward, step=self.total_steps)
            
            # Log action statistics
            if hasattr(self.env.action_space, 'shape') and len(self.env.action_space.shape) > 0:
                # Multi-dimensional action space
                action_array = np.array(action)
                tf.summary.scalar('Step/Action_Mean', 
                                data=float(np.mean(action_array)), step=self.total_steps)
                tf.summary.scalar('Step/Action_Std', 
                                data=float(np.std(action_array)), step=self.total_steps)
                tf.summary.scalar('Step/Action_Min', 
                                data=float(np.min(action_array)), step=self.total_steps)
                tf.summary.scalar('Step/Action_Max', 
                                data=float(np.max(action_array)), step=self.total_steps)
                
                # Log individual action dimensions if reasonable number
                if action_array.size <= 10:
                    for i, a in enumerate(action_array.flatten()):
                        tf.summary.scalar(f'Step/Action_Dim_{i}', 
                                        data=float(a), step=self.total_steps)
            else:
                # Scalar or discrete action
                tf.summary.scalar('Step/Action', 
                                data=float(action), step=self.total_steps)
            
            # Log episode step within current episode
            tf.summary.scalar('Step/Episode_Step', 
                            data=self.episode_step, step=self.total_steps)
    
    def log_episode_metrics(self, episode_length, episode_reward, episode_duration):
        """
        Log episode-level metrics to TensorBoard.
        
        Args:
            episode_length: Number of steps in the episode
            episode_reward: Total reward for the episode
            episode_duration: Wall-clock time for the episode
        """
        with self.file_writer.as_default():
            # Log episode metrics indexed by episode number
            tf.summary.scalar('Episode/Reward', 
                            data=episode_reward, step=self.episode_num)
            tf.summary.scalar('Episode/Length', 
                            data=episode_length, step=self.episode_num)
            tf.summary.scalar('Episode/Duration_Seconds', 
                            data=episode_duration, step=self.episode_num)
            
            # Calculate steps per second
            if episode_duration > 0:
                steps_per_sec = episode_length / episode_duration
                tf.summary.scalar('Episode/Steps_Per_Second', 
                                data=steps_per_sec, step=self.episode_num)
    
    def create_sarsa_data(self, state, action, reward, next_state, done, truncated, info):
        """
        Create SARSA topic data with native formats (no flattening).
        
        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Episode done flag
            truncated: Episode truncated flag
            info: Additional info dictionary
            
        Returns:
            Dictionary containing SARSA data in native formats
        """
        channels = {
            "state": self.convert_for_json(state),
            "action": self.convert_for_json(action),
            "reward": float(reward),
            "next_state": self.convert_for_json(next_state),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": self.convert_for_json(info)
        }
        
        return {
            "channels": channels,
            "timestamp": time.time()
        }
    
    def create_state_data(self, state):
        """
        Create state topic data containing only current state.
        
        Args:
            state: Current observation
            
        Returns:
            Dictionary containing state data
        """
        channels = {
            "state": self.convert_for_json(state)
        }
        
        return {
            "channels": channels,
            "timestamp": time.time()
        }
    
    def create_decomposed_data(self, state, action, reward, next_state, done, truncated, info):
        """
        Create decomposed topic data with flattened arrays for logging/monitoring.
        Flattens arrays while preserving shape information for reconstruction.
        
        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Episode done flag
            truncated: Episode truncated flag
            info: Additional info dictionary
            
        Returns:
            Dictionary containing decomposed data with flattened arrays
        """
        def flatten_array_with_shape(arr, prefix):
            """
            Flatten an array and return both flattened fields and shape info.
            
            Args:
                arr: Array to flatten (list or scalar)
                prefix: Field prefix (e.g., 'state', 'action')
                
            Returns:
                dict: Flattened fields and shape information
            """
            converted = self.convert_for_json(arr)
            fields = {}
            
            if isinstance(converted, list):
                # Multi-dimensional: flatten to indexed fields
                for i, val in enumerate(converted):
                    fields[f"{prefix}_{i}"] = float(val) if isinstance(val, (int, float)) else val
                
                # Add shape information
                fields[f"{prefix}_shape"] = len(converted)
                fields[f"{prefix}_is_array"] = True
                
                # Add summary statistics for numeric arrays
                if all(isinstance(x, (int, float)) for x in converted):
                    numeric_vals = [float(x) for x in converted]
                    fields[f"{prefix}_mean"] = sum(numeric_vals) / len(numeric_vals)
                    fields[f"{prefix}_min"] = min(numeric_vals)
                    fields[f"{prefix}_max"] = max(numeric_vals)
                    fields[f"{prefix}_std"] = (sum((x - fields[f"{prefix}_mean"]) ** 2 for x in numeric_vals) / len(numeric_vals)) ** 0.5
            else:
                # Scalar: store directly
                fields[prefix] = float(converted) if isinstance(converted, (int, float)) else converted
                fields[f"{prefix}_shape"] = 1
                fields[f"{prefix}_is_array"] = False
            
            return fields
        
        # Create channels dictionary with all gymnasium data
        channels = {
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "episode": self.episode_num,
            "episode_step": self.episode_step,
            "total_steps": self.total_steps,
            "environment": self.gym_config['environment']
        }
        
        # Flatten state with shape information
        state_fields = flatten_array_with_shape(state, "state")
        channels.update(state_fields)
        
        # Flatten next_state with shape information
        next_state_fields = flatten_array_with_shape(next_state, "next_state")
        channels.update(next_state_fields)
        
        # Flatten action with shape information
        action_fields = flatten_array_with_shape(action, "action")
        channels.update(action_fields)
        
        # Add reward shape info (always scalar but for consistency)
        channels["reward_shape"] = 1
        channels["reward_is_array"] = False
        
        # Add info fields if they exist and are simple types
        converted_info = self.convert_for_json(info)
        if isinstance(converted_info, dict):
            for key, value in converted_info.items():
                # Only add simple numeric/boolean values
                if isinstance(value, (int, float, bool)):
                    channels[f"info_{key}"] = float(value) if isinstance(value, (int, float)) else value
                    channels[f"info_{key}_shape"] = 1
                    channels[f"info_{key}_is_array"] = False
                elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    # Handle simple numeric arrays in info
                    info_fields = flatten_array_with_shape(value, f"info_{key}")
                    channels.update(info_fields)
        
        return {
            "channels": channels,
            "timestamp": time.time()
        }
    
    def send_state_message(self, state):
        """
        Send current state to the state topic.
        
        Args:
            state: Current observation to send
        """
        try:
            state_data = self.create_state_data(state)
            kafka_topic = self.producer.sanitize_topic_name(self.output_topics['state'])
            self.producer.send_to_kafka(kafka_topic, json.dumps(state_data))
            logging.debug(f"Sent state to topic '{kafka_topic}'")
        except Exception as e:
            logging.error(f"Error sending state message: {e}")
    
    def step_environment(self, action):
        """
        Execute one step in the environment and send results to all Kafka topics.
        
        CRITICAL: This action must be for the current state (self.current_obs).
        The sequence is:
        1. Agent generates action At for state St
        2. Gym receives action At
        3. Gym executes: St+1, reward, done = env.step(At)
        4. Gym sends SARSA(St, At, reward, St+1, done)
        5. If NOT done: Gym sends state St+1 so agent can generate At+1
        6. If done: Gym resets and sends S0 of new episode
        
        Args:
            action: Action to execute on current state
            
        Returns:
            Tuple indicating success and any outputs to send
        """
        try:
            # Reset environment if needed (first step of episode)
            if self.current_obs is None:
                self.current_obs, info = self.env.reset()
                self.episode_reward = 0.0
                self.episode_start_time = time.time()
                self.episode_step = 0
                logging.info(f"[GYM-STEP] Reset environment - Episode {self.episode_num}")
                logging.info(f"[GYM-STEP] Initial state S0: {self.current_obs}")
                
                # Send initial state so agent can generate first action
                self.send_state_message(self.current_obs)
                logging.info(f"[GYM-STEP] Sent initial state S0 to agent")
                
                # Return without stepping - wait for agent's action for S0
                return True, []
            
            # Store current state (St) before stepping
            current_state = self.current_obs.copy()
            
            logging.info("=" * 80)
            logging.info(f"[GYM-STEP] Step {self.total_steps + 1} starting")
            logging.info(f"[GYM-STEP] Current state St (step {self.total_steps}): {current_state}")
            logging.info(f"[GYM-STEP] Action At from agent: {action}")
            logging.info(f"[GYM-STEP] Executing: St+1 = env.step(At)")
            
            # Execute environment step: St+1, reward = env.step(At)
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            logging.info(f"[GYM-STEP] Next state St+1 (step {self.total_steps + 1}): {next_obs}")
            logging.info(f"[GYM-STEP] Reward: {reward:.3f}")
            logging.info(f"[GYM-STEP] Done: {done}, Truncated: {truncated}")
            
            # Update episode reward and counters
            self.episode_reward += reward
            self.episode_step += 1
            self.total_steps += 1
            
            # Log per-step metrics to TensorBoard
            self.log_step_metrics(action, reward)
            
            # Create SARSA tuple: (St, At, Rt, St+1, done)
            sarsa_data = self.create_sarsa_data(
                current_state, action, reward, next_obs, done, truncated, info
            )
            
            # Create decomposed data for monitoring
            decomposed_data = self.create_decomposed_data(
                current_state, action, reward, next_obs, done, truncated, info
            )
            
            logging.info(f"[GYM-STEP] Created SARSA tuple:")
            logging.info(f"  State St (step {self.total_steps - 1}): {current_state[:3]}...")
            logging.info(f"  Action At: {action}")
            logging.info(f"  Reward Rt: {reward:.3f}")
            logging.info(f"  Next_state St+1 (step {self.total_steps}): {next_obs[:3]}...")
            logging.info(f"  Done: {done or truncated}")
            
            # Handle episode end vs. continuation differently
            if done or truncated:
                # Episode ended - log metrics
                episode_duration = time.time() - self.episode_start_time
                
                # Log episode-level metrics to TensorBoard
                self.log_episode_metrics(self.episode_step, self.episode_reward, episode_duration)
                
                logging.info(f"[GYM-EPISODE] Episode {self.episode_num} FINISHED")
                logging.info(f"  Total steps: {self.episode_step}")
                logging.info(f"  Total reward: {self.episode_reward:.3f}")
                logging.info(f"  Duration: {episode_duration:.2f}s")
                logging.info(f"  Done: {done}, Truncated: {truncated}")
                
                # CRITICAL: Reset immediately and send S0 of new episode
                # Don't send the terminal state (St+1) since agent can't act on it
                self.current_obs, info = self.env.reset()
                self.episode_num += 1
                self.episode_step = 0
                self.episode_reward = 0.0
                self.episode_start_time = time.time()
                
                logging.info(f"[GYM-STEP] Auto-reset for Episode {self.episode_num}")
                logging.info(f"[GYM-STEP] New initial state S0: {self.current_obs}")
                
                # Send S0 of new episode so agent can generate first action
                self.send_state_message(self.current_obs)
                logging.info(f"[GYM-STEP] Sent initial state S0 of Episode {self.episode_num} to agent")
                
            else:
                # Episode continuing - send next state
                self.current_obs = next_obs
                
                # Send next state (St+1) so agent can generate next action (At+1)
                self.send_state_message(self.current_obs)
                logging.info(f"[GYM-STEP] Sent next state St+1 (step {self.total_steps}) to agent")
            
            logging.info(f"[GYM-STEP] Step {self.total_steps} complete")
            logging.info("=" * 80)
            
            # Add delay if specified
            if self.step_delay > 0:
                time.sleep(self.step_delay)
            
            # Prepare outputs for Kafka (SARSA and decomposed topics)
            outputs = [
                (self.producer.sanitize_topic_name(self.output_topics['sarsa']), json.dumps(sarsa_data)),
                (self.producer.sanitize_topic_name(self.output_topics['decomposed']), json.dumps(decomposed_data))
            ]
            
            return True, outputs
            
        except Exception as e:
            logging.error(f"[GYM-STEP] Error stepping environment: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False, []
    
    def step_with_default_action(self):
        """
        Execute environment step with default action and send results to Kafka.
        Used in non-blocking mode when no Kafka action is received.
        """
        try:
            action = self.get_default_action()
            success, outputs = self.step_environment(action)
            
            # Send outputs directly since we're not returning from process_message
            if success and outputs:
                for topic, message in outputs:
                    self.producer.send_to_kafka(topic, message)
                    
        except Exception as e:
            logging.error(f"Error in default action step: {e}")
    
    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process action message from Kafka and execute environment step.
        
        This method is called by KafkaStreamingProcessBase when a message is received.
        
        Args:
            message: The message value (JSON string with action)
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            Tuple[bool, List[Tuple]]: Success status and list of outputs to send to Kafka
        """
        try:
            # Parse action from Kafka message
            action = self.parse_action(message)
            logging.debug(f"Received action from Kafka: {action}")
            
            # Execute environment step
            return self.step_environment(action)
            
        except ValueError as e:
            logging.error(f"Invalid action message from topic {topic}: {e}")
            logging.error(f"Message content: {message}")
            return False, []
        except Exception as e:
            logging.error(f"Error processing action message: {e}")
            return False, []
    
    def start(self):
        """
        Start the Kafka Gym wrapper.
        
        This extends the base class start method to handle initial environment reset.
        """
        try:
            logging.info("Starting Kafka Gym wrapper...")
            
            # Call parent start method which sets up Kafka and begins consumption
            super().start()
            
        except Exception as e:
            logging.error(f"Error starting Kafka Gym wrapper: {e}")
            self.cleanup()
            raise
    
    def consume_messages(self):
        """
        Main consumption loop with support for blocking and non-blocking modes.
        
        Overrides the base class method to add default action handling
        when operating in non-blocking mode.
        """
        logging.info(f"Starting Kafka Gym wrapper loop (blocking_mode={self.blocking_mode})...")
        
        # Reset environment if configured to do so
        if self.reset_on_start:
            self.current_obs, info = self.env.reset()
            self.episode_reward = 0.0
            self.episode_start_time = time.time()
            logging.info("Environment reset on startup")
            # Send initial state immediately after reset
            self.send_state_message(self.current_obs)


        while self.running:
            try:
                # Poll for messages with timeout
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if message_batch:
                    # Process Kafka actions normally
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            try:
                                success, outputs = self.process_message(
                                    message=message.value,
                                    topic=message.topic,
                                    partition=message.partition,
                                    offset=message.offset
                                )
                                
                                if not success:
                                    logging.warning(f"Message processing failed for topic {message.topic}, offset {message.offset}")
                                    continue
                                
                                # Send outputs to Kafka
                                if outputs:
                                    for output in outputs:
                                        try:
                                            if len(output) == 2:
                                                topic, message_content = output
                                                key = None
                                            elif len(output) == 3:
                                                topic, message_content, key = output
                                            else:
                                                raise ValueError(f"Invalid output tuple length: {len(output)}")
                                            
                                            record_metadata = self.producer.send_to_kafka(topic, message_content, key)
                                            logging.debug(f"Sent step data to topic '{topic}' - partition {record_metadata.partition}, offset {record_metadata.offset}")
                                            
                                        except Exception as e:
                                            logging.error(f"Failed to send output tuple {output}: {e}")
                                
                            except Exception as e:
                                logging.error(f"Error processing message from topic {message.topic}: {e}")
                                self.handle_processing_error(e, message)
                else:
                    # No messages received
                    if self.blocking_mode:
                        # In blocking mode, just continue waiting
                        continue
                    else:
                        # In non-blocking mode, use default action
                        self.step_with_default_action()
                
            except Exception as e:
                logging.error(f"Error in consumption loop: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """
        Clean up environment, TensorBoard, and Kafka resources.
        """
        # Close TensorBoard file writer
        if hasattr(self, 'file_writer') and self.file_writer:
            try:
                self.file_writer.close()
                logging.info("TensorBoard file writer closed")
            except Exception as e:
                logging.error(f"Error closing TensorBoard file writer: {e}")
        
        # Close gymnasium environment
        if hasattr(self, 'env') and self.env:
            try:
                self.env.close()
                logging.info("Gymnasium environment closed")
            except Exception as e:
                logging.error(f"Error closing gymnasium environment: {e}")
        
        # Call base class cleanup for Kafka resources
        super().cleanup()


def main():
    """
    Main entry point for the Gymnasium Kafka controller.
    """

    setup_logging()

    logging.info("Starting Gymnasium Kafka Controller...")
    
    try:
        # Create and start the controller
        controller = KafkaGymWrapper()
        controller.start()
        
    except KeyboardInterrupt:
        logging.info("Received shutdown signal...")
    except Exception as e:
        logging.error(f"Error in Gymnasium controller: {e}")
        raise
    finally:
        logging.info("Gymnasium Kafka Controller stopped")


if __name__ == "__main__":
    main()  