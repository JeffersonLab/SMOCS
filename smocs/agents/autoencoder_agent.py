import os
import glob
import json
import time
import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from smocs.cores import AgentBase, DataIngestThreadBase, MLTrainingThreadBase, MLInferenceThreadBase
from smocs.utils import ConfigLoader, extract_sensor_values

logging.basicConfig(level=logging.DEBUG)

class AutoencoderDataIngestThread(DataIngestThreadBase):
    """
    Data ingestion thread for autoencoder agent.
    Stores raw time series sensor data to database.
    """
    
    def store_message(self, message, topic, partition, offset) -> bool:
        """
        Parse sensor message and store raw data to database.
        
        Args:
            message: JSON string with sensor readings
            topic: Kafka topic name
            partition: Kafka partition
            offset: Kafka offset
            
        Returns:
            bool: True if storage successful
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            data = json.loads(message)
            
            # Extract timestamp and sensor channels
            if 'timestamp' in data:
                timestamp = datetime.fromtimestamp(data['timestamp'])
            else:
                timestamp = datetime.now()
            
            # Get sensor readings from channels
            channels = data.get('channels', {})
            if not channels:
                logging.warning(f"AEDataIngestThread: No channels found in message: {data}")
                return False
                    
            state_keys, state_values = extract_sensor_values(channels, topic)
            
            sensor_values = np.array(state_values, dtype=np.float32)
            
            # Store in database using existing schema
            sensor_data = {
                'state_source_timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'state_received_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'state': sensor_values
            }
            
            status = self.db_manager.record_sensor_data(sensor_data)
            
            if status == 0:
                logging.debug(f"AEDataIngestThread: Stored sensor data: {len(sensor_values)} channels at {timestamp}")
                return True
            else:
                logging.error(f"AEDataIngestThread: Failed to store sensor data, status: {status}")
                return False
                
        except Exception as e:
            logging.error(f"AEDataIngestThread: Error storing message: {e}")
            return False

class AutoencoderMLTrainingThread(MLTrainingThreadBase):
    """
    ML training thread for autoencoder agent.
    Creates sliding windows from database data and trains TensorFlow autoencoder.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        # Configuration parameters
        self.window_size = config.get('window_size', 50)  # 50 timesteps per window
        self.min_training_samples = config.get('min_training_samples', 10000)
        
        # Model architecture config
        self.encoder_dims = config.get('encoder_dims', [32, 16])  # Hidden layer sizes
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
        
        # Model state
        self.model = None
        self.input_dim = None
        self.last_training_count = 0
        
        super().__init__(agent_id, config)
    
    def build_model(self):
        """Build TensorFlow autoencoder model."""
        # Will build model once we know input dimensions from data
        logging.info("AEMLTrainingThread: Autoencoder model will be built when input dimensions are determined")
    
    def _create_autoencoder(self, input_dim: int):
        """
        Create autoencoder with specified input dimension.
        
        Args:
            input_dim: Size of input/output layer (window_size * n_sensors)
        """
        # Build encoder
        encoder_layers = []
        encoder_layers.append(layers.Dense(self.encoder_dims[0], activation='relu', input_shape=(input_dim,)))
        
        for dim in self.encoder_dims[1:]:
            encoder_layers.append(layers.Dense(dim, activation='relu'))
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        for dim in reversed(self.encoder_dims[:-1]):
            decoder_layers.append(layers.Dense(dim, activation='relu'))
        
        decoder_layers.append(layers.Dense(input_dim, activation='linear'))  # Output layer
        
        # Create full autoencoder
        autoencoder = keras.Sequential([
            *encoder_layers,
            *decoder_layers
        ])
        
        # Compile model
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = autoencoder
        self.input_dim = input_dim
        
        logging.info(f"AEMLTrainingThread: Built autoencoder: input_dim={input_dim}, encoder_dims={self.encoder_dims}")
        logging.info(f"AEMLTrainingThread: Model summary: {autoencoder.summary()}")
    
    def get_training_data(self) -> Optional[np.ndarray]:
        """
        Retrieve and preprocess training data from database.
        
        Returns:
            Windowed training data or None if insufficient data
        """
        try:
            # Check if enough data available
            total_samples = self.db_manager.get_size("agent_inferences")
            
            if total_samples < self.min_training_samples:
                logging.debug(f"AEMLTrainingThread: Not enough samples for training: {total_samples} < {self.min_training_samples}")
                return None
            
            # Check if we have new data since last training
            if total_samples <= self.last_training_count:
                logging.debug("AEMLTrainingThread: No new data since last training")
                return None
            
            # Get recent sensor data from database - specify agent_type="diagnostics"
            batch_data = self.db_manager.sample_batch(
                batch_size=self.batch_size*100, # Change this
                segment_length=self.window_size,  # Use exact window size, not 2x
                agent_type="diagnostics",
                mode="latest"
            )
            logging.info(f"sample_batch returned {len(batch_data['state'])} sequences")
            
            if batch_data is None or len(batch_data['state']) == 0:
                return None
            
            windows = []
            for sequence in batch_data['state']:
                # Each sequence is already the right length (window_size)
                if len(sequence) == self.window_size:
                    try:
                        # Convert each state in sequence to numpy array first to ensure consistent shape
                        processed_sequence = []
                        for state in sequence:
                            if isinstance(state, np.ndarray):
                                processed_sequence.append(state.flatten())
                            else:
                                # Convert to numpy array if it's not already
                                state_array = np.array(state, dtype=np.float32)
                                processed_sequence.append(state_array.flatten())
                        
                        # Ensure all states in sequence have the same length
                        if len(processed_sequence) > 0:
                            expected_length = len(processed_sequence[0])
                            valid_sequence = True
                            for i, state in enumerate(processed_sequence):
                                if len(state) != expected_length:
                                    logging.warning(f"AEMLTrainingThread: Inconsistent state length at position {i}: expected {expected_length}, got {len(state)}")
                                    valid_sequence = False
                                    break
                            
                            if valid_sequence:
                                # Flatten the entire window (sequence of states)
                                window = np.concatenate(processed_sequence)
                                windows.append(window)
                            else:
                                logging.warning(f"AEMLTrainingThread: Skipping sequence with inconsistent state shapes")
                        
                    except Exception as seq_error:
                        logging.warning(f"AEMLTrainingThread: Error processing sequence: {seq_error}")
                        continue
                else:
                    logging.debug(f"AEMLTrainingThread: Skipping sequence with length {len(sequence)}, expected {self.window_size}")
            
            if not windows:
                logging.error("AEMLTrainingThread: No valid windows found after processing")
                return None
            
            # Ensure all windows have the same length before creating array
            if len(windows) > 0:
                expected_window_length = len(windows[0])
                filtered_windows = []
                for i, window in enumerate(windows):
                    if len(window) == expected_window_length:
                        filtered_windows.append(window)
                    else:
                        logging.warning(f"AEMLTrainingThread: Skipping window {i} with length {len(window)}, expected {expected_window_length}")
                
                windows = filtered_windows
            
            if not windows:
                logging.error("AEMLTrainingThread: No windows with consistent length found")
                return None
            
            windowed_array = np.array(windows, dtype=np.float32)
            logging.info(f"Training data shape: {windowed_array.shape}")
            logging.info(f"Training data range: [{np.min(windowed_array):.6f}, {np.max(windowed_array):.6f}]")
            logging.info(f"Training data mean: {np.mean(windowed_array):.6f}")
            logging.info(f"Training data std: {np.std(windowed_array):.6f}")
            logging.info(f"Any NaN values: {np.isnan(windowed_array).any()}")
            logging.info(f"Any infinite values: {np.isinf(windowed_array).any()}")
            
            self.last_training_count = total_samples
            
            logging.info(f"AEMLTrainingThread: Prepared {len(windowed_array)} training windows from {total_samples} total samples")
            return windowed_array
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error getting training data: {e}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return None
    
    def train_model(self, training_data: np.ndarray) -> Dict[str, Any]:
        """
        Train the autoencoder model.
        
        Args:
            training_data: Windowed sensor data
            
        Returns:
            Training metrics
        """
        try:
            # Add normalization
            self.data_mean = np.mean(training_data, axis=0)
            self.data_std = np.std(training_data, axis=0)
            
            # Avoid division by zero
            self.data_std[self.data_std == 0] = 1.0
            
            normalized_data = (training_data - self.data_mean) / self.data_std
            # Build model if not exists
            if self.model is None:
                input_dim = training_data.shape[1]
                self._create_autoencoder(input_dim)
            
            # Train autoencoder (input = output for reconstruction)
            history = self.model.fit(
                normalized_data,
                normalized_data,  # Autoencoder learns to reconstruct input
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                verbose=0
            )
            
            # Extract training metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            metrics = {
                'loss': float(final_loss),
                'val_loss': float(final_val_loss),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(training_data)
            }
            
            logging.info(f"AEMLTrainingThread: Training completed: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error training model: {e}")
            return {'error': str(e)}
    
    def eval_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                return {'error': 'No model to evaluate'}
            
            # Get small batch of recent data for evaluation
            eval_data = self.get_training_data()
            if eval_data is None:
                return {'error': 'No evaluation data available'}
            
            # Use subset for evaluation
            eval_subset = eval_data[:min(100, len(eval_data))]
            
            # Normalize evaluation data using training statistics
            if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
                normalized_eval = (eval_subset - self.data_mean) / self.data_std
            else:
                # Fallback normalization if training stats not available
                normalized_eval = eval_subset
                logging.warning("AEMLTrainingThread: No training normalization stats available for evaluation")
            
            # Evaluate reconstruction error
            reconstructions = self.model.predict(normalized_eval, verbose=0)
            mse_errors = np.mean((normalized_eval - reconstructions) ** 2, axis=1)
            
            eval_metrics = {
                'mean_reconstruction_error': float(np.mean(mse_errors)),
                'std_reconstruction_error': float(np.std(mse_errors)),
                'max_reconstruction_error': float(np.max(mse_errors)),
                'anomaly_threshold_95': float(np.percentile(mse_errors, 95)),
                'eval_samples': len(eval_subset)
            }
            
            logging.info(f"AEMLTrainingThread: Model evaluation: mean_error={eval_metrics['mean_reconstruction_error']:.4f}")
            
            return eval_metrics
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error evaluating model: {e}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """
        Save trained model to local directory with atomic writes.
        Always saves and updates latest model.
        
        Args:
            model_metrics: Training metrics
            eval_results: Evaluation results
        """
        try:
            if self.model is None:
                logging.error("AEMLTrainingThread: No model to save")
                return
            
            # Create models directory
            models_dir = "/app/models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Find next version number
            existing_models = glob.glob(f"{models_dir}/model_v*.h5")
            if existing_models:
                versions = [int(f.split('_v')[1].split('.')[0]) for f in existing_models]
                next_version = max(versions) + 1
            else:
                next_version = 1
            
            version_str = f"{next_version:03d}"
            model_file = f"{models_dir}/model_v{version_str}.h5"
            latest_file = f"{models_dir}/latest_model.json"
            
            # Use temporary files for atomic writes
            model_tmp = f"{model_file}.tmp"
            latest_tmp = f"{latest_file}.tmp"
            
            # Save model to temporary file
            self.model.save(model_tmp)
            
            # Prepare metadata
            metadata = {
                'version': next_version,
                'model_file': f"model_v{version_str}.h5",
                'input_dim': self.input_dim,
                'architecture_config': {
                    'encoder_dims': self.encoder_dims,
                    'window_size': self.window_size,
                    'learning_rate': self.learning_rate
                },
                'training_metrics': model_metrics,
                'eval_metrics': eval_results,
                'timestamp': time.time(),
                'data_mean': self.data_mean.tolist() if hasattr(self, 'data_mean') else None,
                'data_std': self.data_std.tolist() if hasattr(self, 'data_std') else None
            }
            
            # Always save as latest model
            with open(latest_tmp, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic renames
            os.rename(model_tmp, model_file)
            os.rename(latest_tmp, latest_file)
            
            logging.info(f"AEMLTrainingThread: Model v{version_str} saved as latest (val_loss: {eval_results.get('val_loss', 'N/A')})")
                
        except Exception as e:
            # Clean up temporary files
            for tmp_file in [model_tmp, latest_tmp]:
                if 'tmp_file' in locals() and os.path.exists(tmp_file):
                    os.remove(tmp_file)
            logging.error(f"AEMLTrainingThread: Error saving model: {e}")
    
    def _send_training_results(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """
        Send training results to Kafka in validated format.
        
        Args:
            model_metrics: Training metrics
            eval_results: Evaluation results
        """
        try:
            output_topic = self.config.get('kafka_topics', {}).get('training_output', 'autoencoder-training-results')
            
            # Create properly formatted message with timestamp and channels
            channels = {
                'agent_id': self.agent_id,
                'model_version': eval_results.get('version', 1),
                'training_loss': model_metrics.get('loss', 0.0),
                'validation_loss': model_metrics.get('val_loss', 0.0),
                'epochs_trained': model_metrics.get('epochs_trained', 0),
                'training_samples': model_metrics.get('training_samples', 0),
                'mean_reconstruction_error': eval_results.get('mean_reconstruction_error', 0.0),
                'std_reconstruction_error': eval_results.get('std_reconstruction_error', 0.0),
                'anomaly_threshold_95': eval_results.get('anomaly_threshold_95', 0.0),
                'eval_samples': eval_results.get('eval_samples', 0),
                'input_dim': self.input_dim or 0,
                'window_size': self.window_size
            }
            
            # Add error information if present
            if 'error' in model_metrics:
                channels['training_error'] = str(model_metrics['error'])
            if 'error' in eval_results:
                channels['eval_error'] = str(eval_results['error'])
            
            message = {
                'timestamp': time.time(),
                'channels': channels
            }
            
            kafka_topic = self.sanitize_topic_name(output_topic)
            self.send_to_kafka(kafka_topic, json.dumps(message))
            
            logging.info(f"AEMLTrainingThread: Sent training results to topic '{kafka_topic}'")
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error sending training results to Kafka: {e}")

class AutoencoderMLInferenceThread(MLInferenceThreadBase):
    """
    ML inference thread for autoencoder agent.
    Performs anomaly detection on streaming sensor data.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.window_size = config.get('window_size', 50)
        self.anomaly_threshold = None
        self.model = None
        self.input_dim = None
        self.recent_data = []
        self.current_model_version = None
        self.last_model_check = 0
        self.model_check_interval = config.get('model_check_interval', 30)  # seconds
        self.data_mean = None
        self.data_std = None
        
        super().__init__(agent_id, config)
   
    def load_model(self):
        """Load the latest autoencoder model from local directory."""
        try:
            models_dir = "/app/models"
            latest_file = f"{models_dir}/latest_model.json"
            
            if not os.path.exists(latest_file):
                logging.warning("AEMLInferenceThread: No latest model file found")
                return
            
            # Read latest model info
            with open(latest_file, 'r') as f:
                latest_info = json.load(f)
            
            model_version = latest_info['version']
            
            # Check if we already have this version loaded
            if self.current_model_version == model_version:
                return
            
            model_file = f"{models_dir}/{latest_info['model_file']}"
            
            if not os.path.exists(model_file):
                logging.error(f"AEMLInferenceThread: Model file not found: {model_file}")
                return
            
            # Load the TensorFlow model
            self.model = tf.keras.models.load_model(model_file)
            
            # Update instance variables from metadata
            self.input_dim = latest_info['input_dim']
            self.anomaly_threshold = latest_info.get('eval_metrics', {}).get('anomaly_threshold_95', 0.1)
            self.current_model_version = model_version
            
            # Load normalization parameters if available
            if latest_info.get('data_mean') and latest_info.get('data_std'):
                self.data_mean = np.array(latest_info['data_mean'])
                self.data_std = np.array(latest_info['data_std'])
            
            logging.info(f"AEMLInferenceThread: Loaded model v{model_version}: input_dim={self.input_dim}, threshold={self.anomaly_threshold}")
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error loading model: {e}")

    def check_for_model_updates(self):
        """Check if a new model is available and load if necessary."""
        current_time = time.time()
        if current_time - self.last_model_check > self.model_check_interval:
            self.load_model()
            self.last_model_check = current_time

    def perform_inference(self, inference_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform anomaly detection inference with automatic model updates.
        
        Args:
            inference_request: Parsed sensor data
            
        Returns:
            Inference results with reconstruction and anomaly score
        """
        try:
            # Check for model updates periodically
            self.check_for_model_updates()
            
            if self.model is None:
                self.load_model()  # Try loading if no model loaded
                if self.model is None:
                    logging.warning("AEMLInferenceThread: No model available for inference")
                    return None
            
            sensor_values = inference_request['sensor_values']
            
            # Add to recent data buffer
            self.recent_data.append(sensor_values)
            
            # Keep buffer at window size
            if len(self.recent_data) > self.window_size * 2:
                self.recent_data = self.recent_data[-self.window_size * 2:]
            
            # Need at least window_size samples for inference
            if len(self.recent_data) < self.window_size:
                return {
                    'reconstruction': sensor_values.tolist(),
                    'error_score': 0.0,
                    'is_anomaly': False,
                    'status': 'insufficient_data',
                    'buffer_size': len(self.recent_data)
                }
            
            # Create inference window with consistent shape handling
            try:
                window_data = []
                for state in self.recent_data[-self.window_size:]:
                    if isinstance(state, np.ndarray):
                        window_data.append(state.flatten())
                    else:
                        state_array = np.array(state, dtype=np.float32)
                        window_data.append(state_array.flatten())
                
                # Ensure all states have the same length
                if len(window_data) > 0:
                    expected_length = len(window_data[0])
                    for i, state in enumerate(window_data):
                        if len(state) != expected_length:
                            logging.error(f"AEMLInferenceThread: Inconsistent state length at position {i}: expected {expected_length}, got {len(state)}")
                            return {
                                'error': f'Inconsistent state shapes in window',
                                'status': 'error'
                            }
                
                # Flatten the entire window
                flattened_window = np.concatenate(window_data).reshape(1, -1)
                
            except Exception as window_error:
                logging.error(f"AEMLInferenceThread: Error creating inference window: {window_error}")
                return {
                    'error': f'Window creation failed: {str(window_error)}',
                    'status': 'error'
                }
            
            # Normalize using training statistics if available
            if self.data_mean is not None and self.data_std is not None:
                try:
                    if flattened_window.shape[1] != len(self.data_mean):
                        logging.error(f"AEMLInferenceThread: Window size mismatch - expected {len(self.data_mean)}, got {flattened_window.shape[1]}")
                        return {
                            'error': 'Window size mismatch with trained model',
                            'status': 'error'
                        }
                    normalized_window = (flattened_window - self.data_mean) / self.data_std
                except Exception as norm_error:
                    logging.error(f"AEMLInferenceThread: Normalization error: {norm_error}")
                    normalized_window = flattened_window
            else:
                # Fallback to simple normalization
                window_min, window_max = flattened_window.min(), flattened_window.max()
                if window_max > window_min:
                    normalized_window = (flattened_window - window_min) / (window_max - window_min)
                else:
                    normalized_window = flattened_window
            
            # Get reconstruction
            reconstruction = self.model.predict(normalized_window, verbose=0)
            
            # Denormalize reconstruction
            if self.data_mean is not None and self.data_std is not None:
                reconstruction = reconstruction * self.data_std + self.data_mean
            elif 'window_min' in locals() and window_max > window_min:
                reconstruction = reconstruction * (window_max - window_min) + window_min
            
            # Compute reconstruction error
            error_score = float(np.mean((flattened_window - reconstruction) ** 2))
            
            # Determine if anomaly
            is_anomaly = error_score > self.anomaly_threshold if self.anomaly_threshold else False
            
            result = {
                'reconstruction': reconstruction.flatten().tolist(),
                'original_window': flattened_window.flatten().tolist(),
                'error_score': error_score,
                'is_anomaly': is_anomaly,
                'anomaly_threshold': self.anomaly_threshold,
                'model_version': self.current_model_version,
                'timestamp': inference_request['timestamp'],
                'status': 'success'
            }
            
            if is_anomaly:
                logging.warning(f"AEMLInferenceThread: Anomaly detected: error_score={error_score:.4f} > threshold={self.anomaly_threshold:.4f}")
            
            return result
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error performing inference: {e}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return {'error': str(e), 'status': 'error'}

    def parse_inference_request(self, message, topic, partition, offset) -> Optional[Dict[str, Any]]:
        """
        Parse inference request from sensor message.
        
        Args:
            message: Sensor data message
            topic: Kafka topic
            partition: Kafka partition  
            offset: Kafka offset
            
        Returns:
            Parsed sensor data or None
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            data = json.loads(message)
            
            channels = data.get('channels', {})
            if not channels:
                logging.warning(f"AEMLInferenceThread: No channels found in message: {data}")
                return False
            
            state_keys, state_values = extract_sensor_values(channels, topic)

            # Convert to numpy array for storage
            sensor_values = np.array(state_values, dtype=np.float32)
            
            return {
                'sensor_values': sensor_values,
                'timestamp': data.get('timestamp', time.time()),
                'channels': channels
            }
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error parsing inference request: {e}")
            return None

    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process incoming message and return inference results in validated format.
        
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
            
            # Format result for Kafka in validated format
            channels = {
                'agent_id': self.agent_id,
                'model_version': inference_result.get('model_version', 0),
                'error_score': inference_result.get('error_score', 0.0),
                'is_anomaly': inference_result.get('is_anomaly', False),
                'anomaly_threshold': inference_result.get('anomaly_threshold', 0.0),
                'buffer_size': inference_result.get('buffer_size', len(self.recent_data)),
                'status': inference_result.get('status', 'unknown'),
                'input_topic': topic
            }
            
            # Add error information if present
            if 'error' in inference_result:
                channels['error'] = str(inference_result['error'])
            
            # Add reconstruction statistics if available
            if 'reconstruction' in inference_result:
                reconstruction = inference_result['reconstruction']
                if isinstance(reconstruction, list) and reconstruction:
                    channels['reconstruction_mean'] = float(np.mean(reconstruction))
                    channels['reconstruction_std'] = float(np.std(reconstruction))
                    channels['reconstruction_min'] = float(np.min(reconstruction))
                    channels['reconstruction_max'] = float(np.max(reconstruction))
            
            output_message = {
                'timestamp': inference_result.get('timestamp', time.time()),
                'channels': channels
            }
            
            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error processing inference message: {e}")
            return False, []

    def _store_inference_result(self, inference_request: Any, inference_result: Any):
        """Store inference result to database."""
        try:
            # This would use DBManager to store the inference result
            # Implementation depends on specific data structure
            pass
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error storing inference result: {e}")

class AutoencoderAgent(AgentBase):
    """
    Autoencoder agent for time series anomaly detection.
    """
    
    def __init__(self, config_path: str = None):
        super().__init__("AutoencoderAgent")
        
        # Load configuration
        if config_path:
            config_loader = ConfigLoader(config_path)
            autoencoder_config = config_loader.config.get('autoencoder_agent', {})
        else:
            # Default configuration
            autoencoder_config = {
                'window_size': 10,
                'min_training_samples': 1000,
                'encoder_dims': [32, 16],
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'kafka_topics': {
                    'input': 'gymnasium-output',
                    'output': 'autoencoder-anomalies',
                    'training_output': 'autoencoder-training-results'
                }
            }
        
        # Ensure kafka_topics are included in the config passed to threads
        if 'kafka_topics' not in autoencoder_config:
            logging.warning("AEAgent: No kafka_topics found in config, using defaults")
            autoencoder_config['kafka_topics'] = {
                'input': 'gymnasium-output',
                'output': 'autoencoder-anomalies',
                'training_output': 'autoencoder-training-results'
            }
        
        # Add agent_id to config for threads to use
        self.agent_config = autoencoder_config.copy()
        self.agent_config['agent_id'] = self.agent_id
        
        logging.info(f"AEAgent: AutoencoderAgent initialized with config: {self.agent_config}")
    
    def create_data_ingest_component(self):
        """Create data ingestion thread component."""
        return AutoencoderDataIngestThread(self.agent_id, self.agent_config)
    
    def create_ml_training_component(self):
        """Create ML training thread component."""
        return AutoencoderMLTrainingThread(self.agent_id, self.agent_config)
    
    def create_ml_inference_component(self):
        """Create ML inference thread component."""
        return AutoencoderMLInferenceThread(self.agent_id, self.agent_config)

def main():
    """Main entry point for autoencoder agent."""
    import os
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')
    
    try:
        agent = AutoencoderAgent(config_path)
        agent.start()
    except KeyboardInterrupt:
        logging.info("Shutting down autoencoder agent...")
    except Exception as e:
        logging.error(f"Error running autoencoder agent: {e}")
        raise

if __name__ == "__main__":
    main()