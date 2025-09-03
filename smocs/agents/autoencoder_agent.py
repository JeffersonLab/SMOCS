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
from smocs.utils import ConfigLoader

logging.basicConfig(level=logging.INFO)


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
                    
            state_keys = [k for k in channels.keys()
                        if k.startswith('state_')
                        and k != 'state_shape'
                        and k != 'state_is_array'
                        and isinstance(channels[k], (int, float))]
            
            state_keys.sort()

            state_values = []
            for key in state_keys:
                try:
                    value = float(channels[key])
                    state_values.append(value)
                except (ValueError, TypeError):
                    logging.warning(f"AEDataIngestThread: Skipping non-numeric state value: {key}={channels[key]}")
                    continue
            
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
                    # Convert sequence to numpy array and flatten
                    window = np.array([state for state in sequence]).flatten()
                    windows.append(window)
            
            if not windows:
                return None
            
            windowed_array = np.array(windows)

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
            
            # Evaluate reconstruction error
            reconstructions = self.model.predict(eval_subset, verbose=0)
            mse_errors = np.mean((eval_subset - reconstructions) ** 2, axis=1)
            
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
            import tensorflow as tf
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
            
            # Create inference window
            window = np.array(self.recent_data[-self.window_size:])
            flattened_window = window.flatten().reshape(1, -1)
            
            # Normalize using training statistics if available
            if self.data_mean is not None and self.data_std is not None:
                normalized_window = (flattened_window - self.data_mean) / self.data_std
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
                return None

            state_keys = [k for k in channels.keys()
                        if k.startswith('state_')
                        and k != 'state_shape'
                        and k != 'state_is_array'
                        and isinstance(channels[k], (int, float))]
            
            state_keys.sort()

            state_values = []
            for key in state_keys:
                try:
                    value = float(channels[key])
                    state_values.append(value)
                except (ValueError, TypeError):
                    logging.warning(f"AEMLInferenceThread: Skipping non-numeric state value: {key}={channels[key]}")
                    continue

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


class AutoencoderAgent(AgentBase):
    """
    Autoencoder agent for time series anomaly detection.
    TODO: Make configurable and generic with registation for component thread classes allowing for configuration based instantiation of an agent
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
        level=logging.INFO,
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