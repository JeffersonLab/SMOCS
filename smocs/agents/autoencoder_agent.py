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

            # Convert to numpy array for storage
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
        self.min_training_samples = config.get('min_training_samples', 1000)
        
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
            # TODO: Go back and check if I need the sliding window internal function call now that I am recieving data (Check the shapes to make sure it is working correctly)
            batch_data = self.db_manager.sample_batch(
                batch_size=min(10, total_samples // self.window_size),
                segment_length=max(self.window_size * 2, 100),
                agent_type="diagnostics",  # This is now properly supported
                mode="latest"
            )
            
            if batch_data is None or len(batch_data['state']) == 0:
                logging.warning("AEMLTrainingThread: No training data retrieved from database")
                return None
            
            # Convert to windowed format
            windowed_data = self._create_sliding_windows(batch_data['state'])
            
            if windowed_data is None or len(windowed_data) == 0:
                logging.warning("AEMLTrainingThread: No valid windows created from training data")
                return None
            
            self.last_training_count = total_samples
            
            logging.info(f"AEMLTrainingThread: Prepared {len(windowed_data)} training windows from {total_samples} total samples")
            return windowed_data
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error getting training data: {e}")
            return None
    
    def _create_sliding_windows(self, time_series_data: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create sliding windows from time series data.
        
        Args:
            time_series_data: List of sensor reading arrays from database
            
        Returns:
            Array of windowed data for training
        """
        try:
            windows = []
            
            for sequence in time_series_data:
                # sequence is a list of sensor readings over time
                if len(sequence) < self.window_size:
                    continue
                
                # Convert to numpy array if needed
                if isinstance(sequence[0], np.ndarray):
                    # Each element is already a sensor reading array
                    sensor_data = np.array([reading for reading in sequence])
                else:
                    # Convert to proper format
                    sensor_data = np.array(sequence)
                
                # Create sliding windows
                for i in range(len(sensor_data) - self.window_size + 1):
                    window = sensor_data[i:i + self.window_size]
                    # Flatten window to vector for dense autoencoder
                    flattened_window = window.flatten()
                    windows.append(flattened_window)
            
            if not windows:
                return None
            
            windowed_array = np.array(windows)
            
            # Normalize data to [0, 1] range
            windowed_array = (windowed_array - windowed_array.min()) / (windowed_array.max() - windowed_array.min() + 1e-8)
            
            return windowed_array
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error creating sliding windows: {e}")
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
            # Build model if not exists
            if self.model is None:
                input_dim = training_data.shape[1]
                self._create_autoencoder(input_dim)
            
            # Train autoencoder (input = output for reconstruction)
            history = self.model.fit(
                training_data,
                training_data,  # Autoencoder learns to reconstruct input
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
        Save trained model to database.
        
        Args:
            model_metrics: Training metrics
            eval_results: Evaluation results
        """
        try:
            if self.model is None:
                logging.error("AEMLTrainingThread: No model to save")
                return
            
            # Serialize model
            model_bytes = pickle.dumps({
                'model_weights': self.model.get_weights(),
                'model_config': self.model.get_config(),
                'input_dim': self.input_dim,
                'architecture_config': {
                    'encoder_dims': self.encoder_dims,
                    'window_size': self.window_size,
                    'learning_rate': self.learning_rate
                },
                'training_metrics': model_metrics,
                'eval_metrics': eval_results,
                'timestamp': time.time()
            })
            
            # Store model in database (using agent_information table for model storage)
            query = """UPDATE agent_information 
                      SET config = %s, info = %s 
                      WHERE registered_id = %s"""
            
            model_info = pickle.dumps({
                'model_type': 'autoencoder',
                'last_updated': time.time(),
                'training_metrics': model_metrics,
                'eval_metrics': eval_results
            })
            
            values = (model_bytes, model_info, self.agent_id)
            
            status = self.db_manager._DBManager__execute_and_commit(query, values)
            
            if status == 0:
                logging.info("AEMLTrainingThread: Model saved successfully to database")
            else:
                logging.error(f"AEMLTrainingThread: Failed to save model, status: {status}")
                
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error saving model: {e}")


class AutoencoderMLInferenceThread(MLInferenceThreadBase):
    """
    ML inference thread for autoencoder agent.
    Performs anomaly detection on streaming sensor data.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.window_size = config.get('window_size', 50)
        self.anomaly_threshold = None  # Will be set from loaded model
        self.model = None
        self.input_dim = None
        self.recent_data = []  # Buffer for creating inference windows
        
        super().__init__(agent_id, config)
    
    def load_model(self):
        """Load the latest autoencoder model from database."""
        try:
            # Query for latest model using agent_id
            query = f"SELECT config, info FROM agent_information WHERE registered_id = '{self.agent_id}'"
            results = self.db_manager._DBManager__execute_query(query)
            results = self.db_manager.parse_results(results)
            
            if len(results) == 0:
                logging.warning("AEMLInferenceThread: No saved model found in database")
                return
            
            # Load model data
            model_data = pickle.loads(results[0]['config'])
            
            # Rebuild model architecture
            self.input_dim = model_data['input_dim']
            encoder_dims = model_data['architecture_config']['encoder_dims']
            
            # Recreate model structure
            encoder_layers = []
            encoder_layers.append(layers.Dense(encoder_dims[0], activation='relu', input_shape=(self.input_dim,)))
            
            for dim in encoder_dims[1:]:
                encoder_layers.append(layers.Dense(dim, activation='relu'))
            
            decoder_layers = []
            for dim in reversed(encoder_dims[:-1]):
                decoder_layers.append(layers.Dense(dim, activation='relu'))
            
            decoder_layers.append(layers.Dense(self.input_dim, activation='linear'))
            
            # Create model
            self.model = keras.Sequential([*encoder_layers, *decoder_layers])
            self.model.compile(optimizer='adam', loss='mse')
            
            # Load trained weights
            self.model.set_weights(model_data['model_weights'])
            
            # Set anomaly threshold from evaluation metrics
            eval_metrics = model_data.get('eval_metrics', {})
            self.anomaly_threshold = eval_metrics.get('anomaly_threshold_95', 0.1)
            
            logging.info(f"AEMLInferenceThread: Loaded autoencoder model: input_dim={self.input_dim}, threshold={self.anomaly_threshold}")
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error loading model: {e}")
    
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
    
    def perform_inference(self, inference_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform anomaly detection inference.
        
        Args:
            inference_request: Parsed sensor data
            
        Returns:
            Inference results with reconstruction and anomaly score
        """
        try:
            if self.model is None:
                logging.warning("AEMLInferenceThread: No model loaded for inference")
                return None
            
            sensor_values = inference_request['sensor_values']
            
            # Add to recent data buffer
            self.recent_data.append(sensor_values)
            
            # Keep buffer at window size
            if len(self.recent_data) > self.window_size * 2:  # Keep extra for sliding windows
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
            
            # Create inference window (most recent window_size samples)
            window = np.array(self.recent_data[-self.window_size:])
            flattened_window = window.flatten().reshape(1, -1)
            
            # Normalize (simple min-max, ideally should use training statistics)
            window_min, window_max = flattened_window.min(), flattened_window.max()
            if window_max > window_min:
                normalized_window = (flattened_window - window_min) / (window_max - window_min)
            else:
                normalized_window = flattened_window
            
            # Get reconstruction
            reconstruction = self.model.predict(normalized_window, verbose=0)
            
            # Denormalize reconstruction
            if window_max > window_min:
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
                'timestamp': inference_request['timestamp'],
                'episode': inference_request.get('episode', 0),
                'episode_step': inference_request.get('episode_step', 0),
                'status': 'success'
            }
            
            if is_anomaly:
                logging.warning(f"AEMLInferenceThread: Anomaly detected: error_score={error_score:.4f} > threshold={self.anomaly_threshold:.4f}")
            
            return result
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error performing inference: {e}")
            return {'AEMLInferenceThread: error': str(e), 'status': 'error'}


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
                'window_size': 50,
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