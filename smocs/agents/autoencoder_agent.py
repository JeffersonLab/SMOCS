import os
import glob
import json
import time
import logging
import traceback
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from smocs.cores import AgentBase, DataIngestThreadBase, MLTrainingThreadBase, MLInferenceThreadBase
from smocs.utils import ConfigLoader, ChannelFilter
from smocs.preprocessing import PreprocessingManager

class AutoencoderDataIngestThread(DataIngestThreadBase):
    """
    Data ingestion thread for autoencoder agent.
    Stores raw time series sensor data to database.
    """
    
    def store_message(self, message_data, topic, partition, offset) -> bool:
        """
        Parse sensor message and store raw data to database.
        No preprocessing here - just store raw sensor values.
        """
        try:
            # Extract timestamp
            if 'timestamp' in message_data:
                timestamp = datetime.fromtimestamp(message_data['timestamp'])
            else:
                timestamp = datetime.now()
            
            # Get filtered channels (already processed by base class)
            channels = message_data.get('channels', {})
            
            if not channels:
                logging.error("AEDataIngestThread: No channels in filtered message data")
                return False
            
            # Convert channel values directly to numpy array (maintains order)
            channel_values = list(channels.values())
            sensor_values = np.array(channel_values, dtype=np.float32)
            
            # Store RAW data in database (no preprocessing here)
            sensor_data = {
                'state_source_timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'state_received_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'state': sensor_values  # Raw data - preprocessing happens during training
            }
            
            status = self.db_manager.record_sensor_data(sensor_data)
            
            if status == 0:
                logging.debug(f"AEDataIngestThread: Stored raw sensor data: {len(sensor_values)} channels at {timestamp}")
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

        # Preprocessing setup
        self.preprocessing_manager = PreprocessingManager(config)
        self.window_processor = self.preprocessing_manager.get_processor('window_processor')
        self.bounds_normalizer = self.preprocessing_manager.get_processor('bounds_normalizer')
        
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
        Retrieve raw training data from database and apply preprocessing pipeline.
        """
        try:
            # Check if enough data available
            total_samples = self.db_manager.get_size("agent_inferences")
            
            logging.info(f"AEMLTrainingThread: Database contains {total_samples} samples (need {self.min_training_samples})")
            
            if total_samples < self.min_training_samples:
                logging.debug(f"AEMLTrainingThread: Not enough samples for training: {total_samples} < {self.min_training_samples}")
                return None
            
            # Check if we have new data since last training
            if total_samples <= self.last_training_count:
                logging.debug("AEMLTrainingThread: No new data since last training")
                return None
            
            logging.info(f"AEMLTrainingThread: Found {total_samples - self.last_training_count} new samples since last training")
            
            # Get recent sensor data from database - data is RAW (not preprocessed)
            batch_data = self.db_manager.sample_batch(
                batch_size=self.batch_size*100, 
                segment_length=self.window_size,
                agent_type="diagnostics",
                mode="latest"
            )
            
            logging.info(f"AEMLTrainingThread: sample_batch returned {len(batch_data['state']) if batch_data else 0} sequences")
            
            if batch_data is None or len(batch_data['state']) == 0:
                logging.warning("AEMLTrainingThread: No batch data returned from database")
                return None
            
            # Step 1: Normalize each individual state using bounds normalizer
            logging.info("AEMLTrainingThread: Starting preprocessing pipeline...")
            normalized_sequences = []
            
            for seq_idx, sequence in enumerate(batch_data['state']):
                if len(sequence) != self.window_size:
                    logging.debug(f"AEMLTrainingThread: Skipping sequence {seq_idx} with wrong length {len(sequence)}")
                    continue
                
                normalized_sequence = []
                sequence_valid = True
                
                for state_idx, raw_state in enumerate(sequence):
                    try:
                        # Convert to consistent numpy format
                        if isinstance(raw_state, np.ndarray):
                            state_array = raw_state.flatten().astype(np.float32)
                        elif isinstance(raw_state, (list, tuple)):
                            state_array = np.array(raw_state, dtype=np.float32).flatten()
                        elif isinstance(raw_state, (int, float)):
                            state_array = np.array([float(raw_state)], dtype=np.float32)
                        else:
                            logging.warning(f"AEMLTrainingThread: Unknown state type at sequence {seq_idx}, state {state_idx}: {type(raw_state)}")
                            sequence_valid = False
                            break
                        
                        # Validate numeric values
                        if np.any(np.isnan(state_array)) or np.any(np.isinf(state_array)):
                            logging.warning(f"AEMLTrainingThread: Invalid values (NaN/Inf) at sequence {seq_idx}, state {state_idx}")
                            sequence_valid = False
                            break
                        
                        # Apply bounds normalization to this state
                        normalized_state = self.bounds_normalizer.process(state_array)
                        normalized_sequence.append(normalized_state)
                        
                    except Exception as norm_error:
                        logging.warning(f"AEMLTrainingThread: Normalization failed for sequence {seq_idx}, state {state_idx}: {norm_error}")
                        sequence_valid = False
                        break
                
                # Add valid normalized sequences
                if sequence_valid and len(normalized_sequence) == self.window_size:
                    normalized_sequences.append(normalized_sequence)
            
            logging.info(f"AEMLTrainingThread: Normalized {len(normalized_sequences)} sequences")
            
            if not normalized_sequences:
                logging.error("AEMLTrainingThread: No valid sequences after normalization")
                return None
            
            # Step 2: Apply window processing to create training windows
            try:
                windowed_array = self.window_processor.process(normalized_sequences)
                
                if windowed_array is None or len(windowed_array) == 0:
                    logging.error("AEMLTrainingThread: No valid windows created by window processor")
                    return None
                    
            except Exception as window_error:
                logging.error(f"AEMLTrainingThread: Window processing failed: {window_error}")
                # Debug the data format
                logging.error(f"AEMLTrainingThread: Data passed to window processor: {len(normalized_sequences)} sequences")
                if len(normalized_sequences) > 0:
                    logging.error(f"AEMLTrainingThread: First sequence length: {len(normalized_sequences[0])}")
                    logging.error(f"AEMLTrainingThread: First state shape: {normalized_sequences[0][0].shape}")
                return None
            
            # Log final statistics
            logging.info(f"AEMLTrainingThread: Final windowed data shape: {windowed_array.shape}")
            logging.debug(f"AEMLTrainingThread: Data range: [{np.min(windowed_array):.6f}, {np.max(windowed_array):.6f}]")
            logging.debug(f"AEMLTrainingThread: Data mean: {np.mean(windowed_array):.6f}")
            logging.debug(f"AEMLTrainingThread: Data std: {np.std(windowed_array):.6f}")
            
            # Update training count
            self.last_training_count = total_samples
            
            logging.info(f"AEMLTrainingThread: Successfully prepared {len(windowed_array)} training windows using preprocessing pipeline")
            return windowed_array
            
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error getting training data: {e}")
            logging.error(f"AEMLTrainingThread: Exception details: {type(e).__name__}: {str(e)}")
            return None
    
    def train_model(self, training_data: np.ndarray) -> Dict[str, Any]:
        """
        Train the autoencoder model on preprocessed data.
        Data is already normalized by the preprocessing pipeline.
        """
        try:
            # Build model if not exists
            if self.model is None:
                input_dim = training_data.shape[1]
                self._create_autoencoder(input_dim)
            
            logging.info(f"AEMLTrainingThread: Training with preprocessed data shape: {training_data.shape}")
            
            # Train autoencoder (input = output for reconstruction)
            # Data is already normalized by preprocessing pipeline
            history = self.model.fit(
                training_data,
                training_data,  # Autoencoder learns to reconstruct input
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                verbose=1  # Show training progress
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
            logging.error(f"AEMLTrainingThread: Exception details: {type(e).__name__}")
            return {'error': str(e)}
    
    def eval_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model and denormalize results for interpretability.
        """
        try:
            if self.model is None:
                return {'error': 'No model to evaluate'}
            
            # Get small batch of recent normalized data for evaluation
            eval_data = self.get_training_data()
            if eval_data is None:
                return {'error': 'No evaluation data available'}
            
            # Use subset for evaluation
            eval_subset = eval_data[:min(100, len(eval_data))]
            
            # Evaluate reconstruction error on normalized data
            reconstructions = self.model.predict(eval_subset, verbose=0)
            mse_errors = np.mean((eval_subset - reconstructions) ** 2, axis=1)
            
            # Denormalize a sample for interpretability
            try:
                sample_original = self.bounds_normalizer.denormalize(eval_subset[:5])
                sample_reconstructed = self.bounds_normalizer.denormalize(reconstructions[:5])
                
                # Compute denormalized errors for reporting
                denorm_errors = np.mean((sample_original - sample_reconstructed) ** 2, axis=1)
                
                eval_metrics = {
                    'mean_reconstruction_error': float(np.mean(mse_errors)),
                    'std_reconstruction_error': float(np.std(mse_errors)),
                    'max_reconstruction_error': float(np.max(mse_errors)),
                    'anomaly_threshold_95': float(np.percentile(mse_errors, 95)),
                    'eval_samples': len(eval_subset),
                    'mean_denormalized_error': float(np.mean(denorm_errors)),
                    'sample_original_range': [float(np.min(sample_original)), float(np.max(sample_original))],
                    'sample_reconstructed_range': [float(np.min(sample_reconstructed)), float(np.max(sample_reconstructed))]
                }
                
            except Exception as denorm_error:
                logging.warning(f"AEMLTrainingThread: Could not denormalize evaluation samples: {denorm_error}")
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
        """
        try:
            if self.model is None:
                logging.error("AEMLTrainingThread: No model to save")
                return
            
            logging.info("AEMLTrainingThread: Starting model save process...")
            
            # Create models directory
            models_dir = "/app/models"
            os.makedirs(models_dir, exist_ok=True)
            logging.info(f"AEMLTrainingThread: Models directory created/confirmed: {models_dir}")
            
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
            
            logging.info(f"AEMLTrainingThread: Saving model version {next_version}")
            
            # Use temporary files for atomic writes
            model_tmp = f"{model_file}.tmp"
            latest_tmp = f"{latest_file}.tmp"
            
            # Save model to temporary file
            logging.info("AEMLTrainingThread: Saving TensorFlow model...")
            self.model.save(model_tmp)
            logging.info("AEMLTrainingThread: TensorFlow model saved to temporary file")
            
            # Prepare metadata (normalization info stored in bounds normalizer config)
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
                'normalization_note': 'Data normalized at ingestion using bounds from config'
            }
            
            # Save metadata to temporary file
            logging.info("AEMLTrainingThread: Saving metadata...")
            with open(latest_tmp, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info("AEMLTrainingThread: Metadata saved to temporary file")
            
            # Atomic renames
            logging.info("AEMLTrainingThread: Performing atomic file renames...")
            os.rename(model_tmp, model_file)
            os.rename(latest_tmp, latest_file)
            logging.info("AEMLTrainingThread: Atomic renames completed")
            
            logging.info(f"AEMLTrainingThread: Model v{version_str} saved successfully as latest (val_loss: {eval_results.get('val_loss', 'N/A')})")
                
        except Exception as e:
            logging.error(f"AEMLTrainingThread: Error saving model: {e}")
            logging.error(f"AEMLTrainingThread: Exception details: {type(e).__name__}: {str(e)}")
            
            # Clean up temporary files - FIXED BUG
            temp_files = [model_tmp, latest_tmp] if 'model_tmp' in locals() and 'latest_tmp' in locals() else []
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                        logging.info(f"AEMLTrainingThread: Cleaned up temporary file: {tmp_file}")
                    except Exception as cleanup_error:
                        logging.error(f"AEMLTrainingThread: Error cleaning up {tmp_file}: {cleanup_error}")
    
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

        # Preprocessing setup
        self.preprocessing_manager = PreprocessingManager(config)
        self.window_processor = self.preprocessing_manager.get_processor('window_processor')
        self.bounds_normalizer = self.preprocessing_manager.get_processor('bounds_normalizer')
        
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
        Perform anomaly detection inference using preprocessing pipeline.
        """
        try:
            # Check for model updates periodically
            self.check_for_model_updates()
            
            if self.model is None:
                self.load_model()
                if self.model is None:
                    logging.debug("AEMLInferenceThread: No model available for inference")
                    return {
                        'reconstruction_normalized': np.array([]),
                        'reconstruction_original': np.array([]),
                        'error_score': 0.0,
                        'is_anomaly': False,
                        'anomaly_threshold': 0.0,
                        'model_version': 0,
                        'timestamp': inference_request['timestamp'],
                        'status': 'model_not_ready'
                    }
            
            sensor_values = inference_request['sensor_values']
            
            # Normalize sensor values using bounds normalizer
            try:
                normalized_values = self.bounds_normalizer.process(sensor_values)
            except Exception as norm_error:
                logging.error(f"AEMLInferenceThread: Normalization failed: {norm_error}")
                return {'error': str(norm_error), 'status': 'error'}
            
            # Add normalized values to recent data buffer
            self.recent_data.append(normalized_values)
            
            # Keep buffer at reasonable size
            if len(self.recent_data) > self.window_size * 2:
                self.recent_data = self.recent_data[-self.window_size * 2:]
            
            # Need at least window_size samples for inference
            if len(self.recent_data) < self.window_size:
                return {
                    'reconstruction_normalized': normalized_values,
                    'reconstruction_original': sensor_values,
                    'error_score': 0.0,
                    'is_anomaly': False,
                    'status': 'insufficient_data',
                    'buffer_size': len(self.recent_data),
                    'timestamp': inference_request['timestamp']
                }
            
            # Create inference window from normalized data using window processor
            try:
                window_data = self.window_processor.process([self.recent_data[-self.window_size:]])
                if len(window_data) == 0:
                    raise ValueError("No valid windows created")
                
                flattened_window = window_data[0:1]  # Get first (and only) window as batch
                
            except Exception as window_error:
                logging.error(f"AEMLInferenceThread: Window creation failed: {window_error}")
                return {'error': str(window_error), 'status': 'error'}
            
            # Get reconstruction from model (operates on preprocessed data)
            reconstruction_normalized = self.model.predict(flattened_window, verbose=0)
            
            # Denormalize reconstruction to original units
            try:
                reconstruction_original = self.bounds_normalizer.denormalize(reconstruction_normalized)
            except Exception as denorm_error:
                logging.warning(f"AEMLInferenceThread: Denormalization failed: {denorm_error}")
                reconstruction_original = reconstruction_normalized  # Fallback
            
            # Compute reconstruction error on normalized data
            error_score = float(np.mean((flattened_window - reconstruction_normalized) ** 2))
            
            # Determine if anomaly
            is_anomaly = error_score > self.anomaly_threshold if self.anomaly_threshold else False
            
            result = {
                'reconstruction_normalized': reconstruction_normalized.flatten(),
                'reconstruction_original': reconstruction_original.flatten(),
                'original_window_normalized': flattened_window.flatten(),
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

    def parse_inference_request(self, message_data, topic, partition, offset) -> Optional[Dict[str, Any]]:
        """
        Parse inference request from sensor message.
        
        Args:
            message_data: Already parsed and filtered message dict
            topic: Kafka topic
            partition: Kafka partition  
            offset: Kafka offset
            
        Returns:
            Parsed sensor data or None
        """
        try:
            channels = message_data.get('channels', {})
            
            if not channels:
                logging.error("AEMLInferenceThread: No channels in filtered message data")
                return None
            
            # Convert channel values directly to numpy array (maintains order)
            channel_values = list(channels.values())
            sensor_values = np.array(channel_values, dtype=np.float32)
            
            return {
                'sensor_values': sensor_values,
                'timestamp': message_data.get('timestamp', time.time()),
                'channels': channels
            }
            
        except Exception as e:
            logging.error(f"AEMLInferenceThread: Error parsing inference request: {e}")
            return None

    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Process incoming message with optional channel filtering and return inference results.
        
        Args:
            message: The message value
            topic: The topic name
            partition: The partition number
            offset: The message offset
            
        Returns:
            Tuple[bool, List[Tuple]]: Success status and list of outputs to send
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
                    logging.debug(f"MLInfernceThread: Skipping message from {topic}:{partition}:{offset} due to channel filtering")
                    return True, []
                
                channel_names, channel_values = filtered_result
            else:
                # Extract all numeric channels when no filter configured
                filtered_result = ChannelFilter.extract_all_channels(message_data)
                if filtered_result is None:
                    logging.debug(f"MLInfernceThread: Skipping message from {topic}:{partition}:{offset} - no valid channels")
                    return True, []
                
                channel_names, channel_values = filtered_result
            
            # Create clean channel dictionary for agent processing
            filtered_channels = dict(zip(channel_names, channel_values))
            message_data['channels'] = filtered_channels
            
            logging.debug(f"MLInfernceThread: Extracted {len(channel_values)} channels for inference")
            
            # Parse inference request with processed data
            inference_request = self.parse_inference_request(message_data, topic, partition, offset)
            
            if inference_request is None:
                return False, []
            
            # Rest of inference logic remains the same...
            inference_result = self.perform_inference(inference_request)
            
            if inference_result is None:
                return False, []
            
            self._store_inference_result(inference_request, inference_result)
            
            output_message = {
                'timestamp': time.time(),
                'channels': {
                    'agent_id': self.agent_id,
                    'error_score': inference_result.get('error_score', 0.0),
                    'is_anomaly': inference_result.get('is_anomaly', False),
                    'anomaly_threshold': inference_result.get('anomaly_threshold', 0.0),
                    'model_version': inference_result.get('model_version', 0),
                    'status': inference_result.get('status', 'unknown')
                }
            }
            
            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
            
        except json.JSONDecodeError as e:
            logging.error(f"MLInfernceThread: JSON decode error for message from {topic}:{partition}:{offset}: {e}")
            return False, []
        except Exception as e:
            logging.error(f"MLInfernceThread: Error processing inference message: {e}")
            return False, []

    def _store_inference_result(self, inference_request: Dict[str, Any], inference_result: Dict[str, Any]):
        """Store inference result to database using DBManager's record_prediction function."""
        try:
            if 'error' in inference_result or inference_result.get('status') == 'error':
                logging.warning(f"AEMLInferenceThread: Not storing inference result due to error: {inference_result.get('error', 'unknown error')}")
                return

            # Extract timestamp from inference request
            source_timestamp = inference_request.get('timestamp')
            if source_timestamp is None:
                logging.warning("AEMLInferenceThread: No timestamp in inference request, cannot store result")
                return
            
            # Convert timestamp to proper format if needed
            if isinstance(source_timestamp, (int, float)):
                timestamp_dt = datetime.fromtimestamp(source_timestamp)
                source_timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                source_timestamp_str = str(source_timestamp)
            
            # Create prediction array from inference result
            # Use reconstruction_normalized for storage (consistent with training data)
            reconstruction = inference_result.get('reconstruction_normalized')
            if reconstruction is None:
                logging.warning("AEMLInferenceThread: No reconstruction_normalized in inference result, cannot store prediction")
                return
            
            # Convert reconstruction to numpy array
            if isinstance(reconstruction, list):
                prediction_array = np.array(reconstruction, dtype=np.float32)
            elif isinstance(reconstruction, np.ndarray):
                prediction_array = reconstruction.astype(np.float32)
            else:
                logging.warning(f"AEMLInferenceThread: Unexpected reconstruction type: {type(reconstruction)}")
                return
            
            # Get current timestamp for prediction timestamp
            prediction_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Use record_prediction to store the inference result
            status = self.db_manager.record_prediction(
                prediction=prediction_array,
                prediction_timestamp=prediction_timestamp,
                key_value=source_timestamp_str,
                key="state_source_timestamp"
            )
            
            if status == 0:
                logging.debug(f"AEMLInferenceThread: Successfully stored inference result for timestamp {source_timestamp_str}")
            else:
                logging.error(f"AEMLInferenceThread: Failed to store inference result, status: {status}")
            
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
            # Get enabled threads directly from config
            self.enabled_threads = autoencoder_config.get('enabled_threads', ['ingest', 'training', 'inference'])
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
            # Default to all threads when no config file
            self.enabled_threads = ['ingest', 'training', 'inference']
        
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
        logging.info(f"AEAgent: Enabled threads: {self.enabled_threads}")
    
    def create_data_ingest_component(self):
        """Create data ingestion thread component."""
        if 'ingest' in self.enabled_threads:
            return AutoencoderDataIngestThread(self.agent_id, self.agent_config)
        return None
    
    def create_ml_training_component(self):
        """Create ML training thread component."""
        if 'training' in self.enabled_threads:
            return AutoencoderMLTrainingThread(self.agent_id, self.agent_config)
        return None
    
    def create_ml_inference_component(self):
        """Create ML inference thread component."""
        if 'inference' in self.enabled_threads:
            return AutoencoderMLInferenceThread(self.agent_id, self.agent_config)
        return None

def main():
    """Main entry point for autoencoder agent."""

    
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