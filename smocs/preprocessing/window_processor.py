import logging
import numpy as np
from typing import List, Optional, Dict, Any
from .base import BasePreprocessor
from .registry import PreprocessingRegistry

class WindowProcessor(BasePreprocessor):
    """Processor for creating sliding windows from sensor data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = self.config['window_size']
    
    def validate_config(self):
        """Validate configuration parameters."""
        if 'window_size' not in self.config:
            raise ValueError("WindowProcessor: 'window_size' must be specified in config")
        
        if not isinstance(self.config['window_size'], int) or self.config['window_size'] <= 0:
            raise ValueError("WindowProcessor: 'window_size' must be a positive integer")
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Create windows from input data.
        
        Args:
            data: Input data - can be:
                - List of sequences (from database batch_data['state'])
                - 2D array (n_timesteps, n_features) 
                - 3D array (n_sequences, sequence_length, n_features)
            **kwargs: Additional parameters (unused currently)
            
        Returns:
            Windowed data of shape (n_windows, window_size * n_features)
        """
        try:
            # Handle different input formats
            if isinstance(data, list):
                # Convert list of states to array format (original logic)
                processed_sequences = self._process_sequence_list(data)
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    # Handle 2D numpy array input (n_timesteps, n_features)
                    processed_sequences = self._process_array_data(data)
                elif data.ndim == 3:
                    # Handle 3D numpy array input (n_sequences, sequence_length, n_features)
                    processed_sequences = self._process_3d_array_data(data)
                else:
                    raise ValueError(f"WindowProcessor: Unsupported array dimensionality: {data.ndim}D")
            else:
                raise ValueError(f"WindowProcessor: Unsupported data type: {type(data)}")
            
            if len(processed_sequences) == 0:
                raise ValueError("WindowProcessor: No valid sequences found after processing")
            
            # Create final windowed array
            windowed_array = np.array(processed_sequences, dtype=np.float32)
            
            logging.debug(f"WindowProcessor: Created {len(windowed_array)} windows of size {windowed_array.shape[1]}")
            
            return windowed_array
            
        except Exception as e:
            logging.error(f"WindowProcessor: Error processing data: {e}")
            raise

    def _process_3d_array_data(self, data: np.ndarray) -> List[np.ndarray]:
        """Process 3D array data (n_sequences, sequence_length, n_features) into windows."""
        n_sequences, sequence_length, n_features = data.shape
        
        logging.debug(f"WindowProcessor: Processing 3D data: {n_sequences} sequences, {sequence_length} timesteps, {n_features} features")
        
        valid_windows = []
        
        for seq_idx in range(n_sequences):
            sequence = data[seq_idx]  # Shape: (sequence_length, n_features)
            
            # Check if sequence length matches expected window size
            if sequence_length != self.window_size:
                logging.debug(f"WindowProcessor: Skipping sequence {seq_idx} with length {sequence_length}, expected {self.window_size}")
                continue
            
            # Validate the sequence data
            if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                logging.warning(f"WindowProcessor: Invalid values (NaN/Inf) in sequence {seq_idx}")
                continue
            
            # Flatten the sequence into a window
            try:
                flattened_window = sequence.flatten().astype(np.float32)
                valid_windows.append(flattened_window)
            except Exception as flatten_error:
                logging.warning(f"WindowProcessor: Error flattening sequence {seq_idx}: {flatten_error}")
                continue
        
        logging.info(f"WindowProcessor: Processed {len(valid_windows)} valid windows from {n_sequences} 3D sequences")
        return valid_windows
    
    def _process_sequence_list(self, sequences: List) -> List[np.ndarray]:
        """Process a list of sequences into windows."""
        valid_windows = []
        expected_state_length = None
        
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) != self.window_size:
                logging.debug(f"WindowProcessor: Skipping sequence {seq_idx} with length {len(sequence)}, expected {self.window_size}")
                continue
            
            # Process states in sequence
            processed_states = []
            sequence_valid = True
            
            for state_idx, state in enumerate(sequence):
                try:
                    # Convert to numpy array and flatten
                    if isinstance(state, np.ndarray):
                        if state.size == 0:
                            logging.warning(f"WindowProcessor: Empty state array at sequence {seq_idx}, state {state_idx}")
                            sequence_valid = False
                            break
                        state_flat = state.flatten().astype(np.float32)
                    elif isinstance(state, (list, tuple)):
                        if len(state) == 0:
                            logging.warning(f"WindowProcessor: Empty state list at sequence {seq_idx}, state {state_idx}")
                            sequence_valid = False
                            break
                        state_array = np.array(state, dtype=np.float32)
                        state_flat = state_array.flatten()
                    elif isinstance(state, (int, float)):
                        state_flat = np.array([float(state)], dtype=np.float32)
                    else:
                        logging.warning(f"WindowProcessor: Unknown state type at sequence {seq_idx}, state {state_idx}: {type(state)}")
                        sequence_valid = False
                        break
                    
                    # Validate values
                    if np.any(np.isnan(state_flat)) or np.any(np.isinf(state_flat)):
                        logging.warning(f"WindowProcessor: Invalid values (NaN/Inf) at sequence {seq_idx}, state {state_idx}")
                        sequence_valid = False
                        break
                    
                    # Check length consistency
                    if expected_state_length is None:
                        expected_state_length = len(state_flat)
                    elif len(state_flat) != expected_state_length:
                        logging.warning(f"WindowProcessor: Inconsistent state length at sequence {seq_idx}, state {state_idx}: expected {expected_state_length}, got {len(state_flat)}")
                        sequence_valid = False
                        break
                    
                    processed_states.append(state_flat)
                    
                except Exception as e:
                    logging.warning(f"WindowProcessor: Error processing sequence {seq_idx}, state {state_idx}: {e}")
                    sequence_valid = False
                    break
            
            # Add valid windows
            if sequence_valid and len(processed_states) == self.window_size:
                window = np.concatenate(processed_states)
                valid_windows.append(window)
        
        return valid_windows
    
    def _process_array_data(self, data: np.ndarray) -> List[np.ndarray]:
        """Process array data into windows."""
        if data.ndim != 2:
            raise ValueError(f"WindowProcessor: Expected 2D array, got {data.ndim}D")
        
        n_timesteps, n_features = data.shape
        
        if n_timesteps < self.window_size:
            logging.warning(f"WindowProcessor: Not enough timesteps ({n_timesteps}) for window size ({self.window_size})")
            return []
        
        windows = []
        for i in range(n_timesteps - self.window_size + 1):
            window_data = data[i:i + self.window_size]
            flattened_window = window_data.flatten()
            windows.append(flattened_window)
        
        return windows
    
    @classmethod
    def get_name(cls) -> str:
        return "window_processor"

# Register the processor
PreprocessingRegistry().register(WindowProcessor)