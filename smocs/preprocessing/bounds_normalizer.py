import logging
import numpy as np
from typing import List, Dict, Any
from .base import BasePreprocessor
from .registry import PreprocessingRegistry

class BoundsNormalizer(BasePreprocessor):
    """Processor for normalizing data using channel-specific bounds."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bounds = np.array(self.config['bounds'], dtype=np.float32)
        self.channels = self.config.get('channels', [])
    
    def validate_config(self):
        """Validate configuration parameters."""
        if 'bounds' not in self.config:
            raise ValueError("BoundsNormalizer: 'bounds' must be specified in config")
        
        bounds = self.config['bounds']
        if not isinstance(bounds, (list, tuple)):
            raise ValueError("BoundsNormalizer: 'bounds' must be a list or tuple")
        
        # Validate bounds format
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                raise ValueError(f"BoundsNormalizer: bounds[{i}] must be a [min, max] pair")
            if bound[0] >= bound[1]:
                raise ValueError(f"BoundsNormalizer: bounds[{i}] min ({bound[0]}) must be less than max ({bound[1]})")
        
        # Validate bounds match channels if channels provided
        if 'channels' in self.config:
            channels = self.config['channels']
            if len(bounds) != len(channels):
                raise ValueError(f"BoundsNormalizer: Number of bounds ({len(bounds)}) must match number of channels ({len(channels)})")
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Normalize data using min-max scaling with specified bounds.
        
        Args:
            data: Input data to normalize - supports:
                  - 1D: single sample (n_features,)
                  - 2D: multiple samples (n_samples, n_features) or flattened windows
                  - 3D: sequences (n_sequences, sequence_length, n_features) - from database
            **kwargs: Additional parameters (unused currently)
            
        Returns:
            Normalized data scaled to [0, 1] range
        """
        try:
            # Ensure data is numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            else:
                data = data.astype(np.float32)
            
            # Handle different data shapes
            original_shape = data.shape
            
            if data.ndim == 1:
                # Single sample
                normalized_data = self._normalize_sample(data)
                
            elif data.ndim == 2:
                # Multiple samples or windowed data
                if data.shape[1] == len(self.bounds):
                    # Each column corresponds to a channel
                    normalized_data = np.array([self._normalize_sample(sample) for sample in data])
                else:
                    # Windowed data - need to handle differently
                    normalized_data = self._normalize_windowed_data(data)
                    
            elif data.ndim == 3:
                # 3D data from database: (n_sequences, sequence_length, n_features)
                normalized_data = self._normalize_3d_sequences(data)
                
            else:
                raise ValueError(f"BoundsNormalizer: Unsupported data dimensionality: {data.ndim}")
            
            # Validate output
            if np.any(np.isnan(normalized_data)) or np.any(np.isinf(normalized_data)):
                logging.error("BoundsNormalizer: NaN or Inf values in normalized data")
                raise ValueError("BoundsNormalizer: Invalid values in normalized output")
            
            logging.debug(f"BoundsNormalizer: Normalized data shape {original_shape} -> {normalized_data.shape}")
            
            return normalized_data
            
        except Exception as e:
            logging.error(f"BoundsNormalizer: Error normalizing data: {e}")
            raise
    
    def _normalize_3d_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Normalize 3D sequence data from database.
        
        Args:
            sequences: Array of shape (n_sequences, sequence_length, n_features)
            
        Returns:
            Normalized sequences with same shape
        """
        n_sequences, sequence_length, n_features = sequences.shape
        
        logging.debug(f"BoundsNormalizer: Processing 3D sequences: {n_sequences} sequences, {sequence_length} timesteps, {n_features} features")
        
        # Validate feature count matches bounds
        if n_features != len(self.bounds):
            raise ValueError(f"BoundsNormalizer: Number of features ({n_features}) doesn't match number of bounds ({len(self.bounds)})")
        
        # Normalize each sequence
        normalized_sequences = np.zeros_like(sequences)
        
        for seq_idx in range(n_sequences):
            sequence = sequences[seq_idx]  # Shape: (sequence_length, n_features)
            
            # Normalize each timestep in the sequence
            for t in range(sequence_length):
                timestep = sequence[t]  # Shape: (n_features,)
                normalized_sequences[seq_idx, t] = self._normalize_sample(timestep)
        
        logging.debug(f"BoundsNormalizer: Normalized {n_sequences} sequences")
        
        return normalized_sequences
    
    def _normalize_sample(self, sample: np.ndarray) -> np.ndarray:
        """Normalize a single sample."""
        if len(sample) != len(self.bounds):
            raise ValueError(f"BoundsNormalizer: Sample length ({len(sample)}) doesn't match bounds length ({len(self.bounds)})")
        
        normalized = np.zeros_like(sample)
        
        for i, (value, (min_bound, max_bound)) in enumerate(zip(sample, self.bounds)):
            # Clip to bounds and log if clipping occurs
            if value < min_bound:
                logging.warning(f"BoundsNormalizer: Clipping value {value} to lower bound {min_bound} for channel {i}")
                value = min_bound
            elif value > max_bound:
                logging.warning(f"BoundsNormalizer: Clipping value {value} to upper bound {max_bound} for channel {i}")
                value = max_bound
            
            # Min-max normalization to [0, 1]
            range_size = max_bound - min_bound
            if range_size > 0:
                normalized[i] = (value - min_bound) / range_size
            else:
                normalized[i] = 0.5
        
        return normalized
    
    def _normalize_windowed_data(self, windowed_data: np.ndarray) -> np.ndarray:
        """Normalize windowed data where each row is a flattened window."""
        n_samples, window_length = windowed_data.shape
        n_channels = len(self.bounds)
        
        if window_length % n_channels != 0:
            raise ValueError(f"BoundsNormalizer: Window length ({window_length}) not divisible by number of channels ({n_channels})")
        
        window_size = window_length // n_channels
        normalized_data = np.zeros_like(windowed_data)
        
        logging.debug(f"BoundsNormalizer: Processing windowed data: {n_samples} samples, window_length={window_length}, n_channels={n_channels}, window_size={window_size}")
        
        for sample_idx in range(n_samples):
            window = windowed_data[sample_idx]
            
            try:
                # Reshape to (window_size, n_channels)
                reshaped_window = window.reshape(window_size, n_channels)
                
                # Normalize each timestep
                normalized_window = np.array([self._normalize_sample(timestep) for timestep in reshaped_window])
                
                # Flatten back
                normalized_data[sample_idx] = normalized_window.flatten()
                
            except Exception as reshape_error:
                logging.error(f"BoundsNormalizer: Error processing sample {sample_idx}: {reshape_error}")
                logging.error(f"BoundsNormalizer: Window shape: {window.shape}, trying to reshape to ({window_size}, {n_channels})")
                raise
        
        return normalized_data
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original bounds.
        
        Args:
            normalized_data: Normalized data in [0, 1] range
            
        Returns:
            Denormalized data in original bounds
        """
        try:
            if not isinstance(normalized_data, np.ndarray):
                normalized_data = np.array(normalized_data, dtype=np.float32)
            
            if normalized_data.ndim == 1:
                return self._denormalize_sample(normalized_data)
            elif normalized_data.ndim == 2:
                if normalized_data.shape[1] == len(self.bounds):
                    return np.array([self._denormalize_sample(sample) for sample in normalized_data])
                else:
                    return self._denormalize_windowed_data(normalized_data)
            elif normalized_data.ndim == 3:
                return self._denormalize_3d_sequences(normalized_data)
            else:
                raise ValueError(f"BoundsNormalizer: Unsupported data dimensionality for denormalization: {normalized_data.ndim}")
                
        except Exception as e:
            logging.error(f"BoundsNormalizer: Error denormalizing data: {e}")
            raise
    
    def _denormalize_3d_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Denormalize 3D sequence data."""
        n_sequences, sequence_length, n_features = sequences.shape
        denormalized_sequences = np.zeros_like(sequences)
        
        for seq_idx in range(n_sequences):
            for t in range(sequence_length):
                denormalized_sequences[seq_idx, t] = self._denormalize_sample(sequences[seq_idx, t])
        
        return denormalized_sequences
    
    def _denormalize_sample(self, normalized_sample: np.ndarray) -> np.ndarray:
        """Denormalize a single sample."""
        if len(normalized_sample) != len(self.bounds):
            raise ValueError(f"BoundsNormalizer: Sample length ({len(normalized_sample)}) doesn't match bounds length ({len(self.bounds)})")
        
        denormalized = np.zeros_like(normalized_sample)
        
        for i, (norm_value, (min_bound, max_bound)) in enumerate(zip(normalized_sample, self.bounds)):
            range_size = max_bound - min_bound
            denormalized[i] = norm_value * range_size + min_bound
        
        return denormalized
    
    def _denormalize_windowed_data(self, windowed_data: np.ndarray) -> np.ndarray:
        """Denormalize windowed data."""
        n_samples, window_length = windowed_data.shape
        n_channels = len(self.bounds)
        window_size = window_length // n_channels
        
        denormalized_data = np.zeros_like(windowed_data)
        
        for sample_idx in range(n_samples):
            window = windowed_data[sample_idx]
            reshaped_window = window.reshape(window_size, n_channels)
            denormalized_window = np.array([self._denormalize_sample(timestep) for timestep in reshaped_window])
            denormalized_data[sample_idx] = denormalized_window.flatten()
        
        return denormalized_data
    
    @classmethod
    def get_name(cls) -> str:
        return "bounds_normalizer"

# Register the processor
PreprocessingRegistry().register(BoundsNormalizer)