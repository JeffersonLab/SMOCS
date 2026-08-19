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
        Flatten windows of input data for model consumption.

        Args:
            data: Input data - can be:
                - List of windows (from database batch_data['state']), each window
                  itself a list of per-timestep states
                - 2D array of raw timesteps (n_timesteps, n_features), converted into
                  overlapping windows via a sliding window of length self.window_size
                - 3D array of windows (n_windows, window_size, n_features)
            **kwargs: Additional parameters (unused currently)

        Returns:
            Flattened windows, shape (n_windows, window_size * n_features)
        """
        try:
            # Add debug logging
            logging.debug(f"WindowProcessor: Input data type: {type(data)}")
            if isinstance(data, (list, tuple)):
                logging.debug(f"WindowProcessor: List/tuple length: {len(data)}")
                if len(data) > 0:
                    logging.debug(f"WindowProcessor: First element type: {type(data[0])}")
                    if isinstance(data[0], (list, tuple)):
                        logging.debug(f"WindowProcessor: First window length: {len(data[0])}")
                        if len(data[0]) > 0:
                            logging.debug(f"WindowProcessor: First state type: {type(data[0][0])}")
                            logging.debug(f"WindowProcessor: First state: {data[0][0]}")
            elif isinstance(data, np.ndarray):
                logging.debug(f"WindowProcessor: Array shape: {data.shape}")
                logging.debug(f"WindowProcessor: Array dtype: {data.dtype}")

            # Handle different input formats
            if isinstance(data, list):
                # Convert list of windows to array format (original logic)
                flattened_windows = self._process_window_list(data)
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    # Handle 2D numpy array input (n_timesteps, n_features)
                    flattened_windows = self._process_array_data(data)
                elif data.ndim == 3:
                    # Handle 3D numpy array input (n_windows, window_size, n_features)
                    flattened_windows = self._process_3d_array_data(data)
                else:
                    raise ValueError(f"WindowProcessor: Unsupported array dimensionality: {data.ndim}D")
            else:
                raise ValueError(f"WindowProcessor: Unsupported data type: {type(data)}")

            if len(flattened_windows) == 0:
                logging.error("WindowProcessor: No valid windows found after processing")
                raise ValueError("WindowProcessor: No valid windows found after processing")

            # Create final flattened-windows array
            flattened_windows = np.array(flattened_windows, dtype=np.float32)

            logging.debug(f"WindowProcessor: Created {len(flattened_windows)} windows of size {flattened_windows.shape[1]}")

            return flattened_windows

        except Exception as e:
            logging.error(f"WindowProcessor: Error processing data: {e}")
            raise

    def _process_3d_array_data(self, data: np.ndarray) -> List[np.ndarray]:
        """Process 3D array data (n_windows, window_size, n_features) into flattened windows."""
        n_windows, window_size, n_features = data.shape

        logging.debug(f"WindowProcessor: Processing 3D data: {n_windows} windows, {window_size} timesteps, {n_features} features")

        flattened_windows = []

        for window_idx in range(n_windows):
            window = data[window_idx]  # Shape: (window_size, n_features)

            # Check if this window's length matches the expected window_size
            if window_size != self.window_size:
                logging.debug(f"WindowProcessor: Skipping window {window_idx} with length {window_size}, expected {self.window_size}")
                continue

            # Validate the window's data
            if np.any(np.isnan(window)) or np.any(np.isinf(window)):
                logging.warning(f"WindowProcessor: Invalid values (NaN/Inf) in window {window_idx}")
                continue

            # Flatten the window into a flattened_window
            try:
                flattened_window = window.flatten().astype(np.float32)
                flattened_windows.append(flattened_window)
            except Exception as flatten_error:
                logging.warning(f"WindowProcessor: Error flattening window {window_idx}: {flatten_error}")
                continue

        logging.debug(f"WindowProcessor: Processed {len(flattened_windows)} valid flattened windows from {n_windows} 3D windows")
        return flattened_windows

    def _process_window_list(self, windows: List) -> List[np.ndarray]:
        """Process a list of windows into flattened windows."""
        flattened_windows = []
        expected_state_length = None

        logging.debug(f"WindowProcessor: Processing {len(windows)} windows, expected window_size: {self.window_size}")

        for window_idx, window in enumerate(windows):
            # Each window should already be exactly window_size length from database
            if len(window) != self.window_size:
                logging.debug(f"WindowProcessor: Skipping window {window_idx} with length {len(window)}, expected {self.window_size}")
                continue

            # Process states in the window
            processed_states = []
            window_valid = True

            for state_idx, state in enumerate(window):
                try:
                    # States should already be normalized numpy arrays from preprocessing
                    if isinstance(state, np.ndarray):
                        if state.size == 0:
                            logging.warning(f"WindowProcessor: Empty state array at window {window_idx}, state {state_idx}")
                            window_valid = False
                            break
                        state_flat = state.flatten().astype(np.float32)
                    elif isinstance(state, (list, tuple)):
                        if len(state) == 0:
                            logging.warning(f"WindowProcessor: Empty state list at window {window_idx}, state {state_idx}")
                            window_valid = False
                            break
                        state_array = np.array(state, dtype=np.float32)
                        state_flat = state_array.flatten()
                    elif isinstance(state, (int, float)):
                        state_flat = np.array([float(state)], dtype=np.float32)
                    else:
                        logging.warning(f"WindowProcessor: Unknown state type at window {window_idx}, state {state_idx}: {type(state)}")
                        window_valid = False
                        break

                    # Validate values - should be clean from normalization
                    if np.any(np.isnan(state_flat)) or np.any(np.isinf(state_flat)):
                        logging.warning(f"WindowProcessor: Invalid values (NaN/Inf) at window {window_idx}, state {state_idx}")
                        window_valid = False
                        break

                    # Check state length consistency
                    if expected_state_length is None:
                        expected_state_length = len(state_flat)
                        logging.debug(f"WindowProcessor: Expected state length set to {expected_state_length}")
                    elif len(state_flat) != expected_state_length:
                        logging.warning(f"WindowProcessor: Window {window_idx}, state {state_idx} has length {len(state_flat)}, expected {expected_state_length}")
                        window_valid = False
                        break

                    processed_states.append(state_flat)

                except Exception as e:
                    logging.warning(f"WindowProcessor: Error processing window {window_idx}, state {state_idx}: {e}")
                    window_valid = False
                    break

            # Add valid flattened windows - flatten the entire window into one flattened_window
            if window_valid and len(processed_states) == self.window_size:
                try:
                    flattened_window = np.concatenate(processed_states)
                    flattened_windows.append(flattened_window)

                    if window_idx < 3:  # Debug first few flattened windows
                        logging.debug(f"WindowProcessor: Created flattened_window {window_idx} with shape {flattened_window.shape}")

                except Exception as concat_error:
                    logging.warning(f"WindowProcessor: Error concatenating window {window_idx}: {concat_error}")
                    continue

        logging.debug(f"WindowProcessor: Created {len(flattened_windows)} valid flattened windows from {len(windows)} windows")

        return flattened_windows

    def _process_array_data(self, data: np.ndarray) -> List[np.ndarray]:
        """Process a 2D array of raw timesteps into flattened windows via a sliding window."""
        if data.ndim != 2:
            raise ValueError(f"WindowProcessor: Expected 2D array, got {data.ndim}D")

        n_timesteps, n_features = data.shape

        if n_timesteps < self.window_size:
            logging.warning(f"WindowProcessor: Not enough timesteps ({n_timesteps}) for window size ({self.window_size})")
            return []

        flattened_windows = []
        for i in range(n_timesteps - self.window_size + 1):
            window = data[i:i + self.window_size]
            flattened_window = window.flatten()
            flattened_windows.append(flattened_window)

        return flattened_windows

    @classmethod
    def get_name(cls) -> str:
        return "window_processor"

# Register the processor
PreprocessingRegistry().register(WindowProcessor)
