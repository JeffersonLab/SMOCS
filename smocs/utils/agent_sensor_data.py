"""
Utility functions for extracting sensor data from messages across different data sources.
Utilized both in the agent ML inference thread and the Data ingestion thread.
"""

import logging
from typing import Dict, Any, List, Tuple

def extract_sensor_values(channels: Dict[str, Any], topic: str) -> Tuple[List[str], List[float]]:
    """
    Extract numeric sensor values from message channels.
    
    Args:
        channels: Dictionary of channel data from message
        topic: Kafka topic name to determine extraction strategy
        
    Returns:
        Tuple of (state_keys, state_values) - sorted keys and corresponding numeric values
    """
    if not channels:
        logging.warning(f"extract_sensor_values: No channels found in message")
        return [], []
    
    # Handle gymnasium data specially, grab all numeric data for everything else
    if topic == 'gymnasium-output':  # Gymnasium topic
        # Look for state_ prefixed keys only
        state_keys = [k for k in channels.keys()
                    if k.startswith('state_')
                    and k != 'state_shape'
                    and k != 'state_is_array'
                    and isinstance(channels[k], (int, float))]
    else:  # EPICS, MQTT, or other data sources
        # Extract all numeric values
        state_keys = [k for k in channels.keys()
                    if isinstance(channels[k], (int, float))]
    
    state_keys.sort()
    state_values = []
    
    for key in state_keys:
        try:
            value = float(channels[key])
            state_values.append(value)
        except (ValueError, TypeError):
            logging.warning(f"extract_sensor_values: Skipping non-numeric state value: {key}={channels[key]}")
            continue
    
    return state_keys, state_values