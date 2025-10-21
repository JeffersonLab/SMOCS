import numpy as np
import base64
from typing import Union, Dict, Any
import logging

def numpy_to_base64(arr: np.ndarray) -> Dict[str, Any]:
    """
    Convert numpy array to base64-encoded dictionary with metadata.
    
    Args:
        arr: NumPy array to encode
        
    Returns:
        Dictionary with base64-encoded data, dtype, and shape
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr)}")
    
    return {
        '_numpy_': True,  # Flag to identify this as encoded numpy
        'data': base64.b64encode(arr.tobytes()).decode('utf-8'),
        'dtype': str(arr.dtype),
        'shape': list(arr.shape)  # Convert tuple to list for JSON
    }

def base64_to_numpy(encoded: Dict[str, Any]) -> np.ndarray:
    """
    Convert base64-encoded dictionary back to numpy array.
    
    Args:
        encoded: Dictionary with 'data', 'dtype', and 'shape' keys
        
    Returns:
        Reconstructed numpy array
    """
    if not isinstance(encoded, dict) or not encoded.get('_numpy_', False):
        raise ValueError(f"Invalid encoded numpy format: {encoded}")
    
    try:
        data_bytes = base64.b64decode(encoded['data'])
        dtype = np.dtype(encoded['dtype'])
        shape = tuple(encoded['shape'])
        
        arr = np.frombuffer(data_bytes, dtype=dtype)
        arr = arr.reshape(shape)
        
        return arr
    except Exception as e:
        logging.error(f"Error decoding numpy array: {e}")
        raise

def is_encoded_numpy(obj: Any) -> bool:
    """Check if object is an encoded numpy array."""
    return isinstance(obj, dict) and obj.get('_numpy_', False)

def convert_for_base64_json(obj):
    """
    Convert objects for JSON serialization, encoding numpy arrays as base64.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of obj with numpy arrays encoded
    """
    if isinstance(obj, np.ndarray):
        return numpy_to_base64(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_for_base64_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_base64_json(item) for item in obj]
    else:
        return obj

def decode_base64_json(obj):
    """
    Decode objects from JSON, converting base64-encoded arrays back to numpy.
    
    Args:
        obj: Object to decode
        
    Returns:
        Decoded object with numpy arrays restored
    """
    if isinstance(obj, dict):
        if is_encoded_numpy(obj):
            return base64_to_numpy(obj)
        else:
            return {k: decode_base64_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_base64_json(item) for item in obj]
    else:
        return obj