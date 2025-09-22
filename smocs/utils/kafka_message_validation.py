"""
Utility functions for making sure that data being passed to kafka and recieved from kafka adheres framework standards.
"""

import json
from datetime import datetime
from typing import Union

def validate_topic_format(topic: str) -> bool:
    """
    Validate topic follows the required hierarchical format.
    
    Args:
        topic (str): Topic name to validate
        
    Returns:
        bool: True if topic format is valid
        
    Raises:
        ValueError: If topic format is invalid
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"Topic must be a non-empty string, got: {type(topic)}")

    return True

def validate_message_format(message: Union[str, bytes]) -> bool:
    """
    Validate message has required timestamp and channels structure.
    Expected format: {"timestamp": "2025-01-XX", "channels": {...}}
    
    Args:
        message: Message to validate (string or bytes)
        
    Returns:
        bool: True if message format is valid
        
    Raises:
        ValueError: If message format is invalid
    """
    try:
        # Convert bytes to string if necessary
        if isinstance(message, bytes):
            message = message.decode('utf-8')
        
        # Parse JSON
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            raise ValueError(f"Message is not valid JSON: {e}")
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object")
        
        # Check for required timestamp field
        if 'timestamp' not in data:
            raise ValueError("Message must contain 'timestamp' field")
        
        # Validate timestamp can be parsed (flexible format)
        timestamp = data['timestamp']
        if timestamp is not None:  # Allow None timestamps
            try:
                # Try various common timestamp formats
                if isinstance(timestamp, (int, float)):
                    # Unix timestamp
                    datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    # Try ISO format first, then other common formats
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        # Try other formats
                        from dateutil import parser
                        parser.parse(timestamp)
                else:
                    raise ValueError(f"Timestamp must be string, number, or null, got: {type(timestamp)}")
            except (ValueError, OverflowError) as e:
                raise ValueError(f"Invalid timestamp format: {e}")
        
        # Check for required channels field
        if 'channels' not in data:
            raise ValueError("Message must contain 'channels' field")
        
        # Validate channels is a dictionary (content can be anything)
        if not isinstance(data['channels'], dict):
            raise ValueError("'channels' field must be a JSON object")
        
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Message validation failed: {e}")