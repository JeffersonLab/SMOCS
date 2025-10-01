import math
import logging
from typing import Dict, Any, List, Optional, Tuple

class ChannelFilter:
    """
    Utility class for filtering specific channels from Kafka messages.
    
    This class enables selective extraction of channels from incoming messages,
    allowing agents to process only the channels they need rather than entire
    message payloads.
    """
    
    def __init__(self, required_channels: List[str]):
        """
        Initialize the channel filter.
        
        Args:
            required_channels: List of channel names to extract from messages
        """
        self.required_channels = required_channels
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not required_channels:
            self.logger.warning("ChannelFilter initialized with empty channel list")
    
    def filter_channels(self, message_dict: Dict[str, Any]) -> Optional[Tuple[List[str], List[float]]]:
        """
        Extract specified channels from message in configured order.
        
        Args:
            message_dict: Parsed JSON message containing channels
            
        Returns:
            Tuple of (channel_names, channel_values) or None if channels missing/invalid
        """
        try:
            channels = message_dict.get('channels', {})
            
            if not channels:
                self.logger.error("No 'channels' field found in message")
                return None
            
            # Check all required channels exist
            missing_channels = [ch for ch in self.required_channels if ch not in channels]
            if missing_channels:
                self.logger.error(f"Missing required channels: {missing_channels}")
                self.logger.debug(f"Available channels: {list(channels.keys())}")
                return None
            
            # Extract channels in configured order
            filtered_values = []
            for channel in self.required_channels:
                value = channels[channel]
                
                # Check for null/None values
                if value is None:
                    self.logger.error(f"Channel {channel} has null value")
                    return None
                
                # Convert to float, allowing for various numeric types
                try:
                    float_value = float(value)
                    # Check for NaN or infinite values
                    if not self._is_valid_number(float_value):
                        self.logger.error(f"Channel {channel} has invalid numeric value: {value}")
                        return None
                    filtered_values.append(float_value)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Channel {channel} has non-numeric value: {value} ({type(value)})")
                    return None
            
            self.logger.debug(f"Successfully filtered {len(filtered_values)} channels")
            return self.required_channels, filtered_values
            
        except Exception as e:
            self.logger.error(f"Error filtering channels: {e}")
            return None
    
    def _is_valid_number(self, value: float) -> bool:
        """Check if a float value is valid (not NaN or infinite)."""
        return not (math.isnan(value) or math.isinf(value))
    
    def get_required_channels(self) -> List[str]:
        """Get the list of required channels."""
        return self.required_channels.copy()
    
    @staticmethod
    def extract_all_channels(message_dict: Dict[str, Any]) -> Optional[Tuple[List[str], List[float]]]:
        """
        Extract all numeric channels from message in message order (fallback when no filtering configured).
        
        Args:
            message_dict: Parsed JSON message
            
        Returns:
            Tuple of (channel_names, channel_values) or None if no valid channels
        """
        try:
            channels = message_dict.get('channels', {})
            
            if not channels:
                logging.error("No 'channels' field found in message")
                return None
            
            # Extract all numeric channels in message order
            valid_channels = []
            valid_values = []
            
            for channel_name, value in channels.items():
                if value is None:
                    continue
                    
                try:
                    float_value = float(value)
                    # Check for NaN or infinite values
                    import math
                    if math.isnan(float_value) or math.isinf(float_value):
                        continue
                        
                    valid_channels.append(channel_name)
                    valid_values.append(float_value)
                except (ValueError, TypeError):
                    # Skip non-numeric channels
                    continue
            
            if not valid_channels:
                logging.error("No valid numeric channels found in message")
                return None
                
            return valid_channels, valid_values
            
        except Exception as e:
            logging.error(f"Error extracting all channels: {e}")
            return None

    def validate_message_has_channels(self, message_dict: Dict[str, Any]) -> bool:
        """
        Validate that message contains all required channels without extracting values.
        
        Args:
            message_dict: Parsed JSON message
            
        Returns:
            True if all required channels are present, False otherwise
        """
        channels = message_dict.get('channels', {})
        missing_channels = [ch for ch in self.required_channels if ch not in channels]
        
        if missing_channels:
            self.logger.debug(f"Message missing channels: {missing_channels}")
            return False
        
        return True