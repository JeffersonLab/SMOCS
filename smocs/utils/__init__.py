"""Utils for SMOCS."""

__all__ = [
    "ConfigLoader",
    "validate_topic_format",
    "validate_message_format",
    "ChannelFilter",
    "setup_logging"
]

def __getattr__(name):
    if name == "ConfigLoader":
        from .config_loader import ConfigLoader
        return ConfigLoader
    elif name == "validate_topic_format":
        from .kafka_message_validation import validate_topic_format
        return validate_topic_format
    elif name == "validate_message_format":
        from .kafka_message_validation import validate_message_format
        return validate_message_format
    elif name == "ChannelFilter":
        from .channel_filter import ChannelFilter
        return ChannelFilter
    elif name == "setup_logging":
        from .logging_config import setup_logging
        return setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")