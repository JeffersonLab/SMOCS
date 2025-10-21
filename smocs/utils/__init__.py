"""Utils for SMOCS."""

__all__ = [
    "ConfigLoader",
    "validate_topic_format",
    "validate_message_format",
    "ChannelFilter",
    "setup_logging",
    "numpy_to_base64",
    "base64_to_numpy",
    "is_encoded_numpy",
    "convert_for_base64_json",
    "decode_base64_json"
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
    elif name == "numpy_to_base64":
        from .encoding import numpy_to_base64
        return numpy_to_base64
    elif name == "base64_to_numpy":
        from .encoding import base64_to_numpy
        return base64_to_numpy
    elif name == "is_encoded_numpy":
        from .encoding import is_encoded_numpy
        return is_encoded_numpy
    elif name == "convert_for_base64_json":
        from .encoding import convert_for_base64_json
        return convert_for_base64_json
    elif name == "decode_base64_json":
        from .encoding import decode_base64_json
        return decode_base64_json
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")