"""
Control plane implementations for SMOCS.
"""

__all__ = ["KafkaGymWrapper"]

def __getattr__(name):
    if name == "KafkaGymWrapper":
        from .gymnasium_kafka_controller import KafkaGymWrapper
        return KafkaGymWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")