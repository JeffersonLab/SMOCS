"""
Core components SMOCS
"""

from .kafka_producer_base import KafkaProducerBase
from .kafka_consumer_base import KafkaConsumerBase


__version__ = "1.0.0"
__all__ = ["KafkaProducerBase", "KafkaConsumerBase"]