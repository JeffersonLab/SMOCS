"""
Core components for SMOCS Kafka streaming system.
"""

from .kafka_producer_base import KafkaProducerBase
from .kafka_consumer_base import KafkaConsumerBase
from .kafka_streaming_process_base import KafkaStreamingProcessBase

__version__ = "1.0.0"
__all__ = ["KafkaProducerBase", "KafkaConsumerBase", "KafkaStreamingProcessBase"]