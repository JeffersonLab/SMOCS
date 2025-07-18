"""
SMOCS - Streaming Monitoring Optimization Control System

A Python package for Kafka-based streaming data processing
"""

__version__ = "1.0.0"
__author__ = "Jefferson Lab"

from .cores.kafka_producer_base import KafkaProducerBase
from .cores.kafka_consumer_base import KafkaConsumerBase

__all__ = ["KafkaProducerBase", "KafkaConsumerBase"]