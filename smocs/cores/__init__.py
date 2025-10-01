"""
Core components for SMOCS Kafka streaming system.
"""

__version__ = "1.0.0"

# Define what's available without importing
__all__ = [
    "KafkaProducerBase",
    "KafkaConsumerBase", 
    "KafkaStreamingProcessBase",
    "MLInferenceThreadBase",
    "MLTrainingThreadBase",
    "DataIngestThreadBase",
    "AgentBase"
]

# Use __getattr__ for lazy loading
def __getattr__(name):
    if name == "KafkaProducerBase":
        from .kafka_producer_base import KafkaProducerBase
        return KafkaProducerBase
    elif name == "KafkaConsumerBase":
        from .kafka_consumer_base import KafkaConsumerBase
        return KafkaConsumerBase
    elif name == "KafkaStreamingProcessBase":
        from .kafka_streaming_process_base import KafkaStreamingProcessBase
        return KafkaStreamingProcessBase
    elif name == "MLInferenceThreadBase":
        from .ml_inference_thread_base import MLInferenceThreadBase
        return MLInferenceThreadBase
    elif name == "MLTrainingThreadBase":
        from .ml_training_thread_base import MLTrainingThreadBase
        return MLTrainingThreadBase
    elif name == "DataIngestThreadBase":
        from .data_ingest_thread_base import DataIngestThreadBase
        return DataIngestThreadBase
    elif name == "AgentBase":
        from .agent_base import AgentBase
        return AgentBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")