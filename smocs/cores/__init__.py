"""
Core components for SMOCS Kafka streaming system.
"""
try:
    from .kafka_producer_base import KafkaProducerBase
except:
    print("Required packages are not installed")
try:
    from .kafka_consumer_base import KafkaConsumerBase
except:
    print("Required packages are not installed")
try:
    from .kafka_streaming_process_base import KafkaStreamingProcessBase
except:
    print("Required packages are not installed")
try:
    from .ml_inference_thread_base import MLInferenceThreadBase
except:
    print("Required packages are not installed")
try:
    from .ml_training_thread_base import MLTrainingThreadBase
except:
    print("Required packages are not installed")
try:
    from .data_ingest_thread_base import DataIngestThreadBase
except:
    print("Required packages are not installed")
try:
    from .agent_base import AgentBase
except:
    print("Required packages are not installed")

__version__ = "1.0.0"
__all__ = ["KafkaProducerBase", "KafkaConsumerBase", "KafkaStreamingProcessBase", "MLInferenceThreadBase", "MLTrainingThreadBase", "DataIngestThreadBase", "AgentBase"]