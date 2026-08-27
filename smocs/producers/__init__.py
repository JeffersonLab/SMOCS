"""
Producer implementations for SMOCS.
"""

__all__ = ["MQTTKafkaProducer", "EpicsKafkaProducer", "RootFileKafkaProducer"]

def __getattr__(name):
    if name == "MQTTKafkaProducer":
        from .mqtt_kafka_producer import MQTTKafkaProducer
        return MQTTKafkaProducer
    elif name == "EpicsKafkaProducer":
        from .epics_kafka_producer import EpicsKafkaProducer
        return EpicsKafkaProducer
    elif name == "RootFileKafkaProducer":
        from .root_file_kafka_producer import RootFileKafkaProducer
        return RootFileKafkaProducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")