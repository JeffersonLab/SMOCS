"""
Consumer implementations for SMOCS.
"""

__all__ = ["InfluxDBConsumer"]

def __getattr__(name):
    if name == "InfluxDBConsumer":
        from .influxdb_kafka_consumer import InfluxDBConsumer
        return InfluxDBConsumer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")