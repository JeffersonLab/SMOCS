"""
Control plane implementations for SMOCS.
"""

__all__ = ["KafkaGymWrapper"]

def __getattr__(name):
    if name == "KafkaGymWrapper":
        from .gymnasium_kafka_controller import KafkaGymWrapper
        return KafkaGymWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from gymnasium.envs.registration import register

register(
    id='SCORE-IndustryParticleAccelerator-v0',
    entry_point='smocs.control_plane.proxy_industry_env:IndustryParticleAcceleratorEnv',
)