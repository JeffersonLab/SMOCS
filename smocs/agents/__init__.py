"""
Agent implementations for SMOCS.
"""

__all__ = ["AutoencoderAgent"]

def __getattr__(name):
    if name == "AutoencoderAgent":
        from .autoencoder_agent import AutoencoderAgent
        return AutoencoderAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")