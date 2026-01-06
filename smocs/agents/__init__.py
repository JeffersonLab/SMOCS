"""
Agent implementations for SMOCS.
"""

__all__ = ["AutoencoderAgent", "RLControlAgent"]

def __getattr__(name):
    if name == "AutoencoderAgent":
        from .autoencoder_agent import AutoencoderAgent
        return AutoencoderAgent
    elif name == "RLControlAgent":
        from .rl_control_agent import RLControlAgent
        return RLControlAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")