"""
Agent implementations for SMOCS.
"""

__all__ = ["AutoencoderAgent", "RLControlAgent", "RidgeRegressionAgent"]

def __getattr__(name):
    if name == "AutoencoderAgent":
        from .autoencoder_agent import AutoencoderAgent
        return AutoencoderAgent
    elif name == "RLControlAgent":
        from .rl_control_agent import RLControlAgent
        return RLControlAgent
    elif name == "RidgeRegressionAgent":
        from .ridge_regression_agent import RidgeRegressionAgent
        return RidgeRegressionAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")