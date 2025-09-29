"""Preprocessing modules for SMOCS."""

__all__ = [
    'PreprocessingRegistry',
    'PreprocessingManager', 
    'BasePreprocessor',
    'WindowProcessor',
    'BoundsNormalizer'
]

def __getattr__(name):
    if name == "PreprocessingRegistry":
        from .registry import PreprocessingRegistry
        return PreprocessingRegistry
    elif name == "PreprocessingManager":
        from .manager import PreprocessingManager
        return PreprocessingManager
    elif name == "BasePreprocessor":
        from .base import BasePreprocessor
        return BasePreprocessor
    elif name == "WindowProcessor":
        from .window_processor import WindowProcessor
        return WindowProcessor
    elif name == "BoundsNormalizer":
        from .bounds_normalizer import BoundsNormalizer
        return BoundsNormalizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")