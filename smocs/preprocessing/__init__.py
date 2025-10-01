"""Preprocessing modules for SMOCS."""

# Import registry first
from .registry import PreprocessingRegistry
from .manager import PreprocessingManager
from .base import BasePreprocessor

# EAGERLY import and register all processors
from .window_processor import WindowProcessor
from .bounds_normalizer import BoundsNormalizer

__all__ = [
    'PreprocessingRegistry',
    'PreprocessingManager', 
    'BasePreprocessor',
    'WindowProcessor',
    'BoundsNormalizer'
]