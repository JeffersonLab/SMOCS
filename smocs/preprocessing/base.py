from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BasePreprocessor(ABC):
    """Base class for all preprocessing modules."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary for this processor
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self):
        """Validate that the configuration contains required parameters."""
        pass
    
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            **kwargs: Additional parameters for processing
            
        Returns:
            Processed data
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the name for registry registration."""
        pass