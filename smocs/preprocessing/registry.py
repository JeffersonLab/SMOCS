import logging
from typing import Dict, Type, Optional
from .base import BasePreprocessor

class PreprocessingRegistry:
    """Singleton registry for preprocessing modules."""
    
    _instance = None
    _registry: Dict[str, Type[BasePreprocessor]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, processor_class: Type[BasePreprocessor]):
        """
        Register a preprocessing module.
        
        Args:
            processor_class: The processor class to register
        """
        name = processor_class.get_name()
        if name in self._registry:
            logging.warning(f"PreprocessingRegistry: Overwriting existing processor '{name}'")
        
        self._registry[name] = processor_class
        logging.info(f"PreprocessingRegistry: Registered processor '{name}'")
    
    def get_processor_class(self, name: str) -> Optional[Type[BasePreprocessor]]:
        """
        Get a processor class by name.
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor class or None if not found
        """
        return self._registry.get(name)
    
    def list_processors(self) -> list:
        """Return list of registered processor names."""
        return list(self._registry.keys())