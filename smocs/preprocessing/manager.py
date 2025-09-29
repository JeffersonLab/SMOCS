import logging
from typing import Dict, Any, Optional, List
import numpy as np
from .registry import PreprocessingRegistry
from .base import BasePreprocessor

class PreprocessingManager:
    """Manager for creating and configuring preprocessing modules."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize manager with configuration.
        
        Args:
            config: Full agent configuration dictionary
        """
        self.config = config
        self.registry = PreprocessingRegistry()
        self._validate_preprocessing_config()
        self._initialize_pipeline()
    
    def _validate_preprocessing_config(self):
        """Validate preprocessing-related configuration."""
        # Validate model_input section exists
        if 'model_input' not in self.config:
            raise ValueError("PreprocessingManager: 'model_input' section missing from config")
        
        model_input = self.config['model_input']
        
        # Validate channels and bounds
        if 'channels' not in model_input:
            raise ValueError("PreprocessingManager: 'channels' missing from model_input config")
        
        if 'bounds' not in model_input:
            raise ValueError("PreprocessingManager: 'bounds' missing from model_input config")
        
        channels = model_input['channels']
        bounds = model_input['bounds']
        
        if len(channels) != len(bounds):
            raise ValueError(f"PreprocessingManager: Number of channels ({len(channels)}) must match number of bounds ({len(bounds)})")
        
        # Validate window_size exists
        if 'window_size' not in self.config:
            raise ValueError("PreprocessingManager: 'window_size' missing from config")
        
        # Validate preprocessing_pipeline is specified
        if 'preprocessing_pipeline' not in self.config:
            raise ValueError("PreprocessingManager: 'preprocessing_pipeline' must be explicitly specified in config (can be empty list for no preprocessing)")
        
        if not isinstance(self.config['preprocessing_pipeline'], list):
            raise ValueError("PreprocessingManager: 'preprocessing_pipeline' must be a list")
        
        logging.info(f"PreprocessingManager: Configuration validated - {len(channels)} channels, window_size={self.config['window_size']}")
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing pipeline from config."""
        pipeline_names = self.config['preprocessing_pipeline']
        
        # Validate all processors exist in registry
        available_processors = self.registry.list_processors()
        for processor_name in pipeline_names:
            if processor_name not in available_processors:
                raise ValueError(
                    f"PreprocessingManager: Processor '{processor_name}' not found in registry. "
                    f"Available processors: {available_processors}"
                )
        
        # Create processor instances in order
        self.pipeline = []
        for processor_name in pipeline_names:
            processor_instance = self.get_processor(processor_name)
            self.pipeline.append(processor_instance)
        
        # Log pipeline configuration
        if self.pipeline:
            processor_names = [p.__class__.get_name() for p in self.pipeline]
            logging.debug(f"PreprocessingManager: Initialized pipeline with {len(self.pipeline)} processors in order: {processor_names}")
        else:
            logging.debug("PreprocessingManager: No preprocessing pipeline configured (empty list)")
    
    def execute_pipeline(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Execute the full preprocessing pipeline on input data.
        
        Args:
            data: Input data to process
            **kwargs: Additional parameters to pass to processors
            
        Returns:
            Processed data after running through all pipeline stages
        """
        if not self.pipeline:
            logging.debug("PreprocessingManager: No preprocessing pipeline, returning data unchanged")
            return data
        
        processed_data = data
        
        for i, processor in enumerate(self.pipeline):
            processor_name = processor.__class__.get_name()
            logging.debug(f"PreprocessingManager: Executing pipeline stage {i+1}/{len(self.pipeline)}: {processor_name}")
            
            try:
                processed_data = processor.process(processed_data, **kwargs)
                logging.debug(f"PreprocessingManager: Stage {i+1} ({processor_name}) completed successfully")
            except Exception as e:
                logging.error(f"PreprocessingManager: Pipeline failed at stage {i+1} ({processor_name}): {e}")
                raise
        
        logging.debug(f"PreprocessingManager: Pipeline execution complete - processed {len(self.pipeline)} stages")
        return processed_data
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline.
        
        Returns:
            Dictionary containing pipeline metadata
        """
        processor_names = [p.__class__.get_name() for p in self.pipeline]
        return {
            'processor_count': len(self.pipeline),
            'processors': processor_names,
            'execution_order': processor_names
        }
    
    def get_processor(self, processor_name: str, **kwargs) -> BasePreprocessor:
        """
        Create a configured processor instance.
        
        Args:
            processor_name: Name of the processor to create
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured processor instance
        """
        processor_class = self.registry.get_processor_class(processor_name)
        
        if processor_class is None:
            available_processors = self.registry.list_processors()
            raise ValueError(f"PreprocessingManager: Unknown processor '{processor_name}'. Available: {available_processors}")
        
        # Build processor-specific config
        processor_config = self._build_processor_config(processor_name, **kwargs)
        
        try:
            processor_instance = processor_class(processor_config)
            logging.info(f"PreprocessingManager: Created processor '{processor_name}' with config: {processor_config}")
            return processor_instance
        except Exception as e:
            logging.error(f"PreprocessingManager: Failed to create processor '{processor_name}': {e}")
            raise
    
    def _build_processor_config(self, processor_name: str, **kwargs) -> Dict[str, Any]:
        """Build configuration dictionary for a specific processor."""
        processor_config = {}
        
        if processor_name == "window_processor":
            processor_config['window_size'] = self.config['window_size']
        
        elif processor_name == "bounds_normalizer":
            model_input = self.config['model_input']
            processor_config['bounds'] = model_input['bounds']
            processor_config['channels'] = model_input['channels']
        
        else:
            logging.warning(f"PreprocessingManager: No specific config builder for processor '{processor_name}'")
        
        # Apply any override kwargs
        processor_config.update(kwargs)
        
        return processor_config
    
    def list_available_processors(self) -> list:
        """Return list of available processor names."""
        return self.registry.list_processors()