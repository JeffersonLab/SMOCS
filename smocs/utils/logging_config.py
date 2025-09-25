import logging
import os

def setup_logging():
    """
    Set up logging based on SMOCS_LOG_LEVEL environment variable.
    Call this once at the start of each main module.
    """
    # Get log level from environment, default to INFO
    log_level = os.environ.get('SMOCS_LOG_LEVEL', 'INFO').upper()
    
    # Convert to logging constant
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Configure logging with force=True to override any existing config
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # This overwrites existing configurations
    )
    
    # Print confirmation (will appear in Docker logs)
    print(f"SMOCS logging configured: level={log_level}")