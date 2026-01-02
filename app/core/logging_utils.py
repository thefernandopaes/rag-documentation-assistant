import logging
import sys
import json
from typing import Any
from config import Config

class JSONFormatter(logging.Formatter):
    """JSON formatter for structural logging in production."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def configure_logging() -> None:
    """
    Configure logging based on environment.
    
    Production: JSON format, INFO level, reduced noise.
    Development: Standard text format, DEBUG/INFO level.
    """
    
    # Check if production mode
    is_production = Config._is_production()
    log_level = logging.INFO if is_production else logging.DEBUG
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Stream handler (stdout)
    handler = logging.StreamHandler(sys.stdout)
    
    if is_production:
        # JSON Formatter for production (better for Railway/CloudWatch)
        formatter = JSONFormatter()
    else:
        # Standard readable formatter for dev
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Reduce noise from 3rd party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Disable access logs in prod if using a proxy
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Create specific app logger
    logger = logging.getLogger("app")
    logger.setLevel(log_level)
    
    return logger

# Initialize logger instance for imports
logger = configure_logging()
