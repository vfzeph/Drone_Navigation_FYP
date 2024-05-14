import logging
import logging.handlers
import os
import json
from datetime import datetime
from argparse import ArgumentParser
from enum import Enum, auto
from typing import Any, Dict

class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

def setup_logger(name: str, log_file: str, level: LogLevel = LogLevel.INFO,
                 console_level: LogLevel = LogLevel.ERROR) -> logging.Logger:
    """Sets up and returns a logger with file and console handlers.
    
    Args:
        name (str): The name of the logger.
        log_file (str): File path where the log should be stored.
        level (LogLevel): The logging level for the file output.
        console_level (LogLevel): The logging level for the console output.
        
    Returns:
        logging.Logger: Configured logger with specified handlers and level.
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logger = logging.getLogger(name)
        logger.setLevel(min(level.value, console_level.value))

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level.value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    except Exception as e:
        print(f"Failed to set up logger due to: {str(e)}")
        raise

class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Enhances logging output with structured (JSON) data."""
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        if isinstance(msg, dict):
            msg.update(self.extra)
            msg_str = json.dumps(msg)
        else:
            msg_str = f'{self.extra} - {msg}'
        return msg_str, kwargs

def parse_arguments() -> ArgumentParser:
    """Parses command line arguments for logging levels."""
    parser = ArgumentParser(description="Configure logging level and console logging level.")
    parser.add_argument('--log_level', type=LogLevel, choices=list(LogLevel), default=LogLevel.INFO,
                        help='Set the logging level for the file')
    parser.add_argument('--console_level', type=LogLevel, choices=list(LogLevel), default=LogLevel.ERROR,
                        help='Set the logging level for the console')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    log_dir = "logs"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_name = f"{log_dir}/experiment_{current_time}.log"

    logger = setup_logger('structured_logger', log_file_name, level=args.log_level, console_level=args.console_level)
    adapter = StructuredLoggerAdapter(logger, {"application": "drone_navigation"})

    # Logging examples showing structured logging usage.
    adapter.info({"message": "Experiment started", "phase": "initialization"})
    adapter.info({"message": "Training progress", "epoch": 1, "loss": 0.5, "accuracy": 0.85})
