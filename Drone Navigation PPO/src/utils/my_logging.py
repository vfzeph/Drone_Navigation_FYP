import logging
import logging.handlers
import os
import json
from datetime import datetime
import argparse
from typing import Any, Dict

def setup_logger(name: str, log_file: str, level=logging.INFO, console_level=logging.INFO,
                 log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', max_bytes=10485760, backup_count=5) -> logging.Logger:
    """
    Setup for a logger with file rotation and separate console logging level.
    """
    formatter = logging.Formatter(log_format)
    logger = logging.getLogger(name)
    logger.setLevel(min(level, console_level))

    # Ensure the directory for the log_file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler with a separate level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    Custom LoggerAdapter for structured logging.
    """
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        if isinstance(msg, dict):
            msg.update(self.extra)
            msg_str = json.dumps(msg)
        else:
            msg_str = f'{self.extra} - {msg}'
        return msg_str, kwargs

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to configure logging levels.
    """
    parser = argparse.ArgumentParser(description="Configure logging level and console logging level.")
    parser.add_argument('--log_level', type=str, default='INFO', help='Set the logging level for the file (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--console_level', type=str, default='ERROR', help='Set the logging level for the console (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Validate and set log levels
    log_level = getattr(logging, args.log_level.upper(), None)
    console_level = getattr(logging, args.console_level.upper(), None)
    if not isinstance(log_level, int) or not isinstance(console_level, int):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        raise ValueError(f'Invalid log level: {args.log_level} or {args.console_level}. Allowed options are: {", ".join(allowed_levels)}')

    log_dir = "logs"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_name = f"{log_dir}/experiment_{current_time}.log"

    logger = setup_logger('structured_logger', log_file_name, level=log_level, console_level=console_level)
    adapter = StructuredLoggerAdapter(logger, {"application": "drone_navigation"})

    # Example structured logging
    adapter.info({"message": "Experiment started", "phase": "initialization"})
    adapter.info({"message": "Training progress", "epoch": 1, "loss": 0.5, "accuracy": 0.85})
