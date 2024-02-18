import logging
import logging.handlers
import os
import json
from datetime import datetime, timedelta
import argparse
from typing import Any, Dict

def setup_logger(name: str, log_file: str, level=logging.INFO, console_level=logging.INFO,
                 log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', max_bytes=10485760, backup_count=5) -> logging.Logger:
    """
    Setup for a logger with file rotation and separate console logging level.

    Args:
        name (str): The name of the logger.
        log_file (str): File path for the log output.
        level (int): Logging level for the file handler.
        console_level (int): Logging level for the console handler.
        log_format (str): Format for the log messages.
        max_bytes (int): Maximum log file size in bytes before rotating.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: A configured logger instance.
    """
    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(name)
    logger.setLevel(min(level, console_level))  # Set logger to the lower level between file and console

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler with separate level
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
            # Add extra fields to the message dictionary
            msg.update(self.extra)
            msg_str = json.dumps(msg)
        else:
            msg_str = f'{self.extra} - {msg}'
        return msg_str, kwargs

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to configure logging level and console logging level.

    Returns:
        argparse.Namespace: Parsed arguments.
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
        raise ValueError(f'Invalid log level: {args.log_level} or {args.console_level}')

    # Ensure log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_name = f"{log_dir}/experiment_{current_time}.log"

    logger = setup_logger('structured_logger', log_file_name, level=log_level, console_level=console_level)
    adapter = StructuredLoggerAdapter(logger, {"application": "drone_navigation"})

    # Example structured logging
    adapter.info({"message": "Experiment started", "phase": "initialization"})
    adapter.info({"message": "Training progress", "epoch": 1, "loss": 0.5, "accuracy": 0.85})
