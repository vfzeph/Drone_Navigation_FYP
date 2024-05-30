import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
from torch.utils.tensorboard import SummaryWriter

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        log_record = {
            "time": datetime.utcfromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Ensure record.args is a dictionary before updating log_record
        if isinstance(record.args, dict):
            log_record.update(record.args)
        return json.dumps(log_record)

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler using ThreadPoolExecutor."""
    def __init__(self, handler):
        super().__init__(handler.level)
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        self.executor.submit(self._emit, record)

    def _emit(self, record):
        self.handler.emit(record)

    def close(self):
        self.handler.close()
        self.executor.shutdown()
        super().close()

class CustomLogger:
    """
    Custom logger class to handle both structured JSON logging and traditional text logging.
    Outputs detailed logs to a file for debugging and less detailed, readable logs to the console for regular monitoring.
    Includes asynchronous logging for performance optimization and JSON formatted logs for structured parsing.
    """
    def __init__(self, name, log_dir="logs", log_level=logging.DEBUG, tensorboard_log_dir="tensorboard_logs"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.handlers = self.setup_handlers(log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def setup_handlers(self, log_dir):
        handlers = []

        # Setup detailed log file handler
        detailed_log_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
        detailed_handler = logging.FileHandler(detailed_log_path)
        detailed_handler.setLevel(logging.DEBUG)
        detailed_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        async_detailed_handler = AsyncLogHandler(detailed_handler)
        self.logger.addHandler(async_detailed_handler)
        handlers.append(async_detailed_handler)

        # Setup console handler for less detailed logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
        async_console_handler = AsyncLogHandler(console_handler)
        self.logger.addHandler(async_console_handler)
        handlers.append(async_console_handler)

        # Setup JSON file handler for structured logs
        json_log_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.json'))
        json_handler = logging.FileHandler(json_log_path)
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JsonFormatter())
        async_json_handler = AsyncLogHandler(json_handler)
        self.logger.addHandler(async_json_handler)
        handlers.append(async_json_handler)

        return handlers

    def get_logger(self):
        """Return the initialized logger instance."""
        return self.logger

    def close_handlers(self):
        """Close all handlers to properly release resources."""
        for handler in self.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        self.writer.close()

    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

if __name__ == "__main__":
    logger = CustomLogger("AirSimLogger", log_dir="./logs", tensorboard_log_dir="./tensorboard_logs").get_logger()
    try:
        logger.info("Starting the simulation environment", extra={"environment": "AirSim"})
        logger.debug("Detailed configuration parameters", extra={"config": {"param": "value"}})
        logger.error("Simulation error occurred", extra={"error": "specific_error_detail"})

        # Example of logging scalars to TensorBoard
        custom_logger = CustomLogger("AirSimLogger", tensorboard_log_dir="tensorboard_logs")
        for step in range(100):
            custom_logger.log_scalar("example_scalar", step * 2, step)
    finally:
        for handler in logger.handlers:
            handler.flush()
