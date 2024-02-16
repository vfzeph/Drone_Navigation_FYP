import json
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs', log_file='training_log.json'):
        """
        Initializes the logger.
        
        Params:
        - log_dir (str): Directory where log files will be saved.
        - log_file (str): Name of the log file.
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(log_dir, log_file)
        self.data = {}
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize log file if it doesn't exist
        if not os.path.isfile(self.log_path):
            with open(self.log_path, 'w') as file:
                json.dump({}, file)

    def log(self, key, value):
        """
        Logs a key-value pair.
        
        Params:
        - key (str): The metric name.
        - value (float): The value of the metric.
        """
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def save(self):
        """
        Saves the current logs to the log file.
        """
        with open(self.log_path, 'w') as file:
            json.dump(self.data, file, indent=4)
    
    def load(self):
        """
        Loads existing logs from the log file.
        """
        with open(self.log_path, 'r') as file:
            self.data = json.load(file)
    
    def print_latest(self):
        """
        Prints the latest values of all logged metrics.
        """
        for key, values in self.data.items():
            print(f'{key}: {values[-1]}')

# Example usage
if __name__ == "__main__":
    logger = Logger()
    logger.log('episode_reward', 100)
    logger.log('average_reward', 50)
    logger.print_latest()
    logger.save()
