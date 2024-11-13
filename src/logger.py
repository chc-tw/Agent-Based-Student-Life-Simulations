import logging
from datetime import datetime
import os
import json
from typing import Dict, Any

class Logger:
    def __init__(self, name: str, personality: str, log_dir: str = 'logs'):
        """Initialize logger for an agent"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.name = name
        self.personality = personality
        
        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # Create file handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fh = logging.FileHandler(f'logs/{name}_{timestamp}.log')
            fh.setLevel(logging.INFO)
            
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # Log the agent's name and personality at the beginning
        self.logger.info(f"\nPersonality: {self.personality}")

    def log_prompt(self, prompt_name: str, inputs: Dict[str, Any], output: str):
        """Log prompt inputs and outputs"""
        log_data = {
            "prompt_name": prompt_name,
            "inputs": inputs,
            "output": output
        }
        self.logger.info(f"PROMPT: {json.dumps(log_data, indent=4)}")

    def log_action(self, action: str, status: Dict[str, Any], day: int):
        """Log agent actions and status updates"""
        log_data = {
            "day": day,
            "action": action,
            "status": status
        }
        self.logger.info(f"ACTION: {json.dumps(log_data, indent=4)}")