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
            fh = logging.FileHandler(f'{log_dir}/{name}.log')
            fh.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add file handler to logger
            self.logger.addHandler(fh)

        # Log the agent's name and personality at the beginning
        self.logger.info(f"\nPersonality: {self.personality}")

    def log_prompt(self,day: int, prompt_name: str, inputs: str, output: str):
        """Log prompt inputs and outputs"""
        log_data = {
            "day": day,
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
    
    def log_sick(self, day: int, status: Dict[str, Any]):
        log_data = {
            "day": day,
            "action": "Get sick",
            "status": status
        }
        self.logger.info(f"SICK: {json.dumps(log_data, indent=4)}")