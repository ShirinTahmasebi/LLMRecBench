def log(text, only_in_debug: bool = True):
    if only_in_debug and DEBUG_MODE:
        print(f"{text}")
     

import os
import wandb
from datetime import datetime

class Initialization:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Initialization, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load environment variables
        from dotenv import dotenv_values
        from helpers.utils_general import get_absolute_path
        self.all_api_keys = dotenv_values(get_absolute_path('.env.development'))

        # Wandb initialization and login
        wandb.login(key=self.all_api_keys["WANDB_API_KEY"]) 


init = Initialization()

ALL_API_KEYS = init.all_api_keys
DEBUG_MODE = True

from helpers.keywords import Keywords
KEYWORDS = Keywords()

from helpers.constants import Constants
CONSTANTS = Constants()