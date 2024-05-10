# Load environment variables
from helpers.utils_general import get_absolute_path
from dotenv import dotenv_values

ALL_API_KEYS = dotenv_values(get_absolute_path('.env.development'))


DEBUG_MODE = True

def log(text, only_in_debug: bool = True):
    if only_in_debug and DEBUG_MODE:
        print(f"{text}")
        
        
from helpers.constants import Keywords
KEYWORDS = Keywords()