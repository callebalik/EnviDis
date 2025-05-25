# path_setup.py
import os
import sys

# Append the parent directory to sys.path so modules from there can be imported directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "")))

# Import all the variables from config.py
# from config import *

import config

cfg = config
print("config loaded")
