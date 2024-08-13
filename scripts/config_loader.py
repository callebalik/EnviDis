# path_setup.py
import os
import sys

# Append the parent directory to sys.path so modules from there can be imported directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "")))

import config

# Export config to the environment
config = config
