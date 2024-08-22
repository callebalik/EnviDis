# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Root level
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VOCAB_DIR = os.path.join(PROJECT_ROOT, "vocab")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data folders
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR = os.path.join(DATA_DIR, "processed")
DIC_DIR = "dictionaries"
ONTO_DIR = "ontologies"
VOC_DIR = "vocabs"
