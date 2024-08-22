# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DIC_DIR = os.path.join(DATA_DIR, "dictionary")
ONTO_DIR = os.path.join(DATA_DIR, "ontologies")

VOCAB_DIR = os.path.join(PROJECT_ROOT, "vocab")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
