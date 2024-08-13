# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DIC_DIR = os.path.join(DATA_DIR, "dictionary")
ONTO_DIR = os.path.join(DATA_DIR, "ontologies")

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


print("PROJECT_ROOT:", PROJECT_ROOT)
