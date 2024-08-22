from pathlib import Path

import pandas as pd
from config_loader import cfg

# Defines an dictionary class that can be used for loading ontologies in other scripts, e.g. the Envo ontology from the raw data dir can then be called with dot notation
#  from ontologies_loader import envo
# print(envo.file_path)


class Dictionary:
    def __init__(self, name, file_path, dataframe=None):
        self.name = name
        self.file_path = file_path
        self.dataframe = dataframe


env_phen = Dictionary(
    name="Environmental Phenoma",
    file_path=str(
        Path(cfg.DATA_DIR)
        / "raw"
        / "dictionaries"
        / "environmentalphenomena_dictionary_v1.xlsx"
    ),
)

env_phen.dataframe = pd.read_excel(env_phen.file_path)


# Make callable list of ontologies
dictionaries = [env_phen]

# Example usage
# from ontologies_loader import ontologies

# for ont in ontologies:
#     print(ont.name)


print("Dictionary data succesfully loaded")
