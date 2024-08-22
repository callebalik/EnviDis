# Build the EnviDis vocab

import config as cfg
from scripts.vocabs import Vocab

# Example usage
directory = f"{cfg.DATA_DIR}/processed/dictionaries/"

# Build vocab instance
voc = Vocab(directory)

# Get and print valid vocab files
valid_vocab_files = voc.get_valid_files()
print("Valid vocab files:", valid_vocab_files)

# Generates and exports the combined vocab
voc.export_combined_vocab_to_txt(cfg.VOCAB_DIR, "envidis")
