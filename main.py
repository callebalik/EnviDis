# Build the EnviDis vocab

import config as cfg
from scripts.vocabs import Vocab

directory = f"{cfg.PROC_DATA_DIR}/vocabs/"

# Build vocab instance
voc = Vocab(directory)

# Get and print valid vocab files
valid_vocab_files = voc.get_valid_files()
print("Valid vocab files:", valid_vocab_files)

# Generates and exports the combined vocab
voc.export_combined_vocab_to_txt(cfg.VOCAB_DIR, "envidis")
