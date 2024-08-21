from pathlib import Path

from config_loader import cfg

# Defines an ontology class that can be used for loading ontologies in other scripts, e.g. the Envo ontology from the raw data dir can then be called with dot notation
#  from ontologies_loader import envo
# print(envo.file_path)


class Ontology:
    def __init__(self, name, file_path, iri, root_class_iri=None):
        self.name = name
        self.file_path = file_path
        self.iri = iri
        self.root_class_iri = root_class_iri


# Load the ENVO ontology
envo = Ontology(
    name="ENVO",
    file_path=str(Path(cfg.DATA_DIR) / "raw" / "ontologies" / "envo.owl"),
    iri="http://purl.obolibrary.org/obo/envo.owl",
)

# Make callable list of ontologies
ontologies = [envo]

# Example usage
# from ontologies_loader import ontologies

# for ont in ontologies:
#     print(ont.name)


print("Ontology data succesfully loaded")
