import os
from config_loader import cfg

def read_terms(filepath):
    with open(filepath, 'r') as file:
        terms = file.read().splitlines()
    return terms

def generate_plural(term):
    if term.endswith('y') and not term.endswith('ay') and not term.endswith('ey') and not term.endswith('oy') and not term.endswith('uy'):
        return term[:-1] + 'ies'
    elif term.endswith('s') or term.endswith('x') or term.endswith('z') or term.endswith('sh') or term.endswith('ch'):
        return term + 'es'
    elif term.endswith('f'):
        return term[:-1] + 'ves'
    elif term.endswith('fe'):
        return term[:-2] + 'ves'
    else:
        return term + 's'

def add_plurals(terms):
    plural_terms = set()
    for term in terms:
        plural_terms.add(term)
        plural_terms.add(generate_plural(term))
    return sorted(plural_terms)

def write_terms(filepath, terms):
    with open(filepath, 'w') as file:
        for term in terms:
            file.write(term + '\n')

if __name__ == "__main__":
    input_filepath = cfg.VOCAB_DIR + '/envidis_vocab.txt'
    output_filepath = cfg.VOCAB_DIR + '/envidis_vocab_with_plurals.txt'

    terms = read_terms(input_filepath)
    terms_with_plurals = add_plurals(terms)
    write_terms(output_filepath, terms_with_plurals)