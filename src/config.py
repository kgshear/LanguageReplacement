import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "eng_sentences.tsv")

OUTPUT_DIR = os.path.join(DATA_DIR, "output")
REPLACEMENTS_PATH_JSON = os.path.join(OUTPUT_DIR, "replaced_sentences.json")
REPLACEMENTS_PATH_CSV = os.path.join(OUTPUT_DIR, "replaced_sentences.csv")