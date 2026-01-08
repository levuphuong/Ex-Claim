import pickle
import numpy as np
import torch
from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE
import logging

path = "outputs/"
model_path = path + "Pretrained-models/mGENRE/"
with open(path + "Pretrained-models/mGENRE/" + "titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)
logging.info("Entity linking prefix tree loaded")

model = mGENRE.from_pretrained(model_path + "fairseq_multilingual_entity_disambiguation").eval()
logging.info("Entity linking model loaded")