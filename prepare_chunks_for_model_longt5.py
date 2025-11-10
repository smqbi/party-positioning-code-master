import os, json
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import utils

# Absolute paths
DATA_DIR   = "/mount/studenten-temp1/users/qbaibisa/party-positioning-code-master/data"
CHUNK_FILE = os.path.join(DATA_DIR, "chunks_for_manifestos_longt5.json")
RILE_FILE  = os.path.join(DATA_DIR, "RILE_by_chunk_longt5.json")

print('Loading the data...')
data = pd.read_csv(os.path.join(DATA_DIR, "full_data_w_translations_cleaned.csv"))
print('Preparing the data...')
normalised_labels = data.label.map(utils.normalise_label)
del data['label']
data.insert(0, 'label', normalised_labels)
manifesto_ids = data.apply(utils.get_manifesto_id, axis=1)
data.insert(0, 'manifesto_id', manifesto_ids)

# LongT5 tokenizer
model_name = "google/long-t5-tglobal-base"
MAX_CHUNK_TOKENS = 8192
MIN_CHUNK_TOKENS = 1000
tokeniser = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    cache_dir="/mount/studenten-temp1/users/qbaibisa/hf_cache"
)

chunks_by_manifesto = defaultdict(list)
riles_by_manifesto  = defaultdict(list)

for manifesto_id in tqdm(data.manifesto_id.unique(), desc='Splitting into chunks and computing RILEs'):
    sentences = data.loc[data.manifesto_id == manifesto_id].text_translated.to_list()
    labels    = data.loc[data.manifesto_id == manifesto_id].label.to_list()
    chunk_buffer, label_buffer = [], []
    token_count = 0

    for sentence, label in zip(sentences, labels):
        n_tokens = len(tokeniser.tokenize(sentence))
        if token_count + n_tokens <= MAX_CHUNK_TOKENS:
            chunk_buffer.append(sentence)
            label_buffer.append(label)
            token_count += n_tokens
        else:
            if token_count >= MIN_CHUNK_TOKENS:
                chunks_by_manifesto[manifesto_id].append(' '.join(chunk_buffer))
                riles_by_manifesto[manifesto_id].append(utils.compute_rile_from_list(label_buffer))
            chunk_buffer, label_buffer = [sentence], [label]
            token_count = n_tokens

    if token_count >= MIN_CHUNK_TOKENS:
        chunks_by_manifesto[manifesto_id].append(' '.join(chunk_buffer))
        riles_by_manifesto[manifesto_id].append(utils.compute_rile_from_list(label_buffer))

os.makedirs(DATA_DIR, exist_ok=True)
with open(CHUNK_FILE, 'w') as out:
    json.dump(chunks_by_manifesto, out, indent=2)
with open(RILE_FILE, 'w') as out:
    json.dump(riles_by_manifesto, out, indent=2)

print(f'Wrote {CHUNK_FILE} and {RILE_FILE}')
