import os
import json
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import utils

print('Loading the data...')
data_path = "full_data_w_translations_cleaned.csv"
chunk_file = "chunks_for_manifestos_longt5.json"
rile_file = "RILE_by_chunk_longt5.json"

data = pd.read_csv(data_path)
print('Preparing the data...')

normalised_labels = data.label.map(utils.normalise_label)
del data['label']
data.insert(0, 'label', normalised_labels)
manifesto_ids = data.apply(utils.get_manifesto_id, axis=1)
data.insert(0, 'manifesto_id', manifesto_ids)

model_name = "google/long-t5-tglobal-base"
MAX_CHUNK_TOKENS = 8192
MIN_CHUNK_TOKENS = 1000
tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=False)

chunks_by_manifesto = defaultdict(list)
riles_by_manifesto = defaultdict(list)

for manifesto_id in tqdm(data.manifesto_id.unique(), desc='Splitting into chunks and computing RILEs'):
    sentences = data.loc[data.manifesto_id == manifesto_id].text_translated.to_list()
    labels = data.loc[data.manifesto_id == manifesto_id].label.to_list()
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

with open(chunk_file, 'w') as out:
    json.dump(chunks_by_manifesto, out, indent=2)
with open(rile_file, 'w') as out:
    json.dump(riles_by_manifesto, out, indent=2)

print(f'Wrote {chunk_file} and {rile_file}')
