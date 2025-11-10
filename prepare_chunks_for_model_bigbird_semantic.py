import os
import json
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm
from transformers import BigBirdTokenizerFast
from sentence_transformers import SentenceTransformer, util
import utils

print("Loading the data...")
data_path = "full_data_w_translations_cleaned.csv"
chunk_file_path = "chunks_for_manifestos_bigbird_semantic.json"
rile_path = "RILE_by_chunk_bigbird_semantic.json"

data = pd.read_csv(data_path)
print("Preparing the data...")

# Normalize labels and assign manifesto IDs
normalised_labels = data.label.map(utils.normalise_label)
del data["label"]
data.insert(0, "label", normalised_labels)

manifesto_ids = data.apply(utils.get_manifesto_id, axis=1)
data.insert(0, "manifesto_id", manifesto_ids)

# Load tokenizer and SBERT model
model_name = "google/bigbird-roberta-base"
max_chunk_size = 4095
tokeniser = BigBirdTokenizerFast.from_pretrained(model_name)
sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

chunks_by_manifesto = defaultdict(list)
riles_by_manifesto = defaultdict(list)

def segment_manifesto(sentences, max_chunk_size=4095, sim_threshold=0.7):
    """Segment a manifesto into semantically coherent chunks using SBERT + cosine similarity."""
    if not sentences:
        return []

    embeddings = sbert_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    segments = []
    current_segment = [sentences[0]]
    current_length = len(current_segment[0].split())

    for i in range(1, len(sentences)):
        sim = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()

        # Start a new segment if semantic similarity drops
        if sim < sim_threshold:
            segments.append(" ".join(current_segment))
            current_segment = []
            current_length = 0

        current_segment.append(sentences[i])
        current_length += len(sentences[i].split())

        # Enforce transformer max token length
        if current_length >= max_chunk_size:
            segments.append(" ".join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments

# Main loop: process each manifesto
for manifesto_id in tqdm(data.manifesto_id.unique(), desc="Semantic chunking and computing RILEs"):
    sentences = data.loc[data.manifesto_id == manifesto_id].text_translated.to_list()
    labels = data.loc[data.manifesto_id == manifesto_id].label.to_list()

    segments = segment_manifesto(sentences, max_chunk_size=max_chunk_size, sim_threshold=0.7)

    for seg in segments:
        chunks_by_manifesto[manifesto_id].append(seg)
        riles_by_manifesto[manifesto_id].append(utils.compute_rile_from_list(labels))

# Save results
with open(chunk_file_path, "w") as out:
    json.dump(chunks_by_manifesto, out, indent=2)

with open(rile_path, "w") as out:
    json.dump(riles_by_manifesto, out, indent=2)

print(f"Saved semantic chunks to {chunk_file_path}")
print(f"Saved RILE scores to {rile_path}")
