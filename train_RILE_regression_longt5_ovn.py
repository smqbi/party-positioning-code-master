import os
import json
from math import ceil
from random import shuffle, seed

import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LongT5EncoderModel
import utils


class RILERegressionHead(torch.nn.Module):
    def __init__(self, input_dim, inner_dim=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return x


def train_epoch(epoch_n, data, batch_size, tokeniser, encoder, regression_head, optimiser, max_length):
    encoder.train()
    n_steps = ceil(data.shape[0] / batch_size)
    losses = torch.zeros(n_steps)
    for step in tqdm(range(n_steps), desc=f'Epoch {epoch_n+1}, training', leave=False):
        optimiser.zero_grad()
        lo, hi = step * batch_size, step * batch_size + batch_size
        batch_texts = data.text[lo:hi].to_list()
        tokens = tokeniser(batch_texts, truncation=True, max_length=max_length, padding='longest', return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokens.items()}
        with torch.cuda.amp.autocast():
            enc_out = encoder(**model_inputs).last_hidden_state
            reps = utils.mean_pooling(enc_out, model_inputs['attention_mask'])
            preds = regression_head(reps)
            gold = torch.tensor(data.RILE[lo:hi].values, dtype=torch.float32, device='cuda').unsqueeze(-1)
            loss = torch.mean((preds - gold) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(regression_head.parameters()), 1.0)
        optimiser.step()
        losses[step] = loss.item()
    return losses.mean().item()


def validate_epoch(epoch_n, data, batch_size, tokeniser, encoder, regression_head, max_length):
    encoder.eval()
    n_steps = ceil(data.shape[0] / batch_size)
    preds_all = []
    for step in tqdm(range(n_steps), desc=f'Epoch {epoch_n+1}, validation', leave=False):
        lo, hi = step * batch_size, step * batch_size + batch_size
        batch_texts = data.text[lo:hi].to_list()
        tokens = tokeniser(batch_texts, truncation=True, max_length=max_length, padding='longest', return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            enc_out = encoder(**model_inputs).last_hidden_state
            reps = utils.mean_pooling(enc_out, model_inputs['attention_mask'])
            preds = regression_head(reps).detach().cpu().flatten().tolist()
        preds_all.extend(preds)
    return spearmanr(preds_all, data.RILE).correlation, preds_all


def main():
    seed(42)
    torch.manual_seed(42)

    chunk_file = "chunks_for_manifestos_bigbird.json"
    rile_file = "RILE_by_chunk_bigbird.json"
    dataset_tsv = "longt5_chunk_training_data_baseline.csv"
    results_path = "longt5_ovn_baseline_predictions.json"

    with open(chunk_file) as inp:
        chunks_by_manifesto = json.load(inp)
    with open(rile_file) as inp:
        rile_by_chunk = json.load(inp)

    if os.path.exists(dataset_tsv):
        all_data = pd.read_csv(dataset_tsv, sep='\t')
    else:
        recs = []
        for m_id, chunks in tqdm(list(chunks_by_manifesto.items()), desc='Matching chunks to RILEs', leave=False):
            r = rile_by_chunk[m_id]
            assert len(chunks) == len(r), f'RILE count mismatch for {m_id}'
            for text, rile in zip(chunks, r):
                recs.append([m_id, text, rile])
        all_data = pd.DataFrame(recs, columns=['manifesto_id', 'text', 'RILE'])
        all_data.to_csv(dataset_tsv, sep='\t', index=False)

    print(f"Dataset size: {all_data.shape[0]}")

    data_train_all, data_test = utils.split_chunk_data_old_vs_new(all_data)
    idx = list(range(data_train_all.shape[0]))
    shuffle(idx)
    dev_size = len(idx) // 10
    dev_idx, train_idx = idx[:dev_size], idx[dev_size:]
    data_train = data_train_all.iloc[train_idx, :]
    data_dev = data_train_all.iloc[dev_idx, :]

    print(f"Splits sizes: train={len(data_train)}, dev={len(data_dev)}, test={len(data_test)}")

    model_name = "google/long-t5-tglobal-base"
    tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    encoder = LongT5EncoderModel.from_pretrained(model_name)
    encoder.gradient_checkpointing_enable()
    encoder.cuda()

    d_model = encoder.config.d_model
    regression_head = RILERegressionHead(input_dim=d_model).cuda()
    optimiser = torch.optim.AdamW(list(encoder.parameters()) + list(regression_head.parameters()), lr=5e-6)

    n_epochs = 5
    batch_size = 1
    max_length = 4096

    best_dev_r = -1.0
    test_r_at_best = -1.0
    test_preds = []

    for epoch_n in range(n_epochs):
        train_loss = train_epoch(epoch_n, data_train, batch_size, tokeniser, encoder, regression_head, optimiser, max_length)
        dev_r, _ = validate_epoch(epoch_n, data_dev, batch_size, tokeniser, encoder, regression_head, max_length)
        print(f"Epoch {epoch_n+1}: train_mse={train_loss:.4f} dev_spearman={dev_r:.4f}")

        if dev_r is not None and dev_r > best_dev_r:
            best_dev_r = dev_r
            test_r, test_preds = validate_epoch(epoch_n, data_test, batch_size, tokeniser, encoder, regression_head, max_length)
            test_r_at_best = test_r
            with open(results_path, 'w') as out:
                json.dump({'spearman': test_r, 'preds': test_preds}, out)

    print(f"\nBEST dev_spearman={best_dev_r:.4f} | test_spearman_at_best={test_r_at_best:.4f}")

    print("\nPredicted RILE scores (first 20):")
    print(test_preds[:20])


if __name__ == '__main__':
    main()
