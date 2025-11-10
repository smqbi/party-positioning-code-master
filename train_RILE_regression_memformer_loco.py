import os
import json
from math import ceil
from random import shuffle, seed

import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from memorizing_transformers_pytorch import MemorizingTransformer
from transformers import AutoTokenizer
import utils


class RILERegressionHead(torch.nn.Module):
    def __init__(self, input_dim, inner_dim=512):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, 1)

    def forward(self, x):
        return torch.tanh(self.linear2(torch.tanh(self.linear1(x))))


def _epoch(data, tokeniser, encoder, head, batch_size, max_length, train=False, optimiser=None):
    if train:
        encoder.train()
    else:
        encoder.eval()
    n_steps = ceil(data.shape[0] / batch_size)
    preds, losses = [], []
    for step in tqdm(range(n_steps), leave=False):
        lo, hi = step * batch_size, step * batch_size + batch_size
        texts = data.text[lo:hi].to_list()
        toks = tokeniser(texts, truncation=True, max_length=max_length, padding='longest', return_tensors='pt')
        x = toks['input_ids'].cuda()
        with torch.set_grad_enabled(train):
            enc_out = encoder(x)
            reps = enc_out.mean(dim=1)
            out = head(reps)
            if train:
                gold = torch.tensor(data.RILE[lo:hi].values, dtype=torch.float32, device='cuda').unsqueeze(-1)
                loss = torch.mean((out - gold) ** 2)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                losses.append(loss.item())
        preds.extend(out.detach().cpu().flatten().tolist())
    return (sum(losses) / len(losses) if losses else None), preds


def main():
    seed(42)
    torch.manual_seed(42)

    data_path = "memformer_chunk_training_data.csv"
    results_path = "memformer_all_countries_results.json"

    all_data = pd.read_csv(data_path, sep='\t')
    print(f"Dataset size: {all_data.shape[0]}")

    tokeniser = AutoTokenizer.from_pretrained("roberta-base")
    encoder = MemorizingTransformer(
        num_tokens=30522,
        dim=512,
        depth=6,
        max_seq_len=8192,
        memorizing_layers=[3, 6],
        memory_top_k=32
    ).cuda()

    head = RILERegressionHead(input_dim=512).cuda()
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1e-5)

    n_epochs = 5
    batch_size = 1
    max_length = 8192

    results = {}

    for (country, data_train_all, data_test) in utils.split_chunk_data_loco(all_data):
        idx = list(range(data_train_all.shape[0]))
        shuffle(idx)
        dev_size = len(idx) // 10
        dev_idx, train_idx = idx[:dev_size], idx[dev_size:]
        data_train = data_train_all.iloc[train_idx, :]
        data_dev = data_train_all.iloc[dev_idx, :]

        print(f"Running LOCO for {country} | train={len(data_train)}, dev={len(data_dev)}, test={len(data_test)}")

        best_dev, best_test = -1.0, -1.0
        for ep in range(n_epochs):
            tr_loss, _ = _epoch(data_train, tokeniser, encoder, head, batch_size, max_length, train=True, optimiser=opt)
            _, dev_preds = _epoch(data_dev, tokeniser, encoder, head, batch_size, max_length, train=False)
            dev_r = spearmanr(dev_preds, data_dev.RILE).correlation
            print(f"[{country}] epoch {ep+1}: train_mse={tr_loss:.4f} dev_r={dev_r:.4f}")
            if dev_r is not None and dev_r > best_dev:
                best_dev = dev_r
                _, test_preds = _epoch(data_test, tokeniser, encoder, head, batch_size, max_length, train=False)
                test_r = spearmanr(test_preds, data_test.RILE).correlation
                best_test = test_r
                results[country] = {"dev_r": float(best_dev), "test_r": float(best_test)}

    with open(results_path, 'w') as out:
        json.dump(results, out, indent=2)
    print("Saved results:", results)


if __name__ == '__main__':
    main()
