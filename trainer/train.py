import sys
sys.path.append(".")

import numpy as np
import pickle
import torch
import torch.nn as nn
from model.gpt import GPT, GPTConfig

block_size = 64
batch_size = 32

train_data = np.fromfile("data/processed/train.bin", dtype=np.uint16)
val_data = np.fromfile("data/processed/val.bin", dtype=np.uint16)

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

config = GPTConfig(
    vocab_size = tokenizer.vocab_size,
    block_size = block_size,
    n_layer = 4,
    n_head = 4,
    n_embd = 128
)

def get_batch(split, batch_size, block_size):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))

    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return x,y

model = GPT(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)