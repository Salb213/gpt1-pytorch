import sys
sys.path.append(".")

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from model.gpt import GPT, GPTConfig
import torch.nn.functional as F

block_size = 64
batch_size = 32
max_iters = 5000
eval_interval = 100

checkpoint_path = "experiments/checkpoints/gpt1_checkpoint.pt"

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

def save_checkpoint(model, optimizer, step, loss):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }, checkpoint_path)
    print(f"[Checkpoint] saved at step {step}")

def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        loss = checkpoint['loss']
        print(f"[Checkpoint] Loaded checkpoint from step {step}")
        return step, loss
    else:
        print(f"[Checkpoint] No checkpoint found, starting from scratch")
        return 0, None
    
start_step,_ = load_checkpoint(model, optimizer)

for step in range(start_step, max_iters):
    x,y = get_batch("train", batch_size, block_size)
    x,y = x.to(device), y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"Step {step:4d} | Train Loss: {loss.item():4f}")

    if step % 1000 == 0 or step == max_iters - 1:
        save_checkpoint(model, optimizer, step, loss.item())