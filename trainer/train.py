import sys
sys.path.append(".")

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from model.gpt import GPT, GPTConfig
import torch.nn.functional as F
import wandb
import math
from torch.amp import GradScaler, autocast
from contextlib import contextmanager

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except:
    pass

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                p.data, self.shadow[k] = self.shadow[k], p.data        

block_size = 256
batch_size = 32
max_iters = 1_000_000
eval_interval = 500
patience = 8
since_best = 0

checkpoint_path = "experiments/checkpoints/gpt1_checkpoint.pt"
best_path = "experiments/checkpoints/gpt1_best.pt"
best_val = float("inf")

wandb.init(
    project="gpt1-from-scratch",
    config={
        "batch_size": batch_size,
        "block_size": block_size,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 384,
        "learning_rate": 1e-3,
        "dataset": "input.txt",
        "max_iters": max_iters
    }
)

train_data = np.fromfile("data/processed/train.bin", dtype=np.uint16)
val_data = np.fromfile("data/processed/val.bin", dtype=np.uint16)

train_tensor = torch.from_numpy(train_data.astype(np.int64))
val_tensor   = torch.from_numpy(val_data.astype(np.int64))

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=256,
    n_layer=8,
    n_head=8,
    n_embd=384,
    attn_p=0.0, 
    resid_p=0.0, 
    ff_p=0.0
)

model = GPT(config)

@contextmanager
def using_ema(model, ema):
    ema.swap_in(model)
    try:
        yield
    finally:
        ema.swap_in(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")
model = model.to(device)
scaler = GradScaler(enabled=(device == "cuda"))


def param_groups(model, weight_decay=0.1):
    decay, no_decay = set(), set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse = False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or isinstance(m, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)
            else:
                decay.add(fpn)
    assert len(decay & no_decay) == 0
    return [
        {"params":[p for n, p in model.named_parameters() if n in decay], "weight_decay": weight_decay},
        {"params":[p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0}
    ]


optimizer = torch.optim.AdamW(
    param_groups(model, weight_decay=0.1),
    lr=1e-3,
    betas=(0.9, 0.95),
)

wandb.watch(model, log="all")

def get_batch(split, bs, blk):
    data = train_tensor if split == "train" else val_tensor
    ix = torch.randint(0, len(data) - blk - 1, (bs,))
    buf = torch.empty((bs, blk+1), dtype=torch.long)
    for j, i in enumerate(ix):
        buf[j] = data[i:i+blk+1]
    x, y = buf[:, :-1], buf[:, 1:]
    return x.pin_memory(), y.pin_memory()

def evaluate(split):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):
            xb, yb = get_batch(split, batch_size, block_size)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast('cuda', enabled=(device=="cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def save_checkpoint(model, optimizer, step, loss):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
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

start_step, _ = load_checkpoint(model, optimizer)
warmup_steps = 2000
lr_decay_steps = 60_000
max_lr = 1e-4
min_lr = 1e-5
grad_accum = 4
optimizer.zero_grad(set_to_none=True)
ema = EMA(model, decay=0.999)

for step in range(start_step, max_iters):
    for micro in range(grad_accum):
        xb, yb = get_batch("train", batch_size, block_size)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast('cuda', enabled=(device=="cuda")):
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss = loss / grad_accum
        scaler.scale(loss).backward()

    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        decay_step = min(step - warmup_steps, lr_decay_steps)
        progress = decay_step / lr_decay_steps
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    wandb.log({"learning_rate": lr}, step=step)

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    ema.update(model)

    if step % eval_interval == 0:
        with using_ema(model, ema):
            val_loss = evaluate("val")
        pp_train = math.exp(loss.item())
        pp_val = math.exp(val_loss)
        print(f"Step {step:5d} | Train {loss.item():.4f} (pp {pp_train:.2f}) | "
            f"Val {val_loss:.4f} (pp {pp_val:.2f})")
        wandb.log({
            "train_loss": loss.item(),
            "val_loss": val_loss,
            "train_pp": pp_train,
            "val_pp": pp_val,
            "step": step
        }, step=step)

        if val_loss < best_val:
            best_val = val_loss
            since_best = 0
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            with using_ema(model, ema):
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(config),
                    "step": step,
                    "val_loss": val_loss,
                    "ema": True
                }, best_path)
            print(f"[Checkpoint] new best saved @ {step} (val {val_loss:.4f})")
        else:
            since_best += 1
            print(f"[Checkpoint] no improvement, since best {since_best}/{patience}")
            if since_best >= patience:
                print(f"[Checkpoint] Early stopping at step {step} (since best {since_best})")
                break
    if step % 1000 == 0:
        save_checkpoint(model, optimizer, step, loss.item())