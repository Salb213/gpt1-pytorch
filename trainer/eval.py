import sys
sys.path.append(".")

import os
import pickle
import torch
from model.gpt import GPT, GPTConfig

max_new_tokens = 100

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128
    )

model = GPT(config)
checkpoint_path = "experiments/checkpoints/gpt1_checkpoint.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Evaluation] Loaded Checkpoint from step {checkpoint['step']}")
else:
    print("[Evaluation] No Checkpoint found")
    exit()

model.eval()

prompt = input("Enter a prompt: ")

input_ids = tokenizer.encoded(prompt)

input_ids = torch.tensor([input_ids], dtype=torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
input_ids = input_ids.to(device)

for _ in range(max_new_tokens):
    idx_cond = input_ids[:, -config.block_size:]
    logits = model(idx_cond)
    logits = logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)

    next_id = torch.multinomial(probs, num_samples=1)

    input_ids = torch.cat((input_ids, next_id), dim=1)

output = tokenizer.decoded(input_ids[0].tolist())

print("\nGenerated Output: ")
print(output)