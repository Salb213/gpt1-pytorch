import sys
sys.path.append(".")

from tokenizer.tokenizer import Tokenizer
import numpy as np
import os
import pickle

file_path = "data/raw/input.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print("First 100 Chars:\n", text[:100])

tokenizer = Tokenizer()

data = tokenizer.encode(text)
print("Max token id:", max(data), "Vocab size:", tokenizer.vocab_size)
assert max(data) < tokenizer.vocab_size, "Token IDs exceed vocab size!"

n = len(data)

split_idx = int(0.9 * n)

train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train tokens: {len(train_data)}")
print(f"Val tokens: {len(val_data)}")

os.makedirs("data/processed", exist_ok=True)

train_array = np.array(train_data, dtype=np.uint16)
val_array = np.array(val_data, dtype=np.uint16)

train_array.tofile("data/processed/train.bin")
val_array.tofile("data/processed/val.bin")

with open("data/processed/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)