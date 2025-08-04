import os

folders = [
    "config", "data/raw", "model", "trainer", "tokenizer", "utils", 
    "experiments/checkpoints", "experiments/logs"
]
files = {
    "README.md": "# GPT-1 from Scratch\nThis is my own GPT-1 built in PyTorch.",
    "requirements.txt": "torch\nnumpy\ntqdm\n",
    ".gitignore": "__pycache__/\n*.pt\n*.log\n",
    "data/prepare_data.py": "# TODO: Write code to load and tokenize text data\n",
    "model/gpt.py": "# TODO: Write GPT-1 model architecture here\n",
    "trainer/train.py": "# TODO: Write training loop\n",
    "trainer/eval.py": "# TODO: Write text generation code\n",
    "tokenizer/tokenizer.py": "# TODO: Build simple tokenizer (char-level or BPE)\n",
    "utils/logging.py": "# TODO: Write logging helpers (optional)\n",
    "config/gpt_config.json": '{\n  "vocab_size": 50257,\n  "block_size": 512,\n  "n_layer": 12,\n  "n_head": 12,\n  "n_embd": 768\n}\n'
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file, content in files.items():
    with open(file, "w") as f:
        f.write(content)

print("Project structure created.")
