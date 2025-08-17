import sys, os, pickle, torch
sys.path.append(".")
from model.gpt import GPT, GPTConfig

checkpoint_path = "experiments/checkpoints/gpt1_best.pt"
max_new_tokens = 100
temperature = 0.7
top_p = 0.9
top_k = None
repetition_penalty = 1.1
freq_penalty = 0.2
min_length = 10

with open("data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

bos_id = tokenizer.encoder.get("<bos>")
eos_id = tokenizer.encoder.get("<eos>")

ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
saved_cfg = ckpt.get("config") or {"block_size": 256, "n_layer": 8, "n_head": 8, "n_embd": 256}

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=saved_cfg["block_size"],
    n_layer=saved_cfg["n_layer"],
    n_head=saved_cfg["n_head"],
    n_embd=saved_cfg["n_embd"],
)

model = GPT(config)
model.load_state_dict(ckpt["model_state_dict"])
print(f"[Evaluation] Loaded checkpoint from step {ckpt.get('step', '?')}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

prompt = input("Enter a prompt: ")
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    for t in range(max_new_tokens):
        idx_cond = input_ids[:, -config.block_size:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / temperature

        if bos_id is not None and input_ids.size(1) > 1:
            logits[:, bos_id] = float("-inf")

        uniq, counts = torch.unique(input_ids, return_counts=True)
        rep_mask = torch.zeros_like(logits, dtype=torch.bool)
        rep_mask[:, uniq] = True
        logits = torch.where(rep_mask, logits / repetition_penalty, logits)
        pen = torch.zeros_like(logits)
        pen.scatter_add_(1, uniq.unsqueeze(0), counts.to(logits.dtype).unsqueeze(0))
        logits = logits - freq_penalty * pen

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            cutoff = cdf > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, ix, False)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if eos_id is not None and t >= min_length and int(next_id) == eos_id:
            break

print("\nGenerated Output:\n", tokenizer.decode(input_ids[0].tolist()))
