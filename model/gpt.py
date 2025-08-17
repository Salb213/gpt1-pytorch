import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, attn_p=0.1, resid_p=0.1, ff_p=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_p = attn_p
        self.resid_p = resid_p
        self.ff_p = ff_p

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x/rms)*self.weight
    
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def build_rope_cache(T, head_dim, device, base = 10000):
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    ang = torch.einsum("t,f->tf", t, freqs)
    angles = torch.cat([ang, ang], dim=-1)
    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]
    return cos, sin


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_p=0.1, resid_p=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        assert (n_embd // n_head) % 2 == 0, "head_dim must be even for RoPE"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_drop = nn.Dropout(attn_p)
        self.resid_drop = nn.Dropout(resid_p)
        self._rope_cache = None

    def _rope(self, T, device):
        if (self._rope_cache is None or
            self._rope_cache[0].size(2) < T or
            self._rope_cache[0].device != device):
            self._rope_cache = build_rope_cache(T, self.head_dim, device)
        cos, sin = self._rope_cache
        return cos[:, :, :T, :], sin[:, :, :T, :]

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, x.device)
        q, k = apply_rotary(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd, p =0.1):
        super().__init__()
        hidden = int(4*n_embd* 2/3)
        self.w1 = nn.Linear(n_embd, hidden)
        self.w2 = nn.Linear(n_embd, hidden)
        self.w3 = nn.Linear(hidden, n_embd)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x) + self.w2(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_p=0.1, resid_p=0.1, ff_p=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head, attn_p=attn_p, resid_p=resid_p)
        self.ln2 = RMSNorm(n_embd)
        self.ffwd = FeedForward(n_embd, p=ff_p)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        ### self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([
            Block(
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                attn_p=config.attn_p,
                resid_p=config.resid_p,
                ff_p=config.ff_p
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding_table.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits