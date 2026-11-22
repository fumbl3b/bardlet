import math
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# read input training data

device = "mps" if torch.backends.mps.is_available() else "cpu"

text = open("input.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {
    ch: i for i, ch in enumerate(chars)
}
itos = {
    i: ch for ch, i in stoi.items()
}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

n_embd = 256
n_head = 4
n_layer = 6
dropout = 0.2
block_size = 256
batch_size = 64


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward()
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = GPT().to(device)
print(sum(p.numel() for p in model.parameters()), "parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(50)
        for k in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

save_path = "model_v2.pt"   # a new name, so the published model.pt is never overwritten
max_steps = 4000
warmup = 100
max_lr = 1e-3
min_lr = 1e-4
patience = 2                # stop after this many evaluations in a row with no improvement

def lr_at(step):
    # ramp up for the first `warmup` steps, then follow a cosine curve down to min_lr
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

best_val = float("inf")
best_step = 0
bad_evals = 0
started = time.time()

for step in range(max_steps + 1):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)

    if step % 500 == 0:
        losses = estimate_loss()
        minutes = (time.time() - started) / 60
        print(f"step {step}: train {losses['train']:.3f}, val {losses['val']:.3f}, "
              f"lr {lr_at(step):.5f}, {minutes:.1f} min", flush=True)
        # keep the best model so far, in the format chat.py loads
        if losses["val"] < best_val:
            best_val, best_step, bad_evals = losses["val"], step, 0
            torch.save({
                "model": model.state_dict(),
                "chars": chars,
                "config": dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size),
            }, save_path)
        else:
            bad_evals += 1
            if bad_evals == patience:
                print("val stopped improving, stopping early", flush=True)
                break
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"best val {best_val:.3f} at step {best_step}, saved to {save_path}")

start = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(start, 500)[0].tolist()))
