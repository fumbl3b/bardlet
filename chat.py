# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Chat with a tiny Shakespeare GPT (plain text version; bard.py is the fancy one).

    uv run chat.py
    uv run chat.py --temperature 0.6

The model only knows how to continue a play script, so "chat" works by
writing your line into the script under one character's name, then letting
the model write the next character's speech.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg["n_embd"], head_size, bias=False)
        self.query = nn.Linear(cfg["n_embd"], head_size, bias=False)
        self.value = nn.Linear(cfg["n_embd"], head_size, bias=False)
        size = cfg["block_size"]
        self.register_buffer("tril", torch.tril(torch.ones(size, size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg["n_embd"] // cfg["n_head"]
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg["n_head"])])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"])

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = cfg["n_embd"]
        # The Identity stands in for the training-time Dropout so layer
        # numbering matches the saved weights.
        self.net = nn.Sequential(nn.Linear(n, 4 * n), nn.ReLU(), nn.Linear(4 * n, n), nn.Identity())

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.sa = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.block_size = cfg["block_size"]
        self.token_embedding = nn.Embedding(vocab_size, cfg["n_embd"])
        self.position_embedding = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.lm_head = nn.Linear(cfg["n_embd"], vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        return self.lm_head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def next_token(self, idx, temperature, top_k=None):
        logits = self(idx[:, -self.block_size:])[:, -1, :] / temperature
        if top_k:
            # Keep only the k most likely characters; the rest get zero probability.
            kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth_best, float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPT(ckpt["config"], len(ckpt["chars"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["chars"]


def main():
    parser = argparse.ArgumentParser(description="Chat with a tiny Shakespeare GPT.")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--model", default=os.path.join(here, "model.pt"))
    parser.add_argument("--you", default="ROMEO", help="script name used for your lines (hidden)")
    parser.add_argument("--bot", default="JULIET", help="script name used for the model's lines (hidden)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="only pick among the k likeliest characters; 0 turns this off")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="lower = safer and more repetitive, higher = wilder")
    parser.add_argument("--max-chars", type=int, default=400)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, chars = load(args.model, device)
    stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(s):
        # Drop anything the model has never seen (digits, emoji, etc.).
        return [stoi[c] for c in s if c in stoi]

    def clean_name(name, fallback):
        # Keep only characters the model knows; a name can't contain a colon or newline.
        name = "".join(c for c in name if c in stoi and c not in ":\n").strip()
        return name or fallback

    params = sum(p.numel() for p in model.parameters())
    print(f"A tiny Shakespeare GPT, {params:,} parameters. Type a message; /quit to leave.\n")

    # Behind the scenes the conversation is still written as a play script,
    # because that is the only format the model knows.
    you = clean_name(args.you, "ROMEO")
    bot = clean_name(args.bot, "JULIET")

    script = []  # the whole play so far, as token ids
    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in {"/quit", "quit", "exit"}:
            break
        if not encode(line):
            continue

        script += encode(f"{you}:\n{line}\n\n{bot}:\n")
        print("bot: ", end="", flush=True)

        reply = ""
        while len(reply) < args.max_chars and not reply.endswith("\n\n"):
            idx = torch.tensor([script], dtype=torch.long, device=device)
            token = model.next_token(idx, args.temperature, args.top_k).item()
            script.append(token)
            reply += chars[token]
            print(chars[token], end="", flush=True)
        if not reply.endswith("\n\n"):
            # Hit the length cap mid-speech: close the speech in the script too.
            script += encode("\n\n")
            print("\n")


if __name__ == "__main__":
    main()
