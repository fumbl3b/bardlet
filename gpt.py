import torch

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

block_size = 8
batch_size = 4

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

print(vocab_size)
print(encode("hello"), decode(encode("hello")))
x, y = get_batch("train")
print(x.shape, y.shape)
print(x[0])
print(y[0])
