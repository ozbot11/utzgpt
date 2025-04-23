import os
import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]

decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

import random

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

from model import GPTLanguageModel

# -------------------- Hyperparameters --------------------
block_size = 128
batch_size = 16
# max_iters = 100000 # 50000 # 3000
# eval_interval = 500 # 1000 # 300
max_iters = 100_000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
eval_iters = 200
n_embd = 768 # 64
n_head = 12
n_layer = 12 # 4
# -------------------- Model Initialization --------------------
model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -------------------- Evaluation Helper --------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


begin_iter = 0
if os.path.exists('model.pth'):
    print("Loading existing model...")
    model.load_state_dict(torch.load('model.pth'))
    begin_iter = model.state_dict().get('iter', 14000)


# for iter in range(begin_iter, max_iters):
#     print(f"Iteration {iter+1}/{max_iters}", end='\r')
#     # periodically evaluate loss
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#         model.state_dict()['iter'] = iter
#         torch.save(model.state_dict(), 'model.pth')

#     xb, yb = get_batch('train')
#     xb, yb = xb.to(device), yb.to(device)

#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# -------------------- Generate Text --------------------
# context = torch.zeros((1, 1), dtype=torch.long).to(device)  # Start with the token "0"
# generated = model.generate(context, max_new_tokens=200)
# print("\n--- Generated Text ---\n")
# print(decode(generated[0].tolist()))

context = torch.tensor(encode("Winter is coming"), dtype=torch.long).unsqueeze(0).to(device)

# context = torch.zeros((1, 1), dtype=torch.long).to(device)

generated = model.generate(context, max_new_tokens=300, temperature=0.001)
print("\n--- Generated Text ---\n")
print(decode(generated[0].tolist()))
print("\n--- Generated Text ---\n")
